// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Simulate a 2-turn conversation as ONE prefill sequence to isolate
//! whether multi-turn quality issues come from the engine's per-turn
//! state management (KV cache + DeltaNet across separate prefill calls)
//! or from the model itself struggling with the conversation context.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use std::path::Path;

    let model_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/kaden/.hipfire/models/qwen3.5-35b-a3b.mq4".to_string());
    let n_gen: usize = std::env::var("HIPFIRE_GEN")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(80);

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load");

    let mut kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 2048).unwrap();
    let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).unwrap();

    // Build a one-shot multi-turn prompt: user1 → assistant1 (canned answer)
    // → user2 → assistant_prefix. Then greedy-decode a continuation.
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end   = tokenizer.encode("<|im_end|>");
    let nl       = tokenizer.encode("\n");
    let user_t   = tokenizer.encode("user");
    let asst_t   = tokenizer.encode("assistant");

    // Replicate the daemon's exact sequence: prefill turn-1 user prompt,
    // greedy-decode the assistant response (stop on im_end), advance KV
    // through the trailing nl, then prefill turn-2 user prompt and decode.
    let im_end_tok = im_end[0];

    let mut turn1_user: Vec<u32> = Vec::new();
    {
        let mut push1 = |v: &[u32]| { for &t in v { turn1_user.push(t); } };
        push1(&im_start); push1(&user_t); push1(&nl);
        push1(&tokenizer.encode("What is 2 + 2?"));
        push1(&im_end); push1(&nl);
        push1(&im_start); push1(&asst_t); push1(&nl);
    }
    let mut turn2_user: Vec<u32> = Vec::new();
    {
        let mut push2 = |v: &[u32]| { for &t in v { turn2_user.push(t); } };
        push2(&im_start); push2(&user_t); push2(&nl);
        push2(&tokenizer.encode("Now multiply that by 5."));
        push2(&im_end); push2(&nl);
        push2(&im_start); push2(&asst_t); push2(&nl);
    }

    eprintln!("Turn 1 user: {} tokens, Turn 2 user: {} tokens", turn1_user.len(), turn2_user.len());

    let mut pos = 0usize;

    // ── Turn 1 prefill ──
    eprintln!("--- Prefilling turn 1 ---");
    for &tok in &turn1_user {
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch).unwrap();
        pos += 1;
    }

    // ── Turn 1 decode (using GPU sample_top_p to match daemon exactly) ──
    eprintln!("--- Decoding turn 1 via sample_top_p ---");
    let vocab_size = config.vocab_size;
    let use_sample = std::env::var("USE_SAMPLE").ok().as_deref() == Some("1");
    let mut rng_state: u32 = 0x13579BDFu32;
    let (tok0, rng0) = if use_sample {
        gpu.sample_top_p(&scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
            vocab_size, 0.0, 1.0, rng_state, 0, 1.0).unwrap()
    } else {
        let l = gpu.download_f32(&scratch.logits).unwrap();
        (llama::argmax(&l), rng_state)
    };
    let mut next = tok0;
    rng_state = rng0;
    let mut t1_resp = String::new();
    let mut t1_tokens = 0usize;
    loop {
        t1_resp.push_str(&tokenizer.decode(&[next]));
        t1_tokens += 1;
        qwen35::forward_scratch(&mut gpu, &weights, &config, next, pos, &mut kv, &mut dn, &scratch).unwrap();
        pos += 1;
        if next == im_end_tok || next == config.eos_token { break; }
        if t1_tokens >= 200 { break; }
        if use_sample {
            let (tok, rng) = gpu.sample_top_p(&scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
                vocab_size, 0.0, 1.0, rng_state, 0, 1.0).unwrap();
            next = tok; rng_state = rng;
        } else {
            let l = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&l);
        }
    }
    eprintln!("Turn 1 response ({} tokens): {}", t1_tokens, t1_resp);

    // ── Trailing nl after im_end (mirrors the daemon) ──
    for &t in &nl {
        qwen35::forward_scratch(&mut gpu, &weights, &config, t, pos, &mut kv, &mut dn, &scratch).unwrap();
        pos += 1;
    }

    // ── Turn 2 prefill ──
    eprintln!("--- Prefilling turn 2 ---");
    for &tok in &turn2_user {
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch).unwrap();
        pos += 1;
    }
    // ── Turn 2 decode (same sampler choice as turn 1) ──
    let (tok2, rng2) = if use_sample {
        gpu.sample_top_p(&scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
            vocab_size, 0.0, 1.0, rng_state, 0, 1.0).unwrap()
    } else {
        let l = gpu.download_f32(&scratch.logits).unwrap();
        (llama::argmax(&l), rng_state)
    };
    let mut next = tok2;
    rng_state = rng2;
    let mut out = String::new();
    for _ in 0..n_gen {
        out.push_str(&tokenizer.decode(&[next]));
        if next == config.eos_token { break; }
        qwen35::forward_scratch(&mut gpu, &weights, &config, next, pos, &mut kv, &mut dn, &scratch).unwrap();
        pos += 1;
        if use_sample {
            let (tok, rng) = gpu.sample_top_p(&scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
                vocab_size, 0.0, 1.0, rng_state, 0, 1.0).unwrap();
            next = tok; rng_state = rng;
        } else {
            let l = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&l);
        }
    }
    println!("\n=== assistant turn 2 ===\n{}\n", out);
}
