// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure greedy token dump for byte-exact regression comparison.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: greedy_dump <model.hfq> <out_tokens.txt> [prompt...]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let out_path = &args[2];
    let prompt_text = if args.len() > 3 {
        args[3..].join(" ")
    } else {
        "Write a 500-word essay about Federalist No. 10 by James Madison.".to_string()
    };

    let mode = std::env::var("PROMPT_MODE").unwrap_or_else(|_| "thinking".to_string());
    eprintln!("greedy_dump: {model_path} mode={mode}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    let mut prompt_tokens: Vec<u32> = match mode.as_str() {
        "raw" => tokenizer.encode(&prompt_text),
        _ => {
            let im_start = tokenizer.encode("<|im_start|>");
            let im_end = tokenizer.encode("<|im_end|>");
            let user = tokenizer.encode("user");
            let asst = tokenizer.encode("assistant");
            let nl = tokenizer.encode("\n");
            let user_body = tokenizer.encode(&prompt_text);
            let mut chat = Vec::new();
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&user);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&user_body);
            chat.extend_from_slice(&im_end);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&asst);
            chat.extend_from_slice(&nl);
            if mode == "thinking" {
                chat.extend_from_slice(&tokenizer.encode("<think>"));
                chat.extend_from_slice(&nl);
            }
            chat
        }
    };
    eprintln!("prompt: {} tokens", prompt_tokens.len());

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    let kv_seq = 2048usize;
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    eprintln!("greedy_dump: kv_mode={kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "asym3" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        "asym4" => KvCache::new_gpu_asym4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        "asym2" => KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        _ => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let max_gen = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| kv_seq.saturating_sub(prompt_tokens.len() + 8));
    let mut out = std::fs::File::create(out_path).expect("create out");

    // Route prefill through forward_prefill_batch so the quality gate
    // exercises the batched prefill path directly — any future batching
    // regression in that function will be caught here.
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &prompt_tokens, 0,
        &mut kv_cache, &mut dn_state, &scratch,
        None, None, None, None,
    ).expect("prefill forward failed");

    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);
    writeln!(out, "{next_token}").ok();
    prompt_tokens.push(next_token);

    for step in 0..max_gen {
        let pos = prompt_tokens.len() - 1;
        if pos >= kv_seq { break; }
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("forward failed");
        logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
        writeln!(out, "{next_token}").ok();
        prompt_tokens.push(next_token);
        if next_token == config.eos_token { break; }
        if step % 500 == 0 { eprintln!("  step {step:4}"); }
    }
    eprintln!("done");
}
