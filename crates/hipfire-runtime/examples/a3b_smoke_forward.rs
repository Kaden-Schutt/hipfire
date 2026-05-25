// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 1 smoke test: run one forward pass through A3B (or any qwen3_5_moe
//! HFQ) and verify logits are finite. Exercises both the MoE FFN helper and
//! the new DeltaNetMoe/FullAttnMoe match arms in forward_scratch_layers.
//!
//! Usage:
//!   cargo run --release --features deltanet --example a3b_smoke_forward -- \
//!       ~/.hipfire/models/qwen3.5-35b-a3b.mq4
//!
//!   # ParoQuant / HF safetensors directory (auto-routes by path-is-dir):
//!   cargo run --release --features deltanet --example a3b_smoke_forward -- \
//!       ~/.hipfire/models/shisa-Qwen3.6-35B-A3B-PARO-packed
//!
//!   # Optional: generate N tokens greedily to probe short-term stability.
//!   HIPFIRE_SMOKE_STEPS=8 cargo run --release ...

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: a3b_smoke_forward <model.mq4 | safetensors-dir>");
        std::process::exit(1);
    }
    let model_path = Path::new(&args[1]);
    let n_steps: usize = std::env::var("HIPFIRE_SMOKE_STEPS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);

    eprintln!("Opening: {}", model_path.display());
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    // Auto-route safetensors directories (ParoQuant / HF native) — mirrors
    // eval_hipfire.rs:165-176 and daemon.rs:1500-1504. HFQ files take the
    // canonical HFQ path below.
    let (config, weights, tokenizer) = if model_path.is_dir() {
        use hipfire_runtime::safetensors_source::SafetensorsSource;
        use hipfire_runtime::model_source::ModelSource;
        let source = SafetensorsSource::open(model_path).expect("safetensors open");
        let config = qwen35::config_from_safetensors(&source).expect("config_from_safetensors");
        assert!(config.num_experts > 0, "this smoke test expects a MoE model");
        eprintln!("A3B config: dim={}, layers={}, experts={}, top_k={}, moe_inter={}, shared_inter={}",
            config.dim, config.n_layers, config.num_experts, config.num_experts_per_tok,
            config.moe_intermediate_size, config.shared_expert_intermediate_size);
        eprintln!("Loading weights via safetensors (ParoQuant path) ...");
        let weights = qwen35::load_weights_paroquant(&source, &config, &mut gpu)
            .expect("load_weights_paroquant");
        let tok_path = source.tokenizer_json_path()
            .expect("tokenizer.json in safetensors directory");
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .expect("tokenizer parse")
            .expect("tokenizer load");
        (config, weights, tokenizer)
    } else {
        let mut hfq = HfqFile::open(model_path).expect("open model");
        let config = qwen35::config_from_hfq(&hfq).expect("read config");
        assert!(config.num_experts > 0, "this smoke test expects a MoE model");
        eprintln!("A3B config: dim={}, layers={}, experts={}, top_k={}, moe_inter={}, shared_inter={}",
            config.dim, config.n_layers, config.num_experts, config.num_experts_per_tok,
            config.moe_intermediate_size, config.shared_expert_intermediate_size);
        eprintln!("Loading weights ...");
        let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .expect("tokenizer");
        (config, weights, tokenizer)
    };
    eprintln!("Loaded {} layers.", weights.layers.len());

    let kv_seq = std::env::var("HIPFIRE_SMOKE_KV_SEQ")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(256usize);
    // Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the
    // production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit
    // KV path and always-on flash; f32 falls back to plain attention_f32.
    let kv_mode = std::env::var("HIPFIRE_SMOKE_KV").unwrap_or_else(|_| "q8".to_string());
    eprintln!("KV cache mode: {kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
        "asym2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
        _ => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).expect("kv cache alloc"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state alloc");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch alloc");

    // Prompt mode selection:
    //   raw (default for back-compat):  tokenize "Hello" and decode from pos=0.
    //     Useful for finite-logits smoke tests, but the model has no chat
    //     context, so the output will drift into random trained patterns
    //     (e.g. Wikipedia prose). Don't read too much into coherence here.
    //   chat:  wrap the prompt in the Qwen3.5 chat template
    //     (<|im_start|>user ... <|im_end|> <|im_start|>assistant ...) and
    //     prefill the full prompt before greedy decoding. Use this to
    //     sanity-check assistant-style coherence.
    let prompt_mode = std::env::var("HIPFIRE_SMOKE_MODE").unwrap_or_else(|_| "raw".to_string());
    let user_prompt = std::env::var("HIPFIRE_SMOKE_PROMPT")
        .unwrap_or_else(|_| "Hello".to_string());

    let prompt_tokens: Vec<u32> = if prompt_mode == "chat" {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user = tokenizer.encode("user");
        let asst = tokenizer.encode("assistant");
        let nl = tokenizer.encode("\n");
        let body = tokenizer.encode(&user_prompt);
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&user);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&body);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&asst);
        chat.extend_from_slice(&nl);
        chat
    } else {
        tokenizer.encode(&user_prompt)
    };
    eprintln!("Prompt ({prompt_mode} mode): {} tokens", prompt_tokens.len());

    eprintln!("\n=== forward_scratch prefill ===");
    let t0 = std::time::Instant::now();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, tok, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("forward_scratch failed");
    }
    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
    let elapsed = t0.elapsed();
    let n_prompt = prompt_tokens.len();
    let pf_us = elapsed.as_micros() as f64;
    eprintln!("prefill {} toks in {:.2} ms ({:.1} tok/s)",
        n_prompt, pf_us / 1000.0, (n_prompt as f64) * 1_000_000.0 / pf_us);

    // ─── Correctness gates ──────────────────────────────────────────────
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &logits {
        if v.is_nan() { n_nan += 1; }
        else if v.is_infinite() { n_inf += 1; }
        else {
            if v < min_v { min_v = v; }
            if v > max_v { max_v = v; }
        }
    }
    eprintln!("  logits.len = {}", logits.len());
    eprintln!("  finite range: [{:.4}, {:.4}]", min_v, max_v);
    eprintln!("  NaNs: {n_nan}  Infs: {n_inf}");
    assert!(n_nan == 0, "NaN in logits — forward path is broken");
    assert!(n_inf == 0, "Inf in logits — forward path is broken");

    // Top-5
    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    indexed.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed[..5].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  top-5 token ids: {:?}", &indexed[..5]);
    let argmax = indexed[0].0;
    eprintln!("  argmax = {argmax}  (elapsed: {:?})", elapsed);

    // ─── Optional multi-step decode ────────────────────────────────────
    if n_steps > 1 {
        eprintln!("\n=== decoding {} more tokens greedily ===", n_steps - 1);
        let mut next = argmax;
        let base_pos = prompt_tokens.len();
        let mut timings = Vec::with_capacity(n_steps.saturating_sub(1));
        for step in 1..n_steps {
            let t0 = std::time::Instant::now();
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, next, base_pos + step - 1,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward_scratch failed");
            let l = gpu.download_f32(&scratch.logits).expect("download");
            timings.push(t0.elapsed());
            let has_nan = l.iter().any(|v| v.is_nan() || v.is_infinite());
            assert!(!has_nan, "NaN/Inf at step {step}");
            next = llama::argmax(&l);
            let decoded = tokenizer.decode(&[next]);
            eprintln!("  step {step:2} -> {next:6} '{}'  ({} µs)",
                decoded.replace('\n', "\\n"), timings.last().unwrap().as_micros());
        }

        // Steady-state summary — ignore the first 2 steps (graph capture
        // and KV cache warm-up throw off early measurements).
        let settled: Vec<_> = timings.iter().skip(2).collect();
        if !settled.is_empty() {
            let sum: u128 = settled.iter().map(|d| d.as_micros()).sum();
            let avg_us = sum / settled.len() as u128;
            let tok_per_s = 1_000_000.0 / avg_us as f64;
            eprintln!("\nsteady-state decode (n={}): avg {} µs/tok = {:.1} tok/s",
                settled.len(), avg_us, tok_per_s);
        }
    }

    eprintln!("\n=== SMOKE TEST PASSED ===");
}
