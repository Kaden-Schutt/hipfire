//! Phase 3 smoke test: run a Gemma-4-31B forward pass and verify logits are
//! finite. Exercises both layer-type match arms (Sliding + Full) in
//! `gemma4::forward_scratch`, the two KV caches, RoPE (default + proportional),
//! the new rope_partial_halved kernel, sandwich norms, SwiGLU/gelu_tanh, the
//! learned per-layer scalar, and the final logit softcap.
//!
//! Usage:
//!   cargo run --release --features deltanet --example gemma4_smoke_forward -- \
//!       ~/.hipfire/models/gemma-4-31b/gemma-4-31b.mq4
//!
//!   # Optional env:
//!   HIPFIRE_SMOKE_STEPS=8     — greedy-decode N extra tokens after prefill
//!   HIPFIRE_SMOKE_PROMPT=Hi   — custom raw prompt (no chat template yet)
//!   HIPFIRE_SMOKE_KV=asym3    — KV cache quant: asym3 (default) / asym4 / asym2 / q8
//!   HIPFIRE_SMOKE_KV_SEQ=512  — per-cache max seq (full layers)
//!                                (sliding cache is capped at sliding_window=1024)

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::gemma4::{self, Gemma4Scratch};
    use engine::llama::{self, KvCache};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gemma4_smoke_forward <model.mq4>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let n_steps: usize = std::env::var("HIPFIRE_SMOKE_STEPS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);

    eprintln!("Opening: {model_path}");
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    assert_eq!(hfq.arch_id, 7, "expected arch_id=7 (Gemma 4), got {}", hfq.arch_id);
    let config = gemma4::config_from_hfq(&hfq).expect("read config");

    let n_sliding = config.layer_types.iter()
        .filter(|&&t| t == gemma4::LayerType::Sliding).count();
    let n_full = config.layer_types.iter()
        .filter(|&&t| t == gemma4::LayerType::Full).count();
    eprintln!(
        "Gemma 4 config: dim={}, layers={} ({} sliding + {} full), vocab={}, \
         sliding_hd={}, full_hd={}, n_heads={}, softcap={}, tie={}",
        config.dim, config.n_layers, n_sliding, n_full, config.vocab_size,
        config.sliding_head_dim, config.full_head_dim, config.n_heads,
        config.final_logit_softcapping, config.tie_word_embeddings,
    );

    eprintln!("Loading weights...");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = gemma4::load_weights(&hfq, &config, &mut gpu).expect("load weights");

    let kv_seq = std::env::var("HIPFIRE_SMOKE_KV_SEQ")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(256usize);
    let kv_mode = std::env::var("HIPFIRE_SMOKE_KV").unwrap_or_else(|_| "asym3".to_string());
    // Sliding cache must hold every written position (write kernel addresses
    // absolute pos); sliding_window only gates the *read* side. Full KV is
    // forced to FP32 because the quantized flash kernels truncate head_dim>256.
    eprintln!("KV cache: sliding={kv_mode} @ {kv_seq} / full=fp32 @ {kv_seq}");

    let mut kv_sliding = match kv_mode.as_str() {
        "asym4" => KvCache::new_gpu_asym4(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, kv_seq),
        "asym2" => KvCache::new_gpu_asym2(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, kv_seq),
        "q8"    => KvCache::new_gpu_q8(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, kv_seq),
        _       => KvCache::new_gpu_asym3(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, kv_seq),
    }.expect("kv sliding alloc");
    let mut kv_full = KvCache::new_gpu(&mut gpu, n_full, config.full_n_kv_heads, config.full_head_dim, kv_seq)
        .expect("kv full alloc");

    let scratch = Gemma4Scratch::new(&mut gpu, &config, 64).expect("scratch alloc");
    gemma4::init_scratch_constants(&mut gpu, &scratch, config.full_head_dim)
        .expect("scratch constants init");

    // Raw-prompt mode: no chat template yet (Phase 10). Just tokenize + forward.
    // BOS is required — HF's Gemma 4 tokenizer always prepends `<bos>` (id=2).
    let user_prompt = std::env::var("HIPFIRE_SMOKE_PROMPT")
        .unwrap_or_else(|_| "Hello".to_string());
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut prompt_tokens: Vec<u32> = vec![config.bos_token];
    prompt_tokens.extend(tokenizer.encode(&user_prompt));
    eprintln!("Prompt: {} tokens (inc. BOS): {:?}",
        prompt_tokens.len(),
        &prompt_tokens[..prompt_tokens.len().min(16)]);

    eprintln!("\n=== forward_scratch (per-token) ===");
    let t0 = std::time::Instant::now();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        gemma4::forward_scratch(
            &mut gpu, &weights, &config, tok, pos,
            &mut kv_sliding, &mut kv_full, &scratch,
        ).expect("forward_scratch failed");
    }
    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
    let elapsed = t0.elapsed();
    let n_prompt = prompt_tokens.len();
    let pf_us = elapsed.as_micros() as f64;
    eprintln!(
        "processed {} prompt toks in {:.2} ms ({:.1} tok/s)",
        n_prompt, pf_us / 1000.0, (n_prompt as f64) * 1_000_000.0 / pf_us,
    );

    // Correctness gates.
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
    eprintln!("  logits.len = {}  (expected {})", logits.len(), config.vocab_size);
    eprintln!("  finite range: [{:.4}, {:.4}]", min_v, max_v);
    eprintln!("  NaNs: {n_nan}  Infs: {n_inf}");
    assert!(n_nan == 0, "NaN in logits — forward path is broken");
    assert!(n_inf == 0, "Inf in logits — forward path is broken");
    // Soft-cap sanity: logits should be within ±cap.
    if config.final_logit_softcapping > 0.0 {
        let cap = config.final_logit_softcapping;
        assert!(min_v.abs() <= cap + 1e-3, "logit {min_v} below -cap {cap}");
        assert!(max_v.abs() <= cap + 1e-3, "logit {max_v} above +cap {cap}");
        eprintln!("  softcap check: |logit| <= {cap} — PASS");
    }

    // Top-5 argmax.
    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    indexed.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed[..5].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  top-5 tokens: {:?}", &indexed[..5]);
    let argmax = indexed[0].0;
    eprintln!("  argmax = {argmax}  decoded = '{}'  (elapsed: {:?})",
        tokenizer.decode(&[argmax]).replace('\n', "\\n"), elapsed);

    if n_steps > 1 {
        eprintln!("\n=== decoding {} more tokens greedily ===", n_steps - 1);
        let mut next = argmax;
        let base_pos = prompt_tokens.len();
        let mut timings = Vec::with_capacity(n_steps.saturating_sub(1));
        for step in 1..n_steps {
            let t0 = std::time::Instant::now();
            gemma4::forward_scratch(
                &mut gpu, &weights, &config, next, base_pos + step - 1,
                &mut kv_sliding, &mut kv_full, &scratch,
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
