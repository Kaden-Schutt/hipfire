// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inference integration tests. Runs actual token sequences through the
//! full forward path and checks for correctness, hangs, and speed.
//! No model quality checks — just verifies the engine doesn't break.
//! Usage: cargo run --release --features deltanet --example test_inference -- <model.hfq>

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::DeltaNetState;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: test_inference <model.hfq>");
        std::process::exit(1);
    });

    eprintln!("=== hipfire inference integration tests ===");
    eprintln!("Model: {model_path}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not found");

    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, hd={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.head_dim);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("failed to load weights");

    let mut passed = 0;
    let mut failed = 0;

    macro_rules! test {
        ($name:expr, $timeout_ms:expr, $body:expr) => {{
            eprint!("  {:60} ", $name);
            let t = Instant::now();
            let mut closure = || -> Result<String, String> { $body };
            match closure() {
                Ok(msg) => {
                    let ms = t.elapsed().as_millis();
                    if ms > $timeout_ms {
                        failed += 1; eprintln!("SLOW ({ms}ms > {}ms limit)", $timeout_ms);
                    } else {
                        passed += 1; eprintln!("OK ({ms}ms) {msg}");
                    }
                }
                Err(e) => { failed += 1; eprintln!("FAIL: {e}"); }
            }
        }};
    }

    // Test 1: Forward produces finite logits
    eprintln!("\n--- Forward path ---");
    test!("forward() produces finite logits", 10000, {
        let kv_seq = 128;
        let mut kv = llama::KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).map_err(|e| format!("{e}"))?;
        let mut dn = DeltaNetState::new(&mut gpu, &config).map_err(|e| format!("{e}"))?;
        let logits = qwen35::forward(&mut gpu, &weights, &config, 1, 0, &mut kv, &mut dn).map_err(|e| format!("{e}"))?;
        assert_eq!(logits.len(), config.vocab_size, "logits len mismatch");
        assert!(logits[0].is_finite(), "logits[0] is NaN/Inf: {}", logits[0]);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max > -1e10, "logits all very negative: max={max}");
        Ok(format!("logits[0]={:.4} max={:.4}", logits[0], max))
    });

    // Test 2: forward_scratch() matches forward()
    test!("forward_scratch() matches forward()", 10000, {
        let kv_seq = 128;
        let mut kv1 = llama::KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).map_err(|e| format!("{e}"))?;
        let mut dn1 = DeltaNetState::new(&mut gpu, &config).map_err(|e| format!("{e}"))?;
        let mut kv2 = llama::KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).map_err(|e| format!("{e}"))?;
        let mut dn2 = DeltaNetState::new(&mut gpu, &config).map_err(|e| format!("{e}"))?;
        let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &config, 64).map_err(|e| format!("{e}"))?;

        let tok = tokenizer.encode("Hello")[0];
        let logits_a = qwen35::forward(&mut gpu, &weights, &config, tok, 0, &mut kv1, &mut dn1).map_err(|e| format!("{e}"))?;
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, 0, &mut kv2, &mut dn2, &scratch).map_err(|e| format!("{e}"))?;
        let logits_b = gpu.download_f32(&scratch.logits).map_err(|e| format!("{e}"))?;

        let max_diff = logits_a.iter().zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.001, "forward vs scratch diverged: max_diff={max_diff}");
        Ok(format!("max_diff={max_diff:.6}"))
    });

    // Test 3: Multi-token sequence doesn't hang
    test!("10-token sequence completes (no hang)", 15000, {
        let kv_seq = 128;
        let mut kv = llama::KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).map_err(|e| format!("{e}"))?;
        let mut dn = DeltaNetState::new(&mut gpu, &config).map_err(|e| format!("{e}"))?;
        let tokens = tokenizer.encode("What is the capital");
        let t0 = Instant::now();
        for (pos, &tok) in tokens.iter().enumerate() {
            let _ = qwen35::forward(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn).map_err(|e| format!("{e}"))?;
        }
        let ms = t0.elapsed().as_millis();
        let tps = tokens.len() as f64 / (ms as f64 / 1000.0);
        Ok(format!("{} tokens in {ms}ms ({tps:.0} tok/s)", tokens.len()))
    });

    // Test 4: </think> token is detectable
    test!("</think> encodes to detectable token(s)", 1000, {
        let think_end = tokenizer.encode("</think>");
        assert!(!think_end.is_empty(), "</think> encoded to empty");
        Ok(format!("{} token(s): {:?}", think_end.len(), think_end))
    });

    // Test 5: ChatML tokens encode correctly
    test!("ChatML special tokens encode as single tokens", 1000, {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        assert_eq!(im_start.len(), 1, "<|im_start|> is {} tokens: {:?}", im_start.len(), im_start);
        assert_eq!(im_end.len(), 1, "<|im_end|> is {} tokens: {:?}", im_end.len(), im_end);
        Ok(format!("im_start={} im_end={}", im_start[0], im_end[0]))
    });

    // Test 6: Givens4 KV cache allocates correctly
    test!("givens4 KV cache allocates", 5000, {
        let kv_seq = 128;
        let kv = llama::KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).map_err(|e| format!("{e}"))?;
        assert!(kv.quant_asym3, "quant_asym3 should be true");
        assert!(kv.givens_cos.is_some(), "givens_cos missing");
        assert!(kv.givens_sin.is_some(), "givens_sin missing");
        Ok(format!("givens4 allocated for {} layers, hd={}", config.n_layers, config.head_dim))
    });

    // Test 7: Givens4 forward doesn't hang
    test!("givens4 forward completes (no hang)", 15000, {
        let kv_seq = 128;
        let mut kv = llama::KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).map_err(|e| format!("{e}"))?;
        let mut dn = DeltaNetState::new(&mut gpu, &config).map_err(|e| format!("{e}"))?;
        let tokens = tokenizer.encode("Hello world");
        let t0 = Instant::now();
        for (pos, &tok) in tokens.iter().enumerate() {
            let _ = qwen35::forward(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn).map_err(|e| format!("{e}"))?;
        }
        let ms = t0.elapsed().as_millis();
        Ok(format!("{} tokens in {ms}ms", tokens.len()))
    });

    // Test 8: Speed sanity check (should be >10 tok/s for any model)
    test!("decode speed > 10 tok/s", 30000, {
        let kv_seq = 256;
        let mut kv = llama::KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).map_err(|e| format!("{e}"))?;
        let mut dn = DeltaNetState::new(&mut gpu, &config).map_err(|e| format!("{e}"))?;
        // Prefill
        let prompt = tokenizer.encode("The quick brown fox");
        for (pos, &tok) in prompt.iter().enumerate() {
            let _ = qwen35::forward(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn).map_err(|e| format!("{e}"))?;
        }
        // Generate 20 tokens
        let mut tok = 1u32;
        let t0 = Instant::now();
        for i in 0..20 {
            let logits = qwen35::forward(&mut gpu, &weights, &config, tok, prompt.len() + i, &mut kv, &mut dn).map_err(|e| format!("{e}"))?;
            tok = llama::argmax(&logits);
        }
        let ms = t0.elapsed().as_millis();
        let tps = 20.0 / (ms as f64 / 1000.0);
        assert!(tps > 10.0, "decode too slow: {tps:.1} tok/s");
        Ok(format!("{tps:.1} tok/s"))
    });

    // Test 9: VRAM leak detection — alloc/free cycle should return to baseline
    eprintln!("\n--- VRAM lifecycle ---");
    // VRAM leak tests — CRITICAL for model eviction in Bun CLI daemon
    // BUG: DeviceBuffer has no Drop impl. GPU memory is NEVER freed unless
    // gpu.free_tensor() is explicitly called. This means every KvCache,
    // DeltaNetState, Scratch, and VisionWeights that goes out of scope leaks.
    // These tests document the current state and will FAIL until Drop is implemented.
    test!("VRAM: KV cache free_gpu + drain returns memory", 10000, {
        let (free_before, _) = gpu.hip.get_vram_info().map_err(|e| format!("{e}"))?;
        let kv = llama::KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 512
        ).map_err(|e| format!("{e}"))?;
        let (free_during, _) = gpu.hip.get_vram_info().map_err(|e| format!("{e}"))?;
        let alloc_mb = (free_before - free_during) as f64 / 1e6;
        assert!(alloc_mb > 1.0, "KV cache should use >1MB, got {alloc_mb:.1}MB");
        // Explicit free + drain
        kv.free_gpu(&mut gpu);
        gpu.drain_pool();
        let (free_after, _) = gpu.hip.get_vram_info().map_err(|e| format!("{e}"))?;
        let leak_mb = (free_before as i64 - free_after as i64) as f64 / 1e6;
        assert!(leak_mb.abs() < 2.0, "VRAM leak: {leak_mb:.1}MB after free_gpu+drain");
        Ok(format!("alloc={alloc_mb:.1}MB, leak={leak_mb:.2}MB"))
    });

    eprintln!("\n--- Summary ---");
    eprintln!("  Passed:  {passed}");
    eprintln!("  Failed:  {failed}");
    if failed > 0 { std::process::exit(1); }
}
