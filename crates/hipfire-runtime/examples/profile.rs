// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Profile: measure time breakdown within a single forward pass.

use hipfire_runtime::gguf::GgufFile;
use hipfire_runtime::llama::{self, LlamaConfig, KvCache};
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = GgufFile::open(Path::new(model_path)).unwrap();
    let config = LlamaConfig::from_gguf(&gguf).unwrap();
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    let weights = llama::load_weights(&gguf, &config, &mut gpu).unwrap();
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv_cache = KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap();

    // Warmup
    let _ = llama::forward(&mut gpu, &weights, &config, 1, 0, &mut kv_cache);
    let _ = llama::forward(&mut gpu, &weights, &config, 15043, 1, &mut kv_cache);

    // Profile 10 forward passes, averaging
    let n = 10;
    let mut total = 0.0f64;
    for i in 0..n {
        let t = Instant::now();
        let _ = llama::forward(&mut gpu, &weights, &config, 15043, 2 + i, &mut kv_cache);
        total += t.elapsed().as_secs_f64() * 1000.0;
    }
    let avg = total / n as f64;

    eprintln!("Average forward pass: {avg:.1}ms ({n} iterations)");
    eprintln!();

    // Now measure individual components in isolation
    let dim = config.dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let q_dim = config.n_heads * config.head_dim;

    // 1. hipMalloc + hipFree overhead
    let t = Instant::now();
    for _ in 0..100 {
        let buf = gpu.zeros(&[dim], rdna_compute::DType::F32).unwrap();
        gpu.free_tensor(buf).unwrap();
    }
    let malloc_us = t.elapsed().as_secs_f64() * 1e6 / 100.0;
    eprintln!("hipMalloc+Free ({}B): {:.0}us/pair", dim * 4, malloc_us);

    // Count allocs per forward pass: ~20 per layer × 22 layers + a few global = ~450
    let allocs_per_fwd = 20 * config.n_layers + 5;
    let malloc_total_ms = malloc_us * allocs_per_fwd as f64 / 1000.0;
    eprintln!("  Estimated malloc overhead per forward: {malloc_total_ms:.1}ms ({allocs_per_fwd} allocs)");

    // 2. Embedding download (the full vocab table)
    let t = Instant::now();
    for _ in 0..10 {
        let _ = gpu.download_f32(&weights.token_embd);
    }
    let embd_ms = t.elapsed().as_secs_f64() * 1000.0 / 10.0;
    eprintln!("Embedding download ({}MB): {:.1}ms", config.vocab_size * dim * 4 / 1_000_000, embd_ms);

    // 3. Single GEMV (quantized)
    let tmp = gpu.zeros(&[dim], rdna_compute::DType::F32).unwrap();
    let q = gpu.zeros(&[q_dim], rdna_compute::DType::F32).unwrap();
    let t = Instant::now();
    for _ in 0..100 {
        llama::gemv_family().run_auto(&llama::DispatchCtx::new(&mut gpu), &mut gpu, &weights.layers[0].wq.dispatch_ref(), &tmp, &q).unwrap();
    }
    let gemv_ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    eprintln!("Single GEMV wq ({}x{}): {:.2}ms", q_dim, dim, gemv_ms);
    // Per layer: 7 GEMVs (wq, wk, wv, wo, gate, up, down)
    let gemv_total_ms = gemv_ms * 7.0 * config.n_layers as f64;
    // But gate/up/down are larger — estimate 2x for ffn
    let gemv_est_ms = gemv_ms * (4.0 + 3.0 * 2.5) * config.n_layers as f64;
    eprintln!("  Estimated GEMV total per forward: {gemv_est_ms:.1}ms");

    // 4. Q/K download for RoPE (per layer)
    let k_buf = gpu.zeros(&[kv_dim], rdna_compute::DType::F32).unwrap();
    let t = Instant::now();
    for _ in 0..100 {
        let _ = gpu.download_f32(&q);
        let _ = gpu.download_f32(&k_buf);
    }
    let dl_ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    eprintln!("Q+K download ({:.0}KB): {:.2}ms", (q_dim + kv_dim) as f64 * 4.0 / 1024.0, dl_ms);
    let rope_dl_total = dl_ms * config.n_layers as f64;
    eprintln!("  Download total per forward (×{}): {:.1}ms", config.n_layers, rope_dl_total);

    // 5. RoPE CPU computation
    let mut q_data = vec![0.1f32; q_dim];
    let mut k_data = vec![0.1f32; kv_dim];
    let t = Instant::now();
    for _ in 0..1000 {
        llama::apply_rope_cpu_pub(&mut q_data, config.n_heads, config.head_dim, 10);
        llama::apply_rope_cpu_pub(&mut k_data, config.n_kv_heads, config.head_dim, 10);
    }
    let rope_us = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    eprintln!("RoPE CPU compute: {:.1}us/call", rope_us);

    // 6. Upload after RoPE
    let t = Instant::now();
    for _ in 0..100 {
        let up = gpu.upload_f32(&q_data, &[q_dim]).unwrap();
        gpu.free_tensor(up).unwrap();
    }
    let upload_ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    eprintln!("Q upload+free ({:.0}KB): {:.2}ms", q_dim as f64 * 4.0 / 1024.0, upload_ms);
    let upload_total = upload_ms * config.n_layers as f64;
    eprintln!("  Upload total per forward (×{}): {:.1}ms", config.n_layers, upload_total);

    // 7. KV cache write (memcpy_htod_offset)
    let v_data = vec![0.1f32; kv_dim];
    let t = Instant::now();
    for _ in 0..100 {
        kv_cache.store_kv_pub(&gpu, 0, 5, &k_data, &v_data).unwrap();
    }
    let kv_ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    eprintln!("KV cache write ({:.0}KB): {:.2}ms", kv_dim as f64 * 4.0 * 2.0 / 1024.0, kv_ms);

    // Summary
    eprintln!("\n=== Time Budget Estimate (per forward pass) ===");
    eprintln!("  Embedding download:  {embd_ms:.1}ms");
    eprintln!("  GEMV total:          {gemv_est_ms:.1}ms");
    eprintln!("  Q/K/V download:      {rope_dl_total:.1}ms");
    eprintln!("  RoPE CPU:            {:.1}ms", rope_us * config.n_layers as f64 / 1000.0);
    eprintln!("  Q upload:            {upload_total:.1}ms");
    eprintln!("  hipMalloc overhead:  {malloc_total_ms:.1}ms");
    let accounted = embd_ms + gemv_est_ms + rope_dl_total + rope_us * config.n_layers as f64 / 1000.0 + upload_total + malloc_total_ms;
    eprintln!("  Accounted:           {accounted:.1}ms");
    eprintln!("  Actual:              {avg:.1}ms");
    eprintln!("  Unaccounted:         {:.1}ms", avg - accounted);
}
