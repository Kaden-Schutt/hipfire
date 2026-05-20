// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Benchmark: early-exit forward pass vs baseline.
//! Measures exit rate, speed, and quality impact.
//! Usage: bench_early_exit <model.hfq>

use hipfire_runtime::hfq::{self, HfqFile};
use hipfire_runtime::llama::{self, KvCache, ForwardScratch};
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: bench_early_exit <model.hfq>");
        std::process::exit(1);
    });

    let hfq = HfqFile::open(Path::new(&model_path)).expect("failed to parse HFQ");
    let config = hfq::config_from_hfq(&hfq).expect("failed to read config");
    eprintln!("Config: dim={}, layers={}, heads={}", config.dim, config.n_layers, config.n_heads);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let weights = hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("failed to load weights");

    let n_gen = 50;

    // === Test 1: Baseline (forward_scratch) ===
    eprintln!("\n=== Test 1: Baseline forward_scratch ===");
    {
        let mut kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 2048).unwrap();
        let scratch = ForwardScratch::new(&mut gpu, &config).unwrap();
        let mut rng = 42u32;

        // Prefill "Hello"
        let prompt = vec![9906u32]; // "Hello" token (approximate)
        for (pos, &tok) in prompt.iter().enumerate() {
            let (_, r) = llama::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, rng, 0, 1.0).unwrap();
            rng = r;
        }

        let t0 = Instant::now();
        let mut tokens = Vec::new();
        let mut next_rng = rng;
        for i in 0..n_gen {
            let pos = prompt.len() + i;
            let mut out_bytes = [0u8; 8];
            gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
            let tok = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
            tokens.push(tok);
            let (_, r) = llama::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, next_rng, 0, 1.0).unwrap();
            next_rng = r;
        }
        let ms = t0.elapsed().as_millis();
        eprintln!("Baseline: {} tokens in {}ms ({:.1} tok/s)", n_gen, ms, n_gen as f64 / (ms as f64 / 1000.0));
        eprintln!("Tokens: {:?}", &tokens[..10.min(tokens.len())]);
    }

    // === Test 2: Early exit with threshold=0.0 (should NEVER exit, match baseline) ===
    eprintln!("\n=== Test 2: Early exit, threshold=0.0 (never exits) ===");
    {
        let mut kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 2048).unwrap();
        let scratch = ForwardScratch::new(&mut gpu, &config).unwrap();
        let mut rng = 42u32;
        let prompt = vec![9906u32];
        for (pos, &tok) in prompt.iter().enumerate() {
            let (_, r, _) = llama::forward_early_exit(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, rng, 0, 1.0, 0.0, &[]).unwrap();
            rng = r;
        }
        let t0 = Instant::now();
        let mut tokens = Vec::new();
        let mut exits = 0;
        let mut next_rng = rng;
        for i in 0..n_gen {
            let pos = prompt.len() + i;
            let mut out_bytes = [0u8; 8];
            gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
            let tok = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
            tokens.push(tok);
            let (_, r, el) = llama::forward_early_exit(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, next_rng, 0, 1.0, 0.0, &[]).unwrap();
            next_rng = r;
            if el < config.n_layers { exits += 1; }
        }
        let ms = t0.elapsed().as_millis();
        eprintln!("No-exit: {} tokens in {}ms ({:.1} tok/s), exits: {}", n_gen, ms, n_gen as f64 / (ms as f64 / 1000.0), exits);
        eprintln!("Tokens: {:?}", &tokens[..10.min(tokens.len())]);
    }

    // === Test 3: Early exit at layer n_layers/3 with threshold=0.9 ===
    eprintln!("\n=== Test 3: Early exit at layer {}, threshold=0.9 ===", config.n_layers / 3);
    {
        let checkpoint = config.n_layers / 3; // layer 12 for 36-layer model
        let mut kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 2048).unwrap();
        let scratch = ForwardScratch::new(&mut gpu, &config).unwrap();
        let mut rng = 42u32;
        let prompt = vec![9906u32];
        for (pos, &tok) in prompt.iter().enumerate() {
            let (_, r, _) = llama::forward_early_exit(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, rng, 0, 1.0, 0.9, &[checkpoint]).unwrap();
            rng = r;
        }
        let t0 = Instant::now();
        let mut tokens = Vec::new();
        let mut exits = 0;
        let mut next_rng = rng;
        for i in 0..n_gen {
            let pos = prompt.len() + i;
            let mut out_bytes = [0u8; 8];
            gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
            let tok = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
            tokens.push(tok);
            let (_, r, el) = llama::forward_early_exit(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, next_rng, 0, 1.0, 0.9, &[checkpoint]).unwrap();
            next_rng = r;
            if el < config.n_layers { exits += 1; }
        }
        let ms = t0.elapsed().as_millis();
        let exit_pct = exits as f64 / n_gen as f64 * 100.0;
        eprintln!("Exit@L{}: {} tokens in {}ms ({:.1} tok/s), exits: {} ({:.0}%)",
            checkpoint, n_gen, ms, n_gen as f64 / (ms as f64 / 1000.0), exits, exit_pct);
        eprintln!("Tokens: {:?}", &tokens[..10.min(tokens.len())]);
    }

    // === Test 4: Early exit at layer n_layers/3 with threshold=0.5 (more aggressive) ===
    eprintln!("\n=== Test 4: Early exit at layer {}, threshold=0.5 ===", config.n_layers / 3);
    {
        let checkpoint = config.n_layers / 3;
        let mut kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 2048).unwrap();
        let scratch = ForwardScratch::new(&mut gpu, &config).unwrap();
        let mut rng = 42u32;
        let prompt = vec![9906u32];
        for (pos, &tok) in prompt.iter().enumerate() {
            let (_, r, _) = llama::forward_early_exit(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, rng, 0, 1.0, 0.5, &[checkpoint]).unwrap();
            rng = r;
        }
        let t0 = Instant::now();
        let mut tokens = Vec::new();
        let mut exits = 0;
        let mut next_rng = rng;
        for i in 0..n_gen {
            let pos = prompt.len() + i;
            let mut out_bytes = [0u8; 8];
            gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
            let tok = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
            tokens.push(tok);
            let (_, r, el) = llama::forward_early_exit(&mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.01, 0.8, next_rng, 0, 1.0, 0.5, &[checkpoint]).unwrap();
            next_rng = r;
            if el < config.n_layers { exits += 1; }
        }
        let ms = t0.elapsed().as_millis();
        let exit_pct = exits as f64 / n_gen as f64 * 100.0;
        eprintln!("Exit@L{}: {} tokens in {}ms ({:.1} tok/s), exits: {} ({:.0}%)",
            checkpoint, n_gen, ms, n_gen as f64 / (ms as f64 / 1000.0), exits, exit_pct);
        eprintln!("Tokens: {:?}", &tokens[..10.min(tokens.len())]);
    }
}
