// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sweep early-exit thresholds and checkpoints.
use hipfire_runtime::hfq::{self, HfqFile};
use hipfire_runtime::llama::{self, KvCache, ForwardScratch};
use std::path::Path;
use std::time::Instant;

fn run_test(gpu: &mut rdna_compute::Gpu, weights: &hipfire_runtime::llama::LlamaWeights, config: &hipfire_runtime::llama::LlamaConfig,
            threshold: f32, checkpoints: &[usize], label: &str) {
    let n_gen = 50;
    let mut kv = KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, 2048).unwrap();
    let scratch = ForwardScratch::new(gpu, config).unwrap();
    let mut rng = 42u32;
    let prompt = vec![9906u32];
    for (pos, &tok) in prompt.iter().enumerate() {
        let (_, r, _) = llama::forward_early_exit(gpu, weights, config, tok, pos, &mut kv, &scratch, 0.01, 0.8, rng, 0, 1.0, threshold, checkpoints).unwrap();
        rng = r;
    }
    let t0 = Instant::now();
    let mut exits = 0;
    let mut first_tokens = Vec::new();
    for i in 0..n_gen {
        let pos = prompt.len() + i;
        let mut out = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out, &scratch.sample_buf.buf).unwrap();
        let tok = u32::from_ne_bytes([out[0], out[1], out[2], out[3]]);
        if first_tokens.len() < 5 { first_tokens.push(tok); }
        let (_, r, el) = llama::forward_early_exit(gpu, weights, config, tok, pos, &mut kv, &scratch, 0.01, 0.8, rng, 0, 1.0, threshold, checkpoints).unwrap();
        rng = r;
        if el < config.n_layers { exits += 1; }
    }
    let ms = t0.elapsed().as_millis();
    let tps = n_gen as f64 / (ms as f64 / 1000.0);
    let pct = exits as f64 / n_gen as f64 * 100.0;
    eprintln!("| {:<25} | {:5.1} tok/s | {:2} exits ({:4.0}%) | {:?}", label, tps, exits, pct, first_tokens);
}

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| { eprintln!("Usage: sweep_exit <model.hfq>"); std::process::exit(1); });
    let hfq = HfqFile::open(Path::new(&model_path)).expect("parse");
    let config = hfq::config_from_hfq(&hfq).expect("config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    let weights = hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("weights");
    let cp1 = config.n_layers / 3;      // layer 12
    let cp2 = config.n_layers * 2 / 3;  // layer 24

    eprintln!("Qwen3-8B early-exit sweep ({} layers, checkpoints at L{} and L{})", config.n_layers, cp1, cp2);
    eprintln!("| Config                    | Speed      | Exits          | First 5 tokens");
    eprintln!("|---------------------------|------------|----------------|---------------");
    run_test(&mut gpu, &weights, &config, 0.0, &[], "baseline (no exit)");
    run_test(&mut gpu, &weights, &config, 0.95, &[cp1], "L12, t=0.95");
    run_test(&mut gpu, &weights, &config, 0.9, &[cp1], "L12, t=0.90");
    run_test(&mut gpu, &weights, &config, 0.8, &[cp1], "L12, t=0.80");
    run_test(&mut gpu, &weights, &config, 0.7, &[cp1], "L12, t=0.70");
    run_test(&mut gpu, &weights, &config, 0.6, &[cp1], "L12, t=0.60");
    run_test(&mut gpu, &weights, &config, 0.5, &[cp1], "L12, t=0.50");
    run_test(&mut gpu, &weights, &config, 0.8, &[cp1, cp2], "L12+L24, t=0.80");
    run_test(&mut gpu, &weights, &config, 0.7, &[cp1, cp2], "L12+L24, t=0.70");
    run_test(&mut gpu, &weights, &config, 0.6, &[cp1, cp2], "L12+L24, t=0.60");
    run_test(&mut gpu, &weights, &config, 0.8, &[cp2], "L24 only, t=0.80");
    run_test(&mut gpu, &weights, &config, 0.7, &[cp2], "L24 only, t=0.70");
}
