// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throughput micro-bench for ZAYA1: prefill latency + greedy decode tokens/sec.
//!
//! Run: cargo run --release -p hipfire-arch-zaya --example bench -- <hfq> [ntok]

use hipfire_arch_zaya::arch::ZayaModel;
use hipfire_arch_zaya::ZayaConfig;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
            if x > b.1 {
                (i, x)
            } else {
                b
            }
        })
        .0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let hfq_path = args
        .next()
        .unwrap_or_else(|| "/home/sadara/zaya1-8b-native.mq4.hfq".to_string());
    let ntok: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);

    let hfq = HfqFile::open(Path::new(&hfq_path)).expect("open hfq");
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg = ZayaConfig::from_json(meta.get("config").unwrap_or(&meta)).expect("config");

    let mut gpu = Gpu::init().expect("gpu init");
    let t_load = Instant::now();
    let mut model = ZayaModel::from_hfq(&mut gpu, &hfq, cfg, 4096).expect("load");
    let load_s = t_load.elapsed().as_secs_f64();

    // Warm prompt (kernel JIT) before timing.
    let prompt: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    model.prefill(&mut gpu, &prompt).expect("warm prefill");
    let _ = gpu.download_f32(model.logits()).expect("warm sync");

    // Timed prefill.
    let t_pf = Instant::now();
    model.prefill(&mut gpu, &prompt).expect("prefill");
    let _ = gpu.download_f32(model.logits()).expect("sync");
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;

    // Timed greedy decode.
    let t_dec = Instant::now();
    let mut pos = prompt.len();
    for _ in 0..ntok {
        let logits = gpu.download_f32(model.logits()).expect("logits");
        let next = argmax(&logits) as u32;
        model.decode_step(&mut gpu, next, pos).expect("decode");
        pos += 1;
    }
    let dec_s = t_dec.elapsed().as_secs_f64();

    println!(
        "model       : {hfq_path} ({:.2} GB on disk)",
        std::fs::metadata(&hfq_path)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0)
    );
    println!("load        : {load_s:.1} s");
    println!(
        "prefill     : {prefill_ms:.1} ms for {} prompt tokens",
        prompt.len()
    );
    println!(
        "decode      : {ntok} tokens in {dec_s:.2} s = {:.2} tok/s ({:.1} ms/tok)",
        ntok as f64 / dec_s,
        dec_s * 1000.0 / ntok as f64
    );
}
