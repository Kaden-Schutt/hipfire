// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sweep Muse Glimmer's prefill projection shapes across chunk widths.
//!
//! Answers "what chunk B should Glimmer prefill use, and which projection
//! shapes are leaving throughput on the floor" — with synthetic MQ4G256
//! weights, so it needs no model and fits any card.
//!
//! Per-layer FLOP weights (what actually matters for the end-to-end number):
//! gate/up/down are 82.4% of layer params, attention projections 17.6%.
//!
//! Usage: bench_glimmer_prefill_shapes [B ...]

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn build_hfq4g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let bpr = gpr * 136;
    let mut out = vec![0u8; m * bpr];
    let mix = |x: u64| {
        let h = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;
    for row in 0..m {
        for g in 0..gpr {
            let off = row * bpr + g * 136;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.01 + (((r1 as u32) % 4001) as f32) * 1e-5;
            let zero = (((r2 as u32) % 1500) as f32) * 1e-4 - 0.075;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte_i in 0..128 {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (byte_i as u64));
                out[off + 8 + byte_i] = (r & 0xff) as u8;
            }
        }
    }
    out
}

fn main() {
    let batches: Vec<usize> = {
        let a: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|s| s.parse().ok())
            .collect();
        if a.is_empty() {
            vec![64, 128, 256, 384, 512, 768, 1024]
        } else {
            a
        }
    };

    // (label, M, K, share of per-layer FLOPs)
    let dim = 6656usize;
    let ffn = 19968usize;
    let shapes: Vec<(&str, usize, usize)> = vec![
        ("q_proj", 4096, dim),
        ("k_proj", 256, dim),
        ("v_proj", 256, dim),
        ("attn_gate", 4096, dim),
        ("qkvg_fused", 4096 + 256 + 256 + 4096, dim),
        ("o_proj", dim, 4096),
        ("gate_proj", ffn, dim),
        ("up_proj", ffn, dim),
        ("down_proj", dim, ffn),
    ];

    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch: {}", gpu.arch_caps.arch());
    println!(
        "{:<12} {:>6} {:>10} {:>10} {:>9}",
        "shape", "B", "ms", "TFLOP/s", "M"
    );

    for (label, m, k) in &shapes {
        let w = gpu
            .upload_raw(&build_hfq4g256(*m, *k, 0xA7), &[*m, *k])
            .expect("upload weight");
        for &b in &batches {
            let x_f32: Vec<f32> = (0..b * k)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let x = gpu.upload_f32(&x_f32, &[b, *k]).expect("upload x");
            let y = gpu.alloc_tensor(&[b, *m], DType::F32).expect("alloc y");

            // warm-up (JIT + any first-call cache fill), discarded
            for _ in 0..2 {
                let _ = gpu.hip.memset(&y.buf, 0, b * m * 4);
                let _ = gpu.gemm_hfq4g256_residual(&w, &x, &y, *m, *k, b);
            }
            let _ = gpu.hip.device_synchronize();

            let iters = if b >= 512 { 6 } else { 12 };
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = gpu.hip.memset(&y.buf, 0, b * m * 4);
                gpu.gemm_hfq4g256_residual(&w, &x, &y, *m, *k, b)
                    .expect("gemm");
            }
            let _ = gpu.hip.device_synchronize();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let tflops = 2.0 * (*m as f64) * (*k as f64) * (b as f64) / (ms / 1000.0) / 1e12;
            println!("{:<12} {:>6} {:>10.3} {:>10.2} {:>9}", label, b, ms, tflops, m);

            let _ = gpu.free_tensor(y);
            let _ = gpu.free_tensor(x);
        }
        let _ = gpu.free_tensor(w);
    }
}
