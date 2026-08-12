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

    // Sustained warm-up. A 2-iteration warm-up is NOT enough: the first measured
    // shape after idle reads ~40% low while DPM ramps clocks, which silently
    // makes early shapes in a sweep look worse than late ones and corrupts every
    // cross-B comparison. Drive real work for ~1.5 s before measuring anything.
    {
        let (wm, wk, wb) = (19968usize, 6656usize, 256usize);
        let ww = gpu.upload_raw(&build_hfq4g256(wm, wk, 0x5C), &[wm, wk]).expect("warm w");
        let wxv: Vec<f32> = (0..wb * wk).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();
        let wx = gpu.upload_f32(&wxv, &[wb, wk]).expect("warm x");
        let wy = gpu.alloc_tensor(&[wb, wm], DType::F32).expect("warm y");
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < 1.5 {
            for _ in 0..8 { let _ = gpu.gemm_hfq4g256_residual(&ww, &wx, &wy, wm, wk, wb); }
            let _ = gpu.hip.device_synchronize();
        }
        let _ = gpu.free_tensor(wy); let _ = gpu.free_tensor(wx); let _ = gpu.free_tensor(ww);
        println!("(warm-up complete)");
    }
    println!(
        "{:<12} {:>6} {:>10} {:>10} {:>9}",
        "shape", "B", "ms", "TFLOP/s", "M"
    );

    // Fused gate+up: one call producing both outputs from a shared x, versus
    // the two separate residual GEMMs Glimmer issues today. Qwen prefill uses
    // this; measuring whether it is worth adopting on Glimmer's FFN shapes.
    {
        let (gm, um, kk) = (ffn, ffn, dim);
        let ag = gpu.upload_raw(&build_hfq4g256(gm, kk, 0xB1), &[gm, kk]).unwrap();
        let au = gpu.upload_raw(&build_hfq4g256(um, kk, 0xB2), &[um, kk]).unwrap();
        for &b in &batches {
            let xv: Vec<f32> = (0..b * kk).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
            let x = gpu.upload_f32(&xv, &[b, kk]).unwrap();
            let yg = gpu.alloc_tensor(&[b, gm], DType::F32).unwrap();
            let yu = gpu.alloc_tensor(&[b, um], DType::F32).unwrap();
            for _ in 0..2 {
                let _ = gpu.gemm_gate_up_hfq4g256(&ag, &au, &x, &yg, &yu, gm, um, kk, b);
            }
            let _ = gpu.hip.device_synchronize();
            let iters = if b >= 512 { 6 } else { 12 };
            let mut reps: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..iters {
                    gpu.gemm_gate_up_hfq4g256(&ag, &au, &x, &yg, &yu, gm, um, kk, b)
                        .expect("gate_up");
                }
                let _ = gpu.hip.device_synchronize();
                reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
            }
            reps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ms = reps[1];
            let spread = 100.0 * (reps[2] - reps[0]) / reps[1];
            // both outputs: 2 * (gate + up) FLOPs
            let tflops = 2.0 * ((gm + um) as f64) * (kk as f64) * (b as f64) / (ms / 1000.0) / 1e12;
            println!("{:<12} {:>6} {:>10.3} {:>10.2} {:>7.1}% {:>9}", "gate_up_FUSED", b, ms, tflops, spread, gm);
            let _ = gpu.free_tensor(yu); let _ = gpu.free_tensor(yg); let _ = gpu.free_tensor(x);
        }
        let _ = gpu.free_tensor(au); let _ = gpu.free_tensor(ag);
    }

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
            let mut reps: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..iters {
                    let _ = gpu.hip.memset(&y.buf, 0, b * m * 4);
                    gpu.gemm_hfq4g256_residual(&w, &x, &y, *m, *k, b).expect("gemm");
                }
                let _ = gpu.hip.device_synchronize();
                reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
            }
            reps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ms = reps[1];
            let spread = 100.0 * (reps[2] - reps[0]) / reps[1];
            let tflops = 2.0 * (*m as f64) * (*k as f64) * (b as f64) / (ms / 1000.0) / 1e12;
            println!("{:<12} {:>6} {:>10.3} {:>10.2} {:>7.1}% {:>9}", label, b, ms, tflops, spread, m);

            // Muse-owned full-tile sibling. Same BV, same K loop, same WMMA
            // sequence -> the accumulation order is identical, so this must be
            // BIT-identical to the shared kernel, not merely close. Checked
            // here rather than assumed: a tolerance would hide a real
            // reordering, and bit-identity is what lets the existing
            // byte-identical output gate survive this kernel.
            let muse_bt = if b % 192 == 0 {
                12
            } else if b % 128 == 0 {
                8
            } else {
                0
            };
            if muse_bt != 0 {
                let y2 = gpu.alloc_tensor(&[b, *m], DType::F32).expect("alloc y2");
                let _ = gpu.hip.memset(&y.buf, 0, b * m * 4);
                gpu.gemm_hfq4g256_residual(&w, &x, &y, *m, *k, b).expect("shared");
                let _ = gpu.hip.memset(&y2.buf, 0, b * m * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse(&w, &x, &y2, *m, *k, b, muse_bt)
                    .expect("muse");
                let _ = gpu.hip.device_synchronize();
                let a_host = gpu.download_f32(&y).expect("dl shared");
                let b_host = gpu.download_f32(&y2).expect("dl muse");
                let ndiff = a_host
                    .iter()
                    .zip(b_host.iter())
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
                let maxabs = a_host
                    .iter()
                    .zip(b_host.iter())
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0f32, f32::max);

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y2.buf, 0, b * m * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse(&w, &x, &y2, *m, *k, b, muse_bt);
                }
                let _ = gpu.hip.device_synchronize();
                let mut mreps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y2.buf, 0, b * m * 4);
                        gpu.gemm_hfq4g256_residual_muse(&w, &x, &y2, *m, *k, b, muse_bt)
                            .expect("muse bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    mreps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                mreps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mms = mreps[1];
                let mspread = 100.0 * (mreps[2] - mreps[0]) / mreps[1];
                let mtf = 2.0 * (*m as f64) * (*k as f64) * (b as f64) / (mms / 1000.0) / 1e12;
                println!(
                    "{:<12} {:>6} {:>10.3} {:>10.2} {:>7.1}% {:>9}   used={} bitdiff={} maxabs={:.3e} vs_shared={:+.1}%",
                    format!("{}_M{}", label, muse_bt),
                    b, mms, mtf, mspread, m, used, ndiff, maxabs,
                    100.0 * (ms / mms - 1.0)
                );
                let _ = gpu.free_tensor(y2);
            }

            let _ = gpu.free_tensor(y);
            let _ = gpu.free_tensor(x);
        }
        let _ = gpu.free_tensor(w);
    }
}
