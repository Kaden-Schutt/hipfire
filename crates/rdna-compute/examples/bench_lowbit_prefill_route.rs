// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Which prefill route should the low-bit formats take?
//!
//! TQ2G128/BQ1G128 have no tiled GEMM, so a prefill chunk of N tokens today
//! re-reads every weight row N times through the scalar GEMV. This measures
//! the candidate replacement at the real Bonsai-27B shapes:
//!
//!   A) scalar   — N x gemv_{tq2,bq1}g128            (what prefill does now)
//!   B) dequant  — one dequant_*_to_f16 + one gemm_f16 over all N tokens
//!
//! Arm B pays the dequant ONCE per weight per chunk, so its cost is amortised
//! over N. The crossover N is the number that matters: below it, keep the
//! scalar path; above it, the dequant route wins.
//!
//! Usage: bench_lowbit_prefill_route [ternary|binary] [reps]

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let which = argv.get(1).cloned().unwrap_or_else(|| "ternary".into());
    let reps: usize = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let ternary = which != "binary";
    let blk = if ternary { 34usize } else { 18usize };

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "bench_lowbit_prefill_route: arch={} fmt={} reps={reps}",
        gpu.arch,
        if ternary { "tq2g128" } else { "bq1g128" }
    );

    // The FFN up-projection: the biggest per-layer weight, so the one whose
    // re-reads dominate a prefill chunk.
    for (m, k) in [(17408usize, 5120usize), (5120, 5120)] {
        let groups = k / 128;
        let total = m * groups * blk;
        let packed: Vec<u8> = (0..total).map(|i| (i * 37 % 251) as u8).collect();
        let d_a = gpu.upload_raw(&packed, &[total]).expect("upload A");
        // FP16 scratch for the dequantized weight: M*K*2 bytes.
        let w_f16 = gpu.zeros(&[m * k], DType::F16).expect("zeros w_f16");

        for n in [8usize, 32, 128, 512] {
            let xs: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
            let d_y = gpu.zeros(&[n * m], DType::F32).expect("zeros y");
            let d_xf = gpu.upload_f32(&xs, &[n * k]).expect("upload x f32");
            let d_y1 = gpu.zeros(&[m], DType::F32).expect("zeros y1");
            let slices: Vec<_> = (0..n)
                .map(|i| {
                    gpu.upload_f32(&xs[i * k..(i + 1) * k], &[k])
                        .expect("upload slice")
                })
                .collect();

            let scalar = |gpu: &mut Gpu| {
                for xslice in slices.iter() {
                    if ternary {
                        gpu.gemv_tq2g128(&d_a, xslice, &d_y1, m, k).expect("scalar");
                    } else {
                        gpu.gemv_bq1g128(&d_a, xslice, &d_y1, m, k).expect("scalar");
                    }
                }
            };
            // Arm B: native tiled GEMM straight off the packed blocks.
            let native = |gpu: &mut Gpu| {
                if ternary {
                    gpu.gemm_tq2g128_prefill(&d_a, &d_xf, &d_y, m, k, n)
                        .expect("native prefill");
                } else {
                    gpu.gemm_bq1g128_prefill(&d_a, &d_xf, &d_y, m, k, n)
                        .expect("native prefill");
                }
            };
            // Arm D: WMMA straight off the packed blocks (F16 activations).
            let wmma = |gpu: &mut Gpu| {
                if ternary {
                    gpu.gemm_tq2g128_wmma(&d_a, &d_xf, &d_y, m, k, n)
                        .expect("wmma");
                } else {
                    gpu.gemm_bq1g128_wmma(&d_a, &d_xf, &d_y, m, k, n)
                        .expect("wmma");
                }
            };
            // Arm C: dequant -> generic F16-weight GEMM (measured, kept for contrast).
            let dequant_route = |gpu: &mut Gpu| {
                if ternary {
                    gpu.dequantize_tq2g128_to_f16(&d_a.buf, &w_f16.buf, m, k)
                        .expect("dequant");
                } else {
                    gpu.dequantize_bq1g128_to_f16(&d_a.buf, &w_f16.buf, m, k)
                        .expect("dequant");
                }
                gpu.gemm_f16(&w_f16, &d_xf, &d_y, m, k, n)
                    .expect("gemm_f16");
            };

            scalar(&mut gpu);
            native(&mut gpu);
            dequant_route(&mut gpu);
            gpu.hip.device_synchronize().expect("sync");

            let t0 = Instant::now();
            for _ in 0..reps {
                scalar(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let s_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            let t1 = Instant::now();
            for _ in 0..reps {
                dequant_route(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let d_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            wmma(&mut gpu);
            gpu.hip.device_synchronize().expect("sync");
            let tw = Instant::now();
            for _ in 0..reps {
                wmma(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let w_ms = tw.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            let tn = Instant::now();
            for _ in 0..reps {
                native(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let n_ms = tn.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            // Decompose arm C: which half is the cost?
            let t2 = Instant::now();
            for _ in 0..reps {
                if ternary {
                    gpu.dequantize_tq2g128_to_f16(&d_a.buf, &w_f16.buf, m, k)
                        .expect("dequant");
                } else {
                    gpu.dequantize_bq1g128_to_f16(&d_a.buf, &w_f16.buf, m, k)
                        .expect("dequant");
                }
            }
            gpu.hip.device_synchronize().expect("sync");
            let dq_ms = t2.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            println!(
                "M={m:<6} K={k:<5} N={n:<4} scalar {s_ms:8.3} | tiled {n_ms:7.3} ({:4.1}x) | WMMA {w_ms:7.3} ({:5.1}x) | dq+f16 {d_ms:7.3} ({:.2}x) | dequant {dq_ms:7.3}",
                s_ms / n_ms,
                s_ms / w_ms,
                s_ms / d_ms
            );
        }
    }
}
