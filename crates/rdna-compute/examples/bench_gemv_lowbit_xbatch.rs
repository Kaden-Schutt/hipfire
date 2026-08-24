// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! What the x-batch low-bit GEMV buys over looping the scalar kernel.
//!
//! Prefill currently re-reads every weight row once per token. This measures
//! the alternative directly: B scalar launches vs one x-batch launch, at the
//! real Bonsai-27B shapes. The figure of merit is the ratio — that is the
//! prefill speedup ceiling for a chunk of B tokens.
//!
//! Usage: bench_gemv_lowbit_xbatch [ternary|binary] [reps]

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let which = argv.get(1).cloned().unwrap_or_else(|| "ternary".into());
    let reps: usize = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let ternary = which != "binary";
    let blk = if ternary { 34usize } else { 18usize };

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "bench_gemv_lowbit_xbatch: arch={} fmt={} reps={reps}",
        gpu.arch,
        if ternary { "tq2g128" } else { "bq1g128" }
    );

    for (m, k) in [
        (6144usize, 5120usize),
        (5120, 5120),
        (17408, 5120),
        (5120, 17408),
    ] {
        let groups = k / 128;
        let total = m * groups * blk;
        let packed: Vec<u8> = (0..total).map(|i| (i * 37 % 251) as u8).collect();
        let d_a = gpu.upload_raw(&packed, &[total]).expect("upload A");

        for b in [2usize, 4] {
            let xs: Vec<f32> = (0..b * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
            let d_x = gpu.upload_f32(&xs, &[b * k]).expect("upload x");
            let d_y = gpu.zeros(&[b * m], DType::F32).expect("zeros y");
            let d_y1 = gpu.zeros(&[m], DType::F32).expect("zeros y1");

            // Pre-upload the per-token slices OUTSIDE the timed region: the
            // scalar arm must pay only for launches, exactly like the xbatch
            // arm, or the comparison measures host uploads instead of kernels.
            let slices: Vec<_> = (0..b)
                .map(|bi| {
                    gpu.upload_f32(&xs[bi * k..(bi + 1) * k], &[k])
                        .expect("upload slice")
                })
                .collect();

            // Arm A: B scalar launches (what prefill does today, per token).
            let scalar = |gpu: &mut Gpu| {
                for xslice in slices.iter() {
                    if ternary {
                        gpu.gemv_tq2g128(&d_a, xslice, &d_y1, m, k).expect("scalar");
                    } else {
                        gpu.gemv_bq1g128(&d_a, xslice, &d_y1, m, k).expect("scalar");
                    }
                }
            };
            // Arm B: one x-batch launch.
            let batched = |gpu: &mut Gpu| {
                if ternary {
                    gpu.gemv_tq2g128_xbatch(&d_a, &d_x, &d_y, m, k, b)
                        .expect("xbatch");
                } else {
                    gpu.gemv_bq1g128_xbatch(&d_a, &d_x, &d_y, m, k, b)
                        .expect("xbatch");
                }
            };

            scalar(&mut gpu);
            batched(&mut gpu);
            gpu.hip.device_synchronize().expect("sync");

            let t0 = Instant::now();
            for _ in 0..reps {
                scalar(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let s_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            let t1 = Instant::now();
            for _ in 0..reps {
                batched(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            let b_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            println!(
                "M={m:<6} K={k:<6} B={b}  scalar {s_ms:7.3} ms  xbatch {b_ms:7.3} ms  speedup {:.2}x",
                s_ms / b_ms
            );
        }
    }
}
