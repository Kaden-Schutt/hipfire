// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Does fusing the LA preamble actually pay for the low-bit formats?
//!
//! Four separate plain GEMMs (what the integration does today) vs one fused
//! qkvza launch. mq4's prefill advantage is exactly this launch structure, so
//! this measures whether closing it is worth the dispatch wiring.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("bench_fused_qkvza_lowbit: arch={} reps={reps}", gpu.arch);

    let k = 5120usize;
    // Bonsai-27B LA preamble: wqkv is the bulk, wz next, beta/alpha tiny.
    let m = [6144usize, 4096, 128, 128];

    for n in [32usize, 128, 512] {
        let dev: Vec<_> = m
            .iter()
            .map(|mi| {
                let total = mi * (k / 128) * 34;
                let packed: Vec<u8> = (0..total).map(|i| (i * 37 % 251) as u8).collect();
                gpu.upload_raw(&packed, &[total]).expect("upload")
            })
            .collect();
        let xs: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let d_x = gpu.upload_f32(&xs, &[n * k]).expect("upload x");
        let ys: Vec<_> = m
            .iter()
            .map(|mi| gpu.zeros(&[n * mi], DType::F32).expect("zeros"))
            .collect();

        let unfused = |gpu: &mut Gpu| {
            for i in 0..4 {
                gpu.gemm_tq2g128_prefill(&dev[i], &d_x, &ys[i], m[i], k, n)
                    .expect("plain");
            }
        };
        let fused = |gpu: &mut Gpu| {
            gpu.gemm_qkvza_tq2g128(
                &dev[0], &dev[1], &dev[2], &dev[3], &d_x, &ys[0], &ys[1], &ys[2], &ys[3], m[0],
                m[1], m[2], m[3], k, n,
            )
            .expect("fused");
        };

        unfused(&mut gpu);
        fused(&mut gpu);
        gpu.hip.device_synchronize().expect("sync");

        let t0 = Instant::now();
        for _ in 0..reps {
            unfused(&mut gpu);
        }
        gpu.hip.device_synchronize().expect("sync");
        let u_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        let t1 = Instant::now();
        for _ in 0..reps {
            fused(&mut gpu);
        }
        gpu.hip.device_synchronize().expect("sync");
        let f_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        println!(
            "N={n:<4} unfused(4 launches) {u_ms:7.3} ms | fused(1 launch) {f_ms:7.3} ms | {:.2}x",
            u_ms / f_ms
        );
    }
}
