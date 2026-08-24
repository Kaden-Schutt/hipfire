// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Same kernel design, two formats: gemm_qkvza_tq2g128 vs gemm_qkvza_hfq4g256.
//!
//! Both are BATCH_TILE=8, one wave per output row, fused over the four LA
//! projections, scalar FMA, no LDS. Identical m/K/N. Ternary reads 0.27
//! B/weight against mq4's 0.53, so if ours is not at least as fast, the
//! difference is in our implementation, not the format.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("bench_qkvza_ternary_vs_mq4: arch={} reps={reps}", gpu.arch);

    let k = 5120usize;
    let m = [6144usize, 4096, 128, 128];

    for n in [32usize, 128, 512] {
        // Ternary: 34 B per 128 weights.
        let tq2: Vec<_> = m
            .iter()
            .map(|mi| {
                let total = mi * (k / 128) * 34;
                let p: Vec<u8> = (0..total).map(|i| (i * 37 % 251) as u8).collect();
                gpu.upload_raw(&p, &[total]).expect("upload tq2")
            })
            .collect();
        // HFQ4-G256: 136 B per 256 weights ([f32 scale][f32 zero][128 B codes]).
        let hfq4: Vec<_> = m
            .iter()
            .map(|mi| {
                let total = mi * (k / 256) * 136;
                let p: Vec<u8> = (0..total).map(|i| (i * 31 % 251) as u8).collect();
                gpu.upload_raw(&p, &[total]).expect("upload hfq4")
            })
            .collect();

        let xs: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let d_x = gpu.upload_f32(&xs, &[n * k]).expect("upload x");
        let ys: Vec<_> = m
            .iter()
            .map(|mi| gpu.zeros(&[n * mi], DType::F32).expect("zeros"))
            .collect();

        let run_tq2 = |gpu: &mut Gpu| {
            gpu.gemm_qkvza_tq2g128(
                &tq2[0], &tq2[1], &tq2[2], &tq2[3], &d_x, &ys[0], &ys[1], &ys[2], &ys[3], m[0],
                m[1], m[2], m[3], k, n,
            )
            .expect("tq2");
        };
        let run_mq4 = |gpu: &mut Gpu| {
            gpu.gemm_qkvza_hfq4g256(
                &hfq4[0], &hfq4[1], &hfq4[2], &hfq4[3], &d_x, &ys[0], &ys[1], &ys[2], &ys[3], m[0],
                m[1], m[2], m[3], k, n,
            )
            .expect("mq4");
        };

        run_tq2(&mut gpu);
        run_mq4(&mut gpu);
        gpu.hip.device_synchronize().expect("sync");

        let t0 = Instant::now();
        for _ in 0..reps {
            run_tq2(&mut gpu);
        }
        gpu.hip.device_synchronize().expect("sync");
        let t_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        let t1 = Instant::now();
        for _ in 0..reps {
            run_mq4(&mut gpu);
        }
        gpu.hip.device_synchronize().expect("sync");
        let m_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        println!(
            "N={n:<4} qkvza ternary {t_ms:7.3} ms | qkvza mq4 {m_ms:7.3} ms | ternary/mq4 {:.2}x",
            m_ms / t_ms
        );
    }
}
