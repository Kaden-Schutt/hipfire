// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate for `gemv_hfq4g256_xbatch`: one weight pass, B activation vectors.
//!
//! The multi-slot decode path may only adopt this kernel if it is **bitwise**
//! identical to running the existing single-vector GEMV once per slot — the
//! multi-slot forward is gated by a golden test demanding 0.000x divergence
//! from the reference's per-token GEMV, so "close enough" is a failure here.
//!
//! Runs at the real lm_head shape (M=248320, K=2048), because that shape
//! selects a hand-tuned gfx1151 kernel on this box. Comparing at a smaller
//! shape would compare against the generic kernel and prove nothing about
//! what the model actually runs.
//!
//! Also reports wall time for both arms, since the whole point is that the
//! 270 MB weight matrix is read once instead of B times.

use rdna_compute::{DType, Gpu};

const M: usize = 248_320;
const K: usize = 2_048;

fn main() {
    let mut gpu = Gpu::init().unwrap();

    let groups_per_row = K / 256;
    let row_bytes = groups_per_row * 136;
    let total_bytes = M * row_bytes;
    eprintln!(
        "=== gemv_hfq4g256_xbatch gate: M={M} K={K}, weights {:.1} MiB ===",
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    // Varied weights, not a constant pattern: a degenerate matrix can hide a
    // reassociation bug by making every partial sum equal.
    let mut w = vec![0u8; total_bytes];
    let mut seed: u32 = 0x1234_5678;
    let mut next = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for r in 0..M {
        for g in 0..groups_per_row {
            let off = r * row_bytes + g * 136;
            let scale = 0.001 + ((next() >> 8) % 1000) as f32 * 1e-5;
            let zero = -0.004 + ((next() >> 8) % 800) as f32 * 1e-5;
            w[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            w[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for i in 0..128 {
                w[off + 8 + i] = (next() >> 13) as u8;
            }
        }
    }
    let d_w = gpu.upload_raw(&w, &[total_bytes]).unwrap();
    drop(w);

    let max_b = 4usize;
    // x is [max_b x K] row-major, matching the kernel's layout.
    let x: Vec<f32> = (0..max_b * K)
        .map(|i| {
            let t = (i % 977) as f32;
            0.01 * (t * 0.37).sin() + 0.002 * ((i / 977) as f32)
        })
        .collect();
    let d_x = gpu.upload_f32(&x, &[max_b * K]).unwrap();

    let d_y_ref = gpu.zeros(&[max_b * M], DType::F32).unwrap();
    let d_y_cand = gpu.zeros(&[max_b * M], DType::F32).unwrap();

    let mut all_ok = true;

    for b in 1..=max_b {
        // Reference arm: the existing single-vector GEMV, once per row, each
        // writing into its own [M] slice of the same [b x M] block.
        for i in 0..b {
            let x_i = d_x.sub_offset(i * K, K);
            let y_i = d_y_ref.sub_offset(i * M, M);
            gpu.gemv_hfq4g256(&d_w, &x_i, &y_i, M, K).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        gpu.gemv_hfq4g256_xbatch(&d_w, &d_x, &d_y_cand, M, K, b)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let y_ref = gpu.download_f32(&d_y_ref).unwrap();
        let y_cand = gpu.download_f32(&d_y_cand).unwrap();

        let mut mismatches = 0usize;
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for i in 0..b * M {
            if y_ref[i].to_bits() != y_cand[i].to_bits() {
                mismatches += 1;
                let d = (y_ref[i] - y_cand[i]).abs();
                if d > worst {
                    worst = d;
                    worst_at = i;
                }
            }
        }
        if mismatches == 0 {
            eprintln!("  B={b}: BITWISE IDENTICAL over {} elements", b * M);
        } else {
            all_ok = false;
            eprintln!(
                "  B={b}: {mismatches} / {} elements differ; worst |d|={worst:.3e} at {worst_at} \
                 (ref={} cand={})",
                b * M,
                y_ref[worst_at],
                y_cand[worst_at]
            );
        }
    }

    // ---- timing: B separate GEMVs vs one xbatch launch ----
    eprintln!("\n--- wall time (20 iters after 5 warmup) ---");
    for b in 1..=max_b {
        for _ in 0..5 {
            for i in 0..b {
                let x_i = d_x.sub_offset(i * K, K);
                let y_i = d_y_ref.sub_offset(i * M, M);
                gpu.gemv_hfq4g256(&d_w, &x_i, &y_i, M, K).unwrap();
            }
            gpu.gemv_hfq4g256_xbatch(&d_w, &d_x, &d_y_cand, M, K, b)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            for i in 0..b {
                let x_i = d_x.sub_offset(i * K, K);
                let y_i = d_y_ref.sub_offset(i * M, M);
                gpu.gemv_hfq4g256(&d_w, &x_i, &y_i, M, K).unwrap();
            }
        }
        gpu.hip.device_synchronize().unwrap();
        let per_seq = t0.elapsed().as_secs_f64() * 1e3 / 20.0;

        let t1 = std::time::Instant::now();
        for _ in 0..20 {
            gpu.gemv_hfq4g256_xbatch(&d_w, &d_x, &d_y_cand, M, K, b)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let per_batch = t1.elapsed().as_secs_f64() * 1e3 / 20.0;

        eprintln!(
            "  B={b}: {b} x GEMV = {per_seq:.3} ms | xbatch = {per_batch:.3} ms | {:.2}x",
            per_seq / per_batch
        );
    }

    if all_ok {
        eprintln!("\nALL CHECKS PASS (bitwise identical for B=1..={max_b})");
    } else {
        eprintln!("\nFAILED: xbatch is not bitwise identical to per-vector GEMV");
        std::process::exit(1);
    }
}
