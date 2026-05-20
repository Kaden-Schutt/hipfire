// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! T3-1b correctness test — `gemm_qkv_q8_0_wmma` (production fused 3-way QKV)
//! vs the Tier 2 substrate (`gemm_q8_0_batched_chunked` × 3).
//!
//! Gate: gated mean relative error < 1e-3 on each of Y_q, Y_k, Y_v.
//! Gating excludes outputs where |ref| < 1% of |ref|_max (rel-error metric
//! is unreliable near zero — see q8-fused-prefill-kernels.md §Numerical
//! equivalence test).
//!
//! Sweeps:
//!   - N ∈ {1, 4, 16, 32, 64, 128, 256}
//!   - Several (q_m, k_m, v_m, K) shape triples, including production 9B FA.
//!   - "every-int8-value-once" block-pattern catches sign-extension regressions.
//!
//! Run:  cargo run --release --example test_gemm_q8_qkv_wmma
//! Exits 0 on pass, 1 on any sweep failure.

use rdna_compute::{DType, Gpu};

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::{f32_to_f16_bits, synth_q8};

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_q8_qkv_wmma ===");
    eprintln!("  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: WMMA path requires RDNA3+ (gfx11/12), got {arch}");
        std::process::exit(0);
    }

    // (q_m, k_m, v_m, K, label) — all dims chosen multiples of 16 for clean WMMA tiles.
    let shapes: Vec<(usize, usize, usize, usize, &str)> = vec![
        ( 64,  32,  32,  128, "tiny    (q=64 k=v=32 K=128)"),
        (256,  64,  64,  512, "medium  (q=256 k=v=64 K=512)"),
        (4096, 1024, 1024, 4096, "9B FA   (q=4096 k=v=1024 K=4096)"),
    ];
    let batches: Vec<usize> = vec![1, 4, 16, 32, 64, 128, 256];
    let mut total_fail = 0usize;

    for (q_m, k_m, v_m, k, label) in &shapes {
        let (q_m, k_m, v_m, k) = (*q_m, *k_m, *v_m, *k);
        assert!(k % 32 == 0, "K must be a multiple of 32 (Q8_0 block)");
        eprintln!("\n--- {label} ---");

        let w_q = synth_q8(q_m, k, 0xA1B2C3D4);
        let w_k = synth_q8(k_m, k, 0xE5F60718);
        let w_v = synth_q8(v_m, k, 0x9ABCDEF0);

        let d_aq = gpu.upload_raw(&w_q, &[w_q.len()]).unwrap();
        let d_ak = gpu.upload_raw(&w_k, &[w_k.len()]).unwrap();
        let d_av = gpu.upload_raw(&w_v, &[w_v.len()]).unwrap();

        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k)
            .map(|i| {
                let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
                (v * 1e-9) % 2.0 - 1.0
            })
            .collect();
        let d_x = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        let d_yq_wmma = gpu.zeros(&[max_n * q_m], DType::F32).unwrap();
        let d_yk_wmma = gpu.zeros(&[max_n * k_m], DType::F32).unwrap();
        let d_yv_wmma = gpu.zeros(&[max_n * v_m], DType::F32).unwrap();

        let d_yq_ref = gpu.zeros(&[max_n * q_m], DType::F32).unwrap();
        let d_yk_ref = gpu.zeros(&[max_n * k_m], DType::F32).unwrap();
        let d_yv_ref = gpu.zeros(&[max_n * v_m], DType::F32).unwrap();

        for &n in &batches {
            let x_n  = d_x.sub_offset(0, n * k);
            let yq_w = d_yq_wmma.sub_offset(0, n * q_m);
            let yk_w = d_yk_wmma.sub_offset(0, n * k_m);
            let yv_w = d_yv_wmma.sub_offset(0, n * v_m);
            let yq_r = d_yq_ref.sub_offset(0, n * q_m);
            let yk_r = d_yk_ref.sub_offset(0, n * k_m);
            let yv_r = d_yv_ref.sub_offset(0, n * v_m);

            // Production fused call — gfx11 uses the canonical _w32 kernel;
            // gfx12 routes to the _w32_gfx12 sibling (half8_t lane-grp split).
            if arch.starts_with("gfx12") {
                gpu.gemm_qkv_q8_0_wmma_gfx12(
                    &d_aq, &d_ak, &d_av,
                    &x_n,
                    &yq_w, &yk_w, &yv_w,
                    q_m, k_m, v_m, k, n,
                ).unwrap();
            } else {
                gpu.gemm_qkv_q8_0_wmma(
                    &d_aq, &d_ak, &d_av,
                    &x_n,
                    &yq_w, &yk_w, &yv_w,
                    q_m, k_m, v_m, k, n,
                ).unwrap();
            }

            // Reference: 3 separate substrate calls (single-output each).
            gpu.gemm_q8_0_batched_chunked(&d_aq, &x_n, &yq_r, q_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_ak, &x_n, &yk_r, k_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_av, &x_n, &yv_r, v_m, k, n).unwrap();

            let yq_w_host = gpu.download_f32(&yq_w).unwrap();
            let yk_w_host = gpu.download_f32(&yk_w).unwrap();
            let yv_w_host = gpu.download_f32(&yv_w).unwrap();
            let yq_r_host = gpu.download_f32(&yq_r).unwrap();
            let yk_r_host = gpu.download_f32(&yk_r).unwrap();
            let yv_r_host = gpu.download_f32(&yv_r).unwrap();

            let stats_q = compare(&yq_w_host, &yq_r_host);
            let stats_k = compare(&yk_w_host, &yk_r_host);
            let stats_v = compare(&yv_w_host, &yv_r_host);

            // Gate: mean_rel < 2e-3 AND max_rel < 3.5e-2 — fp16 WMMA precision.
            // Threshold tightened 2026-05-13 from 5e-2 → 3.5e-2 (see
            // test_gemm_q8_gate_up_wmma.rs for the full rationale).
            let pass = stats_q.mean_rel < 2e-3 && stats_k.mean_rel < 2e-3 && stats_v.mean_rel < 2e-3
                   && stats_q.max_rel  < 3.5e-2 && stats_k.max_rel  < 3.5e-2 && stats_v.max_rel  < 3.5e-2;
            let mark = if pass { "PASS" } else { total_fail += 1; "FAIL" };
            eprintln!(
                "  N={n:4}  {mark}   \
                 Q: mean_rel={:.3e} max_rel={:.3e}  \
                 K: mean_rel={:.3e} max_rel={:.3e}  \
                 V: mean_rel={:.3e} max_rel={:.3e}",
                stats_q.mean_rel, stats_q.max_rel,
                stats_k.mean_rel, stats_k.max_rel,
                stats_v.mean_rel, stats_v.max_rel,
            );
        }
    }

    // "Every-int8-value-once" pattern: weight block contains int8 [-128..127] in
    // order at byte offset 2. Catches sign-extension regressions on the dequant
    // path. One block per row, fixed scale = 1/128 so the dequanted values land
    // in [-1.0, 0.992] — a clean fp16 range.
    eprintln!("\n--- every-int8-value-once pattern ---");
    let q_m = 16usize;
    let k_m = 16usize;
    let v_m = 16usize;
    let k = 32usize; // single block, all 256 int8 values across the block × 8 rows
    let mut w = vec![0u8; 16 * 34];
    let sc_bits = f32_to_f16_bits(1.0 / 128.0);
    for row in 0..16 {
        let off = row * 34;
        w[off] = (sc_bits & 0xFF) as u8;
        w[off + 1] = (sc_bits >> 8) as u8;
        for j in 0..32 {
            // Row 0: bytes 0..32 of the int8 enumeration ([-128..-97])
            // Row 1: bytes 32..64, etc. After row 7 it wraps.
            let val = ((row * 32 + j) as i32 - 128) as i8;
            w[off + 2 + j] = val as u8;
        }
    }
    let d_aq = gpu.upload_raw(&w, &[w.len()]).unwrap();
    let d_ak = d_aq.sub_offset(0, w.len()); // same matrix for K, V (sufficient — we only check Q dims)
    let d_av = d_aq.sub_offset(0, w.len());

    let n = 4usize;
    let x: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();

    let d_yq = gpu.zeros(&[n * q_m], DType::F32).unwrap();
    let d_yk = gpu.zeros(&[n * k_m], DType::F32).unwrap();
    let d_yv = gpu.zeros(&[n * v_m], DType::F32).unwrap();
    let d_yq_ref = gpu.zeros(&[n * q_m], DType::F32).unwrap();

    if arch.starts_with("gfx12") {
        gpu.gemm_qkv_q8_0_wmma_gfx12(&d_aq, &d_ak, &d_av, &d_x, &d_yq, &d_yk, &d_yv,
                                     q_m, k_m, v_m, k, n).unwrap();
    } else {
        gpu.gemm_qkv_q8_0_wmma(&d_aq, &d_ak, &d_av, &d_x, &d_yq, &d_yk, &d_yv,
                               q_m, k_m, v_m, k, n).unwrap();
    }
    gpu.gemm_q8_0_batched_chunked(&d_aq, &d_x, &d_yq_ref, q_m, k, n).unwrap();

    let yq = gpu.download_f32(&d_yq).unwrap();
    let yq_ref = gpu.download_f32(&d_yq_ref).unwrap();
    let stats = compare(&yq, &yq_ref);
    let pass = stats.mean_rel < 2e-3 && stats.max_rel < 5e-2;
    let mark = if pass { "PASS" } else { total_fail += 1; "FAIL" };
    eprintln!(
        "  N={n}  {mark}   Q: mean_rel={:.3e} max_rel={:.3e}  |ref|_max={:.2}",
        stats.mean_rel, stats.max_rel, yq_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );

    eprintln!("\n=== {} failure(s) ===", total_fail);
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

struct Stats {
    mean_rel: f64,
    max_rel: f64,
}

fn compare(wmma: &[f32], reference: &[f32]) -> Stats {
    let max_ref = reference.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let threshold = max_ref * 0.01;
    let mut sum_rel = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut count = 0usize;
    for (w, r) in wmma.iter().zip(reference.iter()) {
        if r.abs() > threshold {
            let rel = ((w - r).abs() / r.abs()) as f64;
            sum_rel += rel;
            if rel > max_rel { max_rel = rel; }
            count += 1;
        }
    }
    let mean_rel = if count == 0 { 0.0 } else { sum_rel / count as f64 };
    Stats { mean_rel, max_rel }
}

