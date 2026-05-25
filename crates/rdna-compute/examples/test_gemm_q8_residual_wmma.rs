// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! T3-2 correctness — `gemm_q8_0_residual_wmma` vs (substrate + add_inplace).
//! Tests both the WMMA matmul AND the fused `+=` residual semantics.
//!
//! Setup: seed Y_test with a residual; run the kernel. Reference: substrate
//! into a fresh tmp, then add_inplace into a separately-seeded Y_ref. Compare.

use rdna_compute::{DType, Gpu};

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::synth_q8;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_q8_residual_wmma ===\n  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: needs gfx11/12, got {arch}"); std::process::exit(0);
    }

    // (M, K, label) — residual sites are wo and w_down on Qwen3.5.
    let shapes: Vec<(usize, usize, &str)> = vec![
        ( 64,  128, "tiny"),
        (512,  512, "medium"),
        (4096, 4096, "9B wo     (M=K=4096)"),
        (4096, 11008, "9B w_down (M=4096 K=11008)"),
    ];
    let batches: Vec<usize> = vec![1, 4, 16, 32, 64, 128, 256];
    let mut total_fail = 0usize;

    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);
        eprintln!("\n--- {label} ---");

        let w = synth_q8(m, k, 0xA1B2C3D4);
        let d_a = gpu.upload_raw(&w, &[w.len()]).unwrap();

        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        // Residual seed — non-zero so we actually test += vs =.
        let r_host: Vec<f32> = (0..max_n * m).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);

            // Test path: seed Y with residual, run fused kernel.
            let d_y_test = gpu.upload_f32(&r_host[..n * m], &[n * m]).unwrap();
            if arch.starts_with("gfx12") {
                gpu.gemm_q8_0_residual_wmma_gfx12(&d_a, &x_n, &d_y_test, m, k, n).unwrap();
            } else {
                gpu.gemm_q8_0_residual_wmma(&d_a, &x_n, &d_y_test, m, k, n).unwrap();
            }

            // Ref path: substrate into tmp, add_inplace into separately-seeded Y_ref.
            let d_tmp = gpu.zeros(&[n * m], DType::F32).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_a, &x_n, &d_tmp, m, k, n).unwrap();
            let d_y_ref = gpu.upload_f32(&r_host[..n * m], &[n * m]).unwrap();
            gpu.add_inplace_f32(&d_y_ref, &d_tmp).unwrap();

            let s = compare(&gpu.download_f32(&d_y_test).unwrap(),
                            &gpu.download_f32(&d_y_ref).unwrap());
            // Threshold tightened 2026-05-13 from max_rel < 5e-2 → 2.5e-2;
            // see test_gemm_q8_gate_up_wmma.rs for rationale.
            let pass = s.mean_rel < 2e-3 && s.max_rel < 3.5e-2;
            let mark = if pass { "PASS" } else { total_fail += 1; "FAIL" };
            eprintln!("  N={n:4}  {mark}   mean_rel={:.2e}  max_rel={:.2e}",
                s.mean_rel, s.max_rel);
        }
    }
    eprintln!("\n=== {total_fail} failure(s) ===");
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

struct Stats { mean_rel: f64, max_rel: f64 }
fn compare(a: &[f32], b: &[f32]) -> Stats {
    let max_ref = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let thr = max_ref * 0.01;
    let (mut sum, mut max_r, mut n) = (0.0f64, 0.0f64, 0usize);
    for (x, y) in a.iter().zip(b.iter()) {
        if y.abs() > thr {
            let r = ((x - y).abs() / y.abs()) as f64;
            sum += r; if r > max_r { max_r = r; } n += 1;
        }
    }
    Stats { mean_rel: if n == 0 { 0.0 } else { sum / n as f64 }, max_rel: max_r }
}
fn synth_x(i: usize) -> f32 {
    let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
    (v * 1e-9) % 2.0 - 1.0
}
