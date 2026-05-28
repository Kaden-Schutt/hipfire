// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx11 correctness — `gemm_q8_0_wmma_gfx11` vs the scalar
//! `gemm_q8_0_batched` reference. Mirrors `test_gemm_q8_residual_wmma`
//! but tests the no-residual (= rather than +=) write semantics on the
//! new gfx11 WMMA sibling shipped in PR #8 (2026-05-28).
//!
//! Setup: scalar substrate into Y_ref; new WMMA kernel into Y_test;
//! compare. Both should agree within ulp-bounded FP rounding tolerance
//! (Q8 dequant + FP16-WMMA on gfx11 vs scalar Q8 dequant + FP32 accum).

use rdna_compute::{DType, Gpu};

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::synth_q8;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_q8_wmma_gfx11 ===\n  arch = {arch}");
    if !arch.starts_with("gfx11") {
        eprintln!("  SKIPPED: needs gfx11, got {arch}");
        std::process::exit(0);
    }

    // Same shape coverage as the residual sibling, plus a small
    // A3B-lm_head-shaped shape (M=151936 is the Qwen3 vocab — use a
    // proxy M that exercises the row-tiling without OOMing on 24GB).
    let shapes: Vec<(usize, usize, &str)> = vec![
        ( 64,  128,         "tiny"),
        (512,  512,         "medium"),
        (4096, 4096,        "9B wo     (M=K=4096)"),
        (4096, 11008,       "9B w_down (M=4096 K=11008)"),
        (5120, 4096,        "27B lm_head-shaped (M=5120 K=4096)"),
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

        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);

            // Test path: direct WMMA gfx11 kernel.
            let d_y_test = gpu.zeros(&[n * m], DType::F32).unwrap();
            gpu.gemm_q8_0_wmma_gfx11(&d_a, &x_n, &d_y_test, m, k, n).unwrap();

            // Reference path: scalar substrate via gemm_q8_0_batched_chunked
            // with HIPFIRE_Q8_BATCHED_LEGACY=1 (the test process doesn't toggle
            // env vars; call the scalar kernel directly through the substrate
            // by chunking via gemm_q8_0_batched at MAX_BATCH=64).
            let d_y_ref = gpu.zeros(&[n * m], DType::F32).unwrap();
            // Direct scalar dispatch (mirrors the legacy path inside
            // gemm_q8_0_batched_chunked):
            const MAX_BATCH: usize = 64;
            let mut off = 0;
            while off < n {
                let take = (n - off).min(MAX_BATCH);
                let x_sub = x_n.sub_offset(off * k, take * k);
                let y_sub = d_y_ref.sub_offset(off * m, take * m);
                gpu.gemm_q8_0_batched(&d_a, &x_sub, &y_sub, m, k, take).unwrap();
                off += take;
            }

            let s = compare(&gpu.download_f32(&d_y_test).unwrap(),
                            &gpu.download_f32(&d_y_ref).unwrap());
            // Same thresholds as the residual sibling: WMMA-FP32-accum vs
            // scalar-FP32-accum agree to ulp + FP16-dequant rounding.
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
