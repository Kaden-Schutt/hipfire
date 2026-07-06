// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test: gemm_q8_0_wmma vs gemm_q8_0_batched_chunked (scalar reference)
//! on gfx11 (RDNA3 dGPU).
//!
//! Two suites:
//!
//! 1. Static suite — confirms the WMMA kernel is numerically correct at small N
//!    (1,2,4,8,16,64). Includes the exact lm_head shape M=248320,K=2048 at N=2
//!    (the MTP failure shape) and a smaller M=8192,K=2048 sweep.
//!
//! 2. Stale-cache suite — exercises the MTP scenario where the same X tensor
//!    pointer carries DIFFERENT CONTENT across calls (rmsnorm overwrites it each
//!    step). Exposes the `ensure_fp16_x` pointer-keyed stale-cache bug:
//!    after step 1 converts and caches ptr, step 2's new content at the same
//!    ptr should also be converted but is NOT without the fix (→ stale FP16 →
//!    wrong logits → τ collapse). The test calls gemm_q8_0_wmma twice with the
//!    SAME x-buffer (different content each call) and verifies both produce
//!    correct output vs the scalar reference.
//!
//! "Pass" = max-abs logit diff within Q8/fp16 tolerance:
//!   mean_rel < 2e-3 AND max_rel < 3.5e-2
//!   (mirrors test_gemm_q8_residual_wmma.rs thresholds)
//!
//! Run: cargo run --release --example test_gemm_q8_0_wmma_parity
//! Exits 0 on pass, 1 on any failure.

use rdna_compute::{DType, Gpu};

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::synth_q8;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_q8_0_wmma_parity ===");
    eprintln!("  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: WMMA path requires RDNA3+ (gfx11/12), got {arch}");
        std::process::exit(0);
    }

    let mut total_fail = 0usize;

    // ── Suite 1: Static parity (static content each call) ─────────────────
    eprintln!("\n=== Suite 1: static parity ===");
    {
        let m = 8192usize;
        let k = 2048usize;
        eprintln!("\n--- M={m}, K={k} (batch sweep) ---");

        let w = synth_q8(m, k, 0xDEADBEEF);
        let d_a = gpu.upload_raw(&w, &[w.len()]).unwrap();

        let batches: &[usize] = &[1, 2, 4, 8, 16, 64];
        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k).map(synth_x).collect();
        let d_x_full = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        for &n in batches {
            let x_n = d_x_full.sub_offset(0, n * k);

            let d_y_wmma = gpu.zeros(&[n * m], DType::F32).unwrap();
            gpu.gemm_q8_0_wmma(&d_a, &x_n, &d_y_wmma, m, k, n).unwrap();

            let d_y_ref = gpu.zeros(&[n * m], DType::F32).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_a, &x_n, &d_y_ref, m, k, n)
                .unwrap();

            let y_wmma = gpu.download_f32(&d_y_wmma).unwrap();
            let y_ref = gpu.download_f32(&d_y_ref).unwrap();
            let stats = compare(&y_wmma, &y_ref);
            let pass = stats.mean_rel < 2e-3 && stats.max_rel < 3.5e-2;
            let mark = if pass {
                "PASS"
            } else {
                total_fail += 1;
                "FAIL"
            };
            eprintln!(
                "  N={n:4}  {mark}   mean_rel={:.3e}  max_rel={:.3e}",
                stats.mean_rel, stats.max_rel,
            );
        }
    }

    {
        let m = 248320usize;
        let k = 2048usize;
        let n = 2usize;
        eprintln!("\n--- M={m} (lm_head), K={k}, N={n} ---");

        let w = synth_q8(m, k, 0xCAFEBABE);
        let d_a = gpu.upload_raw(&w, &[w.len()]).unwrap();
        let x_host: Vec<f32> = (0..n * k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[n * k]).unwrap();

        let d_y_wmma = gpu.zeros(&[n * m], DType::F32).unwrap();
        gpu.gemm_q8_0_wmma(&d_a, &d_x, &d_y_wmma, m, k, n).unwrap();

        let d_y_ref = gpu.zeros(&[n * m], DType::F32).unwrap();
        gpu.gemm_q8_0_batched_chunked(&d_a, &d_x, &d_y_ref, m, k, n)
            .unwrap();

        let y_wmma = gpu.download_f32(&d_y_wmma).unwrap();
        let y_ref = gpu.download_f32(&d_y_ref).unwrap();
        let stats = compare(&y_wmma, &y_ref);
        let pass = stats.mean_rel < 2e-3 && stats.max_rel < 3.5e-2;
        let mark = if pass {
            "PASS"
        } else {
            total_fail += 1;
            "FAIL"
        };
        eprintln!(
            "  N={n:4}  {mark}   mean_rel={:.3e}  max_rel={:.3e}",
            stats.mean_rel, stats.max_rel,
        );

        let max_abs_diff = y_wmma
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_abs_ref = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "         max_abs_diff={:.4e}  max_abs_ref={:.4e}",
            max_abs_diff, max_abs_ref
        );
    }

    // ── Suite 2: Stale-cache suite (same X buffer, different content each call) ──
    //
    // Simulates the MTP decode scenario: `tmp_batched` is a FIXED tensor that
    // gets new content from rmsnorm_batched each step. The first call to
    // gemm_q8_0_wmma with this pointer triggers ensure_fp16_x → converts + caches
    // the pointer. The SECOND call with the SAME pointer but NEW content must
    // ALSO reconvert. The stale-cache bug returns the first call's FP16 data
    // for all subsequent calls → wrong logits.
    //
    // Fix: gemm_q8_0_wmma must use convert_fp16_x_uncached instead of ensure_fp16_x.
    eprintln!("\n=== Suite 2: stale-cache (same pointer, different content) ===");
    {
        let m = 8192usize;
        let k = 2048usize;
        let n = 2usize;

        let w = synth_q8(m, k, 0xABCD1234);
        let d_a = gpu.upload_raw(&w, &[w.len()]).unwrap();

        // Allocate a REUSABLE x buffer — same address, will be overwritten between calls.
        let x_step0: Vec<f32> = (0..n * k).map(|i| synth_x(i)).collect();
        let x_step1: Vec<f32> = (0..n * k).map(|i| synth_x(i + 100_000)).collect();

        // d_x_reuse = same GpuTensor (same device ptr) used both times.
        let d_x_reuse = gpu.upload_f32(&x_step0, &[n * k]).unwrap();

        // Pre-compute the reference outputs (scalar path, always correct).
        let d_ref0 = gpu.zeros(&[n * m], DType::F32).unwrap();
        gpu.gemm_q8_0_batched_chunked(&d_a, &d_x_reuse, &d_ref0, m, k, n)
            .unwrap();
        let ref0 = gpu.download_f32(&d_ref0).unwrap();

        // Overwrite d_x_reuse with step-1 content BEFORE computing ref1
        // (so both refs use the correct content for each step).
        let x_step1_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x_step1.as_ptr() as *const u8, x_step1.len() * 4) };
        gpu.hip.memcpy_htod(&d_x_reuse.buf, x_step1_bytes).unwrap();
        let d_ref1 = gpu.zeros(&[n * m], DType::F32).unwrap();
        gpu.gemm_q8_0_batched_chunked(&d_a, &d_x_reuse, &d_ref1, m, k, n)
            .unwrap();
        let ref1 = gpu.download_f32(&d_ref1).unwrap();

        // --- Call 1: Load step-0 content, run gemm_q8_0_wmma (warms the FP16 cache) ---
        let x_step0_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x_step0.as_ptr() as *const u8, x_step0.len() * 4) };
        gpu.hip.memcpy_htod(&d_x_reuse.buf, x_step0_bytes).unwrap();
        let d_y0 = gpu.zeros(&[n * m], DType::F32).unwrap();
        gpu.gemm_q8_0_wmma(&d_a, &d_x_reuse, &d_y0, m, k, n)
            .unwrap();
        let y0 = gpu.download_f32(&d_y0).unwrap();

        let stats0 = compare(&y0, &ref0);
        let pass0 = stats0.mean_rel < 2e-3 && stats0.max_rel < 3.5e-2;
        let mark0 = if pass0 {
            "PASS"
        } else {
            total_fail += 1;
            "FAIL"
        };
        eprintln!(
            "  step-0 (cache warm)  {mark0}   mean_rel={:.3e}  max_rel={:.3e}",
            stats0.mean_rel, stats0.max_rel,
        );

        // --- Call 2: Overwrite SAME d_x_reuse with step-1 content, run again ---
        // With the stale-cache bug: ensure_fp16_x sees same ptr → returns FP16 from step-0 → FAIL.
        // With the fix (convert_fp16_x_uncached): always reconverts → correct → PASS.
        gpu.hip.memcpy_htod(&d_x_reuse.buf, x_step1_bytes).unwrap();
        let d_y1 = gpu.zeros(&[n * m], DType::F32).unwrap();
        gpu.gemm_q8_0_wmma(&d_a, &d_x_reuse, &d_y1, m, k, n)
            .unwrap();
        let y1 = gpu.download_f32(&d_y1).unwrap();

        let stats1 = compare(&y1, &ref1);
        let pass1 = stats1.mean_rel < 2e-3 && stats1.max_rel < 3.5e-2;
        let mark1 = if pass1 {
            "PASS"
        } else {
            total_fail += 1;
            "FAIL"
        };
        eprintln!(
            "  step-1 (stale-cache) {mark1}   mean_rel={:.3e}  max_rel={:.3e}",
            stats1.mean_rel, stats1.max_rel,
        );

        // Also show step-0 FP16 vs step-1 data as context (expected mismatch if bug present).
        // This is the delta that shows HOW WRONG the stale-cache result is.
        let stale_err = compare(&y1, &ref0);
        eprintln!(
            "  (step-1 vs step-0 ref, shows 'stale' distance: mean={:.3e} max={:.3e})",
            stale_err.mean_rel, stale_err.max_rel
        );
    }

    eprintln!("\n=== {} failure(s) ===", total_fail);
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

struct Stats {
    mean_rel: f64,
    max_rel: f64,
}
fn compare(wmma: &[f32], reference: &[f32]) -> Stats {
    let max_ref = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let thr = max_ref * 0.01;
    let (mut sum, mut max_r, mut cnt) = (0.0f64, 0.0f64, 0usize);
    for (w, r) in wmma.iter().zip(reference.iter()) {
        if r.abs() > thr {
            let rel = ((w - r).abs() / r.abs()) as f64;
            sum += rel;
            if rel > max_r {
                max_r = rel;
            }
            cnt += 1;
        }
    }
    Stats {
        mean_rel: if cnt == 0 { 0.0 } else { sum / cnt as f64 },
        max_rel: max_r,
    }
}

fn synth_x(i: usize) -> f32 {
    let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
    (v * 1e-9) % 2.0 - 1.0
}
