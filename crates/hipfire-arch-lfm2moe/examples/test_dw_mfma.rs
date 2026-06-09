// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Decisive layout check for the MFMA dW kernel: compute dW[N,K] = sum_m
// dY[m,N]*X[m,K] via the naive f32 kernel and the MFMA kernel, compare.
// A fragment-layout bug => O(1) relative error (wrong elements). Correct
// layout + bf16 input cast => ~1e-2 relative error (bf16 mantissa). Sweeps
// several (M,K,N) including M<16 (mask path) and tile-boundary sizes.

use rdna_compute::{DType, Gpu, GpuTensor};

fn frand(s: usize) -> f32 { ((s.wrapping_mul(2654435761) % 4000) as f32 / 2000.0) - 1.0 }
fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }

fn one(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> (f32, f32) {
    let dy: Vec<f32> = (0..m * n).map(|i| frand(i + 1)).collect();
    let x: Vec<f32> = (0..m * k).map(|i| frand(i + 99991)).collect();
    let dyg = up(gpu, &dy, &[m, n]);
    let xg = up(gpu, &x, &[m, k]);
    let dw_naive = gpu.zeros(&[n, k], DType::F32).unwrap();
    let dw_mfma = gpu.zeros(&[n, k], DType::F32).unwrap();
    gpu.linear_bwd_dw_f32(&dyg, &xg, &dw_naive, m, k, n).unwrap();
    gpu.linear_bwd_dw_mfma_f32(&dyg, &xg, &dw_mfma, m, k, n).unwrap();
    let a = gpu.download_f32(&dw_naive).unwrap();
    let b = gpu.download_f32(&dw_mfma).unwrap();
    // Robust global metric: relative L2 ||a-b|| / ||a||. A fragment-layout bug
    // => O(1) (wrong elements dominate the norm). Correct layout + bf16 cast
    // => ~1e-2. Immune to per-element near-zero sign flips. Also count
    // abs-error outliers (should be a tiny handful = near-zero dW elements).
    let mut num = 0f64;
    let mut den = 0f64;
    let mut outliers = 0usize;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        num += d * d;
        den += (a[i] as f64) * (a[i] as f64);
        if (a[i] - b[i]).abs() > 0.05 { outliers += 1; }
    }
    let l2 = (num / den.max(1e-30)).sqrt() as f32;
    (l2, outliers as f32 / a.len() as f32)
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    // (M, K, N): body-linear-like shapes + tile-boundary + M<16 mask path
    let cases = [
        (16usize, 1024usize, 3072usize), // in_proj-like
        (16, 1024, 4608),                // w1/w3-like
        (16, 4608, 1024),                // w2-like
        (16, 1024, 1024),                // out_proj-like
        (16, 64, 64),                    // tiny exact tile
        (16, 70, 50),                    // ragged (mask on both dims)
        (8, 256, 256),                   // M<16 mask path
        (1, 128, 128),                   // M=1 rank-1
        (32, 1024, 512),                 // M=32 (fc grad, 2 chunks)
        (48, 256, 256),                  // M=48 (3 chunks)
    ];
    let mut allpass = true;
    for (m, k, n) in cases {
        let (l2, outlier_frac) = one(&mut gpu, m, k, n);
        // layout bug => L2 O(1); bf16 precision => ~1e-2. Threshold 5e-2.
        let ok = l2 < 5e-2;
        allpass &= ok;
        println!("  M={m:<3} K={k:<5} N={n:<5}  rel_L2={l2:.3e}  outlier_frac={outlier_frac:.2e}  {}",
                 if ok { "ok" } else { "FAIL (layout?)" });
    }
    if allpass { println!("dw_mfma: PASS (layout correct, bf16-precision agreement)"); }
    else { println!("dw_mfma: FAIL"); std::process::exit(1); }
}
