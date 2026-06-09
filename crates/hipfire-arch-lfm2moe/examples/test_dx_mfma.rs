// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Decisive layout check for the MFMA dX kernel: dX[b,K] = sum_n dY[b,n]*W[n,K]
// via the naive f32 kernel vs the MFMA kernel. Layout bug => rel-L2 O(1);
// correct + bf16 cast => ~1e-2. Sweeps body-linear shapes incl. ragged K/N.

use rdna_compute::{DType, Gpu, GpuTensor};

fn frand(s: usize) -> f32 { ((s.wrapping_mul(2654435761) % 4000) as f32 / 2000.0) - 1.0 }
fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }

fn one(gpu: &mut Gpu, b: usize, k: usize, n: usize) -> (f32, f32) {
    let dy: Vec<f32> = (0..b * n).map(|i| frand(i + 1)).collect();
    let w: Vec<f32> = (0..n * k).map(|i| frand(i + 99991)).collect();
    let dyg = up(gpu, &dy, &[b, n]);
    let wg = up(gpu, &w, &[n, k]);
    let dx_naive = gpu.zeros(&[b, k], DType::F32).unwrap();
    let dx_mfma = gpu.zeros(&[b, k], DType::F32).unwrap();
    gpu.linear_bwd_dx_f32(&dyg, &wg, &dx_naive, b, k, n).unwrap();
    gpu.linear_bwd_dx_mfma_f32(&dyg, &wg, &dx_mfma, b, k, n).unwrap();
    let aa = gpu.download_f32(&dx_naive).unwrap();
    let bb = gpu.download_f32(&dx_mfma).unwrap();
    let mut num = 0f64; let mut den = 0f64; let mut outl = 0usize;
    for i in 0..aa.len() {
        let d = (aa[i] - bb[i]) as f64; num += d * d; den += (aa[i] as f64).powi(2);
        if (aa[i] - bb[i]).abs() > 0.05 { outl += 1; }
    }
    ((num / den.max(1e-30)).sqrt() as f32, outl as f32 / aa.len() as f32)
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    // (batch, K, N): body-linear dX shapes
    let cases = [
        (16usize, 1024usize, 3072usize), // in_proj dX
        (16, 1024, 4608),                // w1/w3 dX
        (16, 4608, 1024),                // w2 dX
        (16, 1024, 1024),                // out_proj dX
        (16, 64, 64),                    // exact tile
        (16, 70, 50),                    // ragged K and N
        (16, 100, 130),                  // ragged, K-strip boundary
        (8, 256, 256),                   // batch<16 mask
    ];
    let mut allpass = true;
    for (b, k, n) in cases {
        let (l2, outl) = one(&mut gpu, b, k, n);
        let ok = l2 < 5e-2;
        allpass &= ok;
        println!("  b={b:<3} K={k:<5} N={n:<5}  rel_L2={l2:.3e}  outlier_frac={outl:.2e}  {}",
                 if ok { "ok" } else { "FAIL (layout?)" });
    }
    if allpass { println!("dx_mfma: PASS (layout correct, bf16-precision agreement)"); }
    else { println!("dx_mfma: FAIL"); std::process::exit(1); }
}
