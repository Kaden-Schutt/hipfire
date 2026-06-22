// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// CPU-reference validation of the DFlash trainer fp32 forward kernels:
//   linear_fwd_f32      — Y[M,N] = X[M,K]·W[N,K]^T (row-major)
//   attn_block_ctx_f32  — naive GQA block attention with injected context.
// Both checked against an f64 host reference.

use rdna_compute::{DType, Gpu};

fn frand(seed: usize) -> f32 {
    ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let mut worst = 0f64;

    // ---------------- linear_fwd_f32 ----------------
    {
        let m = 16usize; // block tokens
        let k = 1024usize; // in features (LFM2.5 hidden)
        let n = 4608usize; // out features (FFN gate width)
        let x: Vec<f32> = (0..m * k).map(|i| frand(i + 1)).collect();
        let w: Vec<f32> = (0..n * k).map(|i| frand(i + 7_000)).collect();
        let xg = gpu.upload_f32(&x, &[m, k]).unwrap();
        let wg = gpu.upload_f32(&w, &[n, k]).unwrap();
        let yg = gpu.zeros(&[m, n], DType::F32).unwrap();
        gpu.linear_fwd_f32(&xg, &wg, &yg, m, k, n).unwrap();
        let y = gpu.download_f32(&yg).unwrap();
        let mut mx = 0f64;
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += x[mm * k + kk] as f64 * w[nn * k + kk] as f64;
                }
                let got = y[mm * n + nn] as f64;
                let den = acc.abs().max(1e-6);
                mx = mx.max((got - acc).abs() / den);
            }
        }
        println!("linear_fwd_f32 [{m}x{k} · {n}x{k}]: max rel err = {mx:.3e}");
        worst = worst.max(mx);
    }

    // ---------------- attn_block_ctx_f32 ----------------
    {
        let b = 16usize; // block (queries)
        let n_ctx = 24usize; // injected context positions
        let l = n_ctx + b;
        let n_h = 16usize;
        let n_kv = 8usize;
        let hd = 64usize;
        let group = n_h / n_kv;
        let scale = 1.0f32 / (hd as f32).sqrt();

        let q: Vec<f32> = (0..b * n_h * hd).map(|i| frand(i + 100)).collect();
        let k: Vec<f32> = (0..l * n_kv * hd).map(|i| frand(i + 50_000)).collect();
        let v: Vec<f32> = (0..l * n_kv * hd).map(|i| frand(i + 90_000)).collect();
        let qg = gpu.upload_f32(&q, &[b, n_h, hd]).unwrap();
        let kg = gpu.upload_f32(&k, &[l, n_kv, hd]).unwrap();
        let vg = gpu.upload_f32(&v, &[l, n_kv, hd]).unwrap();
        let og = gpu.zeros(&[b, n_h, hd], DType::F32).unwrap();
        gpu.attn_block_ctx_f32(&qg, &kg, &vg, &og, b, l, n_h, n_kv, hd, scale)
            .unwrap();
        let o = gpu.download_f32(&og).unwrap();

        let mut mx = 0f64;
        for i in 0..b {
            for h in 0..n_h {
                let g = h / group;
                // scores
                let mut sc = vec![0f64; l];
                let mut mmax = f64::NEG_INFINITY;
                for j in 0..l {
                    let mut d = 0f64;
                    for dd in 0..hd {
                        d += q[(i * n_h + h) * hd + dd] as f64 * k[(j * n_kv + g) * hd + dd] as f64;
                    }
                    sc[j] = d * scale as f64;
                    if sc[j] > mmax {
                        mmax = sc[j];
                    }
                }
                let mut z = 0f64;
                for j in 0..l {
                    sc[j] = (sc[j] - mmax).exp();
                    z += sc[j];
                }
                for dd in 0..hd {
                    let mut acc = 0f64;
                    for j in 0..l {
                        acc += sc[j] * v[(j * n_kv + g) * hd + dd] as f64;
                    }
                    acc /= z;
                    let got = o[(i * n_h + h) * hd + dd] as f64;
                    mx = mx.max((got - acc).abs() / acc.abs().max(1e-4));
                }
            }
        }
        println!("attn_block_ctx_f32 [B={b} n_ctx={n_ctx} nH={n_h} nKV={n_kv} hd={hd}]: max rel err = {mx:.3e}");
        worst = worst.max(mx);
    }

    if worst < 2e-3 {
        println!("dflash_train kernels: PASS (worst rel err {worst:.3e})");
    } else {
        println!("dflash_train kernels: FAIL (worst rel err {worst:.3e})");
        std::process::exit(1);
    }
}
