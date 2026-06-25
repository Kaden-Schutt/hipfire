// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemm_oq4_grouped_f16_wmma` (OQ4+ batched PREFILL, W4A16) vs a CPU
//! oracle. The kernel dequantizes the 4-bit weight to f16 and multiplies by f16
//! activations; the reference computes the same dot in f32 (f16 rounding makes
//! it approximate, so the tolerance is looser than the exact-int decode path).
//!
//!   cargo run --release -p rdna-compute --example parity_gemm_oq4_f16_wmma [M K B]

use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcgf_vals(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1536);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let group = 256usize;
    assert_eq!(k % group, 0);
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP parity_gemm_oq4_f16_wmma: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }

    let wnib = lcg(1, m * (k / 2));
    let wsc: Vec<f32> = lcgf_vals(0x11, m * ng)
        .iter()
        .map(|v| 0.01 + v.abs() * 0.25)
        .collect();
    let mut wbuf = wnib.clone();
    for s in &wsc {
        wbuf.extend_from_slice(&s.to_le_bytes());
    }
    let x: Vec<f32> = lcgf_vals(3, b * k);

    let wd = gpu.upload_raw(&wbuf, &[wbuf.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[b, k]).unwrap();
    let yd = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();

    gpu.gemm_oq4_grouped_f16_wmma(&wd, &xd, &yd, m, k, b, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_gpu = gpu.download_f32(&yd).unwrap(); // [b, m]

    // CPU oracle: W4A16 with f16-rounded weight (sw*nib) and f16-rounded acts.
    let sext = |nib: u8| -> i32 {
        let v = (nib & 0xf) as i32;
        (v << 28) >> 28
    };
    // f32 oracle (no f16 rounding); tol absorbs the kernel's f16 weight/act
    // rounding (~2^-11 relative per element).
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for bi in 0..b {
        for row in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let sw = wsc[row * ng + g];
                for j in 0..group {
                    let kk = g * group + j;
                    let byte = wnib[row * (k / 2) + kk / 2];
                    let nib = if kk & 1 == 0 { byte & 0xf } else { byte >> 4 };
                    acc += sw * sext(nib) as f32 * x[bi * k + kk];
                }
            }
            let got = y_gpu[bi * m + row];
            max_abs = max_abs.max((got - acc).abs());
            max_mag = max_mag.max(acc.abs());
        }
    }
    let tol = 2e-2 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "parity_gemm_oq4_f16_wmma M={m} K={k} B={b} on {}: max_abs={max_abs:.5} (mag={max_mag:.2}, tol={tol:.5}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
