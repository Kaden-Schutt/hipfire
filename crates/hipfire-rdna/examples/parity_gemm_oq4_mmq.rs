// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemm_oq4_residual_mmq` (OQ4+ int8-WMMA MMQ) vs the W4A16 f16-WMMA
//! kernel `gemm_oq4_grouped_f16_wmma` (the validated reference). MMQ uses q8_1
//! int8 activation quant, so it carries ~int8 activation error vs the f16 path;
//! the tolerance allows that while catching layout/sign bugs (which blow up huge).
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_oq4_mmq [M K N]

use hipfire_rdna::{DType, Gpu};

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
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let group = 256usize;
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_gemm_oq4_mmq: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let wnib = lcg(1, m * (k / 2));
    let wsc: Vec<f32> = lcgf_vals(0x11, m * ng)
        .iter()
        .map(|v| 0.01 + v.abs() * 0.1)
        .collect();
    let mut wbuf = wnib.clone();
    for s in &wsc {
        wbuf.extend_from_slice(&s.to_le_bytes());
    }
    let x: Vec<f32> = lcgf_vals(3, n * k);

    let wd = gpu.upload_raw(&wbuf, &[wbuf.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[n, k]).unwrap();

    // Reference: f16 W4A16 (per-batch-row grouped GEMM).
    let yref = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    gpu.gemm_oq4_grouped_f16_wmma(&wd, &xd, &yref, m, k, n, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_ref = gpu.download_f32(&yref).unwrap();

    // MMQ (int8 q8_1 activation), add=0 (SET).
    let ymmq = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    gpu.gemm_oq4_residual_mmq(&wd, &xd, &ymmq, m, k, n, false)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_mmq = gpu.download_f32(&ymmq).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for i in 0..n * m {
        max_abs = max_abs.max((y_mmq[i] - y_ref[i]).abs());
        max_mag = max_mag.max(y_ref[i].abs());
    }
    let rel = max_abs / max_mag.max(1e-6);
    // int8-act error budget: ~1/127 per element, partially averaging over K.
    let pass = rel <= 0.05;
    println!(
        "parity_gemm_oq4_mmq M={m} K={k} N={n} on {}: max_abs={max_abs:.4} mag={max_mag:.2} rel={rel:.4} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
