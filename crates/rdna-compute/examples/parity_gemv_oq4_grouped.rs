// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemv_oq4_grouped` (decode B=1 GEMV) vs the validated
//! `gemm_oq4_grouped_wmma` at batch_size=1. Bit-exact expected (identical
//! integer dots + per-group f32 rescale, just a different tiling).
//!
//!   cargo run --release -p rdna-compute --example parity_gemv_oq4_grouped [M K]

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcgf(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (0.01 + (s as f32 / 2_147_483_648.0) * 0.5).to_le_bytes()
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3584);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let group = 256usize;
    assert_eq!(k % group, 0);
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP parity_gemv_oq4_grouped: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }

    // Combined weight buffer [packed nibbles M*(K/2) | f32 scales M*ng].
    let mut wbuf = lcg(1, m * (k / 2));
    wbuf.extend_from_slice(&lcgf(0x11, m * ng));
    let xq = lcg(2, k / 2);
    let xs = lcgf(3, ng);

    let wd = gpu.upload_raw(&wbuf, &[wbuf.len()]).unwrap();
    let ws = wd.sub_offset(m * (k / 2), m * ng * 4);
    let xqd = gpu.upload_raw(&xq, &[1, k / 2]).unwrap();
    let xsd = gpu.upload_raw(&xs, &[1, ng]).unwrap();

    // GEMV (decode).
    let yg = gpu.upload_raw(&vec![0u8; m * 4], &[1, m]).unwrap();
    gpu.gemv_oq4_grouped(&wd, &ws, &xqd, &xsd, &yg, m, k, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_gemv = gpu.download_f32(&yg).unwrap();

    // Reference: WMMA grouped GEMM at B=1.
    let yr = gpu.upload_raw(&vec![0u8; m * 4], &[1, m]).unwrap();
    gpu.gemm_oq4_grouped_wmma(&wd, &ws, &xqd, &xsd, &yr, m, k, 1, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_ref = gpu.download_f32(&yr).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for i in 0..m {
        max_abs = max_abs.max((y_gemv[i] - y_ref[i]).abs());
        max_mag = max_mag.max(y_ref[i].abs());
    }
    let tol = 1e-3 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "parity_gemv_oq4_grouped M={m} K={k} on {}: max_abs={max_abs:.5} (mag={max_mag:.2}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
