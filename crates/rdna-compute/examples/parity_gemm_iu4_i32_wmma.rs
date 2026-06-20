// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Numeric parity for the generic-library kernel `gemm_iu4_i32_wmma`
//! (signed INT4 × INT4 → INT32, gfx1103 zero-LDS WMMA).
//!
//! Integer accumulation is exact, so this is an EXACT-match test. Inputs are
//! signed 4-bit values in [-8, 7], packed two-per-byte (k_even | k_odd<<4).
//! Run on an RDNA3 box (gfx1103/1100/1151).
//!
//!   cargo run -p rdna-compute --example parity_gemm_iu4_i32_wmma [M K B]

use rdna_compute::Gpu;

/// Deterministic signed int4 stream in [-8, 7].
fn lcg_i4(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (((s >> 16) & 0xf) as i32 - 8) as i8 // [-8, 7]
        })
        .collect()
}

/// Pack a [rows, K] signed-int4 matrix into [rows, K/2] bytes:
/// byte = (k_even & 0xf) | ((k_odd & 0xf) << 4).
fn pack_i4(vals: &[i8], rows: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * (k / 2)];
    for r in 0..rows {
        for kk in (0..k).step_by(2) {
            let lo = (vals[r * k + kk] as u8) & 0xf;
            let hi = (vals[r * k + kk + 1] as u8) & 0xf;
            out[r * (k / 2) + kk / 2] = lo | (hi << 4);
        }
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let b: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    assert_eq!(k % 16, 0, "K must be a multiple of 16");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP gemm_iu4_i32_wmma parity: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let a = lcg_i4(1, m * k); // [M, K]
    let x = lcg_i4(2, b * k); // [B, K]

    // Exact integer reference.
    let mut y_ref = vec![0i32; b * m]; // [B, M]
    for bi in 0..b {
        for mi in 0..m {
            let mut acc: i64 = 0;
            for ki in 0..k {
                acc += a[mi * k + ki] as i64 * x[bi * k + ki] as i64;
            }
            y_ref[bi * m + mi] = acc as i32;
        }
    }

    let a_dev = gpu.upload_raw(&pack_i4(&a, m, k), &[m, k / 2]).unwrap();
    let x_dev = gpu.upload_raw(&pack_i4(&x, b, k), &[b, k / 2]).unwrap();
    let y_dev = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();

    gpu.gemm_iu4_i32_wmma(&a_dev, &x_dev, &y_dev, m, k, b).unwrap();
    gpu.device_synchronize().unwrap();

    let y_bytes = gpu.download_raw(&y_dev, b * m * 4).unwrap();
    let y_gpu: Vec<i32> = y_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut mismatches = 0usize;
    let mut first = None;
    for i in 0..b * m {
        if y_gpu[i] != y_ref[i] {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, y_ref[i], y_gpu[i]));
            }
        }
    }

    let pass = mismatches == 0;
    match first {
        None => println!(
            "gemm_iu4_i32_wmma parity M={m} K={k} B={b} on {}: EXACT match -> PASS",
            gpu.arch
        ),
        Some((i, r, g)) => println!(
            "gemm_iu4_i32_wmma parity M={m} K={k} B={b} on {}: {mismatches} mismatches \
             (first @ b={},m={}: ref={r} gpu={g}) -> FAIL",
            gpu.arch,
            i / m,
            i % m
        ),
    }
    if !pass {
        std::process::exit(1);
    }
}
