// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Numeric parity for the generic-library kernel `gemm_iu8_i32_wmma`
//! (signed INT8 × INT8 → INT32, gfx1103 zero-LDS WMMA).
//!
//! Integer accumulation is exact, so this is an EXACT-match test (any
//! mismatch is a real bug). Run on an RDNA3 box (gfx1103/1100/1151).
//!
//!   cargo run -p rdna-compute --example parity_gemm_iu8_i32_wmma [M K B]

use rdna_compute::Gpu;

/// Deterministic signed int8 stream in [-127, 127].
fn lcg_i8(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (((s >> 16) & 0xff) as i32 - 128).clamp(-127, 127) as i8
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let b: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    assert_eq!(k % 16, 0, "K must be a multiple of 16");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP gemm_iu8_i32_wmma parity: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }

    let a = lcg_i8(1, m * k); // [M, K]
    let x = lcg_i8(2, b * k); // [B, K]

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

    let a_bytes: Vec<u8> = a.iter().map(|&v| v as u8).collect();
    let x_bytes: Vec<u8> = x.iter().map(|&v| v as u8).collect();
    let a_dev = gpu.upload_raw(&a_bytes, &[m, k]).unwrap();
    let x_dev = gpu.upload_raw(&x_bytes, &[b, k]).unwrap();
    let y_dev = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();

    gpu.gemm_iu8_i32_wmma(&a_dev, &x_dev, &y_dev, m, k, b)
        .unwrap();
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
            "gemm_iu8_i32_wmma parity M={m} K={k} B={b} on {}: EXACT match -> PASS",
            gpu.arch
        ),
        Some((i, r, g)) => println!(
            "gemm_iu8_i32_wmma parity M={m} K={k} B={b} on {}: {mismatches} mismatches \
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
