// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the tuned wave64 LDS W3A4 kernel `gemm_w3a4_i32_wmma_lds`: it must
//! (a) match the exact integer reference and (b) agree bit-for-bit with the int4
//! LDS twin `gemm_iu4_i32_wmma_lds` fed the SAME weight values (3-bit weights are a
//! subset of int4, so the two must produce identical int32 outputs). Uses
//! deliberately unaligned M/B to exercise the block-tile bounds guards. Run on
//! RDNA3.5 (wave64 WMMA). `K % 64`.
//!
//!   cargo run -p hipfire-rdna --example parity_gemm_w3a4_i32_wmma_lds [M K B]

use hipfire_rdna::Gpu;

// signed 3-bit weights in [-4, 3]
fn lcg_w3(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (((s >> 16) & 0x7) as i32 - 4) as i8 // [-4, 3]
        })
        .collect()
}

// signed int4 activations in [-8, 7]
fn lcg_i4(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (((s >> 16) & 0xf) as i32 - 8) as i8 // [-8, 7]
        })
        .collect()
}

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

// bit-plane pack signed 3-bit weights: [M, 3K/32] u32, 3 planes per 32-group.
fn pack_w3(vals: &[i8], rows: usize, k: usize) -> Vec<u8> {
    let stw = 3 * k / 32; // u32 per row
    let mut out = vec![0u32; rows * stw];
    for r in 0..rows {
        for g in 0..(k / 32) {
            let (mut p0, mut p1, mut p2) = (0u32, 0u32, 0u32);
            for i in 0..32 {
                let v = (vals[r * k + g * 32 + i] as u8) & 7; // 3-bit two's-comp
                p0 |= ((v & 1) as u32) << i;
                p1 |= (((v >> 1) & 1) as u32) << i;
                p2 |= (((v >> 2) & 1) as u32) << i;
            }
            let base = r * stw + g * 3;
            out[base] = p0;
            out[base + 1] = p1;
            out[base + 2] = p2;
        }
    }
    out.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn download_i32(gpu: &mut Gpu, y_dev: &hipfire_rdna::GpuTensor, n: usize) -> Vec<i32> {
    let bytes = gpu.download_raw(y_dev, n * 4).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    // Unaligned defaults (M not %64, B not %256) to exercise the bounds guards.
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let b: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(300);
    assert_eq!(k % 64, 0, "K must be a multiple of 64");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP gemm_w3a4_i32_wmma_lds parity: {} lacks WMMA",
            gpu.arch
        );
        return;
    }

    let a = lcg_w3(1, m * k); // 3-bit weights (also valid int4)
    let x = lcg_i4(2, b * k); // int4 activations

    let mut y_ref = vec![0i32; b * m];
    for bi in 0..b {
        for mi in 0..m {
            let mut acc: i64 = 0;
            for ki in 0..k {
                acc += a[mi * k + ki] as i64 * x[bi * k + ki] as i64;
            }
            y_ref[bi * m + mi] = acc as i32;
        }
    }

    let a_w3 = gpu
        .upload_raw(&pack_w3(&a, m, k), &[m, 3 * k / 32])
        .unwrap();
    let a_i4 = gpu.upload_raw(&pack_i4(&a, m, k), &[m, k / 2]).unwrap();
    let x_dev = gpu.upload_raw(&pack_i4(&x, b, k), &[b, k / 2]).unwrap();
    let y_w3 = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    let y_i4 = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();

    gpu.gemm_w3a4_i32_wmma_lds(&a_w3, &x_dev, &y_w3, m, k, b)
        .unwrap();
    gpu.gemm_iu4_i32_wmma_lds(&a_i4, &x_dev, &y_i4, m, k, b)
        .unwrap();
    gpu.device_synchronize().unwrap();

    let g_w3 = download_i32(&mut gpu, &y_w3, b * m);
    let g_i4 = download_i32(&mut gpu, &y_i4, b * m);

    let ref_ok = g_w3 == y_ref;
    let twin_ok = g_w3 == g_i4;
    println!(
        "gemm_w3a4_i32_wmma_lds parity M={m} K={k} B={b} on {}:",
        gpu.arch
    );
    println!(
        "  vs exact int reference : {}",
        if ref_ok { "EXACT" } else { "MISMATCH" }
    );
    println!(
        "  vs int4 LDS twin        : {}",
        if twin_ok { "bit-identical" } else { "DIVERGED" }
    );
    if ref_ok && twin_ok {
        println!("-> PASS");
    } else {
        println!("-> FAIL");
        std::process::exit(1);
    }
}
