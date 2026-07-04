// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the tuned wave64 LDS W4A4 kernel `gemm_iu4_i32_wmma_lds`: it must
//! (a) match the exact integer reference and (b) agree bit-for-bit with the
//! production single-chain `gemm_iu4_i32_wmma`. Uses deliberately unaligned M/B to
//! exercise the block-tile bounds guards. Run on RDNA3.5 (wave64 WMMA). `K % 64`.
//!
//!   cargo run -p hipfire-rdna --example parity_gemm_iu4_i32_wmma_lds [M K B]

use hipfire_rdna::Gpu;

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
        println!("SKIP gemm_iu4_i32_wmma_lds parity: {} lacks WMMA", gpu.arch);
        return;
    }

    let a = lcg_i4(1, m * k);
    let x = lcg_i4(2, b * k);

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

    let a_dev = gpu.upload_raw(&pack_i4(&a, m, k), &[m, k / 2]).unwrap();
    let x_dev = gpu.upload_raw(&pack_i4(&x, b, k), &[b, k / 2]).unwrap();
    let y_prod = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    let y_lds = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();

    gpu.gemm_iu4_i32_wmma(&a_dev, &x_dev, &y_prod, m, k, b)
        .unwrap();
    gpu.gemm_iu4_i32_wmma_lds(&a_dev, &x_dev, &y_lds, m, k, b)
        .unwrap();
    gpu.device_synchronize().unwrap();

    let g_prod = download_i32(&mut gpu, &y_prod, b * m);
    let g_lds = download_i32(&mut gpu, &y_lds, b * m);

    let ref_ok = g_lds == y_ref;
    let twin_ok = g_lds == g_prod;
    println!(
        "gemm_iu4_i32_wmma_lds parity M={m} K={k} B={b} on {}:",
        gpu.arch
    );
    println!(
        "  vs exact int reference : {}",
        if ref_ok { "EXACT" } else { "MISMATCH" }
    );
    println!(
        "  vs production kernel    : {}",
        if twin_ok { "bit-identical" } else { "DIVERGED" }
    );
    if ref_ok && twin_ok {
        println!("-> PASS");
    } else {
        println!("-> FAIL");
        std::process::exit(1);
    }
}
