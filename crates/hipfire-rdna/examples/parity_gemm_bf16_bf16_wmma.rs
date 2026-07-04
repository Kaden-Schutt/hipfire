// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Numeric parity for the generic-library kernel `gemm_bf16_bf16_wmma`
//! (BF16 × BF16 → BF16, gfx1103 zero-LDS WMMA).
//!
//! Verifies the WMMA output layout / OPSEL packing is correct by comparing
//! against a CPU reference computed on the SAME bf16-rounded inputs. The
//! hardware accumulates in bf16 precision, so the tolerance is loose — its
//! only job is to separate "layout correct" (diffs ~ a few %) from "layout
//! wrong" (garbage). Run on an RDNA3 box (gfx1103/1100/1151).
//!
//!   cargo run -p hipfire-rdna --example parity_gemm_bf16_bf16_wmma [M K B]

use hipfire_rdna::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((s >> 16) & 0x7fff) as f32 / 32768.0 - 0.5
        })
        .collect()
}

/// f32 -> bf16 (round-to-nearest-even), returned as raw u16.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return (bits >> 16) as u16; // NaN passthrough
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&f32_to_bf16(x).to_le_bytes());
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
        println!(
            "SKIP gemm_bf16_bf16_wmma parity: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }

    let a_f32 = lcg(1, m * k); // [M, K]
    let x_f32 = lcg(2, b * k); // [B, K]

    // Reference: bf16-rounded inputs, F32 accumulation over K (matching the
    // kernel's v_wmma_f32_16x16x16_bf16 path), then a single round to BF16 on
    // store. Diff vs GPU is then just F32 reduction-order + one bf16 ULP.
    let a_r: Vec<f32> = a_f32.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let x_r: Vec<f32> = x_f32.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let mut y_ref = vec![0.0f32; b * m]; // [B, M]
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a_r[mi * k + ki] * x_r[bi * k + ki];
            }
            // single bf16 rounding on store, as the kernel does
            y_ref[bi * m + mi] = bf16_to_f32(f32_to_bf16(acc));
        }
    }

    let a_dev = gpu.upload_raw(&to_bf16_bytes(&a_f32), &[m, k]).unwrap();
    let x_dev = gpu.upload_raw(&to_bf16_bytes(&x_f32), &[b, k]).unwrap();
    let y_dev = gpu.upload_raw(&vec![0u8; b * m * 2], &[b, m]).unwrap();

    gpu.gemm_bf16_bf16_wmma(&a_dev, &x_dev, &y_dev, m, k, b)
        .unwrap();
    gpu.device_synchronize().unwrap();

    let y_bytes = gpu.download_raw(&y_dev, b * m * 2).unwrap();
    let y_gpu: Vec<f32> = y_bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut argmax = 0usize;
    for i in 0..b * m {
        let d = (y_gpu[i] - y_ref[i]).abs();
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
        let denom = y_ref[i].abs().max(1e-3);
        max_rel = max_rel.max(d / denom);
    }

    // F32 accumulate + single bf16 store: residual is ~a couple bf16 ULP of the
    // output magnitude (one rounding) plus tiny f32 reduction-order drift.
    let max_mag = y_ref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let tol_abs = 3.0 * max_mag * 2.0f32.powi(-8) + 1e-3;
    let pass = max_abs <= tol_abs;
    println!(
        "gemm_bf16_bf16_wmma parity M={m} K={k} B={b} on {}: \
         max_abs={max_abs:.4} (ref={:.4}, gpu={:.4} @ b={},m={}) max_rel={max_rel:.3} tol_abs={tol_abs:.4} -> {}",
        gpu.arch,
        y_ref[argmax],
        y_gpu[argmax],
        argmax / m,
        argmax % m,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
