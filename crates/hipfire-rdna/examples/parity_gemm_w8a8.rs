// SPDX-License-Identifier: Apache-2.0
//! W8A8 reference GEMM — full GPU path (reference kernel layer, quant-targets plan).
//!
//! Composes the reference W8A8 cell entirely on-GPU:
//!   quantize_act_int8_per_token(X)  →  gemm_iu8_i32_wmma(W_i8, Xq)  →
//!   dequant_i32_rowcol(·, x_scale, w_scale)
//! and validates the f32 output against an f32 reference (W @ Xᵀ). Weights are
//! per-channel symmetric int8 (host-quantized here; the loader/QuantType emit is a
//! follow-up). Activations are per-token symmetric int8 (the GPU kernel under test).
//!
//! Unlike W4A8 (per-channel W4 floored ~18 dB), per-channel W8 + per-token A8 is the
//! high-fidelity integer baseline — expect cos ≈ 0.9995+ and SQNR ≳ 35 dB.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_w8a8 [M K B]

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    (0..n)
        .map(|_| {
            let u1 = u().max(1e-7);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let b: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    assert_eq!(k % 16, 0, "K must be a multiple of 16");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_gemm_w8a8: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let w = lcg(1, m * k); // [M, K]
    let x = lcg(2, b * k); // [B, K]

    // ── W8: per-channel (row) symmetric int8, host-quantized.
    let mut wq = vec![0i8; m * k];
    let mut w_scale = vec![0.0f32; m];
    for mi in 0..m {
        let amax = (0..k).map(|ki| w[mi * k + ki].abs()).fold(0.0f32, f32::max);
        let s = (amax / 127.0).max(1e-8);
        w_scale[mi] = s;
        for ki in 0..k {
            wq[mi * k + ki] = ((w[mi * k + ki] / s).round()).clamp(-127.0, 127.0) as i8;
        }
    }

    // ── Upload. Weights int8, activations f32 (the GPU quantizes them).
    let w_bytes: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
    let w_dev = gpu.upload_raw(&w_bytes, &[m, k]).unwrap();
    let ws_dev = gpu.upload_f32(&w_scale, &[m]).unwrap();
    let x_dev = gpu.upload_f32(&x, &[b, k]).unwrap();

    // GPU W8A8 path.
    let xq_dev = gpu.upload_raw(&vec![0u8; b * k], &[b, k]).unwrap(); // int8
    let xs_dev = gpu.alloc_tensor(&[b], DType::F32).unwrap();
    let yi_dev = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap(); // int32
    let yf_dev = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();

    gpu.quantize_act_int8_per_token(&x_dev, &xq_dev, &xs_dev, b, k)
        .unwrap();
    gpu.gemm_iu8_i32_wmma(&w_dev, &xq_dev, &yi_dev, m, k, b)
        .unwrap();
    gpu.dequant_i32_rowcol(&yi_dev, &xs_dev, &ws_dev, &yf_dev, b, m)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y = gpu.download_f32(&yf_dev).unwrap();

    // ── f32 reference + quality.
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    let mut dot = 0.0f64;
    let (mut nr, mut ng) = (0.0f64, 0.0f64);
    for bi in 0..b {
        for mi in 0..m {
            let r: f32 = (0..k).map(|ki| w[mi * k + ki] * x[bi * k + ki]).sum();
            let g = y[bi * m + mi];
            sig += (r as f64).powi(2);
            err += ((r - g) as f64).powi(2);
            dot += r as f64 * g as f64;
            nr += (r as f64).powi(2);
            ng += (g as f64).powi(2);
        }
    }
    let sqnr_db = 10.0 * (sig / err.max(1e-30)).log10();
    let cos = dot / (nr.sqrt() * ng.sqrt() + 1e-30);
    let pass = cos > 0.999 && sqnr_db > 30.0;
    println!(
        "W8A8 GPU path (per-token A8 quant + iu8 WMMA + rowcol dequant) M={m} K={k} B={b} on {}: cos={cos:.6} SQNR={sqnr_db:.1}dB -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
