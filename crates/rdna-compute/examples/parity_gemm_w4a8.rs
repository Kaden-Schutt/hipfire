// SPDX-License-Identifier: Apache-2.0
//! W4A8 batched reference GEMM — full GPU path (reference kernel layer).
//!
//! Composes the W4A8 reference cell on-GPU: packed int4 weights are expanded to int8
//! (nibble_expand_int4_to_int8), then run through the validated W8A8 path
//!   quantize_act_int8_per_token(X) → gemm_iu8_i32_wmma → dequant_i32_rowcol.
//! The only NEW kernel vs W8A8 is the nibble expand; this validates it EXACTLY
//! (downloaded int8 must equal the host sign-extend of the codes) and reports the
//! end-to-end quality vs an f32 reference.
//!
//! Per-channel W4 (one scale per row, no rotation/grouping) is the low-quality FLOOR
//! (~cos 0.99, ~18 dB SQNR) — correct, kept for matrix completeness; the per-group
//! grouped-iu8 kernel is the quality overlay (follow-up).
//!
//!   cargo run --release -p rdna-compute --example parity_gemm_w4a8 [M K B]

use rdna_compute::{DType, Gpu};

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
    assert_eq!(k % 2, 0, "K must be even (2 codes/byte)");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_gemm_w4a8: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let w = lcg(1, m * k);
    let x = lcg(2, b * k);

    // ── W4: per-channel symmetric int4 codes in [-8,7], host-quantized + packed.
    let mut codes = vec![0i8; m * k];
    let mut w_scale = vec![0.0f32; m];
    for mi in 0..m {
        let amax = (0..k).map(|ki| w[mi * k + ki].abs()).fold(0.0f32, f32::max);
        let s = (amax / 7.0).max(1e-8); // /7 keeps it inside [-7,7], symmetric
        w_scale[mi] = s;
        for ki in 0..k {
            codes[mi * k + ki] = ((w[mi * k + ki] / s).round()).clamp(-7.0, 7.0) as i8;
        }
    }
    // Pack two 4-bit two's-complement codes per byte (low nibble = even k).
    let mut packed = vec![0u8; m * k / 2];
    for i in 0..m * k {
        let nib = (codes[i] as u8) & 0xf;
        if i % 2 == 0 {
            packed[i / 2] = nib;
        } else {
            packed[i / 2] |= nib << 4;
        }
    }

    let wp_dev = gpu.upload_raw(&packed, &[m, k / 2]).unwrap();
    let w8_dev = gpu.upload_raw(&vec![0u8; m * k], &[m, k]).unwrap(); // int8
    let ws_dev = gpu.upload_f32(&w_scale, &[m]).unwrap();
    let x_dev = gpu.upload_f32(&x, &[b, k]).unwrap();
    let xq_dev = gpu.upload_raw(&vec![0u8; b * k], &[b, k]).unwrap();
    let xs_dev = gpu.alloc_tensor(&[b], DType::F32).unwrap();
    let yi_dev = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    let yf_dev = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();

    // GPU W4A8 path: expand → (W8A8 path).
    gpu.nibble_expand_int4_to_int8(&wp_dev, &w8_dev, m, k).unwrap();
    gpu.quantize_act_int8_per_token(&x_dev, &xq_dev, &xs_dev, b, k).unwrap();
    gpu.gemm_iu8_i32_wmma(&w8_dev, &xq_dev, &yi_dev, m, k, b).unwrap();
    gpu.dequant_i32_rowcol(&yi_dev, &xs_dev, &ws_dev, &yf_dev, b, m).unwrap();
    gpu.device_synchronize().unwrap();

    // ── Validate the NEW expand kernel EXACTLY: downloaded int8 == codes.
    let w8 = gpu.download_raw(&w8_dev, m * k).unwrap();
    let mut expand_mismatch = 0usize;
    for i in 0..m * k {
        if (w8[i] as i8) != codes[i] {
            expand_mismatch += 1;
        }
    }

    // ── End-to-end quality vs f32 reference.
    let y = gpu.download_f32(&yf_dev).unwrap();
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

    let pass = expand_mismatch == 0 && cos > 0.98;
    println!(
        "W4A8 GPU path (int4 expand + iu8 + dequant) M={m} K={k} B={b} on {}: expand={} | quality(per-channel W4): cos={cos:.5} SQNR={sqnr_db:.1}dB -> {}",
        gpu.arch,
        if expand_mismatch == 0 { "EXACT" } else { "BROKEN" },
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
