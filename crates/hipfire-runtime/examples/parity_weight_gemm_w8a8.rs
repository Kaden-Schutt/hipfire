// SPDX-License-Identifier: Apache-2.0
//! Validate the `weight_gemm` W8A8Ref dispatch arm with a constructed weight (no
//! model needed). Builds a W8A8Ref WeightTensor — buffer = [M*K int8 weights |
//! M f32 per-channel scales] (byte-addressed) — runs `weight_gemm`, and compares the
//! f32 output to an f32 reference. Exercises the arm's co-located-scale slicing
//! (sub_offset byte math), the per-token activation quant, the iu8 WMMA, and the
//! rowcol dequant end-to-end.
//!
//! NOTE: this is the consume-side reference cell for the generic weight_gemm path
//! (llama/gemma3/qwen2/minimax). qwen3.5 uses fused GEMMs and does not route through
//! weight_gemm, so wiring W8A8 into a *running* qwen3.5 model needs the fused sites
//! (separate, bigger). No generic-arch bf16 model is on disk to run end-to-end yet.
//!
//!   cargo run --release -p hipfire-runtime --example parity_weight_gemm_w8a8 [M K B]

use hipfire_runtime::weights::{weight_gemm, WeightTensor};
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

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_weight_gemm_w8a8: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let w = lcg(1, m * k);
    let x = lcg(2, b * k);

    // Per-channel symmetric int8 weights + co-located f32 scales: [M*K int8 | M f32].
    let mut bytes: Vec<u8> = Vec::with_capacity(m * k + m * 4);
    let mut scale = vec![0.0f32; m];
    let mut codes = vec![0i8; m * k];
    for mi in 0..m {
        let amax = (0..k).map(|ki| w[mi * k + ki].abs()).fold(0.0f32, f32::max);
        let s = (amax / 127.0).max(1e-8);
        scale[mi] = s;
        for ki in 0..k {
            codes[mi * k + ki] = ((w[mi * k + ki] / s).round()).clamp(-127.0, 127.0) as i8;
        }
    }
    bytes.extend(codes.iter().map(|&c| c as u8));
    for &s in &scale {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let mut buf = gpu.upload_raw(&bytes, &[bytes.len()]).unwrap();
    buf.dtype = DType::W8A8Ref;
    let wt = WeightTensor {
        buf,
        gpu_dtype: DType::W8A8Ref,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    };

    let x_dev = gpu.upload_f32(&x, &[b, k]).unwrap();
    let y_dev = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
    weight_gemm(&mut gpu, &wt, &x_dev, &y_dev, b).unwrap();
    gpu.device_synchronize().unwrap();
    let y = gpu.download_f32(&y_dev).unwrap();

    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    let (mut dot, mut nr, mut ng) = (0.0f64, 0.0f64, 0.0f64);
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
        "weight_gemm W8A8Ref arm M={m} K={k} B={b} on {}: cos={cos:.6} SQNR={sqnr_db:.1}dB -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
