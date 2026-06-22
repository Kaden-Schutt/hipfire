// SPDX-License-Identifier: Apache-2.0
//! W4A8 GEMM numerics on the existing iu8 WMMA core (quantization-targets plan,
//! docs/plans/2026-06-22-quantization-targets.md — first build step).
//!
//! Validates the W4A8 design: affine symmetric int4 weights are EXPANDED to int8
//! (the code IS the integer level) and fed to `gemm_iu8_i32_wmma` against int8
//! activations; the int32 accumulator is then dequantized by w_scale·x_scale
//! (scales never enter the WMMA). This proves the nibble-expand + scale-after-
//! accumulate path that W4A8 and W8A8 share.
//!
//! Scope note: the plain iu8 core accumulates the FULL K into one int32, so this
//! uses a PER-CHANNEL weight scale (one per output row). Per-GROUP weight scales
//! (better quality) need grouped accumulation in the kernel (a follow-up, like the
//! hfq4 mmq path) — out of scope for this numerics check.
//!
//! Quant is lossy (W4 + A8), so this reports SQNR / cos vs an f32 reference, not an
//! exact match. Expect W4A8-class quality (cos > 0.99, SQNR ~30+ dB on benign data).
//!
//!   cargo run --release -p rdna-compute --example parity_w4a8 [M K B]

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    // Box-Muller-ish gaussian from an LCG (no rand dep).
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
        println!("SKIP parity_w4a8: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let w = lcg(1, m * k); // [M, K] weights
    let x = lcg(2, b * k); // [B, K] activations

    // ── W4: per-channel (per output row) symmetric int4, levels [-7, 7].
    let mut wq = vec![0i8; m * k];
    let mut w_scale = vec![0.0f32; m];
    for mi in 0..m {
        let amax = (0..k).map(|ki| w[mi * k + ki].abs()).fold(0.0f32, f32::max);
        let s = (amax / 7.0).max(1e-8);
        w_scale[mi] = s;
        for ki in 0..k {
            wq[mi * k + ki] = ((w[mi * k + ki] / s).round()).clamp(-7.0, 7.0) as i8;
        }
    }

    // ── A8: per-row (per token) symmetric int8, levels [-127, 127].
    let mut xq = vec![0i8; b * k];
    let mut x_scale = vec![0.0f32; b];
    for bi in 0..b {
        let amax = (0..k).map(|ki| x[bi * k + ki].abs()).fold(0.0f32, f32::max);
        let s = (amax / 127.0).max(1e-8);
        x_scale[bi] = s;
        for ki in 0..k {
            xq[bi * k + ki] = ((x[bi * k + ki] / s).round()).clamp(-127.0, 127.0) as i8;
        }
    }

    // ── iu8 WMMA over the expanded int4→int8 weights + int8 activations.
    let w_bytes: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
    let x_bytes: Vec<u8> = xq.iter().map(|&v| v as u8).collect();
    let w_dev = gpu.upload_raw(&w_bytes, &[m, k]).unwrap();
    let x_dev = gpu.upload_raw(&x_bytes, &[b, k]).unwrap();
    let y_dev = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    gpu.gemm_iu8_i32_wmma(&w_dev, &x_dev, &y_dev, m, k, b).unwrap();
    gpu.device_synchronize().unwrap();
    let y_bytes = gpu.download_raw(&y_dev, b * m * 4).unwrap();
    let y_i32: Vec<i32> = y_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // ── NUMERICS (the PASS criterion): the GPU int32 must EXACTLY equal the CPU
    // integer accumulate over the same expanded codes — this proves the
    // expand→iu8→(dequant) path is arithmetically correct, independent of quant
    // quality. (Integer accumulate is exact; the f32 dequant is applied identically
    // on both sides, so we compare the i32 cores directly.)
    let mut mismatches = 0usize;
    for bi in 0..b {
        for mi in 0..m {
            let mut acc: i64 = 0;
            for ki in 0..k {
                acc += wq[mi * k + ki] as i64 * xq[bi * k + ki] as i64;
            }
            if y_i32[bi * m + mi] != acc as i32 {
                mismatches += 1;
            }
        }
    }
    let numerics_ok = mismatches == 0;

    // ── QUALITY (informational): dequant vs the f32 reference. Per-channel W4 (one
    // scale per row, dominated by outliers, NO rotation/grouping) is the low-quality
    // floor — the production recipe adds per-GROUP scales + FWHT rotation (mq4/oq4).
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    let mut dot = 0.0f64;
    let (mut nr, mut ng) = (0.0f64, 0.0f64);
    for bi in 0..b {
        for mi in 0..m {
            let r: f32 = (0..k).map(|ki| w[mi * k + ki] * x[bi * k + ki]).sum();
            let g = y_i32[bi * m + mi] as f32 * w_scale[mi] * x_scale[bi];
            sig += (r as f64).powi(2);
            err += ((r - g) as f64).powi(2);
            dot += r as f64 * g as f64;
            nr += (r as f64).powi(2);
            ng += (g as f64).powi(2);
        }
    }
    let sqnr_db = 10.0 * (sig / err.max(1e-30)).log10();
    let cos = dot / (nr.sqrt() * ng.sqrt() + 1e-30);

    println!(
        "W4A8 (int4→iu8 expand) M={m} K={k} B={b} on {}: numerics={} ({mismatches} mismatch) | quality(per-channel W, no rot): cos={cos:.5} SQNR={sqnr_db:.1}dB -> {}",
        gpu.arch,
        if numerics_ok { "EXACT" } else { "BROKEN" },
        if numerics_ok { "PASS" } else { "FAIL" }
    );
    if !numerics_ok {
        std::process::exit(1);
    }
}
