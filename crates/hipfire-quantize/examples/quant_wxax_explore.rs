// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! First-principles study of activation precision for the fused-iu4 path on
//! gfx1103. Measures GEMM *output* SQNR (what model quality actually depends
//! on) for W4A16 / W8A8 / W4A8 / W4A4, with and without FWHT rotation of both
//! operands.
//!
//! Why output SQNR, not weight reconstruction: activation quant error enters
//! the product directly, and activations have far worse (per-channel) outliers
//! than weights. The fused iu4 WMMA (W4A4) is the max-throughput compute path
//! (prefill/batched), but only viable if A4 quality holds.
//!
//! Rotation identity: Y = X·Wᵀ = (XQ)(WQ)ᵀ for orthogonal Q, so rotating both
//! along K is exact but Gaussianizes both → int4 activations become tolerable
//! (QuaRot/SpinQuant). The engine already FWHT-rotates activations at runtime.
//!
//!   cargo run -p hipfire-quantize --example quant_wxax_explore [M] [K] [B]

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};

fn lcg_gauss(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = u().max(1e-7);
        let u2 = u();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f32::consts::TAU * u2).cos());
        if out.len() < n {
            out.push(r * (std::f32::consts::TAU * u2).sin());
        }
    }
    out
}

/// Per-channel (column) outliers along K — the dominant activation-outlier mode
/// in LLMs (a few feature channels with very large magnitude).
fn with_channel_outliers(
    mut w: Vec<f32>,
    rows: usize,
    k: usize,
    n_hot: usize,
    gain: f32,
    seed: u32,
) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9;
    let mut hot = vec![false; k];
    for _ in 0..n_hot {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        hot[(s as usize) % k] = true;
    }
    for r in 0..rows {
        for c in 0..k {
            if hot[c] {
                w[r * k + c] *= gain;
            }
        }
    }
    w
}

/// Rotate every row's K dimension in 256-blocks in place (K must be %256).
fn rotate_rows(m: &mut [f32], rows: usize, k: usize, s1: &[f32], s2: &[f32]) {
    let mut buf = [0.0f32; 256];
    for r in 0..rows {
        for seg in 0..(k / 256) {
            let base = r * k + seg * 256;
            buf.copy_from_slice(&m[base..base + 256]);
            cpu_fwht_256(&mut buf, s1, s2);
            m[base..base + 256].copy_from_slice(&buf);
        }
    }
}

/// Symmetric uniform per-group quant→dequant of every row (group along K).
fn quant_sym_rows(src: &[f32], rows: usize, k: usize, group: usize, bits: u32) -> Vec<f32> {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut amax = 1e-12f32;
            for c in g0..g1 {
                amax = amax.max(src[r * k + c].abs());
            }
            let scale = amax / qmax;
            for c in g0..g1 {
                let q = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                out[r * k + c] = q * scale;
            }
        }
    }
    out
}

fn matmul_out_sqnr(x: &[f32], w: &[f32], yref: &[f32], m: usize, k: usize, b: usize) -> f64 {
    // Y[b,m] = sum_k X[b,k] * W[m,k]
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[bi * k + ki] * w[mi * k + ki];
            }
            let r = yref[bi * m + mi] as f64;
            sig += r * r;
            let d = r - acc as f64;
            noise += d * d;
        }
    }
    if noise <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / noise).log10()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    assert_eq!(k % 256, 0, "K must be a multiple of 256 for FWHT-256");
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let g = 128usize;

    // Weights: gaussian + mild outliers. Activations: gaussian + STRONG
    // per-channel outliers (the realistic LLM activation regime).
    let w = with_channel_outliers(lcg_gauss(1, m * k), m, k, 8, 6.0, 7);
    let x = with_channel_outliers(lcg_gauss(2, b * k), b, k, 16, 20.0, 9);

    // Exact f32 reference output.
    let mut yref = vec![0.0f32; b * m];
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[bi * k + ki] * w[mi * k + ki];
            }
            yref[bi * m + mi] = acc;
        }
    }

    // Rotated copies (rotation is exact: rotated product == reference).
    let mut wr = w.clone();
    let mut xr = x.clone();
    rotate_rows(&mut wr, m, k, &s1, &s2);
    rotate_rows(&mut xr, b, k, &s1, &s2);

    println!("quant_wxax_explore  M={m} K={k} B={b}  group={g}");
    println!("activations: gaussian + 16 hot channels ×20 ; weights + 8 ×6\n");
    println!("{:<26} {:>10} {:>14}", "scheme", "path", "out SQNR dB");
    println!("{}", "-".repeat(54));

    let f16ish = |v: &[f32]| v.to_vec(); // proxy for A16 (f16 error ≪ int4 error)

    // No rotation
    let w4 = quant_sym_rows(&w, m, k, g, 4);
    let w8 = quant_sym_rows(&w, m, k, g, 8);
    let x4 = quant_sym_rows(&x, b, k, g, 4);
    let x8 = quant_sym_rows(&x, b, k, g, 8);
    let row = |name: &str, path: &str, xx: &[f32], ww: &[f32]| {
        let q = matmul_out_sqnr(xx, ww, &yref, m, k, b);
        println!("{name:<26} {path:>10} {q:>14.2}");
    };
    row("W4A16", "iu4*+deq", &f16ish(&x), &w4);
    row("W8A8", "iu8 wmma", &x8, &w8);
    row("W4A8", "mixed", &x8, &w4);
    row("W4A4", "iu4 wmma", &x4, &w4);

    println!();
    // With rotation (both operands; product preserved). Group=256 aligns with
    // the rotation block; quantize the rotated tensors.
    let w4r = quant_sym_rows(&wr, m, k, g, 4);
    let w8r = quant_sym_rows(&wr, m, k, g, 8);
    let x4r = quant_sym_rows(&xr, b, k, g, 4);
    let x8r = quant_sym_rows(&xr, b, k, g, 8);
    let rowr = |name: &str, path: &str, xx: &[f32], ww: &[f32]| {
        let q = matmul_out_sqnr(xx, ww, &yref, m, k, b);
        println!("{name:<26} {path:>10} {q:>14.2}");
    };
    rowr("W4A16 + FWHT", "iu4*+deq", &f16ish(&xr), &w4r);
    rowr("W8A8 + FWHT", "iu8 wmma", &x8r, &w8r);
    rowr("W4A8 + FWHT", "mixed", &x8r, &w4r);
    rowr("W4A4 + FWHT", "iu4 wmma", &x4r, &w4r);

    println!("\nReads: W4A4 (no rot) shows the raw cost of 4-bit activations on");
    println!("outlier-heavy activations; W4A4+FWHT shows how close rotation pulls");
    println!("it back toward W4A16. iu4-wmma path = fused native int4 matmul.");
}
