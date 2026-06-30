// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! First-principles quantization-scheme exploration for the gfx1103 platform.
//!
//! gfx1103 is a bandwidth-bound RDNA3 UMA APU with native WMMA/dot for f16,
//! bf16, iu8 and iu4 (no fp8). For decode (GEMV) the dominant cost is reading
//! weights from shared system DRAM, so bytes/weight is the lever. This probe
//! measures reconstruction quality (SQNR) per effective-bit for a range of
//! uniform/affine/rotated/non-uniform schemes on controlled weight
//! distributions, so we can pick the best quality-per-byte that still maps to
//! the platform's native integer matmul.
//!
//! Pure CPU, deterministic, no model files. Companion GPU bandwidth probe:
//! `rdna-compute --example bench_gemv_dtype_bw`.
//!
//!   cargo run -p hipfire-quantize --example quant_explore [rows] [k]

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};

// ───────────────────────── weight distributions ─────────────────────────

fn lcg_gauss(seed: u32, n: usize) -> Vec<f32> {
    // Box-Muller from an LCG uniform stream.
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

/// Gaussian bulk with a fraction of large outliers (realistic transformer
/// weights: most weights small, a few very large).
fn gauss_outliers(seed: u32, n: usize, frac: f32, gain: f32) -> Vec<f32> {
    let mut w = lcg_gauss(seed, n);
    let mut s = seed ^ 0xa5a5;
    for v in w.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        if (s as f32 / 2_147_483_648.0) < frac {
            *v *= gain;
        }
    }
    w
}

/// Per-channel (whole-column) outliers — outliers concentrated in a few input
/// channels, as commonly seen in LLM activations/weights.
fn channel_outliers(seed: u32, rows: usize, k: usize, n_hot: usize, gain: f32) -> Vec<f32> {
    let mut w = lcg_gauss(seed, rows * k);
    let mut s = seed ^ 0x1234_5678;
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

// ───────────────────────── quant schemes ─────────────────────────
// Each returns the f32 reconstruction (same length as input) and the effective
// bits-per-weight including per-group scale/zero-point overhead.

struct Recon {
    rec: Vec<f32>,
    bits_per_weight: f64,
}

/// Symmetric uniform: per group, scale = absmax/qmax, q in [-qmax, qmax].
/// Maps directly to native signed iu8/iu4 WMMA (W·scale applied after).
fn quant_sym(w: &[f32], group: usize, bits: u32) -> Recon {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32; // 127 (int8), 7 (int4)
    let mut rec = vec![0.0f32; w.len()];
    for (gi, chunk) in w.chunks(group).enumerate() {
        let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
        let scale = amax / qmax;
        let base = gi * group;
        for (j, &v) in chunk.iter().enumerate() {
            let q = (v / scale).round().clamp(-qmax, qmax);
            rec[base + j] = q * scale;
        }
    }
    Recon {
        rec,
        bits_per_weight: bits as f64 + 16.0 / group as f64, // f16 scale per group
    }
}

/// Affine/asymmetric uniform: per group, scale=(max-min)/(2^bits-1), zp=min.
/// Better for skewed groups but needs a zero-point correction in int matmul.
fn quant_affine(w: &[f32], group: usize, bits: u32) -> Recon {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut rec = vec![0.0f32; w.len()];
    for (gi, chunk) in w.chunks(group).enumerate() {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in chunk {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        let scale = ((mx - mn) / levels).max(1e-12);
        let base = gi * group;
        for (j, &v) in chunk.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, levels);
            rec[base + j] = q * scale + mn;
        }
    }
    Recon {
        rec,
        bits_per_weight: bits as f64 + 32.0 / group as f64, // f16 scale + f16 zp
    }
}

/// Symmetric uniform on FWHT-rotated weights (incoherence processing). The
/// rotation spreads outliers so one group scale fits the bulk; inverse rotation
/// on dequant. Group is fixed at 256 (the FWHT block).
fn quant_sym_fwht(w: &[f32], bits: u32, s1: &[f32], s2: &[f32]) -> Recon {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut rec = vec![0.0f32; w.len()];
    for (gi, chunk) in w.chunks(256).enumerate() {
        if chunk.len() < 256 {
            break;
        }
        let mut buf = [0.0f32; 256];
        buf.copy_from_slice(chunk);
        cpu_fwht_256(&mut buf, s1, s2); // forward rotate
        let amax = buf.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
        let scale = amax / qmax;
        for v in buf.iter_mut() {
            let q = (*v / scale).round().clamp(-qmax, qmax);
            *v = q * scale;
        }
        cpu_fwht_256(&mut buf, s2, s1); // inverse rotate
        rec[gi * 256..gi * 256 + 256].copy_from_slice(&buf);
    }
    Recon {
        rec,
        bits_per_weight: bits as f64 + 16.0 / 256.0,
    }
}

/// NF4: QLoRA-style non-uniform 4-bit, 16 levels placed at quantiles of a
/// unit normal. Per-group absmax scale. Best for Gaussian data, but the matmul
/// path is dequant→f16 (NOT native int4 WMMA) since levels are non-uniform.
const NF4_LEVELS: [f32; 16] = [
    -1.0, -0.6961928, -0.5250730, -0.3949175, -0.2844444, -0.1847203, -0.0911271, 0.0, 0.0795803,
    0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0,
];

fn quant_nf4(w: &[f32], group: usize) -> Recon {
    let mut rec = vec![0.0f32; w.len()];
    for (gi, chunk) in w.chunks(group).enumerate() {
        let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
        let base = gi * group;
        for (j, &v) in chunk.iter().enumerate() {
            let t = v / amax; // in [-1, 1]
                              // nearest NF4 level
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for (li, &lv) in NF4_LEVELS.iter().enumerate() {
                let d = (t - lv).abs();
                if d < bd {
                    bd = d;
                    best = li;
                }
            }
            rec[base + j] = NF4_LEVELS[best] * amax;
        }
    }
    Recon {
        rec,
        bits_per_weight: 4.0 + 16.0 / group as f64,
    }
}

// ───────────────────────── metrics ─────────────────────────

fn sqnr_db(orig: &[f32], rec: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&o, &r) in orig.iter().zip(rec) {
        sig += (o as f64) * (o as f64);
        let d = o as f64 - r as f64;
        noise += d * d;
    }
    if noise <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / noise).log10()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let rows: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let n = rows * k;
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);

    let dists: Vec<(&str, Vec<f32>)> = vec![
        ("gaussian", lcg_gauss(1, n)),
        ("gauss+1%x8 outliers", gauss_outliers(2, n, 0.01, 8.0)),
        ("gauss+0.1%x20 outliers", gauss_outliers(3, n, 0.001, 20.0)),
        (
            "channel outliers (16 cols x12)",
            channel_outliers(4, rows, k, 16, 12.0),
        ),
    ];

    println!("quant_explore  rows={rows} k={k}  (SQNR dB, higher=better)\n");
    println!("{:<34} {:>10} {:>9}", "scheme", "bits/w", "note");
    println!("{}", "-".repeat(80));

    // (label, recon-fn closure result, maps-to-native-int-matmul?)
    type Scheme = (&'static str, Box<dyn Fn(&[f32]) -> Recon>, &'static str);
    let s1c = s1.clone();
    let s2c = s2.clone();
    let schemes: Vec<Scheme> = vec![
        (
            "int8 sym g128",
            Box::new(|w: &[f32]| quant_sym(w, 128, 8)),
            "native iu8",
        ),
        (
            "int8 sym per-row(k)",
            Box::new(move |w: &[f32]| quant_sym(w, k, 8)),
            "native iu8",
        ),
        (
            "int4 sym per-row(k)",
            Box::new(move |w: &[f32]| quant_sym(w, k, 4)),
            "native iu4",
        ),
        (
            "int4 sym g128",
            Box::new(|w: &[f32]| quant_sym(w, 128, 4)),
            "native iu4",
        ),
        (
            "int4 sym g32",
            Box::new(|w: &[f32]| quant_sym(w, 32, 4)),
            "native iu4",
        ),
        (
            "int4 affine g128",
            Box::new(|w: &[f32]| quant_affine(w, 128, 4)),
            "int4+zp corr",
        ),
        (
            "int4 sym+FWHT g256",
            Box::new(move |w: &[f32]| quant_sym_fwht(w, 4, &s1c, &s2c)),
            "iu4 (rot act)",
        ),
        (
            "nf4 g128 (non-uniform)",
            Box::new(|w: &[f32]| quant_nf4(w, 128)),
            "dequant->f16",
        ),
    ];

    for (dname, w) in &dists {
        println!("\n### distribution: {dname}");
        for (label, f, note) in &schemes {
            let r = f(w);
            let q = sqnr_db(w, &r.rec);
            println!(
                "{:<34} {:>10.3} {:>9}  SQNR={:>7.2} dB",
                label, r.bits_per_weight, note, q
            );
        }
    }

    println!("\nNotes:");
    println!("  * native iu8/iu4 = weight stored as symmetric int, runs on the");
    println!("    platform's native integer WMMA (W4A4/W8A8) or cheap dequant.");
    println!("  * affine needs a per-group zero-point correction term in int matmul.");
    println!("  * nf4 is non-uniform -> must dequant to f16 (f16 WMMA), no int path,");
    println!("    but still 4-bit storage (same bandwidth win on this UMA platform).");
}
