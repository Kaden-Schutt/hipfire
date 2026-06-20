// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Head-to-head: MQ4 (as shipped) vs MQ+ vs Opus Quant on gfx1103.
//!
//! - **MQ4**: affine u4 weights (min+scale·q, range/15), FWHT-256, g256; compute
//!   is W4A8 via iu8 WMMA (weight upcast to int8, Q8_1 int8 activations). The
//!   production format today.
//! - **MQ+**: MQ4's affine format + SmoothQuant per-channel migration +
//!   clip-search scale. Same iu8 GEMM kernel; only the offline quant and a
//!   runtime activation rescale change. A free quality upgrade to MQ4.
//! - **Opus Quant**: symmetric s4 weights + clip-search + SmoothQuant, int4
//!   activations → fused iu4 WMMA (W4A4). New compute path; max throughput.
//!
//! Metric: GEMM output SQNR (dB) vs f32, on outlier-heavy activations.
//!
//!   cargo run -p hipfire-quantize --example quant_opus_mqplus [M K B]

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
fn hot(seed: u32, k: usize, n: usize) -> Vec<bool> {
    let mut s = seed ^ 0x9e37_79b9;
    let mut h = vec![false; k];
    for _ in 0..n {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        h[(s as usize) % k] = true;
    }
    h
}
fn gain(mut w: Vec<f32>, rows: usize, k: usize, h: &[bool], g: f32) -> Vec<f32> {
    for r in 0..rows {
        for c in 0..k {
            if h[c] {
                w[r * k + c] *= g;
            }
        }
    }
    w
}
fn signs(seed: u32, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (st >> 16) & 1 == 1 { 1.0 } else { -1.0 }
        })
        .collect()
}
fn fwht256(x: &mut [f32; 256], s1: &[f32], s2: &[f32]) {
    for i in 0..256 {
        x[i] *= s1[i];
    }
    let mut st = 1;
    while st < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..st {
                let a = x[i + j];
                let b = x[i + j + st];
                x[i + j] = a + b;
                x[i + j + st] = a - b;
            }
            i += st * 2;
        }
        st <<= 1;
    }
    for i in 0..256 {
        x[i] *= 0.0625 * s2[i];
    }
}
fn rotate(m: &mut [f32], rows: usize, k: usize, s1: &[f32], s2: &[f32]) {
    let mut buf = [0.0f32; 256];
    for r in 0..rows {
        for seg in 0..(k / 256) {
            let base = r * k + seg * 256;
            buf.copy_from_slice(&m[base..base + 256]);
            fwht256(&mut buf, s1, s2);
            m[base..base + 256].copy_from_slice(&buf);
        }
    }
}
fn smoothquant(x: &[f32], w: &[f32], b: usize, m: usize, k: usize, alpha: f32) -> (Vec<f32>, Vec<f32>) {
    let mut xm = vec![1e-9f32; k];
    let mut wm = vec![1e-9f32; k];
    for r in 0..b {
        for c in 0..k {
            xm[c] = xm[c].max(x[r * k + c].abs());
        }
    }
    for r in 0..m {
        for c in 0..k {
            wm[c] = wm[c].max(w[r * k + c].abs());
        }
    }
    let s: Vec<f32> = (0..k).map(|c| (xm[c].powf(alpha) / wm[c].powf(1.0 - alpha)).max(1e-6)).collect();
    let mut xo = x.to_vec();
    let mut wo = w.to_vec();
    for r in 0..b {
        for c in 0..k {
            xo[r * k + c] /= s[c];
        }
    }
    for r in 0..m {
        for c in 0..k {
            wo[r * k + c] *= s[c];
        }
    }
    (xo, wo)
}

const CLIP_GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];

/// Affine (asymmetric) unsigned-int quant→dequant per group (MQ format).
/// clip shrinks the [min,max] range symmetrically; clip=1 is plain MQ4.
fn quant_affine(src: &[f32], rows: usize, k: usize, group: usize, bits: u32, search: bool) -> Vec<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for c in g0..g1 {
                mn = mn.min(src[r * k + c]);
                mx = mx.max(src[r * k + c]);
            }
            let mid = 0.5 * (mn + mx);
            let half = 0.5 * (mx - mn);
            let grid: &[f32] = if search { &CLIP_GRID } else { &[1.0] };
            let (mut bc, mut be) = (1.0f32, f32::INFINITY);
            for &cl in grid {
                let lo = mid - cl * half;
                let scale = (2.0 * cl * half / levels).max(1e-12);
                let mut e = 0.0f32;
                for c in g0..g1 {
                    let q = ((src[r * k + c] - lo) / scale).round().clamp(0.0, levels);
                    let d = src[r * k + c] - (q * scale + lo);
                    e += d * d;
                }
                if e < be {
                    be = e;
                    bc = cl;
                }
            }
            let lo = mid - bc * half;
            let scale = (2.0 * bc * half / levels).max(1e-12);
            for c in g0..g1 {
                let q = ((src[r * k + c] - lo) / scale).round().clamp(0.0, levels);
                out[r * k + c] = q * scale + lo;
            }
        }
    }
    out
}

/// Symmetric signed-int quant→dequant per group (Opus format).
fn quant_sym(src: &[f32], rows: usize, k: usize, group: usize, bits: u32, search: bool) -> Vec<f32> {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut amax = 1e-12f32;
            for c in g0..g1 {
                amax = amax.max(src[r * k + c].abs());
            }
            let grid: &[f32] = if search { &CLIP_GRID } else { &[1.0] };
            let (mut bc, mut be) = (1.0f32, f32::INFINITY);
            for &cl in grid {
                let scale = cl * amax / qmax;
                let mut e = 0.0f32;
                for c in g0..g1 {
                    let q = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                    let d = src[r * k + c] - q * scale;
                    e += d * d;
                }
                if e < be {
                    be = e;
                    bc = cl;
                }
            }
            let scale = bc * amax / qmax;
            for c in g0..g1 {
                let q = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                out[r * k + c] = q * scale;
            }
        }
    }
    out
}

fn out_sqnr(x: &[f32], w: &[f32], yref: &[f32], m: usize, k: usize, b: usize) -> f64 {
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
    10.0 * (sig / noise.max(1e-30)).log10()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    assert_eq!(k % 256, 0);
    let s1 = signs(42, 256);
    let s2 = signs(1042, 256);

    let w = gain(lcg_gauss(1, m * k), m, k, &hot(7, k, 8), 6.0);
    let x = gain(lcg_gauss(2, b * k), b, k, &hot(9, k, 16), 20.0);

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

    // Helper: apply optional SmoothQuant, then rotation, returning rotated X,W.
    let prep = |smooth: bool| -> (Vec<f32>, Vec<f32>) {
        let (mut xf, mut wf) = if smooth {
            smoothquant(&x, &w, b, m, k, 0.5)
        } else {
            (x.clone(), w.clone())
        };
        rotate(&mut wf, m, k, &s1, &s2);
        rotate(&mut xf, b, k, &s1, &s2);
        (xf, wf)
    };

    println!("quant_opus_mqplus  M={m} K={k} B={b}  (output SQNR dB)\n");
    println!("{:<14} {:<8} {:<7} {:>6} {:>9} {:>10}", "scheme", "W", "A", "bits/w", "compute", "SQNR dB");
    println!("{}", "-".repeat(62));

    // MQ4 as shipped: affine u4 g256, A8 int8 g128, FWHT, no smooth, no clip.
    {
        let (xf, wf) = prep(false);
        let wq = quant_affine(&wf, m, k, 256, 4, false);
        let xq = quant_affine(&xf, b, k, 128, 8, false);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!("{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}", "MQ4", "affine4", "int8", 4.25, "iu8", q);
    }
    // MQ+ : affine u4 g256 + clip + SmoothQuant, A8 int8 g128 + clip.
    {
        let (xf, wf) = prep(true);
        let wq = quant_affine(&wf, m, k, 256, 4, true);
        let xq = quant_affine(&xf, b, k, 128, 8, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!("{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}", "MQ+", "affine4", "int8", 4.25, "iu8", q);
    }
    // Opus Quant : symmetric s4 g128 + clip + SmoothQuant, A4 int4 g32 + clip.
    {
        let (xf, wf) = prep(true);
        let wq = quant_sym(&wf, m, k, 128, 4, true);
        let xq = quant_sym(&xf, b, k, 32, 4, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!("{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}", "Opus Quant", "sym4", "int4", 4.13, "iu4", q);
    }
    // Opus-A8 : symmetric s4 + int8 activations (the symmetric MQ+ analog).
    {
        let (xf, wf) = prep(true);
        let wq = quant_sym(&wf, m, k, 128, 4, true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!("{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}", "Opus-A8", "sym4", "int8", 4.13, "iu8", q);
    }

    println!("\nMQ4→MQ+ : same iu8 kernel, offline quant + runtime act-rescale only.");
    println!("Opus Quant : new fused-iu4 path (W4A4), max prefill throughput.");
}
