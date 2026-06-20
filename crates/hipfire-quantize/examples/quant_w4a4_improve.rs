// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! How far can W4A4 quality be pushed on gfx1103? Stacks iu4-preserving quant
//! techniques on the W4A4 baseline and measures GEMM output SQNR. All schemes
//! except the last keep both operands symmetric int4 → still run on the fused
//! v_wmma_i32_16x16x16_iu4 path.
//!
//! Levers: clip-search scale (MSE-optimal vs absmax), activation group size,
//! rotation block size (FWHT-256 vs full-K Hadamard), SmoothQuant per-channel
//! migration, and (ceiling, NOT pure-iu4) outlier-channel mixed precision.
//!
//!   cargo run -p hipfire-quantize --example quant_w4a4_improve [M] [K] [B]

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

fn hot_channels(seed: u32, k: usize, n_hot: usize) -> Vec<bool> {
    let mut s = seed ^ 0x9e37_79b9;
    let mut hot = vec![false; k];
    for _ in 0..n_hot {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        hot[(s as usize) % k] = true;
    }
    hot
}

fn apply_channel_gain(mut w: Vec<f32>, rows: usize, k: usize, hot: &[bool], gain: f32) -> Vec<f32> {
    for r in 0..rows {
        for c in 0..k {
            if hot[c] {
                w[r * k + c] *= gain;
            }
        }
    }
    w
}

fn gen_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (st >> 16) & 1 == 1 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Generic randomized FWHT of power-of-two size n. signs1/signs2 length >= n.
fn fwht(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    let n = x.len();
    for i in 0..n {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let sc = 1.0 / (n as f32).sqrt();
    for i in 0..n {
        x[i] *= sc * signs2[i];
    }
}

/// Rotate every row's K dim in blocks of `block` (power of two, divides k).
fn rotate_rows(m: &mut [f32], rows: usize, k: usize, block: usize, s1: &[f32], s2: &[f32]) {
    let mut buf = vec![0.0f32; block];
    for r in 0..rows {
        for seg in 0..(k / block) {
            let base = r * k + seg * block;
            buf.copy_from_slice(&m[base..base + block]);
            fwht(&mut buf, s1, s2);
            m[base..base + block].copy_from_slice(&buf);
        }
    }
}

/// Symmetric int quant→dequant per group. clip in (0,1]: scale = clip*absmax/qmax.
/// clip<1 sacrifices the largest values for finer bulk resolution.
fn quant_sym(src: &[f32], rows: usize, k: usize, group: usize, bits: u32, clip: f32) -> Vec<f32> {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut amax = 1e-12f32;
            for c in g0..g1 {
                amax = amax.max(src[r * k + c].abs());
            }
            let scale = clip * amax / qmax;
            for c in g0..g1 {
                let q = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                out[r * k + c] = q * scale;
            }
        }
    }
    out
}

/// Per-group MSE-optimal clip-search symmetric quant (grid over clip factor).
fn quant_sym_clipsearch(src: &[f32], rows: usize, k: usize, group: usize, bits: u32) -> Vec<f32> {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let grid = [1.0f32, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut amax = 1e-12f32;
            for c in g0..g1 {
                amax = amax.max(src[r * k + c].abs());
            }
            let mut best_clip = 1.0f32;
            let mut best_err = f32::INFINITY;
            for &cl in &grid {
                let scale = cl * amax / qmax;
                let mut err = 0.0f32;
                for c in g0..g1 {
                    let q = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                    let d = src[r * k + c] - q * scale;
                    err += d * d;
                }
                if err < best_err {
                    best_err = err;
                    best_clip = cl;
                }
            }
            let scale = best_clip * amax / qmax;
            for c in g0..g1 {
                let q = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                out[r * k + c] = q * scale;
            }
        }
    }
    out
}

/// SmoothQuant per-channel migration: s_j = max_i|X[:,j]|^a / max_i|W[:,j]|^(1-a).
/// Returns (X/s, W*s); the product X'·W'ᵀ == X·Wᵀ exactly.
fn smoothquant(x: &[f32], w: &[f32], b: usize, m: usize, k: usize, alpha: f32) -> (Vec<f32>, Vec<f32>) {
    let mut xmax = vec![1e-9f32; k];
    let mut wmax = vec![1e-9f32; k];
    for r in 0..b {
        for c in 0..k {
            xmax[c] = xmax[c].max(x[r * k + c].abs());
        }
    }
    for r in 0..m {
        for c in 0..k {
            wmax[c] = wmax[c].max(w[r * k + c].abs());
        }
    }
    let s: Vec<f32> = (0..k)
        .map(|c| (xmax[c].powf(alpha) / wmax[c].powf(1.0 - alpha)).max(1e-6))
        .collect();
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
    assert!(k.is_power_of_two(), "K must be power-of-two for full-K Hadamard");

    let s1 = gen_signs(42, k);
    let s2 = gen_signs(1042, k);

    let whot = hot_channels(7, k, 8);
    let xhot = hot_channels(9, k, 16);
    let w = apply_channel_gain(lcg_gauss(1, m * k), m, k, &whot, 6.0);
    let x = apply_channel_gain(lcg_gauss(2, b * k), b, k, &xhot, 20.0);

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

    println!("quant_w4a4_improve  M={m} K={k} B={b}");
    println!("acts: 16 hot ch ×20 ; weights: 8 hot ch ×6\n");
    println!("{:<46} {:>9} {:>8}", "W4A4 scheme (all fused iu4 unless noted)", "SQNR dB", "Δ base");
    println!("{}", "-".repeat(66));

    let mut base = 0.0f64;
    let emit = |name: &str, xq: &[f32], wq: &[f32], set_base: bool, base: &mut f64| {
        let q = out_sqnr(xq, wq, &yref, m, k, b);
        let d = if *base != 0.0 { q - *base } else { 0.0 };
        if set_base {
            *base = q;
        }
        println!("{name:<46} {q:>9.2} {:>+8.2}", d);
    };

    // 0) baseline: FWHT-256 + absmax symmetric int4, A g128.
    {
        let mut wr = w.clone();
        let mut xr = x.clone();
        rotate_rows(&mut wr, m, k, 256, &s1, &s2);
        rotate_rows(&mut xr, b, k, 256, &s1, &s2);
        let wq = quant_sym(&wr, m, k, 128, 4, 1.0);
        let xq = quant_sym(&xr, b, k, 128, 4, 1.0);
        emit("0 baseline: FWHT256 + absmax, A=g128", &xq, &wq, true, &mut base);
    }
    // 1) + clip-search scale (W and A)
    {
        let mut wr = w.clone();
        let mut xr = x.clone();
        rotate_rows(&mut wr, m, k, 256, &s1, &s2);
        rotate_rows(&mut xr, b, k, 256, &s1, &s2);
        let wq = quant_sym_clipsearch(&wr, m, k, 128, 4);
        let xq = quant_sym_clipsearch(&xr, b, k, 128, 4);
        emit("1 + clip-search scale", &xq, &wq, false, &mut base);
    }
    // 2) + finer activation group (g32)
    {
        let mut wr = w.clone();
        let mut xr = x.clone();
        rotate_rows(&mut wr, m, k, 256, &s1, &s2);
        rotate_rows(&mut xr, b, k, 256, &s1, &s2);
        let wq = quant_sym_clipsearch(&wr, m, k, 128, 4);
        let xq = quant_sym_clipsearch(&xr, b, k, 32, 4);
        emit("2 + clip-search + A=g32", &xq, &wq, false, &mut base);
    }
    // 3) full-K Hadamard rotation instead of 256-block
    {
        let mut wr = w.clone();
        let mut xr = x.clone();
        rotate_rows(&mut wr, m, k, k, &s1, &s2);
        rotate_rows(&mut xr, b, k, k, &s1, &s2);
        let wq = quant_sym_clipsearch(&wr, m, k, 128, 4);
        let xq = quant_sym_clipsearch(&xr, b, k, 32, 4);
        emit("3 + full-K Hadamard (vs 256) + clip + A=g32", &xq, &wq, false, &mut base);
    }
    // 4) SmoothQuant migration THEN 256-block rotation + clip + A=g32
    {
        let (xs, ws) = smoothquant(&x, &w, b, m, k, 0.5);
        let mut wr = ws.clone();
        let mut xr = xs.clone();
        rotate_rows(&mut wr, m, k, 256, &s1, &s2);
        rotate_rows(&mut xr, b, k, 256, &s1, &s2);
        let wq = quant_sym_clipsearch(&wr, m, k, 128, 4);
        let xq = quant_sym_clipsearch(&xr, b, k, 32, 4);
        emit("4 + SmoothQuant α0.5 (then FWHT256 + clip + g32)", &xq, &wq, false, &mut base);
    }
    // 5) SmoothQuant + full-K Hadamard + clip + A=g32
    {
        let (xs, ws) = smoothquant(&x, &w, b, m, k, 0.5);
        let mut wr = ws.clone();
        let mut xr = xs.clone();
        rotate_rows(&mut wr, m, k, k, &s1, &s2);
        rotate_rows(&mut xr, b, k, k, &s1, &s2);
        let wq = quant_sym_clipsearch(&wr, m, k, 128, 4);
        let xq = quant_sym_clipsearch(&xr, b, k, 32, 4);
        emit("5 + SmoothQuant + full-K Hadamard + clip + g32", &xq, &wq, false, &mut base);
    }

    // Reference points (not pure iu4) — context, best recipe applied.
    println!();
    {
        // W4A8 (upcast iu8): the no-A4 comparison, same front-end as stage 5.
        let (xs, ws) = smoothquant(&x, &w, b, m, k, 0.5);
        let mut wr = ws.clone();
        let mut xr = xs.clone();
        rotate_rows(&mut wr, m, k, k, &s1, &s2);
        rotate_rows(&mut xr, b, k, k, &s1, &s2);
        let wq = quant_sym_clipsearch(&wr, m, k, 128, 4);
        let xq = quant_sym_clipsearch(&xr, b, k, 32, 8);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!("{:<48} {q:>9.2}  (ref: A8 upcast→iu8)", "  W4A8 + SmoothQuant + full-K + clip");
    }
    {
        // W8A8 (both 8-bit): the high-precision int ceiling, same front-end.
        let (xs, ws) = smoothquant(&x, &w, b, m, k, 0.5);
        let mut wr = ws.clone();
        let mut xr = xs.clone();
        rotate_rows(&mut wr, m, k, k, &s1, &s2);
        rotate_rows(&mut xr, b, k, k, &s1, &s2);
        let wq = quant_sym_clipsearch(&wr, m, k, 128, 8);
        let xq = quant_sym_clipsearch(&xr, b, k, 32, 8);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!("{:<48} {q:>9.2}  (ceiling: iu8 wmma)", "  W8A8 + SmoothQuant + full-K + clip");
    }
}
