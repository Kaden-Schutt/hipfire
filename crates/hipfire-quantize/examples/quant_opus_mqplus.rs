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
//!
//! ## ORDERING NOTE: SmoothQuant runs BEFORE the bit-tier split (whole-layer)
//!
//! `prep()` applies `smoothquant()` over the *entire* layer (one per-input-channel
//! scale `s[c] = max|W_:,c|^α / ...`, applied to ALL rows), then `rotate()`, and
//! every downstream scheme — including the salient/mixed-resolution tiering in
//! `quant_mixed_salient` ("Idea 4") — quantizes the already-smoothed weights.
//! There is NO per-tier or post-split smoothing.
//!
//! This has a real hazard for any mixed-precision / lower-bit-tier scheme (e.g.
//! sending the lowest-scoring 50% of weights to qtip2/int4): SmoothQuant's `W·s`
//! step *migrates activation magnitude into the weights*, keyed on **activation
//! magnitude** — a different axis from the **weight sensitivity** used to tier.
//! So a high-activation / low-sensitivity column gets its weights inflated AND
//! dropped into the low-bit bin — i.e. smoothing dumps difficulty onto exactly
//! the weights about to be crushed hardest. `quant_mixed_salient` keys its
//! upgrade on per-group amax *energy* computed AFTER smoothing, so it catches
//! columns smoothing inflated (a partial, accidental mitigation), but it is NOT
//! the protect-sensitive-first ordering that would avoid the problem.
//!
//! The clean fixes (NOT implemented here): (a) protect/permute the sensitive +
//! hot channels into a high-bit (bf16/Q8) bin FIRST, then smooth only the
//! remaining low tier; or (b) skip SmoothQuant entirely and use activation-
//! magnitude **permutation** to isolate the hot channels — handling the outliers
//! without ever migrating magnitude into the to-be-crushed weights. (b) is the
//! "permutation can replace SmoothQuant" idea; this harness does neither, so its
//! numbers reflect the smooth-first-global ordering only.

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
            if (st >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
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
fn smoothquant(
    x: &[f32],
    w: &[f32],
    b: usize,
    m: usize,
    k: usize,
    alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
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
    let s: Vec<f32> = (0..k)
        .map(|c| (xm[c].powf(alpha) / wm[c].powf(1.0 - alpha)).max(1e-6))
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

const CLIP_GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];

/// Affine (asymmetric) unsigned-int quant→dequant per group (MQ format).
/// clip shrinks the [min,max] range symmetrically; clip=1 is plain MQ4.
fn quant_affine(
    src: &[f32],
    rows: usize,
    k: usize,
    group: usize,
    bits: u32,
    search: bool,
) -> Vec<f32> {
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
fn quant_sym(
    src: &[f32],
    rows: usize,
    k: usize,
    group: usize,
    bits: u32,
    search: bool,
) -> Vec<f32> {
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

/// Idea 1 — non-uniform Lloyd codebook snapped to the int8 lattice.
/// Per group: 1-D k-means with 16 centroids, snap centroids to int8 (this is the
/// 16-entry codebook the IU8 WMMA reads via the 4-bit index), then assign each
/// weight to its nearest int8-snapped level and dequant. Models the "free"
/// non-uniform expansion: 4-bit index + per-group {16×int8 codebook, f32 level
/// scale}. A 16-entry int8 codebook can also be ASYMMETRIC (no zero-point), so
/// it partly subsumes idea 3's benefit for free.
fn quant_lloyd_int8(
    src: &[f32],
    rows: usize,
    k: usize,
    group: usize,
    n_levels: usize,
    iters: usize,
) -> Vec<f32> {
    let l = n_levels;
    let mut out = vec![0.0f32; src.len()];
    let mut vals = vec![0.0f32; group];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let n = g1 - g0;
            vals[..n].copy_from_slice(&src[r * k + g0..r * k + g1]);
            let mut amax = 1e-12f32;
            for &v in &vals[..n] {
                amax = amax.max(v.abs());
            }
            // init centroids uniformly over [-amax, amax]
            let mut cent: Vec<f32> = (0..l)
                .map(|i| -amax + 2.0 * amax * i as f32 / (l - 1).max(1) as f32)
                .collect();
            for _ in 0..iters {
                let mut sum = vec![0.0f64; l];
                let mut cnt = vec![0u32; l];
                for &v in &vals[..n] {
                    let mut bi = 0usize;
                    let mut bd = f32::INFINITY;
                    for (li, &c) in cent.iter().enumerate() {
                        let d = (v - c).abs();
                        if d < bd {
                            bd = d;
                            bi = li;
                        }
                    }
                    sum[bi] += v as f64;
                    cnt[bi] += 1;
                }
                for li in 0..l {
                    if cnt[li] > 0 {
                        cent[li] = (sum[li] / cnt[li] as f64) as f32;
                    }
                }
            }
            // snap centroids to int8 lattice → the actual codebook the WMMA sees
            let cmax = cent.iter().fold(1e-12f32, |a, &c| a.max(c.abs()));
            let lscale = cmax / 127.0;
            let cq: Vec<f32> = cent
                .iter()
                .map(|&c| (c / lscale).round().clamp(-127.0, 127.0) * lscale)
                .collect();
            for (j, &v) in vals[..n].iter().enumerate() {
                let mut bd = f32::INFINITY;
                let mut bv = 0.0f32;
                for &c in &cq {
                    let d = (v - c).abs();
                    if d < bd {
                        bd = d;
                        bv = c;
                    }
                }
                out[r * k + g0 + j] = bv;
            }
        }
    }
    out
}

/// Idea 4 — salient-group mixed resolution. Upgrade only the top `frac` of groups
/// (by amax energy) to the idea-1 Lloyd-int8 codebook; the rest keep cheap uniform
/// symmetric int4. Tests whether protecting just the salient groups recovers most
/// of idea-1's gain at a fraction of the LUT budget.
fn quant_mixed_salient(
    src: &[f32],
    rows: usize,
    k: usize,
    group: usize,
    frac: f32,
    iters: usize,
) -> (Vec<f32>, f32) {
    // per-group amax over all (row, group) cells
    let mut energies: Vec<f32> = Vec::new();
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut amax = 0.0f32;
            for c in g0..g1 {
                amax = amax.max(src[r * k + c].abs());
            }
            energies.push(amax);
        }
    }
    let mut sorted = energies.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let thr = sorted[((frac * sorted.len() as f32) as usize).min(sorted.len() - 1)];
    let uniform = quant_sym(src, rows, k, group, 4, true);
    let lloyd = quant_lloyd_int8(src, rows, k, group, 16, iters);
    let mut out = vec![0.0f32; src.len()];
    let groups_per_row = k.div_ceil(group);
    let mut upgraded = 0usize;
    for r in 0..rows {
        for (gi, g0) in (0..k).step_by(group).enumerate() {
            let g1 = (g0 + group).min(k);
            let e = energies[r * groups_per_row + gi];
            let use_lloyd = e >= thr;
            if use_lloyd {
                upgraded += 1;
            }
            for c in g0..g1 {
                out[r * k + c] = if use_lloyd {
                    lloyd[r * k + c]
                } else {
                    uniform[r * k + c]
                };
            }
        }
    }
    let actual_frac = upgraded as f32 / energies.len() as f32;
    (out, actual_frac)
}

/// Idea 2 — stretch the int4 levels across the int8 range (×factor), folding the
/// inverse 1/factor into the group scale. Computed INDEPENDENTLY of `quant_sym`
/// to empirically confirm the no-op: W_int8·(scale/f) = (f·W_int4)·(scale/f) =
/// W_int4·scale. Storage is still 16 codes; expansion cannot add reconstruction
/// levels, so a uniform stretch is provably SQNR-invariant. Saturation at ±127
/// is the only way it could differ (f·7 ≤ 127 ⇒ f ≤ 18, safe here).
fn quant_sym_int8stretch(
    src: &[f32],
    rows: usize,
    k: usize,
    group: usize,
    factor: f32,
) -> Vec<f32> {
    let qmax = 7.0f32;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut amax = 1e-12f32;
            for c in g0..g1 {
                amax = amax.max(src[r * k + c].abs());
            }
            // same clip-search as quant_sym for a fair baseline match
            let (mut bc, mut be) = (1.0f32, f32::INFINITY);
            for &cl in &CLIP_GRID {
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
            let lscale = scale / factor; // per-group level scale after stretch
            for c in g0..g1 {
                let q4 = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                let i8 = (factor * q4).clamp(-127.0, 127.0); // stretched int8 code the WMMA reads
                out[r * k + c] = i8 * lscale;
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

/// Generic power-of-2 FWHT block (g64/g128/g256), 1/sqrt(n) normalized.
fn fwht_block(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    let n = x.len();
    for i in 0..n {
        x[i] *= s1[i];
    }
    let mut st = 1;
    while st < n {
        let mut i = 0;
        while i < n {
            for j in 0..st {
                let a = x[i + j];
                let bb = x[i + j + st];
                x[i + j] = a + bb;
                x[i + j + st] = a - bb;
            }
            i += st * 2;
        }
        st <<= 1;
    }
    let norm = 1.0 / (n as f32).sqrt();
    for i in 0..n {
        x[i] *= norm * s2[i];
    }
}

/// FWHT-rotate each `group`-sized segment in place (g64 = finer than the g256 default).
fn rotate_g(m: &mut [f32], rows: usize, k: usize, group: usize, s1: &[f32], s2: &[f32]) {
    let mut buf = vec![0.0f32; group];
    for r in 0..rows {
        for seg in 0..(k / group) {
            let base = r * k + seg * group;
            buf.copy_from_slice(&m[base..base + group]);
            fwht_block(&mut buf, s1, s2);
            m[base..base + group].copy_from_slice(&buf);
        }
    }
}

/// Permute the K columns of a row-major [rows×k] matrix (GEMM-invariant when the
/// same perm is applied to both operands). `perm[i]` = source column at slot i.
fn permute_cols(src: &[f32], rows: usize, k: usize, perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for i in 0..k {
            out[r * k + i] = src[r * k + perm[i]];
        }
    }
    out
}

/// Output-level SQNR comparing a precomputed y[b*m] against the f32 reference.
/// (For methods that add an output correction term, not just a dequant-W.)
fn y_sqnr(y: &[f32], yref: &[f32]) -> f64 {
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for i in 0..yref.len() {
        let r = yref[i] as f64;
        s += r * r;
        let d = r - y[i] as f64;
        n += d * d;
    }
    10.0 * (s / n.max(1e-30)).log10()
}

/// Per-channel outlier mask: the top `frac` of K channels by max-abs over the batch.
fn top_channels(x: &[f32], b: usize, k: usize, frac: f32) -> Vec<bool> {
    let mut amax = vec![0.0f32; k];
    for r in 0..b {
        for c in 0..k {
            amax[c] = amax[c].max(x[r * k + c].abs());
        }
    }
    let mut sorted = amax.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let thr = sorted[((frac * k as f32) as usize).min(k - 1)];
    amax.iter().map(|&v| v >= thr).collect()
}

/// Method 6 — symmetric int4 with sequential error-feedback (noise-shaping) per
/// group: carry each rounding residual forward to the next weight. A cheap proxy
/// for GPTQ/OBS (full off-diagonal Hessian feedback = `ldlq::oq4_ldlq_pack`).
fn quant_sym_ef(src: &[f32], rows: usize, k: usize, group: usize) -> Vec<f32> {
    let qmax = 7.0f32;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for g0 in (0..k).step_by(group) {
            let g1 = (g0 + group).min(k);
            let mut amax = 1e-12f32;
            for c in g0..g1 {
                amax = amax.max(src[r * k + c].abs());
            }
            let (mut bc, mut be) = (1.0f32, f32::INFINITY);
            for &cl in &CLIP_GRID {
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
            let mut carry = 0.0f32;
            for c in g0..g1 {
                let v = src[r * k + c] + carry;
                let q = (v / scale).round().clamp(-qmax, qmax);
                let dq = q * scale;
                carry = v - dq;
                out[r * k + c] = dq;
            }
        }
    }
    out
}

/// Method 7 — top-`rank` SVD of a residual matrix (m×k, row-major) via power
/// iteration + deflation. Returns (U[m*rank] with σ folded in, V[k*rank]) so
/// `U·Vᵀ` is the rank-`rank` approximation of the residual.
fn lowrank_resid(
    r0: &[f32],
    m: usize,
    k: usize,
    rank: usize,
    iters: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut r = r0.to_vec();
    let mut uo = vec![0.0f32; m * rank];
    let mut vo = vec![0.0f32; k * rank];
    let mut u = vec![0.0f32; m];
    let mut v = vec![0.0f32; k];
    for j in 0..rank {
        let mut s = (j as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
        for vi in v.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            *vi = (s as f32 / 2_147_483_648.0) - 0.5;
        }
        let mut sigma = 0.0f32;
        for _ in 0..iters {
            for (mi, uu) in u.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += r[mi * k + ki] * v[ki];
                }
                *uu = acc;
            }
            let un = u.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-20);
            for uu in u.iter_mut() {
                *uu /= un;
            }
            for (ki, vv) in v.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for mi in 0..m {
                    acc += r[mi * k + ki] * u[mi];
                }
                *vv = acc;
            }
            sigma = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-20);
            for vv in v.iter_mut() {
                *vv /= sigma;
            }
        }
        for mi in 0..m {
            uo[mi * rank + j] = u[mi] * sigma;
        }
        for ki in 0..k {
            vo[ki * rank + j] = v[ki];
        }
        for mi in 0..m {
            for ki in 0..k {
                r[mi * k + ki] -= sigma * u[mi] * v[ki];
            }
        }
    }
    (uo, vo)
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
    // ORDERING: SmoothQuant here is WHOLE-LAYER and runs BEFORE any bit-tier
    // split — every scheme below (incl. quant_mixed_salient) quantizes these
    // already-smoothed weights. See the module-level "ORDERING NOTE" for the
    // hazard this creates for low-bit tiers (smoothing migrates activation
    // magnitude into weights that may then be crushed to qtip2/int4).
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
    println!(
        "{:<14} {:<8} {:<7} {:>6} {:>9} {:>10}",
        "scheme", "W", "A", "bits/w", "compute", "SQNR dB"
    );
    println!("{}", "-".repeat(62));

    // MQ4 as shipped: affine u4 g256, A8 int8 g128, FWHT, no smooth, no clip.
    {
        let (xf, wf) = prep(false);
        let wq = quant_affine(&wf, m, k, 256, 4, false);
        let xq = quant_affine(&xf, b, k, 128, 8, false);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "MQ4", "affine4", "int8", 4.25, "iu8", q
        );
    }
    // MQ+ : affine u4 g256 + clip + SmoothQuant, A8 int8 g128 + clip.
    {
        let (xf, wf) = prep(true);
        let wq = quant_affine(&wf, m, k, 256, 4, true);
        let xq = quant_affine(&xf, b, k, 128, 8, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "MQ+", "affine4", "int8", 4.25, "iu8", q
        );
    }
    // Opus Quant : symmetric s4 g128 + clip + SmoothQuant, A4 int4 g32 + clip.
    {
        let (xf, wf) = prep(true);
        let wq = quant_sym(&wf, m, k, 128, 4, true);
        let xq = quant_sym(&xf, b, k, 32, 4, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "Opus Quant", "sym4", "int4", 4.13, "iu4", q
        );
    }
    // Opus Plus (Opus-A8) : symmetric s4 + int8 activations (the symmetric MQ+ analog).
    {
        let (xf, wf) = prep(true);
        let wq = quant_sym(&wf, m, k, 128, 4, true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "Opus+ (A8)", "sym4", "int8", 4.13, "iu8", q
        );
    }

    println!("\n--- expansion-time processing (ideas 1/3/4 on the Opus+ iu8 path) ---");
    // Idea 1 — non-uniform Lloyd codebook on the int8 lattice (per g256).
    {
        let (xf, wf) = prep(true);
        let wq = quant_lloyd_int8(&wf, m, k, 256, 16, 12);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "1: Opus+Lloyd", "cb4", "int8", 4.56, "iu8", q
        );
    }
    // Idea 3 — affine-u4 weights via zero-point→activation-sum correction (iu8).
    {
        let (xf, wf) = prep(true);
        let wq = quant_affine(&wf, m, k, 256, 4, true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "3: Opus+affine", "affine4", "int8", 4.25, "iu8", q
        );
    }
    // Idea 4 — salient-group mixed resolution (Lloyd codebook on top-X% groups).
    {
        let (xf, wf) = prep(true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        for frac in [0.10f32, 0.25, 0.50] {
            let (wq, af) = quant_mixed_salient(&wf, m, k, 256, frac, 12);
            let q = out_sqnr(&xq, &wq, &yref, m, k, b);
            let bits = 4.06 + 0.5 * af; // uniform 4.06 + 0.5 b/w codebook on upgraded groups
            println!(
                "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
                format!("4: salient {:.0}%", af * 100.0),
                "mix4",
                "int8",
                bits,
                "iu8",
                q
            );
        }
    }
    // Idea 2a — int8-range stretch (×8). Expect bit-identical to sym4 g256 (no-op).
    {
        let (xf, wf) = prep(true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let base = quant_sym(&wf, m, k, 256, 4, true); // g256 reference (no stretch)
        let qb = out_sqnr(&xq, &base, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "2-ref: g256", "sym4", "int8", 4.06, "iu8", qb
        );
        let wq = quant_sym_int8stretch(&wf, m, k, 256, 8.0);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "2a: stretch×8", "sym4↑i8", "int8", 4.06, "iu8", q
        );
    }
    // Idea 2b — finer WEIGHT-scale granularity (the real magnitude lever): g128/g64.
    {
        let (xf, wf) = prep(true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        for (g, bw) in [(128usize, 4.13f32), (64, 4.25)] {
            let wq = quant_sym(&wf, m, k, g, 4, true);
            let q = out_sqnr(&xq, &wq, &yref, m, k, b);
            println!(
                "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
                format!("2b: sym4 g{g}"),
                "sym4",
                "int8",
                bw,
                "iu8",
                q
            );
        }
    }

    println!("\n--- methods 5–8 (activation-side + weight refinements) ---");
    // Method 5 — activation outlier decomposition: top-frac channels exact (fp16),
    // bulk int8. Outliers split in the UNROTATED domain (where they're sparse).
    {
        for frac in [0.005f32, 0.01, 0.02] {
            let mask = top_channels(&x, b, k, frac);
            let nout = mask.iter().filter(|&&v| v).count();
            let mut xb = x.clone();
            for r in 0..b {
                for c in 0..k {
                    if mask[c] {
                        xb[r * k + c] = 0.0;
                    }
                }
            }
            let (mut xbs, mut ws) = smoothquant(&xb, &w, b, m, k, 0.5);
            rotate(&mut ws, m, k, &s1, &s2);
            rotate(&mut xbs, b, k, &s1, &s2);
            let wq = quant_sym(&ws, m, k, 128, 4, true);
            let xq = quant_sym(&xbs, b, k, 32, 8, true);
            let mut y = vec![0.0f32; b * m];
            for bi in 0..b {
                for mi in 0..m {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        acc += xq[bi * k + ki] * wq[mi * k + ki];
                    }
                    // outlier columns: exact unrotated contribution
                    for ki in 0..k {
                        if mask[ki] {
                            acc += x[bi * k + ki] * w[mi * k + ki];
                        }
                    }
                    y[bi * m + mi] = acc;
                }
            }
            let q = y_sqnr(&y, &yref);
            println!(
                "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
                format!("5: outl {nout}ch"),
                "sym4",
                "i8+f16",
                4.13,
                "iu8+f16",
                q
            );
        }
    }
    // Method 6 — error-feedback (noise-shaping) weight rounding (GPTQ proxy).
    {
        let (xf, wf) = prep(true);
        let wq = quant_sym_ef(&wf, m, k, 128);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let q = out_sqnr(&xq, &wq, &yref, m, k, b);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "6: EF round", "sym4-ef", "int8", 4.13, "iu8", q
        );
    }
    // Method 7 — low-rank fp16 residual correction: y = iu8(Q4·X) + U(Vᵀ·X).
    {
        let (xf, wf) = prep(true);
        let wq_d = quant_sym(&wf, m, k, 128, 4, true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let resid: Vec<f32> = (0..m * k).map(|i| wf[i] - wq_d[i]).collect();
        for rank in [8usize, 16] {
            let (u, v) = lowrank_resid(&resid, m, k, rank, 8);
            let mut y = vec![0.0f32; b * m];
            for bi in 0..b {
                let mut t = vec![0.0f32; rank];
                for ki in 0..k {
                    let xv = xf[bi * k + ki];
                    for j in 0..rank {
                        t[j] += v[ki * rank + j] * xv;
                    }
                }
                for mi in 0..m {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        acc += xq[bi * k + ki] * wq_d[mi * k + ki];
                    }
                    let mut corr = 0.0f32;
                    for j in 0..rank {
                        corr += u[mi * rank + j] * t[j];
                    }
                    y[bi * m + mi] = acc + corr;
                }
            }
            let bits = 4.06 + (rank * (m + k) * 16) as f32 / (m * k) as f32;
            let q = y_sqnr(&y, &yref);
            println!(
                "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
                format!("7: lowrank r{rank}"),
                "sym4+lr",
                "int8",
                bits,
                "iu8+f16",
                q
            );
        }
    }
    // Method 8 — per-output-row bias correction (free epilogue constant).
    {
        let (xf, wf) = prep(true);
        let wq_d = quant_sym(&wf, m, k, 128, 4, true);
        let xq = quant_sym(&xf, b, k, 32, 8, true);
        let mut mu = vec![0.0f32; k];
        for bi in 0..b {
            for ki in 0..k {
                mu[ki] += xf[bi * k + ki];
            }
        }
        for ki in 0..k {
            mu[ki] /= b as f32;
        }
        let cbias: Vec<f32> = (0..m)
            .map(|mi| {
                (0..k)
                    .map(|ki| (wf[mi * k + ki] - wq_d[mi * k + ki]) * mu[ki])
                    .sum()
            })
            .collect();
        let mut y = vec![0.0f32; b * m];
        for bi in 0..b {
            for mi in 0..m {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += xq[bi * k + ki] * wq_d[mi * k + ki];
                }
                y[bi * m + mi] = acc + cbias[mi];
            }
        }
        let q = y_sqnr(&y, &yref);
        println!(
            "{:<14} {:<8} {:<7} {:>6.2} {:>9} {:>10.2}",
            "8: bias-corr", "sym4+bc", "int8", 4.06, "iu8", q
        );
    }

    println!("\n--- method 9: permute + g64-rotate + outlier-groups (combine 5 + 2b [+3]) ---");
    {
        let s1_64 = signs(42, 64);
        let s2_64 = signs(1042, 64);
        let mut amax = vec![0.0f32; k];
        for r in 0..b {
            for c in 0..k {
                amax[c] = amax[c].max(x[r * k + c].abs());
            }
        }
        let mut perm_sorted: Vec<usize> = (0..k).collect();
        perm_sorted.sort_by(|&a, &c| amax[c].partial_cmp(&amax[a]).unwrap()); // desc
        let ident: Vec<usize> = (0..k).collect();
        let cut = 64usize; // first g64 group (highest act) → fp16 activations
        for smooth in [true, false] {
            let (xs, ws) = if smooth {
                smoothquant(&x, &w, b, m, k, 0.5)
            } else {
                (x.clone(), w.clone())
            };
            let _ = &ident;
            for (label, perm) in [("actmag", &perm_sorted)] {
                let xp = permute_cols(&xs, b, k, perm);
                let wp = permute_cols(&ws, m, k, perm);
                let mut xr = xp.clone();
                rotate_g(&mut xr, b, k, 64, &s1_64, &s2_64);
                let mut wr = wp.clone();
                rotate_g(&mut wr, m, k, 64, &s1_64, &s2_64);
                let wq = quant_sym(&wr, m, k, 64, 4, true);
                let wqa = quant_affine(&wr, m, k, 64, 4, true); // idea-3 weights
                let wq8 = quant_sym(&wr, m, k, 64, 8, true); // Q8 outlier weights
                let xq8 = quant_sym(&xr, b, k, 64, 8, true);
                for (wlabel, wqd) in [("sym4", &wq), ("aff4", &wqa)] {
                    // outlier-group precision:
                    //   f16  = W16A16 (exact)            — needs fp16 sidestream
                    //   q8   = Q8A16 (int8 w, fp16 act)  — mixed GEMM
                    //   q8a8 = Q8A8  (int8 w, int8 act)  — SAME iu8 path as bulk
                    for (omode, ow, oxs, obits) in [
                        ("f16", &wr, &xr, 16.0f32),
                        ("q8", &wq8, &xr, 8.0),
                        ("q8a8", &wq8, &xq8, 8.0),
                    ] {
                        let mut y = vec![0.0f32; b * m];
                        for bi in 0..b {
                            for mi in 0..m {
                                let mut acc = 0.0f32;
                                for ki in 0..k {
                                    let (wv, xv) = if ki < cut {
                                        (ow[mi * k + ki], oxs[bi * k + ki]) // outlier
                                    } else {
                                        (wqd[mi * k + ki], xq8[bi * k + ki]) // bulk: int4 × int8
                                    };
                                    acc += wv * xv;
                                }
                                y[bi * m + mi] = acc;
                            }
                        }
                        let q = y_sqnr(&y, &yref);
                        let sm = if smooth { "sm" } else { "no" };
                        let frac = cut as f32 / k as f32;
                        let bits = 4.0 + frac * (obits - 4.0);
                        let _ = label;
                        println!(
                            "{:<26} {:<5} {:>6.2} {:>9} {:>10.2}",
                            format!("9 {sm}+actmag+{wlabel}+o{omode}"),
                            "g64",
                            bits,
                            "outl",
                            q
                        );
                    }
                }
            }
        }
    }

    println!("\n--- method 10: magnitude-tiered W8A8(top1grp) / W4A8(mid) / W4A4(bottom) ---");
    {
        let s1_64 = signs(42, 64);
        let s2_64 = signs(1042, 64);
        let mut amax = vec![0.0f32; k];
        for r in 0..b {
            for c in 0..k {
                amax[c] = amax[c].max(x[r * k + c].abs());
            }
        }
        let mut perm: Vec<usize> = (0..k).collect();
        perm.sort_by(|&a, &c| amax[c].partial_cmp(&amax[a]).unwrap()); // desc
        let (xs, ws) = smoothquant(&x, &w, b, m, k, 0.5);
        let xp = permute_cols(&xs, b, k, &perm);
        let wp = permute_cols(&ws, m, k, &perm);
        let mut xr = xp.clone();
        rotate_g(&mut xr, b, k, 64, &s1_64, &s2_64);
        let mut wr = wp.clone();
        rotate_g(&mut wr, m, k, 64, &s1_64, &s2_64);
        let wq4 = quant_affine(&wr, m, k, 64, 4, true); // bulk int4 (affine, idea 3)
        let wq8 = quant_sym(&wr, m, k, 64, 8, true); // outlier int8
        let xq8 = quant_sym(&xr, b, k, 64, 8, true); // int8 acts
        let xq4 = quant_sym(&xr, b, k, 32, 4, true); // int4 acts (Opus-Quant style g32)
        let cut_hi = 64usize; // top g64 group → W8A8
        for lo_frac in [0.0f32, 0.25, 0.5, 0.75, 0.9] {
            let cut_lo = ((lo_frac * k as f32) as usize / 64) * 64; // g64-aligned bottom
            let lo_start = k - cut_lo;
            let mut y = vec![0.0f32; b * m];
            for bi in 0..b {
                for mi in 0..m {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        let (wv, xv) = if ki < cut_hi {
                            (wq8[mi * k + ki], xq8[bi * k + ki]) // W8A8 (top)
                        } else if ki >= lo_start {
                            (wq4[mi * k + ki], xq4[bi * k + ki]) // W4A4 (bottom, iu4)
                        } else {
                            (wq4[mi * k + ki], xq8[bi * k + ki]) // W4A8 (mid)
                        };
                        acc += wv * xv;
                    }
                    y[bi * m + mi] = acc;
                }
            }
            let q = y_sqnr(&y, &yref);
            let iu4_pct = cut_lo as f32 / k as f32 * 100.0;
            println!(
                "{:<24} {:>7} {:>10.2}",
                format!("10 W4A4-bottom {:.0}%", lo_frac * 100.0),
                format!("{iu4_pct:.0}%iu4"),
                q
            );
        }
    }

    println!("\n--- method 11: bottom-25% weight codec @ A8 (affine4 / qtip4=L16 / qtip2=L4) ---");
    {
        let s1_64 = signs(42, 64);
        let s2_64 = signs(1042, 64);
        let mut amax = vec![0.0f32; k];
        for r in 0..b {
            for c in 0..k {
                amax[c] = amax[c].max(x[r * k + c].abs());
            }
        }
        let mut perm: Vec<usize> = (0..k).collect();
        perm.sort_by(|&a, &c| amax[c].partial_cmp(&amax[a]).unwrap()); // desc
        let (xs, ws) = smoothquant(&x, &w, b, m, k, 0.5);
        let xp = permute_cols(&xs, b, k, &perm);
        let wp = permute_cols(&ws, m, k, &perm);
        let mut xr = xp.clone();
        rotate_g(&mut xr, b, k, 64, &s1_64, &s2_64);
        let mut wr = wp.clone();
        rotate_g(&mut wr, m, k, 64, &s1_64, &s2_64);
        let wq4 = quant_affine(&wr, m, k, 64, 4, true);
        let wq8 = quant_sym(&wr, m, k, 64, 8, true);
        let w_l16 = quant_lloyd_int8(&wr, m, k, 64, 16, 12); // qtip4 proxy (Lloyd-16, int8 cb)
        let w_l4 = quant_lloyd_int8(&wr, m, k, 64, 4, 12); // qtip2 proxy (Lloyd-4, int8 cb)
        let xq8 = quant_sym(&xr, b, k, 64, 8, true);
        let cut_hi = 64usize;
        let cut_lo = ((0.25 * k as f32) as usize / 64) * 64;
        let lo_start = k - cut_lo;
        let frac_hi = cut_hi as f32 / k as f32;
        let frac_lo = cut_lo as f32 / k as f32;
        for (lab, wbot, bb) in [
            ("affine4", &wq4, 4.0f32),
            ("qtip4", &w_l16, 4.0),
            ("qtip2", &w_l4, 2.0),
        ] {
            let mut y = vec![0.0f32; b * m];
            for bi in 0..b {
                for mi in 0..m {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        let wv = if ki < cut_hi {
                            wq8[mi * k + ki] // W8A8 top
                        } else if ki >= lo_start {
                            wbot[mi * k + ki] // bottom-25% codec @ A8
                        } else {
                            wq4[mi * k + ki] // W4A8 mid (affine)
                        };
                        acc += wv * xq8[bi * k + ki]; // A8 everywhere (no iu4)
                    }
                    y[bi * m + mi] = acc;
                }
            }
            let q = y_sqnr(&y, &yref);
            let avg = frac_hi * 8.0 + (1.0 - frac_hi - frac_lo) * 4.0 + frac_lo * bb;
            println!(
                "{:<20} {:>7.2} {:>10.2}",
                format!("11 bot25%={lab}"),
                avg,
                q
            );
        }
    }

    // Diagnostic: WEIGHT-reconstruction SQNR (dequant W vs W), isolating the
    // expansion-time weight quality from the activation noise floor. Same rotated
    // + smoothed weights all schemes see.
    let w_sqnr = |q: &[f32], wf: &[f32]| -> f64 {
        let (mut s, mut n) = (0.0f64, 0.0f64);
        for i in 0..wf.len() {
            let r = wf[i] as f64;
            s += r * r;
            let d = r - q[i] as f64;
            n += d * d;
        }
        10.0 * (s / n.max(1e-30)).log10()
    };
    println!("\n--- WEIGHT-recon SQNR (isolates expansion quality from act floor) ---");
    {
        let (_xf, wf) = prep(true);
        let u = quant_sym(&wf, m, k, 256, 4, true);
        let l = quant_lloyd_int8(&wf, m, k, 256, 16, 12);
        let af = quant_affine(&wf, m, k, 256, 4, true);
        println!("  uniform sym4 (Opus+ base): {:>7.2} dB", w_sqnr(&u, &wf));
        println!("  Lloyd cb4 (idea 1):        {:>7.2} dB", w_sqnr(&l, &wf));
        println!("  affine4 (idea 3):          {:>7.2} dB", w_sqnr(&af, &wf));
    }

    println!("\nMQ4→MQ+ : same iu8 kernel, offline quant + runtime act-rescale only.");
    println!("Opus Quant : new fused-iu4 path (W4A4), max prefill throughput.");
    println!("Ideas 1/3/4 : all run the SAME iu8 WMMA; only the int4→int8 expansion differs.");
}
