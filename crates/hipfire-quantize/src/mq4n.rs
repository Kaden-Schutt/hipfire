// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ4N-G256 — candidate MQ4G256 successor. Encoder + CPU dequant reference.
//!
//! Same 136 B per 256 weights as MQ4G256, same FWHT-256 rotation, same nibble
//! packing. Three things change, all inside the 8-byte header:
//!
//! | | header | grid |
//! |---|---|---|
//! | MQ4G256 | `[f32 scale][f32 min]` | uniform 16-level, min/max fit |
//! | MQ4N    | `[f32 master][4 x u8 E4M3]` | Lloyd-Max Gaussian codebook |
//!
//! Rationale. After a 256-point FWHT each coordinate is a sum of 256 sign-
//! flipped weights, so by the CLT it is near-Gaussian. Three consequences:
//!
//! 1. The distribution is zero-mean and symmetric, so the f32 zero-point is
//!    nearly dead weight. Dropping it frees 4 B.
//! 2. Those 4 B buy four E4M3 sub-scales, one per 64 weights — 4x finer scale
//!    granularity at *identical* total bytes. (MFP4-E8 already spends its 0.25
//!    bpw of scale metadata this way; MQ4G256 spends it on one f32 pair.)
//! 3. The optimal 16-level codebook for a Gaussian is a mathematical constant,
//!    not calibration data. [`CODEBOOK`] is the Lloyd-Max solution, so it needs
//!    no imatrix, no corpus, and costs zero bytes on the wire.
//!
//! Measured on FWHT-rotated LFM2.5-350M/1.2B weights: −35% squared error
//! versus the MQ4G256 min/max fit, entirely data-free. See
//! `benchmarks/kernel-probes/mq4-codebook-decode-cost/` for the decode-cost
//! measurement (net faster on gfx1201).
//!
//! Byte layout, per 256 weights:
//! ```text
//!   [0..4)   f32  master        group scale
//!   [4..8)   u8x4 sub[j]        E4M3, one per 64 weights
//!   [8..136) u8x128             two 4-bit codes per byte, low nibble first
//! ```
//! Reconstruction: `w[i] = CODEBOOK[code[i]] * master * e4m3(sub[i / 64])`.
//!
//! NOTE: no `QuantType` discriminant is assigned yet and no GPU kernel consumes
//! this. It is a validated encoder plus reference decoder, not a shippable
//! format — wiring it up is a separate wire-format decision.

use crate::fp8::{e4m3_decode, e4m3_encode_roundup};

/// Weights per rotation/scale group.
pub const GROUP: usize = 256;
/// Weights sharing one E4M3 sub-scale.
pub const SUB: usize = 64;
/// Sub-scales per group.
pub const N_SUB: usize = GROUP / SUB;
/// Encoded bytes per group — identical to MQ4G256.
pub const GROUP_BYTES: usize = 136;

/// Lloyd-Max optimal 16-level reconstruction levels for a unit Gaussian,
/// normalized so `max|c| == 1`.
///
/// Solved on the Gaussian density itself (not on samples), then symmetrized,
/// so the constant is exact and reproducible rather than a fit to one model.
/// Verified against theory: distortion 0.009495 vs the published 0.009497
/// (20.225 dB vs 20.22 dB), and +0.96 dB over an optimally-stepped uniform
/// 16-level quantizer.
///
/// Ascending, which [`nearest_code`] relies on.
pub const CODEBOOK: [f32; 16] = [
    -1.00000000,
    -0.75716355,
    -0.59212931,
    -0.45972187,
    -0.34485254,
    -0.24034313,
    -0.14200753,
    -0.04698658,
    0.04698658,
    0.14200753,
    0.24034313,
    0.34485254,
    0.45972187,
    0.59212931,
    0.75716355,
    1.00000000,
];

/// Sub-scale search grid, as a fraction of the sub-block maximum. Values above
/// 1.0 are included because the codebook's outermost level sits at ±1: letting
/// the scale overshoot slightly can beat clipping when a sub-block has one
/// dominant outlier.
const ALPHA_LO: f32 = 0.50;
const ALPHA_HI: f32 = 1.15;
const ALPHA_STEPS: usize = 20;

/// Index of the nearest codebook entry.
///
/// `CODEBOOK` is ascending, so this is a midpoint scan; 16 entries and offline,
/// so a branchy loop is fine.
///
/// Exact midpoints round UP (toward the larger level). Any GPU decoder only
/// reads the emitted index, so this rule affects encoding alone — but it must
/// stay fixed, since flipping it would change emitted bytes for values landing
/// exactly on a boundary (`v == 0.0` is the common one: it is equidistant from
/// `CODEBOOK[7]` and `CODEBOOK[8]`, and resolves to 8).
#[inline]
pub fn nearest_code(v: f32) -> u8 {
    let mut idx = 0u8;
    for i in 1..16 {
        let mid = (CODEBOOK[i - 1] + CODEBOOK[i]) * 0.5;
        if v >= mid {
            idx = i as u8;
        } else {
            break;
        }
    }
    idx
}

/// Squared error of one sub-block reconstructed at `eff`.
#[inline]
fn sub_sq_err(vals: &[f32], eff: f32) -> f32 {
    if !(eff > 0.0) {
        return f32::INFINITY;
    }
    let inv = 1.0 / eff;
    let mut e = 0.0f32;
    for &v in vals {
        let d = v - CODEBOOK[nearest_code(v * inv) as usize] * eff;
        e += d * d;
    }
    e
}

/// Encode one already-rotated 256-weight group into `out` (136 B).
///
/// `master` is the group maximum magnitude; each sub-scale is then chosen
/// independently by an MSE search, which is strictly better than one shared
/// alpha because sub-blocks are independent once `master` is fixed.
pub fn encode_group(group: &[f32; GROUP], out: &mut [u8; GROUP_BYTES]) {
    let master = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    out[0..4].copy_from_slice(&master.to_le_bytes());

    if !(master > 0.0) {
        // All-zero group: every reconstruction is 0 regardless of the codes.
        out[4..GROUP_BYTES].fill(0);
        return;
    }
    let inv_master = 1.0 / master;

    let mut eff = [0.0f32; N_SUB];
    for s in 0..N_SUB {
        let vals = &group[s * SUB..(s + 1) * SUB];
        let submax = vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let (mut best_byte, mut best_err) = (0u8, f32::INFINITY);
        for a in 0..ALPHA_STEPS {
            let alpha =
                ALPHA_LO + (ALPHA_HI - ALPHA_LO) * (a as f32) / (ALPHA_STEPS - 1) as f32;
            let byte = e4m3_encode_roundup(submax * alpha * inv_master);
            let cand = master * e4m3_decode(byte);
            let err = sub_sq_err(vals, cand);
            if err < best_err {
                best_err = err;
                best_byte = byte;
            }
        }
        out[4 + s] = best_byte;
        eff[s] = master * e4m3_decode(best_byte);
    }

    for i in 0..GROUP / 2 {
        let (a, b) = (2 * i, 2 * i + 1);
        let ea = eff[a / SUB];
        let eb = eff[b / SUB];
        let lo = if ea > 0.0 { nearest_code(group[a] / ea) } else { 0 };
        let hi = if eb > 0.0 { nearest_code(group[b] / eb) } else { 0 };
        out[8 + i] = lo | (hi << 4);
    }
}

/// Reference decode of one 136-B group back to the rotated domain.
///
/// Mirrors exactly what a GPU kernel must compute; any kernel is required to
/// agree with this bit-for-bit.
pub fn decode_group(blk: &[u8; GROUP_BYTES], out: &mut [f32; GROUP]) {
    let master = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
    let mut eff = [0.0f32; N_SUB];
    for s in 0..N_SUB {
        eff[s] = master * e4m3_decode(blk[4 + s]);
    }
    for i in 0..GROUP / 2 {
        let byte = blk[8 + i];
        out[2 * i] = CODEBOOK[(byte & 0xF) as usize] * eff[(2 * i) / SUB];
        out[2 * i + 1] = CODEBOOK[(byte >> 4) as usize] * eff[(2 * i + 1) / SUB];
    }
}

/// Encode a flat weight slice, applying the FWHT per 256-group first.
///
/// `signs1`/`signs2` are the engine's rotation sign tables; the caller supplies
/// them so this stays identical to the MQ4G256 path. A trailing partial group
/// is zero-padded, matching `quantize_mq4g256`.
pub fn encode(f32_data: &[f32], fwht: impl Fn(&mut [f32]) + Sync) -> Vec<u8> {
    use rayon::prelude::*;
    let n = f32_data.len();
    let n_blocks = n.div_ceil(GROUP);
    let mut out = vec![0u8; n_blocks * GROUP_BYTES];
    out.par_chunks_mut(GROUP_BYTES)
        .enumerate()
        .for_each(|(b, chunk)| {
            let start = b * GROUP;
            let end = (start + GROUP).min(n);
            let mut g = [0.0f32; GROUP];
            g[..end - start].copy_from_slice(&f32_data[start..end]);
            fwht(&mut g);
            let arr: &mut [u8; GROUP_BYTES] = chunk.try_into().unwrap();
            encode_group(&g, arr);
        });
    out
}

/// Decode a whole buffer back to the rotated domain.
pub fn decode(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / GROUP_BYTES;
    let mut out = vec![0.0f32; n_blocks * GROUP];
    for b in 0..n_blocks {
        let blk: &[u8; GROUP_BYTES] = bytes[b * GROUP_BYTES..(b + 1) * GROUP_BYTES]
            .try_into()
            .unwrap();
        let dst: &mut [f32; GROUP] = (&mut out[b * GROUP..(b + 1) * GROUP])
            .try_into()
            .unwrap();
        decode_group(blk, dst);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16777216.0).clamp(1e-7, 1.0 - 1e-7)
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (next(), next());
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect()
    }

    fn block(seed: u32) -> [f32; GROUP] {
        let mut g = [0.0f32; GROUP];
        for (i, v) in gaussian(GROUP, seed).into_iter().enumerate() {
            g[i] = v;
        }
        g
    }

    /// Legacy MQ4G256 fit, for the quality comparison.
    fn legacy_err(g: &[f32; GROUP]) -> f64 {
        let lo = g.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = g.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = hi - lo;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv = if range > 0.0 { 1.0 / scale } else { 0.0 };
        g.iter()
            .map(|&v| {
                let q = (((v - lo) * inv + 0.5) as u8).min(15);
                let d = v - (q as f32 * scale + lo);
                (d as f64) * (d as f64)
            })
            .sum()
    }

    fn mq4n_err(g: &[f32; GROUP]) -> f64 {
        let mut enc = [0u8; GROUP_BYTES];
        encode_group(g, &mut enc);
        let mut dec = [0.0f32; GROUP];
        decode_group(&enc, &mut dec);
        g.iter()
            .zip(dec.iter())
            .map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64))
            .sum()
    }

    #[test]
    fn codebook_is_sorted_symmetric_and_unit_peak() {
        for i in 1..16 {
            assert!(CODEBOOK[i] > CODEBOOK[i - 1], "not ascending at {i}");
        }
        for i in 0..8 {
            assert!(
                (CODEBOOK[i] + CODEBOOK[15 - i]).abs() < 1e-7,
                "asymmetric at {i}"
            );
        }
        assert!((CODEBOOK[15] - 1.0).abs() < 1e-7);
        assert!((CODEBOOK[0] + 1.0).abs() < 1e-7);
    }

    #[test]
    fn nearest_code_is_the_true_argmin() {
        for step in 0..4001 {
            let v = -1.6 + 3.2 * (step as f32) / 4000.0;
            let got = nearest_code(v) as usize;
            // Compare by distance, not index: exact midpoints are genuine ties
            // and the documented rule is to round up, whereas `min_by` keeps
            // the first minimum. Both are optimal; only the distance matters.
            let best = (0..16)
                .map(|i| (CODEBOOK[i] - v).abs())
                .fold(f32::INFINITY, f32::min);
            assert!(
                (CODEBOOK[got] - v).abs() <= best + 1e-9,
                "v={v}: picked {got} (err {}) but best err is {best}",
                (CODEBOOK[got] - v).abs()
            );
        }
    }

    /// The documented tie rule: exact midpoints round up.
    #[test]
    fn exact_midpoints_round_up() {
        assert_eq!(nearest_code(0.0), 8, "zero is the canonical tie");
        for i in 1..16 {
            let mid = (CODEBOOK[i - 1] + CODEBOOK[i]) * 0.5;
            assert_eq!(nearest_code(mid), i as u8, "midpoint below index {i}");
        }
    }

    #[test]
    fn group_is_exactly_136_bytes_like_mq4g256() {
        assert_eq!(GROUP_BYTES, 136);
        let data = gaussian(GROUP * 5, 3);
        let out = encode(&data, |_| {});
        assert_eq!(out.len(), 5 * 136);
    }

    #[test]
    fn encode_decode_round_trips_through_the_reference() {
        for seed in 0..16u32 {
            let g = block(seed * 131 + 7);
            let mut enc = [0u8; GROUP_BYTES];
            encode_group(&g, &mut enc);
            let mut dec = [0.0f32; GROUP];
            decode_group(&enc, &mut dec);
            // Every output must be an exact codebook*scale product.
            let master = f32::from_le_bytes([enc[0], enc[1], enc[2], enc[3]]);
            for i in 0..GROUP {
                let eff = master * e4m3_decode(enc[4 + i / SUB]);
                let code = if i % 2 == 0 {
                    enc[8 + i / 2] & 0xF
                } else {
                    enc[8 + i / 2] >> 4
                };
                assert_eq!(dec[i].to_bits(), (CODEBOOK[code as usize] * eff).to_bits());
            }
        }
    }

    #[test]
    fn all_zero_group_decodes_to_all_zero() {
        let g = [0.0f32; GROUP];
        let mut enc = [0u8; GROUP_BYTES];
        encode_group(&g, &mut enc);
        let mut dec = [0.0f32; GROUP];
        decode_group(&enc, &mut dec);
        assert!(dec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn constant_nonzero_group_survives() {
        let g = [0.75f32; GROUP];
        let mut enc = [0u8; GROUP_BYTES];
        encode_group(&g, &mut enc);
        let mut dec = [0.0f32; GROUP];
        decode_group(&enc, &mut dec);
        // One level sits at exactly +1.0 * master, so a constant block is exact.
        for &v in dec.iter() {
            assert!((v - 0.75).abs() < 1e-6, "got {v}");
        }
    }

    /// The headline claim: materially lower error than the MQ4G256 min/max fit
    /// on post-FWHT (near-Gaussian) data, at identical bytes.
    #[test]
    fn beats_mq4g256_min_max_by_over_1_5_db() {
        let (mut new, mut old) = (0.0f64, 0.0f64);
        for seed in 0..64u32 {
            let g = block(seed * 7919 + 3);
            new += mq4n_err(&g);
            old += legacy_err(&g);
        }
        let gain_db = 10.0 * (old / new).log10();
        assert!(
            gain_db > 1.5,
            "MQ4N gained only {gain_db:.3} dB over MQ4G256 min/max; expected > 1.5 dB"
        );
    }

    /// Sub-scales must actually be exercised: a group whose four 64-blocks have
    /// wildly different dynamic range is exactly the case MQ4G256 handles badly
    /// and MQ4N is meant to fix.
    #[test]
    fn heterogeneous_sub_blocks_pick_distinct_scales() {
        let mut g = [0.0f32; GROUP];
        for (i, v) in gaussian(GROUP, 99).into_iter().enumerate() {
            g[i] = v * [1.0f32, 0.05, 8.0, 0.4][i / SUB];
        }
        let mut enc = [0u8; GROUP_BYTES];
        encode_group(&g, &mut enc);
        let subs = [enc[4], enc[5], enc[6], enc[7]];
        assert!(
            subs.iter().collect::<std::collections::HashSet<_>>().len() >= 3,
            "expected distinct sub-scales, got {subs:?}"
        );
        let gain_db = 10.0 * (legacy_err(&g) / mq4n_err(&g)).log10();
        assert!(
            gain_db > 6.0,
            "heterogeneous block gained only {gain_db:.2} dB; sub-scales not working"
        );
    }
}
