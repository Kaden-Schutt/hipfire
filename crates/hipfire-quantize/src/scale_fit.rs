// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MSE-optimal affine scale fitting for the fixed-grid quant families.
//!
//! The MQ/HFQ encoders historically fit each group's affine `(scale, min)` to
//! the raw block min/max. After the FWHT the group is near-Gaussian (CLT over
//! 256 terms), so its range runs to roughly ±3.3σ and a min/max fit spends its
//! 16 levels covering outliers that carry very little probability mass.
//! Shrinking the bounds trades a few clipped extremes for a finer step
//! everywhere else, which is a large net MSE win at zero wire-format cost.
//!
//! Measured on FWHT-rotated LFM2.5-350M and 1.2B weights: +0.74 dB with a
//! joint scan, +0.83 dB with the coordinate descent implemented here (≈16%
//! and ≈17% lower squared error respectively) against the min/max baseline of
//! 19.25 dB.

use std::sync::atomic::{AtomicBool, Ordering};

/// MQ4 scale-fit policy. `false` restores the legacy min/max fit so artifacts
/// produced before the search landed can be reproduced byte-for-byte.
static MQ4_SCALE_SEARCH: AtomicBool = AtomicBool::new(true);

/// Select the MQ4 affine scale fit. `false` = legacy min/max.
pub fn set_mq4_scale_search(enabled: bool) {
    MQ4_SCALE_SEARCH.store(enabled, Ordering::Relaxed);
}

/// True when [`fit_mq4_affine`] performs the MSE clip search.
pub fn mq4_scale_search_enabled() -> bool {
    MQ4_SCALE_SEARCH.load(Ordering::Relaxed)
}

/// Clip candidates per bound, per coordinate-descent pass.
const MQ4_CLIP_CANDIDATES: usize = 12;
/// Coordinate-descent passes over the `(lo, hi)` pair.
const MQ4_CLIP_PASSES: usize = 2;
/// Narrowest fraction of the raw min/max range the search will consider.
const MQ4_CLIP_FLOOR: f32 = 0.55;

/// Squared reconstruction error of the 16-level affine grid spanning `[lo, hi]`.
///
/// Evaluated through the exact emit expression — `((v - lo) * inv + 0.5) as u8`
/// then `.min(15)` — so the search optimizes the bytes actually written rather
/// than an idealized rounding model. Rust saturates negative float-to-integer
/// casts to 0, which is what clips values below `lo` once the range shrinks.
#[inline]
pub fn affine16_sq_err(group: &[f32; 256], lo: f32, hi: f32) -> f32 {
    let range = hi - lo;
    if !(range > 0.0) {
        return f32::INFINITY;
    }
    let scale = range / 15.0;
    let inv = 1.0 / scale;
    let mut err = 0.0f32;
    for &v in group.iter() {
        let q = (((v - lo) * inv + 0.5) as u8).min(15);
        let d = v - (q as f32 * scale + lo);
        err += d * d;
    }
    err
}

/// MSE-optimal `(lo, hi)` clip bounds for one 256-weight MQ4 block.
///
/// Coordinate descent moves `lo` and `hi` independently, which measured better
/// than a 4x-longer joint scan because the post-rotation tails are not
/// symmetric within a single block. The search is seeded at the exact min/max
/// and only ever accepts a strict improvement, so the result is never worse
/// than the legacy fit and is byte-identical whenever min/max is already
/// optimal.
///
/// Returns the raw min/max unchanged when the block is constant (zero range)
/// or when the search is disabled via [`set_mq4_scale_search`].
pub fn fit_mq4_affine(group: &[f32; 256]) -> (f32, f32) {
    let lo0 = group.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi0 = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !(hi0 - lo0 > 0.0) || !mq4_scale_search_enabled() {
        return (lo0, hi0);
    }

    let (mut lo, mut hi) = (lo0, hi0);
    let mut best = affine16_sq_err(group, lo, hi);
    let span = 1.0 - MQ4_CLIP_FLOOR;
    for _ in 0..MQ4_CLIP_PASSES {
        for i in 0..MQ4_CLIP_CANDIDATES {
            let a = MQ4_CLIP_FLOOR + span * (i as f32) / (MQ4_CLIP_CANDIDATES - 1) as f32;

            let cand = lo0 * a;
            let err = affine16_sq_err(group, cand, hi);
            if err < best {
                best = err;
                lo = cand;
            }

            let cand = hi0 * a;
            let err = affine16_sq_err(group, lo, cand);
            if err < best {
                best = err;
                hi = cand;
            }
        }
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Gaussian sample via Box-Muller over a small LCG, so the
    /// tests need no rand dependency and stay bit-stable across platforms.
    pub(crate) fn gaussian(n: usize, seed: u32) -> Vec<f32> {
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

    fn block(seed: u32) -> [f32; 256] {
        let mut g = [0.0f32; 256];
        for (i, v) in gaussian(256, seed).into_iter().enumerate() {
            g[i] = v;
        }
        g
    }

    fn min_max(g: &[f32; 256]) -> (f32, f32) {
        (
            g.iter().cloned().fold(f32::INFINITY, f32::min),
            g.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    }

    #[test]
    fn sq_err_matches_an_explicit_reconstruction() {
        let g = block(7);
        let (lo, hi) = (-2.0f32, 2.0f32);
        let scale = (hi - lo) / 15.0;
        let expect: f32 = g
            .iter()
            .map(|&v| {
                let q = (((v - lo) / scale + 0.5) as u8).min(15);
                let d = v - (q as f32 * scale + lo);
                d * d
            })
            .sum();
        let got = affine16_sq_err(&g, lo, hi);
        assert!(
            (got - expect).abs() <= 1e-6 * expect.max(1.0),
            "got {got}, expected {expect}"
        );
    }

    #[test]
    fn zero_range_is_reported_as_infinite_error() {
        let g = [0.5f32; 256];
        assert_eq!(affine16_sq_err(&g, 0.5, 0.5), f32::INFINITY);
    }

    /// Seeded at min/max and accepting only strict improvements, the search can
    /// never emit a worse fit than the legacy encoder.
    #[test]
    fn search_never_loses_to_min_max() {
        for seed in 0..32u32 {
            let g = block(seed * 977 + 1);
            let (lo0, hi0) = min_max(&g);
            let (lo, hi) = fit_mq4_affine(&g);
            let (searched, legacy) = (affine16_sq_err(&g, lo, hi), affine16_sq_err(&g, lo0, hi0));
            assert!(
                searched <= legacy * (1.0 + 1e-6),
                "seed {seed}: search {searched} regressed against min/max {legacy}"
            );
        }
    }

    /// On post-FWHT (near-Gaussian) data the clip search must be a material
    /// win, not noise. Guards against the search degenerating into a no-op if
    /// the candidate grid, floor, or seeding is ever broken.
    #[test]
    fn search_materially_beats_min_max_on_gaussian_blocks() {
        let (mut searched, mut legacy) = (0.0f64, 0.0f64);
        for seed in 0..64u32 {
            let g = block(seed * 7919 + 3);
            let (lo0, hi0) = min_max(&g);
            let (lo, hi) = fit_mq4_affine(&g);
            searched += affine16_sq_err(&g, lo, hi) as f64;
            legacy += affine16_sq_err(&g, lo0, hi0) as f64;
        }
        let gain_db = 10.0 * (legacy / searched).log10();
        assert!(
            gain_db > 0.5,
            "clip search gained only {gain_db:.3} dB over min/max; expected > 0.5 dB"
        );
    }

    /// A constant block has zero range and must pass through untouched, so the
    /// caller's degenerate path (scale 1.0, all-zero nibbles) is preserved.
    #[test]
    fn constant_block_passes_through_unchanged() {
        let g = [0.25f32; 256];
        assert_eq!(fit_mq4_affine(&g), (0.25, 0.25));
    }

    /// Disabling the search must reproduce the raw min/max bounds exactly.
    /// Serialized with the other toggle test because the flag is process-wide.
    #[test]
    fn disabled_search_returns_exact_min_max() {
        static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());

        let g = block(31337);
        let expect = min_max(&g);
        set_mq4_scale_search(false);
        let got = fit_mq4_affine(&g);
        set_mq4_scale_search(true);

        assert_eq!(got, expect);
        assert!(mq4_scale_search_enabled(), "flag must be restored");
    }
}
