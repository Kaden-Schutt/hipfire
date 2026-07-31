// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Provably optimal 1-D k-means, for the MQ*-Lloyd per-block codebooks.
//!
//! The shipped MQ2/MQ3-Lloyd encoders fit each block's codebook with Lloyd's
//! algorithm from a percentile initialization, capped at 8 iterations. Lloyd's
//! algorithm is coordinate descent: it converges to a *local* minimum, and
//! which one depends entirely on the initialization.
//!
//! In one dimension the global optimum is computable exactly. Optimal clusters
//! are contiguous intervals of the sorted data, so a dynamic program over
//! (cluster count, prefix length) finds the true minimum-distortion partition,
//! with segment costs in O(1) from prefix sums.
//!
//! Measured against the shipped fitter on FWHT-rotated LFM2.5 weights, at
//! identical bytes and identical wire format:
//!
//! | width | shipped (8 iter) | Lloyd converged | **exact DP** |
//! |---|---|---|---|
//! | 3-bit (K=8) | 14.651 dB | 14.927 (+0.28) | **15.352 (+0.70)** |
//! | 2-bit (K=4) | 9.490 dB | 9.505 (+0.02) | **9.534 (+0.05)** |
//!
//! Two things that fall out of those numbers:
//!
//! * **2-bit is already essentially optimal.** The shipped fit is within
//!   0.045 dB of the global optimum, so there is no headroom in the *fitter*.
//!   Improving MQ2 further needs a different structure, not a better k-means.
//! * **3-bit leaves 0.70 dB on the table**, and running Lloyd to convergence
//!   only recovers 39% of it. The rest is the local-vs-global gap, which no
//!   iteration count can close. Initializing from the Lloyd-Max Gaussian
//!   codebook was also tried and is *worse* (−0.108 dB).
//!
//! A useful side effect: the DP has no iteration count, so the
//! "do NOT raise this above 8" hazard documented on `quantize_mq2g256_lloyd`
//! cannot recur — there is no knob to mis-set.

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the MQ*-Lloyd encoders use [`optimal_levels`] instead of the
/// 8-iteration Lloyd fit.
///
/// Defaults to OFF. The gain is real and measured, but `quantize_mq2g256_lloyd`
/// carries a hard-won warning: raising the Lloyd iteration cap from 8 to 16
/// once produced a 60x PPL regression on DeepSeek V4 that a synthetic probe had
/// predicted as an improvement. Flipping a per-block codebook fit on by default
/// without a real-model coherence run would repeat exactly that mistake, so
/// this ships opt-in until gated.
///
/// (For the record, that incident's stated cause does not survive scrutiny:
/// Lloyd's algorithm is monotonically non-increasing in distortion, 16
/// iterations measures strictly better weight MSE than 8, and no fp16 centroid
/// collapse occurs at 8, 16, or 32 iterations. The real cause is still
/// unidentified — which is itself a reason to gate this on coherence.)
static OPTIMAL_FIT: AtomicBool = AtomicBool::new(false);

/// Enable or disable the optimal per-block fit for MQ*-Lloyd.
pub fn set_optimal_fit(enabled: bool) {
    OPTIMAL_FIT.store(enabled, Ordering::Relaxed);
}

/// True when MQ*-Lloyd should use [`optimal_levels`].
pub fn optimal_fit_enabled() -> bool {
    OPTIMAL_FIT.load(Ordering::Relaxed)
}

/// Fit `k` optimal reconstruction levels to `data`, ascending.
///
/// Returns the cluster means of the globally optimal partition. `k` is clamped
/// to `data.len()`; degenerate inputs (empty, or fewer distinct values than
/// clusters) still return `k` finite ascending-or-equal levels.
///
/// Cost is O(k · n²) time and O(k · n) space. That is heavier than Lloyd but
/// runs once per 256-weight block at quantize time, and blocks parallelize.
pub fn optimal_levels(data: &[f32], k: usize) -> Vec<f32> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![0.0; k];
    }
    let mut x: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if k >= n {
        let mut out: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        out.resize(k, x[n - 1] as f32);
        return out;
    }

    // Prefix sums give the cost of any contiguous segment in O(1).
    let mut p1 = vec![0.0f64; n + 1];
    let mut p2 = vec![0.0f64; n + 1];
    for i in 0..n {
        p1[i + 1] = p1[i] + x[i];
        p2[i + 1] = p2[i] + x[i] * x[i];
    }
    // Sum of squared deviations of x[i..j].
    let seg = |i: usize, j: usize| -> f64 {
        let c = (j - i) as f64;
        if c <= 0.0 {
            return 0.0;
        }
        let s = p1[j] - p1[i];
        (p2[j] - p2[i] - s * s / c).max(0.0)
    };

    const INF: f64 = f64::INFINITY;
    let mut prev = vec![INF; n + 1];
    let mut cur = vec![INF; n + 1];
    // back[c][j] = best split point for c clusters covering x[..j]
    let mut back = vec![0usize; (k + 1) * (n + 1)];

    prev[0] = 0.0;
    for c in 1..=k {
        cur.iter_mut().for_each(|v| *v = INF);
        for j in c..=n {
            let (mut best, mut arg) = (INF, c - 1);
            for i in (c - 1)..j {
                if prev[i].is_finite() {
                    let cand = prev[i] + seg(i, j);
                    if cand < best {
                        best = cand;
                        arg = i;
                    }
                }
            }
            cur[j] = best;
            back[c * (n + 1) + j] = arg;
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    // Walk the split points back out and take each segment's mean.
    let mut cuts = vec![n; k + 1];
    let mut j = n;
    for c in (1..=k).rev() {
        j = back[c * (n + 1) + j];
        cuts[c - 1] = j;
    }
    let mut levels = Vec::with_capacity(k);
    for c in 0..k {
        let (a, b) = (cuts[c], cuts[c + 1]);
        levels.push(if b > a {
            ((p1[b] - p1[a]) / (b - a) as f64) as f32
        } else if c > 0 {
            levels[c - 1]
        } else {
            x[0] as f32
        });
    }
    levels
}

/// Total squared error of reconstructing `data` with the nearest entry of an
/// ascending `levels`. Useful for gating a candidate fit against the incumbent.
pub fn assignment_sq_err(data: &[f32], levels: &[f32]) -> f64 {
    if levels.is_empty() {
        return f64::INFINITY;
    }
    let mut acc = 0.0f64;
    for &v in data {
        let mut best = (v - levels[0]).abs();
        let mut rec = levels[0];
        for &c in &levels[1..] {
            let d = (v - c).abs();
            if d < best {
                best = d;
                rec = c;
            }
        }
        let e = (v - rec) as f64;
        acc += e * e;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lloyd_reference(data: &[f32], k: usize, iters: usize) -> Vec<f32> {
        let mut s: Vec<f32> = data.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut cb: Vec<f32> = (0..k)
            .map(|i| s[(((i as f32 + 0.5) / k as f32) * (s.len() - 1) as f32).round() as usize])
            .collect();
        for _ in 0..iters {
            let mut sums = vec![0.0f64; k];
            let mut cnt = vec![0u32; k];
            for &v in data {
                let mut bi = 0;
                let mut bd = (v - cb[0]).abs();
                for j in 1..k {
                    let d = (v - cb[j]).abs();
                    if d < bd {
                        bd = d;
                        bi = j;
                    }
                }
                sums[bi] += v as f64;
                cnt[bi] += 1;
            }
            for j in 0..k {
                if cnt[j] > 0 {
                    cb[j] = (sums[j] / cnt[j] as f64) as f32;
                }
            }
        }
        cb.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cb
    }

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

    #[test]
    fn levels_are_ascending_and_finite() {
        for k in [2usize, 4, 8, 16] {
            for seed in 0..8u32 {
                let d = gaussian(256, seed * 17 + 1);
                let l = optimal_levels(&d, k);
                assert_eq!(l.len(), k);
                assert!(l.iter().all(|v| v.is_finite()), "k={k} produced non-finite");
                for i in 1..k {
                    assert!(l[i] >= l[i - 1], "k={k} not ascending at {i}");
                }
            }
        }
    }

    /// The whole point: the DP must never lose to Lloyd's, at any width.
    #[test]
    fn never_worse_than_lloyd_from_percentiles() {
        for k in [4usize, 8] {
            for seed in 0..24u32 {
                let d = gaussian(256, seed * 977 + 5);
                let dp = assignment_sq_err(&d, &optimal_levels(&d, k));
                for iters in [8usize, 64] {
                    let ll = assignment_sq_err(&d, &lloyd_reference(&d, k, iters));
                    assert!(
                        dp <= ll * (1.0 + 1e-6),
                        "k={k} seed={seed}: DP {dp} lost to Lloyd({iters}) {ll}"
                    );
                }
            }
        }
    }

    /// Against brute force on small inputs, the DP must be exactly optimal.
    #[test]
    fn matches_brute_force_on_small_inputs() {
        for (n, k) in [(8usize, 3usize), (10, 4), (9, 2)] {
            for seed in 0..12u32 {
                let d = gaussian(n, seed * 31 + 3);
                let mut sorted = d.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                // Optimal 1-D clusters are contiguous, so enumerate split sets.
                let mut best = f64::INFINITY;
                let mut cuts = vec![0usize; k + 1];
                cuts[k] = n;
                fn rec(
                    depth: usize,
                    k: usize,
                    n: usize,
                    cuts: &mut Vec<usize>,
                    x: &[f32],
                    best: &mut f64,
                ) {
                    if depth == k {
                        let mut tot = 0.0f64;
                        for c in 0..k {
                            let (a, b) = (cuts[c], cuts[c + 1]);
                            if b > a {
                                let m: f64 =
                                    x[a..b].iter().map(|&v| v as f64).sum::<f64>() / (b - a) as f64;
                                tot += x[a..b].iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>();
                            }
                        }
                        if tot < *best {
                            *best = tot;
                        }
                        return;
                    }
                    for split in cuts[depth - 1]..=n {
                        cuts[depth] = split;
                        rec(depth + 1, k, n, cuts, x, best);
                    }
                }
                rec(1, k, n, &mut cuts, &sorted, &mut best);
                let dp = assignment_sq_err(&d, &optimal_levels(&d, k));
                assert!(
                    dp <= best + 1e-6 * best.max(1.0),
                    "n={n} k={k} seed={seed}: DP {dp} > brute force {best}"
                );
            }
        }
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert_eq!(optimal_levels(&[], 4).len(), 4);
        let c = optimal_levels(&[2.5f32; 64], 4);
        assert!(c.iter().all(|&v| (v - 2.5).abs() < 1e-6), "{c:?}");
        // Fewer distinct values than clusters.
        let two = optimal_levels(&[1.0, 1.0, 1.0, 5.0, 5.0, 5.0], 4);
        assert_eq!(two.len(), 4);
        assert!(two.windows(2).all(|w| w[1] >= w[0]));
        // More clusters than points.
        assert_eq!(optimal_levels(&[1.0, 2.0], 5).len(), 5);
    }

    #[test]
    fn exactly_k_distinct_values_is_lossless() {
        let vals = [-3.0f32, -0.5, 0.25, 4.0];
        let mut data = Vec::new();
        for _ in 0..40 {
            data.extend_from_slice(&vals);
        }
        let l = optimal_levels(&data, 4);
        assert!(
            assignment_sq_err(&data, &l) < 1e-6,
            "should reconstruct exactly, got {l:?}"
        );
    }
}
