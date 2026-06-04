// SPDX-License-Identifier: see LICENSE
// hipfire — see LICENSE
// Copyright (c) 2026 Kaden Schutt
//
//! Small CPU-side utilities shared across the qwen35 arch crate.

/// CPU argmax over a logits slice, returning the index of the maximum as `u32`.
///
/// Uses a strict `>` comparison so the FIRST maximal index wins on ties and
/// NaNs are skipped (`x > bv` is false for NaN, so a NaN never displaces the
/// running best). This mirrors the GPU `argmax.hip` reduction's tie/NaN
/// convention — keep them in lockstep or greedy decode can diverge between the
/// CPU verify path and the GPU sampler. Collapses three byte-identical copies
/// (mtp_probe / speculative / mtp_compose).
#[inline]
pub fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_max_index() {
        assert_eq!(argmax_u32(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax_u32(&[3.0, 2.0, 1.0]), 0);
        assert_eq!(argmax_u32(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn first_wins_on_tie() {
        // Strict `>` => the first occurrence of the max wins.
        assert_eq!(argmax_u32(&[1.0, 1.0, 1.0]), 0);
        assert_eq!(argmax_u32(&[0.0, 5.0, 5.0]), 1);
    }

    #[test]
    fn nan_is_skipped() {
        // NaN must never become the argmax (x > bv is false for NaN).
        assert_eq!(argmax_u32(&[f32::NAN, 2.0, 1.0]), 1);
        assert_eq!(argmax_u32(&[1.0, f32::NAN, 3.0]), 2);
    }

    #[test]
    fn empty_returns_zero() {
        assert_eq!(argmax_u32(&[]), 0);
    }
}
