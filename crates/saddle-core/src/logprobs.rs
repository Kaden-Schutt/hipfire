// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Top-K log-probabilities from a host logits slice.
//!
//! Placed in `saddle-core` because it is arithmetic over `&[f32]` with no GPU,
//! no tokenizer and no architecture knowledge. That keeps it at layer 4, below
//! every arch crate, so any generate path can call it without a new dependency
//! edge — `scripts/check-layering.py` enforces the direction.
//!
//! ## Why this is cheap where it is used
//!
//! `hipfire-arch-deepseek4::sampling::sample_token` takes `logits: &[f32]`, so
//! the dense decode path already has the full distribution resident on the host
//! at sampling time. Computing logprobs there costs one pass for the max, one
//! for the sum, and a bounded selection — no extra device-to-host copy. A prior
//! optimisation removed a per-token D2H round-trip from that path and this must
//! not reintroduce one.
//!
//! ## Numerics
//!
//! `log_softmax(x)_i = x_i - (m + ln Σ exp(x_j - m))`, with `m = max(x)`. The
//! shift is not optional: `exp` of an unshifted logit overflows to `inf` for
//! values above ~88 in f32, and real lm_head outputs reach that. Accumulation is
//! f64 because a 250k-entry vocabulary sums enough small terms for f32 rounding
//! to move the last decimal place of every reported logprob.

/// One candidate token and its log-probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

/// Top-`k` tokens by logit, descending, with log-softmax probabilities.
///
/// `k` is clamped to the vocabulary size. `k == 0` yields an empty vector.
/// Non-finite logits are skipped for selection — a `NaN` from a corrupt buffer
/// must not silently become the argmax, and returning it as a "most likely
/// token" would be worse than omitting it.
pub fn top_k_logprobs(logits: &[f32], k: usize) -> Vec<TokenLogprob> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let lse = log_sum_exp(logits);
    if !lse.is_finite() {
        return Vec::new();
    }

    let k = k.min(logits.len());
    // Partial selection: a full sort of a 250k vocabulary to report 5 entries is
    // the kind of avoidable work that shows up as a per-token cost.
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.retain(|&i| logits[i as usize].is_finite());
    if idx.is_empty() {
        return Vec::new();
    }
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    idx.into_iter()
        .map(|i| TokenLogprob {
            token_id: i,
            logprob: (f64::from(logits[i as usize]) - lse) as f32,
        })
        .collect()
}

/// Log-probability of one specific token — the sampled one, which need not be
/// in the top-K when temperature or top-p is in play.
pub fn logprob_of(logits: &[f32], token_id: u32) -> Option<f32> {
    let v = *logits.get(token_id as usize)?;
    if !v.is_finite() {
        return None;
    }
    let lse = log_sum_exp(logits);
    lse.is_finite().then(|| (f64::from(v) - lse) as f32)
}

/// `m + ln Σ exp(x_i - m)` in f64. Returns non-finite if every logit is.
fn log_sum_exp(logits: &[f32]) -> f64 {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v.is_finite() && v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return f64::NAN;
    }
    let m = f64::from(max);
    let sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| (f64::from(v) - m).exp())
        .sum();
    if sum <= 0.0 {
        return f64::NAN;
    }
    m + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_sum_to_one() {
        let logits = [0.5f32, -1.25, 3.0, 2.0, -0.75];
        let all = top_k_logprobs(&logits, logits.len());
        let total: f64 = all.iter().map(|t| f64::from(t.logprob).exp()).sum();
        assert!((total - 1.0).abs() < 1e-6, "sum was {total}");
    }

    #[test]
    fn ordered_descending_and_truncated() {
        let logits = [1.0f32, 5.0, 3.0, 4.0, 2.0];
        let top = top_k_logprobs(&logits, 3);
        assert_eq!(top.iter().map(|t| t.token_id).collect::<Vec<_>>(), vec![1, 3, 2]);
        assert!(top[0].logprob > top[1].logprob && top[1].logprob > top[2].logprob);
    }

    #[test]
    fn large_logits_do_not_overflow() {
        // exp(200) is inf in f32; without the max-shift every logprob is NaN.
        let logits = [200.0f32, 199.0, 198.0];
        let top = top_k_logprobs(&logits, 3);
        assert_eq!(top.len(), 3);
        assert!(top.iter().all(|t| t.logprob.is_finite()), "{top:?}");
        assert!(top[0].logprob < 0.0 && top[0].logprob > -1.0, "{:?}", top[0]);
    }

    #[test]
    fn uniform_distribution_gives_ln_one_over_n() {
        let logits = [1.0f32; 8];
        let top = top_k_logprobs(&logits, 1);
        let expected = (1.0f64 / 8.0).ln() as f32;
        assert!((top[0].logprob - expected).abs() < 1e-6, "{:?}", top[0]);
    }

    #[test]
    fn nan_is_never_selected_as_most_likely() {
        let logits = [1.0f32, f32::NAN, 2.0];
        let top = top_k_logprobs(&logits, 3);
        assert!(top.iter().all(|t| t.token_id != 1), "NaN token selected: {top:?}");
        assert_eq!(top[0].token_id, 2);
    }

    #[test]
    fn k_larger_than_vocab_is_clamped_and_k_zero_is_empty() {
        let logits = [1.0f32, 2.0];
        assert_eq!(top_k_logprobs(&logits, 99).len(), 2);
        assert!(top_k_logprobs(&logits, 0).is_empty());
        assert!(top_k_logprobs(&[], 5).is_empty());
    }

    #[test]
    fn logprob_of_matches_the_top_k_entry() {
        let logits = [0.25f32, 4.0, -2.0, 1.5];
        let top = top_k_logprobs(&logits, 4);
        for t in &top {
            let direct = logprob_of(&logits, t.token_id).expect("finite");
            assert!((direct - t.logprob).abs() < 1e-6, "{t:?} vs {direct}");
        }
        assert!(logprob_of(&logits, 99).is_none(), "out of range must be None");
    }

    #[test]
    fn ties_break_on_token_id_so_output_is_deterministic() {
        let logits = [2.0f32, 2.0, 2.0];
        let a = top_k_logprobs(&logits, 2);
        let b = top_k_logprobs(&logits, 2);
        assert_eq!(a.iter().map(|t| t.token_id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(
            a.iter().map(|t| t.token_id).collect::<Vec<_>>(),
            b.iter().map(|t| t.token_id).collect::<Vec<_>>()
        );
    }
}
