// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure fp64 reduction + KLD scoring math.
//!
//! Faithful extraction of the two implementations that previously lived
//! (duplicated) in `build_kld_ref_hipfire::log_softmax_top_k_row` and
//! `eval_hipfire::score_position`. The reference reduction and the candidate
//! scoring now share this one module, so they cannot diverge.

use crate::refblock::RefBlock;

/// Log-partition `log Z = log Σ_i exp(logit_i)`, computed in fp64 with the
/// standard max-shift for numerical stability. The shift is folded back in, so
/// the result is the true (unshifted) log-sum-exp.
pub fn log_z(logits: &[f32]) -> f64 {
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_logit {
            max_logit = v;
        }
    }
    if !max_logit.is_finite() {
        // All-empty or all -inf: defined as the max (handles len()==0 → -inf).
        return max_logit as f64;
    }
    let mut sum_exp = 0.0f64;
    for &v in logits {
        sum_exp += ((v - max_logit) as f64).exp();
    }
    (max_logit as f64) + sum_exp.ln()
}

/// Top-K log-softmax reduction of a full logit row — the reference-side
/// representation written into a `.kldref`. Returns the K highest-logit token
/// ids (descending, ties broken by ascending id), their log-probabilities, and
/// the residual probability mass not captured by the top-K (`1 - Σ p_topk`,
/// clamped to ≥0).
#[derive(Debug, Clone, PartialEq)]
pub struct TopKReduction {
    pub indices: Vec<u32>,
    pub log_probs: Vec<f32>,
    pub residual_mass: f32,
}

pub fn top_k_log_softmax(logits: &[f32], top_k: usize) -> TopKReduction {
    let k = top_k.min(logits.len());
    let lz = log_z(logits);

    // Rank by logit descending, tie-break by id ascending (matches the
    // historical heap/sort order so byte-identical refs reproduce).
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then_with(|| a.cmp(&b))
    });

    let mut indices = vec![0u32; top_k];
    let mut log_probs = vec![f32::NEG_INFINITY; top_k];
    let mut top_p_sum = 0.0f64;
    for i in 0..k {
        let idx = order[i];
        let log_p = ((logits[idx as usize] as f64) - lz) as f32;
        indices[i] = idx;
        log_probs[i] = log_p;
        top_p_sum += (log_p as f64).exp();
    }
    let residual_mass = (1.0 - top_p_sum).max(0.0) as f32;
    TopKReduction {
        indices,
        log_probs,
        residual_mass,
    }
}

/// Per-position score: forward KLD `D(P_ref ‖ Q_cand)` plus the candidate NLL of
/// the actual next token.
///
/// **Wide accumulate, narrow store.** Both fields are computed via f64
/// accumulators (the partition sum and the K-term cross-entropy need the wide
/// accumulator — see [`log_z`]) but are returned as `f32`: a KLD of ~1e-3 needs
/// ~4 significant figures and f32 gives 7, so storing f64 here would only bloat
/// the high-volume per-position arrays (mean/p99 reductions, per-layer
/// divergence matrices) without adding information. f64 stays an accumulator
/// type, not a storage type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionScore {
    /// `Σ_{i∈topK(P_ref)} P_ref(i)·(log P_ref(i) − log Q_cand(i))` plus the
    /// residual cross-term. Clamped at 0 (Gibbs; tiny negatives are fp64 roundoff).
    pub kld: f32,
    /// `−log Q_cand(actual_next)`, or `None` if `actual_next` is out of range.
    pub nll: Option<f32>,
}

/// Score candidate logits against a reference block for one position.
///
/// Mirrors the historical `eval_hipfire::score_position` exactly: KLD over the
/// reference's top-K support with a single lumped residual cross-term, and NLL
/// from the candidate's log-softmax at `actual_next`. `log Z` of the candidate
/// is computed once here.
pub fn score_position(
    ref_block: &RefBlock,
    cand_logits: &[f32],
    actual_next: usize,
) -> PositionScore {
    let lz = log_z(cand_logits);

    let mut kld = 0.0f64;
    let mut sum_p_cand_at_ref_top = 0.0f64;
    for (j, &ref_idx) in ref_block.top_indices.iter().enumerate() {
        let ref_idx = ref_idx as usize;
        if ref_idx >= cand_logits.len() {
            continue;
        }
        let log_p_ref = ref_block.top_log_probs[j] as f64;
        if !log_p_ref.is_finite() {
            // Padding slot (fewer than top_k real entries): skip.
            continue;
        }
        let log_p_cand = (cand_logits[ref_idx] as f64) - lz;
        let p_ref = log_p_ref.exp();
        let p_cand = log_p_cand.exp();
        kld += p_ref * (log_p_ref - log_p_cand);
        sum_p_cand_at_ref_top += p_cand;
    }

    let sum_p_residual_ref = ref_block.residual_mass as f64;
    let sum_p_residual_cand = (1.0 - sum_p_cand_at_ref_top).max(0.0);
    if sum_p_residual_ref > 1e-9 && sum_p_residual_cand > 1e-9 {
        kld += sum_p_residual_ref * (sum_p_residual_ref.ln() - sum_p_residual_cand.ln());
    }
    // KLD ≥ 0 by Gibbs' inequality. The residual cross-term plus ~K-term fp64
    // sums accumulate cancellation roundoff up to ~1e-8 when P_ref ≈ Q_cand
    // (the self-consistency limit); a real math bug yields O(0.1+) negatives.
    // Guard against the latter, clamp the former.
    debug_assert!(kld >= -1e-6, "negative KLD beyond fp roundoff: {kld}");

    // Narrow at the boundary: f64 was the accumulator, f32 is the stored value.
    let kld = kld.max(0.0) as f32;
    let nll = if actual_next < cand_logits.len() {
        Some((-((cand_logits[actual_next] as f64) - lz)) as f32)
    } else {
        None
    };
    PositionScore { kld, nll }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_z_matches_naive() {
        let logits = [0.0f32, 1.0, 2.0, -1.0];
        // log(e^0 + e^1 + e^2 + e^-1)
        let naive = (1.0f64 + std::f64::consts::E + (2.0f64).exp() + (-1.0f64).exp()).ln();
        assert!((log_z(&logits) - naive).abs() < 1e-12);
    }

    #[test]
    fn top_k_picks_highest_and_orders_descending() {
        let logits = [0.0f32, 3.0, 1.0, 2.0];
        let r = top_k_log_softmax(&logits, 2);
        assert_eq!(r.indices, vec![1, 3]); // logits 3.0 then 2.0
                                           // log-probs descending, finite
        assert!(r.log_probs[0] > r.log_probs[1]);
        assert!(r.residual_mass > 0.0 && r.residual_mass < 1.0);
    }

    #[test]
    fn top_k_full_support_has_zero_residual() {
        let logits = [0.0f32, 1.0, 2.0];
        let r = top_k_log_softmax(&logits, 3);
        assert!(r.residual_mass < 1e-6, "residual {}", r.residual_mass);
    }

    #[test]
    fn top_k_breaks_ties_by_ascending_id() {
        let logits = [2.0f32, 2.0, 2.0, 0.0];
        let r = top_k_log_softmax(&logits, 2);
        assert_eq!(r.indices, vec![0, 1]); // equal logits → lowest ids first
    }

    /// THE invariant the whole refactor enforces: scoring a candidate against a
    /// reference built from *identical* logits yields KLD ≈ 0. This is the
    /// self-consistency property — at the math level it must be exact-to-roundoff.
    #[test]
    fn self_consistency_is_zero() {
        let logits: Vec<f32> = (0..512)
            .map(|i| ((i * 7 % 23) as f32) * 0.31 - 3.0)
            .collect();
        let red = top_k_log_softmax(&logits, 64);
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };
        let s = score_position(&rb, &logits, 5);
        assert!(
            s.kld < 1e-9,
            "self-consistency KLD should be ~0, got {}",
            s.kld
        );
    }

    #[test]
    fn kld_is_positive_for_different_distribution() {
        let ref_logits: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05).collect();
        let red = top_k_log_softmax(&ref_logits, 32);
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };
        // A clearly different candidate (reversed slope).
        let cand_logits: Vec<f32> = (0..128).map(|i| -(i as f32) * 0.05).collect();
        let s = score_position(&rb, &cand_logits, 0);
        assert!(s.kld > 0.1, "expected sizeable KLD, got {}", s.kld);
    }

    #[test]
    fn nll_of_argmax_is_smallest() {
        let logits = [0.0f32, 5.0, 1.0];
        let red = top_k_log_softmax(&logits, 3);
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };
        let nll_argmax = score_position(&rb, &logits, 1).nll.unwrap();
        let nll_other = score_position(&rb, &logits, 0).nll.unwrap();
        assert!(nll_argmax < nll_other);
        assert!(nll_argmax > 0.0);
    }
}
