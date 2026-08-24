//! CPU oracle for turning a logit row into a reference block.
//!
//! This is the contract a `k >= 256` device kernel must satisfy. It is the
//! *degraded* path — someone without a GPU can still build a reference
//! overnight, but the target is on-device batched top-k + log-sum-exp.
//!
//! See `crate::eval::mod` docs for background.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::{Estimator, QuantError, Result};

// ---------------------------------------------------------------------------
// TopKBlock
// ---------------------------------------------------------------------------

/// One scored position's truncated reference.
///
/// `top` is descending by log-probability (i.e. descending by logit), length
/// `k`. Ties (exactly equal logits) are broken by ascending token index so
/// the output is deterministic.
///
/// `residual_logprob` is `ln(max(1 - sum(exp(top logprobs)), 0))`: the log of
/// the probability mass *not* captured by the top-k. When that mass is zero
/// or negative after clamping (e.g. `k == vocab`) the honest value is
/// `f64::NEG_INFINITY` — the log of zero mass — and it must not become a
/// silent `0.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct TopKBlock {
    pub residual_logprob: f64,
    pub top: Vec<(u32, f32)>,
}

// ---------------------------------------------------------------------------
// log_sum_exp
// ---------------------------------------------------------------------------

/// Numerically stable `log(sum(exp(logits)))`.
///
/// Finds the max, accumulates `exp(v - max)` in `f64`, returns
/// `max + ln(sum)`. Empty input returns `f64::NEG_INFINITY`.
pub fn log_sum_exp(logits: &[f32]) -> f64 {
    if logits.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    // All entries were -inf (or empty, handled above).
    if max == f32::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    // If max is +inf or NaN, propagate. NaN case: sum will be NaN.
    if !max.is_finite() {
        return max as f64;
    }
    let max_f = max as f64;
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v as f64) - max_f).exp();
    }
    // sum > 0 because at least the max entry contributes exp(0)=1.
    max_f + sum.ln()
}

// ---------------------------------------------------------------------------
// Heap entry — bounded selection without vocab-sized allocation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct HeapEntry {
    logit: f32,
    idx: u32,
}

// Equality ignores the subtle -0 vs 0 distinction for the Eq bound; we use
// bitwise equality so `Eq` is well-defined. Ordering uses `total_cmp` which
// is a total order over f32 (including NaN, -0).
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.logit.to_bits() == other.logit.to_bits() && self.idx == other.idx
    }
}
impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let ord = self.logit.total_cmp(&other.logit);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
        // Larger idx is "smaller" (worse).
        other.idx.cmp(&self.idx)
    }
}

// ---------------------------------------------------------------------------
// topk_block
// ---------------------------------------------------------------------------

/// Oracle: turn one logit row into a [`TopKBlock`].
///
/// * `Err(Malformed)` if `k == 0` or `k > logits.len()`.
/// * Working-set allocation is `O(k)` — a bounded `BinaryHeap` of size `k`
///   in a single pass. No `Vec` proportional to the vocabulary is allocated.
/// * Ties: exactly equal logits (via `==` / identical bit pattern under
///   `total_cmp`) order by ascending token index, deterministically.
/// * Log-probabilities are computed as `logit - log_z` in `f64` and stored as
///   `f32`.
pub fn topk_block(logits: &[f32], k: usize) -> Result<TopKBlock> {
    if k == 0 {
        return Err(QuantError::Malformed("topk_block: k == 0".into()));
    }
    if k > logits.len() {
        return Err(QuantError::Malformed(format!(
            "topk_block: k ({}) > vocab ({})",
            k,
            logits.len()
        )));
    }

    let log_z = log_sum_exp(logits);

    // Bounded min-heap selection — O(vocab * log k) time, O(k) space.
    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k);
    for (idx, &logit) in logits.iter().enumerate() {
        let entry = HeapEntry {
            logit,
            idx: idx as u32,
        };
        if heap.len() < k {
            heap.push(Reverse(entry));
        } else if let Some(worst) = heap.peek().map(|r| r.0.clone()) {
            // entry > worst means entry is strictly better (larger logit, or
            // equal logit but smaller index).
            if entry > worst {
                heap.pop();
                heap.push(Reverse(entry));
            }
        }
    }

    // Extract and sort descending by logit (and ascending idx on tie).
    let mut items: Vec<HeapEntry> = heap.into_iter().map(|r| r.0).collect();
    items.sort_by(|a, b| {
        // Descending by logit, ascending idx when exactly equal.
        // We treat total_cmp == Equal (bit-equal) as tie. For broader "== "
        // equality (-0 == 0) the branch still gives ascending idx which is
        // deterministic and matches the spec's intent.
        let ord = b.logit.total_cmp(&a.logit);
        if ord == std::cmp::Ordering::Equal {
            a.idx.cmp(&b.idx)
        } else {
            ord
        }
    });

    // Compute log-probs in f64, store as f32.
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    for e in &items {
        let logp_f64 = (e.logit as f64) - log_z;
        top.push((e.idx, logp_f64 as f32));
    }

    // Residual mass, summed DIRECTLY over the entries outside the top-k.
    //
    // The obvious formulation — `1 - sum(top-k mass)` — catastrophically
    // cancels in exactly the case that matters. For a trained LM at k=256 the
    // top-k routinely captures >99.9% of the probability mass, so that
    // subtraction throws away every significant digit of the answer. Concrete
    // case, from the unit tests below: one logit of 100.0 among 99 zeros has a
    // true residual of 95*exp(-100) ~= 3.5e-42, which f64 represents
    // comfortably, yet `1 - captured` rounds to exactly 0 and would report
    // NEG_INFINITY. Summing the complement has no cancellation at all, and it
    // also makes `k == vocab` fall out correctly: there are no entries outside
    // the top-k, the sum stays 0, and NEG_INFINITY is the honest answer.
    let mut top_idx_sorted: Vec<u32> = items.iter().map(|e| e.idx).collect();
    top_idx_sorted.sort_unstable();
    let mut residual_mass = 0.0f64;
    for (idx, &logit) in logits.iter().enumerate() {
        if top_idx_sorted.binary_search(&(idx as u32)).is_ok() {
            continue;
        }
        let logp = (logit as f64) - log_z;
        if logp.is_finite() {
            residual_mass += logp.exp();
        }
    }
    let residual_logprob = if residual_mass > 0.0 && residual_mass.is_finite() {
        residual_mass.ln()
    } else {
        f64::NEG_INFINITY
    };

    Ok(TopKBlock {
        residual_logprob,
        top,
    })
}

// ---------------------------------------------------------------------------
// nll_of
// ---------------------------------------------------------------------------

/// Negative log-likelihood of the true next token: `-(logit[actual] - log_z)`.
///
/// `Err(Malformed)` if `actual_next` is out of range.
pub fn nll_of(logits: &[f32], actual_next: usize) -> Result<f64> {
    if actual_next >= logits.len() {
        return Err(QuantError::Malformed(format!(
            "nll_of: actual_next {} out of range for vocab {}",
            actual_next,
            logits.len()
        )));
    }
    let log_z = log_sum_exp(logits);
    Ok(-((logits[actual_next] as f64) - log_z))
}

// ---------------------------------------------------------------------------
// kld_from_block
// ---------------------------------------------------------------------------

/// Truncated KL (`reference || candidate`) approximated from a [`TopKBlock`].
///
/// ```text
/// KL ≈ Σ_{i in top-k} p_ref(i) * (logp_ref(i) - logp_cand(i))
///      + p_residual_ref * (log p_residual_ref - log p_residual_cand)
/// ```
/// where `p_ref(i) = exp(logp_ref(i))`, `logp_cand(i) = cand_logit(i) - log_z_cand`,
/// `log_z_cand = log_sum_exp(candidate_logits)`, and
/// `p_residual_ref = exp(reference.residual_logprob)` (0 when `NEG_INFINITY`).
/// `p_residual_cand = max(1 - Σ_{i in top-k} p_cand(i), 0)`.
///
/// # Approximation
///
/// This is a **truncated estimator**. Only the `k` outcomes stored in the
/// reference contribute individually; the remaining `vocab - k` outcomes are
/// lumped into a single aggregated "residual" bucket. That is cheaper to
/// store and score (about 2 KB/position at `k = 256` vs 993 KB for the full
/// row) but it is *not* the full-vocabulary KL. Its bias depends on how much
/// mass the top-k retains, which [`measure_truncation_bias`] quantifies. Do
/// not compare numbers from a truncated estimator against full-vocab numbers
/// (e.g. llama.cpp `--kl-divergence`) without the `bias_vs_full` carried in
/// [`Estimator::TopK`].
pub fn kld_from_block(reference: &TopKBlock, candidate_logits: &[f32]) -> Result<f64> {
    if candidate_logits.is_empty() {
        return Err(QuantError::Malformed(
            "kld_from_block: candidate_logits empty".into(),
        ));
    }
    // Validate that every reference index is in range for the candidate.
    for (idx, _) in &reference.top {
        if (*idx as usize) >= candidate_logits.len() {
            return Err(QuantError::Malformed(format!(
                "kld_from_block: reference index {} out of range for candidate vocab {}",
                idx,
                candidate_logits.len()
            )));
        }
    }

    let log_z = log_sum_exp(candidate_logits);

    let mut kld = 0.0f64;
    let mut sum_p_cand_at_top = 0.0f64;

    for (idx, logp_ref_f32) in &reference.top {
        let logp_ref = *logp_ref_f32 as f64;
        let p_ref = logp_ref.exp();
        // If p_ref is 0 (logp_ref == -inf) contribution is 0.
        if !(p_ref > 0.0) {
            continue;
        }
        let logp_cand = (candidate_logits[*idx as usize] as f64) - log_z;
        let p_cand = logp_cand.exp();
        sum_p_cand_at_top += if p_cand.is_finite() { p_cand } else { 0.0 };
        // p_ref * (logp_ref - logp_cand)
        kld += p_ref * (logp_ref - logp_cand);
    }

    let p_residual_ref = if reference.residual_logprob.is_finite() {
        reference.residual_logprob.exp()
    } else {
        0.0
    };
    let p_residual_cand = (1.0 - sum_p_cand_at_top).max(0.0);

    // Mirror eval_hipfire.rs: ignore residual cross-term when either side is
    // negligible (< 1e-9) to avoid amplifying float noise into a large log
    // difference. When both sides are substantial, include it.
    if p_residual_ref > 1e-9 && p_residual_cand > 1e-9 {
        kld += p_residual_ref * (reference.residual_logprob - p_residual_cand.ln());
    }
    // Gibbs' inequality: KL >= 0 up to roundoff.
    if kld < 0.0 && kld > -1e-9 {
        kld = 0.0;
    } else {
        kld = kld.max(0.0);
    }
    Ok(kld)
}

// ---------------------------------------------------------------------------
// TruncationBias
// ---------------------------------------------------------------------------

/// Measured bias of a top-k truncated estimator versus the full vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncationBias {
    pub k: u32,
    pub mean_abs_delta: f64,
    pub max_abs_delta: f64,
    pub captured_mass_mean: f64,
    pub n: usize,
}

/// Measure how much a top-k truncation distorts a per-row entropy-style
/// divergence term.
///
/// For each row we compute:
///
/// * `full_term = Σ_v p(v) * ln p(v)`  (negative entropy, exact over the full
///   vocabulary, with `p = softmax(logits)`), and
/// * `trunc_term = Σ_{top-k} p(v) * ln p(v) + p_residual * ln p_residual`
///   where the `vocab - k` tail outcomes are lumped into a single bucket of
///   mass `p_residual = exp(residual_logprob)`.
///
/// `|full_term - trunc_term|` is the per-row truncation delta; its mean and
/// max are reported as `mean_abs_delta` / `max_abs_delta`. That delta is the
/// quantity stored in [`Estimator::TopK::bias_vs_full`]: it quantifies the
/// irreducible error from lumping the tail, turning an uncalibrated truncated
/// estimator into a comparable one. `captured_mass_mean` is the mean of
/// `Σ_{top-k} p(v)` — the probability mass the top-k actually retains.
/// At `k = 256` over a 248,320 vocab the top-k is only 0.103% of the *entries*;
/// the point of this measurement is to see how much *mass* it retains.
pub fn measure_truncation_bias(rows: &[&[f32]], k: usize) -> Result<TruncationBias> {
    if k == 0 {
        return Err(QuantError::Malformed(
            "measure_truncation_bias: k == 0".into(),
        ));
    }
    if rows.is_empty() {
        return Err(QuantError::Malformed(
            "measure_truncation_bias: no rows".into(),
        ));
    }
    for (i, r) in rows.iter().enumerate() {
        if k > r.len() {
            return Err(QuantError::Malformed(format!(
                "measure_truncation_bias: k ({}) > vocab ({}) at row {}",
                k,
                r.len(),
                i
            )));
        }
        if r.is_empty() {
            return Err(QuantError::Malformed(format!(
                "measure_truncation_bias: empty row at index {}",
                i
            )));
        }
    }

    let mut sum_abs = 0.0f64;
    let mut max_abs: f64 = 0.0;
    let mut sum_captured = 0.0f64;

    for row in rows {
        let log_z = log_sum_exp(row);
        // Full term: Σ p * ln p
        let mut full_term = 0.0f64;
        for &logit in *row {
            let logp = (logit as f64) - log_z;
            if !logp.is_finite() {
                // logp == -inf => p == 0 => 0 * (-inf) == 0 by continuity.
                continue;
            }
            let p = logp.exp();
            if p > 0.0 {
                full_term += p * logp;
            }
        }

        let block = topk_block(row, k)?;
        let mut trunc_term = 0.0f64;
        let mut captured = 0.0f64;
        for (_idx, logp_f32) in &block.top {
            let logp = *logp_f32 as f64;
            if !logp.is_finite() {
                continue;
            }
            let p = logp.exp();
            if p > 0.0 {
                trunc_term += p * logp;
                captured += p;
            }
        }
        if block.residual_logprob.is_finite() {
            let p_res = block.residual_logprob.exp();
            if p_res > 0.0 {
                trunc_term += p_res * block.residual_logprob;
            }
        }
        // captured_mass for this row is Σ_{topk} p; computed above but also
        // verify via captured variable (same). Use it for mean.
        sum_captured += captured;

        let delta = (full_term - trunc_term).abs();
        sum_abs += delta;
        if delta > max_abs {
            max_abs = delta;
        }
    }

    let n = rows.len();
    Ok(TruncationBias {
        k: k as u32,
        mean_abs_delta: sum_abs / n as f64,
        max_abs_delta: max_abs,
        captured_mass_mean: sum_captured / n as f64,
        n,
    })
}

// ---------------------------------------------------------------------------
// estimator_for
// ---------------------------------------------------------------------------

/// Build the [`Estimator`] value to record in a reference header.
///
/// Carries `bias_vs_full` when a [`TruncationBias`] measurement exists.
pub fn estimator_for(k: usize, bias: Option<&TruncationBias>) -> Estimator {
    match bias {
        Some(b) => Estimator::TopK {
            k: k as u32,
            bias_vs_full: Some(b.mean_abs_delta),
        },
        None => Estimator::TopK {
            k: k as u32,
            bias_vs_full: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    // ---- log_sum_exp ----

    #[test]
    fn log_sum_exp_small_vector_within_1e12() {
        // Hand-computed: logits [0, 1, 2]
        // LSE = ln(exp0+exp1+exp2) = ln(1 + e + e^2)
        let v = [0.0f32, 1.0, 2.0];
        let got = log_sum_exp(&v);
        let expected = (1.0f64 + std::f64::consts::E + std::f64::consts::E.powi(2)).ln();
        // ln(1+e+e^2) = about 2.40760596444
        assert!(
            approx_eq(got, expected, 1e-12),
            "got {got} expected {expected} diff {}",
            (got - expected).abs()
        );
        // Also check against direct f64 calc
        let direct = (0.0f64.exp() + 1.0f64.exp() + 2.0f64.exp()).ln();
        assert!(approx_eq(got, direct, 1e-12));
    }

    #[test]
    fn log_sum_exp_empty_is_neg_inf() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_does_not_overflow_at_1e30() {
        let v = [1e30f32, 1e30f32, 1e30f32 - 1.0];
        let got = log_sum_exp(&v);
        assert!(got.is_finite(), "overflowed to {got}");
        // Compare against the f32 literal widened to f64, NOT 1e30f64:
        // `1e30f32` is 1.0000000150474662e30, so an absolute tolerance
        // against 1e30f64 is off by ~1.5e22 before the function is even
        // called. At this magnitude f64 resolution is ~2.2e14, so the
        // + ln(2 + exp(-1)) term is below the representable step and the
        // answer is exactly the max. Relative tolerance is the only
        // meaningful check here.
        let max = 1e30f32 as f64;
        assert!(
            ((got - max) / max).abs() < 1e-12,
            "expected near {max}, got {got}"
        );
    }

    #[test]
    fn log_sum_exp_large_negative_does_not_underflow() {
        let v = [-1e30f32, -1e30f32];
        let got = log_sum_exp(&v);
        assert!(got.is_finite() || got == f64::NEG_INFINITY);
        // With both at -1e30, lse = -1e30 + ln2
        let expected = -1e30f64 + (2.0f64).ln();
        assert!(approx_eq(got, expected, 1e-6 * expected.abs().max(1.0)));
    }

    // ---- topk_block ----

    #[test]
    fn topk_block_known_small_vector() {
        // logits: idx0=0, 1=3, 2=1, 3=2
        // sorted descending: idx1(3), idx3(2), idx2(1), idx0(0)
        let logits = [0.0f32, 3.0, 1.0, 2.0];
        let k = 2;
        let block = topk_block(&logits, k).unwrap();
        assert_eq!(block.top.len(), 2);
        // Expect descending order: (1, logp for 3), (3, logp for 2)
        assert_eq!(block.top[0].0, 1);
        assert_eq!(block.top[1].0, 3);
        // Check ordering descending by logprob (stored as f32)
        assert!(block.top[0].1 >= block.top[1].1);
        // Verify logprobs are correct: logp = logit - log_z
        let lse = log_sum_exp(&logits);
        for (idx, lp) in &block.top {
            let expected = (logits[*idx as usize] as f64 - lse) as f32;
            assert!(
                (*lp - expected).abs() < 1e-6,
                "logprob mismatch for idx {idx}: got {lp} expected {expected}"
            );
        }
        // Residual: captured + exp(residual) == 1
        let captured: f64 = block.top.iter().map(|(_, lp)| (*lp as f64).exp()).sum();
        let residual_mass = if block.residual_logprob.is_finite() {
            block.residual_logprob.exp()
        } else {
            0.0
        };
        assert!(approx_eq(captured + residual_mass, 1.0, 1e-6));
    }

    #[test]
    fn topk_block_determinism_ties_ascending_index() {
        // Several exactly equal logits
        let logits = [5.0f32, 5.0, 5.0, 5.0, 1.0];
        let block = topk_block(&logits, 3).unwrap();
        assert_eq!(block.top.len(), 3);
        // Among tied 5.0's, must be ascending indices 0,1,2
        assert_eq!(block.top[0].0, 0);
        assert_eq!(block.top[1].0, 1);
        assert_eq!(block.top[2].0, 2);
        // Two calls identical
        let block2 = topk_block(&logits, 3).unwrap();
        assert_eq!(block.top, block2.top);
        assert_eq!(block.residual_logprob, block2.residual_logprob);
    }

    #[test]
    fn topk_block_ties_mixed() {
        // Mix of values with ties not at top
        // logits: 0:10, 1:5, 2:5, 3:5, 4:1
        let logits = [10.0f32, 5.0, 5.0, 5.0, 1.0];
        let block = topk_block(&logits, 3).unwrap();
        // Sorted: idx0 (10), then among 5's ascending: 1,2
        assert_eq!(block.top[0].0, 0);
        assert_eq!(block.top[1].0, 1);
        assert_eq!(block.top[2].0, 2);
    }

    #[test]
    fn topk_block_errors() {
        let v = [1.0f32, 2.0, 3.0];
        assert!(matches!(topk_block(&v, 0), Err(QuantError::Malformed(_))));
        assert!(matches!(topk_block(&v, 4), Err(QuantError::Malformed(_))));
        // ok at boundary
        assert!(topk_block(&v, 3).is_ok());
    }

    #[test]
    fn topk_block_residual_k_equals_vocab() {
        let logits = [1.0f32, 2.0, 3.0, 0.5];
        let block = topk_block(&logits, logits.len()).unwrap();
        assert_eq!(block.residual_logprob, f64::NEG_INFINITY);
        assert!(!block.residual_logprob.is_finite());
        // Captured mass ~1
        let captured: f64 = block.top.iter().map(|(_, lp)| (*lp as f64).exp()).sum();
        // Log-probs are stored as f32, so the recovered mass carries ~1e-7
        // relative error. 1e-9 is below the storage precision of the format.
        assert!(approx_eq(captured, 1.0, 1e-6));
        assert!(!block.residual_logprob.is_nan());
    }

    #[test]
    fn topk_block_residual_peaked_small_k() {
        // Peaked distribution: one large logit
        let mut logits = vec![0.0f32; 100];
        logits[42] = 100.0;
        let block = topk_block(&logits, 2).unwrap();
        assert!(block.residual_logprob.is_finite());
        let captured: f64 = block.top.iter().map(|(_, lp)| (*lp as f64).exp()).sum();
        let residual = block.residual_logprob.exp();
        assert!(approx_eq(captured + residual, 1.0, 1e-6));
        // With one huge logit, captured should be ~1.0
        assert!(captured > 0.99);
    }

    #[test]
    fn topk_block_bounded_allocation_large_vocab() {
        // 100k synthetic vocab, k=256
        let n = 100_000usize;
        let mut logits = Vec::with_capacity(n);
        for i in 0..n {
            // deterministic pseudo-random: make distinct values so ordering is clear
            // logit = (i % 1000) as f32 + (i as f32)*1e-6
            logits.push((i % 1000) as f32 + (i as f32) * 1e-6);
        }
        let k = 256;
        let block = topk_block(&logits, k).unwrap();
        assert_eq!(block.top.len(), 256);
        // Correctly ordered descending
        for w in block.top.windows(2) {
            let (idx_a, lp_a) = w[0];
            let (idx_b, lp_b) = w[1];
            // lp_a >= lp_b
            assert!(
                lp_a >= lp_b,
                "not descending: idx {idx_a} lp {lp_a} vs idx {idx_b} lp {lp_b}"
            );
            if logits[idx_a as usize] == logits[idx_b as usize] {
                assert!(idx_a < idx_b, "tie not ascending: {} vs {}", idx_a, idx_b);
            }
        }
        // Check that top indices are indeed the largest logits
        // The synthetic construction makes logits increasing with i within each mod cycle,
        // so the top values will be the highest i with highest mod. Instead of
        // computing exact, just check that none of the tail exceeds the smallest top.
        let min_top_logit = logits[block.top.last().unwrap().0 as usize];
        // Brute check that all tail logits <= min_top_logit (or equal with larger idx)
        // We do a quick scan, but with bounded logic we already know the heap did.
        // Verify a few tail entries are not better.
        for (i, &logit) in logits.iter().enumerate().take(1000) {
            if block.top.iter().any(|(idx, _)| *idx as usize == i) {
                continue;
            }
            assert!(
                logit < min_top_logit
                    || (logit == min_top_logit && (i as u32) > block.top.last().unwrap().0),
                "tail entry {i} with logit {logit} should not beat min top {min_top_logit}"
            );
        }
        // Residual + captured ==1 within 1e-6
        let captured: f64 = block.top.iter().map(|(_, lp)| (*lp as f64).exp()).sum();
        let residual = if block.residual_logprob.is_finite() {
            block.residual_logprob.exp()
        } else {
            0.0
        };
        assert!(approx_eq(captured + residual, 1.0, 1e-6));
    }

    // ---- nll_of ----

    #[test]
    fn nll_of_matches_formula_and_errors() {
        let logits = [0.0f32, 1.0, 2.0];
        let lse = log_sum_exp(&logits);
        for i in 0..logits.len() {
            let nll = nll_of(&logits, i).unwrap();
            let expected = -((logits[i] as f64) - lse);
            assert!(
                approx_eq(nll, expected, 1e-12),
                "idx {i}: {nll} vs {expected}"
            );
        }
        assert!(matches!(nll_of(&logits, 10), Err(QuantError::Malformed(_))));
        assert!(matches!(nll_of(&[], 0), Err(QuantError::Malformed(_))));
    }

    // ---- kld_from_block ----

    #[test]
    fn kld_from_block_identical_is_zero() {
        let logits = [1.0f32, 2.0, 3.0, 0.5, -1.0];
        let block = topk_block(&logits, 3).unwrap();
        let kld = kld_from_block(&block, &logits).unwrap();
        assert!(
            kld.abs() < 1e-6,
            "identical logits KLD should be ~0, got {kld}"
        );
    }

    #[test]
    fn kld_from_block_different_is_positive() {
        let ref_logits = [5.0f32, 0.0, 0.0, 0.0, 0.0];
        let cand_logits = [0.0f32, 5.0, 0.0, 0.0, 0.0];
        let block = topk_block(&ref_logits, 2).unwrap();
        let kld = kld_from_block(&block, &cand_logits).unwrap();
        assert!(
            kld > 0.1,
            "clearly different candidate should have strictly positive KLD, got {kld}"
        );
    }

    #[test]
    fn kld_from_block_errors() {
        let block = topk_block(&[1.0f32, 2.0, 3.0], 2).unwrap();
        assert!(matches!(
            kld_from_block(&block, &[]),
            Err(QuantError::Malformed(_))
        ));
        // Candidate vocab smaller than reference top index
        assert!(matches!(
            kld_from_block(&block, &[1.0f32, 2.0]),
            Err(QuantError::Malformed(_))
        ));
    }

    #[test]
    fn kld_from_block_with_full_vocab_residual() {
        // k == vocab => residual NEG_INF, KL with identical should be 0
        let logits = [1.0f32, 2.0, 3.0];
        let block = topk_block(&logits, 3).unwrap();
        assert_eq!(block.residual_logprob, f64::NEG_INFINITY);
        let kld = kld_from_block(&block, &logits).unwrap();
        assert!(kld.abs() < 1e-9);
    }

    // ---- truncation bias ----

    #[test]
    fn truncation_bias_peaked_vs_uniform() {
        // Peaked rows: one token dominates
        let mut peaked_logits: Vec<Vec<f32>> = Vec::new();
        for _ in 0..10 {
            let mut v = vec![0.0f32; 1000];
            v[0] = 20.0;
            peaked_logits.push(v);
        }
        let peaked_refs: Vec<&[f32]> = peaked_logits.iter().map(|v| v.as_slice()).collect();
        let bias_peaked = measure_truncation_bias(&peaked_refs, 4).unwrap();
        assert!(
            bias_peaked.captured_mass_mean > 0.99,
            "peaked captured should be ~1, got {}",
            bias_peaked.captured_mass_mean
        );
        assert!(
            bias_peaked.mean_abs_delta < 0.1,
            "peaked delta should be near 0, got {}",
            bias_peaked.mean_abs_delta
        );

        // Near-uniform rows: all logits 0 => uniform distribution
        let mut uniform_logits: Vec<Vec<f32>> = Vec::new();
        for _ in 0..10 {
            uniform_logits.push(vec![0.0f32; 1000]);
        }
        let uniform_refs: Vec<&[f32]> = uniform_logits.iter().map(|v| v.as_slice()).collect();
        let bias_uniform = measure_truncation_bias(&uniform_refs, 4).unwrap();
        // captured = 4/1000 = 0.004
        assert!(
            bias_uniform.captured_mass_mean < 0.01,
            "uniform captured should be small, got {}",
            bias_uniform.captured_mass_mean
        );
        assert!(
            bias_uniform.mean_abs_delta > 0.5,
            "uniform delta should be materially larger, got {}",
            bias_uniform.mean_abs_delta
        );
        // Peaked delta must be much smaller than uniform delta
        assert!(
            bias_peaked.mean_abs_delta < bias_uniform.mean_abs_delta,
            "peaked {} should be < uniform {}",
            bias_peaked.mean_abs_delta,
            bias_uniform.mean_abs_delta
        );
    }

    #[test]
    fn truncation_bias_large_vocab_uniform_small_k() {
        // 10k vocab, k=4, uniform
        let row = vec![0.0f32; 10_000];
        let rows: Vec<&[f32]> = vec![&row];
        let bias = measure_truncation_bias(&rows, 4).unwrap();
        assert!(bias.captured_mass_mean < 0.001);
        assert!(bias.mean_abs_delta > 1.0);
    }

    #[test]
    fn truncation_bias_errors() {
        let row = vec![0.0f32; 10];
        let rows: Vec<&[f32]> = vec![&row];
        assert!(matches!(
            measure_truncation_bias(&rows, 0),
            Err(QuantError::Malformed(_))
        ));
        assert!(matches!(
            measure_truncation_bias(&[], 2),
            Err(QuantError::Malformed(_))
        ));
        assert!(matches!(
            measure_truncation_bias(&rows, 11),
            Err(QuantError::Malformed(_))
        ));
    }

    #[test]
    fn estimator_for_builds() {
        let bias = TruncationBias {
            k: 4,
            mean_abs_delta: 0.123,
            max_abs_delta: 0.2,
            captured_mass_mean: 0.9,
            n: 10,
        };
        let e = estimator_for(4, Some(&bias));
        assert_eq!(
            e,
            Estimator::TopK {
                k: 4,
                bias_vs_full: Some(0.123)
            }
        );
        let e2 = estimator_for(8, None);
        assert_eq!(
            e2,
            Estimator::TopK {
                k: 8,
                bias_vs_full: None
            }
        );
    }

    // Ensure no panic on edge: single element vocab
    #[test]
    fn single_element_vocab() {
        let logits = [5.0f32];
        let block = topk_block(&logits, 1).unwrap();
        assert_eq!(block.top.len(), 1);
        assert_eq!(block.top[0].0, 0);
        assert_eq!(block.residual_logprob, f64::NEG_INFINITY);
        let nll = nll_of(&logits, 0).unwrap();
        assert!(approx_eq(nll, 0.0, 1e-12)); // only one outcome => p=1 => nll 0
        let kld = kld_from_block(&block, &logits).unwrap();
        assert!(kld.abs() < 1e-9);
        let rows: Vec<&[f32]> = vec![&logits];
        let bias = measure_truncation_bias(&rows, 1).unwrap();
        assert!(bias.captured_mass_mean > 0.999);
        assert!(bias.mean_abs_delta < 1e-9);
    }
}
