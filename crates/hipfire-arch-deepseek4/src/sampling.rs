// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampler for DeepSeek V4. Pure greedy argmax on a quantized instruct
//! model falls into self-reinforcing loops (observed: prompt → coherent
//! prose → `import hashlib\nimport hashlib\n...` once the model enters a
//! code-fence context). The HuggingFace card for `deepseek-ai/DeepSeek-V4-Flash`
//! recommends `temperature = 1.0, top_p = 1.0` for local deployment.
//!
//! `sample_token` supports both top-k and top-p (nucleus) filters in
//! that order; either or both can be disabled. The PRNG is xorshift64*
//! — tiny, deterministic, zero deps.

/// xorshift64* PRNG. Reproducible from a seed; non-zero seed forces a
/// canonical splash for the 0 → 0 fixed-point case.
pub struct Xorshift {
    s: u64,
}

impl Xorshift {
    pub fn new(seed: u64) -> Self {
        Self {
            s: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}

/// Sample next token from `logits`.
///
/// - `temp <= 0`: greedy argmax (deterministic; ignores top_k / top_p).
/// - Otherwise: optional top-k filter → softmax with temperature →
///   optional top-p (nucleus) filter → multinomial draw via inverse CDF.
///
/// `top_k == 0` or `top_k >= |logits|` disables the top-k filter.
/// `top_p >= 1.0` (or `<= 0.0`) disables the top-p filter.
///
/// HF DeepSeek V4 Flash recommended defaults: `temp = 1.0, top_p = 1.0,
/// top_k = 0`.
pub fn sample_token(
    logits: &[f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut Xorshift,
) -> u32 {
    if temp <= 0.0 {
        // hunt3 M-B (FinalFix): explicit `>`-based fold mirrors the GPU kernel
        // (argmax.hip: `if (data[i] > lmax)`, seed -1e30). `v > best` is false
        // for NaN, so a NaN logit never wins and never displaces the real max —
        // unlike `max_by(partial_cmp.unwrap_or(Less))`, which keeps a trailing
        // NaN candidate. Seeded at NEG_INFINITY so the first finite logit wins.
        return logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 {
                    (i, v)
                } else {
                    best
                }
            })
            .0 as u32;
    }
    let n = logits.len();

    // hunt3 M-B (FinalFix): drop NaN logits up front. A non-total comparator
    // (NaN compares Less in both directions) makes select_nth_unstable_by /
    // sort_unstable_by yield an *unspecified* partition — empirically NaNs can
    // land at the HEAD of the top-k and displace real finalists. Partitioning
    // NaN out here guarantees no NaN index ever reaches the top-k or the
    // softmax, matching the GPU kernel's NaN-dropping argmax. The remaining
    // comparators then only ever see finite values, so partial_cmp is total.
    let mut idx: Vec<usize> = (0..n).filter(|&i| !logits[i].is_nan()).collect();
    if idx.is_empty() {
        // All-NaN logits: nothing finite to sample. Return token 0 rather than
        // panicking; the recurrent-state guard upstream treats this as a
        // degenerate request.
        return 0;
    }
    let m = idx.len();
    let k = if top_k == 0 || top_k >= m { m } else { top_k };

    // 1. Pick top-k indices by raw logit (descending). Only finite logits
    //    remain, so the comparator is a total order and never panics.
    if k < m {
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
    }

    // 2. Temperature-scaled softmax over the filtered set.
    let max_l = idx
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - max_l) / temp).exp())
        .collect();
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        // hunt3 M-B (FinalFix): degenerate-softmax fallback. `idx` holds only
        // finite-logit indices (NaNs were partitioned out above), but use the
        // same GPU-parity `>`-based fold for consistency — NaN can never win.
        return idx
            .iter()
            .fold((idx[0], f32::NEG_INFINITY), |best, &i| {
                if logits[i] > best.1 {
                    (i, logits[i])
                } else {
                    best
                }
            })
            .0 as u32;
    }
    for w in weights.iter_mut() {
        *w /= sum;
    }

    // 3. Optional top-p (nucleus) prune. Sort idx by probability desc,
    //    drop the tail once cumulative mass reaches top_p, renormalise.
    if top_p > 0.0 && top_p < 1.0 {
        let mut order: Vec<usize> = (0..idx.len()).collect();
        // hunt3 M-B (FinalFix): all weights are finite here — NaN logits were
        // dropped at the top-k stage and the `!sum.is_finite()` guard above
        // already returned for any +inf weight, so partial_cmp is total and the
        // unwrap_or branch is unreachable. Equal is the safe neutral fallback.
        order.sort_unstable_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0;
        let mut cutoff = order.len();
        for (rank, &j) in order.iter().enumerate() {
            cum += weights[j];
            if cum >= top_p {
                cutoff = rank + 1;
                break;
            }
        }
        let keep: std::collections::HashSet<usize> = order.iter().take(cutoff).copied().collect();
        let mut new_idx = Vec::with_capacity(cutoff);
        let mut new_w = Vec::with_capacity(cutoff);
        for (j, &id) in idx.iter().enumerate() {
            if keep.contains(&j) {
                new_idx.push(id);
                new_w.push(weights[j]);
            }
        }
        let new_sum: f32 = new_w.iter().sum();
        if new_sum > 0.0 && new_sum.is_finite() {
            for w in new_w.iter_mut() {
                *w /= new_sum;
            }
            idx = new_idx;
            weights = new_w;
        }
    }

    // 4. Multinomial draw via inverse CDF.
    let r = rng.next_f32();
    let mut acc = 0.0;
    for (j, &w) in weights.iter().enumerate() {
        acc += w;
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    // hunt3 M-B (FinalFix): NaN logits must never be selected and must never
    // displace the true max, mirroring argmax.hip's `data[i] > lmax` semantics.

    #[test]
    fn greedy_nan_at_last_index_not_selected() {
        let mut rng = Xorshift::new(1);
        // GPU kernel returns idx 1 (5.0) for this input; the old max_by idiom
        // returned idx 3 (the NaN).
        let logits = [1.0f32, 5.0, 3.0, f32::NAN];
        assert_eq!(sample_token(&logits, 0.0, 0, 1.0, &mut rng), 1);
    }

    #[test]
    fn greedy_nan_does_not_drop_true_max() {
        let mut rng = Xorshift::new(1);
        // The true max (5.0) precedes the NaN; the old idiom dropped it.
        let logits = [5.0f32, f32::NAN, 0.1, 2.0];
        assert_eq!(sample_token(&logits, 0.0, 0, 1.0, &mut rng), 0);
    }

    #[test]
    fn greedy_all_nan_does_not_panic() {
        let mut rng = Xorshift::new(1);
        let logits = [f32::NAN, f32::NAN, f32::NAN];
        // No finite max exists; fold seed (idx 0) is returned, no panic.
        let _ = sample_token(&logits, 0.0, 0, 1.0, &mut rng);
    }

    #[test]
    fn sampled_path_drops_nan_from_topk() {
        let mut rng = Xorshift::new(42);
        // Two scattered NaNs around three real finalists. With temp>0 + top_k,
        // a surviving NaN would make the softmax sum NaN and route to the
        // fallback. Repeated draws must only ever return a finite-logit index.
        let logits = [1.0f32, f32::NAN, 3.0, 2.0, f32::NAN, 0.5];
        let finite: std::collections::HashSet<u32> = [0u32, 2, 3, 5].into_iter().collect();
        for _ in 0..256 {
            let tok = sample_token(&logits, 1.0, 3, 0.9, &mut rng);
            assert!(
                finite.contains(&tok),
                "sampler returned a NaN-indexed token: {tok}"
            );
        }
    }

    #[test]
    fn sampled_path_all_nan_returns_zero() {
        let mut rng = Xorshift::new(7);
        let logits = [f32::NAN, f32::NAN];
        assert_eq!(sample_token(&logits, 1.0, 0, 1.0, &mut rng), 0);
    }

    #[test]
    fn greedy_finite_unaffected() {
        let mut rng = Xorshift::new(1);
        // Sanity: no NaN → behaves exactly like a plain argmax.
        let logits = [0.1f32, 0.2, 0.9, 0.3, 0.4];
        assert_eq!(sample_token(&logits, 0.0, 0, 1.0, &mut rng), 2);
    }
}
