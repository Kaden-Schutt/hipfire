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

    // 1. Pick top-k indices by raw logit (descending). NaN logits are not
    // valid candidates; dropping them matches the GPU argmax path where
    // `NaN > best` is false and prevents comparator panics.
    let mut idx: Vec<usize> = (0..n).filter(|&i| !logits[i].is_nan()).collect();
    if idx.is_empty() {
        return 0;
    }
    let k = if top_k == 0 || top_k >= idx.len() {
        idx.len()
    } else {
        top_k
    };
    if k < idx.len() {
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

    #[test]
    fn greedy_nan_logits_do_not_win() {
        let mut rng = Xorshift::new(1);
        assert_eq!(
            sample_token(&[1.0, 5.0, 3.0, f32::NAN], 0.0, 0, 1.0, &mut rng),
            1
        );
        assert_eq!(
            sample_token(&[5.0, f32::NAN, 0.1, 2.0], 0.0, 0, 1.0, &mut rng),
            0
        );
    }

    #[test]
    fn sampled_path_drops_nan_logits() {
        let mut rng = Xorshift::new(42);
        for _ in 0..64 {
            let tok = sample_token(
                &[1.0, f32::NAN, 3.0, 2.0, f32::NAN, 0.5],
                1.0,
                3,
                0.9,
                &mut rng,
            );
            assert!([0, 2, 3, 5].contains(&tok));
        }
    }

    #[test]
    fn all_nan_logits_return_zero_without_panic() {
        let mut rng = Xorshift::new(7);
        assert_eq!(
            sample_token(&[f32::NAN, f32::NAN], 0.0, 0, 1.0, &mut rng),
            0
        );
        assert_eq!(
            sample_token(&[f32::NAN, f32::NAN], 1.0, 0, 1.0, &mut rng),
            0
        );
    }
}
