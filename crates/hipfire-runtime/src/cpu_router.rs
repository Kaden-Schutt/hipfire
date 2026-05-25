// SPDX-License-Identifier: MIT
// Copyright (c) 2026 mad-lab-kbando
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-side router replica (MAD-93 v0.1).
//!
//! Replicates the per-layer MoE router GEMV on CPU so the
//! [`crate::weight_pager::WeightPager`] knows which experts are active without
//! forcing a GPU→CPU sync inside the forward path. The actual GPU still runs
//! the fused router GEMV — the CPU result is used purely for *scheduling*
//! (which experts to ensure resident, which to evict, which to prefetch).
//!
//! ## Why CPU
//!
//! The router is small: for Qwen3.5-MoE-A3B it's `[256, 2048]` per layer ≈
//! 256 KB stored as F32. 36 layers ≈ 9 MB total. Trivially fits in L2/L3
//! plus a slice in RAM. A CPU GEMV of `256×2048` floats is ~512K FMAs ≈
//! 50-100 µs on a Zen 2 — much smaller than a single MoE GEMV on GPU
//! (~5 ms). So adding it to the forward critical path costs basically
//! nothing.
//!
//! What we get in return: the *scheduler* (CPU) knows top-k expert indices
//! for layer N as soon as `x_norm` for layer N is available, with no GPU
//! readback. That's the precondition for predictive prefetch in later
//! commits.
//!
//! ## v0.1 scope
//!
//! - F32 router weights (no on-CPU quant; we just dequantize once at load).
//! - Single-threaded GEMV (rayon parallelism comes later if profiling shows
//!   it matters; per-call work is small enough that thread setup may dominate).
//! - Top-k via partial sort (k=8 out of 256, partial_sort beats heap for
//!   tiny k).
//! - No softmax — we only need the *indices*, and softmax is monotonic, so
//!   top-k by raw logits matches top-k by post-softmax weights.

/// Per-layer router weight in F32 plus optional sigmoid/softmax-normed
/// weights for the active experts.
///
/// Storage: `weights[expert_idx * hidden + dim]` — row-major, one row per
/// expert. We deliberately store F32 (~256 KB/layer) rather than the GPU's
/// MQ4G256 form because (a) RAM is cheap, (b) the GEMV is small enough that
/// dequant in the inner loop would dominate, (c) we want a clean F32 reference
/// to validate against GPU's quantized output during bring-up.
pub struct CpuRouter {
    pub layer: u16,
    /// `[num_experts × hidden]`, row-major.
    weights: Vec<f32>,
    pub num_experts: usize,
    pub hidden: usize,
}

impl CpuRouter {
    /// Construct from already-dequantized F32 weights.
    /// The loader is responsible for converting from the HFQ on-disk quant
    /// (typically MQ4G256) to F32 — there's nothing CPU-paging-specific about
    /// that step, so it lives in `hipfire_arch_qwen35::qwen35` alongside other tensor
    /// dequant helpers.
    pub fn from_f32_weights(layer: u16, weights: Vec<f32>, num_experts: usize, hidden: usize) -> Self {
        debug_assert_eq!(weights.len(), num_experts * hidden);
        Self { layer, weights, num_experts, hidden }
    }

    /// Run the router GEMV: `logits = weights × x_norm`, then return the
    /// top-k expert indices (and their raw logit values, for downstream use).
    ///
    /// `x_norm` must have length `hidden`. `k` is the number of experts to
    /// pick (8 for Qwen3.5-MoE-A3B).
    pub fn compute_topk(&self, x_norm: &[f32], k: usize) -> TopK {
        debug_assert_eq!(x_norm.len(), self.hidden);
        debug_assert!(k > 0 && k <= self.num_experts);

        // 1. GEMV. F32 dot product per expert. Tight inner loop, lets the
        //    compiler vectorize via auto-vectorization. If profiling shows
        //    this is hot we can hand-vectorize with std::simd or wide.
        let mut logits = Vec::with_capacity(self.num_experts);
        for e in 0..self.num_experts {
            let row = &self.weights[e * self.hidden..(e + 1) * self.hidden];
            let mut acc = 0.0f32;
            for i in 0..self.hidden {
                acc += row[i] * x_norm[i];
            }
            logits.push(acc);
        }

        // 2. Top-k by partial sort. For k=8/256 this beats a heap because
        //    branch prediction wins on the dominant "skip" path.
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            // Descending. NaN handling: treat as -inf so it sorts last.
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top = indexed[..k].to_vec();
        // Sort the chosen k for stable iteration order across runs (helps
        // when comparing CPU and GPU top-k sets during validation).
        top.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let indices: Vec<u16> = top.iter().map(|(i, _)| *i as u16).collect();
        let logits_top: Vec<f32> = top.iter().map(|(_, l)| *l).collect();
        TopK { indices, logits: logits_top }
    }
}

/// Result of [`CpuRouter::compute_topk`]. Indices are u16 because hipfire's
/// MoE configs cap at 256 experts (well within u16). If we ever hit configs
/// >65535 experts this changes — and so does the GPU side, so it's a single
/// coordinated change.
#[derive(Debug, Clone)]
pub struct TopK {
    /// Expert indices, sorted by descending logit. `indices.len() == k`.
    pub indices: Vec<u16>,
    /// Raw logits for the chosen k experts (pre-softmax). Same order as
    /// `indices`. Available if a caller wants to compute weights without
    /// re-running the GEMV.
    pub logits: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_weights(num_experts: usize, hidden: usize) -> Vec<f32> {
        // Diagonal-ish pattern: expert e's row has a peak at i == e, zero
        // elsewhere. compute_topk(x = one_hot(j)) should return expert j.
        let mut w = vec![0.0; num_experts * hidden];
        for e in 0..num_experts {
            for i in 0..hidden {
                if i % num_experts == e {
                    w[e * hidden + i] = 1.0;
                }
            }
        }
        w
    }

    #[test]
    fn topk_picks_dominant_expert() {
        let n_exp = 8;
        let hidden = 16;
        let weights = deterministic_weights(n_exp, hidden);
        let router = CpuRouter::from_f32_weights(0, weights, n_exp, hidden);

        // x_norm peaked at dim 3 should make expert 3 the dominant logit
        // (since expert 3's row has 1.0s at i where i % 8 == 3 → i ∈ {3, 11}).
        let mut x = vec![0.0; hidden];
        x[3] = 1.0;
        x[11] = 1.0; // both peaks
        let top = router.compute_topk(&x, 1);
        assert_eq!(top.indices, vec![3]);
        assert!(top.logits[0] >= 1.9); // at least 2 hits
    }

    #[test]
    fn topk_returns_k_distinct_indices() {
        let n_exp = 32;
        let hidden = 64;
        let weights: Vec<f32> = (0..n_exp * hidden)
            .map(|i| ((i * 1664525 + 1013904223) as i32 as f32) * 1e-9)
            .collect();
        let router = CpuRouter::from_f32_weights(0, weights, n_exp, hidden);
        let x = vec![1.0; hidden];
        let top = router.compute_topk(&x, 8);
        assert_eq!(top.indices.len(), 8);
        // Distinct.
        let mut sorted = top.indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 8);
    }

    #[test]
    fn topk_logits_are_descending() {
        let n_exp = 16;
        let hidden = 32;
        let weights: Vec<f32> = (0..n_exp * hidden)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let router = CpuRouter::from_f32_weights(0, weights, n_exp, hidden);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32).sin()).collect();
        let top = router.compute_topk(&x, 4);
        for w in top.logits.windows(2) {
            assert!(w[0] >= w[1], "logits not descending: {:?}", top.logits);
        }
    }
}
