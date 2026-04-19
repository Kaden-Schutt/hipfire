//! CASK: core-aware selective KV compression (arXiv:2604.10900).
//!
//! Sits atop TriAttention. TriAttention scores tokens and keeps the top-B;
//! CASK reinterprets the bottom tail as *mergeable scratch* rather than
//! discard, then folds groups of `m` scratch tokens into single
//! representative slots via weighted-average K and V.
//!
//! Paper §2.1: split the cache into protected **core** (top-α·B tokens,
//! never merged) and **mergeable scratch** (next m·(1-α)·B tokens by score,
//! folded into (1-α)·B groups). Output size equals `budget`, same as
//! TriAttention, but effective coverage is α·B + m·(1-α)·B tokens.
//!
//! v1 is Q8-only, L2-grouped, softmax-weighted merge on CPU. asym2/3/4
//! and GPU-side merge are follow-ups — at the cadence we evict, per-layer
//! CPU fold is <1 ms and not a wall-clock factor.
//!
//! Weight choice: `a_i = exp(z_i)` where z_i is the same z-score +
//! max-GQA aggregation TriAttention uses for top-B. The paper offers
//! "score mass, similarity, or position-aware" as options; softmax over
//! score is the score-mass variant with a numerically stable normalizer.

use crate::llama::{f16_to_f32, f32_to_f16, KvCache};
use crate::triattn::{EvictionCtx, EvictionResult};
use hip_bridge::HipResult;
use rdna_compute::Gpu;

/// Hard cap on per-slot fold factor. The per-slot slot table is a fixed
/// stack array of size MAX_FOLD_M to avoid per-eviction heap allocation
/// of m-sized vectors. m=2 and m=4 are typical; m=8 is an upper bound.
const MAX_FOLD_M: usize = 8;

/// Core-Aware Selective KV Compression policy.
///
/// Wraps a TriAttention `EvictionCtx` — scoring, kernels, scratch all
/// reused. Adds only the CASK post-processing step: split retained
/// tokens into core vs scratch, greedy-group scratch by L2-K similarity,
/// fold groups via weighted avg, re-quantize back into the cache.
pub struct CaskCtx {
    pub base: EvictionCtx,
    /// Fraction of budget reserved for singleton core tokens. `core_frac = 1.0`
    /// degenerates to plain TriAttention; `core_frac = 0.0` folds every kept
    /// slot into an m-group.
    pub core_frac: f32,
    /// Merge group size. 2 is the conservative default (1.5× coverage at
    /// α=0.5); 4 gives 2.5× coverage but relies more on within-group
    /// similarity holding up.
    pub fold_m: usize,
}

impl CaskCtx {
    pub fn new(base: EvictionCtx, core_frac: f32, fold_m: usize) -> Self {
        assert!((0.0..=1.0).contains(&core_frac), "core_frac must be in [0, 1]");
        assert!(fold_m >= 2, "fold_m must be >= 2 (use plain TriAttention for m=1)");
        Self { base, core_frac, fold_m }
    }

    pub fn eviction_count(&self) -> usize {
        self.base.eviction_count.get()
    }

    /// Release all GPU buffers held by the underlying EvictionCtx.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.base.free_gpu(gpu);
    }

    /// Same trigger + return convention as `EvictionCtx::maybe_evict`.
    /// Falls back to plain TriAttention for non-Q8 modes in v1.
    ///
    /// Returns an `EvictionResult` with `retain_mask = Vec::new()` on the
    /// m-fold path — merge-smoothed slots don't correspond to any single
    /// source position, so callers (e.g. DFlash draft) that need a retain
    /// selection treat the empty mask as "can't mirror, skip".
    pub fn maybe_evict(
        &self,
        gpu: &mut Gpu,
        kv: &mut KvCache,
        current_physical: usize,
    ) -> HipResult<Option<EvictionResult>> {
        if current_physical < self.base.budget + self.base.beta {
            return Ok(None);
        }

        // Detect KV mode and layout. V is always Q8_0 across modes (the
        // write path only rotates K), so V fold uses kv_fold_q8 uniformly.
        #[derive(Copy, Clone)]
        enum KMode { Q8, Asym3, Asym4, Asym2 }
        let k_mode = if kv.quant_q8 { KMode::Q8 }
            else if kv.quant_asym3 { KMode::Asym3 }
            else if kv.quant_asym4 { KMode::Asym4 }
            else if kv.quant_asym2 { KMode::Asym2 }
            else {
                // Unknown quant (e.g. quant_int8 legacy) — fall back to TriAttn.
                return self.base.maybe_evict(gpu, kv, current_physical);
            };

        let absolute_pos = current_physical + kv.compact_offset;
        let p_q = absolute_pos as f32;

        // Budget math: output always has exactly `budget` slots (c core + s merged,
        // c + s = budget). Input constraint: c + m*s ≤ physical.
        // Solve for max merge_slots s = min(target_merge, (physical-budget)/(m-1)).
        // When physical == budget + beta (threshold), s = beta/(m-1) at m=2 → s=beta.
        let budget = self.base.budget;
        let target_core = (budget as f32 * self.core_frac).floor() as usize;
        let target_merge = budget - target_core;
        let merge_slots = if self.fold_m > 1 && current_physical > budget {
            let max_by_input = (current_physical - budget) / (self.fold_m - 1);
            target_merge.min(max_by_input)
        } else {
            0
        };
        let core_slots = budget - merge_slots;
        let merge_pool = merge_slots * self.fold_m;
        // Sanity: we consume exactly core_slots + merge_pool = budget + merge_slots*(m-1) tokens.
        debug_assert!(core_slots + merge_pool <= current_physical);

        let n_kv = self.base.n_kv_heads;
        let d = self.base.head_dim;
        let n_blocks = d / 32;
        let v_row_bytes = n_kv * n_blocks * 34;   // V is always Q8_0.
        let k_row_bytes = match k_mode {
            KMode::Q8    => v_row_bytes,
            KMode::Asym3 => n_kv * (4 + (d * 3) / 8),
            KMode::Asym4 => n_kv * (4 + d / 2),
            KMode::Asym2 => n_kv * (4 + d / 4),
        };

        // Scratch GPU buffers for per-layer (indices, weights) table. Small
        // (budget × m × 4 B each), allocated once per call. Reusing across
        // layers to avoid repeated allocs.
        let table_len = budget * self.fold_m;
        let indices_dev = gpu.alloc_tensor(&[table_len], rdna_compute::DType::F32)?;
        let weights_dev = gpu.alloc_tensor(&[table_len], rdna_compute::DType::F32)?;

        for (fa_i, &layer_idx) in self.base.fa_layer_ids.iter().enumerate() {
            // 1. TriAttention scoring (GPU), mode-appropriate.
            let offset = fa_i * self.base.centers_per_layer;
            let centers_layer = self.base.centers_dev
                .sub_offset(offset, self.base.centers_per_layer);
            match k_mode {
                KMode::Q8 => gpu.triattn_score_q8(
                    &kv.k_gpu[layer_idx], &centers_layer, &self.base.scores_buf,
                    self.base.n_heads, self.base.n_kv_heads, self.base.head_dim,
                    self.base.n_rot, self.base.rope_theta, p_q, current_physical,
                )?,
                KMode::Asym3 => gpu.triattn_score_asym3(
                    &kv.k_gpu[layer_idx], &centers_layer,
                    kv.givens_cos.as_ref().expect("asym3 KV must have cos table"),
                    kv.givens_sin.as_ref().expect("asym3 KV must have sin table"),
                    &self.base.scores_buf,
                    self.base.n_heads, self.base.n_kv_heads, self.base.head_dim,
                    self.base.n_rot, self.base.rope_theta, p_q, current_physical,
                )?,
                KMode::Asym4 => gpu.triattn_score_asym4(
                    &kv.k_gpu[layer_idx], &centers_layer,
                    kv.givens_cos.as_ref().expect("asym4 KV must have cos table"),
                    kv.givens_sin.as_ref().expect("asym4 KV must have sin table"),
                    &self.base.scores_buf,
                    self.base.n_heads, self.base.n_kv_heads, self.base.head_dim,
                    self.base.n_rot, self.base.rope_theta, p_q, current_physical,
                )?,
                KMode::Asym2 => gpu.triattn_score_asym2(
                    &kv.k_gpu[layer_idx], &centers_layer,
                    kv.givens_cos.as_ref().expect("asym2 KV must have cos table"),
                    kv.givens_sin.as_ref().expect("asym2 KV must have sin table"),
                    &self.base.scores_buf,
                    self.base.n_heads, self.base.n_kv_heads, self.base.head_dim,
                    self.base.n_rot, self.base.rope_theta, p_q, current_physical,
                )?,
            }
            gpu.hip.device_synchronize()?;
            let scores = gpu.download_f32(&self.base.scores_buf)?;

            // 2. Aggregate (CPU, small): per-head z-score, max across heads.
            let agg = aggregate_scores(
                &scores[..self.base.n_heads * current_physical],
                self.base.n_heads, current_physical,
            );

            // 3. Rank tokens; top `core_slots` = core, next `merge_pool` = scratch.
            let mut ranked: Vec<(f32, usize)> =
                agg.iter().copied().enumerate().map(|(i, s)| (s, i)).collect();
            ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let core_ranked = &ranked[..core_slots];
            let scratch_ranked = &ranked[core_slots..core_slots + merge_pool];

            let core_idx: Vec<u32> = core_ranked.iter().map(|(_, i)| *i as u32).collect();
            let scratch_idx: Vec<u32> = scratch_ranked.iter().map(|(_, i)| *i as u32).collect();
            let scratch_scores: Vec<f32> = scratch_ranked.iter().map(|(s, _)| *s).collect();

            // 4. Build per-slot (indices, weights) table. Core slots pad with
            //    weight=0 for unused entries (kernel skips them). Merge slots
            //    hold m source positions with softmax weights.
            //
            //    Slots are sorted by effective position: core = its own pos,
            //    merge = weighted-centroid of group. Gives temporal order in
            //    the compacted cache.
            let mut entries: Vec<(u32, [(u32, f32); MAX_FOLD_M])> = Vec::with_capacity(budget);

            for &pos in &core_idx {
                let mut slot = [(0u32, 0.0f32); MAX_FOLD_M];
                slot[0] = (pos, 1.0);
                for s in &mut slot[1..self.fold_m] { s.0 = pos; }  // indices safe; weights 0
                entries.push((pos, slot));
            }

            if merge_slots > 0 {
                // For Q8, group by L2-K similarity (K rows downloaded for
                // dequant-feature extraction). For asym modes, fall back to
                // rank-based pairing (consecutive scratch tokens by score)
                // — simpler, no per-mode CPU dequant path. L2 grouping for
                // asym is a follow-up once the simple version is validated.
                let groups: Vec<Vec<usize>> = match k_mode {
                    KMode::Q8 => {
                        let mut k_all = vec![0u8; current_physical * k_row_bytes];
                        gpu.hip.memcpy_dtoh(&mut k_all, &kv.k_gpu[layer_idx].buf)?;
                        greedy_group_by_l2(&k_all, &scratch_idx, n_kv, d, self.fold_m)
                    }
                    _ => {
                        // Rank-based: pair scratch tokens in score-sorted
                        // order (already given by scratch_idx's order in
                        // this loop). Consecutive-m groups.
                        let n = scratch_idx.len();
                        (0..n).step_by(self.fold_m)
                            .filter(|&start| start + self.fold_m <= n)
                            .map(|start| (start..start + self.fold_m).collect())
                            .collect()
                    }
                };
                for group in &groups {
                    let abs_positions: Vec<u32> = group.iter().map(|&gi| scratch_idx[gi]).collect();
                    let raw_scores: Vec<f32> = group.iter().map(|&gi| scratch_scores[gi]).collect();
                    let weights = softmax(&raw_scores);

                    let centroid: f32 = abs_positions.iter().zip(weights.iter())
                        .map(|(&p, &w)| p as f32 * w).sum();
                    let mut slot = [(0u32, 0.0f32); MAX_FOLD_M];
                    for i in 0..self.fold_m {
                        slot[i] = (abs_positions[i], weights[i]);
                    }
                    entries.push((centroid as u32, slot));
                }
            }

            entries.sort_by_key(|&(c, _)| c);

            // Flatten to two arrays: [budget × m] each.
            let mut flat_indices = Vec::with_capacity(table_len);
            let mut flat_weights = Vec::with_capacity(table_len);
            for (_, slot) in &entries {
                for i in 0..self.fold_m {
                    flat_indices.push(slot[i].0 as i32);
                    flat_weights.push(slot[i].1);
                }
            }

            // 5. Upload table and run the GPU fold kernel for K and V.
            //    src → k_compact/v_compact (scratch on EvictionCtx), then
            //    memcpy back into the cache (matching TriAttn pattern).
            let idx_bytes: Vec<u8> = flat_indices.iter()
                .flat_map(|&x| x.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&indices_dev.buf, &idx_bytes)?;
            let w_bytes: Vec<u8> = flat_weights.iter()
                .flat_map(|&x| x.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&weights_dev.buf, &w_bytes)?;

            // K fold uses the mode-specific kernel. V is always Q8_0
            // (rotation is K-only in RotorQuant), so V always uses kv_fold_q8.
            match k_mode {
                KMode::Q8 => gpu.kv_fold_q8(
                    &kv.k_gpu[layer_idx], &self.base.k_compact,
                    &indices_dev, &weights_dev,
                    n_kv, n_blocks, self.fold_m, budget,
                )?,
                KMode::Asym3 => gpu.kv_fold_asym3(
                    &kv.k_gpu[layer_idx], &self.base.k_compact,
                    &indices_dev, &weights_dev,
                    n_kv, d, self.fold_m, budget,
                )?,
                KMode::Asym4 => gpu.kv_fold_asym4(
                    &kv.k_gpu[layer_idx], &self.base.k_compact,
                    &indices_dev, &weights_dev,
                    n_kv, d, self.fold_m, budget,
                )?,
                KMode::Asym2 => gpu.kv_fold_asym2(
                    &kv.k_gpu[layer_idx], &self.base.k_compact,
                    &indices_dev, &weights_dev,
                    n_kv, d, self.fold_m, budget,
                )?,
            }
            gpu.kv_fold_q8(
                &kv.v_gpu[layer_idx], &self.base.v_compact,
                &indices_dev, &weights_dev,
                n_kv, n_blocks, self.fold_m, budget,
            )?;
            gpu.hip.device_synchronize()?;

            gpu.hip.memcpy_dtod_at(
                &kv.k_gpu[layer_idx].buf, 0,
                &self.base.k_compact.buf, 0,
                budget * k_row_bytes,
            )?;
            gpu.hip.memcpy_dtod_at(
                &kv.v_gpu[layer_idx].buf, 0,
                &self.base.v_compact.buf, 0,
                budget * v_row_bytes,
            )?;
        }

        // Output size is always `budget` slots (core_slots + merge_slots = budget).
        kv.compact_offset += current_physical - budget;
        self.base.eviction_count.set(self.base.eviction_count.get() + 1);
        // m-fold merges multiple source positions per output slot — no single
        // retain_mask captures the mapping. Empty mask signals "can't mirror"
        // to callers that shadow the eviction into non-KV buffers.
        Ok(Some(EvictionResult { new_physical: budget, retain_mask: Vec::new() }))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Per-head z-score, then max across heads. Same aggregation TriAttention
/// uses for top-B but returned directly so we can rank-by-score.
pub fn aggregate_scores(scores: &[f32], n_heads: usize, seq_len: usize) -> Vec<f32> {
    let mut agg = vec![f32::NEG_INFINITY; seq_len];
    for h in 0..n_heads {
        let row = &scores[h * seq_len..(h + 1) * seq_len];
        let mean: f32 = row.iter().sum::<f32>() / seq_len as f32;
        let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
            / seq_len as f32;
        let std = var.sqrt().max(1e-6);
        for p in 0..seq_len {
            let z = (row[p] - mean) / std;
            if z > agg[p] { agg[p] = z; }
        }
    }
    agg
}

/// Stable softmax over a small vector.
pub fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Greedy L2-NN grouping: process scratch tokens in input order (which is
/// arbitrary — caller passes scratch sorted by score desc). For each
/// ungrouped anchor, pick the (m-1) nearest-L2 ungrouped neighbors in
/// dequantized K space. Returns groups as indices into the input array.
pub fn greedy_group_by_l2(
    k_all: &[u8],
    scratch_idx: &[u32],
    n_kv: usize,
    head_dim: usize,
    m: usize,
) -> Vec<Vec<usize>> {
    let n = scratch_idx.len();
    let n_blocks = head_dim / 32;
    let row_bytes = n_kv * n_blocks * 34;
    let feat_dim = n_kv * head_dim;

    let mut feats = vec![0f32; n * feat_dim];
    for (i, &pos) in scratch_idx.iter().enumerate() {
        let row = &k_all[pos as usize * row_bytes..(pos as usize + 1) * row_bytes];
        dequant_q8_row(row, &mut feats[i * feat_dim..(i + 1) * feat_dim], n_kv, head_dim);
    }

    let mut used = vec![false; n];
    let n_groups = n / m;
    let mut groups = Vec::with_capacity(n_groups);

    for anchor in 0..n {
        if used[anchor] || groups.len() == n_groups { continue; }
        used[anchor] = true;
        let mut group = vec![anchor];
        let anchor_feat = &feats[anchor * feat_dim..(anchor + 1) * feat_dim];

        // Collect distances to all unused candidates.
        let mut cands: Vec<(f32, usize)> = (0..n)
            .filter(|&j| !used[j])
            .map(|j| {
                let fj = &feats[j * feat_dim..(j + 1) * feat_dim];
                let d2: f32 = anchor_feat.iter().zip(fj)
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                (d2, j)
            })
            .collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for &(_, j) in cands.iter().take(m - 1) {
            used[j] = true;
            group.push(j);
        }
        if group.len() == m {
            groups.push(group);
        }
    }
    groups
}

/// Dequantize a single Q8_0 row to flat f32. `out.len() == n_kv * head_dim`.
pub fn dequant_q8_row(row: &[u8], out: &mut [f32], n_kv: usize, head_dim: usize) {
    let n_blocks = head_dim / 32;
    for h in 0..n_kv {
        for b in 0..n_blocks {
            let block_off = (h * n_blocks + b) * 34;
            let scale = f16_to_f32(u16::from_le_bytes([row[block_off], row[block_off + 1]]));
            for q in 0..32 {
                let v = row[block_off + 2 + q] as i8;
                out[h * head_dim + b * 32 + q] = scale * (v as f32);
            }
        }
    }
}

/// Weighted average of multiple Q8_0 rows, requantized per-block.
/// Output layout matches a single Q8_0 row: `n_kv * n_blocks * 34` bytes.
pub fn weighted_avg_q8(
    all_rows: &[u8],
    positions: &[u32],
    weights: &[f32],
    n_kv: usize,
    head_dim: usize,
) -> Vec<u8> {
    assert_eq!(positions.len(), weights.len());
    let n_blocks = head_dim / 32;
    let row_bytes = n_kv * n_blocks * 34;
    let feat_dim = n_kv * head_dim;

    let mut acc = vec![0f32; feat_dim];
    let mut deq = vec![0f32; feat_dim];
    for (i, &pos) in positions.iter().enumerate() {
        let w = weights[i];
        let row = &all_rows[pos as usize * row_bytes..(pos as usize + 1) * row_bytes];
        dequant_q8_row(row, &mut deq, n_kv, head_dim);
        for j in 0..feat_dim {
            acc[j] += w * deq[j];
        }
    }

    let mut out = vec![0u8; row_bytes];
    for h in 0..n_kv {
        for b in 0..n_blocks {
            let block_off = (h * n_blocks + b) * 34;
            let slice = &acc[h * head_dim + b * 32..h * head_dim + b * 32 + 32];
            let max_abs = slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 127.0;
            let inv_scale = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
            let s16 = f32_to_f16(scale);
            out[block_off] = (s16 & 0xFF) as u8;
            out[block_off + 1] = (s16 >> 8) as u8;
            for q in 0..32 {
                let qv = (slice[q] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                out[block_off + 2 + q] = qv as u8;
            }
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_q8_row(vals: &[f32], n_kv: usize, head_dim: usize) -> Vec<u8> {
        let n_blocks = head_dim / 32;
        let row_bytes = n_kv * n_blocks * 34;
        let mut out = vec![0u8; row_bytes];
        for h in 0..n_kv {
            for b in 0..n_blocks {
                let block_off = (h * n_blocks + b) * 34;
                let slice = &vals[h * head_dim + b * 32..h * head_dim + b * 32 + 32];
                let max_abs = slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 127.0;
                let inv = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
                let s16 = f32_to_f16(scale);
                out[block_off] = (s16 & 0xFF) as u8;
                out[block_off + 1] = (s16 >> 8) as u8;
                for q in 0..32 {
                    let qv = (slice[q] * inv).round().clamp(-127.0, 127.0) as i8;
                    out[block_off + 2 + q] = qv as u8;
                }
            }
        }
        out
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn softmax_numerical_stability_large_inputs() {
        let s = softmax(&[1000.0, 1001.0, 1002.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax overflowed on large z");
    }

    #[test]
    fn aggregate_z_score_max_gqa() {
        // 2 heads, 4 positions. Head 0: [1,2,3,4], Head 1: [4,3,2,1].
        // Per-head z-score then max-across-heads should give position 0 and 3
        // the highest aggregate (extremes on one head each).
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let agg = aggregate_scores(&scores, 2, 4);
        assert!(agg[0] > agg[1]);
        assert!(agg[3] > agg[2]);
        assert!((agg[0] - agg[3]).abs() < 1e-5, "symmetric case, endpoints should tie");
    }

    #[test]
    fn dequant_requant_q8_near_exact() {
        let n_kv = 2;
        let head_dim = 32;
        let vals: Vec<f32> = (0..n_kv * head_dim)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let packed = pack_q8_row(&vals, n_kv, head_dim);
        let mut back = vec![0f32; n_kv * head_dim];
        dequant_q8_row(&packed, &mut back, n_kv, head_dim);
        for i in 0..vals.len() {
            assert!((vals[i] - back[i]).abs() < 0.02, "dequant drift at {}: {} vs {}", i, vals[i], back[i]);
        }
    }

    #[test]
    fn weighted_avg_of_identical_rows_is_identity() {
        let n_kv = 2;
        let head_dim = 32;
        let vals: Vec<f32> = (0..n_kv * head_dim).map(|i| (i as f32) * 0.05).collect();
        let row = pack_q8_row(&vals, n_kv, head_dim);
        let mut all = Vec::new();
        for _ in 0..3 { all.extend_from_slice(&row); }

        let merged = weighted_avg_q8(&all, &[0, 1, 2], &[0.2, 0.3, 0.5], n_kv, head_dim);
        let mut back = vec![0f32; n_kv * head_dim];
        dequant_q8_row(&merged, &mut back, n_kv, head_dim);
        for i in 0..vals.len() {
            assert!((vals[i] - back[i]).abs() < 0.02, "identity merge drift at {}", i);
        }
    }

    #[test]
    fn weighted_avg_of_two_orthogonal_is_blend() {
        let n_kv = 1;
        let head_dim = 32;
        let a: Vec<f32> = (0..head_dim).map(|i| if i < 16 { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = (0..head_dim).map(|i| if i >= 16 { 1.0 } else { 0.0 }).collect();
        let mut all = pack_q8_row(&a, n_kv, head_dim);
        all.extend_from_slice(&pack_q8_row(&b, n_kv, head_dim));

        // Equal weights: merged ≈ [0.5, 0.5, …]
        let merged = weighted_avg_q8(&all, &[0, 1], &[0.5, 0.5], n_kv, head_dim);
        let mut back = vec![0f32; head_dim];
        dequant_q8_row(&merged, &mut back, n_kv, head_dim);
        for i in 0..head_dim {
            assert!((back[i] - 0.5).abs() < 0.05, "blend drift at {}: {}", i, back[i]);
        }
    }

    #[test]
    fn greedy_group_pairs_nearby_tokens() {
        // Two clusters: tokens 0,1 similar; tokens 2,3 similar (different cluster).
        // Expect groups {0,1} and {2,3} (or swap), never {0,2} or {0,3}.
        let n_kv = 1;
        let head_dim = 32;
        let t0: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin() * 0.5).collect();
        let mut t1 = t0.clone();
        t1[0] += 0.01; // near-duplicate
        let t2: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 1.3).cos() * 0.5).collect();
        let mut t3 = t2.clone();
        t3[0] += 0.01;

        let mut all = Vec::new();
        for v in [&t0, &t1, &t2, &t3] {
            all.extend_from_slice(&pack_q8_row(v, n_kv, head_dim));
        }

        let scratch_idx: Vec<u32> = vec![0, 1, 2, 3];
        let groups = greedy_group_by_l2(&all, &scratch_idx, n_kv, head_dim, 2);
        assert_eq!(groups.len(), 2);
        // Each group should have indices both from cluster A (0,1) OR both from B (2,3).
        for g in &groups {
            let all_low = g.iter().all(|&i| i < 2);
            let all_high = g.iter().all(|&i| i >= 2);
            assert!(all_low || all_high, "group crossed clusters: {:?}", g);
        }
    }
}
