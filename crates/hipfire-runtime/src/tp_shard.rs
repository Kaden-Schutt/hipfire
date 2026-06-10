// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor-parallel (TP) shard configuration.
//!
//! `ShardConfig` is the pure-CPU description of how a single model copy is
//! split across `tp_size` ranks. It owns no GPU state — it answers "which
//! Q heads / KV heads / experts / weight-matrix sub-ranges does rank `r`
//! own?" so the weight loader (`load_weights_tp`) and the sharded forward
//! path (`forward_scratch_tp`) can slice consistently.
//!
//! See `docs/plans/multi-gpu-tp-a3b.md` §3 for the sharding axes and
//! `docs/investigations/2026-05-28-tp-comm-baseline-hiptrx.md` for why the
//! comm path is RCCL-backed.
//!
//! ## Sharding axes (Megatron convention)
//!
//! - **`wq` (column-shard, "tensor-parallel column"):** Qwen3.5/3.6 store
//!   `wq` as `[n_heads * head_dim * 2, dim]` — Q and an output **gate**
//!   interleaved per head as `[Q_h0(hd), Gate_h0(hd), Q_h1(hd), ...]`
//!   (see `kernels/src/deinterleave.hip`). Rank `r` owns a contiguous block
//!   of `q_heads_per_rank` heads, i.e. rows
//!   `[r·hpr·2·head_dim, (r+1)·hpr·2·head_dim)`. Slicing along **rows** of a
//!   row-major quant blob is contiguous, and the `2·head_dim` block keeps
//!   each head's Q-slice and gate-slice together — a naive `head_dim`-stride
//!   split would corrupt the gate↔Q correspondence.
//! - **`wk`/`wv`:** replicated when `tp_kv_replicate` (TP=4 on A3B, since
//!   `n_kv_heads=2 < tp=4`), else clean GQA split (TP=2 → 1 KV head/rank).
//! - **`wo` (row-shard, "tensor-parallel row"):** `[dim, n_heads·head_dim]`,
//!   sharded along the **input** dim (columns) so each rank consumes only its
//!   local heads' attention output and produces a *partial* residual, which
//!   is then `all_reduce_sum`'d across ranks. NB: slicing `wo` along columns
//!   of a row-major quant matrix is a per-row gather, not a contiguous byte
//!   range — see the loader notes in the TP plan §5 (Stage 3 milestone 3b).
//! - **Routed experts:** rank owns `n_experts / tp_size` experts per
//!   `expert_to_rank` (Stage 5).

use std::ops::Range;

/// Routed-expert → rank assignment policy (A3B MoE, Stage 5).
///
/// `Stride` (default) load-balances better when top-k draws cluster on hot
/// indices; `Contiguous` keeps expert blocks together. See TP plan §3.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertAssign {
    /// Rank `r` owns experts `[r·per, (r+1)·per)`.
    Contiguous,
    /// Expert `e` → rank `e % tp_size`.
    Stride,
}

impl ExpertAssign {
    /// Resolve from `HIPFIRE_TP_EXPERT_ASSIGN` (`contiguous` | `stride`).
    /// Default `Stride` per TP plan §3.6.
    pub fn from_env() -> Self {
        match std::env::var("HIPFIRE_TP_EXPERT_ASSIGN").ok().as_deref() {
            Some("contiguous") | Some("block") => ExpertAssign::Contiguous,
            _ => ExpertAssign::Stride,
        }
    }
}

/// Pure-CPU TP shard descriptor. Cheap to clone; carries no GPU handles.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of TP ranks this model copy is split across (`>= 1`).
    pub tp_size: usize,
    /// When `true`, every rank holds all `n_kv_heads` KV heads (required
    /// when `tp_size > n_kv_heads`, e.g. TP=4 on A3B). When `false`, KV
    /// heads split evenly (clean GQA, e.g. TP=2 → 1 KV head/rank).
    pub tp_kv_replicate: bool,
    /// Length `n_experts`; `expert_to_rank[e]` is the owning rank. Empty for
    /// dense models (`n_experts == 0`).
    pub expert_to_rank: Vec<u8>,
}

impl ShardConfig {
    /// TP=1 degenerate config — byte-identical to the single-card path.
    pub fn single() -> Self {
        Self {
            tp_size: 1,
            tp_kv_replicate: false,
            expert_to_rank: Vec::new(),
        }
    }

    /// Build a shard config for `tp_size` ranks.
    ///
    /// `n_experts == 0` yields an empty `expert_to_rank` (dense model).
    /// Errors if `tp_size == 0`, or `n_experts` is non-zero and not divisible
    /// by `tp_size` (v1 requires balanced expert blocks — see TP plan §3.6).
    pub fn new(
        tp_size: usize,
        tp_kv_replicate: bool,
        n_experts: usize,
        assign: ExpertAssign,
    ) -> Result<Self, String> {
        if tp_size == 0 {
            return Err("ShardConfig: tp_size must be >= 1".to_string());
        }
        let expert_to_rank = if n_experts == 0 {
            Vec::new()
        } else {
            if n_experts % tp_size != 0 {
                return Err(format!(
                    "ShardConfig: n_experts ({n_experts}) not divisible by tp_size \
                     ({tp_size}); v1 requires balanced expert blocks"
                ));
            }
            let per = n_experts / tp_size;
            (0..n_experts)
                .map(|e| {
                    let rank = match assign {
                        ExpertAssign::Contiguous => e / per,
                        ExpertAssign::Stride => e % tp_size,
                    };
                    rank as u8
                })
                .collect()
        };
        Ok(Self {
            tp_size,
            tp_kv_replicate,
            expert_to_rank,
        })
    }

    /// True for the single-rank degenerate case.
    #[inline]
    pub fn is_single(&self) -> bool {
        self.tp_size == 1
    }

    /// Validate the head geometry against this shard config. Call at model
    /// load once `n_heads`/`n_kv_heads` are known.
    ///
    /// - `n_heads` must divide evenly across ranks (Q-head column shard).
    /// - When `!tp_kv_replicate`, `n_kv_heads` must divide evenly and
    ///   `tp_size <= n_kv_heads` (clean GQA split). When replicated, any
    ///   `n_kv_heads >= 1` is fine.
    pub fn validate(&self, n_heads: usize, n_kv_heads: usize) -> Result<(), String> {
        if n_heads % self.tp_size != 0 {
            return Err(format!(
                "ShardConfig::validate: n_heads ({n_heads}) not divisible by tp_size ({})",
                self.tp_size
            ));
        }
        if !self.tp_kv_replicate {
            if n_kv_heads == 0 || n_kv_heads % self.tp_size != 0 {
                return Err(format!(
                    "ShardConfig::validate: n_kv_heads ({n_kv_heads}) not divisible by \
                     tp_size ({}) and tp_kv_replicate=false — set tp_kv_replicate=true \
                     (default on a 4-card box) or pick a tp_size that divides n_kv_heads",
                    self.tp_size
                ));
            }
        }
        Ok(())
    }

    // ── Attention head ranges ──────────────────────────────────────────

    /// Q heads owned per rank (`n_heads / tp_size`). Caller must have
    /// `validate`d divisibility.
    #[inline]
    pub fn q_heads_per_rank(&self, n_heads: usize) -> usize {
        n_heads / self.tp_size
    }

    /// KV heads present per rank: all of them when replicated, else split.
    #[inline]
    pub fn kv_heads_per_rank(&self, n_kv_heads: usize) -> usize {
        if self.tp_kv_replicate {
            n_kv_heads
        } else {
            n_kv_heads / self.tp_size
        }
    }

    /// Half-open Q-head range owned by `rank`.
    #[inline]
    pub fn q_head_range(&self, rank: usize, n_heads: usize) -> Range<usize> {
        let hpr = self.q_heads_per_rank(n_heads);
        (rank * hpr)..((rank + 1) * hpr)
    }

    /// Half-open KV-head range present on `rank`. Full set when replicated.
    #[inline]
    pub fn kv_head_range(&self, rank: usize, n_kv_heads: usize) -> Range<usize> {
        if self.tp_kv_replicate {
            0..n_kv_heads
        } else {
            let kpr = n_kv_heads / self.tp_size;
            (rank * kpr)..((rank + 1) * kpr)
        }
    }

    // ── DeltaNet (LinearAttention) head ranges (Stage 3c) ──────────────
    //
    // DeltaNet shards by VALUE head; KEY/QUERY heads follow the GQA
    // repeat-interleave ratio `n_value_heads / n_key_heads`. The wqkv output
    // is `[q(k_dim) | k(k_dim) | v(v_dim)]` so the column shard takes local
    // KEY heads from the q+k blocks and local VALUE heads from the v block.

    /// Validate DeltaNet head geometry: both head counts split evenly and the
    /// value/key ratio is preserved per rank (so GQA repeat-interleave still
    /// works on the local shard). Call once at load.
    pub fn validate_deltanet(&self, n_value_heads: usize, n_key_heads: usize) -> Result<(), String> {
        if n_value_heads % self.tp_size != 0 {
            return Err(format!(
                "validate_deltanet: linear_num_value_heads ({n_value_heads}) not divisible by tp_size ({})",
                self.tp_size
            ));
        }
        if n_key_heads == 0 || n_key_heads % self.tp_size != 0 {
            return Err(format!(
                "validate_deltanet: linear_num_key_heads ({n_key_heads}) not divisible by tp_size ({})",
                self.tp_size
            ));
        }
        // Ratio preserved: (n_value/tp) / (n_key/tp) == n_value / n_key.
        if n_value_heads % n_key_heads != 0 {
            return Err(format!(
                "validate_deltanet: n_value_heads ({n_value_heads}) not a multiple of \
                 n_key_heads ({n_key_heads}) — GQA ratio undefined"
            ));
        }
        let ratio = n_value_heads / n_key_heads;
        let local_ratio = (n_value_heads / self.tp_size) / (n_key_heads / self.tp_size);
        if local_ratio != ratio {
            return Err(format!(
                "validate_deltanet: per-rank value/key ratio {local_ratio} != global {ratio} \
                 (tp_size={} splits the GQA group)",
                self.tp_size
            ));
        }
        Ok(())
    }

    /// Value heads owned per rank (`n_value_heads / tp_size`).
    #[inline]
    pub fn dn_value_heads_per_rank(&self, n_value_heads: usize) -> usize {
        n_value_heads / self.tp_size
    }

    /// Key heads owned per rank (`n_key_heads / tp_size`).
    #[inline]
    pub fn dn_key_heads_per_rank(&self, n_key_heads: usize) -> usize {
        n_key_heads / self.tp_size
    }

    /// Half-open VALUE-head range owned by `rank`.
    #[inline]
    pub fn dn_value_head_range(&self, rank: usize, n_value_heads: usize) -> Range<usize> {
        let vpr = n_value_heads / self.tp_size;
        (rank * vpr)..((rank + 1) * vpr)
    }

    /// Half-open KEY-head range owned by `rank` (q + k blocks of wqkv).
    #[inline]
    pub fn dn_key_head_range(&self, rank: usize, n_key_heads: usize) -> Range<usize> {
        let kpr = n_key_heads / self.tp_size;
        (rank * kpr)..((rank + 1) * kpr)
    }

    // ── Expert parallelism (Stage 3e — EP-MoE) ─────────────────────────
    //
    // All-reduce EP: each rank loads + computes ONLY its owned experts (see
    // `owns_expert` / `experts_on_rank` below, Stage 5) into a `[N×dim]`
    // partial, then the partials are all-reduced. See `docs/plans/tp-3e-ep-moe.md`.

    /// Routed experts owned per rank (`n_experts / tp_size`). Dense model
    /// (`expert_to_rank` empty) → `n_experts` (single owner).
    #[inline]
    pub fn experts_per_rank(&self, n_experts: usize) -> usize {
        if self.expert_to_rank.is_empty() {
            n_experts
        } else {
            n_experts / self.tp_size
        }
    }

    /// Validate the MoE expert split: `n_experts` must be divisible by
    /// `tp_size` (v1 requires balanced expert blocks — `new` already enforces
    /// this when constructed with `n_experts`; this re-checks at the call site).
    pub fn validate_moe(&self, n_experts: usize) -> Result<(), String> {
        if n_experts % self.tp_size != 0 {
            return Err(format!(
                "validate_moe: n_experts ({n_experts}) not divisible by tp_size ({})",
                self.tp_size
            ));
        }
        Ok(())
    }

    // ── Weight-matrix sub-ranges (row-major quant blobs) ───────────────

    /// Row range of the gated `wq` (`[n_heads·head_dim·2, dim]`) owned by
    /// `rank`. Each Q head occupies a contiguous `2·head_dim` block
    /// (interleaved Q+gate), so the range is head-block aligned and keeps
    /// each head's Q and gate slices together. Slicing along rows of a
    /// row-major quant matrix is contiguous → cheap to load per rank.
    #[inline]
    pub fn wq_row_range(&self, rank: usize, n_heads: usize, head_dim: usize) -> Range<usize> {
        let hpr = self.q_heads_per_rank(n_heads);
        let block = 2 * head_dim; // Q(hd) + gate(hd) per head
        (rank * hpr * block)..((rank + 1) * hpr * block)
    }

    /// Column range of `wo` (`[dim, n_heads·head_dim]`) consumed by `rank`
    /// (one `head_dim` block per local Q head). NB: this slices the **input**
    /// dim, which is a per-row gather for a row-major quant matrix — not a
    /// contiguous byte range. The forward path produces a partial residual
    /// over this range and `all_reduce_sum`s across ranks.
    #[inline]
    pub fn wo_col_range(&self, rank: usize, n_heads: usize, head_dim: usize) -> Range<usize> {
        let hpr = self.q_heads_per_rank(n_heads);
        (rank * hpr * head_dim)..((rank + 1) * hpr * head_dim)
    }

    // ── Routed-expert ownership (Stage 5) ──────────────────────────────

    /// Whether `rank` owns routed expert `e`.
    #[inline]
    pub fn owns_expert(&self, rank: usize, e: usize) -> bool {
        self.expert_to_rank
            .get(e)
            .is_some_and(|&r| r as usize == rank)
    }

    /// Routed experts owned by `rank`, ascending.
    pub fn experts_on_rank(&self, rank: usize) -> Vec<usize> {
        self.expert_to_rank
            .iter()
            .enumerate()
            .filter_map(|(e, &r)| (r as usize == rank).then_some(e))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_is_degenerate() {
        let s = ShardConfig::single();
        assert!(s.is_single());
        assert_eq!(s.tp_size, 1);
        assert!(s.expert_to_rank.is_empty());
        // Whole-tensor ranges on rank 0.
        assert_eq!(s.q_head_range(0, 8), 0..8);
        assert_eq!(s.wq_row_range(0, 8, 256), 0..(8 * 512));
        assert_eq!(s.wo_col_range(0, 8, 256), 0..(8 * 256));
    }

    #[test]
    fn rejects_zero_tp() {
        assert!(ShardConfig::new(0, false, 0, ExpertAssign::Stride).is_err());
    }

    #[test]
    fn dense_has_no_experts() {
        let s = ShardConfig::new(2, false, 0, ExpertAssign::Stride).unwrap();
        assert!(s.expert_to_rank.is_empty());
        assert!(s.experts_on_rank(0).is_empty());
    }

    #[test]
    fn expert_assign_non_divisible_errors() {
        // 256 experts across tp=3 is not balanced.
        assert!(ShardConfig::new(3, true, 256, ExpertAssign::Stride).is_err());
    }

    #[test]
    fn expert_assign_contiguous_a3b_tp4() {
        let s = ShardConfig::new(4, true, 256, ExpertAssign::Contiguous).unwrap();
        assert_eq!(s.expert_to_rank.len(), 256);
        // Rank r owns [r*64, (r+1)*64).
        assert_eq!(s.expert_to_rank[0], 0);
        assert_eq!(s.expert_to_rank[63], 0);
        assert_eq!(s.expert_to_rank[64], 1);
        assert_eq!(s.expert_to_rank[255], 3);
        assert_eq!(s.experts_on_rank(1), (64..128).collect::<Vec<_>>());
        assert!(s.owns_expert(2, 130));
        assert!(!s.owns_expert(2, 64));
    }

    #[test]
    fn expert_assign_stride_a3b_tp4() {
        let s = ShardConfig::new(4, true, 256, ExpertAssign::Stride).unwrap();
        // Expert e -> rank e % 4.
        assert_eq!(s.expert_to_rank[0], 0);
        assert_eq!(s.expert_to_rank[1], 1);
        assert_eq!(s.expert_to_rank[5], 1);
        assert_eq!(s.expert_to_rank[255], 3);
        // Each rank owns exactly 64, strided.
        let r1 = s.experts_on_rank(1);
        assert_eq!(r1.len(), 64);
        assert_eq!(r1[0], 1);
        assert_eq!(r1[1], 5);
    }

    #[test]
    fn validate_q_head_divisibility() {
        let s = ShardConfig::new(2, false, 0, ExpertAssign::Stride).unwrap();
        // n_heads=8 splits, n_kv_heads=2 splits cleanly at tp=2.
        assert!(s.validate(8, 2).is_ok());
        // n_heads=7 not divisible by 2.
        assert!(s.validate(7, 2).is_err());
    }

    #[test]
    fn validate_kv_replicate_vs_split() {
        // TP=4, n_kv_heads=2: replicate required (4 > 2).
        let rep = ShardConfig::new(4, true, 0, ExpertAssign::Stride).unwrap();
        assert!(rep.validate(8, 2).is_ok());
        assert_eq!(rep.kv_heads_per_rank(2), 2); // all KV heads on every rank
        assert_eq!(rep.kv_head_range(3, 2), 0..2);

        // TP=4, n_kv_heads=2, no replicate: must fail (can't split 2 four ways).
        let split = ShardConfig::new(4, false, 0, ExpertAssign::Stride).unwrap();
        assert!(split.validate(8, 2).is_err());
    }

    #[test]
    fn attn_ranges_tp2() {
        // 0.8B-ish geometry: n_heads=8, n_kv_heads=2, head_dim=256, tp=2.
        let s = ShardConfig::new(2, false, 0, ExpertAssign::Stride).unwrap();
        s.validate(8, 2).unwrap();
        assert_eq!(s.q_heads_per_rank(8), 4);
        assert_eq!(s.q_head_range(0, 8), 0..4);
        assert_eq!(s.q_head_range(1, 8), 4..8);
        // KV clean split: 1 head/rank.
        assert_eq!(s.kv_heads_per_rank(2), 1);
        assert_eq!(s.kv_head_range(0, 2), 0..1);
        assert_eq!(s.kv_head_range(1, 2), 1..2);
        // wq rows: 4 heads × 2 × 256 = 2048 rows/rank.
        assert_eq!(s.wq_row_range(0, 8, 256), 0..2048);
        assert_eq!(s.wq_row_range(1, 8, 256), 2048..4096);
        // wo cols: 4 heads × 256 = 1024 cols/rank.
        assert_eq!(s.wo_col_range(0, 8, 256), 0..1024);
        assert_eq!(s.wo_col_range(1, 8, 256), 1024..2048);
    }

    #[test]
    fn deltanet_ranges_27b_tp2() {
        // 27B-3.6: linear_num_key_heads=16, linear_num_value_heads=48 (ratio 3).
        let s = ShardConfig::new(2, false, 0, ExpertAssign::Stride).unwrap();
        s.validate_deltanet(48, 16).unwrap();
        assert_eq!(s.dn_value_heads_per_rank(48), 24);
        assert_eq!(s.dn_key_heads_per_rank(16), 8);
        assert_eq!(s.dn_value_head_range(0, 48), 0..24);
        assert_eq!(s.dn_value_head_range(1, 48), 24..48);
        assert_eq!(s.dn_key_head_range(0, 16), 0..8);
        assert_eq!(s.dn_key_head_range(1, 16), 8..16);
        // ratio preserved: 24/8 == 48/16 == 3.
    }

    #[test]
    fn deltanet_ranges_tp4_and_08b() {
        // 27B tp=4: 12 value + 4 key/rank, ratio 3 preserved.
        let s4 = ShardConfig::new(4, true, 0, ExpertAssign::Stride).unwrap();
        s4.validate_deltanet(48, 16).unwrap();
        assert_eq!(s4.dn_value_heads_per_rank(48), 12);
        assert_eq!(s4.dn_key_heads_per_rank(16), 4);
        // 0.8B: 16 value + 16 key (ratio 1), tp=2 → 8+8.
        let s2 = ShardConfig::new(2, false, 0, ExpertAssign::Stride).unwrap();
        s2.validate_deltanet(16, 16).unwrap();
        assert_eq!(s2.dn_value_heads_per_rank(16), 8);
        assert_eq!(s2.dn_key_heads_per_rank(16), 8);
    }

    #[test]
    fn deltanet_validate_rejects_split_gqa_group() {
        // Hypothetical: n_value=48, n_key=16, tp=8 → 6 value + 2 key (ratio 3 ok),
        // but tp=16 would give 3 value + 1 key (ratio 3 ok too). A bad case:
        // n_value=4, n_key=4, tp=2 → 2+2 ratio 1 ok. Force a ratio break:
        // n_value=6, n_key=4 is not a clean multiple → undefined ratio.
        let s = ShardConfig::new(2, false, 0, ExpertAssign::Stride).unwrap();
        assert!(s.validate_deltanet(6, 4).is_err()); // 6 % 4 != 0
        assert!(s.validate_deltanet(48, 3).is_err()); // 3 not divisible by tp=2
    }

    #[test]
    fn wq_blocks_partition_full_matrix() {
        // The union of per-rank wq row ranges must exactly tile the full
        // [n_heads·head_dim·2] row space with no gaps/overlap.
        let s = ShardConfig::new(4, true, 0, ExpertAssign::Stride).unwrap();
        let (n_heads, head_dim) = (8usize, 256usize);
        let full = n_heads * head_dim * 2;
        let mut covered = 0usize;
        let mut prev_end = 0usize;
        for r in 0..s.tp_size {
            let rg = s.wq_row_range(r, n_heads, head_dim);
            assert_eq!(rg.start, prev_end, "rank {r} wq range not contiguous");
            covered += rg.len();
            prev_end = rg.end;
        }
        assert_eq!(covered, full);
        assert_eq!(prev_end, full);
    }

    // ── Stage 3e EP-MoE expert ownership ──────────────────────────────

    #[test]
    fn experts_per_rank_splits_evenly() {
        let s = ShardConfig::new(2, true, 256, ExpertAssign::Stride).unwrap();
        assert_eq!(s.experts_per_rank(256), 128);
        let s4 = ShardConfig::new(4, true, 256, ExpertAssign::Contiguous).unwrap();
        assert_eq!(s4.experts_per_rank(256), 64);
        // dense model → single owner gets all
        let dense = ShardConfig::new(2, true, 0, ExpertAssign::Stride).unwrap();
        assert_eq!(dense.experts_per_rank(0), 0);
    }

    #[test]
    fn stride_assignment_ownership_and_local_ids() {
        let s = ShardConfig::new(2, true, 256, ExpertAssign::Stride).unwrap();
        // Stride: expert e → rank e % 2. Rank 0 owns evens, rank 1 owns odds.
        assert!(s.owns_expert(0, 0) && s.owns_expert(0, 2) && s.owns_expert(0, 254));
        assert!(s.owns_expert(1, 1) && s.owns_expert(1, 3) && s.owns_expert(1, 255));
        assert!(!s.owns_expert(0, 1) && !s.owns_expert(1, 0));
        let r0 = s.experts_on_rank(0);
        let r1 = s.experts_on_rank(1);
        assert_eq!(r0.len(), 128);
        assert_eq!(r1.len(), 128);
        assert_eq!(r0[0], 0);
        assert_eq!(r0[1], 2);
        assert_eq!(r1[0], 1);
        // every expert owned by exactly one rank (partition)
        for e in 0..256 {
            assert_eq!(s.owns_expert(0, e) as usize + s.owns_expert(1, e) as usize, 1);
        }
    }

    #[test]
    fn contiguous_assignment_blocks() {
        let s = ShardConfig::new(4, true, 256, ExpertAssign::Contiguous).unwrap();
        // Contiguous: rank r owns [r*64, (r+1)*64).
        assert!(s.owns_expert(0, 0) && s.owns_expert(0, 63) && !s.owns_expert(0, 64));
        assert!(s.owns_expert(3, 192) && s.owns_expert(3, 255) && !s.owns_expert(3, 191));
        let ids = s.experts_on_rank(2);
        assert_eq!(ids.first(), Some(&128));
        assert_eq!(ids.last(), Some(&191));
        assert_eq!(ids.len(), 64);
    }

    #[test]
    fn validate_moe_divisibility() {
        let s2 = ShardConfig::new(2, true, 256, ExpertAssign::Stride).unwrap();
        assert!(s2.validate_moe(256).is_ok());
        // 100 experts on 3 ranks is unbalanced.
        let s3 = ShardConfig::new(3, true, 0, ExpertAssign::Stride).unwrap();
        assert!(s3.validate_moe(100).is_err());
        assert!(s3.validate_moe(255).is_ok());
    }

    #[test]
    fn dense_model_no_experts() {
        // Dense (no experts): expert_to_rank empty → owns_expert false, no ids.
        let s = ShardConfig::new(2, true, 0, ExpertAssign::Stride).unwrap();
        assert!(!s.owns_expert(0, 5));
        assert!(!s.owns_expert(1, 5));
        assert!(s.experts_on_rank(0).is_empty());
    }
}
