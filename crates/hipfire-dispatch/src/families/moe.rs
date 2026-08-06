// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! MoE kernel family: dispatching expert GEMM operations.
//!
//! Supports 3 variants:
//! - **IndexedGateUp**: gate+up projection for a single expert (indexed by token)
//! - **IndexedDown**: down projection for a single expert (indexed by token)
//! - **GroupedGemm**: batched grouped-expert GEMM (all experts in one launch)
//!
//! # Current status
//!
//! `run()` is the centralized single-token MoE decode entry — it delegates to
//! [`crate::pipeline::run_moe_decode`] (the GPU top-K fast path plus the generic
//! CPU-top-K fallback). The family owns resolution (`MoeDtypes` → `MoeResolution`);
//! the model passes only the dtype snapshot + k. One `DispatchCtx` is threaded
//! end-to-end from the call site through every inner GEMV. Scratch stays model-owned.
//! Grouped-GEMM prefill is a future arm (gated on `ShapeInfo.batch_size`).

use rdna_compute::DType;
use rdna_compute::GpuTensor;
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, GivensRef, WeightRef};
use crate::tables::moe_table;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

// ── MoE eligibility lattice ────────────────────────────

/// Routed-expert tiers the mixed-tier bucketed decode path can execute: the
/// tiers for which per-tier indexed gate_up/down GEMV kernels exist (see
/// `run_moe_decode_mixed`). A per-expert tier table containing any other DType
/// cannot be served by the mixed path and is rejected up front with a clear
/// error rather than failing deep in the per-bucket dispatch.
pub const MIXED_SUPPORTED_TIERS: [DType; 3] = [DType::MQ4G256, DType::MQ6G256, DType::ParoQ4G128];

/// The routing operation selected for an MoE layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterSelection {
    SoftmaxTopK,
    SigmoidTopK,
    BiasAwareTopK,
    Hash,
    Precomputed,
}

/// Typed routing plan shared by decode and prefill MoE execution.
///
/// `normalize` describes whether selected routing weights are renormalized;
/// `route_scale` is applied after that combination. Bias is intentionally only
/// present on [`RouterPlan::BiasAwareTopK`], and hash operands are only present
/// on [`RouterPlan::Hash`], so those semantics cannot be silently dropped by a
/// generic boolean configuration.
pub enum RouterPlan<'a> {
    SoftmaxTopK {
        scores: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        k_top: usize,
        normalize: bool,
        route_scale: f32,
    },
    SigmoidTopK {
        scores: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        k_top: usize,
        normalize: bool,
        route_scale: f32,
    },
    BiasAwareTopK {
        scores: &'a GpuTensor,
        gate_bias: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        k_top: usize,
        normalize: bool,
        route_scale: f32,
    },
    Hash {
        scores: &'a GpuTensor,
        tokens: &'a GpuTensor,
        tid2eid: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        k_top: usize,
        normalize: bool,
        route_scale: f32,
    },
    Precomputed {
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        k_top: usize,
        normalize: bool,
        route_scale: f32,
    },
}

impl RouterPlan<'_> {
    pub fn selection(&self) -> RouterSelection {
        match self {
            Self::SoftmaxTopK { .. } => RouterSelection::SoftmaxTopK,
            Self::SigmoidTopK { .. } => RouterSelection::SigmoidTopK,
            Self::BiasAwareTopK { .. } => RouterSelection::BiasAwareTopK,
            Self::Hash { .. } => RouterSelection::Hash,
            Self::Precomputed { .. } => RouterSelection::Precomputed,
        }
    }

    pub fn k_top(&self) -> usize {
        match self {
            Self::SoftmaxTopK { k_top, .. }
            | Self::SigmoidTopK { k_top, .. }
            | Self::BiasAwareTopK { k_top, .. }
            | Self::Hash { k_top, .. }
            | Self::Precomputed { k_top, .. } => *k_top,
        }
    }

    pub fn normalizes(&self) -> bool {
        match self {
            Self::SoftmaxTopK { normalize, .. }
            | Self::SigmoidTopK { normalize, .. }
            | Self::BiasAwareTopK { normalize, .. }
            | Self::Hash { normalize, .. }
            | Self::Precomputed { normalize, .. } => *normalize,
        }
    }

    pub fn route_scale(&self) -> f32 {
        match self {
            Self::SoftmaxTopK { route_scale, .. }
            | Self::SigmoidTopK { route_scale, .. }
            | Self::BiasAwareTopK { route_scale, .. }
            | Self::Hash { route_scale, .. }
            | Self::Precomputed { route_scale, .. } => *route_scale,
        }
    }
}

/// Expert execution shape. This is an execution choice, not a dtype
/// eligibility lattice; [`MoeResolution`] remains the owner of the latter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertExecutionPlan {
    IndexedQuantized,
    GroupedQuantized,
    PerExpertFallback,
}

/// Per-layer dtype snapshot the MoE eligibility lattice reads. Built by the
/// model from its weight structs; kept dtype-only so this stays GPU-free and
/// the dispatch crate needs no dependency on any arch crate.
///
/// `experts_all_gate_up_mq4` mirrors the `ffn.experts.iter().all(..)` clause
/// the original `gate_side_mq4` check used (qwen35.rs:4598-4605); the routed
/// fields use experts[0] as representative (the loader builds all experts in a
/// layer with matching dtype, so [0] == all — same invariant the original
/// routed_* checks relied on).
#[derive(Clone)]
pub struct MoeDtypes {
    pub router: DType,
    pub shared_gate: DType,        // ffn.shared_expert_gate
    pub shared_expert_gate: DType, // ffn.shared_expert.gate
    pub shared_expert_up: DType,   // ffn.shared_expert.up
    pub shared_expert_down: DType, // ffn.shared_expert.down
    pub experts_all_gate_up_mq4: bool,
    pub routed_gate_up: DType, // ffn.experts[0].gate_up
    pub routed_down: DType,    // ffn.experts[0].down
    /// Per-expert mixed routed dtype: experts in one layer carry DIFFERENT
    /// gate_up and/or down dtypes (N-tier graded: MQ6 hot / MQ4 mid / MQ2L
    /// or MQ3L or E8-family cold), so `routed_gate_up` / `routed_down`
    /// (= experts[0]) are NOT representative. Built by the model as
    /// `ffn.expert_dtype_tags.is_some()` — the tag table is built iff any
    /// expert's gate_up or down dtype differs from experts[0]. Tags:
    ///   0 = MQ6G256       (200 B/grp affine)
    ///   1 = MQ2G256Lloyd  ( 72 B/grp codebook)
    ///   2 = MQ4G256       (136 B/grp affine)
    ///   3 = MQ3G256Lloyd  (112 B/grp codebook)
    ///   4 = MFP4G32E8     (16 B hdr + (K/32)*17 B; 4-bit E8 lattice, 4.25 bpw)
    ///   5 = MFP3G32E8     (16 B hdr + (K/32)*13 B; 3-bit E8 lattice, 3.25 bpw)
    ///   6 = MFP2G32E8     (16 B hdr + (K/32)*9  B; 2-bit E8 lattice, 2.25 bpw)
    /// Drives the merged dtype-tag-branched gate_up AND down decode kernels.
    pub routed_has_mixed_experts: bool,
    pub has_paro_shared: bool, // ffn.paro_shared.is_some()
    /// True when any gate-side projection (router, shared_expert_gate/scalar,
    /// shared gate, shared up) carries an AWQ companion.  When true the fused
    /// gate kernel is disabled — each weight uses its individual WeightRef path
    /// which applies the per-weight AWQ scale.
    pub gate_side_has_awq: bool,
    /// True when the routed-down projection carries per-expert AWQ companion
    /// scales.  When true the batched prefill Path 2 (grouped-GEMM) is disabled;
    /// Path 0/1 (indexed GEMV, per-token fallback) remain eligible.
    pub routed_down_has_awq: bool,
    /// Per-expert gate_up tiers for intra-layer mixed-tier dispatch. `None`
    /// (default) ⇒ today's uniform path (representative `routed_gate_up` drives
    /// resolution). `Some(table)` with >1 distinct DType marks the layer
    /// `mixed`; a `Some` table that is all-equal collapses to the uniform path.
    pub per_expert_gate_up: Option<Vec<DType>>,
    /// Per-expert down tiers (parallel to `per_expert_gate_up`). Same semantics.
    pub per_expert_down: Option<Vec<DType>>,
}

impl MoeDtypes {
    pub fn has_mq6_projection(&self) -> bool {
        [
            self.shared_expert_gate,
            self.shared_expert_up,
            self.shared_expert_down,
            self.routed_gate_up,
            self.routed_down,
        ]
        .iter()
        .any(|dt| matches!(*dt, DType::MQ6G256))
    }
}

/// Resolved fused-vs-fallback eligibility for one MoE decode layer. This IS the
/// routing-config logic, relocated from `moe_ffn_decode_impl` into one typed,
/// testable place (review finding #1). Pure function of `MoeDtypes` + k.
#[derive(Clone, Copy, Debug)]
pub struct MoeResolution {
    pub gate_side_mq4: bool,
    /// Router + shared expert are MQ4 (fused gate path applicable, independent
    /// of routed-expert dtype). True for uniform MQ4 AND graded files whose
    /// gate-side is MQ4 (e.g. the redline mq4r).
    pub gate_fusable: bool,
    pub routed_indexable_mq4: bool,
    pub routed_indexable_mq5: bool,
    pub routed_indexable_mq6: bool,
    /// Mixed routed experts: gate_up MQ4, down MQ6 (the "mq6-down" lever —
    /// promote only the sensitive residual-write projection to 6-bit while
    /// gate_up stays 4-bit). Indexable on the decode GPU-top-K path: gate_up
    /// uses the MQ4 indexed GEMV, down uses the MQ6 indexed GEMV, silu+rotate
    /// (optionally AWQ) is weight-agnostic. Decode-only (prefill Path-0 on
    /// gfx9* has no MQ6 down arm; eval scores per-token = decode).
    pub routed_indexable_mixed_gu4_dn6: bool,
    pub routed_indexable_paro: bool,
    /// Uniform all-MFP4G32E8 routed experts on the RDNA3 wave32-WMMA family
    /// (`arch_has_e8_wmma`).  Indexable on the decode GPU-top-K path via the
    /// E8 indexed/grouped MoE GEMV kernels.  Arch-agnostic `resolve()` never
    /// admits E8 (`arch_has_e8_wmma=false`); `resolve_arch` with a wave32-WMMA
    /// arch does.
    pub routed_indexable_e8: bool,
    /// Uniform all-MQ2-Lloyd routed experts (gate_up == down == MQ2G256Lloyd).
    /// Reuses the ds4/minimax indexed Lloyd MoE GEMVs on the decode GPU-top-K
    /// path: gate_up uses the MQ2-Lloyd indexed GEMV, down uses the MQ2-Lloyd
    /// atomic-residual GEMV (self-combining -> no separate down combine).
    pub routed_indexable_mq2lloyd: bool,
    /// Uniform all-MQ3-Lloyd routed experts (gate_up == down == MQ3G256Lloyd).
    /// Same indexed-Lloyd decode path as mq2lloyd, MQ3 launchers.
    pub routed_indexable_mq3lloyd: bool,
    /// Per-expert N-tier graded routed experts (MQ6 hot / MQ4 mid / MQ2L or
    /// MQ3L cold, applied to BOTH gate_up and down). Indexable on the decode
    /// GPU-top-K path via the merged dtype-tag-branched gate_up AND down
    /// kernels. The merged down writes the EXPANDED buffer for all dtypes →
    /// the single shared `moe_down_combine_k8_batched` runs (NOT Lloyd atomic
    /// self-combine). silu+rotate is weight-agnostic (unchanged).
    pub routed_indexable_mixed_per_expert: bool,
    pub use_gpu_topk: bool,
    pub needs_x_rot_local: bool,
    /// True when a per-expert tier table is `Some` AND contains >1 distinct
    /// DType — the layer's routed experts span multiple quant tiers and need
    /// the bucketed dispatch path (Task 3). `None` tables or all-equal `Some`
    /// tables leave this `false` ⇒ unchanged uniform fast path.
    pub mixed: bool,
}

impl MoeResolution {
    /// Arch-agnostic entry. The E8 indexed/grouped kernels exist on the RDNA3
    /// wave32-WMMA family (gfx11; `arch_has_e8_wmma`); passing `false` here routes
    /// E8 to the CPU-top-K fallback — preserving every existing caller + test.
    pub fn resolve(d: &MoeDtypes, k: usize) -> Self {
        Self::resolve_arch(d, k, false)
    }

    pub fn resolve_arch(d: &MoeDtypes, k: usize, arch_has_e8_wmma: bool) -> Self {
        use DType::*;
        // Gate-side weights (router + shared expert) all MQ4 → the fused gate
        // kernel (fused_qkvza_hfq4g256 on one rotated xr) is applicable. This is
        // INDEPENDENT of the routed-expert dtype (all MQ-family share the same
        // FwhtG256 rotation), so it can fire on graded files too (redline mq4r).
        // When any gate-side projection carries an AWQ companion, the fused gate
        // kernel is disabled — each weight uses its individual WeightRef path
        // which applies the per-weight AWQ scale.
        let dtypes_all_mq4 = d.router == MQ4G256
            && d.shared_gate == MQ4G256
            && d.shared_expert_gate == MQ4G256
            && d.shared_expert_up == MQ4G256;
        let gate_fusable = dtypes_all_mq4 && !d.gate_side_has_awq;
        // gate_side_mq4 keeps the stricter all-MQ4 meaning (incl. routed experts)
        // for the rotate/AWQ branch + callers that assume a uniform-MQ4 FFN.
        // Gate-side AWQ also disables gate_side_mq4 (the fused rotate+gemv path
        // cannot interleave AWQ divides).
        let gate_side_mq4 = gate_fusable && d.experts_all_gate_up_mq4;

        let routed_gate_up_mq4 = d.routed_gate_up == MQ4G256;
        let routed_gate_up_mq5 = d.routed_gate_up == MQ5G256;
        let routed_gate_up_mq6 = d.routed_gate_up == MQ6G256;
        let routed_gate_up_paro = d.routed_gate_up == ParoQ4G128 && d.has_paro_shared;
        let routed_gate_up_mq2lloyd = d.routed_gate_up == MQ2G256Lloyd;
        let routed_gate_up_mq3lloyd = d.routed_gate_up == MQ3G256Lloyd;

        let routed_indexable_mq4 = (d.routed_down == MQ4G256) && routed_gate_up_mq4;
        let routed_indexable_mq5 = (d.routed_down == MQ5G256) && routed_gate_up_mq5;
        let routed_indexable_mq6 = (d.routed_down == MQ6G256) && routed_gate_up_mq6;
        let routed_indexable_mixed_gu4_dn6 = routed_gate_up_mq4 && (d.routed_down == MQ6G256);
        let routed_indexable_mq2lloyd = (d.routed_down == MQ2G256Lloyd) && routed_gate_up_mq2lloyd;
        let routed_indexable_mq3lloyd = (d.routed_down == MQ3G256Lloyd) && routed_gate_up_mq3lloyd;
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;
        // Per-expert mixed: the model already verified the experts carry
        // different down dtypes and built the tag table (single source of
        // truth). gate_up stays uniform MQ4, so it pairs with the MQ4 indexed
        // gate_up GEMV; the merged dtype-tag kernel serves the down step.
        let routed_indexable_mixed_per_expert = d.routed_has_mixed_experts;
        // MFP4G32E8 grouped experts (RDNA3 wave32-WMMA): the
        // gemv_mfp4g32_e8_moe_{gate_up,down}_k8_indexed kernels exist for MFP4G32E8
        // ONLY. MFP3G32E8 and MFP2G32E8 have no indexed MoE decode path and are
        // rejected at the dtype-pair level (fallible_dtype_tag) and here.
        // FWHT-rotated (FwhtG256), same as MQ4, so shared silu+mul+rotate applies.
        let routed_gate_up_e8 = d.routed_gate_up == MFP4G32E8;
        let routed_indexable_e8 =
            arch_has_e8_wmma && routed_gate_up_e8 && d.routed_down == MFP4G32E8;

        let routed_dtype_indexable = routed_indexable_mq4
            || routed_indexable_mq5
            || routed_indexable_mq6
            || routed_indexable_mixed_gu4_dn6
            || routed_indexable_mixed_per_expert
            || routed_indexable_mq2lloyd
            || routed_indexable_mq3lloyd
            || routed_indexable_paro
            || routed_indexable_e8;

        let use_gpu_topk = k == 8 && routed_dtype_indexable;
        let needs_x_rot_local = gate_side_mq4
            || routed_indexable_mixed_per_expert
            || routed_gate_up_mq4
            || routed_gate_up_mq5
            || routed_gate_up_mq6
            || routed_gate_up_mq2lloyd
            || routed_gate_up_mq3lloyd
            || routed_gate_up_paro
            || routed_indexable_e8;

        // A per-expert tier table is "mixed" only when it is Some AND spans more
        // than one distinct DType. A Some table that is all-equal collapses to
        // the uniform fast path (mixed = false), so existing arches — which pass
        // None for both tables — are always uniform and byte-identical to today.
        let table_varies = |t: &Option<Vec<DType>>| {
            t.as_ref()
                .and_then(|v| v.split_first())
                .map(|(first, rest)| rest.iter().any(|dt| dt != first))
                .unwrap_or(false)
        };
        let mixed = table_varies(&d.per_expert_gate_up) || table_varies(&d.per_expert_down);

        Self {
            gate_side_mq4,
            gate_fusable,
            routed_indexable_mq4,
            routed_indexable_mq5,
            routed_indexable_mq6,
            routed_indexable_mixed_gu4_dn6,
            routed_indexable_mq2lloyd,
            routed_indexable_mq3lloyd,
            routed_indexable_mixed_per_expert,
            routed_indexable_paro,
            routed_indexable_e8,
            use_gpu_topk,
            needs_x_rot_local,
            mixed,
        }
    }

    pub fn routed_indexable(&self) -> bool {
        self.routed_indexable_mq4
            || self.routed_indexable_mq5
            || self.routed_indexable_mq6
            || self.routed_indexable_mixed_gu4_dn6
            || self.routed_indexable_mixed_per_expert
            || self.routed_indexable_mq2lloyd
            || self.routed_indexable_mq3lloyd
            || self.routed_indexable_paro
            || self.routed_indexable_e8
    }
}

// ── Dispatch parameters ────────────────────────────────

/// Everything the MoE decode executor arm reads, marshaled by the model from
/// its weight/config/scratch structs. Resolution is owned by the family
/// (the model passes only the dtype snapshot + k); the executor computes
/// [`MoeResolution`] from [`MoeDtypes`] on entry.
pub struct MoeParams<'a> {
    pub dtypes: MoeDtypes,
    /// Token-batch width. Decode = 1. >1 must route to grouped prefill (Step 8).
    /// Guarded at runtime matching the bias-aware decode guard.
    pub batch_size: usize,
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub smi: usize,
    pub k: usize,
    pub n_exp: usize,
    pub norm_topk_prob: bool,
    pub x_rot_prerotated: bool,
    /// Safetensors layer index (== `MoeFfnWeights.layer_idx`). Only used
    /// by native GPTQ-on-E8 Hessian capture in the CPU-top-K fallback to
    /// build the per-(tensor,expert) key; ignored on the hot path.
    pub layer_idx: u16,
    // activations / residual
    pub x_norm: &'a GpuTensor,
    pub x_residual: &'a GpuTensor,
    /// EP (expert-parallel, Ship 6 substrate-EP) routed-output redirect. When
    /// `Some`, the routed combine AND the shared-expert down accumulate into
    /// this **zeroed** partial buffer instead of `x_residual`; the EP executor
    /// then all-reduces the partial across ranks and adds it into `x_residual`
    /// once. `None` (default) = single-GPU: accumulate directly into
    /// `x_residual`, byte-identical to pre-EP behavior.
    pub routed_out: Option<&'a GpuTensor>,
    /// EP: skip the shared-expert **down** projection so the replicated shared
    /// expert is computed on rank 0 only (not summed N× by the all-reduce).
    /// `false` (default) = run it (single-GPU). Router + shared gate/up still
    /// run on every rank (they share the fused gate-side GEMV with the router).
    pub skip_shared: bool,
    // gate-side weights
    pub router: WeightRef<'a>,
    pub shared_expert_gate: WeightRef<'a>,
    pub shared_gate_w: WeightRef<'a>,
    pub shared_up_w: WeightRef<'a>,
    pub shared_down_w: WeightRef<'a>,
    // routed expert pointer tables + dims
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Route A MoE-AWQ: per-routed-expert down `awq_scale` pointer table
    /// (`[2·n_exp]` f32 = n_exp `u64` ptrs → each expert's `[routed_down_k]`
    /// f32 scale). `Some` only when the `.hfq` carries per-expert
    /// `down_proj.awq_scale` sidecars; the executor then runs the AWQ-aware
    /// indexed silu+rotate (`x/s` before the FWHT). `None` (default) = the
    /// plain silu+rotate, byte-identical to pre-AWQ.
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// Per-expert mixed-precision decode: `[n_exp]` u8 (DType::Raw, 1 B/exp)
    /// dtype-tag table, `Some` iff any expert's gate_up or down dtype differs
    /// from experts[0] (N-tier graded files). The merged dtype-tag-branched
    /// gate_up AND down kernels read `dtype_tags[expert_id]` per block:
    ///   0=MQ6G256 (200 B/grp), 1=MQ2G256Lloyd (72 B/grp),
    ///   2=MQ4G256 (136 B/grp), 3=MQ3G256Lloyd (112 B/grp).
    /// `None` (default) ⇒ uniform path, byte-identical to pre-mixed.
    pub expert_dtype_tags: Option<&'a GpuTensor>,
    pub routed_gate_up_k: usize,
    pub routed_down_m: usize,
    pub routed_down_k: usize,
    /// Per-expert (gate_up, down) weight refs for the generic CPU-top-K
    /// fallback (`!use_gpu_topk`: k != 8 OR routed dtype not indexable).
    /// Master's `moe_ffn_decode_impl` indexed `ffn.experts[expert_idx]` in a
    /// host loop; the indexed-kernel pointer tables above can't drive that
    /// path (they assume k=8 + an indexable routed dtype). One ref pair per
    /// expert, length `n_exp`. **Empty** when the layer is paged (the indexed
    /// GPU-top-K path is the only mode in paged residency) — the fallback
    /// asserts non-empty before use, matching master's `ffn.experts[..]`
    /// indexing (which also required resident experts).
    pub routed_experts: &'a [(WeightRef<'a>, WeightRef<'a>)],
    // paro sidecars
    pub routed_gate_up_paro: Option<GivensRef<'a>>,
    pub routed_down_paro: Option<GivensRef<'a>>,
    // scratch buffers
    pub router_logits: &'a GpuTensor,
    pub scalar_buf: &'a GpuTensor,
    pub x_rot_local: &'a GpuTensor,
    /// Fused [gate||up] scratch of length `2 * max(mi, smi)`. Used by the
    /// generic CPU-top-K fallback to receive a single routed expert's fused
    /// gate_up GEMV output (master wrote `expert.gate_up` into one buffer of
    /// width `2*mi`, then sliced gate/up halves). The GPU-top-K fast path
    /// does not read this field.
    pub gate_up_buf: &'a GpuTensor,
    pub gate_buf: &'a GpuTensor,
    pub up_buf: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
}

// ── DeepSeek-V4 bias-aware decode parameters ───────────

/// Parameters for the deepseek4 bias-aware MoE decode arm (k=6, MQ2-Lloyd routed
/// experts). Kept distinct from [`MoeParams`] because the ds4 sub-graph has no
/// fused gate-side and no shared-expert block: the shared expert is a separate
/// model-owned step (`ffn_stub`) that runs first and seeds `ffn_out`, and this
/// arm's routed-down kernel atomic-accumulates into that same buffer.
///
/// `scores` is the post-`sqrt_softplus(gate·x)` router output — the model owns
/// the router GEMV + activation. Selection adds `gate_bias` while the routing
/// weights use the *unbiased* `scores`; the bias-aware kernel handles that
/// two-score semantic and folds in `route_scale`, all in one launch. The model
/// pre-rotates the activation, so `x_rot` is consumed as-is (no re-rotation).
pub struct MoeBiasAwareParams<'a> {
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub k_top: usize,
    pub n_exp: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// Token-batch width. Decode = 1. A value > 1 must route to the grouped
    /// prefill executor (Step 8), never this decode arm — guarded in the executor.
    pub batch_size: usize,
    // activations / residual
    /// FWHT-rotated activation (model pre-rotates; this arm does not re-rotate).
    pub x_rot: &'a GpuTensor,
    /// Residual stream the routed-down kernel atomic-accumulates into. The
    /// model's shared-expert step must have run first to seed this buffer.
    pub ffn_out: &'a GpuTensor,
    // router
    pub scores: &'a GpuTensor, // post-sqrt_softplus gate·x (weights use these)
    pub gate_bias: &'a GpuTensor, // per-expert routing bias (selection only)
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    // scratch buffers (model-owned)
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    /// `[k_top × hidden]` per-expert down outputs for the deterministic combine.
    pub down_expanded: &'a GpuTensor,
}

// ── DeepSeek-V4 batched/prefill MoE parameters ─────────

/// Router-selection mode for the batched/prefill MoE path. DeepSeek-V4 uses
/// static hash routing for the first `num_hash_layers` layers and bias-aware
/// top-k for the rest; the executor branches on this.
pub enum MoePrefillRouting<'a> {
    /// Bias-aware batched top-k (select on `scores + gate_bias`, weight on the
    /// unbiased `scores`, normalize, `*route_scale`).
    BiasAware { gate_bias: &'a GpuTensor },
    /// Static `tid2eid` hash routing (layers `0..num_hash_layers`). `tokens` is
    /// the device-side `[B]` i32 token-id buffer.
    Hash {
        tid2eid: &'a GpuTensor,
        tokens: &'a GpuTensor,
    },
}

/// Parameters for the deepseek4 batched/prefill MoE (k=6, MQ2-Lloyd). The
/// model owns RMSNorm, the shared expert, the router GEMV + `sqrt_softplus`
/// (producing `scores`); this arm runs routing → routed experts → combine,
/// accumulating into `ffn_out` (the shared expert already seeded it).
///
/// Picks the grouped-GEMM path when `batch_size >= HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE`
/// (default 128), else the scalar K4 indexed path — mirroring `ffn_batched`.
pub struct MoeBiasAwarePrefillParams<'a> {
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub n_exp: usize,
    pub k_top: usize,
    pub batch_size: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub layer_idx: usize, // for the optional HIPFIRE_DEEPSEEK4_DUMP_TOPK header
    // routing
    pub routing: MoePrefillRouting<'a>,
    pub scores: &'a GpuTensor, // post-sqrt_softplus moe_scores_batch [B, n_exp]
    pub topk_indices: &'a GpuTensor, // [B, k_top] (routing out, expert in)
    pub topk_weights: &'a GpuTensor, // [B, k_top]
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    // activation / residual
    pub x_rot: &'a GpuTensor,   // ffn_x_rot_batch [B, hidden]
    pub ffn_out: &'a GpuTensor, // ffn_out_batch [B, hidden] (accumulate target)
    // grouped-path scratch
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // shared scratch (grouped + scalar)
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // scalar-path scratch (expanded deterministic down)
    pub down_expert_outputs: &'a GpuTensor,
}

// ── Qwen3.5 softmax-top-k MoE prefill parameters (Ship 4.2) ──

/// Parameters for the qwen35 batched/prefill MoE routed-expert block.
///
/// Distinct from [`MoeBiasAwarePrefillParams`] — qwen35 uses softmax top-k
/// routing (k=8) with MQ4/MQ6/Paro routed experts, a fused gate-side, and a
/// shared expert that seeds `x_batch` before this arm runs.
///
/// The model owns RMSNorm, the router GEMV + softmax top-k (producing
/// `topk_indices` / `topk_weights`), and the shared expert (which already
/// accumulated into `x_batch`). This arm runs scatter → gate_up → unscatter →
/// SwiGLU+rotate → down → combine, accumulating into `x_batch`.
///
/// All tensor refs are `&'a GpuTensor` (shared, not `&mut` — GpuTensor is Copy).
/// Scratch tensors are model-owned; the family holds only references.
pub struct MoePrefillParams<'a> {
    // dtype snapshot
    pub dtypes: MoeDtypes,
    // dims
    pub batch_size: usize,
    pub mi: usize,
    pub down_m: usize,
    pub down_k: usize,
    pub gate_up_k: usize,
    pub k_top: usize,
    pub n_exp: usize,
    /// m_total upper bound pre-computed by the model via
    /// `moe_grouped_m_total_bound(total_slots, n_exp)`. Used by Path 2
    /// scatter + grouped GEMM for grid sizing.
    pub m_total_max: usize,
    /// Model-level safety fence for promoted/mixed MQ6 checkpoints. When true,
    /// MQ4 grouped prefill calls use FP16 WMMA even for layers whose local
    /// routed dtype snapshot is pure MQ4. This keeps pure MQ4 models on the
    /// existing i8 default while avoiding mixed-checkpoint corruption.
    pub force_mq4_grouped_fp16: bool,
    // routing inputs (model-produced)
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    // destination = x_batch (residual; combine accumulates here)
    pub x_batch: &'a GpuTensor,
    // activation buffers
    pub x_norm_batch: &'a GpuTensor,
    pub x_rot_batch: &'a GpuTensor,
    // routed gate_up/down pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Route A MoE-AWQ: per-routed-expert down `awq_scale` pointer table (see
    /// [`MoeParams::expert_down_awq_ptrs`]). When `Some`, the prefill silu+rotate
    /// uses the indexed AWQ kernel (per-slot scale via `topk_indices`),
    /// superseding the single-scale `down_awq_scale` stub below for routed
    /// experts. `None` (default) = plain silu+rotate.
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// Per-expert mixed-precision prefill: `[n_exp]` u8 dtype-tag table,
    /// `Some` iff the routed experts carry mixed dtypes (graded T3-3L). Drives
    /// the merged grouped-WMMA prefill kernel. `None` ⇒ uniform path, byte-identical.
    pub expert_dtype_tags: Option<&'a GpuTensor>,
    // intermediate buffers
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // Path 1 expanded-down scratch
    pub down_expanded: &'a GpuTensor,
    // Path 2 scatter scratch (model-owned)
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // paro sidecars (per-layer shared Givens rotation tables)
    pub paro_gate_up: Option<GivensRef<'a>>,
    pub paro_down: Option<GivensRef<'a>>,
    /// AWQ scale for the routed down weight (experts[0].down.awq_scale).
    /// Used by the AWQ-aware silu+rotate step. `None` when the routed
    /// experts are non-AWQ (the common case for A3B).
    pub down_awq_scale: Option<&'a GpuTensor>,
    /// EP (Ship 6 substrate-EP prefill): when `Some`, the **routed** combine
    /// accumulates into this **zeroed** `[batch × dim]` partial instead of
    /// `x_batch`; the EP prefill driver then all-reduce-sums the partial across
    /// ranks and adds it into each rank's `x_batch`. The **shared** expert stays
    /// in `x_batch` (replicated per rank — added once to each rank's own copy,
    /// no all-reduce). `None` (the default) accumulates routed into `x_batch`,
    /// byte-identical to pre-EP behavior.
    pub routed_out: Option<&'a GpuTensor>,
}

/// Resolved dispatch plan for the qwen35 batched MoE prefill routed block.
///
/// Distinct from [`MoeResolution`] (decode) — prefill adds the Path 0/1/2
/// grouped-vs-scalar down selection and the Paro i8/k8 levers.
/// Pure function of [`MoeDtypes`] + arch + [`FeatureFlags`].
pub struct MoePrefillResolution {
    /// Gate_up + down via grouped-GEMM scatter pipeline (Path 2).
    /// Requires WMMA-capable arch (gfx11/gfx12) + `moe_grouped_gemm` flag.
    pub use_path2: bool,
    /// Down uses atomic-accumulate GEMV (Path 0) instead of atomic-free
    /// expanded+combine (Path 1). gfx9* wave64 archs (gfx906/gfx908/gfx94x).
    pub down_path0: bool,
    /// gfx1151 Paro i8 MMQ grouped GEMM (Path 2 only).
    pub use_paro_i8: bool,
    /// gfx1151 Paro i8 MMQ k8 grouped GEMM (Path 2 only).
    pub use_paro_i8_k8: bool,
    /// Routed experts use ParoQ4G128 (determines SwiGLU+rotate kernel selection).
    pub paro_mode: bool,
    /// gfx1151's HFQ4 grouped-i8 path is correct for pure MQ4, but corrupts
    /// MQ6-promoted A3B MTP prefill when the same MoE layer mixes MQ4 and MQ6
    /// projections. Default mixed layers back to FP16 WMMA; explicit
    /// HIPFIRE_MOE_GROUPED_I8=1 still opts into the research path.
    pub force_mq4_grouped_fp16: bool,
}

impl MoePrefillResolution {
    /// Resolve the prefill dispatch plan from dtypes, arch, and flags.
    ///
    /// Reads MoE prefill env levers from `flags` (parsed once at `Gpu::init`),
    /// not `std::env` — mid-prefill env mutation is not honored.
    pub fn resolve(
        d: &MoeDtypes,
        arch: &rdna_compute::arch_caps::ArchCaps,
        flags: &rdna_compute::feature_flags::FeatureFlags,
    ) -> Self {
        let paro_mode = d.routed_gate_up == DType::ParoQ4G128 && d.has_paro_shared;
        let use_path2 = flags.moe_grouped_gemm && arch.has_wmma();
        // MQ6 grouped-WMMA: gfx11 `_k2` kernel now exists (alongside the
        // gfx12 `_gfx12` kernel). Only suppress Path 2 for MQ6 on archs that
        // have NEITHER (gfx9*, gfx1010/1030, CDNA) — i.e. no wmma_w32 and not
        // gfx12. gfx1100/1101/1102/1103/1150/1151/1152 all have wmma_w32.
        // (Master's narrower gfx1151-only MQ6 admit (dfed8cc6) is subsumed by
        // this wider gfx11 widen (8d555fc6); master's mixed-checkpoint safety
        // is preserved separately via `force_mq4_grouped_fp16` below.)
        let mq6_on_non_wmma = d.routed_gate_up == DType::MQ6G256
            && !arch.has_wmma_w32()
            && !(arch.is_gfx1200() || arch.is_gfx1201());
        let use_path2 = use_path2 && !mq6_on_non_wmma;
        // MQ5 grouped-WMMA (`gemm_hfq5g256_moe_grouped_wmma`) is gfx12-only
        // (same as MQ6) — fall back to Path 1 (indexed batched GEMV) on
        // gfx11/gfx9 to avoid the gfx12-only kernel panic.
        let mq5_on_non_gfx12 =
            d.routed_gate_up == DType::MQ5G256 && !(arch.is_gfx1200() || arch.is_gfx1201());
        let use_path2 = use_path2 && !mq5_on_non_gfx12;
        // Mixed per-expert: the merged grouped kernel covers all four dtype
        // tags on any WMMA arch (gfx11 _k2 or gfx12 .gfx12). The routed
        // representative dtype may be MQ6/MQ5 and trip the suppression above,
        // so re-admit Path 2 when the file is graded-mixed (tag table present).
        let use_path2 =
            use_path2 || (d.routed_has_mixed_experts && flags.moe_grouped_gemm && arch.has_wmma());
        // mfp4-E8 routed experts: use Path 2 (grouped-WMMA) on gfx1151 and gfx12
        // (RDNA4). Both have a native E8 grouped-WMMA GEMM kernel:
        //   gfx1151 → gemm_mfp4g32_e8_moe_grouped_wmma (gfx1151.hip)
        //   gfx12   → gemm_mfp4g32_e8_moe_grouped_wmma_gfx12 (gfx12.hip)
        // Other archs (gfx1100 dGPU, gfx9*/CDNA) have no grouped E8 sister → Path 1.
        // mfp4-E8 grouped-WMMA prefill on ALL WMMA arches (RDNA3 gfx11 + RDNA4
        // gfx12). The gfx1151 kernel uses the RDNA3 wave32-WMMA builtin and runs
        // correctly on gfx1100/1101/1102; gfx12 uses its .gfx12 sister. The prior
        // "gfx1151-only / gfx1100 wash" call rested on pp512 97.5-vs-97.6 — which is
        // DECODE tok/s, not prefill throughput (a prefill change can't move decode
        // tok/s). Real prefill throughput is what bench_sweep measures, so route
        // gfx1100 through Path 2 and re-measure. Only ever active under the
        // HIPFIRE_E8_GFX12 batched-prefill gate.
        let e8_no_grouped = matches!(
            d.routed_gate_up,
            DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8
        ) && !(arch.is_rdna3() || arch.is_rdna4());
        let use_path2 = use_path2 && !e8_no_grouped;
        // Path 0: gfx9* wave64 archs (gfx906/gfx908/gfx94x) — cheap HBM
        // atomics make the atomic GEMV pattern competitive vs expanded scratch.
        let down_path0 = arch.is_gcn5() || arch.is_cdna1() || arch.is_cdna3();
        let is_gfx1151 = arch.is_gfx1151();
        let use_paro_i8 = paro_mode && use_path2 && is_gfx1151 && flags.moe_paro_i8.unwrap_or(true);
        let use_paro_i8_k8 = use_paro_i8 && flags.moe_paro_i8_k8.unwrap_or(true);
        // Routed-down AWQ suppresses Path 2 (grouped-GEMM): the AWQ divide
        // must interleave per-expert silu+rotate, which the grouped hot-path
        // does not support.  Path 0/1 (indexed GEMV paths) remain eligible.
        let use_path2 = use_path2 && !d.routed_down_has_awq;

        let force_mq4_grouped_fp16 =
            use_path2 && is_gfx1151 && d.has_mq6_projection() && flags.moe_grouped_i8.is_none();
        Self {
            use_path2,
            down_path0,
            use_paro_i8,
            use_paro_i8_k8,
            paro_mode,
            force_mq4_grouped_fp16,
        }
    }
}

// ── Family ─────────────────────────────────────────────

pub struct MoeFamily {
    registry: KernelRegistry,
}

impl MoeFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        moe_table::populate(&mut registry);
        registry
            .validate()
            .expect("moe kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Resolve the best kernel key for the given MoE variant.
    ///
    /// Applies arch gating through `KernelRegistry::resolve`.
    pub fn resolve(
        &self,
        variant: MoeVariant,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = match variant {
            MoeVariant::IndexedGateUp => KernelKey::MoeIndexedGateUpLloyd,
            MoeVariant::IndexedDown => KernelKey::MoeIndexedDownLloyd,
            MoeVariant::GroupedGemm => KernelKey::MoeGroupedGemm,
        };
        self.registry.resolve(key, ctx, shape)
    }

    /// Run a single-token MoE decode step through the centralized executor.
    ///
    /// Delegates to [`crate::pipeline::run_moe_decode`], which dispatches the
    /// GPU top-K fast path (k=8 with an indexable routed dtype ∈ {MQ4G256,
    /// MQ6G256, ParoQ4G128}) or the generic CPU-top-K fallback (k != 8 or a
    /// non-indexable routed dtype). Resolution is owned here (the family
    /// resolves [`MoeDtypes`] → [`MoeResolution`]), and `ctx` is threaded
    /// through every inner GEMV so the call site builds one `DispatchCtx`
    /// per token (not 6+). Scratch stays model-owned.
    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode(ctx, gpu, params)
    }

    /// Run a single-token deepseek4 bias-aware MoE decode step (k=6, MQ2-Lloyd
    /// routed experts). Delegates to [`crate::pipeline::run_moe_decode_bias_aware`].
    ///
    /// The model owns the router GEMV + `sqrt_softplus` (producing
    /// `params.scores`) and the shared expert (`ffn_stub`, which seeds
    /// `params.ffn_out`); this entry runs only the bias-aware top-k + routed
    /// MQ2-Lloyd expert sub-graph.
    ///
    /// Takes no `DispatchCtx`: the bias-aware path dispatches fixed MQ2-Lloyd
    /// kernels with no arch-gated sub-dispatch, so building a `DispatchCtx`
    /// per layer per token (an uncached `FeatureFlags::from_env` parse) would
    /// be pure waste on the decode hot path.
    pub fn run_bias_aware(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeBiasAwareParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode_bias_aware(gpu, params)
    }

    /// Run a batched/prefill deepseek4 MoE step (k=6, MQ2-Lloyd): routing
    /// (bias-aware or hash) → routed experts (grouped GEMM when
    /// `batch_size >= gate`, else scalar K4 indexed) → combine, accumulating
    /// into `params.ffn_out`. Delegates to
    /// [`crate::pipeline::run_moe_prefill_bias_aware`]. The model owns RMSNorm,
    /// the shared expert, and the router GEMV + `sqrt_softplus`.
    pub fn run_bias_aware_prefill(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeBiasAwarePrefillParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_prefill_bias_aware(gpu, params)
    }

    /// Run a batched/prefill qwen35 MoE routed-expert block (k=8, softmax
    /// top-k, MQ4/MQ6/Paro routed experts): scatter → gate_up → unscatter →
    /// SwiGLU+rotate → down → combine, accumulating into `params.x_batch`.
    ///
    /// The model owns RMSNorm, the router GEMV + softmax top-k, and the
    /// shared expert. Family owns resolution (`MoeDtypes` + arch + flags →
    /// [`MoePrefillResolution`]) and the full routed pipeline. `ctx` is
    /// decision-only (arch/env) — threaded once per chunk, not per layer.
    /// Delegates to [`crate::pipeline::run_moe_prefill`].
    pub fn run_prefill(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut rdna_compute::Gpu,
        params: &MoePrefillParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_prefill(ctx, gpu, params)
    }
}

impl KernelFamily for MoeFamily {
    fn name(&self) -> &'static str {
        "moe"
    }
}

// ── Placement-agnostic expert reference ──────────────────

/// Placement-agnostic view over an arch's MoE expert weight pointer tables.
/// All fields are borrowed from arch-owned layer structs; no data is copied.
///
/// Passed to the Step-IR launch helpers below so `execute_steps` arms can
/// dispatch the right kernel without importing any arch crate or matching on
/// arch-internal types.
///
/// **Field naming:** `expert_m` = intermediate dimension (inter); gate_up
/// writes `2 * expert_m` (fused gate||up), down reads `expert_m`.
/// `expert_k` = hidden dimension; gate_up reads `expert_k`, down writes
/// `expert_k`.
///
/// **Dropped from the brief:** `dummy_down` — no arch allocates a dummy down
/// buffer; only `dummy_gate_up` exists (minimax.rs:405, ds4 arch.rs:337).
pub struct MoeExpertRef<'a> {
    /// `[n_experts]` u64 device-pointer table; each entry points to one
    /// expert's fused gate||up weight buffer `[2·expert_m, expert_k]`.
    pub gate_up_ptrs: &'a GpuTensor,
    /// `[n_experts]` u64 device-pointer table; each entry points to one
    /// expert's down weight buffer `[expert_k, expert_m]`.
    pub down_ptrs: &'a GpuTensor,
    /// EP-shard dummy gate_up buffer (zeroed). Non-owned expert slots in
    /// `gate_up_ptrs` point here so SwiGLU(0,0)=0 → zero contribution. Must
    /// outlive `gate_up_ptrs`. `None` for single-GPU / fully-owned shards.
    pub dummy_gate_up: Option<&'a GpuTensor>,
    /// Expert weight dtype (uniform: gate_up and down share the same tier).
    pub dtype: DType,
    /// Total logical expert count for this layer.
    pub n_experts: usize,
    /// Intermediate dimension: gate_up writes `2 * expert_m`; down reads `expert_m`.
    pub expert_m: usize,
    /// Hidden dimension: gate_up reads `expert_k`; down writes `expert_k`.
    pub expert_k: usize,
    /// Locally-owned expert indices for EP context. Empty slice = all owned
    /// (single-GPU or non-EP path).
    pub owned: &'a [usize],
}

// ── Step-IR launch helpers ────────────────────────────────

use crate::pipeline::{GemvInput, MoeActivationVariant, MoeProj, QwenDownMode, ScoreActKind, Step};
/// Per-arch SwiGLU + FWHT rotate of the gate/up MoE intermediate.
///
/// - `MinimaxFused`: one fused kernel writes `rot_out` directly.
///   `awq_scale = None` → `gpu.fused_silu_mul_rotate_mq_batched` (gemv.rs:2500).
///   `awq_scale = Some(s)` → `gpu.fused_silu_mul_rotate_mq_awq_batched` (gemv.rs:2640).
/// - `Ds4ClampRotate`: two kernels.
///   1. `gpu.deepseek4_silu_mul_clamp_f32_batched(gate, up, gate, inter, k_top, swiglu_limit)`
///      (norm.rs:3977) — silu·mul·clamp in-place into `gate`.
///   2. `gpu.rotate_x_mq_batched(gate, rot_out, inter, k_top)`
///      (gemv.rs:2822) — FWHT-rotate the clamped `gate` into `rot_out`.
pub fn launch_moe_activation(
    gpu: &mut rdna_compute::Gpu,
    variant: &MoeActivationVariant<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    rot_out: &GpuTensor,
    inter: usize,
    k_top: usize,
) -> Result<(), DispatchError> {
    match variant {
        MoeActivationVariant::MinimaxFused { awq_scale: None } => gpu
            .fused_silu_mul_rotate_mq_batched(gate, up, rot_out, inter, k_top)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        MoeActivationVariant::MinimaxFused {
            awq_scale: Some(awq),
        } => gpu
            .fused_silu_mul_rotate_mq_awq_batched(gate, up, awq, rot_out, inter, k_top)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        MoeActivationVariant::Ds4ClampRotate { swiglu_limit } => {
            gpu.deepseek4_silu_mul_clamp_f32_batched(gate, up, gate, inter, k_top, *swiglu_limit)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.rotate_x_mq_batched(gate, rot_out, inter, k_top)
                .map_err(|e| DispatchError::Hip(e.to_string()))
        }
        // qwen Route A MoE-AWQ: per-routed-expert down.awq_scale selected by
        // topk_indices[krank]. Divides silu(g)*u by the expert's scale before
        // the FWHT (AWQ math (W·s)·(x/s)=W·x).
        MoeActivationVariant::QwenAwqIndexed {
            awq_ptrs,
            topk_indices,
        } => gpu
            .fused_silu_mul_rotate_mq_awq_indexed_batched(
                gate,
                up,
                awq_ptrs,
                topk_indices,
                rot_out,
                inter,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        // qwen Paro: fused silu·mul + Givens rotate (same kernel for decode
        // k_top and prefill batch·k_top row counts).
        MoeActivationVariant::QwenParo {
            pairs,
            theta,
            scales,
            krot,
        } => gpu
            .fused_silu_mul_givens_rotate_f32(
                gate, up, rot_out, pairs, theta, scales, k_top, inter, *krot,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
    }
}

/// In-place score activation before routing. Thin wrapper over the arch's
/// sigmoid / sqrt_softplus kernels (rdna-compute norm.rs:1643 / 3828).
/// `Sigmoid` requires the `deltanet` feature (same gate as `gpu.sigmoid_f32`).
pub fn launch_score_activation(
    gpu: &mut rdna_compute::Gpu,
    scores: &GpuTensor,
    kind: ScoreActKind,
) -> Result<(), DispatchError> {
    match kind {
        ScoreActKind::Sigmoid => {
            #[cfg(feature = "deltanet")]
            return gpu
                .sigmoid_f32(scores)
                .map_err(|e| DispatchError::Hip(e.to_string()));
            #[cfg(not(feature = "deltanet"))]
            return Err(DispatchError::UnsupportedVariant {
                family: "score_activation",
                variant: "sigmoid-requires-deltanet",
                arch: "",
                quant: "",
            });
        }
        ScoreActKind::SqrtSoftplus => gpu
            .sqrt_softplus_f32(scores)
            .map_err(|e| DispatchError::Hip(e.to_string())),
    }
}

/// Bias-aware top-K routing: select on `scores + gate_bias`, weight on the
/// unbiased `scores`, normalize, fold in `route_scale` — all in one launch.
/// Thin wrapper over `gpu.deepseek4_moe_topk_bias_aware_f32`.
pub fn launch_moe_route(
    gpu: &mut rdna_compute::Gpu,
    scores: &GpuTensor,
    gate_bias: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    n_exp: usize,
    k_top: usize,
    route_scale: f32,
) -> Result<(), DispatchError> {
    gpu.deepseek4_moe_topk_bias_aware_f32(
        scores,
        gate_bias,
        topk_indices,
        topk_weights,
        n_exp as i32,
        k_top as i32,
        route_scale,
    )
    .map_err(|e| DispatchError::Hip(e.to_string()))
}

/// Softmax + renormalized top-K routing (qwen decode). Two launches in one
/// helper — `softmax_f32(logits)` then `moe_topk_renorm_k8` — preserving the
/// legacy `run_moe_decode` launch order. Backs [`Step::MoeSoftmaxTopK`] and
/// the CPU-top-K fallback's k==8 branch.
pub fn launch_moe_softmax_topk(
    gpu: &mut rdna_compute::Gpu,
    logits: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    n_exp: usize,
    norm_topk_prob: bool,
) -> Result<(), DispatchError> {
    gpu.softmax_f32(logits)
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    gpu.moe_topk_renorm_k8(logits, topk_indices, topk_weights, n_exp, norm_topk_prob)
        .map_err(|e| DispatchError::Hip(e.to_string()))
}

/// In-place scaled add with a device-side scalar:
/// `x += y * scale` via `scaled_add_inplace_gpu_scalar_f32`. Backs
/// [`Step::ScaledAdd`] and the shared-expert down's non-MQ4 arm.
pub fn launch_scaled_add_gpu_scalar(
    gpu: &mut rdna_compute::Gpu,
    x: &GpuTensor,
    y: &GpuTensor,
    scale: &GpuTensor,
) -> Result<(), DispatchError> {
    gpu.scaled_add_inplace_gpu_scalar_f32(x, y, scale)
        .map_err(|e| DispatchError::Hip(e.to_string()))
}

/// Shared gate/up prerotation decision (extracted from `run_moe_decode`):
/// whether the already-rotated `x_rot_local` can feed the shared gate/up
/// GEMVs via [`GemvVariant::Prerotated`] instead of per-call re-rotation.
/// Requires no AWQ on either shared weight, a Prerotated post-rotation
/// variant, and an arch that has the prerotated MQ GEMV.
pub fn shared_prerotation_applies(
    x_rot_local: Option<&GpuTensor>,
    shared_gate_w: &WeightRef,
    shared_up_w: &WeightRef,
    ctx: &DispatchCtx,
) -> bool {
    x_rot_local.is_some()
        && shared_gate_w.awq_scale.is_none()
        && shared_up_w.awq_scale.is_none()
        && matches!(
            crate::types::dtype_post_rotation_variant(shared_gate_w.dtype),
            crate::types::GemvVariant::Prerotated
        )
        && matches!(
            crate::types::dtype_post_rotation_variant(shared_up_w.dtype),
            crate::types::GemvVariant::Prerotated
        )
        && crate::types::KernelKey::dtype_arch_predicate(shared_gate_w.dtype).eval_arch(ctx)
        && crate::types::KernelKey::dtype_arch_predicate(shared_up_w.dtype).eval_arch(ctx)
}

/// Fused gate-side projection (qwen decode, MQ4 gate side): one launch of
/// `fused_qkvza_hfq4g256` over the single FWHT-rotated `x_rot`, writing
/// router logits, the shared-expert scalar, and the `[0, smi)` slice views of
/// `gate_buf`/`up_buf` (the shared gate/up). Backs
/// [`Step::MoeFusedSharedGate`] and the CPU-fallback gate side.
#[allow(clippy::too_many_arguments)]
pub fn launch_fused_shared_gate(
    gpu: &mut rdna_compute::Gpu,
    router: &WeightRef<'_>,
    shared_expert_gate: &WeightRef<'_>,
    shared_gate_w: &WeightRef<'_>,
    shared_up_w: &WeightRef<'_>,
    x_rot: &GpuTensor,
    router_logits: &GpuTensor,
    scalar_buf: &GpuTensor,
    gate_buf: &GpuTensor,
    up_buf: &GpuTensor,
    smi: usize,
) -> Result<(), DispatchError> {
    // SAFETY: the slice views alias device memory owned by the caller's
    // scratch tensors (the [0, smi) shared gate/up halves of the fused
    // gate||up buffer).
    let shared_gate = unsafe { slice_f32_view(gate_buf, 0, smi) };
    let shared_up = unsafe { slice_f32_view(up_buf, 0, smi) };
    gpu.fused_qkvza_hfq4g256(
        router.buf,
        shared_expert_gate.buf,
        shared_gate_w.buf,
        shared_up_w.buf,
        x_rot,
        router_logits,
        scalar_buf,
        &shared_gate,
        &shared_up,
        router.m,
        shared_expert_gate.m,
        shared_gate_w.m,
        shared_up_w.m,
        router.k,
    )
    .map_err(|e| DispatchError::Hip(e.to_string()))
}

/// Per-weight gate-side projection (qwen decode, non-fusable gate side): the
/// router and shared-expert gate GEMVs always re-rotate from `x_norm`; the
/// shared gate/up GEMVs reuse the pre-rotated `x_rot_local` when
/// [`shared_prerotation_applies`], else re-rotate. Backs
/// [`Step::MoeSharedGateSide`] and the CPU-fallback gate side.
#[allow(clippy::too_many_arguments)]
pub fn launch_shared_gate_side(
    ctx: &DispatchCtx,
    gpu: &mut rdna_compute::Gpu,
    router: &WeightRef<'_>,
    shared_expert_gate: &WeightRef<'_>,
    shared_gate_w: &WeightRef<'_>,
    shared_up_w: &WeightRef<'_>,
    x_norm: &GpuTensor,
    x_rot_local: Option<&GpuTensor>,
    router_logits: &GpuTensor,
    scalar_buf: &GpuTensor,
    gate_buf: &GpuTensor,
    up_buf: &GpuTensor,
    smi: usize,
) -> Result<(), DispatchError> {
    static GEMV_GATE: OnceLock<GemvFamily> = OnceLock::new();
    let gemv = GEMV_GATE.get_or_init(GemvFamily::new);
    gemv.run_auto(ctx, gpu, router, x_norm, router_logits)
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    gemv.run_auto(ctx, gpu, shared_expert_gate, x_norm, scalar_buf)
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    // SAFETY: slice views alias device memory owned by the caller's scratch.
    let shared_gate = unsafe { slice_f32_view(gate_buf, 0, smi) };
    let shared_up = unsafe { slice_f32_view(up_buf, 0, smi) };
    if shared_prerotation_applies(x_rot_local, shared_gate_w, shared_up_w, ctx) {
        let xr = x_rot_local.expect("shared_prerotation_applies implies x_rot_local");
        gemv.run(
            ctx,
            gpu,
            &crate::families::gemv::GemvParams {
                w: shared_gate_w,
                x: xr,
                y: &shared_gate,
                variant: crate::types::GemvVariant::Prerotated,
                residual: None,
                gate: None,
                up: None,
            },
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
        gemv.run(
            ctx,
            gpu,
            &crate::families::gemv::GemvParams {
                w: shared_up_w,
                x: xr,
                y: &shared_up,
                variant: crate::types::GemvVariant::Prerotated,
                residual: None,
                gate: None,
                up: None,
            },
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    } else {
        gemv.run_auto(ctx, gpu, shared_gate_w, x_norm, &shared_gate)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        gemv.run_auto(ctx, gpu, shared_up_w, x_norm, &shared_up)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
    }
    Ok(())
}

/// Shared-expert down (qwen decode), extracted from `run_moe_decode` so the
/// Step program and the CPU-top-K fallback share ONE implementation.
///
/// MQ4: `ensure_mq_signs` → fused silu·mul·rotate (AWQ-aware when the weight
/// carries a scale) into the `mq_x_rot` scratch alias → residual-scaled GEMV
/// accumulating into `out_target`.
///
/// Non-MQ4 (deltanet): sigmoid(`scalar_buf`) → silu·mul of the `[0, smi)`
/// gate/up views into `ffn_hidden` → plain GEMV into `ffn_out` → scaled add
/// into `out_target`. The `ffn_hidden` slice view is created here (ephemeral).
///
/// `out_target` is the EP partial when `routed_out` is set, else `x_residual`.
/// The CPU fallback passes `p.x_residual` (it rejects `routed_out` up front)
/// and never consults `skip_shared` — identical to the legacy fallback body.
/// `ctx`, `ffn_hidden`, and `ffn_out` are consumed only by the non-MQ4 arm
/// (deltanet).
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn launch_shared_expert_down(
    ctx: &DispatchCtx,
    gpu: &mut rdna_compute::Gpu,
    w: &WeightRef<'_>,
    gate_buf: &GpuTensor,
    up_buf: &GpuTensor,
    scalar_buf: &GpuTensor,
    ffn_hidden: &GpuTensor,
    ffn_out: &GpuTensor,
    out_target: &GpuTensor,
    smi: usize,
) -> Result<(), DispatchError> {
    // The shared-down BODY, then the non-MQ4 scaled add — the exact legacy
    // sequence. The Step program expresses the same two phases as
    // [Step::MoeSharedDown, Step::ScaledAdd] (the scaled add is NOT fused
    // into the down step on the Step path); this leaf form keeps the
    // fallback's single-call shape.
    launch_shared_expert_down_body(
        ctx, gpu, w, gate_buf, up_buf, scalar_buf, ffn_hidden, ffn_out, out_target, smi,
    )?;
    if w.dtype != DType::MQ4G256 {
        launch_scaled_add_gpu_scalar(gpu, out_target, ffn_out, scalar_buf)?;
    }
    Ok(())
}

/// The shared-expert down projection body (extracted from
/// `run_moe_decode`): the MQ4 fused arm (ensure signs → fused silu·mul·rotate
/// → residual-scaled GEMV into `out_target`) or the non-MQ4 arm
/// (sigmoid → silu·mul → GEMV into `ffn_out`). The non-MQ4 arm intentionally
/// stops BEFORE the scaled add: the Step program emits the standalone
/// [`Step::ScaledAdd`] next (exact legacy launch order), and the CPU-top-K
/// fallback calls [`launch_shared_expert_down`] which appends it.
///
/// `ctx`, `ffn_hidden`, and `ffn_out` are consumed only by the non-MQ4 arm
/// (deltanet).
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn launch_shared_expert_down_body(
    ctx: &DispatchCtx,
    gpu: &mut rdna_compute::Gpu,
    w: &WeightRef<'_>,
    gate_buf: &GpuTensor,
    up_buf: &GpuTensor,
    scalar_buf: &GpuTensor,
    ffn_hidden: &GpuTensor,
    ffn_out: &GpuTensor,
    out_target: &GpuTensor,
    smi: usize,
) -> Result<(), DispatchError> {
    macro_rules! hip {
        ($e:expr) => {
            $e.map_err(|e| DispatchError::Hip(e.to_string()))
        };
    }
    // SAFETY: slice views alias device memory owned by the caller's scratch.
    let shared_gate = unsafe { slice_f32_view(gate_buf, 0, smi) };
    let shared_up = unsafe { slice_f32_view(up_buf, 0, smi) };
    if w.dtype == DType::MQ4G256 {
        hip!(gpu.ensure_mq_signs())?;
        let x_rot_alias = unsafe {
            GpuTensor {
                buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                dtype: DType::F32,
            }
        };
        if let Some(awq) = w.awq_scale {
            hip!(gpu.fused_silu_mul_rotate_mq_awq(
                &shared_gate,
                &shared_up,
                awq,
                &x_rot_alias,
                smi
            ))?;
        } else {
            hip!(gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, smi))?;
        }
        hip!(gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
            w.buf,
            &x_rot_alias,
            out_target,
            scalar_buf,
            w.m,
            w.k,
        ))?;
    } else {
        // Non-MQ4 shared expert down: only reached when the A3B shared expert
        // uses a non-MQ4 dtype. Requires deltanet for sigmoid_f32; returns
        // UnsupportedVariant for builds without the feature.
        #[cfg(feature = "deltanet")]
        {
            hip!(gpu.sigmoid_f32(scalar_buf))?;
            let shared_hid = unsafe { slice_f32_view(ffn_hidden, 0, smi) };
            hip!(gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid))?;
            static GEMV_DOWN: OnceLock<GemvFamily> = OnceLock::new();
            let gemv = GEMV_DOWN.get_or_init(GemvFamily::new);
            // Propagate run_auto's DispatchError as-is (mirrors the legacy
            // fallback body's `?`).
            gemv.run_auto(ctx, gpu, w, &shared_hid, ffn_out)?;
        }
        #[cfg(not(feature = "deltanet"))]
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "shared-down-non-mq4-requires-deltanet",
            arch: "",
            quant: "",
        });
    }
    Ok(())
}

/// Structural rejection of residual-fused grouped down projections: the
/// grouped kernel family has no residual-fused down, so a grouped down MUST
/// be the expanded projection (separate grouped combine follows). Called by
/// the [`Step::GroupedMoeGemm`] launcher before every grouped down launch.
pub fn grouped_down_projection(which: &MoeProj<'_>) -> Result<(), DispatchError> {
    match which {
        MoeProj::DownExpanded => Ok(()),
        MoeProj::DownResidual { .. } | MoeProj::DownResidualI64 { .. } => Err(DispatchError::Hip(
            "grouped down requires MoeProj::DownExpanded: the grouped kernel family has \
                 no residual-fused down; use GroupedMoeGemm(DownExpanded) + \
                 MoeCombine(inverse_perm=Some)"
                .to_string(),
        )),
        MoeProj::GateUp { .. } => Err(DispatchError::Hip(
            "grouped_down_projection: GateUp is not a down projection".to_string(),
        )),
    }
}

/// Slice a subrange of a flat F32 GpuTensor by element offset + length.
/// Mirrors `crate::pipeline::slice_moe_f32_view` — unsafe because it aliases
/// device memory.
unsafe fn slice_f32_view(src: &GpuTensor, offset_elems: usize, len_elems: usize) -> GpuTensor {
    let base = src.buf.as_ptr() as *mut u8;
    let ptr = base.add(offset_elems * 4);
    GpuTensor {
        buf: hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, len_elems * 4),
        shape: vec![len_elems],
        dtype: DType::F32,
    }
}
/// Indexed gate||up GEMV for the top-K selected experts (single token,
/// decode). Dispatches per `experts.dtype` to the exact kernel the arch
/// calls today:
/// - MQ4G256/HFQ4G256 → `gemv_hfq4g256_moe_gate_up_k8_indexed`
/// - MQ6G256/HFQ6G256 → `gemv_hfq6g256_moe_gate_up_k8_indexed`
/// - MQ2G256Lloyd      → `deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed`
/// - MQ3G256Lloyd      → `deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed`
///
/// Requires FWHT-pre-rotated `x_rot`. Output: `gate_batch` and `up_batch`
/// each `[k_top × expert_m]` f32. Call `fused_silu_mul_rotate_mq_batched_for`
/// (arch-side) between this and [`launch_indexed_down`].
#[allow(clippy::too_many_arguments)]
pub fn launch_indexed_gate_up(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    gate_batch: &GpuTensor,
    up_batch: &GpuTensor,
    k_top: usize,
) -> Result<(), DispatchError> {
    let m = 2 * experts.expert_m; // fused gate||up rows
    let k = experts.expert_k;
    match experts.dtype {
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemv_hfq4g256_moe_gate_up_k8_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ6G256 | DType::HFQ6G256 => gpu
            .gemv_hfq6g256_moe_gate_up_k8_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ2G256Lloyd => gpu
            .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ3G256Lloyd => gpu
            .deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_indexed_gate_up: unsupported dtype {other:?}"
        ))),
    }
}

/// Batched twin of [`launch_indexed_gate_up`] for the DeepSeek batched indexed
/// protocol (batch_size > 1): one launch covers `batch_size` tokens (grid
/// z = batch) via the existing MQ2-Lloyd `_batched_k4` kernel. Buffer layouts
/// are the per-token layouts stacked by token: `topk_indices` [N × k_top],
/// `x_rot` [N × hidden], `gate_batch`/`up_batch` [N × k_top × inter_local].
///
/// **MQ2G256Lloyd only** — every other dtype is rejected explicitly; there is
/// never a scalar fallback for a batched call.
#[allow(clippy::too_many_arguments)]
pub fn launch_indexed_gate_up_batched(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    gate_batch: &GpuTensor,
    up_batch: &GpuTensor,
    k_top: usize,
    batch_size: usize,
) -> Result<(), DispatchError> {
    let m = 2 * experts.expert_m; // fused gate||up rows
    let k = experts.expert_k;
    match experts.dtype {
        DType::MQ2G256Lloyd => gpu
            .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_indexed_gate_up_batched: only MQ2G256Lloyd supported, got {other:?}"
        ))),
    }
}

/// Indexed down GEMV — **expanded path**: writes per-expert outputs to
/// `down_expanded` `[batch_size × k_top × expert_k]` with no atomic
/// accumulation. A separate [`launch_moe_combine`] call folds them with
/// `topk_weights` into `ffn_out`.
///
/// Dispatches per `experts.dtype`:
/// - MQ4G256/HFQ4G256 → `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`
/// - MQ6G256/HFQ6G256 → `gemv_hfq6g256_moe_down_k8_indexed_batched_expanded`
/// - MQ2G256Lloyd      → `deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4`
///
/// **MQ3G256Lloyd is not supported here**: no `*_mq3*_moe_down_expanded_k4`
/// kernel exists. Use [`launch_indexed_down_residual`] instead for MQ3-Lloyd
/// (and optionally MQ2-Lloyd when the atomic self-combining path is preferred,
/// e.g. minimax forward.rs:767-778).
#[allow(clippy::too_many_arguments)]
pub fn launch_indexed_down(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    rot_batch: &GpuTensor,
    down_expanded: &GpuTensor,
    k_top: usize,
    batch_size: usize,
) -> Result<(), DispatchError> {
    let m = experts.expert_k; // down output = hidden
    let k = experts.expert_m; // down input  = inter
    match experts.dtype {
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                down_expanded,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ6G256 | DType::HFQ6G256 => gpu
            .gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                down_expanded,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ2G256Lloyd => gpu
            .deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                down_expanded,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_indexed_down: no expanded-k4 kernel for dtype {other:?}; \
             use launch_indexed_down_residual for Lloyd residual path"
        ))),
    }
}

/// Indexed down GEMV — **residual-scaled path** (atomic accumulate + combine
/// in one launch). Writes *directly into `ffn_out`*; no separate combine
/// step is needed. Used by minimax for MQ2-Lloyd and MQ3-Lloyd experts
/// (forward.rs:754-778).
///
/// Dispatches per `experts.dtype`:
/// - MQ2G256Lloyd → `deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed`
/// - MQ3G256Lloyd → `deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed`
///
/// This is a 5th helper beyond the brief's four: added because MQ3-Lloyd
/// has no `_expanded_k4` kernel, so [`launch_indexed_down`] cannot serve it.
/// Calling [`launch_moe_combine`] after this would double-accumulate.
#[allow(clippy::too_many_arguments)]
pub fn launch_indexed_down_residual(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    rot_batch: &GpuTensor,
    ffn_out: &GpuTensor,
    k_top: usize,
) -> Result<(), DispatchError> {
    let m = experts.expert_k; // down output = hidden
    let k = experts.expert_m; // down input  = inter
    match experts.dtype {
        DType::MQ2G256Lloyd => gpu
            .deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                experts.down_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                ffn_out,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ3G256Lloyd => gpu
            .deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
                experts.down_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                ffn_out,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_indexed_down_residual: unsupported dtype {other:?}"
        ))),
    }
}

/// Reproducible int64 down path (MQ2G256Lloyd + MQ3G256Lloyd): accumulates
/// S-scaled int64 values into `residual_i64` (pre-zeroed by the caller).
/// After conversion via `moe_i64_residual_to_f32`, the result is FP32.
/// Used on both the TP path (AllReduceI64Tp then convert) and the EP i64 path
/// (convert per rank then AllReduce{Ep} in FP32).
pub fn launch_indexed_down_residual_i64(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    rot_batch: &GpuTensor,
    residual_i64: &GpuTensor,
    k_top: usize,
) -> Result<(), DispatchError> {
    let m = experts.expert_k; // down output = hidden
    let k = experts.expert_m; // down input  = inter
    match experts.dtype {
        DType::MQ2G256Lloyd => gpu
            .moe_down_mq2g256_lloyd_residual_i64_indexed(
                experts.down_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                residual_i64,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ3G256Lloyd => gpu
            .moe_down_mq3g256_lloyd_residual_i64_indexed(
                experts.down_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                residual_i64,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_indexed_down_residual_i64: only MQ2G256Lloyd/MQ3G256Lloyd supported, got {other:?}"
        ))),
    }
}

/// Batched twin of [`launch_indexed_down_residual_i64`]: accumulates the routed
/// down projection for `batch_size` tokens into `residual_i64` [N × M] in ONE
/// launch (grid z = batch). Buffer layouts are the per-token layouts stacked by
/// token: `topk_indices`/`topk_weights` are [N × k_top], `rot_batch` is
/// [N × k_top × K], `residual_i64` is [N × M] (raw i64, S-scaled). Because i64
/// integer add is associative, the result is BIT-IDENTICAL to calling the
/// per-token variant in a loop — same partition invariance, fewer launches.
/// `residual_i64` must be zeroed by the caller before the launch.
#[allow(clippy::too_many_arguments)]
pub fn launch_indexed_down_residual_i64_batched(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    rot_batch: &GpuTensor,
    residual_i64: &GpuTensor,
    k_top: usize,
    batch_size: usize,
) -> Result<(), DispatchError> {
    let m = experts.expert_k; // down output = hidden
    let k = experts.expert_m; // down input  = inter (per-rank shard under TP)
    match experts.dtype {
        DType::MQ2G256Lloyd => gpu
            .moe_down_mq2g256_lloyd_residual_i64_indexed_batched(
                experts.down_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                residual_i64,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_indexed_down_residual_i64_batched: only MQ2G256Lloyd supported, got {other:?}"
        ))),
    }
}

/// Kernel form selected for a qwen indexed gate_up launch. `Scalar` =
/// decode kernels (batch_size == 1); `Batched` = the existing `_batched`
/// prefill kernels (batch_size > 1). Pure so the no-GPU tests can pin the
/// batch>1 → batched contract at the decision point the launcher uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenIndexedForm {
    Scalar,
    Batched,
}

/// The gate_up indexed kernel form for `batch_size`. Every dtype the qwen
/// indexed launcher serves has an existing `_batched` sister kernel, so the
/// form is purely batch-driven: batch 1 = decode scalar kernels, batch > 1 =
/// the batched prefill kernels. The launcher branches on this.
pub fn qwen_gate_up_indexed_form(dtype: DType, batch_size: usize) -> QwenIndexedForm {
    match dtype {
        DType::MQ4G256
        | DType::HFQ4G256
        | DType::MQ5G256
        | DType::MQ6G256
        | DType::HFQ6G256
        | DType::MFP4G32E8
        | DType::ParoQ4G128 => {
            if batch_size > 1 {
                QwenIndexedForm::Batched
            } else {
                QwenIndexedForm::Scalar
            }
        }
        _ => QwenIndexedForm::Scalar,
    }
}

// ── DeepSeek batched indexed protocol (Phase 3 shared lane) ───────────────
// `Step::IndexedMoeGemv.batch_size` is authoritative: batch one keeps the
// existing scalar launchers byte-identically, batch > 1 selects the MQ2-Lloyd
// batched kernels, and anything else rejects explicitly — there is NEVER a
// scalar fallback for a batched form. These pure selectors are the exact
// decision point the step executor branches on, so the no-GPU tests pin the
// batch>1 contract without pretending to launch kernels.

/// Kernel form for a DeepSeek batched indexed projection. `Scalar` keeps the
/// existing per-token launcher; `Batched` selects the MQ2 `_batched_k4`
/// kernels; `Unsupported` is an explicit rejection (zero batch, or a
/// non-MQ2-Lloyd dtype at batch > 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepSeekIndexedForm {
    Scalar,
    Batched,
    Unsupported,
}

/// Routed gate-up form for `IndexedMoeGemv::GateUp`. Batch one uses the
/// scalar launcher for every dtype it serves; batch > 1 is batched ONLY for
/// MQ2G256Lloyd (the existing `_batched_k4` kernel). Zero batch is
/// `Unsupported` and rejected before dispatch.
pub fn deepseek_gate_up_indexed_form(dtype: DType, batch_size: usize) -> DeepSeekIndexedForm {
    match batch_size {
        0 => DeepSeekIndexedForm::Unsupported,
        1 => DeepSeekIndexedForm::Scalar,
        _ => match dtype {
            DType::MQ2G256Lloyd => DeepSeekIndexedForm::Batched,
            _ => DeepSeekIndexedForm::Unsupported,
        },
    }
}

/// Reproducible int64 down form for `IndexedMoeGemv::DownResidualI64`. Batch
/// one keeps the scalar launcher for EVERY dtype — the launcher's own dtype
/// validation is the authority, so an unsupported scalar dtype reaches the
/// exact same launcher error it always has (never a batched/unrecognized
/// form error). Batch > 1 is the MQ2-Lloyd `_batched_k4` launcher only; every
/// other batched dtype is `Unsupported` with no scalar fallback. Zero batch
/// is `Unsupported`.
pub fn deepseek_i64_down_indexed_form(dtype: DType, batch_size: usize) -> DeepSeekIndexedForm {
    match batch_size {
        0 => DeepSeekIndexedForm::Unsupported,
        1 => DeepSeekIndexedForm::Scalar,
        _ => match dtype {
            DType::MQ2G256Lloyd => DeepSeekIndexedForm::Batched,
            _ => DeepSeekIndexedForm::Unsupported,
        },
    }
}

/// FP32 residual-fused down form for `IndexedMoeGemv::DownResidual`. Batch
/// one keeps the scalar launcher; batch > 1 is explicitly rejected (no
/// batched FP32 residual kernel exists and no scalar fallback is permitted).
pub fn deepseek_f32_down_indexed_form(batch_size: usize) -> DeepSeekIndexedForm {
    match batch_size {
        0 => DeepSeekIndexedForm::Unsupported,
        1 => DeepSeekIndexedForm::Scalar,
        _ => DeepSeekIndexedForm::Unsupported,
    }
}

/// Zero-batch guard for `Step::IndexedMoeGemv`: the step's `batch_size` is
/// authoritative and a zero routed batch is rejected before any launcher
/// dispatch. Pure so the no-GPU tests pin the contract at the exact decision
/// point the step executor uses.
pub fn indexed_moe_batch_guard(batch_size: usize) -> Result<(), DispatchError> {
    if batch_size == 0 {
        Err(DispatchError::Hip(
            "IndexedMoeGemv: batch_size must be nonzero before dispatch".into(),
        ))
    } else {
        Ok(())
    }
}

/// Qwen routed gate_up indexed GEMV (STEP-002 Phase 1). Covers the forms
/// [`launch_indexed_gate_up`] cannot express: Paro (Givens), MFP4-E8, MQ5,
/// per-expert mixed dtype tags, and the batched prefill (Path 1) forms.
///
/// `batch_size == 1` selects the decode kernels; `> 1` the batched kernels.
/// `dtype_tags = Some` selects the merged per-expert mixed kernel — decode
/// only; graded prefill runs the grouped path, so a tagged batched call is
/// rejected (mirrors the legacy prefill dispatch, which never passes tags to
/// the indexed gate_up).
///
/// `m` = 2·expert_m (fused gate||up rows), `k` = expert_k (hidden). The Paro
/// and E8 kernels are k8-implicit and take no `k_top`.
#[allow(clippy::too_many_arguments)]
pub fn launch_qwen_gate_up_indexed(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    gate_batch: &GpuTensor,
    up_batch: &GpuTensor,
    k_top: usize,
    batch_size: usize,
    dtype_tags: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    let m = 2 * experts.expert_m;
    let k = experts.expert_k;
    if let Some(tags) = dtype_tags {
        if batch_size != 1 {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "mixed-indexed-gate-up-batched-unsupported (graded prefill uses Path 2)",
                arch: "",
                quant: "",
            });
        }
        return gpu
            .gemv_mixed_moe_gate_up_k8_indexed_batched(
                experts.gate_up_ptrs,
                tags,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
                1,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()));
    }
    // Single decision point for scalar-vs-batched (the pure form selector the
    // no-GPU tests pin): batch 1 = decode scalar kernels, batch > 1 = the
    // existing `_batched` prefill kernels. Every dtype has a batched sister.
    match (
        experts.dtype,
        qwen_gate_up_indexed_form(experts.dtype, batch_size),
    ) {
        (DType::ParoQ4G128, QwenIndexedForm::Scalar) => gpu
            .gemv_paro_q4g128_moe_gate_up_k8_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::ParoQ4G128, QwenIndexedForm::Batched) => gpu
            .gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MFP4G32E8, QwenIndexedForm::Scalar) => gpu
            .gemv_mfp4g32_e8_moe_gate_up_k8_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MFP4G32E8, QwenIndexedForm::Batched) => gpu
            .gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MQ5G256, QwenIndexedForm::Scalar) => gpu
            .gemv_hfq5g256_moe_gate_up_k8_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MQ5G256, QwenIndexedForm::Batched) => gpu
            .gemv_hfq5g256_moe_gate_up_k8_indexed_batched(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MQ4G256 | DType::HFQ4G256, QwenIndexedForm::Scalar) => gpu
            .gemv_hfq4g256_moe_gate_up_k8_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MQ4G256 | DType::HFQ4G256, QwenIndexedForm::Batched) => gpu
            .gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MQ6G256 | DType::HFQ6G256, QwenIndexedForm::Scalar) => gpu
            .gemv_hfq6g256_moe_gate_up_k8_indexed(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (DType::MQ6G256 | DType::HFQ6G256, QwenIndexedForm::Batched) => gpu
            .gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                experts.gate_up_ptrs,
                topk_indices,
                x_rot,
                gate_batch,
                up_batch,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        (other, _) => Err(DispatchError::Hip(format!(
            "launch_qwen_gate_up_indexed: unsupported dtype {other:?}"
        ))),
    }
}

/// Qwen routed down indexed GEMV (STEP-002 Phase 1). Covers the forms
/// [`launch_indexed_down`] cannot express (Paro, E8, MQ5, mixed tags) plus
/// the batched prefill forms and the atomic residual-scaled Path 0 down.
///
/// `m` = expert_k (hidden), `k` = expert_m (inter). `batch_size == 1`
/// selects the decode kernels; `> 1` the batched kernels.
///
/// - [`QwenDownMode::Expanded`]: writes per-expert outputs to `out`
///   (`down_expanded`); a separate [`launch_moe_combine`] follows.
/// - [`QwenDownMode::ResidualScaled`]: atomic weighted accumulation into
///   `out` (the EP partial / `x_batch`); no combine follows (MQ4 only,
///   prefill Path 0).
///
/// `dtype_tags = Some` selects the merged per-expert mixed kernel — decode
/// only (`batch_size == 1`); graded prefill runs the grouped path.
#[allow(clippy::too_many_arguments)]
pub fn launch_qwen_down_indexed(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    topk_indices: &GpuTensor,
    rot_batch: &GpuTensor,
    out: &GpuTensor,
    k_top: usize,
    batch_size: usize,
    mode: &QwenDownMode<'_>,
    dtype_tags: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    let m = experts.expert_k; // down output = hidden
    let k = experts.expert_m; // down input  = inter
    if let QwenDownMode::ResidualScaled { topk_weights } = mode {
        // Atomic residual-scaled accumulation (prefill Path 0): MQ4 only —
        // mirrors the legacy prefill Path 0 dispatch.
        return match experts.dtype {
            DType::MQ4G256 => gpu
                .gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
                    experts.down_ptrs,
                    topk_indices,
                    topk_weights,
                    rot_batch,
                    out,
                    m,
                    k,
                    k_top,
                    batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string())),
            _other => Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "prefill-down-path0-dtype",
                arch: "",
                quant: "",
            }),
        };
    }
    if let Some(tags) = dtype_tags {
        if batch_size != 1 {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "mixed-indexed-down-batched-unsupported (graded prefill uses Path 2)",
                arch: "",
                quant: "",
            });
        }
        return gpu
            .gemv_mixed_moe_down_k8_indexed_batched_expanded(
                experts.down_ptrs,
                tags,
                topk_indices,
                rot_batch,
                out,
                m,
                k,
                k_top,
                1,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()));
    }
    match experts.dtype {
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                out,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ5G256 => gpu
            .gemv_hfq5g256_moe_down_k8_indexed_batched_expanded(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                out,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ6G256 | DType::HFQ6G256 => gpu
            .gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                out,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MFP4G32E8 => gpu
            .gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                out,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::ParoQ4G128 => gpu
            .gemv_paro_q4g128_moe_down_k8_indexed_batched(
                experts.down_ptrs,
                topk_indices,
                rot_batch,
                out,
                m,
                k,
                k_top,
                batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        _other => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "prefill-down-path1-dtype",
            arch: "",
            quant: "",
        }),
    }
}

/// Weighted combine of per-expert expanded down outputs into `ffn_out`.
/// Thin wrapper over `gpu.moe_down_combine_k8_batched`. Call after
/// [`launch_indexed_down`] (the expanded path). Do NOT call after
/// [`launch_indexed_down_residual`] — that path already accumulates.
pub fn launch_moe_combine(
    gpu: &mut rdna_compute::Gpu,
    down_expanded: &GpuTensor,
    topk_weights: &GpuTensor,
    ffn_out: &GpuTensor,
    hidden: usize,
    k_top: usize,
    batch_size: usize,
) -> Result<(), DispatchError> {
    gpu.moe_down_combine_k8_batched(
        down_expanded,
        topk_weights,
        ffn_out,
        hidden,
        k_top,
        batch_size,
    )
    .map_err(|e| DispatchError::Hip(e.to_string()))
}

// ── Prefill grouped-GEMM launch helpers (Task 5) ─────────────────────────────

/// Frozen grouped-GEMM block width (WMMA tile row count) shared by the Qwen
/// builder, the direct DeepSeek grouped dispatch, and the runtime grouped
/// grammar: every grouped bound is a multiple of 16 and the tile count is
/// m_total_max / 16.
pub const MOE_GROUPED_BLOCK_M: usize = 16;

/// Checked grouped geometry for the shared DeepSeek batched protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeepSeekGroupedBounds {
    pub total_slots: usize,
    pub m_total_max: usize,
    pub tile_count: usize,
}

/// Checked grouped bound formula: `total_slots = batch*k`,
/// `expert_pad = n_experts*MOE_GROUPED_BLOCK_M`, `raw = total_slots +
/// expert_pad`, `aligned = checked align-up(raw, 16)`, `tiles = aligned/16`.
/// Fails closed on multiply/add/align overflow. Zero batch/k_top keeps the
/// expert-pad-only allocation — the same degenerate shape the existing
/// direct dispatch path launches — so the formula stays defined and the
/// launch-side zero gating is unchanged.
pub fn checked_deepseek_grouped_bounds(
    batch_size: usize,
    k_top: usize,
    n_experts: usize,
) -> Result<DeepSeekGroupedBounds, DispatchError> {
    let total_slots = batch_size.checked_mul(k_top).ok_or_else(|| {
        DispatchError::Hip("checked_deepseek_grouped_bounds: batch_size*k_top overflow".into())
    })?;
    let expert_pad = n_experts.checked_mul(MOE_GROUPED_BLOCK_M).ok_or_else(|| {
        DispatchError::Hip("checked_deepseek_grouped_bounds: n_experts*16 overflow".into())
    })?;
    let raw = total_slots.checked_add(expert_pad).ok_or_else(|| {
        DispatchError::Hip("checked_deepseek_grouped_bounds: total_slots+pad overflow".into())
    })?;
    let aligned = raw
        .checked_add(MOE_GROUPED_BLOCK_M - 1)
        .map(|v| (v / MOE_GROUPED_BLOCK_M) * MOE_GROUPED_BLOCK_M)
        .ok_or_else(|| {
            DispatchError::Hip("checked_deepseek_grouped_bounds: align-up overflow".into())
        })?;
    Ok(DeepSeekGroupedBounds {
        total_slots,
        m_total_max: aligned,
        tile_count: aligned / MOE_GROUPED_BLOCK_M,
    })
}

/// Pure guard for the scatter launcher: `block_m` must be nonzero and divide
/// `m_total_max` exactly (the tile count would truncate otherwise). Checked
/// before any GPU work.
pub fn scatter_block_guard(block_m: usize, m_total_max: usize) -> Result<(), DispatchError> {
    if block_m == 0 || m_total_max % block_m != 0 {
        return Err(DispatchError::Hip(format!(
            "launch_moe_scatter: block_m={block_m} must be nonzero and divide m_total_max={m_total_max}"
        )));
    }
    Ok(())
}

/// Thin wrapper over `gpu.moe_scatter_fused_k8`.
/// Produces `sorted_slot_index`, `expert_tile_ids`, and `inverse_perm` from
/// `topk_indices`; also fills the histogram (`expert_token_counts`) and the
/// exclusive-scan offsets (`expert_offsets`). Must run before the grouped
/// GEMMs. The block geometry guard runs before any GPU work.
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_scatter(
    gpu: &mut rdna_compute::Gpu,
    topk_indices: &GpuTensor,
    expert_token_counts: &GpuTensor,
    expert_offsets: &GpuTensor,
    sorted_slot_index: &GpuTensor,
    expert_tile_ids: &GpuTensor,
    inverse_perm: &GpuTensor,
    total_slots: usize,
    n_experts: usize,
    m_total_max: usize,
    block_m: usize,
) -> Result<(), DispatchError> {
    scatter_block_guard(block_m, m_total_max)?;
    gpu.moe_scatter_fused_k8(
        topk_indices,
        expert_token_counts,
        expert_offsets,
        sorted_slot_index,
        expert_tile_ids,
        inverse_perm,
        total_slots,
        n_experts,
        m_total_max,
        block_m,
    )
    .map_err(|e| DispatchError::Hip(e.to_string()))
}

/// Grouped gate||up GEMM (Path 2 prefill): one launch covers all expert tokens
/// sorted by `sorted_slot_index`. Thin wrapper over the shared
/// [`crate::pipeline::dispatch_grouped_gemm`] — the same dispatch helper the
/// legacy `run_moe_prefill` used — so the Step program can never drift from
/// the production kernel selection.
///
/// `dtype_tags` (graded files), `force_mq4_fp16`, `paro_i8`, `paro_i8_k8`
/// carry the grouped controls from `MoePrefillResolution`.
///
/// Dims: `m = 2 * experts.expert_m`, `k = experts.expert_k`,
/// `x_row_div = k_top` (gate_up slots = N·k_top, divided by k_top → N rows of x),
/// `rows = batch_size` (number of input tokens).
#[allow(clippy::too_many_arguments)]
pub fn launch_grouped_gate_up(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    sorted_slot_index: &GpuTensor,
    expert_tile_ids: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m_total: usize,
    k_top: usize,
    batch_size: usize,
    dtype_tags: Option<&GpuTensor>,
    force_mq4_fp16: bool,
    paro_i8: bool,
    paro_i8_k8: bool,
) -> Result<(), DispatchError> {
    let m = 2 * experts.expert_m; // fused gate||up rows
    let k = experts.expert_k;
    crate::pipeline::dispatch_grouped_gemm(
        gpu,
        experts.dtype,
        dtype_tags,
        experts.gate_up_ptrs,
        expert_tile_ids,
        sorted_slot_index,
        x,
        y,
        m,
        k,
        k_top,
        m_total,
        batch_size,
        force_mq4_fp16,
        paro_i8,
        paro_i8_k8,
    )
}

/// Grouped down GEMM (Path 2 prefill): one launch covers all expert tokens
/// sorted by `sorted_slot_index`. Thin wrapper over the shared
/// [`crate::pipeline::dispatch_grouped_gemm`] (same kernels as gate_up,
/// different dims).
///
/// Dims: `m = experts.expert_k` (hidden), `k = experts.expert_m` (inter),
/// `x_row_div = 1` (every row of rot_batch is a distinct slot),
/// `rows = batch_size * k_top`.
#[allow(clippy::too_many_arguments)]
pub fn launch_grouped_down(
    gpu: &mut rdna_compute::Gpu,
    experts: &MoeExpertRef<'_>,
    sorted_slot_index: &GpuTensor,
    expert_tile_ids: &GpuTensor,
    x: &GpuTensor, // rot_batch [batch*k_top × inter]
    y: &GpuTensor, // y_down_grouped [m_total × hidden]
    m_total: usize,
    k_top: usize,
    batch_size: usize,
    dtype_tags: Option<&GpuTensor>,
    force_mq4_fp16: bool,
    paro_i8: bool,
    paro_i8_k8: bool,
) -> Result<(), DispatchError> {
    let m = experts.expert_k; // down output = hidden
    let k = experts.expert_m; // down input  = inter
    crate::pipeline::dispatch_grouped_gemm(
        gpu,
        experts.dtype,
        dtype_tags,
        experts.down_ptrs,
        expert_tile_ids,
        sorted_slot_index,
        x,
        y,
        m,
        k,
        1, /* x_row_div */
        m_total,
        batch_size * k_top,
        force_mq4_fp16,
        paro_i8,
        paro_i8_k8,
    )
}

/// Deinterleave grouped gate_up result: `y_grouped → gate_batch + up_batch`.
/// Thin wrapper over `gpu.moe_gate_up_unscatter_k8`.
/// Call after [`launch_grouped_gate_up`] (before SwiGLU+rotate).
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_gate_up_unscatter(
    gpu: &mut rdna_compute::Gpu,
    y_grouped: &GpuTensor,
    sorted_slot_index: &GpuTensor,
    gate_batch: &GpuTensor,
    up_batch: &GpuTensor,
    inter: usize,
    k_top: usize,
    m_total: usize,
) -> Result<(), DispatchError> {
    gpu.moe_gate_up_unscatter_k8(
        y_grouped,
        sorted_slot_index,
        gate_batch,
        up_batch,
        inter,
        k_top,
        m_total,
    )
    .map_err(|e| DispatchError::Hip(e.to_string()))
}

/// Weighted combine for the grouped prefill down path. Reads `y_down_grouped`
/// via `inverse_perm` and accumulates into `out` (the EP partial or `x_batch`).
/// Thin wrapper over `gpu.moe_down_combine_grouped_k8`.
/// Call after [`launch_grouped_down`]; do NOT call [`launch_moe_combine`]
/// (the decode path) after a grouped down — the combine kernels differ.
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_combine_grouped(
    gpu: &mut rdna_compute::Gpu,
    y_down_grouped: &GpuTensor,
    inverse_perm: &GpuTensor,
    topk_weights: &GpuTensor,
    out: &GpuTensor,
    hidden: usize,
    k_top: usize,
    batch_size: usize,
) -> Result<(), DispatchError> {
    gpu.moe_down_combine_grouped_k8(
        y_down_grouped,
        inverse_perm,
        topk_weights,
        out,
        hidden,
        k_top,
        batch_size,
    )
    .map_err(|e| DispatchError::Hip(e.to_string()))
}

// ── ds4 MoE tail launch helper (Task 6) ─────────────────────────────────────

/// ds4 MoE tail: mixes the EP all-reduced `ffn_out` partial into `residual_streams`.
///
/// This is an **arch-owned tail hook**, NOT a `Step` variant. The two view
/// operands (`comb_view = hc_c.sub_offset(8, 16)` and
/// `post_view = hc_c.sub_offset(4, 4)`) are ephemeral `GpuTensor` values with
/// no stable backing storage; they cannot be `&'a GpuTensor` borrows in a Step
/// (see the note at the end of the `Step` enum). Task 8's `forward_ep` calls
/// this helper directly after `execute_steps_parallel` returns.
///
/// Reproduces exactly the `hc_mix_4stream` + `memcpy_dtod_auto` sequence from
/// `hipfire-arch-deepseek4::forward::hc_ffn_mix` (forward.rs:3747–3759).
///
/// # Arguments
/// - `streams`     — `state.residual_streams`: `[4, hidden]` HC stream bank.
/// - `hc_c`        — `state.hc_c`: parent buffer; `post_view` and `comb_view`
///                   are derived via `sub_offset` using the fixed offsets from
///                   `hc_ffn_mix` (post @ elem 4 len 4; comb @ elem 8 len 16).
/// - `ffn_out`     — `state.ffn_out`: `[hidden]` MoE partial (already all-reduced).
/// - `streams_out` — `state.q`: `[4, hidden]` scratch for the kernel output.
/// - `hidden`      — `cfg.hidden_size`.
/// - `hc_bytes`    — `cfg.hc_mult * cfg.hidden_size * 4` (byte count for D2D copy).
pub fn launch_hc_ffn_mix(
    gpu: &mut rdna_compute::Gpu,
    streams: &GpuTensor,
    hc_c: &GpuTensor,
    ffn_out: &GpuTensor,
    streams_out: &GpuTensor,
    hidden: usize,
    hc_bytes: usize,
) -> Result<(), DispatchError> {
    // Reproduce exactly the sub_offset layout from hc_ffn_mix (forward.rs:3743-3744).
    // post_view → [4] fp32 per-stream scale   (hc_c elems [4..8]).
    // comb_view → [4, 4] fp32 Sinkhorn matrix (hc_c elems [8..24]).
    let post_view = hc_c.sub_offset(4, 4);
    let comb_view = hc_c.sub_offset(8, 16);
    gpu.hc_mix_4stream(
        streams,
        &comb_view,
        &post_view,
        ffn_out,
        streams_out,
        hidden as i32,
    )
    .map_err(|e| DispatchError::Hip(format!("hc_mix_4stream (ffn tail): {e:?}")))?;
    gpu.memcpy_dtod_auto(&streams.buf, &streams_out.buf, hc_bytes)
        .map_err(|e| DispatchError::Hip(format!("hc_ffn_mix d2d streams←streams_out: {e:?}")))
}

// ── Qwen MoE Step-native program (STEP-002 Phase 1) ─────────────────────────

/// Borrowed structured phase decomposition of one Qwen MoE layer program.
///
/// The phases mirror the legacy `run_moe_decode` / `run_moe_prefill` launch
/// order: a decode layer is `rotate → gate_side → route → shared_down →
/// routed`; a prefill layer keeps the first four phases model-owned
/// (`None`/empty) and carries the whole routed block (Path 0/1/2) in
/// `routed`. Flattening ([`MoeStepPhases::into_build`]) emits the exact
/// current launch order.
pub struct MoeStepPhases<'a> {
    /// Scalar activation rotation (decode only): FWHT for MQ/E8 tiers,
    /// Givens for Paro. `None` when the model pre-rotates (`x_rot_prerotated`)
    /// or no rotation is needed.
    pub rotate: Option<Step<'a>>,
    /// Gate-side projection (decode only): one [`Step::MoeFusedSharedGate`]
    /// for the MQ4 fused gate side, else one [`Step::MoeSharedGateSide`]
    /// (the four per-weight GEMVs).
    pub gate_side: Vec<Step<'a>>,
    /// Softmax + renormalized top-K (decode only; prefill routing is
    /// model-owned).
    pub route: Option<Step<'a>>,
    /// Shared-expert down (decode only; empty when `skip_shared`). MQ4
    /// shared down is one [`Step::MoeSharedDown`]; the non-MQ4 form decomposes
    /// honestly into [`Step::MoeSharedDown`] followed by the standalone
    /// [`Step::ScaledAdd`]. The prefill shared expert stays model-owned.
    pub shared_down: Vec<Step<'a>>,
    /// Routed-expert block in exact launch order:
    /// - decode: [gate_up, activation, down, combine?]
    /// - prefill Path 2: [scatter, (givens), grouped gate_up, unscatter,
    ///   activation, grouped down, grouped combine]
    /// - prefill Path 0: [gate_up, activation, atomic down] (no combine)
    /// - prefill Path 1: [gate_up, activation, down, combine]
    pub routed: Vec<Step<'a>>,
    /// Index into `routed` of the step writing the EP partial (the routed
    /// combine, or the residual-fused/atomic down), when `routed_out` is set.
    /// `None` = single-GPU accumulation into the residual stream directly.
    pub routed_partial: Option<usize>,
}

impl<'a> MoeStepPhases<'a> {
    /// Flatten into the executable program. The step sequence is exactly the
    /// current launch order; `ep_partial` maps the routed partial index onto
    /// the flattened step list.
    pub fn into_build(self) -> MoeStepProgram<'a> {
        let prefix_len = usize::from(self.rotate.is_some())
            + self.gate_side.len()
            + usize::from(self.route.is_some())
            + self.shared_down.len();
        let ep_partial = self.routed_partial.map(|i| prefix_len + i);
        let mut steps = Vec::with_capacity(prefix_len + self.routed.len());
        steps.extend(self.rotate);
        steps.extend(self.gate_side);
        steps.extend(self.route);
        steps.extend(self.shared_down);
        steps.extend(self.routed);
        MoeStepProgram { steps, ep_partial }
    }
}

/// Built Qwen MoE layer program: the flattened step list plus the EP-partial
/// metadata the mesh executor needs.
pub struct MoeStepProgram<'a> {
    /// All Steps in exact current launch order.
    pub steps: Vec<Step<'a>>,
    /// Index into `steps` of the step whose `out` is the EP partial (routed
    /// combine, or residual-fused/atomic down) when the layer redirects
    /// routed output (`routed_out`). `None` = accumulate into the residual
    /// stream directly (single GPU).
    pub ep_partial: Option<usize>,
}

/// Build selection for one Qwen MoE decode layer (STEP-002 binding contract).
///
/// - [`MoeStepBuild::Gpu`]: the Step-native program, selected EXACTLY when
///   `resolution.use_gpu_topk` (k == 8 with an indexable routed dtype).
/// - [`MoeStepBuild::CpuFallback`]: the explicit non-Step leaf, selected
///   EXACTLY when `!resolution.use_gpu_topk`. `run_moe_decode` matches it and
///   calls the preserved CPU-top-K fallback (with the gate-side projection
///   run directly beforehand), never a Step program.
pub enum MoeStepBuild<'a> {
    /// Step-native GPU program (built when `resolution.use_gpu_topk`).
    Gpu(MoeStepPhases<'a>),
    /// Explicit non-Step leaf: the CPU-top-K fallback (k != 8 or a
    /// non-indexable routed dtype).
    CpuFallback,
}

impl<'a> MoeStepBuild<'a> {
    /// True when the layer selects the explicit CPU-top-K fallback leaf.
    pub fn is_cpu_fallback(&self) -> bool {
        matches!(self, MoeStepBuild::CpuFallback)
    }

    /// Unwrap the Step-native phases; `None` for the fallback selection.
    pub fn into_gpu(self) -> Option<MoeStepPhases<'a>> {
        match self {
            MoeStepBuild::Gpu(phases) => Some(phases),
            MoeStepBuild::CpuFallback => None,
        }
    }
}

/// Expert reference pair for the decode program. The gate_up and down steps
/// carry their OWN representative dtype (the "mq6-down" lever mixes gate_up
/// MQ4 with down MQ6), so two refs. Owned by the CALLER (the Steps borrow
/// them), which is why the builders take them as parameters.
pub fn decode_expert_refs<'a>(p: &'a MoeParams<'a>) -> (MoeExpertRef<'a>, MoeExpertRef<'a>) {
    (
        MoeExpertRef {
            gate_up_ptrs: p.expert_gate_up_ptrs,
            down_ptrs: p.expert_down_ptrs,
            dummy_gate_up: None,
            dtype: p.dtypes.routed_gate_up,
            n_experts: p.n_exp,
            expert_m: p.mi,
            expert_k: p.routed_gate_up_k,
            owned: &[],
        },
        MoeExpertRef {
            gate_up_ptrs: p.expert_gate_up_ptrs,
            down_ptrs: p.expert_down_ptrs,
            dummy_gate_up: None,
            dtype: p.dtypes.routed_down,
            n_experts: p.n_exp,
            expert_m: p.routed_down_k,
            expert_k: p.routed_down_m,
            owned: &[],
        },
    )
}

/// Expert reference pair for the prefill program (same borrow contract as
/// [`decode_expert_refs`]; dims come from the prefill parameter set).
pub fn prefill_expert_refs<'a>(
    p: &'a MoePrefillParams<'a>,
) -> (MoeExpertRef<'a>, MoeExpertRef<'a>) {
    (
        MoeExpertRef {
            gate_up_ptrs: p.expert_gate_up_ptrs,
            down_ptrs: p.expert_down_ptrs,
            dummy_gate_up: None,
            dtype: p.dtypes.routed_gate_up,
            n_experts: p.n_exp,
            expert_m: p.mi,
            expert_k: p.gate_up_k,
            owned: &[],
        },
        MoeExpertRef {
            gate_up_ptrs: p.expert_gate_up_ptrs,
            down_ptrs: p.expert_down_ptrs,
            dummy_gate_up: None,
            dtype: p.dtypes.routed_down,
            n_experts: p.n_exp,
            expert_m: p.down_k,
            expert_k: p.down_m,
            owned: &[],
        },
    )
}

/// Build the decode Step program for a Qwen MoE layer (GPU top-K path).
///
/// `res` must satisfy `res.use_gpu_topk` — the CPU-top-K fallback is an
/// explicit NON-Step leaf (`run_moe_decode` calls it directly) and this
/// builder refuses to express it as a program.
///
/// `gu_experts`/`dn_experts` are the caller-owned expert refs (see
/// [`decode_expert_refs`]); the Steps borrow them.
///
/// The step sequence reproduces the legacy `run_moe_decode` kernel launch
/// order exactly: rotate → gate-side → softmax+top-K → shared down →
/// [gate_up → activation → down → combine?].
pub fn build_moe_decode_steps<'a>(
    p: &'a MoeParams<'a>,
    res: &MoeResolution,
    gu_experts: &'a MoeExpertRef<'a>,
    dn_experts: &'a MoeExpertRef<'a>,
) -> Result<MoeStepBuild<'a>, DispatchError> {
    // Binding contract: the CPU-top-K fallback is an explicit NON-Step leaf —
    // selected exactly when `!use_gpu_topk` — never expressed as a program.
    if !res.use_gpu_topk {
        return Ok(MoeStepBuild::CpuFallback);
    }
    let out_target: &'a GpuTensor = p.routed_out.unwrap_or(p.x_residual);
    let mut phases = MoeStepPhases {
        rotate: None,
        gate_side: Vec::new(),
        route: None,
        shared_down: Vec::new(),
        routed: Vec::new(),
        routed_partial: None,
    };

    // ── 1. Activation rotation (mirrors the legacy x_rot_local block) ─────
    if res.needs_x_rot_local && !p.x_rot_prerotated {
        phases.rotate = Some(if res.routed_indexable_paro {
            let paro = p
                .routed_gate_up_paro
                .as_ref()
                .ok_or(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "decode-step-build-paro-without-gate-up-sidecar",
                    arch: "",
                    quant: "",
                })?;
            Step::GivensRotateBatched {
                x: p.x_norm,
                out: p.x_rot_local,
                pairs: paro.pairs,
                theta: paro.theta,
                scales: paro.scales,
                batch: 1,
                dim: p.hidden,
                krot: paro.krot,
            }
        } else {
            // gate_side_mq4 routes the router AWQ scale onto the shared
            // rotation; every other form rotates plain.
            let awq_scale = if res.gate_side_mq4 {
                p.router.awq_scale
            } else {
                None
            };
            Step::RotateFwhtBatched {
                x: p.x_norm,
                out: p.x_rot_local,
                awq_scale,
                k: p.hidden,
                batch: 1,
            }
        });
    }
    let xr = if res.needs_x_rot_local {
        Some(p.x_rot_local)
    } else {
        None
    };

    // ── 2. Gate side ──────────────────────────────────────────────────────
    if res.gate_fusable {
        phases.gate_side.push(Step::MoeFusedSharedGate {
            router: &p.router,
            shared_expert_gate: &p.shared_expert_gate,
            shared_gate_w: &p.shared_gate_w,
            shared_up_w: &p.shared_up_w,
            x_rot: xr.expect("gate_fusable implies x_rot_local"),
            router_logits: p.router_logits,
            scalar_buf: p.scalar_buf,
            gate_buf: p.gate_buf,
            up_buf: p.up_buf,
            smi: p.smi,
        });
    } else {
        phases.gate_side.push(Step::MoeSharedGateSide {
            router: &p.router,
            shared_expert_gate: &p.shared_expert_gate,
            shared_gate_w: &p.shared_gate_w,
            shared_up_w: &p.shared_up_w,
            x_norm: p.x_norm,
            x_rot_local: xr,
            router_logits: p.router_logits,
            scalar_buf: p.scalar_buf,
            gate_buf: p.gate_buf,
            up_buf: p.up_buf,
            smi: p.smi,
        });
    }

    // ── 3. Softmax + renormalized top-K ───────────────────────────────────
    phases.route = Some(Step::MoeSoftmaxTopK {
        logits: p.router_logits,
        topk_indices: p.topk_indices,
        topk_weights: p.topk_weights,
        n_exp: p.n_exp,
        norm_topk_prob: p.norm_topk_prob,
    });

    // ── 4. Shared-expert down (EP: skipped on rank>0) ─────────────────────
    // MQ4 shared down is one fused step; the non-MQ4 form is decomposed
    // honestly into the down step (sigmoid → silu·mul → GEMV) followed by
    // the standalone scaled-add step — the exact legacy launch order, with
    // no multi-launch opaque step and no dead scaled-add variant.
    if !p.skip_shared {
        phases.shared_down.push(Step::MoeSharedDown {
            w: &p.shared_down_w,
            gate_buf: p.gate_buf,
            up_buf: p.up_buf,
            scalar_buf: p.scalar_buf,
            ffn_hidden: p.ffn_hidden,
            ffn_out: p.ffn_out,
            out_target,
            smi: p.smi,
        });
        if p.shared_down_w.dtype != DType::MQ4G256 {
            phases.shared_down.push(Step::ScaledAdd {
                x: out_target,
                y: p.ffn_out,
                scale: p.scalar_buf,
            });
        }
    }

    // ── 5. Routed experts ─────────────────────────────────────────────────
    let tags = p.expert_dtype_tags;

    // gate_up: Paro, graded-mixed, MQ5, and E8 need the qwen indexed step;
    // uniform MQ4/MQ6/MQ2L/MQ3L reuse IndexedMoeGemv (identical kernels).
    // A graded file with UNIFORM MQ4 gate_up takes the fast uniform MQ4
    // GEMV (the tag table only drives the graded down).
    let gate_up_mixed = tags.is_some() && !p.dtypes.experts_all_gate_up_mq4;
    if res.routed_indexable_paro
        || gate_up_mixed
        || matches!(p.dtypes.routed_gate_up, DType::MQ5G256 | DType::MFP4G32E8)
    {
        phases.routed.push(Step::MoeGateUpIndexed {
            experts: &gu_experts,
            topk_indices: p.topk_indices,
            x_rot: xr.expect("use_gpu_topk implies x_rot_local"),
            gate_batch: p.gate_batch,
            up_batch: p.up_batch,
            k_top: p.k,
            batch_size: 1,
            dtype_tags: gate_up_mixed.then_some(tags).flatten(),
        });
    } else {
        phases.routed.push(Step::IndexedMoeGemv {
            experts: &gu_experts,
            which: MoeProj::GateUp { up_out: p.up_batch },
            topk_indices: p.topk_indices,
            input: GemvInput::Prerotated(xr.expect("use_gpu_topk implies x_rot_local")),
            out: p.gate_batch,
            k_top: p.k,
            batch_size: 1,
        });
    }

    // activation: Paro → Givens; per-expert AWQ → indexed AWQ; else plain.
    let activation = if res.routed_indexable_paro {
        let paro = p
            .routed_down_paro
            .as_ref()
            .ok_or(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "decode-step-build-paro-without-down-sidecar",
                arch: "",
                quant: "",
            })?;
        MoeActivationVariant::QwenParo {
            pairs: paro.pairs,
            theta: paro.theta,
            scales: paro.scales,
            krot: paro.krot,
        }
    } else if let Some(awq_ptrs) = p.expert_down_awq_ptrs {
        MoeActivationVariant::QwenAwqIndexed {
            awq_ptrs,
            topk_indices: p.topk_indices,
        }
    } else {
        MoeActivationVariant::MinimaxFused { awq_scale: None }
    };
    phases.routed.push(Step::MoeActivation {
        variant: activation,
        gate: p.gate_batch,
        up: p.up_batch,
        rot_out: p.rot_batch,
        inter: p.mi,
        k_top: p.k,
    });

    // down: Lloyd self-combines atomically (no combine follows); every other
    // indexable dtype writes the expanded buffer + shared combine. Per-expert
    // mixed tags force the merged expanded kernel (combine MUST run).
    let routed_down_self_combines = tags.is_none()
        && matches!(
            p.dtypes.routed_down,
            DType::MQ2G256Lloyd | DType::MQ3G256Lloyd
        );
    if routed_down_self_combines {
        phases.routed.push(Step::IndexedMoeGemv {
            experts: &dn_experts,
            which: MoeProj::DownResidual {
                topk_weights: p.topk_weights,
            },
            topk_indices: p.topk_indices,
            input: GemvInput::Prerotated(p.rot_batch),
            out: out_target,
            k_top: p.k,
            batch_size: 1,
        });
        phases.routed_partial = p.routed_out.is_some().then_some(phases.routed.len() - 1);
    } else {
        phases.routed.push(Step::MoeDownIndexed {
            experts: &dn_experts,
            topk_indices: p.topk_indices,
            rot_batch: p.rot_batch,
            out: p.down_expanded,
            k_top: p.k,
            batch_size: 1,
            mode: QwenDownMode::Expanded,
            dtype_tags: tags,
        });
        phases.routed.push(Step::MoeCombine {
            down_out: p.down_expanded,
            topk_weights: p.topk_weights,
            out: out_target,
            k: p.k,
            hidden: p.routed_down_m,
            batch_size: 1,
            inverse_perm: None,
        });
        phases.routed_partial = p.routed_out.is_some().then_some(phases.routed.len() - 1);
    }

    Ok(MoeStepBuild::Gpu(phases))
}

/// Build the prefill Step program for a Qwen MoE layer's routed block
/// (Path 0/1/2). The model owns RMSNorm, routing (top-k), and the shared
/// expert — this builder emits only the routed block, in the exact legacy
/// `run_moe_prefill` launch order.
pub fn build_moe_prefill_steps<'a>(
    p: &'a MoePrefillParams<'a>,
    res: &MoePrefillResolution,
    gu_experts: &'a MoeExpertRef<'a>,
    dn_experts: &'a MoeExpertRef<'a>,
) -> Result<MoeStepPhases<'a>, DispatchError> {
    let (n, inter, k_top, n_exp) = (p.batch_size, p.mi, p.k_top, p.n_exp);
    let (down_m, gate_up_k) = (p.down_m, p.gate_up_k);
    let total_slots = n * k_top;
    let force_mq4_grouped_fp16 = res.force_mq4_grouped_fp16 || p.force_mq4_grouped_fp16;
    let out_target: &'a GpuTensor = p.routed_out.unwrap_or(p.x_batch);
    let mut phases = MoeStepPhases {
        rotate: None,
        gate_side: Vec::new(),
        route: None,
        shared_down: Vec::new(),
        routed: Vec::new(),
        routed_partial: None,
    };

    // ── Gate_up ───────────────────────────────────────────────────────────
    if res.use_path2 {
        // Path 2: scatter → (Paro Givens preamble) → grouped gate_up →
        // gate-up unscatter. Mirrors the legacy scatter-before-preamble order.
        phases.routed.push(Step::MoeScatter {
            topk_indices: p.topk_indices,
            expert_token_counts: p.expert_token_counts,
            expert_offsets: p.expert_offsets,
            sorted_slot_index: p.sorted_slot_index,
            expert_tile_ids: p.expert_tile_ids,
            inverse_perm: p.inverse_perm,
            total_slots,
            n_experts: n_exp,
            m_total_max: p.m_total_max,
            block_m: MOE_GROUPED_BLOCK_M,
        });
        if res.paro_mode {
            let paro = p
                .paro_gate_up
                .as_ref()
                .ok_or(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-step-build-paro-without-gate-up-sidecar",
                    arch: "",
                    quant: "",
                })?;
            phases.routed.push(Step::GivensRotateBatched {
                x: p.x_norm_batch,
                out: p.x_rot_batch,
                pairs: paro.pairs,
                theta: paro.theta,
                scales: paro.scales,
                batch: n,
                dim: gate_up_k,
                krot: paro.krot,
            });
        }
        // Down-only-graded redline: a uniform MQ4 gate_up must NOT receive the
        // down dtype-tag table (the mixed grouped kernel would misread it).
        let gate_up_tags = if p.dtypes.experts_all_gate_up_mq4 {
            None
        } else {
            p.expert_dtype_tags
        };
        phases.routed.push(Step::GroupedMoeGemm {
            experts: &gu_experts,
            which: MoeProj::GateUp { up_out: p.up_batch },
            sorted_slot_index: p.sorted_slot_index,
            expert_tile_ids: p.expert_tile_ids,
            x: p.x_rot_batch,
            y: p.y_gate_up_grouped,
            m_total: p.m_total_max,
            batch_size: n,
            k_top,
            dtype_tags: gate_up_tags,
            force_mq4_fp16: force_mq4_grouped_fp16,
            paro_i8: res.use_paro_i8,
            paro_i8_k8: res.use_paro_i8_k8,
        });
        phases.routed.push(Step::MoeGateUpUnscatter {
            y_grouped: p.y_gate_up_grouped,
            sorted_slot_index: p.sorted_slot_index,
            gate_batch: p.gate_batch,
            up_batch: p.up_batch,
            inter,
            k_top,
            m_total: p.m_total_max,
        });
    } else {
        // Path 0/1: the indexed batched gate_up (representative dtype;
        // graded prefill runs the grouped path). Mirrors the legacy dispatch:
        // paro_mode serves Paro with its mandatory Givens preamble + the
        // batched paro indexed kernel; every other dtype dispatches by
        // representative dtype, and a Paro-routed layer WITHOUT the sidecar
        // falls to `_other` (same rejection as the legacy prefill dispatch).
        if res.paro_mode {
            let paro = p
                .paro_gate_up
                .as_ref()
                .ok_or(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-step-build-paro-without-gate-up-sidecar",
                    arch: "",
                    quant: "",
                })?;
            phases.routed.push(Step::GivensRotateBatched {
                x: p.x_norm_batch,
                out: p.x_rot_batch,
                pairs: paro.pairs,
                theta: paro.theta,
                scales: paro.scales,
                batch: n,
                dim: gate_up_k,
                krot: paro.krot,
            });
            phases.routed.push(Step::MoeGateUpIndexed {
                experts: &gu_experts,
                topk_indices: p.topk_indices,
                x_rot: p.x_rot_batch,
                gate_batch: p.gate_batch,
                up_batch: p.up_batch,
                k_top,
                batch_size: n,
                dtype_tags: None,
            });
        } else {
            match p.dtypes.routed_gate_up {
                DType::MQ4G256 | DType::MQ5G256 | DType::MQ6G256 | DType::MFP4G32E8 => {
                    phases.routed.push(Step::MoeGateUpIndexed {
                        experts: &gu_experts,
                        topk_indices: p.topk_indices,
                        x_rot: p.x_rot_batch,
                        gate_batch: p.gate_batch,
                        up_batch: p.up_batch,
                        k_top,
                        batch_size: n,
                        dtype_tags: None,
                    });
                }
                _other => {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "prefill-gate-up-path1-dtype",
                        arch: "",
                        quant: "other",
                    })
                }
            }
        }
    }

    // ── SwiGLU + rotate over [N*K_TOP × mi] ───────────────────────────────
    let activation = if res.paro_mode {
        let paro = p
            .paro_down
            .as_ref()
            .ok_or(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "prefill-step-build-paro-without-down-sidecar",
                arch: "",
                quant: "",
            })?;
        MoeActivationVariant::QwenParo {
            pairs: paro.pairs,
            theta: paro.theta,
            scales: paro.scales,
            krot: paro.krot,
        }
    } else if p.expert_dtype_tags.is_some() {
        // Graded/mixed routed experts: the silu+rotate is weight-agnostic
        // (mirrors the decode path).
        MoeActivationVariant::MinimaxFused { awq_scale: None }
    } else {
        match p.dtypes.routed_down {
            DType::MQ4G256
            | DType::MQ5G256
            | DType::MQ6G256
            | DType::MFP4G32E8
            | DType::MFP3G32E8
            | DType::MFP2G32E8 => {
                if let Some(awq_ptrs) = p.expert_down_awq_ptrs {
                    MoeActivationVariant::QwenAwqIndexed {
                        awq_ptrs,
                        topk_indices: p.topk_indices,
                    }
                } else if let Some(awq) = p.down_awq_scale {
                    MoeActivationVariant::MinimaxFused {
                        awq_scale: Some(awq),
                    }
                } else {
                    MoeActivationVariant::MinimaxFused { awq_scale: None }
                }
            }
            _other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-silu-rotate-dtype",
                    arch: "",
                    quant: "other",
                })
            }
        }
    };
    phases.routed.push(Step::MoeActivation {
        variant: activation,
        gate: p.gate_batch,
        up: p.up_batch,
        rot_out: p.rot_batch,
        inter,
        k_top: total_slots,
    });

    // ── Down projection ───────────────────────────────────────────────────
    if res.use_path2 {
        // Path 2: grouped down (structurally DownExpanded — the residual
        // projections are rejected) + grouped combine via inverse_perm.
        phases.routed.push(Step::GroupedMoeGemm {
            experts: &dn_experts,
            which: MoeProj::DownExpanded,
            sorted_slot_index: p.sorted_slot_index,
            expert_tile_ids: p.expert_tile_ids,
            x: p.rot_batch,
            y: p.y_down_grouped,
            m_total: p.m_total_max,
            batch_size: n,
            k_top,
            dtype_tags: p.expert_dtype_tags,
            force_mq4_fp16: force_mq4_grouped_fp16,
            paro_i8: res.use_paro_i8,
            paro_i8_k8: res.use_paro_i8_k8,
        });
        phases.routed.push(Step::MoeCombine {
            down_out: p.y_down_grouped,
            topk_weights: p.topk_weights,
            out: out_target,
            k: k_top,
            hidden: down_m,
            batch_size: n,
            inverse_perm: Some(p.inverse_perm),
        });
        phases.routed_partial = p.routed_out.is_some().then_some(phases.routed.len() - 1);
    } else if res.down_path0 {
        // Path 0 (gfx9* wave64): atomic residual-scaled GEMV, MQ4 only —
        // no expanded write, no combine.
        match p.dtypes.routed_down {
            DType::MQ4G256 => {
                phases.routed.push(Step::MoeDownIndexed {
                    experts: &dn_experts,
                    topk_indices: p.topk_indices,
                    rot_batch: p.rot_batch,
                    out: out_target,
                    k_top,
                    batch_size: n,
                    mode: QwenDownMode::ResidualScaled {
                        topk_weights: p.topk_weights,
                    },
                    dtype_tags: None,
                });
                phases.routed_partial = p.routed_out.is_some().then_some(phases.routed.len() - 1);
            }
            _other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-down-path0-dtype",
                    arch: "",
                    quant: "other",
                })
            }
        }
    } else {
        // Path 1: expanded write + combine.
        match p.dtypes.routed_down {
            DType::MQ4G256
            | DType::MQ5G256
            | DType::MQ6G256
            | DType::MFP4G32E8
            | DType::ParoQ4G128 => {
                phases.routed.push(Step::MoeDownIndexed {
                    experts: &dn_experts,
                    topk_indices: p.topk_indices,
                    rot_batch: p.rot_batch,
                    out: p.down_expanded,
                    k_top,
                    batch_size: n,
                    mode: QwenDownMode::Expanded,
                    dtype_tags: None,
                });
            }
            _other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-down-path1-dtype",
                    arch: "",
                    quant: "other",
                })
            }
        }
        phases.routed.push(Step::MoeCombine {
            down_out: p.down_expanded,
            topk_weights: p.topk_weights,
            out: out_target,
            k: k_top,
            hidden: down_m,
            batch_size: n,
            inverse_perm: None,
        });
        phases.routed_partial = p.routed_out.is_some().then_some(phases.routed.len() - 1);
    }

    Ok(phases)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DispatchCtx;
    use crate::pipeline::{execute_steps_mesh, MoeProj, QwenDownMode, Step};

    // ── STEP-002 Phase 1: Qwen MoE Step builders (no-GPU) ──────────────────

    /// Unwrap the Step-native phases from the discriminated build; panics on
    /// the explicit CPU-top-K fallback selection.
    fn gpu_phases(build: MoeStepBuild<'_>) -> MoeStepPhases<'_> {
        match build {
            MoeStepBuild::Gpu(phases) => phases,
            MoeStepBuild::CpuFallback => {
                panic!("expected a Step program, got the CPU-top-K fallback")
            }
        }
    }

    /// Short step label for Debug-free panic messages (Step has no Debug).
    fn step_label(step: &Step<'_>) -> &'static str {
        match step {
            Step::RotateFwhtBatched { .. } => "rotate",
            Step::GivensRotateBatched { .. } => "givens",
            Step::MoeFusedSharedGate { .. } => "gate",
            Step::MoeSharedGateSide { .. } => "gate",
            Step::MoeSoftmaxTopK { .. } => "route",
            Step::MoeSharedDown { .. } => "shared",
            Step::IndexedMoeGemv { .. } => "indexed",
            Step::MoeActivation { .. } => "act",
            Step::MoeDownIndexed { .. } => "down",
            Step::MoeGateUpIndexed { .. } => "gate_up",
            Step::MoeCombine { .. } => "combine",
            Step::MoeScatter { .. } => "scatter",
            Step::GroupedMoeGemm { .. } => "grouped",
            Step::MoeGateUpUnscatter { .. } => "gate_up_unscatter",
            other => {
                // Step has no Debug; classify by the op tag instead.
                match crate::pipeline::steps::step_op_kind(other) {
                    crate::types::PipelineOp::Gemv => "gemv",
                    _ => "other",
                }
            }
        }
    }

    /// Distinct fake tensors (stable addresses, never dereferenced — the
    /// builders are pure metadata shaping).
    struct Fakes {
        tensors: Vec<GpuTensor>,
    }

    impl Fakes {
        fn new() -> Self {
            Self {
                tensors: (0..96u32)
                    .map(|i| GpuTensor {
                        buf: unsafe {
                            hip_bridge::DeviceBuffer::from_raw(
                                (0x1000 + i as usize) as *mut std::ffi::c_void,
                                4096,
                            )
                        },
                        shape: vec![8],
                        dtype: DType::F32,
                    })
                    .collect(),
            }
        }

        fn t(&self, i: usize) -> &GpuTensor {
            &self.tensors[i]
        }

        fn w(&self, i: usize, dtype: DType, m: usize, k: usize) -> WeightRef<'_> {
            WeightRef {
                buf: self.t(i),
                dtype,
                m,
                k,
                row_stride: k,
                rotation: None,
                awq_scale: None,
            }
        }
    }

    fn decode_params<'a>(f: &'a Fakes, dtypes: MoeDtypes, k: usize) -> MoeParams<'a> {
        MoeParams {
            dtypes,
            batch_size: 1,
            hidden: 512,
            mi: 256,
            smi: 128,
            k,
            n_exp: 8,
            norm_topk_prob: true,
            x_rot_prerotated: false,
            layer_idx: 0,
            x_norm: f.t(0),
            x_residual: f.t(1),
            routed_out: None,
            skip_shared: false,
            router: f.w(2, DType::MQ4G256, 8, 512),
            shared_expert_gate: f.w(3, DType::MQ4G256, 1, 512),
            shared_gate_w: f.w(4, DType::MQ4G256, 128, 512),
            shared_up_w: f.w(5, DType::MQ4G256, 128, 512),
            shared_down_w: f.w(6, DType::MQ4G256, 512, 128),
            expert_gate_up_ptrs: f.t(7),
            expert_down_ptrs: f.t(8),
            expert_down_awq_ptrs: None,
            expert_dtype_tags: None,
            routed_gate_up_k: 512,
            routed_down_m: 512,
            routed_down_k: 256,
            routed_experts: &[],
            routed_gate_up_paro: None,
            routed_down_paro: None,
            router_logits: f.t(9),
            scalar_buf: f.t(10),
            x_rot_local: f.t(11),
            gate_up_buf: f.t(12),
            gate_buf: f.t(13),
            up_buf: f.t(14),
            ffn_hidden: f.t(15),
            ffn_out: f.t(16),
            gate_batch: f.t(17),
            up_batch: f.t(18),
            rot_batch: f.t(19),
            topk_indices: f.t(20),
            topk_weights: f.t(21),
            down_expanded: f.t(22),
        }
    }

    fn prefill_params<'a>(f: &'a Fakes, dtypes: MoeDtypes, n: usize) -> MoePrefillParams<'a> {
        MoePrefillParams {
            dtypes,
            batch_size: n,
            mi: 256,
            down_m: 512,
            down_k: 256,
            gate_up_k: 512,
            k_top: 8,
            n_exp: 8,
            m_total_max: n * 8 + 8 * 16,
            force_mq4_grouped_fp16: false,
            topk_indices: f.t(40),
            topk_weights: f.t(41),
            x_batch: f.t(42),
            x_norm_batch: f.t(43),
            x_rot_batch: f.t(44),
            expert_gate_up_ptrs: f.t(45),
            expert_down_ptrs: f.t(46),
            expert_down_awq_ptrs: None,
            expert_dtype_tags: None,
            gate_batch: f.t(47),
            up_batch: f.t(48),
            rot_batch: f.t(49),
            down_expanded: f.t(50),
            expert_token_counts: f.t(51),
            expert_offsets: f.t(52),
            sorted_slot_index: f.t(53),
            expert_tile_ids: f.t(54),
            inverse_perm: f.t(55),
            y_gate_up_grouped: f.t(56),
            y_down_grouped: f.t(57),
            paro_gate_up: None,
            paro_down: None,
            down_awq_scale: None,
            routed_out: None,
        }
    }

    #[test]
    fn qwen_mq4_decode_builder_orders_every_gpu_phase() {
        let f = Fakes::new();
        let d = uniform_mq4();
        let p = decode_params(&f, d.clone(), 8);
        let res = MoeResolution::resolve(&d, 8);
        assert!(
            res.use_gpu_topk,
            "uniform MQ4 k=8 must use the GPU top-K path"
        );
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());

        // Every GPU phase in exact legacy launch order.
        assert!(
            matches!(
                phases.rotate,
                Some(Step::RotateFwhtBatched { batch: 1, .. })
            ),
            "MQ4 decode must open with the scalar FWHT rotate"
        );
        assert_eq!(phases.gate_side.len(), 1, "uniform MQ4 gate side fuses");
        assert!(
            matches!(&phases.gate_side[0], Step::MoeFusedSharedGate { .. }),
            "gate_fusable must build the fused shared-gate step"
        );
        assert!(
            matches!(&phases.route, Some(Step::MoeSoftmaxTopK { n_exp: 8, .. })),
            "softmax + renormalized top-K must be an explicit routing step"
        );
        assert_eq!(phases.shared_down.len(), 1, "MQ4 shared down is one step");
        assert!(
            matches!(&phases.shared_down[0], Step::MoeSharedDown { .. }),
            "shared-expert down must be an explicit step"
        );
        assert_eq!(phases.routed.len(), 4, "gate_up, activation, down, combine");
        assert!(matches!(
            &phases.routed[0],
            Step::IndexedMoeGemv {
                which: MoeProj::GateUp { .. },
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[1],
            Step::MoeActivation {
                variant: MoeActivationVariant::MinimaxFused { awq_scale: None },
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[2],
            Step::MoeDownIndexed {
                mode: QwenDownMode::Expanded,
                batch_size: 1,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[3],
            Step::MoeCombine {
                inverse_perm: None,
                ..
            }
        ));

        // Flattening must preserve the phase order exactly.
        let build = phases.into_build();
        let kinds: Vec<&str> = build
            .steps
            .iter()
            .map(|s| match s {
                Step::RotateFwhtBatched { .. } => "rotate",
                Step::MoeFusedSharedGate { .. } => "gate",
                Step::MoeSoftmaxTopK { .. } => "route",
                Step::MoeSharedDown { .. } => "shared",
                Step::IndexedMoeGemv { .. } => "gate_up",
                Step::MoeActivation { .. } => "act",
                Step::MoeDownIndexed { .. } => "down",
                Step::MoeCombine { .. } => "combine",
                other => panic!(
                    "unexpected step in MQ4 decode program: {}",
                    step_label(other)
                ),
            })
            .collect();
        assert_eq!(
            kinds,
            ["rotate", "gate", "route", "shared", "gate_up", "act", "down", "combine"]
        );
        assert!(build.ep_partial.is_none(), "no routed_out → no EP partial");
    }

    #[test]
    fn qwen_skip_shared_omits_shared_down() {
        let f = Fakes::new();
        let d = uniform_mq4();
        let mut p = decode_params(&f, d.clone(), 8);
        p.skip_shared = true;
        let res = MoeResolution::resolve(&d, 8);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());
        assert!(
            phases.shared_down.is_empty(),
            "skip_shared must omit the shared-expert down steps"
        );
        // Everything else stays.
        assert!(phases.rotate.is_some());
        assert_eq!(phases.gate_side.len(), 1);
        assert!(phases.route.is_some());
        assert_eq!(phases.routed.len(), 4);
    }

    #[test]
    fn qwen_non_mq4_shared_down_emits_scaled_add_after_down() {
        // The non-MQ4 shared-down must decompose honestly: the down step
        // (sigmoid → silu·mul → GEMV) followed by the standalone scaled-add
        // step — exact legacy launch order, no fused multi-launch step, and
        // the standalone ScaledAdd step is actually constructed (not dead).
        let f = Fakes::new();
        let d = uniform_mq4();
        let mut p = decode_params(&f, d.clone(), 8);
        p.shared_down_w = f.w(6, DType::Q8_0, 512, 128);
        let res = MoeResolution::resolve(&d, 8);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());

        assert_eq!(
            phases.shared_down.len(),
            2,
            "non-MQ4 shared down = [MoeSharedDown, ScaledAdd]"
        );
        assert!(matches!(&phases.shared_down[0], Step::MoeSharedDown { .. }));
        match &phases.shared_down[1] {
            Step::ScaledAdd { x, y, scale } => {
                assert_eq!(x.buf.as_ptr(), p.x_residual.buf.as_ptr());
                assert_eq!(y.buf.as_ptr(), p.ffn_out.buf.as_ptr());
                assert_eq!(scale.buf.as_ptr(), p.scalar_buf.buf.as_ptr());
            }
            other => panic!(
                "second shared-down phase must be the standalone ScaledAdd, got {}",
                step_label(other)
            ),
        }

        // Flattened order: … route → shared-down → scaled-add → routed block.
        let program = phases.into_build();
        let kinds: Vec<&str> = program
            .steps
            .iter()
            .map(|s| match s {
                Step::RotateFwhtBatched { .. } => "rotate",
                Step::MoeFusedSharedGate { .. } => "gate",
                Step::MoeSoftmaxTopK { .. } => "route",
                Step::MoeSharedDown { .. } => "shared",
                Step::ScaledAdd { .. } => "scaled_add",
                Step::IndexedMoeGemv { .. } => "gate_up",
                Step::MoeActivation { .. } => "act",
                Step::MoeDownIndexed { .. } => "down",
                Step::MoeCombine { .. } => "combine",
                other => panic!("unexpected step: {}", step_label(other)),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rotate",
                "gate",
                "route",
                "shared",
                "scaled_add",
                "gate_up",
                "act",
                "down",
                "combine"
            ]
        );
    }

    #[test]
    fn qwen_lloyd_self_combining_down_omits_combine() {
        let f = Fakes::new();
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MQ2G256Lloyd;
        d.routed_down = DType::MQ2G256Lloyd;
        d.experts_all_gate_up_mq4 = false;
        let p = decode_params(&f, d.clone(), 8);
        let res = MoeResolution::resolve(&d, 8);
        assert!(res.routed_indexable_mq2lloyd && res.use_gpu_topk);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());

        // Down self-combines atomically → IndexedMoeGemv DownResidual and NO
        // MoeCombine (a combine would double-accumulate).
        assert_eq!(
            phases.routed.len(),
            3,
            "gate_up, activation, self-combining down"
        );
        assert!(matches!(
            &phases.routed[2],
            Step::IndexedMoeGemv {
                which: MoeProj::DownResidual { .. },
                ..
            }
        ));
        assert!(
            phases
                .routed
                .iter()
                .all(|s| !matches!(s, Step::MoeCombine { .. })),
            "Lloyd self-combining down must omit the expanded combine"
        );
    }

    #[test]
    fn qwen_cpu_fallback_is_explicit_and_not_a_step_program() {
        let f = Fakes::new();
        // k != 8 selects the CPU-top-K fallback variant, never a Step program.
        let d = uniform_mq4();
        let res_k4 = MoeResolution::resolve(&d, 4);
        assert!(!res_k4.use_gpu_topk, "k=4 must not be GPU-top-K");
        let p = decode_params(&f, d.clone(), 4);
        let (gu, dn) = decode_expert_refs(&p);
        let build = build_moe_decode_steps(&p, &res_k4, &gu, &dn).unwrap();
        assert!(
            matches!(&build, MoeStepBuild::CpuFallback),
            "!use_gpu_topk must select the explicit CPU-top-K fallback, not a Step program"
        );
        assert!(build.is_cpu_fallback());

        // A non-indexable routed dtype at k=8 also selects the fallback.
        let mut dq = uniform_mq4();
        dq.routed_gate_up = DType::Q8_0;
        dq.routed_down = DType::Q8_0;
        let res_q8 = MoeResolution::resolve(&dq, 8);
        assert!(!res_q8.use_gpu_topk, "Q8_0 routed must not be GPU-top-K");
        let pq = decode_params(&f, dq, 8);
        let (gu, dn) = decode_expert_refs(&pq);
        let build_q8 = build_moe_decode_steps(&pq, &res_q8, &gu, &dn).unwrap();
        assert!(
            matches!(&build_q8, MoeStepBuild::CpuFallback),
            "non-indexable routed dtypes must select the CPU-top-K fallback"
        );

        // And use_gpu_topk must select the Gpu program, never the fallback.
        let res_mq4 = MoeResolution::resolve(&d, 8);
        let pm = decode_params(&f, d, 8);
        let (gu, dn) = decode_expert_refs(&pm);
        let build_mq4 = build_moe_decode_steps(&pm, &res_mq4, &gu, &dn).unwrap();
        assert!(
            matches!(&build_mq4, MoeStepBuild::Gpu(_)),
            "use_gpu_topk must select the Step program"
        );
    }

    #[test]
    fn qwen_prefill_path2_orders_scatter_grouped_gate_up_unscatter_and_combine() {
        let f = Fakes::new();
        let d = uniform_mq4();
        let p = prefill_params(&f, d.clone(), 4);
        let ctx = DispatchCtx::for_test("gfx1100");
        let res = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        assert!(res.use_path2, "uniform MQ4 on gfx1100 must take Path 2");
        let (gu, dn) = prefill_expert_refs(&p);
        let phases = build_moe_prefill_steps(&p, &res, &gu, &dn).unwrap();

        // Prefill routing/shared-expert/rotation stay model-owned.
        assert!(phases.rotate.is_none());
        assert!(phases.gate_side.is_empty());
        assert!(phases.route.is_none());
        assert!(phases.shared_down.is_empty());

        // Path 2: scatter → grouped gate_up → gate-up unscatter →
        // activation → grouped down → grouped combine.
        assert_eq!(phases.routed.len(), 6);
        assert!(matches!(
            &phases.routed[0],
            Step::MoeScatter {
                total_slots: 32,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[1],
            Step::GroupedMoeGemm {
                which: MoeProj::GateUp { .. },
                batch_size: 4,
                ..
            }
        ));
        assert!(matches!(&phases.routed[2], Step::MoeGateUpUnscatter { .. }));
        assert!(matches!(
            &phases.routed[3],
            Step::MoeActivation {
                variant: MoeActivationVariant::MinimaxFused { awq_scale: None },
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[4],
            Step::GroupedMoeGemm {
                which: MoeProj::DownExpanded,
                batch_size: 4,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[5],
            Step::MoeCombine {
                inverse_perm: Some(_),
                batch_size: 4,
                ..
            }
        ));

        let build = phases.into_build();
        let kinds: Vec<&str> = build
            .steps
            .iter()
            .map(|s| match s {
                Step::MoeScatter { .. } => "scatter",
                Step::GroupedMoeGemm { .. } => "grouped",
                Step::MoeGateUpUnscatter { .. } => "gate_up_unscatter",
                Step::MoeActivation { .. } => "act",
                Step::MoeCombine { .. } => "combine",
                other => panic!(
                    "unexpected step in Path 2 prefill program: {}",
                    step_label(other)
                ),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "scatter",
                "grouped",
                "gate_up_unscatter",
                "act",
                "grouped",
                "combine",
            ]
        );
    }

    #[test]
    fn qwen_prefill_path1_orders_indexed_and_combine() {
        // Path 1 (indexed, no WMMA, no atomic Path 0): gate_up → activation →
        // expanded down → plain combine with inverse_perm None.
        let f = Fakes::new();
        let d = uniform_mq4();
        let p = prefill_params(&f, d.clone(), 4);
        let ctx = DispatchCtx::for_test("gfx1030");
        let res = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        assert!(
            !res.use_path2 && !res.down_path0,
            "gfx1030 (RDNA2) must take indexed Path 1"
        );
        let (gu, dn) = prefill_expert_refs(&p);
        let phases = build_moe_prefill_steps(&p, &res, &gu, &dn).unwrap();

        assert_eq!(phases.routed.len(), 4);
        assert!(matches!(
            &phases.routed[0],
            Step::MoeGateUpIndexed { batch_size: 4, .. }
        ));
        assert!(matches!(
            &phases.routed[1],
            Step::MoeActivation {
                variant: MoeActivationVariant::MinimaxFused { awq_scale: None },
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[2],
            Step::MoeDownIndexed {
                batch_size: 4,
                mode: QwenDownMode::Expanded,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[3],
            Step::MoeCombine {
                inverse_perm: None,
                batch_size: 4,
                ..
            }
        ));

        let program = phases.into_build();
        let kinds: Vec<&str> = program
            .steps
            .iter()
            .map(|s| match s {
                Step::MoeGateUpIndexed { .. } => "gate_up",
                Step::MoeActivation { .. } => "act",
                Step::MoeDownIndexed { .. } => "down",
                Step::MoeCombine { .. } => "combine",
                other => panic!("unexpected Path 1 step: {}", step_label(other)),
            })
            .collect();
        assert_eq!(kinds, ["gate_up", "act", "down", "combine"]);
    }

    #[test]
    fn qwen_prefill_path0_omits_expanded_combine() {
        let f = Fakes::new();
        let d = uniform_mq4();
        let p = prefill_params(&f, d.clone(), 4);
        let ctx = DispatchCtx::for_test("gfx942");
        let res = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        assert!(
            res.down_path0,
            "gfx942 (CDNA3) must use the atomic Path 0 down"
        );
        assert!(!res.use_path2, "gfx942 has no WMMA grouped path");
        let (gu, dn) = prefill_expert_refs(&p);
        let phases = build_moe_prefill_steps(&p, &res, &gu, &dn).unwrap();

        // Path 0: indexed batched gate_up → activation → atomic residual-scaled
        // down. NO expanded write and NO combine.
        assert_eq!(phases.routed.len(), 3);
        assert!(matches!(
            &phases.routed[0],
            Step::MoeGateUpIndexed { batch_size: 4, .. }
        ));
        assert!(matches!(
            &phases.routed[2],
            Step::MoeDownIndexed {
                mode: QwenDownMode::ResidualScaled { .. },
                batch_size: 4,
                ..
            }
        ));
        assert!(
            phases
                .routed
                .iter()
                .all(|s| !matches!(s, Step::MoeCombine { .. })),
            "Path 0 accumulates atomically and must omit the expanded combine"
        );
    }

    // ── STEP-002 remediation: indexed prefill Path-1 batch>1 selection ────

    #[test]
    fn qwen_prefill_path1_e8_batched() {
        // E8 on a non-WMMA arch must take the indexed Path 1, and a
        // batch>1 E8 gate_up must select the existing _batched kernel —
        // never the decode scalar kernel (blocking regression 1).
        let f = Fakes::new();
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MFP4G32E8;
        d.routed_down = DType::MFP4G32E8;
        d.experts_all_gate_up_mq4 = false;
        let p = prefill_params(&f, d.clone(), 4);
        let ctx = DispatchCtx::for_test("gfx1030");
        let res = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        assert!(
            !res.use_path2 && !res.down_path0,
            "gfx1030 (RDNA2) must take indexed Path 1 for E8"
        );
        let (gu, dn) = prefill_expert_refs(&p);
        let phases = build_moe_prefill_steps(&p, &res, &gu, &dn).unwrap();
        assert_eq!(phases.routed.len(), 4, "gate_up, activation, down, combine");
        assert!(matches!(
            &phases.routed[0],
            Step::MoeGateUpIndexed {
                batch_size: 4,
                dtype_tags: None,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[3],
            Step::MoeCombine {
                inverse_perm: None,
                ..
            }
        ));

        // Selection contract: the launcher must branch on batch_size and use
        // the existing batched E8 kernel for batch>1 (the scalar kernel is
        // decode-only).
        assert_eq!(
            qwen_gate_up_indexed_form(DType::MFP4G32E8, 1),
            QwenIndexedForm::Scalar
        );
        assert_eq!(
            qwen_gate_up_indexed_form(DType::MFP4G32E8, 4),
            QwenIndexedForm::Batched
        );
    }

    #[test]
    fn qwen_prefill_paro_path1() {
        // Paro on a non-WMMA arch must take the indexed Path 1 WITH the
        // mandatory Givens preamble and the batched paro gate_up kernel
        // (blocking regression 2 — the builder previously rejected Paro).
        let f = Fakes::new();
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::ParoQ4G128;
        d.routed_down = DType::ParoQ4G128;
        d.has_paro_shared = true;
        let mut p = prefill_params(&f, d.clone(), 4);
        p.paro_gate_up = Some(GivensRef {
            pairs: f.t(62),
            theta: f.t(63),
            scales: f.t(64),
            krot: 128,
        });
        p.paro_down = Some(GivensRef {
            pairs: f.t(65),
            theta: f.t(66),
            scales: f.t(67),
            krot: 128,
        });
        let ctx = DispatchCtx::for_test("gfx1030");
        let res = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        assert!(res.paro_mode && !res.use_path2 && !res.down_path0);
        let (gu, dn) = prefill_expert_refs(&p);
        let phases = build_moe_prefill_steps(&p, &res, &gu, &dn).unwrap();

        // givens preamble → batched indexed gate_up → Paro activation →
        // expanded down → plain combine (inverse_perm None).
        assert_eq!(phases.routed.len(), 5);
        assert!(matches!(
            &phases.routed[0],
            Step::GivensRotateBatched { batch: 4, .. }
        ));
        match &phases.routed[1] {
            Step::MoeGateUpIndexed {
                batch_size,
                dtype_tags: None,
                ..
            } => assert_eq!(*batch_size, 4),
            other => panic!(
                "paro Path 1 gate_up must be the batched indexed step, got {}",
                step_label(other)
            ),
        }
        assert!(matches!(
            &phases.routed[2],
            Step::MoeActivation {
                variant: MoeActivationVariant::QwenParo { .. },
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[3],
            Step::MoeDownIndexed {
                batch_size: 4,
                mode: QwenDownMode::Expanded,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[4],
            Step::MoeCombine {
                inverse_perm: None,
                ..
            }
        ));
        // Selection contract: batch>1 paro gate_up uses the batched kernel.
        assert_eq!(
            qwen_gate_up_indexed_form(DType::ParoQ4G128, 4),
            QwenIndexedForm::Batched
        );
    }

    #[test]
    fn qwen_grouped_down_residual_is_rejected() {
        let f = Fakes::new();
        let d = uniform_mq4();
        let p = prefill_params(&f, d.clone(), 4);
        let ctx = DispatchCtx::for_test("gfx1100");
        let res = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        let (gu, dn) = prefill_expert_refs(&p);
        let phases = build_moe_prefill_steps(&p, &res, &gu, &dn).unwrap();

        // Structurally: the Path 2 down step is always the expanded projection
        // (the grouped kernel family has no residual-fused down).
        match &phases.routed[4] {
            Step::GroupedMoeGemm {
                which: MoeProj::DownExpanded,
                batch_size,
                ..
            } => assert_eq!(
                *batch_size, 4,
                "grouped down stores token batch; launch_grouped_down applies k_top exactly once"
            ),
            other => panic!(
                "Path 2 down must be DownExpanded, got {}",
                step_label(other)
            ),
        }
        // The shared rejection predicate refuses the residual projections.
        assert!(grouped_down_projection(&MoeProj::DownExpanded).is_ok());
        assert!(
            grouped_down_projection(&MoeProj::DownResidual {
                topk_weights: f.t(60)
            })
            .is_err(),
            "grouped DownResidual must be rejected"
        );
        assert!(
            grouped_down_projection(&MoeProj::DownResidualI64 {
                topk_weights: f.t(61)
            })
            .is_err(),
            "grouped DownResidualI64 must be rejected"
        );
    }

    #[test]
    fn qwen_decode_mixed_tags_route_tagged_steps() {
        let f = Fakes::new();
        let mut d = uniform_mq4();
        d.routed_has_mixed_experts = true;
        d.experts_all_gate_up_mq4 = false;
        d.per_expert_gate_up = Some(vec![DType::MQ4G256, DType::MQ6G256]);
        d.per_expert_down = Some(vec![DType::MQ4G256, DType::MQ6G256]);
        let mut p = decode_params(&f, d.clone(), 8);
        p.expert_dtype_tags = Some(f.t(30));
        let res = MoeResolution::resolve(&d, 8);
        assert!(res.routed_indexable_mixed_per_expert && res.use_gpu_topk);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());

        // Graded gate_up + down use the merged dtype-tag kernels; the shared
        // expanded combine still runs (tags force expanded writes).
        assert!(matches!(
            &phases.routed[0],
            Step::MoeGateUpIndexed {
                dtype_tags: Some(_),
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[2],
            Step::MoeDownIndexed {
                dtype_tags: Some(_),
                mode: QwenDownMode::Expanded,
                ..
            }
        ));
        assert!(matches!(&phases.routed[3], Step::MoeCombine { .. }));
    }

    #[test]
    fn qwen_decode_awq_activation_uses_indexed_variant() {
        let f = Fakes::new();
        let d = uniform_mq4();
        let mut p = decode_params(&f, d.clone(), 8);
        p.expert_down_awq_ptrs = Some(f.t(31));
        let res = MoeResolution::resolve(&d, 8);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());
        match &phases.routed[1] {
            Step::MoeActivation {
                variant: MoeActivationVariant::QwenAwqIndexed { awq_ptrs, .. },
                ..
            } => assert!(awq_ptrs.buf.as_ptr() == f.t(31).buf.as_ptr()),
            other => panic!(
                "AWQ decode must use the indexed AWQ activation, got {}",
                step_label(other)
            ),
        }
    }

    #[test]
    fn qwen_decode_paro_orders_givens_rotate_and_paro_steps() {
        let f = Fakes::new();
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::ParoQ4G128;
        d.routed_down = DType::ParoQ4G128;
        d.has_paro_shared = true;
        let mut p = decode_params(&f, d.clone(), 8);
        p.routed_gate_up_paro = Some(GivensRef {
            pairs: f.t(32),
            theta: f.t(33),
            scales: f.t(34),
            krot: 128,
        });
        p.routed_down_paro = Some(GivensRef {
            pairs: f.t(35),
            theta: f.t(36),
            scales: f.t(37),
            krot: 128,
        });
        let res = MoeResolution::resolve(&d, 8);
        assert!(res.routed_indexable_paro && res.use_gpu_topk);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());

        assert!(matches!(
            phases.rotate,
            Some(Step::GivensRotateBatched { batch: 1, .. })
        ));
        assert!(matches!(
            &phases.routed[0],
            Step::MoeGateUpIndexed {
                dtype_tags: None,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[1],
            Step::MoeActivation {
                variant: MoeActivationVariant::QwenParo { .. },
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[2],
            Step::MoeDownIndexed {
                dtype_tags: None,
                ..
            }
        ));
        assert!(matches!(&phases.routed[3], Step::MoeCombine { .. }));
    }

    #[test]
    fn qwen_decode_e8_routes_indexed_steps() {
        let f = Fakes::new();
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MFP4G32E8;
        d.routed_down = DType::MFP4G32E8;
        d.experts_all_gate_up_mq4 = false;
        let p = decode_params(&f, d.clone(), 8);
        let res = MoeResolution::resolve_arch(&d, 8, true);
        assert!(res.routed_indexable_e8 && res.use_gpu_topk);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());
        assert!(matches!(
            &phases.routed[0],
            Step::MoeGateUpIndexed {
                dtype_tags: None,
                ..
            }
        ));
        assert!(matches!(
            &phases.routed[2],
            Step::MoeDownIndexed {
                dtype_tags: None,
                mode: QwenDownMode::Expanded,
                ..
            }
        ));
        assert!(matches!(&phases.routed[3], Step::MoeCombine { .. }));
    }

    #[test]
    fn qwen_decode_mq5_mq6_route_indexed_steps() {
        let f = Fakes::new();
        for (gu, dn) in [
            (DType::MQ5G256, DType::MQ5G256),
            (DType::MQ6G256, DType::MQ6G256),
            (DType::MQ4G256, DType::MQ6G256), // mq6-down lever
        ] {
            let mut d = uniform_mq4();
            d.routed_gate_up = gu;
            d.routed_down = dn;
            d.experts_all_gate_up_mq4 = false;
            let p = decode_params(&f, d.clone(), 8);
            let res = MoeResolution::resolve(&d, 8);
            assert!(res.use_gpu_topk, "routed ({gu:?},{dn:?}) must be GPU-top-K");
            let (gu, dn) = decode_expert_refs(&p);
            let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());
            assert_eq!(phases.routed.len(), 4, "gate_up, activation, down, combine");
            assert!(matches!(&phases.routed[3], Step::MoeCombine { .. }));
        }
    }

    #[test]
    fn qwen_decode_routed_out_marks_ep_partial() {
        let f = Fakes::new();
        let d = uniform_mq4();
        let mut p = decode_params(&f, d.clone(), 8);
        p.routed_out = Some(f.t(40));
        let res = MoeResolution::resolve(&d, 8);
        let (gu, dn) = decode_expert_refs(&p);
        let phases = gpu_phases(build_moe_decode_steps(&p, &res, &gu, &dn).unwrap());
        let build = phases.into_build();

        // The EP partial writer is the routed combine.
        let idx = build
            .ep_partial
            .expect("routed_out must mark the EP partial step");
        assert!(matches!(
            &build.steps[idx],
            Step::MoeCombine { out, .. } if out.buf.as_ptr() == f.t(40).buf.as_ptr()
        ));
        // The shared-expert down accumulates into the same partial.
        let shared_idx = 3; // rotate, gate, route, shared
        assert!(matches!(
            &build.steps[shared_idx],
            Step::MoeSharedDown { out_target, .. } if out_target.buf.as_ptr() == f.t(40).buf.as_ptr()
        ));
    }

    fn uniform_mq4() -> MoeDtypes {
        MoeDtypes {
            router: DType::MQ4G256,
            shared_gate: DType::MQ4G256,
            shared_expert_gate: DType::MQ4G256,
            shared_expert_up: DType::MQ4G256,
            shared_expert_down: DType::MQ4G256,
            experts_all_gate_up_mq4: true,
            routed_gate_up: DType::MQ4G256,
            routed_down: DType::MQ4G256,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            gate_side_has_awq: false,
            routed_down_has_awq: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        }
    }

    #[test]
    fn resolve_none_per_expert_is_not_mixed() {
        let d = uniform_mq4();
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.mixed);
    }

    #[test]
    fn resolve_some_per_expert_with_varied_tiers_is_mixed() {
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(vec![DType::MQ4G256, DType::MQ6G256]); // varies
        d.per_expert_down = Some(vec![DType::MQ4G256, DType::MQ6G256]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.mixed);
    }

    #[test]
    fn resolve_empty_per_expert_table_is_not_mixed_and_does_not_panic() {
        // A degenerate empty table must not index v[0]; it collapses to uniform.
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(vec![]);
        d.per_expert_down = Some(vec![]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.mixed);
    }

    #[test]
    fn resolve_some_per_expert_all_same_is_not_mixed() {
        // a per-expert table that is uniform should NOT trigger the mixed path
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(vec![DType::MQ4G256, DType::MQ4G256]);
        d.per_expert_down = Some(vec![DType::MQ4G256, DType::MQ4G256]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(
            !r.mixed,
            "a uniform per-expert table must take the fast uniform path"
        );
    }

    #[test]
    fn router_plan_preserves_normalization_and_scale() {
        let tensor = GpuTensor::null_for_test();
        let plan = RouterPlan::BiasAwareTopK {
            scores: &tensor,
            gate_bias: &tensor,
            topk_indices: &tensor,
            topk_weights: &tensor,
            k_top: 6,
            normalize: true,
            route_scale: 0.25,
        };

        assert_eq!(plan.k_top(), 6);
        assert!(plan.normalizes());
        assert_eq!(plan.route_scale(), 0.25);
    }

    #[test]
    fn router_plan_variants_keep_compatible_operands() {
        let tensor = GpuTensor::null_for_test();
        let plans = [
            RouterPlan::SoftmaxTopK {
                scores: &tensor,
                topk_indices: &tensor,
                topk_weights: &tensor,
                k_top: 8,
                normalize: true,
                route_scale: 1.0,
            },
            RouterPlan::SigmoidTopK {
                scores: &tensor,
                topk_indices: &tensor,
                topk_weights: &tensor,
                k_top: 8,
                normalize: false,
                route_scale: 1.0,
            },
            RouterPlan::BiasAwareTopK {
                scores: &tensor,
                gate_bias: &tensor,
                topk_indices: &tensor,
                topk_weights: &tensor,
                k_top: 6,
                normalize: true,
                route_scale: 0.5,
            },
            RouterPlan::Hash {
                scores: &tensor,
                tokens: &tensor,
                tid2eid: &tensor,
                topk_indices: &tensor,
                topk_weights: &tensor,
                k_top: 6,
                normalize: true,
                route_scale: 1.0,
            },
            RouterPlan::Precomputed {
                topk_indices: &tensor,
                topk_weights: &tensor,
                k_top: 4,
                normalize: false,
                route_scale: 1.0,
            },
        ];

        assert_eq!(plans[0].selection(), RouterSelection::SoftmaxTopK);
        assert_eq!(plans[1].selection(), RouterSelection::SigmoidTopK);
        assert_eq!(plans[2].selection(), RouterSelection::BiasAwareTopK);
        assert_eq!(plans[3].selection(), RouterSelection::Hash);
        assert_eq!(plans[4].selection(), RouterSelection::Precomputed);
    }

    #[test]
    fn router_plan_hash_carries_scores_and_preserves_selection() {
        let scores = GpuTensor::null_for_test();
        let tokens = GpuTensor::null_for_test();
        let tid2eid = GpuTensor::null_for_test();
        let topk_indices = GpuTensor::null_for_test();
        let topk_weights = GpuTensor::null_for_test();
        let plan = RouterPlan::Hash {
            scores: &scores,
            tokens: &tokens,
            tid2eid: &tid2eid,
            topk_indices: &topk_indices,
            topk_weights: &topk_weights,
            k_top: 6,
            normalize: true,
            route_scale: 1.0,
        };

        match &plan {
            RouterPlan::Hash { scores: actual, .. } => {
                assert!(std::ptr::eq(*actual, &scores));
            }
            _ => unreachable!("constructed a hash routing plan"),
        }
        assert_eq!(plan.selection(), RouterSelection::Hash);
    }

    // ── STEP-002 Phase 1: GPU old-vs-Step parity gates (ignored) ────────────
    //
    // Each gate runs the SAME layer twice: once through a TEST-ONLY legacy
    // reference (the exact kernel sequence of the deleted `run_moe_decode` /
    // `run_moe_prefill` GPU bodies, composed from the extracted helpers and
    // direct kernel calls), once through the flattened Step program
    // (`build_*_steps` + `execute_steps_mesh`), and compares the residual /
    // partial BYTE-FOR-BYTE. Garbage quantized payloads are fine: both
    // executions read the identical bytes, so any wiring divergence (step
    // order, argument shuffle, wrong kernel form) shows up as a bit
    // mismatch. These references are TEST-ONLY — the Step program is the
    // only production Qwen GPU path.
    //
    // Run serially under the repository GPU lock:
    //   source scripts/gpu-lock.sh && gpu_acquire step002-phase1
    //   cargo test -p hipfire-dispatch --lib -- --ignored qwen_step_parity --nocapture

    const PH_HIDDEN: usize = 256;
    // mi/smi must be ≥ 256: the FWHT-rotated kernels tile per-256 group and a
    // smaller inter would produce grid.x = 0 (invalid launch).
    const PH_MI: usize = 256;
    const PH_SMI: usize = 256;
    const PH_K: usize = 8;
    const PH_N_EXP: usize = 8;

    fn ph_lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32) / (1u64 << 24) as f32
    }

    fn ph_upload(gpu: &mut rdna_compute::Gpu, shape: &[usize], seed: &mut u64) -> GpuTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| ph_lcg(seed)).collect();
        gpu.upload_f32(&data, shape).unwrap()
    }

    fn ph_upload_raw(gpu: &mut rdna_compute::Gpu, bytes: &[u8]) -> GpuTensor {
        let tensor = gpu.alloc_tensor(&[bytes.len()], DType::Raw).unwrap();
        gpu.hip.memcpy_htod(&tensor.buf, bytes).unwrap();
        tensor
    }

    fn ph_upload_f16_ones(gpu: &mut rdna_compute::Gpu, shape: &[usize]) -> GpuTensor {
        let tensor = gpu.alloc_tensor(shape, DType::F16).unwrap();
        let mut bytes = Vec::with_capacity(shape.iter().product::<usize>() * 2);
        for _ in 0..shape.iter().product::<usize>() {
            bytes.extend_from_slice(&0x3c00u16.to_ne_bytes());
        }
        gpu.hip.memcpy_htod(&tensor.buf, &bytes).unwrap();
        tensor
    }

    fn ph_fill_f32(gpu: &mut rdna_compute::Gpu, tensor: &GpuTensor, value: f32) {
        let values = vec![value; tensor.buf.size() / 4];
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        gpu.hip.memcpy_htod(&tensor.buf, bytes).unwrap();
    }

    fn ph_valid_paro_sidecars(gpu: &mut rdna_compute::Gpu) -> (GpuTensor, GpuTensor, GpuTensor) {
        const KROT: usize = 64;
        const GROUP_SIZE: usize = 128;

        let mut pair_bytes = Vec::with_capacity(KROT * PH_HIDDEN * 2);
        for _ in 0..KROT {
            for channel in 0..PH_HIDDEN {
                let local_channel = (channel % GROUP_SIZE) as i16;
                pair_bytes.extend_from_slice(&local_channel.to_ne_bytes());
            }
        }

        let theta_bytes = vec![0u8; KROT * (PH_HIDDEN / 2) * 2];
        let mut scale_bytes = Vec::with_capacity(PH_HIDDEN * 2);
        for _ in 0..PH_HIDDEN {
            scale_bytes.extend_from_slice(&0x3c00u16.to_ne_bytes());
        }

        (
            ph_upload_raw(gpu, &pair_bytes),
            ph_upload_raw(gpu, &theta_bytes),
            ph_upload_raw(gpu, &scale_bytes),
        )
    }

    fn ph_zero(gpu: &mut rdna_compute::Gpu, shape: &[usize]) -> GpuTensor {
        gpu.alloc_tensor(shape, DType::F32).unwrap()
    }

    /// [n_exp] u64 device-pointer table stored in a 2·n_exp F32 tensor
    /// (mirrors the loader's table layout).
    fn ph_ptr_table(gpu: &mut rdna_compute::Gpu, bufs: &[GpuTensor]) -> GpuTensor {
        let table = gpu.alloc_tensor(&[2 * bufs.len()], DType::F32).unwrap();
        let bytes: Vec<u8> = bufs
            .iter()
            .flat_map(|t| (t.buf.as_ptr() as u64).to_ne_bytes())
            .collect();
        gpu.hip.memcpy_htod(&table.buf, &bytes).unwrap();
        table
    }

    fn ph_repeated_ptr_table(
        gpu: &mut rdna_compute::Gpu,
        tensor: &GpuTensor,
        count: usize,
    ) -> GpuTensor {
        let table = gpu.alloc_tensor(&[2 * count], DType::F32).unwrap();
        let bytes: Vec<u8> = (0..count)
            .flat_map(|_| (tensor.buf.as_ptr() as u64).to_ne_bytes())
            .collect();
        gpu.hip.memcpy_htod(&table.buf, &bytes).unwrap();
        table
    }

    /// i32 top-k indices in the F32-typed 4-byte-slot convention.
    fn ph_topk_indices(gpu: &mut rdna_compute::Gpu) -> GpuTensor {
        let bits: Vec<f32> = (0..PH_K as u32).map(f32::from_bits).collect();
        gpu.upload_f32(&bits, &[PH_K]).unwrap()
    }

    /// Real MQ4 gate-side + shared weights (all gate-side projections must be
    /// MQ4 for `gate_fusable`; the routed dtype varies per test).
    fn ph_mq4_dtypes(routed_gate_up: DType, routed_down: DType) -> MoeDtypes {
        MoeDtypes {
            router: DType::MQ4G256,
            shared_gate: DType::MQ4G256,
            shared_expert_gate: DType::MQ4G256,
            shared_expert_up: DType::MQ4G256,
            shared_expert_down: DType::MQ4G256,
            experts_all_gate_up_mq4: routed_gate_up == DType::MQ4G256,
            routed_gate_up,
            routed_down,
            routed_has_mixed_experts: false,
            has_paro_shared: matches!(routed_gate_up, DType::ParoQ4G128),
            gate_side_has_awq: false,
            routed_down_has_awq: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        }
    }

    /// One real (garbage-weight) MQ4/MQ6/E8/Paro/mixed MoE decode layer.
    /// `params()` borrows the rig's tensors — two rigs with the same seed
    /// hold byte-identical inputs for the two executions.
    struct DecodeRig {
        dtypes: MoeDtypes,
        // gate-side weight buffers (MQ4 quantized layout, garbage bytes)
        router_buf: GpuTensor,
        shared_expert_gate_buf: GpuTensor,
        shared_gate_w_buf: GpuTensor,
        shared_up_w_buf: GpuTensor,
        shared_down_w_buf: GpuTensor,
        // routed experts
        _expert_gu_bufs: Vec<GpuTensor>,
        _expert_dn_bufs: Vec<GpuTensor>,
        expert_gate_up_ptrs: GpuTensor,
        expert_down_ptrs: GpuTensor,
        expert_down_awq_ptrs: Option<GpuTensor>,
        _awq_scale_bufs: Vec<GpuTensor>,
        expert_dtype_tags: Option<GpuTensor>,
        // activations / scratch
        x_norm: GpuTensor,
        x_residual: GpuTensor,
        x_rot_local: GpuTensor,
        router_logits: GpuTensor,
        scalar_buf: GpuTensor,
        gate_buf: GpuTensor,
        up_buf: GpuTensor,
        ffn_hidden: GpuTensor,
        ffn_out: GpuTensor,
        gate_batch: GpuTensor,
        up_batch: GpuTensor,
        rot_batch: GpuTensor,
        topk_indices: GpuTensor,
        topk_weights: GpuTensor,
        down_expanded: GpuTensor,
        // paro sidecars
        paro_gu: Option<(GpuTensor, GpuTensor, GpuTensor)>,
        paro_dn: Option<(GpuTensor, GpuTensor, GpuTensor)>,
    }

    impl DecodeRig {
        fn new(
            gpu: &mut rdna_compute::Gpu,
            dtypes: MoeDtypes,
            seed: &mut u64,
            tags: Option<&[u8]>,
            with_awq: bool,
            with_paro: bool,
        ) -> Self {
            let mut expert_gu_bufs = Vec::new();
            let mut expert_dn_bufs = Vec::new();
            for _ in 0..PH_N_EXP {
                expert_gu_bufs.push(ph_upload(gpu, &[2 * PH_MI * PH_HIDDEN], seed));
                expert_dn_bufs.push(ph_upload(gpu, &[PH_HIDDEN * PH_MI], seed));
            }
            let expert_gate_up_ptrs = ph_ptr_table(gpu, &expert_gu_bufs);
            let expert_down_ptrs = ph_ptr_table(gpu, &expert_dn_bufs);
            let mut awq_scale_bufs = Vec::new();
            let expert_down_awq_ptrs = if with_awq {
                for _ in 0..PH_N_EXP {
                    awq_scale_bufs.push(ph_upload(gpu, &[PH_MI], seed));
                }
                Some(ph_ptr_table(gpu, &awq_scale_bufs))
            } else {
                None
            };
            let expert_dtype_tags = tags.map(|t| {
                let slot = gpu.alloc_tensor(&[PH_N_EXP], DType::Raw).unwrap();
                gpu.hip.memcpy_htod(&slot.buf, t).unwrap();
                slot
            });
            let paro_gu = with_paro.then(|| ph_valid_paro_sidecars(gpu));
            let paro_dn = with_paro.then(|| ph_valid_paro_sidecars(gpu));
            Self {
                dtypes,
                router_buf: ph_upload(gpu, &[PH_N_EXP * PH_HIDDEN], seed),
                shared_expert_gate_buf: ph_upload(gpu, &[1 * PH_HIDDEN], seed),
                shared_gate_w_buf: ph_upload(gpu, &[PH_SMI * PH_HIDDEN], seed),
                shared_up_w_buf: ph_upload(gpu, &[PH_SMI * PH_HIDDEN], seed),
                shared_down_w_buf: ph_upload(gpu, &[PH_HIDDEN * PH_SMI], seed),
                _expert_gu_bufs: expert_gu_bufs,
                _expert_dn_bufs: expert_dn_bufs,
                expert_gate_up_ptrs,
                expert_down_ptrs,
                expert_down_awq_ptrs,
                _awq_scale_bufs: awq_scale_bufs,
                expert_dtype_tags,
                x_norm: ph_upload(gpu, &[PH_HIDDEN], seed),
                x_residual: ph_zero(gpu, &[PH_HIDDEN]),
                x_rot_local: ph_zero(gpu, &[PH_HIDDEN]),
                router_logits: ph_zero(gpu, &[PH_N_EXP]),
                scalar_buf: ph_zero(gpu, &[1]),
                gate_buf: ph_zero(gpu, &[2 * PH_MI]),
                up_buf: ph_zero(gpu, &[2 * PH_MI]),
                ffn_hidden: ph_zero(gpu, &[PH_MI]),
                ffn_out: ph_zero(gpu, &[PH_HIDDEN]),
                gate_batch: ph_zero(gpu, &[PH_K * PH_MI]),
                up_batch: ph_zero(gpu, &[PH_K * PH_MI]),
                rot_batch: ph_zero(gpu, &[PH_K * PH_MI]),
                topk_indices: ph_topk_indices(gpu),
                topk_weights: ph_upload(gpu, &[PH_K], seed),
                down_expanded: ph_zero(gpu, &[PH_K * PH_HIDDEN]),
                paro_gu,
                paro_dn,
            }
        }

        fn weight<'r>(
            &'r self,
            buf: &'r GpuTensor,
            dtype: DType,
            m: usize,
            k: usize,
        ) -> WeightRef<'r> {
            WeightRef {
                buf,
                dtype,
                m,
                k,
                row_stride: k,
                rotation: None,
                awq_scale: None,
            }
        }

        fn params(&self) -> MoeParams<'_> {
            MoeParams {
                dtypes: self.dtypes.clone(),
                batch_size: 1,
                hidden: PH_HIDDEN,
                mi: PH_MI,
                smi: PH_SMI,
                k: PH_K,
                n_exp: PH_N_EXP,
                norm_topk_prob: true,
                x_rot_prerotated: false,
                layer_idx: 0,
                x_norm: &self.x_norm,
                x_residual: &self.x_residual,
                routed_out: None,
                skip_shared: false,
                router: self.weight(&self.router_buf, DType::MQ4G256, PH_N_EXP, PH_HIDDEN),
                shared_expert_gate: self.weight(
                    &self.shared_expert_gate_buf,
                    DType::MQ4G256,
                    1,
                    PH_HIDDEN,
                ),
                shared_gate_w: self.weight(
                    &self.shared_gate_w_buf,
                    DType::MQ4G256,
                    PH_SMI,
                    PH_HIDDEN,
                ),
                shared_up_w: self.weight(&self.shared_up_w_buf, DType::MQ4G256, PH_SMI, PH_HIDDEN),
                shared_down_w: self.weight(
                    &self.shared_down_w_buf,
                    self.dtypes.shared_expert_down,
                    PH_HIDDEN,
                    PH_SMI,
                ),
                expert_gate_up_ptrs: &self.expert_gate_up_ptrs,
                expert_down_ptrs: &self.expert_down_ptrs,
                expert_down_awq_ptrs: self.expert_down_awq_ptrs.as_ref(),
                expert_dtype_tags: self.expert_dtype_tags.as_ref(),
                routed_gate_up_k: PH_HIDDEN,
                routed_down_m: PH_HIDDEN,
                routed_down_k: PH_MI,
                routed_experts: &[],
                routed_gate_up_paro: self.paro_gu.as_ref().map(|(pairs, theta, scales)| {
                    GivensRef {
                        pairs,
                        theta,
                        scales,
                        krot: 64,
                    }
                }),
                routed_down_paro: self
                    .paro_dn
                    .as_ref()
                    .map(|(pairs, theta, scales)| GivensRef {
                        pairs,
                        theta,
                        scales,
                        krot: 64,
                    }),
                router_logits: &self.router_logits,
                scalar_buf: &self.scalar_buf,
                x_rot_local: &self.x_rot_local,
                gate_up_buf: &self.gate_buf,
                gate_buf: &self.gate_buf,
                up_buf: &self.up_buf,
                ffn_hidden: &self.ffn_hidden,
                ffn_out: &self.ffn_out,
                gate_batch: &self.gate_batch,
                up_batch: &self.up_batch,
                rot_batch: &self.rot_batch,
                topk_indices: &self.topk_indices,
                topk_weights: &self.topk_weights,
                down_expanded: &self.down_expanded,
            }
        }
    }

    /// Byte-for-byte comparison (bitwise — NaN bits must match too).
    fn ph_assert_bits_eq(name: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{name}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{name}: byte-for-byte parity mismatch at elem {i} ({x} vs {y})"
            );
        }
    }

    /// TEST-ONLY legacy decode reference, part 1: rotate → fused gate side →
    /// softmax+top-K → shared down. The exact kernel sequence of the deleted
    /// `run_moe_decode` GPU path (uniform-MQ4 gate side).
    fn legacy_decode_front(
        gpu: &mut rdna_compute::Gpu,
        ctx: &DispatchCtx,
        p: &MoeParams<'_>,
        out_target: &GpuTensor,
        res: &MoeResolution,
    ) {
        if res.routed_indexable_paro {
            let paro = p.routed_gate_up_paro.as_ref().unwrap();
            gpu.givens_rotate_to(
                p.x_norm,
                p.x_rot_local,
                paro.pairs,
                paro.theta,
                paro.scales,
                1,
                p.hidden,
                paro.krot,
            )
            .unwrap();
        } else {
            gpu.rotate_x_mq(p.x_norm, p.x_rot_local, p.hidden).unwrap();
        }
        super::launch_fused_shared_gate(
            gpu,
            &p.router,
            &p.shared_expert_gate,
            &p.shared_gate_w,
            &p.shared_up_w,
            p.x_rot_local,
            p.router_logits,
            p.scalar_buf,
            p.gate_buf,
            p.up_buf,
            p.smi,
        )
        .unwrap();
        super::launch_moe_softmax_topk(
            gpu,
            p.router_logits,
            p.topk_indices,
            p.topk_weights,
            p.n_exp,
            p.norm_topk_prob,
        )
        .unwrap();
        super::launch_shared_expert_down_body(
            ctx,
            gpu,
            &p.shared_down_w,
            p.gate_buf,
            p.up_buf,
            p.scalar_buf,
            p.ffn_hidden,
            p.ffn_out,
            out_target,
            p.smi,
        )
        .unwrap();
    }

    /// TEST-ONLY legacy decode reference, routed block (uniform MQ4/MQ6):
    /// indexed gate_up → activation → expanded down → combine.
    fn legacy_decode_routed_uniform(
        gpu: &mut rdna_compute::Gpu,
        p: &MoeParams<'_>,
        out_target: &GpuTensor,
    ) {
        let (gu, dn) = decode_expert_refs(p);
        super::launch_indexed_gate_up(
            gpu,
            &gu,
            p.topk_indices,
            p.x_rot_local,
            p.gate_batch,
            p.up_batch,
            p.k,
        )
        .unwrap();
        super::launch_moe_activation(
            gpu,
            &MoeActivationVariant::MinimaxFused { awq_scale: None },
            p.gate_batch,
            p.up_batch,
            p.rot_batch,
            p.mi,
            p.k,
        )
        .unwrap();
        super::launch_indexed_down(
            gpu,
            &dn,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.k,
            1,
        )
        .unwrap();
        super::launch_moe_combine(
            gpu,
            p.down_expanded,
            p.topk_weights,
            out_target,
            p.routed_down_m,
            p.k,
            1,
        )
        .unwrap();
    }

    /// Run the flattened Step program for a decode layer (one segment — the
    /// HIPFIRE_DUMP_HIDDEN split is a run_moe_decode concern, not a kernel
    /// one).
    fn step_decode(
        gpu: &mut rdna_compute::Gpu,
        ctx: &DispatchCtx,
        p: &MoeParams<'_>,
        res: &MoeResolution,
    ) {
        let (gu, dn) = decode_expert_refs(p);
        let build = build_moe_decode_steps(p, res, &gu, &dn).unwrap();
        let phases = build
            .into_gpu()
            .expect("use_gpu_topk layer must build a Gpu program");
        let program = phases.into_build();
        execute_steps_mesh(
            &hipfire_hardware::DeviceMesh::single(),
            gpu,
            ctx,
            &program.steps,
        )
        .unwrap();
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_mq4_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        let res = MoeResolution::resolve(&d, PH_K);
        assert!(res.use_gpu_topk);
        let mut seed_a = 1u64;
        let mut seed_b = 1u64;
        let rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, None, false, false);
        let rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, None, false, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        legacy_decode_front(&mut gpu, &ctx, &pa, pa.x_residual, &res);
        legacy_decode_routed_uniform(&mut gpu, &pa, pa.x_residual);
        step_decode(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_residual).unwrap();
        let out_b = gpu.download_f32(pb.x_residual).unwrap();
        ph_assert_bits_eq("decode MQ4", &out_a, &out_b);
    }

    #[cfg(feature = "deltanet")]
    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_non_mq4_shared_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let mut d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        d.shared_expert_down = DType::F16;
        let res = MoeResolution::resolve(&d, PH_K);
        assert!(res.use_gpu_topk);
        let mut seed_a = 12u64;
        let mut seed_b = 12u64;
        let mut rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, None, false, false);
        let mut rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, None, false, false);
        rig_a.shared_down_w_buf = ph_upload_f16_ones(&mut gpu, &[PH_HIDDEN, PH_SMI]);
        rig_b.shared_down_w_buf = ph_upload_f16_ones(&mut gpu, &[PH_HIDDEN, PH_SMI]);
        ph_fill_f32(&mut gpu, &rig_a.gate_buf, 1.0);
        ph_fill_f32(&mut gpu, &rig_a.up_buf, 1.0);
        ph_fill_f32(&mut gpu, &rig_a.scalar_buf, 0.0);
        ph_fill_f32(&mut gpu, &rig_b.gate_buf, 1.0);
        ph_fill_f32(&mut gpu, &rig_b.up_buf, 1.0);
        ph_fill_f32(&mut gpu, &rig_b.scalar_buf, 0.0);
        let pa = rig_a.params();
        let pb = rig_b.params();

        super::launch_shared_expert_down(
            &ctx,
            &mut gpu,
            &pa.shared_down_w,
            pa.gate_buf,
            pa.up_buf,
            pa.scalar_buf,
            pa.ffn_hidden,
            pa.ffn_out,
            pa.x_residual,
            pa.smi,
        )
        .unwrap();
        let (gu, dn) = decode_expert_refs(&pb);
        let phases = build_moe_decode_steps(&pb, &res, &gu, &dn)
            .unwrap()
            .into_gpu()
            .unwrap();
        execute_steps_mesh(
            &hipfire_hardware::DeviceMesh::single(),
            &mut gpu,
            &ctx,
            &phases.shared_down,
        )
        .unwrap();

        let out_a = gpu.download_f32(pa.x_residual).unwrap();
        let out_b = gpu.download_f32(pb.x_residual).unwrap();
        ph_assert_bits_eq("decode non-MQ4 shared", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_mq6_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MQ6G256, DType::MQ6G256);
        let res = MoeResolution::resolve(&d, PH_K);
        assert!(res.use_gpu_topk);
        let mut seed_a = 2u64;
        let mut seed_b = 2u64;
        let rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, None, false, false);
        let rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, None, false, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        legacy_decode_front(&mut gpu, &ctx, &pa, pa.x_residual, &res);
        legacy_decode_routed_uniform(&mut gpu, &pa, pa.x_residual);
        step_decode(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_residual).unwrap();
        let out_b = gpu.download_f32(pb.x_residual).unwrap();
        ph_assert_bits_eq("decode MQ6", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_mixed_tags_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let mut d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        d.routed_has_mixed_experts = true;
        d.experts_all_gate_up_mq4 = false;
        let tags: [u8; PH_N_EXP] = [0, 0, 2, 2, 3, 3, 1, 1]; // MQ6, MQ4, MQ3L, MQ2L
        let res = MoeResolution::resolve(&d, PH_K);
        assert!(res.use_gpu_topk);
        let mut seed_a = 3u64;
        let mut seed_b = 3u64;
        let rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, Some(&tags), false, false);
        let rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, Some(&tags), false, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        // Legacy: merged dtype-tag gate_up + down, plain activation, combine.
        legacy_decode_front(&mut gpu, &ctx, &pa, pa.x_residual, &res);
        let (gu, dn) = decode_expert_refs(&pa);
        gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
            pa.expert_gate_up_ptrs,
            pa.expert_dtype_tags.unwrap(),
            pa.topk_indices,
            pa.x_rot_local,
            pa.gate_batch,
            pa.up_batch,
            2 * pa.mi,
            pa.routed_gate_up_k,
            pa.k,
            1,
        )
        .unwrap();
        gpu.fused_silu_mul_rotate_mq_batched(pa.gate_batch, pa.up_batch, pa.rot_batch, pa.mi, pa.k)
            .unwrap();
        gpu.gemv_mixed_moe_down_k8_indexed_batched_expanded(
            pa.expert_down_ptrs,
            pa.expert_dtype_tags.unwrap(),
            pa.topk_indices,
            pa.rot_batch,
            pa.down_expanded,
            pa.routed_down_m,
            pa.routed_down_k,
            pa.k,
            1,
        )
        .unwrap();
        gpu.moe_down_combine_k8_batched(
            pa.down_expanded,
            pa.topk_weights,
            pa.x_residual,
            pa.routed_down_m,
            pa.k,
            1,
        )
        .unwrap();
        drop(gu);
        drop(dn);
        step_decode(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_residual).unwrap();
        let out_b = gpu.download_f32(pb.x_residual).unwrap();
        ph_assert_bits_eq("decode mixed tags", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_paro_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::ParoQ4G128, DType::ParoQ4G128);
        let res = MoeResolution::resolve(&d, PH_K);
        assert!(res.routed_indexable_paro && res.use_gpu_topk);
        let mut seed_a = 4u64;
        let mut seed_b = 4u64;
        let rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, None, false, true);
        let rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, None, false, true);
        let pa = rig_a.params();
        let pb = rig_b.params();
        // Legacy: givens rotate → fused gate → softmax/topk → shared down →
        // paro indexed gate_up → paro activation → paro down → combine.
        legacy_decode_front(&mut gpu, &ctx, &pa, pa.x_residual, &res);
        let (gu, dn) = decode_expert_refs(&pa);
        gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed(
            pa.expert_gate_up_ptrs,
            pa.topk_indices,
            pa.x_rot_local,
            pa.gate_batch,
            pa.up_batch,
            2 * pa.mi,
            pa.routed_gate_up_k,
            pa.k,
        )
        .unwrap();
        let paro = pa.routed_down_paro.as_ref().unwrap();
        gpu.fused_silu_mul_givens_rotate_f32(
            pa.gate_batch,
            pa.up_batch,
            pa.rot_batch,
            paro.pairs,
            paro.theta,
            paro.scales,
            pa.k,
            pa.mi,
            paro.krot,
        )
        .unwrap();
        gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
            pa.expert_down_ptrs,
            pa.topk_indices,
            pa.rot_batch,
            pa.down_expanded,
            pa.routed_down_m,
            pa.routed_down_k,
            pa.k,
            1,
        )
        .unwrap();
        gpu.moe_down_combine_k8_batched(
            pa.down_expanded,
            pa.topk_weights,
            pa.x_residual,
            pa.routed_down_m,
            pa.k,
            1,
        )
        .unwrap();
        drop(gu);
        drop(dn);
        step_decode(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_residual).unwrap();
        let out_b = gpu.download_f32(pb.x_residual).unwrap();
        ph_assert_bits_eq("decode Paro", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_e8_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MFP4G32E8, DType::MFP4G32E8);
        let res = MoeResolution::resolve_arch(&d, PH_K, true);
        assert!(res.routed_indexable_e8 && res.use_gpu_topk);
        let mut seed_a = 5u64;
        let mut seed_b = 5u64;
        let rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, None, false, false);
        let rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, None, false, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        // Legacy: FWHT rotate → fused gate → softmax/topk → shared down →
        // E8 indexed gate_up → activation → E8 down → combine.
        legacy_decode_front(&mut gpu, &ctx, &pa, pa.x_residual, &res);
        let (gu, dn) = decode_expert_refs(&pa);
        gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed(
            pa.expert_gate_up_ptrs,
            pa.topk_indices,
            pa.x_rot_local,
            pa.gate_batch,
            pa.up_batch,
            2 * pa.mi,
            pa.routed_gate_up_k,
        )
        .unwrap();
        gpu.fused_silu_mul_rotate_mq_batched(pa.gate_batch, pa.up_batch, pa.rot_batch, pa.mi, pa.k)
            .unwrap();
        gpu.gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
            pa.expert_down_ptrs,
            pa.topk_indices,
            pa.rot_batch,
            pa.down_expanded,
            pa.routed_down_m,
            pa.routed_down_k,
            pa.k,
            1,
        )
        .unwrap();
        gpu.moe_down_combine_k8_batched(
            pa.down_expanded,
            pa.topk_weights,
            pa.x_residual,
            pa.routed_down_m,
            pa.k,
            1,
        )
        .unwrap();
        drop(gu);
        drop(dn);
        step_decode(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_residual).unwrap();
        let out_b = gpu.download_f32(pb.x_residual).unwrap();
        ph_assert_bits_eq("decode E8", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_awq_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        let res = MoeResolution::resolve(&d, PH_K);
        assert!(res.use_gpu_topk);
        let mut seed_a = 6u64;
        let mut seed_b = 6u64;
        let rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, None, true, false);
        let rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, None, true, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        // Legacy: indexed AWQ activation between gate_up and down.
        legacy_decode_front(&mut gpu, &ctx, &pa, pa.x_residual, &res);
        let (gu, dn) = decode_expert_refs(&pa);
        super::launch_indexed_gate_up(
            &mut gpu,
            &gu,
            pa.topk_indices,
            pa.x_rot_local,
            pa.gate_batch,
            pa.up_batch,
            pa.k,
        )
        .unwrap();
        gpu.fused_silu_mul_rotate_mq_awq_indexed_batched(
            pa.gate_batch,
            pa.up_batch,
            pa.expert_down_awq_ptrs.unwrap(),
            pa.topk_indices,
            pa.rot_batch,
            pa.mi,
            pa.k,
        )
        .unwrap();
        super::launch_indexed_down(
            &mut gpu,
            &dn,
            pa.topk_indices,
            pa.rot_batch,
            pa.down_expanded,
            pa.k,
            1,
        )
        .unwrap();
        super::launch_moe_combine(
            &mut gpu,
            pa.down_expanded,
            pa.topk_weights,
            pa.x_residual,
            pa.routed_down_m,
            pa.k,
            1,
        )
        .unwrap();
        step_decode(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_residual).unwrap();
        let out_b = gpu.download_f32(pb.x_residual).unwrap();
        ph_assert_bits_eq("decode AWQ", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_decode_routed_out_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        let res = MoeResolution::resolve(&d, PH_K);
        assert!(res.use_gpu_topk);
        let mut seed_a = 7u64;
        let mut seed_b = 7u64;
        let mut rig_a = DecodeRig::new(&mut gpu, d.clone(), &mut seed_a, None, false, false);
        let mut rig_b = DecodeRig::new(&mut gpu, d, &mut seed_b, None, false, false);
        rig_a.x_residual = ph_upload(&mut gpu, &[PH_HIDDEN], &mut seed_a);
        rig_b.x_residual = ph_upload(&mut gpu, &[PH_HIDDEN], &mut seed_b);
        let partial_a = ph_zero(&mut gpu, &[PH_HIDDEN]);
        let partial_b = ph_zero(&mut gpu, &[PH_HIDDEN]);
        let mut pa = rig_a.params();
        let mut pb = rig_b.params();
        pa.routed_out = Some(&partial_a);
        pb.routed_out = Some(&partial_b);
        // Legacy: routed combine (and shared down) accumulate into the
        // zeroed partial instead of x_residual.
        legacy_decode_front(&mut gpu, &ctx, &pa, &partial_a, &res);
        legacy_decode_routed_uniform(&mut gpu, &pa, &partial_a);
        step_decode(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(&partial_a).unwrap();
        let out_b = gpu.download_f32(&partial_b).unwrap();
        ph_assert_bits_eq("decode routed_out", &out_a, &out_b);
    }

    // ── Prefill parity rig (Path 1 / Path 2) ────────────────────────────────

    const PH_N: usize = 4; // batch tokens
    const PH_M_TOTAL: usize = PH_N * PH_K + PH_N_EXP * 16;

    /// One real (garbage-weight) MQ4/E8/Paro MoE prefill layer. `x_rot_batch`
    /// is FWHT-pre-rotated in setup for MQ/E8 (the legacy model-owned
    /// rotation); Paro rigs leave it for the in-program Givens preamble.
    struct PrefillRig {
        dtypes: MoeDtypes,
        _expert_gu_bufs: Vec<GpuTensor>,
        _expert_dn_bufs: Vec<GpuTensor>,
        expert_gate_up_ptrs: GpuTensor,
        expert_down_ptrs: GpuTensor,
        expert_dtype_tags: Option<GpuTensor>,
        x_batch: GpuTensor,
        x_norm_batch: GpuTensor,
        x_rot_batch: GpuTensor,
        gate_batch: GpuTensor,
        up_batch: GpuTensor,
        rot_batch: GpuTensor,
        down_expanded: GpuTensor,
        y_gate_up_grouped: GpuTensor,
        y_down_grouped: GpuTensor,
        expert_token_counts: GpuTensor,
        expert_offsets: GpuTensor,
        sorted_slot_index: GpuTensor,
        expert_tile_ids: GpuTensor,
        inverse_perm: GpuTensor,
        topk_indices: GpuTensor,
        topk_weights: GpuTensor,
        paro_gu: Option<(GpuTensor, GpuTensor, GpuTensor)>,
        paro_dn: Option<(GpuTensor, GpuTensor, GpuTensor)>,
    }

    impl PrefillRig {
        fn new(
            gpu: &mut rdna_compute::Gpu,
            dtypes: MoeDtypes,
            seed: &mut u64,
            pre_rotate: bool,
            with_paro: bool,
        ) -> Self {
            let mut expert_gu_bufs = Vec::new();
            let mut expert_dn_bufs = Vec::new();
            for _ in 0..PH_N_EXP {
                expert_gu_bufs.push(ph_upload(gpu, &[2 * PH_MI * PH_HIDDEN], seed));
                expert_dn_bufs.push(ph_upload(gpu, &[PH_HIDDEN * PH_MI], seed));
            }
            let expert_gate_up_ptrs = ph_ptr_table(gpu, &expert_gu_bufs);
            let expert_down_ptrs = ph_ptr_table(gpu, &expert_dn_bufs);
            let x_norm_batch = ph_upload(gpu, &[PH_N * PH_HIDDEN], seed);
            let x_rot_batch = ph_zero(gpu, &[PH_N * PH_HIDDEN]);
            if pre_rotate {
                gpu.rotate_x_mq_batched(&x_norm_batch, &x_rot_batch, PH_HIDDEN, PH_N)
                    .unwrap();
            }
            // topk: token t, rank r → expert (t + r) % n_exp (full coverage).
            let mut idx_bits = Vec::with_capacity(PH_N * PH_K);
            for t in 0..PH_N {
                for r in 0..PH_K {
                    idx_bits.push(f32::from_bits(((t + r) % PH_N_EXP) as u32));
                }
            }
            let topk_indices = gpu.upload_f32(&idx_bits, &[PH_N * PH_K]).unwrap();
            let paro_gu = with_paro.then(|| ph_valid_paro_sidecars(gpu));
            let paro_dn = with_paro.then(|| ph_valid_paro_sidecars(gpu));
            Self {
                dtypes,
                _expert_gu_bufs: expert_gu_bufs,
                _expert_dn_bufs: expert_dn_bufs,
                expert_gate_up_ptrs,
                expert_down_ptrs,
                expert_dtype_tags: None,
                x_batch: ph_zero(gpu, &[PH_N * PH_HIDDEN]),
                x_norm_batch,
                x_rot_batch,
                gate_batch: ph_zero(gpu, &[PH_N * PH_K * PH_MI]),
                up_batch: ph_zero(gpu, &[PH_N * PH_K * PH_MI]),
                rot_batch: ph_zero(gpu, &[PH_N * PH_K * PH_MI]),
                down_expanded: ph_zero(gpu, &[PH_N * PH_K * PH_HIDDEN]),
                y_gate_up_grouped: ph_zero(gpu, &[PH_M_TOTAL * 2 * PH_MI]),
                y_down_grouped: ph_zero(gpu, &[PH_M_TOTAL * PH_HIDDEN]),
                expert_token_counts: ph_zero(gpu, &[PH_N_EXP]),
                expert_offsets: ph_zero(gpu, &[PH_N_EXP + 1]),
                sorted_slot_index: ph_zero(gpu, &[PH_M_TOTAL]),
                expert_tile_ids: ph_zero(gpu, &[PH_M_TOTAL]),
                inverse_perm: ph_zero(gpu, &[PH_N * PH_K]),
                topk_indices,
                topk_weights: ph_upload(gpu, &[PH_N * PH_K], seed),
                paro_gu,
                paro_dn,
            }
        }

        fn params(&self) -> MoePrefillParams<'_> {
            MoePrefillParams {
                dtypes: self.dtypes.clone(),
                batch_size: PH_N,
                mi: PH_MI,
                down_m: PH_HIDDEN,
                down_k: PH_MI,
                gate_up_k: PH_HIDDEN,
                k_top: PH_K,
                n_exp: PH_N_EXP,
                m_total_max: PH_M_TOTAL,
                force_mq4_grouped_fp16: false,
                topk_indices: &self.topk_indices,
                topk_weights: &self.topk_weights,
                x_batch: &self.x_batch,
                x_norm_batch: &self.x_norm_batch,
                x_rot_batch: &self.x_rot_batch,
                expert_gate_up_ptrs: &self.expert_gate_up_ptrs,
                expert_down_ptrs: &self.expert_down_ptrs,
                expert_down_awq_ptrs: None,
                expert_dtype_tags: self.expert_dtype_tags.as_ref(),
                gate_batch: &self.gate_batch,
                up_batch: &self.up_batch,
                rot_batch: &self.rot_batch,
                down_expanded: &self.down_expanded,
                expert_token_counts: &self.expert_token_counts,
                expert_offsets: &self.expert_offsets,
                sorted_slot_index: &self.sorted_slot_index,
                expert_tile_ids: &self.expert_tile_ids,
                inverse_perm: &self.inverse_perm,
                y_gate_up_grouped: &self.y_gate_up_grouped,
                y_down_grouped: &self.y_down_grouped,
                paro_gate_up: self
                    .paro_gu
                    .as_ref()
                    .map(|(pairs, theta, scales)| GivensRef {
                        pairs,
                        theta,
                        scales,
                        krot: 64,
                    }),
                paro_down: self
                    .paro_dn
                    .as_ref()
                    .map(|(pairs, theta, scales)| GivensRef {
                        pairs,
                        theta,
                        scales,
                        krot: 64,
                    }),
                down_awq_scale: None,
                routed_out: None,
            }
        }
    }

    /// Run the flattened prefill Step program.
    fn step_prefill(
        gpu: &mut rdna_compute::Gpu,
        ctx: &DispatchCtx,
        p: &MoePrefillParams<'_>,
        res: &MoePrefillResolution,
    ) {
        let (gu, dn) = prefill_expert_refs(p);
        let phases = build_moe_prefill_steps(p, res, &gu, &dn).unwrap();
        let program = phases.into_build();
        execute_steps_mesh(
            &hipfire_hardware::DeviceMesh::single(),
            gpu,
            ctx,
            &program.steps,
        )
        .unwrap();
    }

    /// gfx1151 resolution with an explicit grouped-gemm lever.
    fn ph_prefill_res(
        _gpu: &rdna_compute::Gpu,
        d: &MoeDtypes,
        grouped: bool,
    ) -> MoePrefillResolution {
        let mut flags = rdna_compute::feature_flags::FeatureFlags::from_env_for_test("gfx1151");
        flags.moe_grouped_gemm = grouped;
        let arch =
            rdna_compute::arch_caps::ArchCaps::new("gfx1151", std::sync::Arc::new(flags.clone()));
        MoePrefillResolution::resolve(d, &arch, &flags)
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); forces the Path-0 kernel on gfx1151 and runs under scripts/gpu-lock.sh"]
    fn qwen_step_parity_prefill_path0_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        let res = MoePrefillResolution {
            use_path2: false,
            down_path0: true,
            use_paro_i8: false,
            use_paro_i8_k8: false,
            paro_mode: false,
            force_mq4_grouped_fp16: false,
        };
        let mut seed_a = 13u64;
        let mut seed_b = 13u64;
        let mut rig_a = PrefillRig::new(&mut gpu, d.clone(), &mut seed_a, true, false);
        let mut rig_b = PrefillRig::new(&mut gpu, d, &mut seed_b, true, false);
        let a_gate_up = ph_repeated_ptr_table(&mut gpu, &rig_a._expert_gu_bufs[0], PH_N_EXP);
        let a_down = ph_repeated_ptr_table(&mut gpu, &rig_a._expert_dn_bufs[0], PH_N_EXP);
        let b_gate_up = ph_repeated_ptr_table(&mut gpu, &rig_b._expert_gu_bufs[0], PH_N_EXP);
        let b_down = ph_repeated_ptr_table(&mut gpu, &rig_b._expert_dn_bufs[0], PH_N_EXP);
        rig_a.expert_gate_up_ptrs = a_gate_up;
        rig_a.expert_down_ptrs = a_down;
        rig_b.expert_gate_up_ptrs = b_gate_up;
        rig_b.expert_down_ptrs = b_down;
        ph_fill_f32(&mut gpu, &rig_a.topk_weights, 0.125);
        ph_fill_f32(&mut gpu, &rig_b.topk_weights, 0.125);
        let pa = rig_a.params();
        let pb = rig_b.params();

        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            pa.expert_gate_up_ptrs,
            pa.topk_indices,
            pa.x_rot_batch,
            pa.gate_batch,
            pa.up_batch,
            2 * pa.mi,
            pa.gate_up_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        gpu.fused_silu_mul_rotate_mq_batched(
            pa.gate_batch,
            pa.up_batch,
            pa.rot_batch,
            pa.mi,
            pa.batch_size * pa.k_top,
        )
        .unwrap();
        gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
            pa.expert_down_ptrs,
            pa.topk_indices,
            pa.topk_weights,
            pa.rot_batch,
            pa.x_batch,
            pa.down_m,
            pa.down_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        step_prefill(&mut gpu, &ctx, &pb, &res);

        let out_a = gpu.download_f32(pa.x_batch).unwrap();
        let out_b = gpu.download_f32(pb.x_batch).unwrap();
        // Every K=8 slot uses identical expert weights and exact 1/8 routing
        // weight, making the atomic sum order-independent and bitwise testable.
        ph_assert_bits_eq("prefill forced Path 0", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_prefill_path1_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        let res = ph_prefill_res(&gpu, &d, false);
        assert!(!res.use_path2 && !res.down_path0, "grouped off → Path 1");
        let mut seed_a = 8u64;
        let mut seed_b = 8u64;
        let rig_a = PrefillRig::new(&mut gpu, d.clone(), &mut seed_a, true, false);
        let rig_b = PrefillRig::new(&mut gpu, d, &mut seed_b, true, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        // Legacy Path 1: indexed batched gate_up → activation → expanded
        // down → combine.
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            pa.expert_gate_up_ptrs,
            pa.topk_indices,
            pa.x_rot_batch,
            pa.gate_batch,
            pa.up_batch,
            2 * pa.mi,
            pa.gate_up_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        gpu.fused_silu_mul_rotate_mq_batched(
            pa.gate_batch,
            pa.up_batch,
            pa.rot_batch,
            pa.mi,
            pa.batch_size * pa.k_top,
        )
        .unwrap();
        gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
            pa.expert_down_ptrs,
            pa.topk_indices,
            pa.rot_batch,
            pa.down_expanded,
            pa.down_m,
            pa.down_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        gpu.moe_down_combine_k8_batched(
            pa.down_expanded,
            pa.topk_weights,
            pa.x_batch,
            pa.down_m,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        step_prefill(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_batch).unwrap();
        let out_b = gpu.download_f32(pb.x_batch).unwrap();
        ph_assert_bits_eq("prefill Path 1", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_prefill_path2_gpu() {
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MQ4G256, DType::MQ4G256);
        let res = ph_prefill_res(&gpu, &d, true);
        assert!(res.use_path2, "grouped on → Path 2");
        let mut seed_a = 9u64;
        let mut seed_b = 9u64;
        let rig_a = PrefillRig::new(&mut gpu, d.clone(), &mut seed_a, true, false);
        let rig_b = PrefillRig::new(&mut gpu, d, &mut seed_b, true, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        // Legacy Path 2: scatter → grouped gate_up → unscatter → activation →
        // grouped down → grouped combine.
        gpu.moe_scatter_fused_k8(
            pa.topk_indices,
            pa.expert_token_counts,
            pa.expert_offsets,
            pa.sorted_slot_index,
            pa.expert_tile_ids,
            pa.inverse_perm,
            pa.batch_size * pa.k_top,
            pa.n_exp,
            pa.m_total_max,
            16,
        )
        .unwrap();
        gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
            pa.expert_gate_up_ptrs,
            pa.expert_tile_ids,
            pa.sorted_slot_index,
            pa.x_rot_batch,
            pa.y_gate_up_grouped,
            2 * pa.mi,
            pa.gate_up_k,
            pa.k_top,
            pa.m_total_max,
            pa.batch_size,
        )
        .unwrap();
        gpu.moe_gate_up_unscatter_k8(
            pa.y_gate_up_grouped,
            pa.sorted_slot_index,
            pa.gate_batch,
            pa.up_batch,
            pa.mi,
            pa.k_top,
            pa.m_total_max,
        )
        .unwrap();
        gpu.fused_silu_mul_rotate_mq_batched(
            pa.gate_batch,
            pa.up_batch,
            pa.rot_batch,
            pa.mi,
            pa.batch_size * pa.k_top,
        )
        .unwrap();
        gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
            pa.expert_down_ptrs,
            pa.expert_tile_ids,
            pa.sorted_slot_index,
            pa.rot_batch,
            pa.y_down_grouped,
            pa.down_m,
            pa.down_k,
            1,
            pa.m_total_max,
            pa.batch_size * pa.k_top,
        )
        .unwrap();
        gpu.moe_down_combine_grouped_k8(
            pa.y_down_grouped,
            pa.inverse_perm,
            pa.topk_weights,
            pa.x_batch,
            pa.down_m,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        step_prefill(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_batch).unwrap();
        let out_b = gpu.download_f32(pb.x_batch).unwrap();
        ph_assert_bits_eq("prefill Path 2", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_prefill_path1_e8_gpu() {
        // GPU proof of remediation 1: batch>1 E8 gate_up must use the
        // _batched kernel (the Step program would diverge from the legacy
        // batched sequence otherwise).
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::MFP4G32E8, DType::MFP4G32E8);
        let res = ph_prefill_res(&gpu, &d, false);
        assert!(!res.use_path2 && !res.down_path0, "grouped off → Path 1");
        let mut seed_a = 10u64;
        let mut seed_b = 10u64;
        let rig_a = PrefillRig::new(&mut gpu, d.clone(), &mut seed_a, true, false);
        let rig_b = PrefillRig::new(&mut gpu, d, &mut seed_b, true, false);
        let pa = rig_a.params();
        let pb = rig_b.params();
        gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
            pa.expert_gate_up_ptrs,
            pa.topk_indices,
            pa.x_rot_batch,
            pa.gate_batch,
            pa.up_batch,
            2 * pa.mi,
            pa.gate_up_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        gpu.fused_silu_mul_rotate_mq_batched(
            pa.gate_batch,
            pa.up_batch,
            pa.rot_batch,
            pa.mi,
            pa.batch_size * pa.k_top,
        )
        .unwrap();
        gpu.gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
            pa.expert_down_ptrs,
            pa.topk_indices,
            pa.rot_batch,
            pa.down_expanded,
            pa.down_m,
            pa.down_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        gpu.moe_down_combine_k8_batched(
            pa.down_expanded,
            pa.topk_weights,
            pa.x_batch,
            pa.down_m,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        step_prefill(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_batch).unwrap();
        let out_b = gpu.download_f32(pb.x_batch).unwrap();
        ph_assert_bits_eq("prefill Path 1 E8", &out_a, &out_b);
    }

    #[test]
    #[ignore = "requires an AMD GPU (gfx11); run under the repo GPU lock (scripts/gpu-lock.sh)"]
    fn qwen_step_parity_prefill_path1_paro_gpu() {
        // GPU proof of remediation 2: batch>1 Paro gate_up must use the
        // _batched kernel behind the mandatory Givens preamble.
        let mut gpu = rdna_compute::Gpu::init().expect("GPU required");
        let ctx = DispatchCtx::new(&gpu);
        let d = ph_mq4_dtypes(DType::ParoQ4G128, DType::ParoQ4G128);
        let res = ph_prefill_res(&gpu, &d, false);
        assert!(res.paro_mode && !res.use_path2 && !res.down_path0);
        let mut seed_a = 11u64;
        let mut seed_b = 11u64;
        let rig_a = PrefillRig::new(&mut gpu, d.clone(), &mut seed_a, false, true);
        let rig_b = PrefillRig::new(&mut gpu, d, &mut seed_b, false, true);
        let pa = rig_a.params();
        let pb = rig_b.params();
        // Legacy: givens preamble → paro batched gate_up → paro activation →
        // paro down → combine.
        let paro_gu = pa.paro_gate_up.as_ref().unwrap();
        gpu.givens_rotate_to(
            pa.x_norm_batch,
            pa.x_rot_batch,
            paro_gu.pairs,
            paro_gu.theta,
            paro_gu.scales,
            pa.batch_size,
            pa.gate_up_k,
            paro_gu.krot,
        )
        .unwrap();
        gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
            pa.expert_gate_up_ptrs,
            pa.topk_indices,
            pa.x_rot_batch,
            pa.gate_batch,
            pa.up_batch,
            2 * pa.mi,
            pa.gate_up_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        let paro_dn = pa.paro_down.as_ref().unwrap();
        gpu.fused_silu_mul_givens_rotate_f32(
            pa.gate_batch,
            pa.up_batch,
            pa.rot_batch,
            paro_dn.pairs,
            paro_dn.theta,
            paro_dn.scales,
            pa.batch_size * pa.k_top,
            pa.mi,
            paro_dn.krot,
        )
        .unwrap();
        gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
            pa.expert_down_ptrs,
            pa.topk_indices,
            pa.rot_batch,
            pa.down_expanded,
            pa.down_m,
            pa.down_k,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        gpu.moe_down_combine_k8_batched(
            pa.down_expanded,
            pa.topk_weights,
            pa.x_batch,
            pa.down_m,
            pa.k_top,
            pa.batch_size,
        )
        .unwrap();
        step_prefill(&mut gpu, &ctx, &pb, &res);
        let out_a = gpu.download_f32(pa.x_batch).unwrap();
        let out_b = gpu.download_f32(pb.x_batch).unwrap();
        ph_assert_bits_eq("prefill Path 1 Paro", &out_a, &out_b);
    }

    // ── DeepSeek batched indexed protocol (Phase 3 shared lane) ─────────────
    // The Step executor keeps `IndexedMoeGemv.batch_size` authoritative:
    // batch one keeps the scalar launchers byte-identically, batch > 1 selects
    // the MQ2-Lloyd batched kernels, and anything else rejects explicitly —
    // there is NEVER a scalar fallback for a batched form.

    #[test]
    fn deepseek_gate_up_form_batch_one_keeps_scalar_launcher() {
        for dtype in [
            DType::MQ2G256Lloyd,
            DType::MQ3G256Lloyd,
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::HFQ6G256,
        ] {
            assert_eq!(
                deepseek_gate_up_indexed_form(dtype, 1),
                DeepSeekIndexedForm::Scalar,
                "{dtype:?} at batch 1 must keep the scalar launcher"
            );
        }
    }

    #[test]
    fn deepseek_gate_up_form_batch_gt_one_is_batched_only_for_mq2_lloyd() {
        assert_eq!(
            deepseek_gate_up_indexed_form(DType::MQ2G256Lloyd, 2),
            DeepSeekIndexedForm::Batched
        );
        // Unsupported batched dtypes reject; never a scalar fallback.
        for dtype in [
            DType::MQ3G256Lloyd,
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::HFQ6G256,
            DType::MQ5G256,
            DType::F32,
        ] {
            assert_eq!(
                deepseek_gate_up_indexed_form(dtype, 2),
                DeepSeekIndexedForm::Unsupported,
                "{dtype:?} batched gate_up must reject without scalar fallback"
            );
        }
    }

    #[test]
    fn deepseek_i64_down_form_batch_one_keeps_scalar_lloyd_launchers() {
        assert_eq!(
            deepseek_i64_down_indexed_form(DType::MQ2G256Lloyd, 1),
            DeepSeekIndexedForm::Scalar
        );
        assert_eq!(
            deepseek_i64_down_indexed_form(DType::MQ3G256Lloyd, 1),
            DeepSeekIndexedForm::Scalar
        );
    }

    #[test]
    fn deepseek_i64_down_form_batch_one_keeps_scalar_for_unsupported_dtypes() {
        // Batch one preserves the scalar launcher for EVERY dtype: the
        // launcher's own dtype validation is the authority (byte-identical
        // error behavior). An unsupported scalar dtype must select Scalar —
        // never a batched/Unsupported error — so the launcher reports the
        // same "only MQ2G256Lloyd/MQ3G256Lloyd supported" error it always has.
        for dtype in [
            DType::F32,
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::HFQ6G256,
            DType::MQ5G256,
            DType::ParoQ4G128,
        ] {
            assert_eq!(
                deepseek_i64_down_indexed_form(dtype, 1),
                DeepSeekIndexedForm::Scalar,
                "{dtype:?} at batch 1 must keep the scalar launcher, not reject"
            );
        }
    }

    #[test]
    fn deepseek_i64_down_form_batch_gt_one_is_batched_only_for_mq2_lloyd() {
        assert_eq!(
            deepseek_i64_down_indexed_form(DType::MQ2G256Lloyd, 2),
            DeepSeekIndexedForm::Batched
        );
        assert_eq!(
            deepseek_i64_down_indexed_form(DType::MQ3G256Lloyd, 2),
            DeepSeekIndexedForm::Unsupported,
            "batched MQ3 i64 down must reject; no scalar fallback"
        );
        assert_eq!(
            deepseek_i64_down_indexed_form(DType::MQ4G256, 2),
            DeepSeekIndexedForm::Unsupported
        );
    }

    #[test]
    fn deepseek_f32_down_form_rejects_batch_gt_one() {
        assert_eq!(
            deepseek_f32_down_indexed_form(1),
            DeepSeekIndexedForm::Scalar
        );
        assert_eq!(
            deepseek_f32_down_indexed_form(2),
            DeepSeekIndexedForm::Unsupported,
            "batched FP32 residual down must reject explicitly"
        );
    }

    #[test]
    fn deepseek_forms_reject_zero_batch() {
        assert_eq!(
            deepseek_gate_up_indexed_form(DType::MQ2G256Lloyd, 0),
            DeepSeekIndexedForm::Unsupported
        );
        assert_eq!(
            deepseek_i64_down_indexed_form(DType::MQ2G256Lloyd, 0),
            DeepSeekIndexedForm::Unsupported
        );
        assert_eq!(
            deepseek_f32_down_indexed_form(0),
            DeepSeekIndexedForm::Unsupported
        );
    }

    #[test]
    fn checked_deepseek_grouped_bounds_aligned_geometry() {
        // 128 tokens × k8 + 8 experts × 16 pad: already 16-aligned.
        let bounds = checked_deepseek_grouped_bounds(128, 8, 8).unwrap();
        assert_eq!(bounds.total_slots, 1024);
        assert_eq!(bounds.m_total_max, 1024 + 128);
        assert_eq!(bounds.tile_count, (1024 + 128) / 16);
    }

    #[test]
    fn checked_deepseek_grouped_bounds_reachable_batch129_k6() {
        // 129×6 = 774 + 8×16 = 902 → align-up to 912 → 57 tiles. The old
        // unaligned local formula produced floor(902/16) = 56 tiles.
        let bounds = checked_deepseek_grouped_bounds(129, 6, 8).unwrap();
        assert_eq!(bounds.total_slots, 774);
        assert_eq!(bounds.m_total_max, 912);
        assert_eq!(bounds.tile_count, 57);
    }

    #[test]
    fn checked_deepseek_grouped_bounds_zero_geometry_is_defined() {
        // Zero batch/k_top keep the expert-pad-only allocation, consistent
        // with the existing direct dispatch path; the formula stays defined.
        let zero_batch = checked_deepseek_grouped_bounds(0, 6, 8).unwrap();
        assert_eq!(zero_batch.total_slots, 0);
        assert_eq!(zero_batch.m_total_max, 128);
        assert_eq!(zero_batch.tile_count, 8);
        let zero_k = checked_deepseek_grouped_bounds(4, 0, 8).unwrap();
        assert_eq!(zero_k.total_slots, 0);
        assert_eq!(zero_k.m_total_max, 128);
    }

    #[test]
    fn checked_deepseek_grouped_bounds_fails_closed_on_overflow() {
        // batch*k overflow.
        assert!(checked_deepseek_grouped_bounds(usize::MAX, 2, 8).is_err());
        // total_slots + expert_pad overflow.
        assert!(checked_deepseek_grouped_bounds(usize::MAX - 8, 1, 8).is_err());
        // align-up (+15) overflow while the sum fits.
        assert!(checked_deepseek_grouped_bounds(usize::MAX - 135, 1, 8).is_err());
        // n_experts * 16 overflow.
        assert!(checked_deepseek_grouped_bounds(4, 1, usize::MAX).is_err());
    }

    #[test]
    fn scatter_block_guard_rejects_zero_or_nonaligned_block() {
        assert!(scatter_block_guard(0, 160).is_err());
        assert!(scatter_block_guard(16, 161).is_err());
        assert!(scatter_block_guard(16, 160).is_ok());
    }

    #[test]
    fn indexed_moe_batch_guard_rejects_zero_before_dispatch() {
        assert!(indexed_moe_batch_guard(0).is_err());
        assert!(indexed_moe_batch_guard(1).is_ok());
        assert!(indexed_moe_batch_guard(2).is_ok());
    }
}
