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

use crate::context::DispatchCtx;
use crate::families::gemv::{GivensRef, WeightRef};
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

use crate::pipeline::MoeActivationVariant;
use crate::pipeline::ScoreActKind;

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

/// Scatter+histogram for grouped-GEMM prefill.
/// Thin wrapper over `gpu.moe_scatter_fused_k8`.
/// Produces `sorted_slot_index`, `expert_tile_ids`, and `inverse_perm` from
/// `topk_indices`; also fills the histogram (`expert_token_counts`) and the
/// exclusive-scan offsets (`expert_offsets`). Must run before the grouped GEMMs.
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
/// sorted by `sorted_slot_index`. Dispatches per `experts.dtype`:
/// - MQ2G256Lloyd → arch-selected variant via `dispatch_grouped_lloyd`
///   (matches `run_moe_prefill_bias_aware`: `Lloyd4w` on gfx11+/gfx12+, `Base` otherwise)
/// - MQ3G256Lloyd → `gemm_mq3g256_lloyd_moe_grouped_wmma`
/// - MQ4G256/HFQ4G256 → `gemm_hfq4g256_moe_grouped_wmma_k2`
/// - MQ6G256/HFQ6G256 → `gemm_hfq6g256_moe_grouped_wmma`
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
) -> Result<(), DispatchError> {
    let m = 2 * experts.expert_m; // fused gate||up rows
    let k = experts.expert_k;
    let x_row_div = k_top;
    let rows = batch_size;
    match experts.dtype {
        DType::MQ2G256Lloyd => {
            // Route through the same variant-selection logic as
            // `run_moe_prefill_bias_aware` so the kernel chosen here can never
            // drift from production (Lloyd4w on gfx11+/gfx12+ by default).
            let arch_4w = gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12");
            let lloyd_4w_base = match std::env::var("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            };
            let n32 = std::env::var("HIPFIRE_DEEPSEEK4_MOE_N32").as_deref() == Ok("1");
            let cnd = std::env::var("HIPFIRE_DEEPSEEK4_MOE_CND").as_deref() == Ok("1");
            let eightw = std::env::var("HIPFIRE_DEEPSEEK4_MOE_8W").as_deref() == Ok("1");
            let mmqload_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_MMQLOAD").as_deref() == Ok("1");
            let nosync_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_NOSYNC").as_deref() == Ok("1");
            // Alignment check mirrors production: (2*im)%64==0 && hidden%256==0 → m%64==0 && k%256==0.
            let use_lloyd_4w = lloyd_4w_base.unwrap_or(arch_4w) && m % 64 == 0 && k % 256 == 0;
            let use_mmqload = use_lloyd_4w && mmqload_env;
            let use_nosync = use_mmqload && nosync_env;
            let variant = crate::pipeline::select_grouped_lloyd_variant(
                use_lloyd_4w,
                n32,
                cnd,
                eightw,
                use_mmqload,
                use_nosync,
            );
            crate::pipeline::dispatch_grouped_lloyd(
                gpu,
                variant,
                experts.gate_up_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
        }
        DType::MQ3G256Lloyd => gpu
            .gemm_mq3g256_lloyd_moe_grouped_wmma(
                experts.gate_up_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemm_hfq4g256_moe_grouped_wmma_k2(
                experts.gate_up_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ6G256 | DType::HFQ6G256 => gpu
            .gemm_hfq6g256_moe_grouped_wmma(
                experts.gate_up_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_grouped_gate_up: unsupported dtype {other:?}"
        ))),
    }
}

/// Grouped down GEMM (Path 2 prefill): one launch covers all expert tokens
/// sorted by `sorted_slot_index`. Dispatches per `experts.dtype` (same
/// kernels as gate_up, different dims).
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
) -> Result<(), DispatchError> {
    let m = experts.expert_k; // down output = hidden
    let k = experts.expert_m; // down input  = inter
    let x_row_div = 1;
    let rows = batch_size * k_top;
    match experts.dtype {
        DType::MQ2G256Lloyd => {
            // Route through the same variant-selection logic as
            // `run_moe_prefill_bias_aware` so the kernel chosen here can never
            // drift from production (Lloyd4w on gfx11+/gfx12+ by default).
            let arch_4w = gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12");
            let lloyd_4w_base = match std::env::var("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            };
            let n32 = std::env::var("HIPFIRE_DEEPSEEK4_MOE_N32").as_deref() == Ok("1");
            let cnd = std::env::var("HIPFIRE_DEEPSEEK4_MOE_CND").as_deref() == Ok("1");
            let eightw = std::env::var("HIPFIRE_DEEPSEEK4_MOE_8W").as_deref() == Ok("1");
            let mmqload_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_MMQLOAD").as_deref() == Ok("1");
            let nosync_env = std::env::var("HIPFIRE_DEEPSEEK4_MOE_NOSYNC").as_deref() == Ok("1");
            // Alignment check mirrors production: hidden%64==0 && im%256==0 → m%64==0 && k%256==0.
            let use_lloyd_4w = lloyd_4w_base.unwrap_or(arch_4w) && m % 64 == 0 && k % 256 == 0;
            let use_mmqload = use_lloyd_4w && mmqload_env;
            let use_nosync = use_mmqload && nosync_env;
            let variant = crate::pipeline::select_grouped_lloyd_variant(
                use_lloyd_4w,
                n32,
                cnd,
                eightw,
                use_mmqload,
                use_nosync,
            );
            crate::pipeline::dispatch_grouped_lloyd(
                gpu,
                variant,
                experts.down_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
        }
        DType::MQ3G256Lloyd => gpu
            .gemm_mq3g256_lloyd_moe_grouped_wmma(
                experts.down_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemm_hfq4g256_moe_grouped_wmma_k2(
                experts.down_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ6G256 | DType::HFQ6G256 => gpu
            .gemm_hfq6g256_moe_grouped_wmma(
                experts.down_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        other => Err(DispatchError::Hip(format!(
            "launch_grouped_down: unsupported dtype {other:?}"
        ))),
    }
}

/// Unscatter grouped gate_up result: `y_grouped → gate_batch + up_batch`.
/// Thin wrapper over `gpu.moe_gate_up_unscatter_k8`.
/// Call after [`launch_grouped_gate_up`] (before SwiGLU+rotate).
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_unscatter(
    gpu: &mut rdna_compute::Gpu,
    y_grouped: &GpuTensor,
    sorted_slot_index: &GpuTensor,
    gate_batch: &GpuTensor,
    up_batch: &GpuTensor,
    mi: usize,
    k_top: usize,
    m_total: usize,
) -> Result<(), DispatchError> {
    gpu.moe_gate_up_unscatter_k8(
        y_grouped,
        sorted_slot_index,
        gate_batch,
        up_batch,
        mi,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
