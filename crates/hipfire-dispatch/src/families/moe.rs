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
use rdna_compute::{Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, GemvParams, GivensRef, WeightRef};
use crate::tables::moe_table;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;
use std::sync::{LazyLock, OnceLock};

use crate::pipeline::steps::{MoeActivationVariant, MoeProj, QwenDownMode, ScoreActKind, Step};

// ── MoE eligibility lattice ────────────────────────────

/// Routed-expert tiers the mixed-tier graded decode path can execute: the
/// tiers for which per-tier indexed gate_up/down GEMV kernels exist (served
/// on-device via `run_moe_decode`'s `expert_dtype_tags` branch). A per-expert
/// tier table containing any other DType
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
    /// Uniform all-MQ2-Lloyd routed experts (gate_up == down == MQ2G256Lloyd).
    /// Reuses the ds4/minimax indexed Lloyd MoE GEMVs on the decode GPU-top-K
    /// path: gate_up uses the MQ2-Lloyd indexed GEMV, down uses the MQ2-Lloyd
    /// atomic-residual GEMV (self-combining -> no separate down combine).
    pub routed_indexable_mq2lloyd: bool,
    /// Uniform all-MQ3-Lloyd routed experts (gate_up == down == MQ3G256Lloyd).
    /// Same indexed-Lloyd decode path as mq2lloyd, MQ3 launchers.
    pub routed_indexable_mq3lloyd: bool,
    /// Routed experts whose gate_up and down are each drawn, INDEPENDENTLY, from
    /// the codebook family `{MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL}` —
    /// the per-projection allocation (e.g. gate_up 2-bit, down 3-bit) that puts
    /// the cheap bits on the larger projection and the accurate ones on the
    /// residual write. Indexable because `run_moe_decode` already picks the
    /// gate_up and down GEMVs from their own dtypes rather than a coupled flag,
    /// and because ALL FOUR down kernels self-combine via atomicAdd — so
    /// `routed_down_self_combines` (keyed on `routed_down` alone) stays correct
    /// and the shared down-combine is skipped exactly once. silu+rotate is
    /// weight-agnostic (it reads activations only). Subsumes the two uniform
    /// Lloyd arms above and the uniform GL cases.
    ///
    /// Lloyd and GL are freely mixable across the two projections: both are
    /// FWHT-G256 formats consuming the same rotated activation, and each GEMV is
    /// selected from its own projection's dtype. The ONLY thing that differs is
    /// where the codebook comes from (per-group fp16 header vs scalar kernel
    /// args), which is entirely inside the launcher.
    ///
    /// Decode-only: batched prefill rejects MoE MQ3-Lloyd outright (see
    /// `moe_ffn_has_mq3_experts_uniform` in hipfire-arch-qwen35), which already
    /// blocks the pre-existing uniform MQ3-Lloyd path too; the GL dtypes are
    /// likewise not admitted by `moe_ffn_batched_admissible_for_dtypes`, so a
    /// GL model prefills through the per-token path.
    pub routed_indexable_mixed_lloyd: bool,
    /// Per-expert N-tier graded routed experts (MQ6 hot / MQ4 mid / MQ2L or
    /// MQ3L cold, applied to BOTH gate_up and down). Indexable on the decode
    /// GPU-top-K path via the merged dtype-tag-branched gate_up AND down
    /// kernels. The merged down writes the EXPANDED buffer for all dtypes →
    /// the single shared `moe_down_combine_k8_batched` runs (NOT Lloyd atomic
    /// self-combine). silu+rotate is weight-agnostic (unchanged).
    pub routed_indexable_mixed_per_expert: bool,
    /// Uniform E8-family routed experts admitted only on wave32-WMMA arches.
    pub routed_indexable_e8: bool,
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
        // gate_up on one of the codebook (Lloyd / GL) formats — needed both for
        // the per-projection mix below and for `needs_x_rot_local` (all four are
        // FwhtG256 and consume the pre-rotated activation).
        let routed_gate_up_gl = matches!(d.routed_gate_up, MQ2G256GL | MQ3G256GL);
        // Per-projection codebook mix (e.g. gate_up MQ2-GL + down MQ3-GL, the
        // 2-bit-gate/3-bit-down allocation; or any Lloyd×GL cross). Subsumes the
        // two uniform Lloyd arms above and the uniform GL cases; the OR below
        // makes the overlap harmless.
        //
        // SAFETY INVARIANT: every dtype admitted here MUST have (a) an indexed
        // gate_up GEMV arm in `run_moe_decode`, (b) an ATOMIC SELF-COMBINING
        // down GEMV arm there, and (c) membership in the
        // `routed_down_self_combines` set in pipeline/mod.rs. Admitting a dtype
        // that misses (c) double-counts every MoE layer, silently.
        const CODEBOOK_INDEXABLE: [DType; 4] = [MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL];
        let routed_indexable_mixed_lloyd = CODEBOOK_INDEXABLE.contains(&d.routed_gate_up)
            && CODEBOOK_INDEXABLE.contains(&d.routed_down);
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;
        // Per-expert mixed: the model already verified the experts carry
        // different down dtypes and built the tag table (single source of
        // truth). gate_up stays uniform MQ4, so it pairs with the MQ4 indexed
        // gate_up GEMV; the merged dtype-tag kernel serves the down step.
        let routed_indexable_mixed_per_expert = d.routed_has_mixed_experts;
        // mfp4/mfp3/mfp2-E8 grouped experts (RDNA3 wave32-WMMA): uniform E8-family
        // gate_up + down → the gemv_mfp4g32_e8_moe_{gate_up,down}_k8_indexed kernels
        // (for uniform E8 models). FWHT-rotated (FwhtG256), same as MQ4, so the
        // shared silu+mul+rotate plumbing applies. Graded mixed-E8 uses the tag-table
        // path (routed_indexable_mixed_per_expert) rather than this uniform arm.
        let routed_gate_up_e8 = matches!(d.routed_gate_up, MFP4G32E8 | MFP3G32E8 | MFP2G32E8);
        let routed_indexable_e8 = arch_has_e8_wmma
            && routed_gate_up_e8
            && matches!(d.routed_down, MFP4G32E8 | MFP3G32E8 | MFP2G32E8);

        let routed_dtype_indexable = routed_indexable_mq4
            || routed_indexable_mq5
            || routed_indexable_mq6
            || routed_indexable_mixed_gu4_dn6
            || routed_indexable_mixed_per_expert
            || routed_indexable_mq2lloyd
            || routed_indexable_mq3lloyd
            || routed_indexable_mixed_lloyd
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
            // MQ2/MQ3-G256-GL are FWHT-G256 formats: their gate_up kernel reads
            // `x_rot`, so the local rotation MUST be produced. Missing this is a
            // silent garbage-output failure (unrotated x into a rotated weight).
            || routed_gate_up_gl
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
            routed_indexable_mixed_lloyd,
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
            || self.routed_indexable_mixed_lloyd
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
    /// Single-GPU lowered-decode experiment: leave the atomic-free routed
    /// output expanded so the architecture layer can combine it into the
    /// residual while producing the next layer's normalized activation.
    pub defer_routed_combine: bool,
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

/// Exact-device MQ2-Lloyd operations used by the DeepSeek4 bias-aware decode
/// executor.
///
/// The dispatch crate deliberately has no architecture detection here. A
/// model crate may provide this capability only after its loader has admitted
/// a model-owned backend. The implementation must still fail closed when the
/// supplied [`Gpu`] is not the device proven by that backend.
pub trait MoeBiasAwareMq2Backend {
    #[allow(clippy::too_many_arguments)]
    fn gate_up(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String>;

    fn rotate_x_batched(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> Result<(), String>;

    #[allow(clippy::too_many_arguments)]
    fn down_expanded(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> Result<(), String>;

    #[allow(clippy::too_many_arguments)]
    fn down_residual_scaled(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String>;
}

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
    /// Model-local dispatch policy. The DS4 loader derives this from the
    /// verified MQ2R backend; generic GPU state must not influence it.
    pub uses_atomic_moe_down: bool,
    /// Optional exact-device MQ2 backend selected by the loaded DeepSeek4
    /// model. `None` retains the portable dispatcher for every other model and
    /// architecture.
    pub native_mq2_backend: Option<&'a dyn MoeBiasAwareMq2Backend>,
    /// EP-shard-only zero weight buffer. Exact-device backends may compare
    /// selected gate/up pointers against it to skip non-owned expert work
    /// while retaining the fixed graph shape. `None` on unsharded models.
    pub nonowned_gate_up_dummy: Option<&'a GpuTensor>,
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

impl<'a> MoeBiasAwareParams<'a> {
    /// Borrow the routed-expert portion after route selection has already
    /// populated `topk_indices` and `topk_weights`. Heterogeneous DS4 uses
    /// this boundary to select routes on the dense owner and execute only the
    /// selected experts on the routed owner.
    pub fn selected(&self) -> MoeSelectedParams<'_> {
        MoeSelectedParams {
            hidden: self.hidden,
            mi: self.mi,
            k_top: self.k_top,
            swiglu_limit: self.swiglu_limit,
            uses_atomic_moe_down: self.uses_atomic_moe_down,
            native_mq2_backend: self.native_mq2_backend,
            nonowned_gate_up_dummy: self.nonowned_gate_up_dummy,
            batch_size: self.batch_size,
            x_rot: self.x_rot,
            ffn_out: self.ffn_out,
            expert_gate_up_ptrs: self.expert_gate_up_ptrs,
            expert_down_ptrs: self.expert_down_ptrs,
            topk_indices: self.topk_indices,
            topk_weights: self.topk_weights,
            gate_batch: self.gate_batch,
            up_batch: self.up_batch,
            rot_batch: self.rot_batch,
            down_expanded: self.down_expanded,
        }
    }
}

/// Selected routed-expert decode subgraph. Route selection is intentionally
/// absent: callers must provide the exact normalized IDs and weights produced
/// by the model-owned router. This is useful for split ownership where the
/// router and expert weights cannot reside on the same device.
pub struct MoeSelectedParams<'a> {
    pub hidden: usize,
    pub mi: usize,
    pub k_top: usize,
    pub swiglu_limit: f32,
    pub uses_atomic_moe_down: bool,
    pub native_mq2_backend: Option<&'a dyn MoeBiasAwareMq2Backend>,
    pub nonowned_gate_up_dummy: Option<&'a GpuTensor>,
    pub batch_size: usize,
    pub x_rot: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
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
    /// Model-local dispatch policy. The DS4 loader derives this from the
    /// verified MQ2R backend; generic GPU state must not influence it.
    pub uses_atomic_moe_down: bool,
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
        // Routed-down AWQ suppresses Path 2 (grouped-GEMM): the AWQ divide
        // must interleave per-expert silu+rotate, which the grouped hot-path
        // does not support.  Path 0/1 (indexed GEMV paths) remain eligible.
        let use_path2 = use_path2 && !d.routed_down_has_awq;
        // Path 0: gfx9* wave64 archs (gfx906/gfx908/gfx94x) — cheap HBM
        // atomics make the atomic GEMV pattern competitive vs expanded scratch.
        let down_path0 = arch.is_gcn5() || arch.is_cdna1() || arch.is_cdna3();
        let is_gfx1151 = arch.is_gfx1151();
        let use_paro_i8 = paro_mode && use_path2 && is_gfx1151 && flags.moe_paro_i8.unwrap_or(true);
        let use_paro_i8_k8 = use_paro_i8 && flags.moe_paro_i8_k8.unwrap_or(true);
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
    /// per layer per token (an uncached generic policy parse) would
    /// be pure waste on the decode hot path.
    pub fn run_bias_aware(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeBiasAwareParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode_bias_aware(gpu, params)
    }

    /// Run only the selected-expert portion of the single-token DeepSeek4
    /// MQ2-Lloyd subgraph. The caller owns route selection and must already
    /// have populated `topk_indices` and `topk_weights`.
    pub fn run_selected(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeSelectedParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode_selected(gpu, params)
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

// ────────────────────────────────────────────────────────────────────
// Step-IR launch helpers (STEP-002..004 device-mesh feature).
// Ported additively from backup/device-mesh-pre-main-merge-20260826 so the
// Step executor (pipeline/steps.rs) can compile against mainline dispatch.
// ────────────────────────────────────────────────────────────────────

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

/// Complete borrowed storage view for Gemma's GELU-routed experts.
///
/// The pointer tables serve the indexed fast path.  The raw pools and their
/// byte strides are retained so the same Step can fall back to non-owning
/// per-expert views when a dtype pair has no indexed kernel.
pub struct MoeGeluExpertsRef<'a> {
    pub gate_up_pool: &'a GpuTensor,
    pub down_pool: &'a GpuTensor,
    pub gate_up_ptrs: &'a GpuTensor,
    pub down_ptrs: &'a GpuTensor,
    pub gate_up_dtype: DType,
    pub down_dtype: DType,
    pub gate_up_bytes: usize,
    pub down_bytes: usize,
    pub n_experts: usize,
}

/// Dispatch backend for the typed Gemma GELU expert Step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeGeluBackend {
    Indexed,
    PerExpert,
}

/// Select the indexed backend only for the Gemma expert pairs backed by real
/// indexed kernels.  HFQ4G128 has an API-shaped launcher, but that launcher is
/// still a Phase-4 stub; routing it here would turn a valid model into a
/// known dispatch error, so it intentionally stays on the generic fallback.
pub fn select_moe_gelu_backend(gate_up_dtype: DType, down_dtype: DType) -> MoeGeluBackend {
    match (gate_up_dtype, down_dtype) {
        (DType::MQ4G256, DType::Q8_0) | (DType::Q8_0, DType::Q8_0) => {
            MoeGeluBackend::Indexed
        }
        _ => MoeGeluBackend::PerExpert,
    }
}

/// Validate the shape-independent invariants required by the typed GELU
/// expert step before any GPU work or top-K download.
pub fn validate_moe_gelu_shape(
    k_top: usize,
    hidden_dim: usize,
    expert_dim: usize,
    n_experts: usize,
) -> Result<(), String> {
    if k_top == 0 {
        return Err("MoeGeluExperts: k_top must be nonzero".to_owned());
    }
    if hidden_dim == 0 {
        return Err("MoeGeluExperts: hidden_dim must be nonzero".to_owned());
    }
    if expert_dim == 0 {
        return Err("MoeGeluExperts: expert_dim must be nonzero".to_owned());
    }
    if n_experts == 0 {
        return Err("MoeGeluExperts: n_experts must be nonzero".to_owned());
    }
    if k_top > n_experts {
        return Err(format!(
            "MoeGeluExperts: k_top={k_top} exceeds n_experts={n_experts}"
        ));
    }
    k_top
        .checked_mul(expert_dim)
        .ok_or_else(|| "MoeGeluExperts: k_top * expert_dim overflows".to_owned())?;
    hidden_dim
        .checked_mul(4)
        .ok_or_else(|| "MoeGeluExperts: hidden_dim byte size overflows".to_owned())?;
    Ok(())
}

/// Return a non-owning raw-byte view for one expert in a pooled allocation.
/// The explicit bounds check keeps malformed metadata from reaching
/// `GpuTensor::sub_offset`, whose assertion is intentionally infallible.
fn moe_pool_expert_view(
    pool: &GpuTensor,
    bytes_per_expert: usize,
    expert: usize,
    n_experts: usize,
    kind: &str,
) -> Result<GpuTensor, DispatchError> {
    if bytes_per_expert == 0 {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} bytes_per_expert must be nonzero"
        )));
    }
    if expert >= n_experts {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} expert index {expert} out of range (n_experts={n_experts})"
        )));
    }
    if pool.dtype != DType::Raw {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} pool must be Raw, got {:?}",
            pool.dtype
        )));
    }
    let offset = expert.checked_mul(bytes_per_expert).ok_or_else(|| {
        DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} expert byte offset overflows"
        ))
    })?;
    let end = offset.checked_add(bytes_per_expert).ok_or_else(|| {
        DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} expert byte range overflows"
        ))
    })?;
    if end > pool.buf.size() {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} expert view [{offset}, {end}) exceeds pool bytes {}",
            pool.buf.size()
        )));
    }
    Ok(pool.sub_offset(offset, bytes_per_expert))
}

fn f32_prefix_view(
    tensor: &GpuTensor,
    elements: usize,
    kind: &str,
) -> Result<GpuTensor, DispatchError> {
    if tensor.dtype != DType::F32 {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} scratch must be F32, got {:?}",
            tensor.dtype
        )));
    }
    let bytes = elements.checked_mul(4).ok_or_else(|| {
        DispatchError::Hip(format!("MoeGeluExperts: {kind} byte size overflows"))
    })?;
    if bytes > tensor.buf.size() {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: {kind} scratch needs {bytes} bytes, has {}",
            tensor.buf.size()
        )));
    }
    Ok(tensor.sub_offset(0, elements))
}

fn zero_moe_output(
    gpu: &mut Gpu,
    out: &GpuTensor,
    hidden_dim: usize,
) -> Result<(), DispatchError> {
    let bytes = hidden_dim.checked_mul(4).ok_or_else(|| {
        DispatchError::Hip("MoeGeluExperts: output byte size overflows".to_owned())
    })?;
    if out.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: output needs {bytes} bytes, has {}",
            out.buf.size()
        )));
    }
    if let Some(stream) = gpu.active_stream.as_ref() {
        gpu.hip
            .memset_async(&out.buf, 0, bytes, stream)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    } else {
        gpu.hip
            .memset(&out.buf, 0, bytes)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }
}

fn launch_moe_weight(
    gemv: &GemvFamily,
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    input: &GpuTensor,
    output: &GpuTensor,
    prerotated: bool,
) -> Result<(), DispatchError> {
    if prerotated {
        return gemv.run(
            ctx,
            gpu,
            &GemvParams {
                w: weight,
                x: input,
                y: output,
                variant: crate::types::GemvVariant::Prerotated,
                residual: None,
                gate: None,
                up: None,
            },
        );
    }
    if crate::types::dtype_rotation_plan(weight.dtype) == crate::types::RotationPlan::None {
        gemv.run(
            ctx,
            gpu,
            &GemvParams {
                w: weight,
                x: input,
                y: output,
                variant: crate::types::GemvVariant::Plain,
                residual: None,
                gate: None,
                up: None,
            },
        )
    } else {
        gemv.run_auto(ctx, gpu, weight, input, output)
    }
}

/// Execute Gemma's selected GELU-tanh experts through one typed Step
/// contract.  Indexed pairs stay entirely on device; all other pairs use the
/// legacy top-K download and non-owning pool-view fallback.
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_gelu_experts(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    experts: &MoeGeluExpertsRef<'_>,
    input: &GpuTensor,
    input_rot: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    expert_scales: &GpuTensor,
    expert_scales_host: &[f32],
    gate: &GpuTensor,
    up: &GpuTensor,
    hidden: &GpuTensor,
    out: &GpuTensor,
    hidden_dim: usize,
    expert_dim: usize,
    k_top: usize,
) -> Result<(), DispatchError> {
    validate_moe_gelu_shape(k_top, hidden_dim, expert_dim, experts.n_experts)
        .map_err(DispatchError::Hip)?;
    if expert_scales_host.len() < experts.n_experts {
        return Err(DispatchError::Hip(format!(
            "MoeGeluExperts: expert_scales_host has {} entries, needs {}",
            expert_scales_host.len(),
            experts.n_experts
        )));
    }
    let backend = if k_top == 8 {
        select_moe_gelu_backend(experts.gate_up_dtype, experts.down_dtype)
    } else {
        MoeGeluBackend::PerExpert
    };
    match backend {
        MoeGeluBackend::Indexed => {
            zero_moe_output(gpu, out, hidden_dim)?;
            match experts.gate_up_dtype {
                DType::MQ4G256 => {
                    gpu.rotate_x_mq(input, input_rot, hidden_dim)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    gpu.gemv_mq4g256_moe_gate_up_k8_indexed(
                        experts.gate_up_ptrs,
                        topk_indices,
                        input_rot,
                        gate,
                        up,
                        2 * expert_dim,
                        hidden_dim,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
                DType::Q8_0 => gpu
                    .gemv_q8_0_moe_gate_up_k8_indexed(
                        experts.gate_up_ptrs,
                        topk_indices,
                        input,
                        gate,
                        up,
                        2 * expert_dim,
                        hidden_dim,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))?,
                other => {
                    return Err(DispatchError::Hip(format!(
                        "MoeGeluExperts: indexed gate_up dtype {other:?} is unsupported"
                    )))
                }
            }
            let activation_n = k_top * expert_dim;
            gpu.gelu_tanh_f32(gate, hidden, activation_n)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.mul_f32(hidden, up, hidden)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            match experts.down_dtype {
                DType::Q8_0 => gpu
                    .gemv_q8_0_moe_down_residual_scaled_k8_indexed(
                        experts.down_ptrs,
                        topk_indices,
                        topk_weights,
                        expert_scales,
                        hidden,
                        out,
                        hidden_dim,
                        expert_dim,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string())),
                other => Err(DispatchError::Hip(format!(
                    "MoeGeluExperts: indexed down dtype {other:?} is unsupported"
                ))),
            }
        }
        MoeGeluBackend::PerExpert => {
            let index_data = gpu
                .download_f32(topk_indices)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let weight_data = gpu
                .download_f32(topk_weights)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            if index_data.len() < k_top {
                return Err(DispatchError::Hip(format!(
                    "MoeGeluExperts: topk_indices has {} entries, needs {k_top}",
                    index_data.len()
                )));
            }
            if weight_data.len() < k_top {
                return Err(DispatchError::Hip(format!(
                    "MoeGeluExperts: topk_weights has {} entries, needs {k_top}",
                    weight_data.len()
                )));
            }
            let index_words =
                unsafe { std::slice::from_raw_parts(index_data.as_ptr() as *const i32, k_top) };
            for &raw in index_words {
                if raw < 0 || (raw as usize) >= experts.n_experts {
                    return Err(DispatchError::Hip(format!(
                        "MoeGeluExperts: topk index {raw} out of range (n_experts={})",
                        experts.n_experts
                    )));
                }
            }
            if experts.gate_up_dtype == DType::MQ4G256 {
                gpu.rotate_x_mq(input, input_rot, hidden_dim)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }

            zero_moe_output(gpu, out, hidden_dim)?;
            // The indexed scratch buffers are large enough for one fused
            // gate||up result and one hidden-dimensional down result.  Reuse
            // them rather than creating a GPU owner for the fallback.
            let gate_up_workspace = f32_prefix_view(gate, 2 * expert_dim, "gate")?;
            let gate_view = gate_up_workspace.sub_offset(0, expert_dim);
            let up_view = gate_up_workspace.sub_offset(expert_dim, expert_dim);
            let hidden_view = f32_prefix_view(hidden, expert_dim, "hidden")?;
            let down_view = f32_prefix_view(up, hidden_dim, "up/down")?;
            static GEMV: LazyLock<GemvFamily> = LazyLock::new(GemvFamily::new);
            let gemv: &GemvFamily = &GEMV;

            for slot in 0..k_top {
                let expert = index_words[slot] as usize;
                let gate_up_buf = moe_pool_expert_view(
                    experts.gate_up_pool,
                    experts.gate_up_bytes,
                    expert,
                    experts.n_experts,
                    "gate_up",
                )?;
                let down_buf = moe_pool_expert_view(
                    experts.down_pool,
                    experts.down_bytes,
                    expert,
                    experts.n_experts,
                    "down",
                )?;
                let gate_up_weight = WeightRef {
                    buf: &gate_up_buf,
                    dtype: experts.gate_up_dtype,
                    m: 2 * expert_dim,
                    k: hidden_dim,
                    row_stride: 0,
                    rotation: None,
                    awq_scale: None,
                };
                let down_weight = WeightRef {
                    buf: &down_buf,
                    dtype: experts.down_dtype,
                    m: hidden_dim,
                    k: expert_dim,
                    row_stride: 0,
                    rotation: None,
                    awq_scale: None,
                };
                launch_moe_weight(
                    gemv,
                    ctx,
                    gpu,
                    &gate_up_weight,
                    if experts.gate_up_dtype == DType::MQ4G256 {
                        input_rot
                    } else {
                        input
                    },
                    &gate_up_workspace,
                    experts.gate_up_dtype == DType::MQ4G256,
                )?;
                gpu.gelu_tanh_f32(&gate_view, &hidden_view, expert_dim)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.mul_f32(&hidden_view, &up_view, &hidden_view)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                launch_moe_weight(
                    gemv,
                    ctx,
                    gpu,
                    &down_weight,
                    &hidden_view,
                    &down_view,
                    false,
                )?;
                let scale = weight_data[slot] * expert_scales_host[expert];
                gpu.scaled_add_inplace_cpu_scalar_f32(out, &down_view, scale)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            Ok(())
        }
    }
}

// ── Step-IR launch helpers ────────────────────────────────

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

/// Fused softmax + renormalized top-K routing. Unlike
/// [`launch_moe_softmax_topk`], this preserves the hand-route's single
/// `moe_softmax_topk_renorm_k8` launch and leaves the input logits untouched.
pub fn launch_moe_softmax_topk_fused(
    gpu: &mut rdna_compute::Gpu,
    logits: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    n_exp: usize,
    norm_topk_prob: bool,
) -> Result<(), DispatchError> {
    gpu.moe_softmax_topk_renorm_k8(
        logits,
        topk_indices,
        topk_weights,
        n_exp,
        norm_topk_prob,
    )
    .map_err(|e| DispatchError::Hip(e.to_string()))
}

/// Architecture-selected Qwen decode router backend. The generic
/// two-launch route (`softmax_f32` + `moe_topk_renorm_k8`) is numerically
/// distinct from the wave64 fused routers (different reduction order and
/// direct division), so the sealed Step path must select the same backend
/// the direct `run_moe_decode` executor picks — never silently substitute
/// the generic form. `Default` remains the byte-identical generic route for
/// callers that need the two-launch semantics; `FusedSoftmaxTopK` is an
/// explicit opt-in for callers whose reference uses the fused router kernel.
///
/// The optional fused router/shared optimization
/// (`HIPFIRE_MOE_ROUTER_SHARED_FUSE`) is intentionally NOT exposed on the
/// sealed Step path: the fused kernel cannot share the `Step::MoeSoftmaxTopK`
/// signature (it also consumes the shared gate/up + rotation operands). The
/// `FusedSoftmaxTopK` backend is the standalone router kernel and does not
/// enable that shared optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRouterBackend {
    /// `moe_softmax_topk_renorm_k8` (one fused launch, leaves logits raw).
    FusedSoftmaxTopK,
    /// `softmax_f32(logits)` + `moe_topk_renorm_k8` (two launches).
    Default,
    /// `moe_router_softmax_topk_k8_wave64` (gfx1201 default, gfx1100
    /// `HIPFIRE_GFX1100_ROUTER_W64=approx` research mode).
    Wave64,
    /// `moe_router_softmax_topk_k8_wave64_exact` (n_exp==256 on gfx1100/
    /// gfx1151, unless the gfx1100 env opt-out says otherwise).
    ExactWave64,
}

/// Pure arch/n_exp/env decision for [`MoeRouterBackend`]. MUST stay in
/// lockstep with the inline block in `run_moe_decode`
/// (crates/hipfire-dispatch/src/pipeline/mod.rs): the sealed Step path and
/// the architecture-side fused leaf decide through this classifier, the
/// direct executor keeps its own copy — a divergence is a routing-numerics
/// regression. `gfx1100_mode` is the `HIPFIRE_GFX1100_ROUTER_W64` value
/// (`None` = unset, `"0"`/`"approx"` disable the exact backend);
/// `gfx1201_w64` is `HIPFIRE_GFX1201_ROUTER_W64 != "0"` (default true).
fn select_moe_router_backend_modes(
    arch: &rdna_compute::arch_caps::ArchCaps,
    n_exp: usize,
    gfx1100_mode: Option<&str>,
    gfx1201_w64: bool,
) -> MoeRouterBackend {
    let exact_wave64_router = n_exp == 256
        && ((arch.is_gfx1100() && !matches!(gfx1100_mode, Some("0" | "approx")))
            || arch.is_gfx1151());
    if exact_wave64_router {
        return MoeRouterBackend::ExactWave64;
    }
    let wave64_router = (arch.is_gfx1201() && gfx1201_w64)
        || (arch.is_gfx1100() && n_exp == 256 && gfx1100_mode == Some("approx"));
    if wave64_router {
        return MoeRouterBackend::Wave64;
    }
    MoeRouterBackend::Default
}

/// Architecture-selected Qwen decode router backend for the current
/// [`DispatchCtx`], with the same gfx/n_exp/env rules as the direct
/// `run_moe_decode` executor. Reads the live environment first so tests can
/// mutate it deterministically, then falls back to the hipfire-config
/// process snapshot (which honors TOML-set developer values); for env-set
/// values the two sources agree, which is the supported spelling for these
/// developer vars.
pub fn select_moe_router_backend(ctx: &DispatchCtx, n_exp: usize) -> MoeRouterBackend {
    let gfx1100_mode = std::env::var("HIPFIRE_GFX1100_ROUTER_W64")
        .ok()
        .or_else(|| hipfire_config::developer_var("HIPFIRE_GFX1100_ROUTER_W64").ok());
    let gfx1201_w64 = std::env::var("HIPFIRE_GFX1201_ROUTER_W64")
        .map(|value| value != "0")
        .unwrap_or_else(|_| {
            hipfire_config::developer_var("HIPFIRE_GFX1201_ROUTER_W64")
                .map(|value| value != "0")
                .unwrap_or(true)
        });
    select_moe_router_backend_modes(&ctx.arch, n_exp, gfx1100_mode.as_deref(), gfx1201_w64)
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
        // MQ2/MQ3-G256-GL: 2-/8-entry TENSOR-GLOBAL codebook (scalar kernel
        // args) + per-block fp16 scale, SoA — the same call shape as the
        // MQ2/MQ3-Lloyd arm (y_gate/y_up separate, m = 2·expert_m). Mirrors
        // the direct `run_moe_decode` GL arms exactly; the sealed decode
        // route must not return UnsupportedVariant for these dtypes.
        DType::MQ2G256GL => gpu
            .gemv_mq2g256gl_moe_gate_up_indexed(
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
        DType::MQ3G256GL => gpu
            .gemv_mq3g256gl_moe_gate_up_indexed(
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
                false, // deepseek4_gfx1151_route: generic dispatch crate path (as mainline pipeline/mod.rs)
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
        // MQ2/MQ3-G256-GL: atomic, weighted, SELF-COMBINING residual GEMV —
        // the same epilogue contract as the Lloyd down kernels (one launch
        // does down → × topk_weight[krank] → atomicAdd into ffn_out; NO
        // separate combine). Mirrors the direct `run_moe_decode` GL arms;
        // the sealed decode route must not return UnsupportedVariant.
        DType::MQ2G256GL => gpu
            .gemv_mq2g256gl_moe_down_residual_scaled_indexed(
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
        DType::MQ3G256GL => gpu
            .gemv_mq3g256gl_moe_down_residual_scaled_indexed(
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

    // ── Router backend selection (sealed-path parity with run_moe_decode) ──

    #[test]
    fn router_backend_pure_matrix_matches_direct_rules() {
        // gfx1100: n_exp==256 → exact unless the env opt-out says "0"/"approx".
        let g1100 = DispatchCtx::for_arch("gfx1100");
        assert_eq!(
            select_moe_router_backend_modes(&g1100.arch, 256, None, true),
            MoeRouterBackend::ExactWave64
        );
        assert_eq!(
            select_moe_router_backend_modes(&g1100.arch, 256, Some("0"), true),
            MoeRouterBackend::Default
        );
        assert_eq!(
            select_moe_router_backend_modes(&g1100.arch, 256, Some("approx"), true),
            MoeRouterBackend::Wave64
        );
        assert_eq!(
            select_moe_router_backend_modes(&g1100.arch, 128, None, true),
            MoeRouterBackend::Default,
            "n_exp != 256 keeps the generic route on gfx1100"
        );
        // gfx1151: n_exp==256 → exact unconditionally (the env is a gfx1100
        // lever; radiowave fusion is always on for 256 experts).
        let g1151 = DispatchCtx::for_arch("gfx1151");
        assert_eq!(
            select_moe_router_backend_modes(&g1151.arch, 256, Some("0"), true),
            MoeRouterBackend::ExactWave64
        );
        assert_eq!(
            select_moe_router_backend_modes(&g1151.arch, 128, None, true),
            MoeRouterBackend::Default
        );
        // gfx1201: wave64 by default at ANY n_exp; HIPFIRE_GFX1201_ROUTER_W64=0
        // restores the generic route.
        let g1201 = DispatchCtx::for_arch("gfx1201");
        assert_eq!(
            select_moe_router_backend_modes(&g1201.arch, 256, None, true),
            MoeRouterBackend::Wave64
        );
        assert_eq!(
            select_moe_router_backend_modes(&g1201.arch, 128, None, true),
            MoeRouterBackend::Wave64
        );
        assert_eq!(
            select_moe_router_backend_modes(&g1201.arch, 256, None, false),
            MoeRouterBackend::Default
        );
        // Everything else stays on the generic two-launch route.
        for other in ["gfx942", "gfx1010", "gfx906"] {
            let ctx = DispatchCtx::for_arch(other);
            assert_eq!(
                select_moe_router_backend_modes(&ctx.arch, 256, None, true),
                MoeRouterBackend::Default,
                "{other} must keep the generic router"
            );
        }
    }

    /// Serializes env-mutating router tests (the classifier reads the live
    /// environment first, like the grammar-config resolvers, so the
    /// hipfire-config snapshot is never consulted while a case is mutated).
    fn router_env_lock() -> std::sync::MutexGuard<'static, ()> {
        static GLOBAL: std::sync::LazyLock<std::sync::Mutex<()>> =
            std::sync::LazyLock::new(|| std::sync::Mutex::new(()));
        static INIT_SNAPSHOT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        let guard = GLOBAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Initialize the hipfire-config process snapshot with a clean env
        // BEFORE any test's set_var can pollute it (same pattern as
        // hipfire-arch-qwen35/src/grammar_config.rs). The classifier reads
        // the live env first and falls back to this frozen snapshot only when
        // the live var is absent, so a snapshot captured mid-mutation would
        // otherwise leak "0"/"approx" into later clean cases.
        INIT_SNAPSHOT.get_or_init(|| {
            std::env::remove_var("HIPFIRE_GFX1100_ROUTER_W64");
            std::env::remove_var("HIPFIRE_GFX1201_ROUTER_W64");
            let _ = hipfire_config::developer_var("HIPFIRE_GFX1100_ROUTER_W64");
            let _ = hipfire_config::developer_var("HIPFIRE_GFX1201_ROUTER_W64");
        });
        guard
    }

    #[test]
    fn router_backend_env_plumbing_matches_direct_rules() {
        let _guard = router_env_lock();
        std::env::set_var("HIPFIRE_GFX1100_ROUTER_W64", "0");
        std::env::remove_var("HIPFIRE_GFX1201_ROUTER_W64");
        let g1100 = DispatchCtx::for_arch("gfx1100");
        assert_eq!(
            select_moe_router_backend(&g1100, 256),
            MoeRouterBackend::Default
        );
        std::env::set_var("HIPFIRE_GFX1100_ROUTER_W64", "approx");
        assert_eq!(
            select_moe_router_backend(&g1100, 256),
            MoeRouterBackend::Wave64
        );
        std::env::remove_var("HIPFIRE_GFX1100_ROUTER_W64");
        assert_eq!(
            select_moe_router_backend(&g1100, 256),
            MoeRouterBackend::ExactWave64
        );
        std::env::set_var("HIPFIRE_GFX1201_ROUTER_W64", "0");
        let g1201 = DispatchCtx::for_arch("gfx1201");
        assert_eq!(
            select_moe_router_backend(&g1201, 256),
            MoeRouterBackend::Default
        );
        std::env::remove_var("HIPFIRE_GFX1201_ROUTER_W64");
        assert_eq!(
            select_moe_router_backend(&g1201, 256),
            MoeRouterBackend::Wave64
        );
    }

    #[test]
    fn gelu_expert_step_rejects_zero_topk_before_launch() {
        let err = validate_moe_gelu_shape(0, 2816, 704, 30).unwrap_err();
        assert_eq!(err.to_string(), "MoeGeluExperts: k_top must be nonzero");
    }

    #[test]
    fn gelu_expert_backend_preserves_unknown_dtype_fallback() {
        assert_eq!(
            select_moe_gelu_backend(DType::HFQ4G128, DType::MQ4G256),
            MoeGeluBackend::PerExpert
        );
    }

    #[test]
    fn gelu_expert_backend_selects_only_implemented_pairs() {
        assert_eq!(
            select_moe_gelu_backend(DType::MQ4G256, DType::Q8_0),
            MoeGeluBackend::Indexed
        );
        assert_eq!(
            select_moe_gelu_backend(DType::Q8_0, DType::Q8_0),
            MoeGeluBackend::Indexed
        );
        assert_eq!(
            select_moe_gelu_backend(DType::MQ4G256, DType::HFQ4G128),
            MoeGeluBackend::PerExpert
        );
    }

    #[test]
    fn gelu_expert_fallback_views_use_pool_byte_offsets() {
        let pool = GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    0x1000usize as *mut std::ffi::c_void,
                    3 * 24,
                )
            },
            shape: vec![3 * 24],
            dtype: DType::Raw,
        };
        let view = moe_pool_expert_view(&pool, 24, 2, 3, "gate_up").unwrap();
        assert_eq!(view.buf.as_ptr() as usize, 0x1000 + 2 * 24);
        assert_eq!(view.buf.size(), 24);
        let down_view = moe_pool_expert_view(&pool, 24, 1, 3, "down").unwrap();
        assert_eq!(down_view.buf.as_ptr() as usize, 0x1000 + 24);
        assert_eq!(down_view.buf.size(), 24);
        assert!(
            moe_pool_expert_view(&pool, 24, 3, 3, "gate_up").is_err(),
            "an expert at n_experts must not produce a view"
        );
    }
    #[test]
    fn gelu_expert_passthrough_dtypes_use_generic_fallback() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert_eq!(
                select_moe_gelu_backend(dtype, dtype),
                MoeGeluBackend::PerExpert,
                "{dtype:?} must use the generic per-expert GEMV fallback"
            );
        }
        assert!(validate_moe_gelu_shape(2, 4, 2, 3).is_ok());
    }
}
