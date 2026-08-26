// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 weight structs (dense / MoE layers), EP shard provenance and seals,
//! `Qwen35Weights`, and the persistent DeltaNet state (`DeltaNetState`).

use super::batch::PrefillBatchScratch;
use super::config::LayerType;
use super::config::Qwen35Config;
use super::forward::Qwen35Scratch;
use crate::store::MoeFfnStorage;
use crate::store::Qwen35MoeBindError;
use crate::store::Qwen35MoeLayerProjection;
use crate::store::Qwen35MoeResident;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::gpu_cleanup::{
    free_tensor_retained, free_weight_all_checked, free_weight_sidecars_checked, GpuCleanupFailure,
    RetainedGpuTensor, RetryableOwner,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::moe_plan::MoEExecutionPolicy;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::screen_weight_tensor;
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::weight_store::WeightCellId;
use hipfire_runtime::MmqScreenable;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;
// ─── Weight structs ─────────────────────────────────────────────────────

/// Weights for a DeltaNet (linear attention) layer.
pub struct DeltaNetLayerWeights {
    pub attn_norm: GpuTensor,   // input_layernorm [dim]
    pub wqkv: WeightTensor,     // in_proj_qkv [6144, dim] → Q+K+V concat
    pub wz: WeightTensor,       // in_proj_z [2048, dim] → gate Z
    pub w_alpha: WeightTensor,  // in_proj_a [n_heads, dim] → decay
    pub w_beta: WeightTensor,   // in_proj_b [n_heads, dim] → update
    pub a_log: GpuTensor,       // A_log [n_heads] — learnable log-decay
    pub dt_bias: GpuTensor,     // dt_bias [n_heads]
    pub conv_weight: GpuTensor, // conv1d.weight [conv_channels, 1, 4] → F32
    pub norm_weight: GpuTensor, // norm.weight [head_dim] — gated output norm
    pub wo: WeightTensor,       // out_proj [dim, d_inner]
    pub ffn_norm: GpuTensor,    // post_attention_layernorm [dim]
    pub w_gate: WeightTensor,   // mlp.gate_proj
    pub w_up: WeightTensor,     // mlp.up_proj
    pub w_down: WeightTensor,   // mlp.down_proj
}

/// Weights for a full attention (gated) layer — similar to Qwen3 but with q+gate split.
pub struct FullAttnLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,  // q_proj [4096, dim] — 2x wide (query + gate)
    pub wk: WeightTensor,  // k_proj
    pub wv: WeightTensor,  // v_proj
    pub wo: WeightTensor,  // o_proj
    pub q_norm: GpuTensor, // q_norm [head_dim]
    pub k_norm: GpuTensor, // k_norm [head_dim]
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

// ─── MoE FFN weights (Qwen3.5-MoE / A3B) ────────────────────────────────
//
// Replaces the dense (w_gate, w_up, w_down) triple with N+1 expert FFNs
// gated by a router, plus a shared always-on expert.
//
// A3B specifics:
//   num_experts = 256, top_k = 8, moe_intermediate = 512, hidden = 2048
//   shared_expert_intermediate = 512 (same as routed)
//
// Per-layer storage:
//   router:               [num_experts, hidden]  MQ4G256 / Q8
//   shared_expert_gate:   [1, hidden]            MQ4G256 / Q8 — projects to scalar
//   experts[X].gate_up:   [2*moe_intermediate, hidden]  MQ4G256
//   experts[X].down:      [hidden, moe_intermediate]    MQ4G256
//   shared_expert.gate:   [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.up:     [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.down:   [hidden, shared_expert_intermediate]   MQ4G256
//
// The quantizer (hipfire-quantize) splits the safetensors 3D
// `mlp.experts.gate_up_proj` / `down_proj` tensors per-expert into
// `mlp.experts.{X}.gate_up_proj.weight` / `down_proj.weight` so the loader
// can fish them out by index. The shared expert is stored with separate
// gate_proj + up_proj + down_proj (it is not fused in safetensors either).

pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2 * moe_intermediate, hidden] — fused (gate || up)
    pub down: WeightTensor,    // [hidden, moe_intermediate]
}

/// Owning storage for a layer's packed uniform-MQ4 routed experts.
///
/// `experts` still carries one [`WeightTensor`] view per routed expert so the
/// CPU fallback and every existing indexed dispatch keep their exact metadata
/// and pointer-table ABI. Those views are non-owning subranges of these two
/// buffers; only this owner pair may be returned to the GPU pool.
pub(crate) struct PackedExpertOwners {
    pub(crate) gate_up: GpuTensor,
    pub(crate) down: GpuTensor,
}

/// SP2: build the per-expert (gate_up, down) quant-tier tables that
/// [`hipfire_dispatch::families::moe::MoeDtypes`] uses to detect an
/// intra-layer mixed-tier layer.
///
/// A table is `Some(vec)` only when the layer genuinely spans >1 distinct
/// tier; a uniform layer — or paged mode where `experts` is empty — yields
/// `None`, which `MoeResolution::resolve` collapses to the unchanged uniform
/// fast path. We pre-filter to `None` for the uniform/empty cases here so the
/// common path allocates nothing and is byte-identical to before SP2.
pub(crate) fn per_expert_tier_tables(
    ffn: &MoeFfnWeights,
) -> (Option<Vec<DType>>, Option<Vec<DType>>) {
    if let Some(global) = ffn.global_expert_dtypes.as_ref() {
        let gu: Vec<DType> = global.iter().map(|(g, _)| *g).collect();
        let dn: Vec<DType> = global.iter().map(|(_, d)| *d).collect();
        (mixed_tier_table(gu), mixed_tier_table(dn))
    } else {
        let gu: Vec<DType> = ffn.experts.iter().map(|e| e.gate_up.gpu_dtype).collect();
        let dn: Vec<DType> = ffn.experts.iter().map(|e| e.down.gpu_dtype).collect();
        (mixed_tier_table(gu), mixed_tier_table(dn))
    }
}

/// Collapse a per-expert dtype column to `None` when it is empty or uniform,
/// `Some` only when it spans >1 distinct tier. Pure (no GPU weights) so it is
/// unit-testable in isolation; `per_expert_tier_tables` is the GPU-weight
/// adapter over it.
fn mixed_tier_table(tiers: Vec<DType>) -> Option<Vec<DType>> {
    match tiers.first() {
        // Empty (paged mode) or uniform → uniform fast path.
        None => None,
        Some(&first) if tiers.iter().all(|&d| d == first) => None,
        Some(_) => Some(tiers),
    }
}
/// Fallible per-expert tag mapping for the pinned graded MQ4R family.
/// Admits exactly 13 ordered (gate, down) pairs; every other ordered pair,
/// every GL dtype in either position, and every unknown dtype is `Err`.
/// Single source of truth consumed by both projections via one stored tag;
/// the uniform MQ4 gate optimisation is the only reason the mixed MQ4 set is valid.
pub fn mixed_expert_tag(gate_dtype: DType, down_dtype: DType) -> HipResult<u8> {
    // GL in either position is always rejected – the tag-branched decoder has
    // no GL branch and would silently mis-decode as MQ4.
    if matches!(gate_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
        || matches!(down_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
    {
        return Err(HipError::new(
            0,
            &format!("graded EP: GL dtype not supported (gate={gate_dtype:?} down={down_dtype:?})"),
        ));
    }
    match (gate_dtype, down_dtype) {
        // Mixed MQ4 gate family (7 pairs) — MQ4G256V2 is the same container family
        // as MQ4G256 for the grouped dispatch, so every gate pair that admits
        // MQ4 also admits V2 (including the cross V2↔MQ4 uniform gate which
        // keeps mixed-shard models loadable). The kernel selection is container-aware
        // via forward_slots::fused_*_key_for, so admitting here never misroutes.
        (DType::MQ4G256, DType::MQ6G256) => Ok(0),
        (DType::MQ4G256V2, DType::MQ6G256) => Ok(0),
        (DType::MQ4G256, DType::MQ2G256Lloyd) => Ok(1),
        (DType::MQ4G256V2, DType::MQ2G256Lloyd) => Ok(1),
        (DType::MQ4G256, DType::MQ4G256) => Ok(2),
        (DType::MQ4G256V2, DType::MQ4G256V2) => Ok(2),
        (DType::MQ4G256V2, DType::MQ4G256) => Ok(2),
        (DType::MQ4G256, DType::MQ4G256V2) => Ok(2),
        (DType::MQ4G256, DType::MQ3G256Lloyd) => Ok(3),
        (DType::MQ4G256V2, DType::MQ3G256Lloyd) => Ok(3),
        (DType::MQ4G256, DType::MFP4G32E8) => Ok(4),
        (DType::MQ4G256V2, DType::MFP4G32E8) => Ok(4),
        (DType::MQ4G256, DType::MFP3G32E8) => Ok(5),
        (DType::MQ4G256V2, DType::MFP3G32E8) => Ok(5),
        (DType::MQ4G256, DType::MFP2G32E8) => Ok(6),
        (DType::MQ4G256V2, DType::MFP2G32E8) => Ok(6),
        // Matching non-MQ4 pairs (6 pairs)
        (DType::MQ6G256, DType::MQ6G256) => Ok(0),
        (DType::MQ2G256Lloyd, DType::MQ2G256Lloyd) => Ok(1),
        (DType::MQ3G256Lloyd, DType::MQ3G256Lloyd) => Ok(3),
        (DType::MFP4G32E8, DType::MFP4G32E8) => Ok(4),
        (DType::MFP3G32E8, DType::MFP3G32E8) => Ok(5),
        (DType::MFP2G32E8, DType::MFP2G32E8) => Ok(6),
        _ => Err(HipError::new(
            0,
            &format!("graded EP: unsupported dtype pair gate={gate_dtype:?} down={down_dtype:?}"),
        )),
    }
}

pub(crate) fn dtype_from_quant_type(qt: u8) -> HipResult<DType> {
    match qt {
        13 => Ok(DType::MQ4G256),
        15 => Ok(DType::MQ6G256),
        19 => Ok(DType::MQ2G256Lloyd),
        20 => Ok(DType::MQ3G256Lloyd),
        30 => Ok(DType::MQ4G256Lloyd),
        34 => Ok(DType::MFP4G32E8),
        36 => Ok(DType::MFP3G32E8),
        38 => Ok(DType::MQ2G256GL),
        39 => Ok(DType::MQ3G256GL),
        40 => Ok(DType::TQ2G128),
        41 => Ok(DType::BQ1G128),
        44 => Ok(DType::MQ4G256V2),
        45 => Ok(DType::MQ4CG256),
        // Neutral-size Magnum V2 family (qt47-50): preserve qtype distinction
        // through WeightTensor/GpuTensor; do not map to legacy MQ2/3/5/6.
        47 => Ok(DType::MQ6G256V2),
        48 => Ok(DType::MQ5G256V2),
        49 => Ok(DType::MQ3G256V2),
        50 => Ok(DType::MQ2G256V2),
        // qt=6 (HFQ4G256) and qt=37 (MFP2G32E8) are shipped formats and MUST stay
        // mapped here. Dropping an arm from this match is not a compile error — it
        // degrades to "graded EP: unsupported quant_type", so the loss stays
        // invisible until a model of that format fails to load.
        6 => Ok(DType::HFQ4G256),
        37 => Ok(DType::MFP2G32E8),
        3 => Ok(DType::Q8_0),
        1 => Ok(DType::F16),
        2 => Ok(DType::F32),
        other => Err(HipError::new(
            0,
            &format!("graded EP: unsupported quant_type {other}"),
        )),
    }
}

/// Shared expert storage — unlike routed experts, gate_proj and up_proj are
/// NOT fused in the safetensors, so we keep them separate here too. The
/// forward path does two GEMVs + silu_mul + down GEMV.
pub struct SharedExpertWeights {
    pub gate: WeightTensor, // [shared_expert_intermediate, hidden]
    pub up: WeightTensor,   // [shared_expert_intermediate, hidden]
    pub down: WeightTensor, // [hidden, shared_expert_intermediate]
}

pub struct MoeFfnWeights {
    pub router: WeightTensor, // [num_experts, hidden]
    /// Routed expert weights. Populated when this layer is fully resident
    /// (`paged_experts == false`); **empty `Vec`** when `paged_experts == true`
    /// (the [`hipfire_runtime::weight_pager::WeightPager`] owns the buffers, and the
    /// indexed kernels read pointers from `expert_*_ptrs` which the pager
    /// patches per-token via `patch_expert_ptr_table`).
    pub experts: Vec<ExpertWeights>, // num_experts (= 256 for A3B); empty in paged mode
    /// Two allocation owners for the uniform MQ4 packed path. `None` preserves
    /// the literal per-expert ownership used by mixed quant, Paro, paged, and
    /// EP-streaming routes.
    pub(crate) packed_expert_owners: Option<PackedExpertOwners>,
    pub shared_expert: SharedExpertWeights,
    pub shared_expert_gate: WeightTensor, // [1, hidden] — row-vector projecting to scalar
    /// Device-side array of `unsigned long long` pointers, one per
    /// expert's `gate_up.buf`. Indexed at runtime by the GPU top-K
    /// kernel's output so the indexed MoE GEMV can stay capture-safe.
    pub expert_gate_up_ptrs: GpuTensor, // [num_experts * 2] f32 slots = num_experts × u64
    pub expert_down_ptrs: GpuTensor,      // [num_experts * 2] f32 slots = num_experts × u64

    /// Route A MoE-AWQ: per-expert down `awq_scale` pointer table
    /// (`[num_experts * 2]` f32 = num_experts × u64). `Some` only when the
    /// `.hfq` carries per-expert `down_proj.awq_scale` sidecars (all-or-none).
    /// Holds *non-owning* device pointers into each `experts[i].down.awq_scale`
    /// — freed as a buffer only; the scales are freed via
    /// `ExpertWeights::down.free_all`.
    pub expert_down_awq_ptrs: Option<GpuTensor>,

    /// Per-expert mixed-precision decode: `[num_experts]` u8 (DType::Raw,
    /// 1 B/expert) dtype-tag table. `Some` only when the layer's routed
    /// experts carry MIXED down dtypes (graded MQ6 hot / MQ2-Lloyd cold);
    /// the merged dtype-tag-branched down kernel reads `tags[expert_id]`
    /// per block (0=MQ6, 1=MQ2-Lloyd). `None` ⇒ uniform path, byte-identical.
    /// Owned device buffer (no aliasing) — freed as a buffer in free_moe_ffn.
    pub expert_dtype_tags: Option<GpuTensor>,

    /// Layer index. Stable identity used to key
    /// [`hipfire_runtime::weight_pager::WeightId::Expert`] entries.
    pub layer_idx: u16,

    /// Per-expert tensor shapes. `None` in non-paged mode (shapes are read
    /// from `experts[i].gate_up.{m, k}` etc.); `Some` in paged mode where
    /// `experts` is empty but kernels still need m/k for kernel-arg setup.
    /// Qwen3.5-MoE-A3B has uniform per-expert shape so one descriptor per
    /// layer suffices for v0.1.
    pub expert_shape: Option<hipfire_runtime::weight_pager::ExpertShape>,

    /// ParoQuant only: shared per-layer rotation sidecars for the routed
    /// experts. shisa-ai's PARO checkpoint quantizes all 256 experts with
    /// one rotation tuple per projection-group (gate||up vs down), so we
    /// upload the sidecars ONCE per layer and broadcast a non-owning
    /// `ParoRotation` (built via `DeviceBuffer::from_raw`) into every
    /// `ExpertWeights.gate_up.paro` / `ExpertWeights.down.paro`. The
    /// owning storage lives here so the aliases stay valid for the
    /// lifetime of the layer. `None` for HFQ MoE (per-tensor PARO sidecars
    /// or no PARO at all).
    pub paro_shared: Option<MoeParoSidecars>,

    /// EP global (gate_up_dtype, down_dtype) table — CPU-side immutable
    /// snapshot of the *full-model* expert dtypes (`len == num_experts`).
    /// `Some` only on the EP `load_weights_ep_rank` path; `None` preserves
    /// byte-identical single-GPU behavior. When present, every graded-mix
    /// decision (uniform/mixed flags, representative dtypes, tier tables,
    /// dummy layout sizes, device tag upload) is derived from this global
    /// table, never from the compact local `experts` slice.
    pub(crate) global_expert_dtypes: Option<Box<[(DType, DType)]>>,

    /// EP streaming dummies: one owned zero buffer per distinct
    /// non-owned storage layout. Non-owned global slots alias into the
    /// matching entry. Owned so `free_moe_ffn` can reclaim them.
    pub(crate) ep_dummy_buffers: Vec<GpuTensor>,
}

/// Owning storage for the per-layer shared ParoQuant rotation sidecars.
/// One tuple per projection-group:
///   - `gate_up_*`: applied to the post-RMSNorm hidden activation (K = hidden_dim).
///     Shared by all 256 experts' gate AND up projections, and by the fused
///     gate_up `WeightTensor`'s `paro` alias.
///   - `down_*`: applied to the post-SiLU intermediate activation (K = mi).
///     Shared by all 256 experts' down projection.
pub struct MoeParoSidecars {
    pub gate_up_pairs: GpuTensor,
    pub gate_up_theta: GpuTensor,
    pub gate_up_channel_scales: GpuTensor,
    pub down_pairs: GpuTensor,
    pub down_theta: GpuTensor,
    pub down_channel_scales: GpuTensor,
    pub krot: u32,
    pub group_size: u32,
}

pub struct DeltaNetMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wqkv: WeightTensor,
    pub wz: WeightTensor,
    pub w_alpha: WeightTensor,
    pub w_beta: WeightTensor,
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    pub conv_weight: GpuTensor,
    pub norm_weight: GpuTensor,
    pub wo: WeightTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnStorage,
}

pub struct FullAttnMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnStorage,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
    // A3B / qwen3_5_moe: same attention as above, MoE FFN instead of dense.
    // Loader + forward path TODO — adding the variants now so the enum is
    // forward-compatible and downstream code that pattern-matches gets a
    // compile-time hint to handle the new case.
    DeltaNetMoe(DeltaNetMoeLayerWeights),
    FullAttnMoe(FullAttnMoeLayerWeights),
}

pub(crate) enum MoeFfnView<'a> {
    Legacy(&'a MoeFfnWeights),
    Frozen(crate::store::MoeFfnBindings<'a>),
}

/// Private MoE execution selector (STEP-002 Task 8, Phase 2B).
///
/// `Single` is the production selection — every production forward wrapper
/// passes it and the byte behavior is unchanged.  `EmulatedEp2` (harness
/// feature only) runs the two logical expert-ownership ranks sequentially
/// into reusable partial buffers with only the gate-up pointer tables
/// overridden via [`Qwen35Weights::moe_ffn_view_ep2`]; down/AWQ/tags/router/
/// shared/Paro stay canonical and borrowed from the single Frozen owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoeExecution {
    Single,
    #[cfg(feature = "emulated-ep2-harness")]
    EmulatedEp2,
}

/// Emulated-EP2 prefill marker threaded through the batched-prefill chain
/// (harness feature only).  Carries no data: the per-layer rank views come
/// from [`Qwen35Weights::moe_ffn_view_ep2`] inside the MoE body and the
/// partial buffers live on the caller-owned [`PrefillBatchScratch`].  The
/// production and legacy-EP drivers always pass `None`.
#[cfg(feature = "emulated-ep2-harness")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ep2PrefillCtx;

/// Emulated-EP2 decode execution contract (pure policy, pinned by CPU
/// tests): rank 0 runs the shared-expert down contribution, rank 1 skips
/// it, so the shared expert is counted exactly once after the partial
/// combine.  `true` = pass `skip_shared` to the MoE decode family.
#[cfg(feature = "emulated-ep2-harness")]
pub(crate) fn ep2_rank_skip_shared(rank: u8) -> bool {
    rank != 0
}

// ── Helper: build a WeightRef from a GpuTensor + metadata ────────────────

fn wt_ref_from_tensor<'a>(
    buf: &'a GpuTensor,
    dtype: DType,
    m: usize,
    k: usize,
    awq_scale: Option<&'a GpuTensor>,
    paro: Option<&'a MoeParoSidecars>,
) -> hipfire_dispatch::families::gemv::WeightRef<'a> {
    hipfire_dispatch::families::gemv::WeightRef {
        buf,
        dtype,
        m,
        k,
        row_stride: 0,
        rotation: paro.map(|p| hipfire_dispatch::families::gemv::GivensRef {
            pairs: &p.gate_up_pairs,
            theta: &p.gate_up_theta,
            scales: &p.gate_up_channel_scales,
            krot: p.krot as usize,
        }),
        awq_scale,
    }
}

fn wt_ref_from_weight_tensor(wt: &WeightTensor) -> hipfire_dispatch::families::gemv::WeightRef<'_> {
    hipfire_dispatch::families::gemv::WeightRef {
        buf: &wt.buf,
        dtype: wt.gpu_dtype,
        m: wt.m,
        k: wt.k,
        row_stride: wt.row_stride,
        rotation: wt
            .paro
            .as_ref()
            .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &p.pairs,
                theta: &p.theta,
                scales: &p.channel_scales,
                krot: p.krot as usize,
            }),
        awq_scale: wt.awq_scale.as_ref(),
    }
}
impl<'a> MoeFfnView<'a> {
    fn frozen_bindings(&self) -> &crate::store::MoeFfnBindings<'a> {
        match self {
            MoeFfnView::Legacy(_) => panic!("frozen_bindings called on Legacy variant"),
            MoeFfnView::Frozen(b) => b,
        }
    }

    fn proj(&self) -> Option<&Qwen35MoeLayerProjection<WeightCellId>> {
        match self {
            MoeFfnView::Legacy(_) => None,
            MoeFfnView::Frozen(b) => Some(b.descriptors()),
        }
    }

    // ── Metadata: router ──────────────────────────────────────────────

    pub(crate) fn router_dtype(&self) -> DType {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.router.gpu_dtype,
            MoeFfnView::Frozen { .. } => self.proj().map_or(DType::F32, |p| p.router.dtype),
        }
    }

    pub(crate) fn router_m(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.router.m,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.router.m),
        }
    }

    /// Router inner dim (hidden).
    pub(crate) fn router_k(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.router.k,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.router.k),
        }
    }

    pub(crate) fn router_has_awq(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.router.awq_scale.is_some(),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .is_some_and(|p| p.router.awq_companion_key.is_some()),
        }
    }

    pub(crate) fn shared_expert_gate_has_awq(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert_gate.awq_scale.is_some(),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .is_some_and(|p| p.shared_expert_gate.awq_companion_key.is_some()),
        }
    }

    pub(crate) fn shared_gate_has_awq(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.gate.awq_scale.is_some(),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .is_some_and(|p| p.shared_gate.awq_companion_key.is_some()),
        }
    }

    pub(crate) fn shared_up_has_awq(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.up.awq_scale.is_some(),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .is_some_and(|p| p.shared_up.awq_companion_key.is_some()),
        }
    }

    // ── Metadata: shared expert ───────────────────────────────────────

    pub(crate) fn shared_expert_gate_m(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert_gate.m,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_expert_gate.m),
        }
    }

    pub(crate) fn shared_expert_gate_k(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert_gate.k,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_expert_gate.k),
        }
    }

    pub(crate) fn shared_expert_gate_dtype(&self) -> DType {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert_gate.gpu_dtype,
            MoeFfnView::Frozen { .. } => self
                .proj()
                .map_or(DType::F32, |p| p.shared_expert_gate.dtype),
        }
    }

    pub(crate) fn shared_gate_dtype(&self) -> DType {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.gate.gpu_dtype,
            MoeFfnView::Frozen { .. } => self.proj().map_or(DType::F32, |p| p.shared_gate.dtype),
        }
    }

    pub(crate) fn shared_up_dtype(&self) -> DType {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.up.gpu_dtype,
            MoeFfnView::Frozen { .. } => self.proj().map_or(DType::F32, |p| p.shared_up.dtype),
        }
    }

    pub(crate) fn shared_down_dtype(&self) -> DType {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.down.gpu_dtype,
            MoeFfnView::Frozen { .. } => self.proj().map_or(DType::F32, |p| p.shared_down.dtype),
        }
    }

    pub(crate) fn shared_gate_m(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.gate.m,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_gate.m),
        }
    }

    pub(crate) fn shared_gate_k(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.gate.k,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_gate.k),
        }
    }

    pub(crate) fn shared_up_m(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.up.m,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_up.m),
        }
    }

    pub(crate) fn shared_up_k(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.up.k,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_up.k),
        }
    }

    pub(crate) fn shared_down_m(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.down.m,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_down.m),
        }
    }

    pub(crate) fn shared_down_k(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.shared_expert.down.k,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.shared_down.k),
        }
    }

    // ── Metadata: routed experts ──────────────────────────────────────

    pub(crate) fn expert_count(&self) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.experts.len(),
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.expert_gate_up.len()),
        }
    }

    pub(crate) fn expert_gate_up_dtype(&self, idx: usize) -> DType {
        match self {
            MoeFfnView::Legacy(ffn) => ffn
                .experts
                .get(idx)
                .map_or(DType::F32, |e| e.gate_up.gpu_dtype),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .and_then(|p| p.expert_gate_up.get(idx))
                .map_or(DType::F32, |d| d.dtype),
        }
    }

    pub(crate) fn expert_down_dtype(&self, idx: usize) -> DType {
        match self {
            MoeFfnView::Legacy(ffn) => ffn
                .experts
                .get(idx)
                .map_or(DType::F32, |e| e.down.gpu_dtype),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .and_then(|p| p.expert_down.get(idx))
                .map_or(DType::F32, |d| d.dtype),
        }
    }

    pub(crate) fn expert_gate_up_k(&self, idx: usize) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.experts.get(idx).map_or(0, |e| e.gate_up.k),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .and_then(|p| p.expert_gate_up.get(idx))
                .map_or(0, |d| d.k),
        }
    }

    pub(crate) fn expert_down_m(&self, idx: usize) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.experts.get(idx).map_or(0, |e| e.down.m),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .and_then(|p| p.expert_down.get(idx))
                .map_or(0, |d| d.m),
        }
    }

    pub(crate) fn expert_down_k(&self, idx: usize) -> usize {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.experts.get(idx).map_or(0, |e| e.down.k),
            MoeFfnView::Frozen { .. } => self
                .proj()
                .and_then(|p| p.expert_down.get(idx))
                .map_or(0, |d| d.k),
        }
    }

    pub(crate) fn experts_all_gate_up_mq4(&self) -> bool {
        (0..self.expert_count()).all(|i| self.expert_gate_up_dtype(i) == DType::MQ4G256)
    }

    pub(crate) fn all_experts_gate_up_dtype(&self, dt: DType) -> bool {
        (0..self.expert_count()).all(|i| self.expert_gate_up_dtype(i) == dt)
    }

    pub(crate) fn all_experts_down_dtype(&self, dt: DType) -> bool {
        (0..self.expert_count()).all(|i| self.expert_down_dtype(i) == dt)
    }

    /// First expert's gate_up dtype for dimension/dtype queries.
    pub(crate) fn first_expert_gate_up_dtype(&self) -> DType {
        if self.expert_count() == 0 {
            return DType::F32;
        }
        self.expert_gate_up_dtype(0)
    }

    /// First expert's down dtype.
    pub(crate) fn first_expert_down_dtype(&self) -> DType {
        if self.expert_count() == 0 {
            return DType::F32;
        }
        self.expert_down_dtype(0)
    }

    /// First expert's gate_up k (inner dim).
    pub(crate) fn first_expert_gate_up_k(&self) -> usize {
        self.expert_gate_up_k(0)
    }

    /// First expert's down m (outer dim).
    pub(crate) fn first_expert_down_m(&self) -> usize {
        self.expert_down_m(0)
    }

    /// First expert's down k (inner dim).
    pub(crate) fn first_expert_down_k(&self) -> usize {
        self.expert_down_k(0)
    }

    // ── Metadata: composite dtype helpers ─────────────────────────────

    fn per_expert_gate_up_tiers(&self) -> Option<Vec<DType>> {
        let n = self.expert_count();
        let tiers: Vec<DType> = (0..n).map(|i| self.expert_gate_up_dtype(i)).collect();
        mixed_tier_table(tiers)
    }

    fn per_expert_down_tiers(&self) -> Option<Vec<DType>> {
        let n = self.expert_count();
        let tiers: Vec<DType> = (0..n).map(|i| self.expert_down_dtype(i)).collect();
        mixed_tier_table(tiers)
    }

    pub(crate) fn per_expert_tier_tables(&self) -> (Option<Vec<DType>>, Option<Vec<DType>>) {
        (
            self.per_expert_gate_up_tiers(),
            self.per_expert_down_tiers(),
        )
    }

    // ── Metadata: optional derived descriptors ────────────────────────

    pub(crate) fn expert_dtype_tags_present(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.expert_dtype_tags.is_some(),
            MoeFfnView::Frozen { .. } => self.proj().and_then(|p| p.dtype_tags.as_ref()).is_some(),
        }
    }

    pub(crate) fn paro_shared_present(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.paro_shared.is_some(),
            MoeFfnView::Frozen { .. } => false,
        }
    }

    /// True when the routed-down projection carries per-expert AWQ scales.
    pub(crate) fn routed_down_awq_present(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.expert_down_awq_ptrs.is_some(),
            MoeFfnView::Frozen { .. } => self.proj().is_some_and(|p| p.expert_down_awq.is_some()),
        }
    }

    /// Layer index (stable identity for pager keying).
    fn layer_idx(&self) -> u16 {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.layer_idx,
            MoeFfnView::Frozen { .. } => self.proj().map_or(0, |p| p.layer_idx as u16),
        }
    }

    // ── Metadata: predicates ──────────────────────────────────────────

    /// All gate-side + routed weights are MQ4G256.
    /// All gate-side + routed weights are MQ4G256.
    fn all_mq4(&self) -> bool {
        self.to_snapshot().all_mq4()
    }

    /// Gate-side only MQ4 (router + shared expert), independent of routed experts.
    fn gate_side_mq4(&self) -> bool {
        self.to_snapshot().gate_side_mq4()
    }

    /// Any MQ3G256 / MQ3G256Lloyd in STRUCTURAL parts.
    fn has_mq3_structural(&self) -> bool {
        self.to_snapshot().has_mq3_structural()
    }

    /// MQ3 in ROUTED experts WITHOUT a tag table (uniform MQ3, not graded).
    fn has_mq3_experts_uniform(&self) -> bool {
        self.to_snapshot().has_mq3_experts_uniform()
    }

    /// Any MQ6G256 anywhere in the FFN — shared predicate with the Frozen
    /// path ([`crate::store::MoeFfnMetaView::has_mq6`]) so the two storage
    /// kinds cannot diverge: router, shared_expert_gate, shared gate/up/
    /// down, or ANY routed expert gate_up/down (uniform or graded).
    #[cfg(test)]
    fn has_mq6(&self) -> bool {
        match self {
            MoeFfnView::Legacy(ffn) => {
                crate::store::MoeFfnMetaView::<'_, WeightCellId>::Legacy(ffn).has_mq6()
            }
            MoeFfnView::Frozen(b) => {
                crate::store::MoeFfnMetaView::Frozen(b.descriptors()).has_mq6()
            }
        }
    }

    /// Extract MoeDtypeSnapshot from this view (metadata only, no tensor binding).
    fn to_snapshot(&self) -> MoeDtypeSnapshot {
        MoeDtypeSnapshot {
            router: self.router_dtype(),
            shared_expert_scalar_gate: self.shared_expert_gate_dtype(),
            shared_gate: self.shared_gate_dtype(),
            shared_up: self.shared_up_dtype(),
            shared_down: self.shared_down_dtype(),
            expert_gate_up: self.first_expert_gate_up_dtype(),
            expert_down: self.first_expert_down_dtype(),
            expert_gate_up_uniform: self
                .all_experts_gate_up_dtype(self.first_expert_gate_up_dtype()),
            expert_down_uniform: self.all_experts_down_dtype(self.first_expert_down_dtype()),
            expert_dtype_tags_present: self.expert_dtype_tags_present(),
            expert_count: self.expert_count(),
            gate_side_has_awq: self.router_has_awq()
                || self.shared_expert_gate_has_awq()
                || self.shared_gate_has_awq()
                || self.shared_up_has_awq(),
        }
    }

    /// Router weight tensor reference.
    pub(crate) fn router_ref(
        &self,
    ) -> Result<hipfire_dispatch::families::gemv::WeightRef<'_>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(wt_ref_from_weight_tensor(&ffn.router)),
            MoeFfnView::Frozen { .. } => {
                let fb = self.frozen_bindings();
                let t = fb.router()?;
                let p = fb.descriptors();
                let awq = fb.router_awq()?;
                Ok(wt_ref_from_tensor(
                    t,
                    p.router.dtype,
                    p.router.m,
                    p.router.k,
                    awq,
                    None,
                ))
            }
        }
    }

    /// Shared expert scalar gate tensor reference.
    pub(crate) fn shared_expert_gate_ref(
        &self,
    ) -> Result<hipfire_dispatch::families::gemv::WeightRef<'_>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(wt_ref_from_weight_tensor(&ffn.shared_expert_gate)),
            MoeFfnView::Frozen { .. } => {
                let fb = self.frozen_bindings();
                let t = fb.shared_expert_gate()?;
                let p = fb.descriptors();
                let awq = fb.shared_expert_gate_awq()?;
                Ok(wt_ref_from_tensor(
                    t,
                    p.shared_expert_gate.dtype,
                    p.shared_expert_gate.m,
                    p.shared_expert_gate.k,
                    awq,
                    None,
                ))
            }
        }
    }

    /// Shared expert gate projection reference.
    pub(crate) fn shared_gate_ref(
        &self,
    ) -> Result<hipfire_dispatch::families::gemv::WeightRef<'_>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(wt_ref_from_weight_tensor(&ffn.shared_expert.gate)),
            MoeFfnView::Frozen { .. } => {
                let fb = self.frozen_bindings();
                let t = fb.shared_gate()?;
                let p = fb.descriptors();
                let awq = fb.shared_gate_awq()?;
                Ok(wt_ref_from_tensor(
                    t,
                    p.shared_gate.dtype,
                    p.shared_gate.m,
                    p.shared_gate.k,
                    awq,
                    None,
                ))
            }
        }
    }

    /// Shared expert up projection reference.
    pub(crate) fn shared_up_ref(
        &self,
    ) -> Result<hipfire_dispatch::families::gemv::WeightRef<'_>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(wt_ref_from_weight_tensor(&ffn.shared_expert.up)),
            MoeFfnView::Frozen { .. } => {
                let fb = self.frozen_bindings();
                let t = fb.shared_up()?;
                let p = fb.descriptors();
                let awq = fb.shared_up_awq()?;
                Ok(wt_ref_from_tensor(
                    t,
                    p.shared_up.dtype,
                    p.shared_up.m,
                    p.shared_up.k,
                    awq,
                    None,
                ))
            }
        }
    }

    /// Shared expert down projection reference.
    pub(crate) fn shared_down_ref(
        &self,
    ) -> Result<hipfire_dispatch::families::gemv::WeightRef<'_>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(wt_ref_from_weight_tensor(&ffn.shared_expert.down)),
            MoeFfnView::Frozen { .. } => {
                let fb = self.frozen_bindings();
                let t = fb.shared_down()?;
                let p = fb.descriptors();
                let awq = fb.shared_down_awq()?;
                Ok(wt_ref_from_tensor(
                    t,
                    p.shared_down.dtype,
                    p.shared_down.m,
                    p.shared_down.k,
                    awq,
                    None,
                ))
            }
        }
    }

    /// Per-expert gate-up pointer table tensor.
    pub(crate) fn expert_gate_up_ptrs_tensor(&self) -> Result<&GpuTensor, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(&ffn.expert_gate_up_ptrs),
            MoeFfnView::Frozen { .. } => self.frozen_bindings().gate_up_ptrs(),
        }
    }

    /// Per-expert down pointer table tensor.
    pub(crate) fn expert_down_ptrs_tensor(&self) -> Result<&GpuTensor, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(&ffn.expert_down_ptrs),
            MoeFfnView::Frozen { .. } => self.frozen_bindings().down_ptrs(),
        }
    }

    /// Optional per-expert down AWQ pointer table tensor.
    pub(crate) fn expert_down_awq_ptrs_tensor(
        &self,
    ) -> Result<Option<&GpuTensor>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(ffn.expert_down_awq_ptrs.as_ref()),
            MoeFfnView::Frozen { .. } => self.frozen_bindings().down_awq_ptrs(),
        }
    }

    /// Optional per-expert dtype tags tensor.
    pub(crate) fn expert_dtype_tags_tensor(
        &self,
    ) -> Result<Option<&GpuTensor>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => Ok(ffn.expert_dtype_tags.as_ref()),
            MoeFfnView::Frozen { .. } => self.frozen_bindings().dtype_tags(),
        }
    }

    /// Per-expert gate-up weight reference.
    fn expert_gate_up_ref(
        &self,
        idx: usize,
    ) -> Result<hipfire_dispatch::families::gemv::WeightRef<'_>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => {
                let e = ffn
                    .experts
                    .get(idx)
                    .ok_or(Qwen35MoeBindError::LayerOutOfRange {
                        requested: idx,
                        count: ffn.experts.len(),
                    })?;
                Ok(wt_ref_from_weight_tensor(&e.gate_up))
            }
            MoeFfnView::Frozen { .. } => {
                let t = self.frozen_bindings().expert_gate_up(idx)?;
                let desc = self.frozen_bindings().expert_gate_up_desc(idx).ok_or(
                    Qwen35MoeBindError::LayerOutOfRange {
                        requested: idx,
                        count: self.frozen_bindings().num_experts(),
                    },
                )?;
                Ok(wt_ref_from_tensor(
                    t, desc.dtype, desc.m, desc.k, None, None,
                ))
            }
        }
    }

    /// Per-expert down weight reference.
    fn expert_down_ref(
        &self,
        idx: usize,
    ) -> Result<hipfire_dispatch::families::gemv::WeightRef<'_>, Qwen35MoeBindError> {
        match self {
            MoeFfnView::Legacy(ffn) => {
                let e = ffn
                    .experts
                    .get(idx)
                    .ok_or(Qwen35MoeBindError::LayerOutOfRange {
                        requested: idx,
                        count: ffn.experts.len(),
                    })?;
                Ok(wt_ref_from_weight_tensor(&e.down))
            }
            MoeFfnView::Frozen { .. } => {
                let t = self.frozen_bindings().expert_down(idx)?;
                let desc = self.frozen_bindings().expert_down_desc(idx).ok_or(
                    Qwen35MoeBindError::LayerOutOfRange {
                        requested: idx,
                        count: self.frozen_bindings().num_experts(),
                    },
                )?;
                Ok(wt_ref_from_tensor(
                    t, desc.dtype, desc.m, desc.k, None, None,
                ))
            }
        }
    }

    /// Build the per-expert (gate_up, down) `WeightRef` Vec.
    fn routed_expert_refs(
        &self,
    ) -> Result<
        Vec<(
            hipfire_dispatch::families::gemv::WeightRef<'_>,
            hipfire_dispatch::families::gemv::WeightRef<'_>,
        )>,
        Qwen35MoeBindError,
    > {
        #[cfg(test)]
        if routed_ref_seam::INSTRUMENT.load(std::sync::atomic::Ordering::Relaxed) {
            routed_ref_seam::RESOLUTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let n = self.expert_count();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push((self.expert_gate_up_ref(i)?, self.expert_down_ref(i)?));
        }
        Ok(out)
    }

    /// First expert's gate-up Paro rotation (if any).
    pub(crate) fn first_expert_gate_up_paro(
        &self,
    ) -> Option<hipfire_dispatch::families::gemv::GivensRef<'_>> {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.experts.first().and_then(|e| {
                e.gate_up
                    .paro
                    .as_ref()
                    .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                        pairs: &p.pairs,
                        theta: &p.theta,
                        scales: &p.channel_scales,
                        krot: p.krot as usize,
                    })
            }),
            MoeFfnView::Frozen { .. } => None,
        }
    }

    /// First expert's down Paro rotation (if any).
    pub(crate) fn first_expert_down_paro(
        &self,
    ) -> Option<hipfire_dispatch::families::gemv::GivensRef<'_>> {
        match self {
            MoeFfnView::Legacy(ffn) => ffn.experts.first().and_then(|e| {
                e.down
                    .paro
                    .as_ref()
                    .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                        pairs: &p.pairs,
                        theta: &p.theta,
                        scales: &p.channel_scales,
                        krot: p.krot as usize,
                    })
            }),
            MoeFfnView::Frozen { .. } => None,
        }
    }
}

/// Test-only routed-ref resolution instrumentation (call-count seam).
///
/// The O(1) Frozen binding contract is: the Frozen decode/prefill path
/// NEVER materializes the per-expert `routed_expert_refs()` Vec (the C2
/// indexed GPU route — pointer tables + dtype tags — is guaranteed for
/// every admitted Frozen layer).  This seam lets the tests prove that
/// contract with a call counter instead of inspecting allocations.
/// `INSTRUMENT` defaults off so unrelated tests never observe it.
#[cfg(test)]
pub(crate) mod routed_ref_seam {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Mutex, MutexGuard};

    /// Serializes seam tests so their delta assertions cannot observe
    /// each other's increments.
    pub static LOCK: Mutex<()> = Mutex::new(());
    /// When set, every [`super::MoeFfnView::routed_expert_refs`] call
    /// increments [`RESOLUTIONS`].
    pub static INSTRUMENT: AtomicBool = AtomicBool::new(false);
    /// Number of routed-ref resolutions performed while instrumented.
    pub static RESOLUTIONS: AtomicUsize = AtomicUsize::new(0);

    /// Reset the instrumentation state (call while holding [`LOCK`]).
    pub fn reset() {
        INSTRUMENT.store(false, Ordering::Relaxed);
        RESOLUTIONS.store(0, Ordering::Relaxed);
    }

    /// RAII guard: enables the counter for its lifetime (holding [`LOCK`]
    /// so delta assertions are race-free) and restores the counter to off
    /// on drop.
    pub struct SeamGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl SeamGuard {
        pub fn on() -> Self {
            let _lock = LOCK.lock().unwrap();
            reset();
            INSTRUMENT.store(true, Ordering::Relaxed);
            SeamGuard { _lock }
        }
    }

    impl Drop for SeamGuard {
        fn drop(&mut self) {
            INSTRUMENT.store(false, Ordering::Relaxed);
        }
    }
}

/// Routed-expert refs for `MoeParams`, O(1) on the Frozen path.
///
/// Frozen layers never materialize the per-expert Vec: the C2 indexed GPU
/// route (`expert_gate_up_ptrs` / `expert_down_ptrs` / AWQ pointer table /
/// dtype tags) is guaranteed for every admitted Frozen layer, so building
/// `n_exp` `WeightRef` pairs would be O(n_exp) dead work per decode token.
/// They pass an EMPTY slice — dispatch's `check_moe_decode_supported` guard
/// rejects empty refs on the CPU-top-K fallback, so no fake refs/aliases
/// are ever passed and a Frozen layer that somehow lacks the indexed route
/// fails loudly instead of mis-executing.
///
/// Legacy layers materialize one (gate_up, down) pair per expert exactly as
/// before — the CPU-top-K fallback iterates them.
pub(crate) fn routed_expert_refs_for_params<'a>(
    view: &'a MoeFfnView<'a>,
) -> Result<
    Vec<(
        hipfire_dispatch::families::gemv::WeightRef<'a>,
        hipfire_dispatch::families::gemv::WeightRef<'a>,
    )>,
    Qwen35MoeBindError,
> {
    match view {
        MoeFfnView::Frozen(_) => Ok(Vec::new()),
        MoeFfnView::Legacy(_) => view.routed_expert_refs(),
    }
}

pub(crate) struct MoeDtypeSnapshot {
    pub(crate) router: DType,
    pub(crate) shared_expert_scalar_gate: DType,
    pub(crate) shared_gate: DType,
    pub(crate) shared_up: DType,
    pub(crate) shared_down: DType,
    pub(crate) expert_gate_up: DType,
    pub(crate) expert_down: DType,
    pub(crate) expert_gate_up_uniform: bool,
    pub(crate) expert_down_uniform: bool,
    pub(crate) expert_dtype_tags_present: bool,
    pub(crate) expert_count: usize,
    /// True when any of router / shared_expert_gate / shared gate/up
    /// carries an AWQ sidecar.  When true, `gate_side_mq4` returns false
    /// and gate-fused execution paths are disabled (each weight uses its
    /// individual WeightRef path which applies the per-weight AWQ scale).
    pub(crate) gate_side_has_awq: bool,
}

impl MoeDtypeSnapshot {
    pub(crate) fn all_mq4(&self) -> bool {
        self.gate_side_mq4()
            && self.expert_count > 0
            && self.expert_gate_up_uniform
            && self.expert_gate_up == DType::MQ4G256
    }

    pub(crate) fn gate_side_mq4(&self) -> bool {
        !self.gate_side_has_awq
            && self.router == DType::MQ4G256
            && self.shared_expert_scalar_gate == DType::MQ4G256
            && self.shared_gate == DType::MQ4G256
            && self.shared_up == DType::MQ4G256
    }

    pub(crate) fn has_mq3_structural(&self) -> bool {
        let is_mq3 = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
        is_mq3(self.router)
            || is_mq3(self.shared_expert_scalar_gate)
            || is_mq3(self.shared_gate)
            || is_mq3(self.shared_up)
            || is_mq3(self.shared_down)
    }

    pub(crate) fn has_mq3_experts_uniform(&self) -> bool {
        let is_mq3 = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
        !self.expert_dtype_tags_present
            && self.expert_count > 0
            && self.expert_gate_up_uniform
            && (is_mq3(self.expert_gate_up) || is_mq3(self.expert_down))
    }
    /// Batched-prefill admission for this dtype snapshot.  Delegates to the
    /// prefill module's env-gated lattice (E8/PARO/codebook gates) so the
    /// snapshot path can never drift from the `MoeFfnWeights`-based one.
    pub(crate) fn batched_admissible(&self, admit_mq6: bool, arch: &str) -> bool {
        crate::qwen35::snapshot_batched_admissible(self, admit_mq6, arch)
    }

    /// The prefill dtype view of this snapshot (the same
    /// `MoePrefillDtypes::from_snapshot` conversion `snapshot_batched_admissible`
    /// consumes), exposed for store-side tests that assert per-field dtype
    /// fidelity of the snapshot → prefill conversion. `None` when the layer
    /// has no routed experts.
    pub(crate) fn prefill_dtypes(&self) -> Option<crate::qwen35::prefill::MoePrefillDtypes> {
        crate::qwen35::prefill::MoePrefillDtypes::from_snapshot(self)
    }
}
/// Build MoeDtypes from a MoeFfnView using metadata only (no tensor binding).
fn moe_dtypes_from_view(view: &MoeFfnView<'_>) -> hipfire_dispatch::families::moe::MoeDtypes {
    let (per_expert_gate_up, per_expert_down) = view.per_expert_tier_tables();
    let gate_side_has_awq = view.router_has_awq()
        || view.shared_expert_gate_has_awq()
        || view.shared_gate_has_awq()
        || view.shared_up_has_awq();
    hipfire_dispatch::families::moe::MoeDtypes {
        router: view.router_dtype(),
        shared_gate: view.shared_expert_gate_dtype(),
        shared_expert_gate: view.shared_gate_dtype(),
        shared_expert_up: view.shared_up_dtype(),
        shared_expert_down: view.shared_down_dtype(),
        experts_all_gate_up_mq4: view.experts_all_gate_up_mq4(),
        routed_gate_up: view.first_expert_gate_up_dtype(),
        routed_down: view.first_expert_down_dtype(),
        routed_has_mixed_experts: view.expert_dtype_tags_present(),
        has_paro_shared: view.paro_shared_present(),
        gate_side_has_awq,
        routed_down_has_awq: view.routed_down_awq_present(),
        per_expert_gate_up,
        per_expert_down,
    }
}

/// Immutable source identity captured before any EP GPU allocation.
/// Exact equality over canonical path, platform file identity (dev, ino),
/// length, mtime, arch_id, exact metadata_json, ordered tensor manifest
/// (name, quant_type, shape, group_size, data_offset, data_size) with
/// absolute offsets (base offset included), and overlay status.
/// Not a hash – any reordering or header difference is inequality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35HfqSourceIdentity {
    pub canonical_path: std::path::PathBuf,
    pub dev: u64,
    pub ino: u64,
    pub file_len: u64,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
    pub arch_id: u32,
    pub metadata_json: String,
    pub tensor_manifest: Vec<(String, u8, Vec<u32>, u32, usize, usize)>,
    pub has_overlay: bool,
}

impl Qwen35HfqSourceIdentity {
    pub fn capture(hfq: &HfqFile) -> Self {
        let path = hfq.path().to_path_buf();
        let canonical = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let (dev, ino, file_len, mtime_secs, mtime_nanos) = {
            match std::fs::metadata(&path) {
                Ok(md) => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        let mtime = md
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                        (
                            md.dev(),
                            md.ino(),
                            md.len(),
                            mtime.map(|d| d.as_secs() as i64).unwrap_or(0),
                            mtime.map(|d| d.subsec_nanos()).unwrap_or(0),
                        )
                    }
                    #[cfg(not(unix))]
                    {
                        let mtime = md
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                        (
                            0u64,
                            0u64,
                            md.len(),
                            mtime.map(|d| d.as_secs() as i64).unwrap_or(0),
                            mtime.map(|d| d.subsec_nanos()).unwrap_or(0),
                        )
                    }
                }
                Err(_) => (0, 0, 0, 0, 0),
            }
        };
        let tensors = hfq.tensors();
        let manifest = tensors
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    t.quant_type,
                    t.shape.clone(),
                    t.group_size,
                    t.data_offset,
                    t.data_size,
                )
            })
            .collect();
        Self {
            canonical_path: canonical,
            dev,
            ino,
            file_len,
            mtime_secs,
            mtime_nanos,
            arch_id: hfq.arch_id,
            metadata_json: hfq.metadata_json.clone(),
            tensor_manifest: manifest,
            has_overlay: hfq.has_overlay(),
        }
    }
}

/// Frozen config fingerprint for EP seal. Contains every Qwen35Config primitive.
/// Equality is exact (f32 via to_bits). EP admission still rejects paged/REAP but
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35EpConfigFingerprint {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps_bits: u32,
    pub eos_token: u32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta_bits: u32,
    pub partial_rotary_factor_bits: u32,
    pub is_vl_text: bool,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub conv_kernel_dim: usize,
    pub hidden_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub has_shared_expert: bool,
    pub norm_topk_prob: bool,
    pub layer_types: Vec<LayerType>,
    pub paged_experts: bool,
    pub vram_budget_bytes: u64,
    pub has_reap_keep: bool,
}

impl Qwen35EpConfigFingerprint {
    pub fn capture(config: &Qwen35Config) -> Self {
        Self {
            dim: config.dim,
            n_layers: config.n_layers,
            vocab_size: config.vocab_size,
            norm_eps_bits: config.norm_eps.to_bits(),
            eos_token: config.eos_token,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            rope_theta_bits: config.rope_theta.to_bits(),
            partial_rotary_factor_bits: config.partial_rotary_factor.to_bits(),
            is_vl_text: config.is_vl_text,
            mrope_interleaved: config.mrope_interleaved,
            mrope_section: config.mrope_section,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            conv_kernel_dim: config.conv_kernel_dim,
            hidden_dim: config.hidden_dim,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_expert_intermediate_size: config.shared_expert_intermediate_size,
            has_shared_expert: config.has_shared_expert,
            norm_topk_prob: config.norm_topk_prob,
            layer_types: config.layer_types.clone(),
            paged_experts: config.paged_experts,
            vram_budget_bytes: config.vram_budget_bytes,
            has_reap_keep: config.reap_keep.is_some(),
        }
    }
}

/// Device-pointer-free descriptor for a GpuTensor. Excludes DeviceBuffer pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTensorDescriptor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub byte_len: usize,
}

impl GpuTensorDescriptor {
    pub fn from_tensor(t: &GpuTensor) -> Self {
        Self {
            shape: t.shape.clone(),
            dtype: t.dtype,
            byte_len: t.buf.size(),
        }
    }
}

/// Paro sidecar descriptor excluding pointers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParoDescriptor {
    pub krot: u32,
    pub group_size: u32,
    pub is_alias: bool,
    pub pairs: GpuTensorDescriptor,
    pub theta: GpuTensorDescriptor,
    pub channel_scales: GpuTensorDescriptor,
}

/// Weight tensor descriptor excluding device pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightTensorDescriptor {
    pub gpu_dtype: DType,
    pub m: usize,
    pub k: usize,
    pub row_stride: usize,
    pub buf: GpuTensorDescriptor,
    pub awq_scale: Option<GpuTensorDescriptor>,
    pub paro: Option<ParoDescriptor>,
}

impl WeightTensorDescriptor {
    pub fn from_weight(w: &WeightTensor) -> Self {
        Self {
            gpu_dtype: w.gpu_dtype,
            m: w.m,
            k: w.k,
            row_stride: w.row_stride,
            buf: GpuTensorDescriptor::from_tensor(&w.buf),
            awq_scale: w.awq_scale.as_ref().map(GpuTensorDescriptor::from_tensor),
            paro: w.paro.as_ref().map(|p| ParoDescriptor {
                krot: p.krot,
                group_size: p.group_size,
                is_alias: p.is_alias,
                pairs: GpuTensorDescriptor::from_tensor(&p.pairs),
                theta: GpuTensorDescriptor::from_tensor(&p.theta),
                channel_scales: GpuTensorDescriptor::from_tensor(&p.channel_scales),
            }),
        }
    }
}

/// Per-expert local descriptor: global id + gate_up/down descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LocalExpertDescriptor {
    pub global_expert_id: usize,
    pub gate_up: WeightTensorDescriptor,
    pub down: WeightTensorDescriptor,
}
/// Complete rank weight/layout seal. Immutable, allocation-free comparison via `matches_*`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35RankSeal {
    pub token_embd: GpuTensorDescriptor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensorDescriptor,
    pub output: WeightTensorDescriptor,
    pub moe_has_mq6: bool,
    pub has_pager: bool,
    pub lm_head_aliases_embd: bool,
    pub layer_seals: Vec<Qwen35LayerSeal>,
    pub global_expert_dtypes: Vec<Vec<(DType, DType)>>,
    pub local_expert_descriptors: Vec<Vec<Qwen35LocalExpertDescriptor>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qwen35LayerSeal {
    DeltaNet {
        attn_norm: GpuTensorDescriptor,
        wqkv: WeightTensorDescriptor,
        wz: WeightTensorDescriptor,
        w_alpha: WeightTensorDescriptor,
        w_beta: WeightTensorDescriptor,
        a_log: GpuTensorDescriptor,
        dt_bias: GpuTensorDescriptor,
        conv_weight: GpuTensorDescriptor,
        norm_weight: GpuTensorDescriptor,
        wo: WeightTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        w_gate: WeightTensorDescriptor,
        w_up: WeightTensorDescriptor,
        w_down: WeightTensorDescriptor,
    },
    FullAttn {
        attn_norm: GpuTensorDescriptor,
        wq: WeightTensorDescriptor,
        wk: WeightTensorDescriptor,
        wv: WeightTensorDescriptor,
        wo: WeightTensorDescriptor,
        q_norm: GpuTensorDescriptor,
        k_norm: GpuTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        w_gate: WeightTensorDescriptor,
        w_up: WeightTensorDescriptor,
        w_down: WeightTensorDescriptor,
    },
    DeltaNetMoe {
        attn_norm: GpuTensorDescriptor,
        wqkv: WeightTensorDescriptor,
        wz: WeightTensorDescriptor,
        w_alpha: WeightTensorDescriptor,
        w_beta: WeightTensorDescriptor,
        a_log: GpuTensorDescriptor,
        dt_bias: GpuTensorDescriptor,
        conv_weight: GpuTensorDescriptor,
        norm_weight: GpuTensorDescriptor,
        wo: WeightTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        moe: Qwen35MoeFfnSeal,
    },
    FullAttnMoe {
        attn_norm: GpuTensorDescriptor,
        wq: WeightTensorDescriptor,
        wk: WeightTensorDescriptor,
        wv: WeightTensorDescriptor,
        wo: WeightTensorDescriptor,
        q_norm: GpuTensorDescriptor,
        k_norm: GpuTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        moe: Qwen35MoeFfnSeal,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeFfnSeal {
    pub router: WeightTensorDescriptor,
    pub shared_gate: WeightTensorDescriptor,
    pub shared_up: WeightTensorDescriptor,
    pub shared_down: WeightTensorDescriptor,
    pub shared_expert_gate: WeightTensorDescriptor,
    pub expert_gate_up_ptrs: GpuTensorDescriptor,
    pub expert_down_ptrs: GpuTensorDescriptor,
    pub expert_down_awq_ptrs: Option<GpuTensorDescriptor>,
    pub expert_dtype_tags: Option<GpuTensorDescriptor>,
    pub layer_idx: u16,
    pub has_packed_owners: bool,
    pub global_expert_dtypes: Option<Vec<(DType, DType)>>,
    pub num_local_experts: usize,
}

impl Qwen35RankSeal {
    pub fn capture(weights: &Qwen35Weights, expert_to_rank: Option<&[u8]>, rank: usize) -> Self {
        let mut global_expert_dtypes = Vec::with_capacity(weights.layers.len());
        let mut local_expert_descriptors = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            // EP rank seals are captured on the Legacy load path only
            // (`load_weights_ep_rank` builds owned layers); Frozen storage
            // cannot reach this surface.
            let ffn = match layer {
                LayerWeights::DeltaNetMoe(weights) => weights.ffn.as_legacy(),
                LayerWeights::FullAttnMoe(weights) => weights.ffn.as_legacy(),
                _ => None,
            };
            let Some(ffn) = ffn else {
                global_expert_dtypes.push(Vec::new());
                local_expert_descriptors.push(Vec::new());
                continue;
            };
            global_expert_dtypes.push(
                ffn.global_expert_dtypes
                    .as_ref()
                    .map(|dtypes| dtypes.to_vec())
                    .unwrap_or_default(),
            );
            let owned: Vec<usize> = match expert_to_rank {
                Some(map) => map
                    .iter()
                    .enumerate()
                    .filter_map(|(global_id, &owner)| (owner as usize == rank).then_some(global_id))
                    .collect(),
                None => (0..ffn.experts.len()).collect(),
            };
            let mut locals = Vec::with_capacity(ffn.experts.len());
            for (local_pos, expert) in ffn.experts.iter().enumerate() {
                locals.push(Qwen35LocalExpertDescriptor {
                    global_expert_id: owned.get(local_pos).copied().unwrap_or(local_pos),
                    gate_up: WeightTensorDescriptor::from_weight(&expert.gate_up),
                    down: WeightTensorDescriptor::from_weight(&expert.down),
                });
            }
            locals.sort_by_key(|descriptor| descriptor.global_expert_id);
            local_expert_descriptors.push(locals);
        }
        let layer_seals = weights
            .layers
            .iter()
            .map(|l| match l {
                LayerWeights::DeltaNet(w) => Qwen35LayerSeal::DeltaNet {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wqkv: WeightTensorDescriptor::from_weight(&w.wqkv),
                    wz: WeightTensorDescriptor::from_weight(&w.wz),
                    w_alpha: WeightTensorDescriptor::from_weight(&w.w_alpha),
                    w_beta: WeightTensorDescriptor::from_weight(&w.w_beta),
                    a_log: GpuTensorDescriptor::from_tensor(&w.a_log),
                    dt_bias: GpuTensorDescriptor::from_tensor(&w.dt_bias),
                    conv_weight: GpuTensorDescriptor::from_tensor(&w.conv_weight),
                    norm_weight: GpuTensorDescriptor::from_tensor(&w.norm_weight),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    w_gate: WeightTensorDescriptor::from_weight(&w.w_gate),
                    w_up: WeightTensorDescriptor::from_weight(&w.w_up),
                    w_down: WeightTensorDescriptor::from_weight(&w.w_down),
                },
                LayerWeights::FullAttn(w) => Qwen35LayerSeal::FullAttn {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wq: WeightTensorDescriptor::from_weight(&w.wq),
                    wk: WeightTensorDescriptor::from_weight(&w.wk),
                    wv: WeightTensorDescriptor::from_weight(&w.wv),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    q_norm: GpuTensorDescriptor::from_tensor(&w.q_norm),
                    k_norm: GpuTensorDescriptor::from_tensor(&w.k_norm),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    w_gate: WeightTensorDescriptor::from_weight(&w.w_gate),
                    w_up: WeightTensorDescriptor::from_weight(&w.w_up),
                    w_down: WeightTensorDescriptor::from_weight(&w.w_down),
                },
                LayerWeights::DeltaNetMoe(w) => {
                    let ffn = w
                        .ffn
                        .as_legacy()
                        .expect("qwen35 rank seal: EP load path builds Legacy MoE storage only");
                    Qwen35LayerSeal::DeltaNetMoe {
                        attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                        wqkv: WeightTensorDescriptor::from_weight(&w.wqkv),
                        wz: WeightTensorDescriptor::from_weight(&w.wz),
                        w_alpha: WeightTensorDescriptor::from_weight(&w.w_alpha),
                        w_beta: WeightTensorDescriptor::from_weight(&w.w_beta),
                        a_log: GpuTensorDescriptor::from_tensor(&w.a_log),
                        dt_bias: GpuTensorDescriptor::from_tensor(&w.dt_bias),
                        conv_weight: GpuTensorDescriptor::from_tensor(&w.conv_weight),
                        norm_weight: GpuTensorDescriptor::from_tensor(&w.norm_weight),
                        wo: WeightTensorDescriptor::from_weight(&w.wo),
                        ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                        moe: Qwen35MoeFfnSeal {
                            router: WeightTensorDescriptor::from_weight(&ffn.router),
                            shared_gate: WeightTensorDescriptor::from_weight(
                                &ffn.shared_expert.gate,
                            ),
                            shared_up: WeightTensorDescriptor::from_weight(&ffn.shared_expert.up),
                            shared_down: WeightTensorDescriptor::from_weight(
                                &ffn.shared_expert.down,
                            ),
                            shared_expert_gate: WeightTensorDescriptor::from_weight(
                                &ffn.shared_expert_gate,
                            ),
                            expert_gate_up_ptrs: GpuTensorDescriptor::from_tensor(
                                &ffn.expert_gate_up_ptrs,
                            ),
                            expert_down_ptrs: GpuTensorDescriptor::from_tensor(
                                &ffn.expert_down_ptrs,
                            ),
                            expert_down_awq_ptrs: ffn
                                .expert_down_awq_ptrs
                                .as_ref()
                                .map(GpuTensorDescriptor::from_tensor),
                            expert_dtype_tags: ffn
                                .expert_dtype_tags
                                .as_ref()
                                .map(GpuTensorDescriptor::from_tensor),
                            layer_idx: ffn.layer_idx,
                            has_packed_owners: ffn.packed_expert_owners.is_some(),
                            global_expert_dtypes: ffn
                                .global_expert_dtypes
                                .as_ref()
                                .map(|b| b.to_vec()),
                            num_local_experts: ffn.experts.len(),
                        },
                    }
                }
                LayerWeights::FullAttnMoe(w) => {
                    let ffn = w
                        .ffn
                        .as_legacy()
                        .expect("qwen35 rank seal: EP load path builds Legacy MoE storage only");
                    Qwen35LayerSeal::FullAttnMoe {
                        attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                        wq: WeightTensorDescriptor::from_weight(&w.wq),
                        wk: WeightTensorDescriptor::from_weight(&w.wk),
                        wv: WeightTensorDescriptor::from_weight(&w.wv),
                        wo: WeightTensorDescriptor::from_weight(&w.wo),
                        q_norm: GpuTensorDescriptor::from_tensor(&w.q_norm),
                        k_norm: GpuTensorDescriptor::from_tensor(&w.k_norm),
                        ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                        moe: Qwen35MoeFfnSeal {
                            router: WeightTensorDescriptor::from_weight(&ffn.router),
                            shared_gate: WeightTensorDescriptor::from_weight(
                                &ffn.shared_expert.gate,
                            ),
                            shared_up: WeightTensorDescriptor::from_weight(&ffn.shared_expert.up),
                            shared_down: WeightTensorDescriptor::from_weight(
                                &ffn.shared_expert.down,
                            ),
                            shared_expert_gate: WeightTensorDescriptor::from_weight(
                                &ffn.shared_expert_gate,
                            ),
                            expert_gate_up_ptrs: GpuTensorDescriptor::from_tensor(
                                &ffn.expert_gate_up_ptrs,
                            ),
                            expert_down_ptrs: GpuTensorDescriptor::from_tensor(
                                &ffn.expert_down_ptrs,
                            ),
                            expert_down_awq_ptrs: ffn
                                .expert_down_awq_ptrs
                                .as_ref()
                                .map(GpuTensorDescriptor::from_tensor),
                            expert_dtype_tags: ffn
                                .expert_dtype_tags
                                .as_ref()
                                .map(GpuTensorDescriptor::from_tensor),
                            layer_idx: ffn.layer_idx,
                            has_packed_owners: ffn.packed_expert_owners.is_some(),
                            global_expert_dtypes: ffn
                                .global_expert_dtypes
                                .as_ref()
                                .map(|b| b.to_vec()),
                            num_local_experts: ffn.experts.len(),
                        },
                    }
                }
            })
            .collect();
        Self {
            token_embd: GpuTensorDescriptor::from_tensor(&weights.token_embd),
            embd_format: weights.embd_format,
            output_norm: GpuTensorDescriptor::from_tensor(&weights.output_norm),
            output: WeightTensorDescriptor::from_weight(&weights.output),
            moe_has_mq6: weights.moe_has_mq6,
            has_pager: weights.pager.is_some(),
            lm_head_aliases_embd: weights.lm_head_aliases_embd,
            layer_seals,
            global_expert_dtypes,
            local_expert_descriptors,
        }
    }
    pub fn matches_config(&self, other: &Self) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35EpShardInfo {
    pub(crate) rank: u8,
    pub(crate) rank_count: u8,
    pub(crate) expert_to_rank: Box<[u8]>,
    pub device_id: i32,
    pub source_identity: std::sync::Arc<Qwen35HfqSourceIdentity>,
    pub config_fingerprint: Qwen35EpConfigFingerprint,
    pub rank_seal: Qwen35RankSeal,
}

impl Qwen35EpShardInfo {
    /// Owning rank for this shard (0 <= rank < rank_count).
    pub fn rank(&self) -> u8 {
        self.rank
    }
    /// Total number of ranks in the EP group (exactly 4 for the MQ4R route).
    pub fn rank_count(&self) -> u8 {
        self.rank_count
    }
    /// Global expert → owning rank map (`len == config.num_experts`, each entry < rank_count).
    pub fn expert_to_rank(&self) -> &[u8] {
        &self.expert_to_rank
    }
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    pub fn source_identity(&self) -> &Qwen35HfqSourceIdentity {
        &self.source_identity
    }
    pub fn config_fingerprint(&self) -> &Qwen35EpConfigFingerprint {
        &self.config_fingerprint
    }
    pub fn rank_seal(&self) -> &Qwen35RankSeal {
        &self.rank_seal
    }
}
/// Transactional pending owner for EP `load_moe_ffn`. Allocations publish only on commit;
/// any failure rolls back every populated field on the owner device with sync/first-error preservation.
pub(crate) struct PendingEpMoeFfn {
    pub(crate) router: Option<WeightTensor>,
    pub(crate) shared_gate: Option<WeightTensor>,
    pub(crate) shared_up: Option<WeightTensor>,
    pub(crate) shared_down: Option<WeightTensor>,
    pub(crate) shared_gate_scalar: Option<WeightTensor>,
    pub(crate) experts: Vec<ExpertWeights>,
    pub(crate) packed_owners: Option<PackedExpertOwners>,
    pub(crate) dummy_buffers: Vec<GpuTensor>,
    pub(crate) gate_up_ptrs: Option<GpuTensor>,
    pub(crate) down_ptrs: Option<GpuTensor>,
    pub(crate) awq_ptrs: Option<GpuTensor>,
    pub(crate) dtype_tags: Option<GpuTensor>,
    pub(crate) global_dtypes: Option<Box<[(DType, DType)]>>,
    pub(crate) layer_idx: u16,
}

impl PendingEpMoeFfn {
    pub(crate) fn new(layer_idx: u16) -> Self {
        Self {
            router: None,
            shared_gate: None,
            shared_up: None,
            shared_down: None,
            shared_gate_scalar: None,
            experts: Vec::new(),
            packed_owners: None,
            dummy_buffers: Vec::new(),
            gate_up_ptrs: None,
            down_ptrs: None,
            awq_ptrs: None,
            dtype_tags: None,
            global_dtypes: None,
            layer_idx,
        }
    }
    /// Roll back every populated field on the owner device. Binds + synchronizes the
    /// owner, attempts every free, preserves the initiating error as primary and attaches
    /// the first cleanup failure as context. Returns the enriched error.
    pub(crate) fn rollback(mut self, gpu: &mut Gpu, err: HipError) -> HipError {
        let mut first_cleanup: Option<HipError> = None;
        let mut record_cleanup = |e: HipError| {
            if first_cleanup.is_none() {
                first_cleanup = Some(e);
            }
        };
        let _ = gpu.bind_thread();
        let _ = gpu.hip.device_synchronize();
        if let Some(t) = self.dtype_tags.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        if let Some(t) = self.awq_ptrs.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        if let Some(t) = self.down_ptrs.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        if let Some(t) = self.gate_up_ptrs.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        for d in self.dummy_buffers.drain(..) {
            if let Err(e) = gpu.free_tensor(d) {
                record_cleanup(e);
            }
        }
        if let Some(owners) = self.packed_owners.take() {
            for e in self.experts.drain(..) {
                free_weight_metadata_only(gpu, e.gate_up);
                free_weight_metadata_only(gpu, e.down);
            }
            if let Err(e) = gpu.free_tensor(owners.gate_up) {
                record_cleanup(e);
            }
            if let Err(e) = gpu.free_tensor(owners.down) {
                record_cleanup(e);
            }
        } else {
            for e in self.experts.drain(..) {
                if let Some(e1) = free_weight_checked(gpu, e.gate_up) {
                    record_cleanup(e1);
                }
                if let Some(e2) = free_weight_checked(gpu, e.down) {
                    record_cleanup(e2);
                }
            }
        }
        if let Some(w) = self.shared_gate_scalar.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.shared_down.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.shared_up.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.shared_gate.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.router.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(cleanup) = first_cleanup {
            HipError::new(
                0,
                &format!("{} (cleanup: {})", err.message, cleanup.message),
            )
        } else {
            err
        }
    }
    pub(crate) fn commit(
        self,
        shared_expert: SharedExpertWeights,
        gate_up_ptrs: GpuTensor,
        down_ptrs: GpuTensor,
        awq_ptrs: Option<GpuTensor>,
        dtype_tags: Option<GpuTensor>,
    ) -> MoeFfnWeights {
        let router = self.router.expect("pending commit: router missing");
        let shared_gate_scalar = self
            .shared_gate_scalar
            .expect("pending commit: shared_gate_scalar missing");
        MoeFfnWeights {
            router,
            experts: self.experts,
            packed_expert_owners: self.packed_owners,
            shared_expert,
            shared_expert_gate: shared_gate_scalar,
            expert_gate_up_ptrs: gate_up_ptrs,
            expert_down_ptrs: down_ptrs,
            expert_down_awq_ptrs: awq_ptrs,
            expert_dtype_tags: dtype_tags,
            layer_idx: self.layer_idx,
            expert_shape: None,
            paro_shared: None,
            global_expert_dtypes: self.global_dtypes,
            ep_dummy_buffers: self.dummy_buffers,
        }
    }
}

/// Internal checked free for a WeightTensor: attempts every sidecar and buffer free,
/// returns the first HipError if any, otherwise None. Non-public; used only by Pending rollback.
fn free_weight_checked(gpu: &mut Gpu, w: WeightTensor) -> Option<HipError> {
    let mut first: Option<HipError> = None;
    let mut record = |e: HipError| {
        if first.is_none() {
            first = Some(e);
        }
    };
    if let Some(paro) = w.paro {
        if !paro.is_alias {
            if let Err(e) = gpu.free_tensor(paro.pairs) {
                record(e);
            }
            if let Err(e) = gpu.free_tensor(paro.theta) {
                record(e);
            }
            if let Err(e) = gpu.free_tensor(paro.channel_scales) {
                record(e);
            }
        }
    }
    if let Some(awq) = w.awq_scale {
        if let Err(e) = gpu.free_tensor(awq) {
            record(e);
        }
    }
    if let Err(e) = gpu.free_tensor(w.buf) {
        record(e);
    }
    first
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    /// True when any MoE FFN projection in the loaded model is MQ6. gfx1151's
    /// grouped-i8 MQ4 shortcut is model-level unsafe for these promoted A3B
    /// checkpoints, even in layers whose local routed experts remain MQ4.
    pub moe_has_mq6: bool,

    /// Weight pager (MAD-93 v0.1). `Some` only when the model was loaded
    /// with `Qwen35Config::paged_experts == true`. The forward path uses
    /// interior mutability (`borrow_mut`) at the MoE dispatch site to call
    /// `ensure_resident` / `patch_expert_ptr_table`. `None` means the model
    /// is fully resident — no behavior change vs main.
    pub pager: Option<std::cell::RefCell<hipfire_runtime::weight_pager::WeightPager>>,

    /// True when the tied lm_head aliases the embedding table buffer
    /// (single-GPU path). When true, `output.buf` is a non-owning view of
    /// `token_embd.buf` and must NOT be freed in `free_gpu`.
    pub lm_head_aliases_embd: bool,
    /// EP shard provenance, if loaded via `load_weights_ep_rank`.  `None`
    /// on every ordinary single-GPU/TP/paged load.  CPU-side (source
    /// identity, config fingerprint, rank seals — no GPU ownership):
    /// attached by the EP loader after a complete successful load and
    /// dropped with the weights; no teardown involvement.
    pub(crate) ep_shard: Option<Qwen35EpShardInfo>,

    // ── Lane 2b: Frozen MoE resident (device-mesh) ────────────────
    /// Optional resident owner for Frozen MoE storage.
    ///
    /// * `None` — all MoE layers are `MoeFfnStorage::Legacy`
    ///   (today's behavior).
    /// * `Some(resident)` — all MoE layers are `MoeFfnStorage::Frozen`
    ///   and the resident owns their GPU allocations.
    ///
    /// Initialized to `None` in every constructor.  Set by the device-mesh
    /// loader after publication.
    ///
    /// # Invariant
    ///
    /// `is_some()` ⇔ every MoE layer's `ffn` is `MoeFfnStorage::Frozen`.
    /// Enforced by [`crate::store::validate_moe_pairing`] at publication seams.
    pub(crate) moe_resident: Option<Qwen35MoeResident>,

    /// STEP-002 Task 9: model-owned, immutable per-layer expert-group plans,
    /// resolved ONCE through the validated manifest authority
    /// ([`Qwen35MoeGroupPlans::resolve`]) and borrowed by layer during
    /// decode/prefill — never reconstructed per token. The cell is a
    /// write-once `OnceLock` because the config-less `load_weights`
    /// orchestrator path cannot resolve at construction; every construction
    /// site initializes the cell empty and the first forward resolves
    /// through the authority (see [`Qwen35Weights::moe_group_plans`]). The
    /// entry is KEYED by the config identity consumed by resolution
    /// ([`Qwen35MoeGroupPlanKey`]): the same identity borrows the cached
    /// success/failure; a different identity is refused explicitly — never
    /// silent stale reuse.
    pub(crate) moe_group_plans: std::sync::OnceLock<Qwen35MoeGroupPlansCacheEntry>,
}

impl Qwen35Weights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        // Lane 2b: hard-refuse when a Frozen MoE resident is present.
        // The resident must be freed independently before calling free_gpu.
        if self.moe_resident.is_some() {
            panic!("free_gpu: moe_resident is present; free the resident separately before calling free_gpu");
        }
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        if !self.lm_head_aliases_embd {
            self.output.free_all(gpu);
        }
        for layer in self.layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_storage(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_storage(gpu, l.ffn);
                }
            }
        }
        // MAD-93 v0.1: in paged mode, the pager owns expert weight allocations
        // (the per-layer `free_moe_ffn` loops ran no-ops since `ffn.experts`
        // was empty). Drain the pager's resident set back to the GPU pool here.
        if let Some(pager_cell) = self.pager {
            pager_cell.into_inner().free_all(gpu);
        }
    }

    /// Multi-GPU companion to `free_gpu`. Each layer freed on its
    /// band-owning device per `gpus.device_for_layer(i)`; `token_embd`
    /// freed on dev 0; `output_norm + output` on `gpus.output_device`.
    /// Mirror of `load_weights_multi` placement. The `pager` field is
    /// always `None` on the multi path (paged-experts is not wired into
    /// pp>1 yet); a non-None pager would need its own per-band drain
    /// strategy and is rejected at load.
    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        // Lane 2b: hard-refuse Frozen MoE (multi-device path stays Legacy-only).
        if self.moe_resident.is_some() {
            panic!("free_gpu_multi: moe_resident is present; multi-device Frozen is unsupported");
        }
        for layer in &self.layers {
            match layer {
                LayerWeights::DeltaNetMoe(l) if l.ffn.is_frozen() => {
                    panic!("free_gpu_multi: Frozen DeltaNetMoe storage is unsupported in multi-device path");
                }
                LayerWeights::FullAttnMoe(l) if l.ffn.is_frozen() => {
                    panic!("free_gpu_multi: Frozen FullAttnMoe storage is unsupported in multi-device path");
                }
                _ => {}
            }
        }
        debug_assert!(
            self.pager.is_none(),
            "free_gpu_multi: pager must be None on pp>1 path"
        );
        let _ = gpus.devices[0].free_tensor(self.token_embd);
        let out_dev = gpus.output_device;
        let _ = gpus.devices[out_dev].free_tensor(self.output_norm);
        self.output.free_all(&mut gpus.devices[out_dev]);
        for (i, layer) in self.layers.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let gpu = &mut gpus.devices[dev_idx];
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_storage(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_storage(gpu, l.ffn);
                }
            }
        }
    }
    /// Exact-retention checked GPU cleanup.  Consumes `self`, attempts every
    /// owned weight even after failures, retains the exact original
    /// `GpuTensor` on failure, and returns only the failures.
    ///
    /// Uses `Gpu::free_tensor_checked(&mut Option<GpuTensor>)` everywhere so
    /// that on bind/driver failure the tensor ownership is preserved for
    /// retry.
    ///
    /// Frozen MoE storage (both per-layer markers and the optional resident)
    /// is freed through the resident's `free_checked` path; failures are
    /// aggregated into the returned [`GpuCleanupFailure`].
    ///
    /// The pager is freed as before (unchecked, since it owns no individual
    /// tensors at this level).
    pub fn free_gpu_checked(self, gpu: &mut Gpu) -> Result<(), GpuCleanupFailure> {
        let mut failures: Vec<RetainedGpuTensor> = Vec::new();

        // ── Top-level tensors ───────────────────────────────────────────
        free_tensor_retained("token_embd", self.token_embd, gpu, &mut failures);
        free_tensor_retained("output_norm", self.output_norm, gpu, &mut failures);

        // Output / LM head: skip buf when aliased.
        if self.lm_head_aliases_embd {
            free_weight_sidecars_checked("output", self.output, gpu, &mut failures);
        } else {
            free_weight_all_checked("output", self.output, gpu, &mut failures);
        }

        // ── Per-layer weights ───────────────────────────────────────────
        for (i, layer) in self.layers.into_iter().enumerate() {
            let lp = |field: &str| format!("layers[{i}].{field}");
            match layer {
                LayerWeights::DeltaNet(l) => {
                    free_tensor_retained(lp("attn_norm"), l.attn_norm, gpu, &mut failures);
                    free_weight_all_checked(&lp("wqkv"), l.wqkv, gpu, &mut failures);
                    free_weight_all_checked(&lp("wz"), l.wz, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_alpha"), l.w_alpha, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_beta"), l.w_beta, gpu, &mut failures);
                    free_tensor_retained(lp("a_log"), l.a_log, gpu, &mut failures);
                    free_tensor_retained(lp("dt_bias"), l.dt_bias, gpu, &mut failures);
                    free_tensor_retained(lp("conv_weight"), l.conv_weight, gpu, &mut failures);
                    free_tensor_retained(lp("norm_weight"), l.norm_weight, gpu, &mut failures);
                    free_weight_all_checked(&lp("wo"), l.wo, gpu, &mut failures);
                    free_tensor_retained(lp("ffn_norm"), l.ffn_norm, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_gate"), l.w_gate, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_up"), l.w_up, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_down"), l.w_down, gpu, &mut failures);
                }
                LayerWeights::FullAttn(l) => {
                    free_tensor_retained(lp("attn_norm"), l.attn_norm, gpu, &mut failures);
                    free_weight_all_checked(&lp("wq"), l.wq, gpu, &mut failures);
                    free_weight_all_checked(&lp("wk"), l.wk, gpu, &mut failures);
                    free_weight_all_checked(&lp("wv"), l.wv, gpu, &mut failures);
                    free_weight_all_checked(&lp("wo"), l.wo, gpu, &mut failures);
                    free_tensor_retained(lp("q_norm"), l.q_norm, gpu, &mut failures);
                    free_tensor_retained(lp("k_norm"), l.k_norm, gpu, &mut failures);
                    free_tensor_retained(lp("ffn_norm"), l.ffn_norm, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_gate"), l.w_gate, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_up"), l.w_up, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_down"), l.w_down, gpu, &mut failures);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    free_tensor_retained(lp("attn_norm"), l.attn_norm, gpu, &mut failures);
                    free_weight_all_checked(&lp("wqkv"), l.wqkv, gpu, &mut failures);
                    free_weight_all_checked(&lp("wz"), l.wz, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_alpha"), l.w_alpha, gpu, &mut failures);
                    free_weight_all_checked(&lp("w_beta"), l.w_beta, gpu, &mut failures);
                    free_tensor_retained(lp("a_log"), l.a_log, gpu, &mut failures);
                    free_tensor_retained(lp("dt_bias"), l.dt_bias, gpu, &mut failures);
                    free_tensor_retained(lp("conv_weight"), l.conv_weight, gpu, &mut failures);
                    free_tensor_retained(lp("norm_weight"), l.norm_weight, gpu, &mut failures);
                    free_weight_all_checked(&lp("wo"), l.wo, gpu, &mut failures);
                    free_tensor_retained(lp("ffn_norm"), l.ffn_norm, gpu, &mut failures);
                    match l.ffn {
                        MoeFfnStorage::Legacy(ffn) => {
                            free_moe_ffn_checked(&lp("ffn"), ffn, gpu, &mut failures);
                        }
                        MoeFfnStorage::Frozen => {
                            // Frozen marker: the resident owns the GPU
                            // allocations. Nothing to free here.
                        }
                    }
                }
                LayerWeights::FullAttnMoe(l) => {
                    free_tensor_retained(lp("attn_norm"), l.attn_norm, gpu, &mut failures);
                    free_weight_all_checked(&lp("wq"), l.wq, gpu, &mut failures);
                    free_weight_all_checked(&lp("wk"), l.wk, gpu, &mut failures);
                    free_weight_all_checked(&lp("wv"), l.wv, gpu, &mut failures);
                    free_weight_all_checked(&lp("wo"), l.wo, gpu, &mut failures);
                    free_tensor_retained(lp("q_norm"), l.q_norm, gpu, &mut failures);
                    free_tensor_retained(lp("k_norm"), l.k_norm, gpu, &mut failures);
                    free_tensor_retained(lp("ffn_norm"), l.ffn_norm, gpu, &mut failures);
                    match l.ffn {
                        MoeFfnStorage::Legacy(ffn) => {
                            free_moe_ffn_checked(&lp("ffn"), ffn, gpu, &mut failures);
                        }
                        MoeFfnStorage::Frozen => {
                            // Frozen marker: the resident owns the GPU
                            // allocations. Nothing to free here.
                        }
                    }
                }
            }
        }

        // ── Frozen MoE resident ─────────────────────────────────────────
        // Non-tensor owner category: kept whole (never flattened), retried
        // via the RetryableOwner boxed path.
        let mut frozen_failures: Vec<Box<dyn RetryableOwner>> = Vec::new();
        if let Some(resident) = self.moe_resident {
            if let Err(f) = resident.free_checked(gpu) {
                frozen_failures.push(Box::new(f));
            }
        }

        // ── Pager ───────────────────────────────────────────────────────
        if let Some(pager_cell) = self.pager {
            pager_cell.into_inner().free_all(gpu);
        }

        // Drop metadata fields (Copy types just need mentioning).
        let _ = self.embd_format;
        let _ = self.moe_has_mq6;
        let _ = self.lm_head_aliases_embd;

        if failures.is_empty() && frozen_failures.is_empty() {
            Ok(())
        } else {
            Err(GpuCleanupFailure {
                failed_tensors: failures,
                other: frozen_failures,
            })
        }
    }
}
impl Qwen35Weights {
    /// Immutable EP shard provenance, if loaded via `load_weights_ep_rank`.
    /// `None` on every ordinary single-GPU/TP/paged load.
    pub fn ep_shard(&self) -> Option<&Qwen35EpShardInfo> {
        self.ep_shard.as_ref()
    }
}

impl Qwen35Weights {
    pub(crate) fn moe_group_plans(
        &self,
        config: &Qwen35Config,
    ) -> Result<&Qwen35MoeGroupPlans, String> {
        Self::moe_group_plans_with(&self.moe_group_plans, config, Qwen35MoeGroupPlans::resolve)
    }

    /// The cache protocol behind [`Qwen35Weights::moe_group_plans`], generic
    /// over the resolver so the exactly-once contract is testable with a
    /// counting resolver through a private seam (no globals, no timing).
    /// The returned borrow is tied to the CELL ('a), never to the config.
    fn moe_group_plans_with<'a, 'b>(
        cell: &'a std::sync::OnceLock<Qwen35MoeGroupPlansCacheEntry>,
        config: &'b Qwen35Config,
        resolve: impl FnOnce(&'b Qwen35Config) -> Result<Qwen35MoeGroupPlans, String>,
    ) -> Result<&'a Qwen35MoeGroupPlans, String> {
        // Zero-alloc hot-path request identity: borrows the config's
        // layer_types slice — the OWNED key (which clones the Vec) is built
        // only inside the initializer below (and on the cold mismatch path).
        let request = Qwen35MoeGroupPlanKeyRef::from_config(config);
        // Resolve AT MOST ONCE per model: `get_or_init` runs the initializer
        // exactly once — the winning caller's config identity is resolved
        // and stored; every other caller (concurrent or later) blocks or
        // returns and receives the SAME stored entry, so there is no stale
        // result, retry, replacement, or discarded second resolution. The
        // complete five-field key is compared with the request below before
        // any result is returned.
        let entry = cell.get_or_init(|| Qwen35MoeGroupPlansCacheEntry {
            key: Qwen35MoeGroupPlanKey::from_config(config),
            result: resolve(config),
        });
        if entry.key == request {
            entry.result.as_ref().map_err(|error| error.clone())
        } else {
            // Cold path: constructing the owned key here is acceptable — the
            // mismatch is a programmer error, never the hot path.
            let requested = Qwen35MoeGroupPlanKey::from_config(config);
            Err(format!(
                "qwen35 plan cache: config identity mismatch — cached {entry_key} but requested \
                 {requested_key}; refusing silent stale reuse",
                entry_key = entry.key,
                requested_key = requested,
            ))
        }
    }
}

impl Qwen35Weights {
    /// Central view constructor: select MoE FFN storage for a global model
    /// layer index and pair it with the optional resident (Frozen path).
    ///
    /// Returns `MoeFfnView::Legacy` or `MoeFfnView::Frozen` based on the
    /// layer's storage variant.  Errors on:
    /// * Non-MoE layer at `layer_idx` (the layer doesn't have an FFN).
    /// * `Frozen` storage without a resident set.
    /// * `Frozen` storage where `resident.bind_layer` fails (OOB / store
    ///   corruption).
    ///
    /// O(1) — no iteration over experts.
    pub(crate) fn moe_ffn_view(
        &self,
        layer_idx: usize,
    ) -> Result<MoeFfnView<'_>, Qwen35MoeBindError> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(Qwen35MoeBindError::LayerOutOfRange {
                requested: layer_idx,
                count: self.layers.len(),
            })?;
        let storage = match layer {
            LayerWeights::DeltaNetMoe(l) => &l.ffn,
            LayerWeights::FullAttnMoe(l) => &l.ffn,
            _ => {
                return Err(Qwen35MoeBindError::LayerOutOfRange {
                    requested: layer_idx,
                    count: self.layers.len(),
                });
            }
        };
        match storage {
            MoeFfnStorage::Legacy(ffn) => Ok(MoeFfnView::Legacy(ffn)),
            MoeFfnStorage::Frozen => {
                let resident = self.moe_resident.as_ref().ok_or_else(|| {
                    Qwen35MoeBindError::TensorLookup(
                        "moe_resident".into(),
                        hipfire_runtime::weight_store::WeightCellLookupError::InvalidSlot,
                    )
                })?;
                let bindings = resident.bind_layer(layer_idx)?;
                Ok(MoeFfnView::Frozen(bindings))
            }
        }
    }

    /// Emulated EP2 view seam (test-only harness, feature
    /// `emulated-ep2-harness`): like [`Self::moe_ffn_view`] but the Frozen
    /// bindings carry the rank-masked gate-up pointer-table override.
    ///
    /// * Out-of-range layer index → [`Qwen35MoeBindError::LayerOutOfRange`].
    /// * Legacy (owned) storage → refused — EP2 rank masking exists only for
    ///   Frozen storage, and silently returning an unmasked view would fake
    ///   the partition evidence.
    /// * Frozen storage without a resident → `TensorLookup`.
    ///
    /// There is NO sequential GPU execution here — this is the binding
    /// surface only (Phase 2A); the harness driver consumes it in Phase 2B.
    #[cfg(feature = "emulated-ep2-harness")]
    pub(crate) fn moe_ffn_view_ep2(
        &self,
        layer_idx: usize,
        rank: usize,
    ) -> Result<MoeFfnView<'_>, Qwen35MoeBindError> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(Qwen35MoeBindError::LayerOutOfRange {
                requested: layer_idx,
                count: self.layers.len(),
            })?;
        let storage = match layer {
            LayerWeights::DeltaNetMoe(l) => &l.ffn,
            LayerWeights::FullAttnMoe(l) => &l.ffn,
            _ => {
                return Err(Qwen35MoeBindError::LayerOutOfRange {
                    requested: layer_idx,
                    count: self.layers.len(),
                });
            }
        };
        match storage {
            MoeFfnStorage::Legacy(_) => Err(Qwen35MoeBindError::Ep2RequiresFrozenStorage),
            MoeFfnStorage::Frozen => {
                let resident = self.moe_resident.as_ref().ok_or_else(|| {
                    Qwen35MoeBindError::TensorLookup(
                        "moe_resident".into(),
                        hipfire_runtime::weight_store::WeightCellLookupError::InvalidSlot,
                    )
                })?;
                let bindings = resident.bind_layer_ep2(layer_idx, rank)?;
                Ok(MoeFfnView::Frozen(bindings))
            }
        }
    }

    /// Infallible metadata-only view for the MoE FFN at `layer_idx`.
    ///
    /// ## Invariant
    ///
    /// This must only be called when the pairing invariant holds:
    /// `validate_moe_pairing` has passed.  Under that invariant:
    /// * Legacy MoE layers always have `MoeFfnStorage::Legacy`.
    /// * Frozen MoE layers always have `MoeFfnStorage::Frozen` AND
    ///   `moe_resident` is `Some` AND `resident.layer_metadata(layer_idx)`
    ///   succeeds AND the returned projection's `layer_idx` equals
    ///   `layer_idx`.
    ///
    /// If the invariant is violated, this method uses a single `expect`
    /// per failure case.  This is acceptable because the invariant is
    /// constructor-proven (publication seam check), not runtime-dependent.
    pub(crate) fn moe_ffn_metadata_view(
        &self,
        layer_idx: usize,
    ) -> crate::store::MoeFfnMetaView<'_> {
        let layer = self
            .layers
            .get(layer_idx)
            .expect("moe_ffn_metadata_view: layer_idx OOB");
        match layer {
            LayerWeights::DeltaNetMoe(l) => match &l.ffn {
                MoeFfnStorage::Legacy(ffn) => crate::store::MoeFfnMetaView::Legacy(ffn),
                MoeFfnStorage::Frozen => {
                    let resident = self
                        .moe_resident
                        .as_ref()
                        .expect("moe_ffn_metadata_view: Frozen layer without resident");
                    let proj = resident
                        .layer_metadata(layer_idx)
                        .expect("moe_ffn_metadata_view: resident missing projection for layer_idx");
                    debug_assert_eq!(proj.layer_idx, layer_idx);
                    crate::store::MoeFfnMetaView::Frozen(proj)
                }
            },
            LayerWeights::FullAttnMoe(l) => match &l.ffn {
                MoeFfnStorage::Legacy(ffn) => crate::store::MoeFfnMetaView::Legacy(ffn),
                MoeFfnStorage::Frozen => {
                    let resident = self
                        .moe_resident
                        .as_ref()
                        .expect("moe_ffn_metadata_view: Frozen layer without resident");
                    let proj = resident
                        .layer_metadata(layer_idx)
                        .expect("moe_ffn_metadata_view: resident missing projection for layer_idx");
                    debug_assert_eq!(proj.layer_idx, layer_idx);
                    crate::store::MoeFfnMetaView::Frozen(proj)
                }
            },
            _ => panic!("moe_ffn_metadata_view: layer {layer_idx} is not an MoE layer"),
        }
    }
}

/// Reject Frozen MoE storage in multi-device / PP / TP / EP paths.
/// Must be called before any operation in `forward_scratch_multi`,
/// `forward_prefill_batch_multi`, and `forward_scratch_layers_multi`.
pub(crate) fn reject_frozen_multi(site: &str, weights: &Qwen35Weights) -> HipResult<()> {
    if weights.moe_resident.is_some() {
        return Err(HipError::new(
            0,
            &format!("{site}: Frozen MoE resident present, multi-device path requires Legacy"),
        ));
    }
    for layer in &weights.layers {
        match layer {
            LayerWeights::DeltaNetMoe(l) if l.ffn.is_frozen() => {
                return Err(HipError::new(
                    0,
                    &format!("{site}: Frozen MoE storage in DeltaNetMoe layer"),
                ));
            }
            LayerWeights::FullAttnMoe(l) if l.ffn.is_frozen() => {
                return Err(HipError::new(
                    0,
                    &format!("{site}: Frozen MoE storage in FullAttnMoe layer"),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn frozen_eligible(config: &Qwen35Config) -> bool {
    if config.num_experts == 0 {
        return false;
    }
    // Every layer must be MoE-capable.
    if !config
        .layer_types
        .iter()
        .all(|t| matches!(t, LayerType::LinearAttention | LayerType::FullAttention))
    {
        return false;
    }
    // Additional eligibility gates can be added here:
    // - Paro quant rejection
    // - Paged expert rejection
    // - A3B routing variant rejection
    true
}

// ── MoE expert-group plan machinery (STEP-002 Task 9) ───────────────────

pub(crate) struct Qwen35MoeGroupPlanKey {
    n_layers: usize,
    layer_types: Vec<LayerType>,
    num_experts: usize,
    dim: usize,
    moe_intermediate_size: usize,
}

impl Qwen35MoeGroupPlanKey {
    fn from_config(cfg: &Qwen35Config) -> Self {
        #[cfg(test)]
        if plan_key_seam::INSTRUMENT.load(std::sync::atomic::Ordering::Relaxed) {
            plan_key_seam::CONSTRUCTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Self {
            n_layers: cfg.n_layers,
            layer_types: cfg.layer_types.clone(),
            num_experts: cfg.num_experts,
            dim: cfg.dim,
            moe_intermediate_size: cfg.moe_intermediate_size,
        }
    }
}

impl std::fmt::Display for Qwen35MoeGroupPlanKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n_layers={}, layer_types={:?}, num_experts={}, dim={}, moe_intermediate_size={}",
            self.n_layers, self.layer_types, self.num_experts, self.dim, self.moe_intermediate_size
        )
    }
}

/// Borrowed view of the config identity consumed by plan resolution — the
/// hot-path comparison side. Constructing this performs ZERO heap
/// allocation (the `layer_types` slice is borrowed, never cloned); the
/// owned [`Qwen35MoeGroupPlanKey`] is built only inside the `get_or_init`
/// initializer and on the cold mismatch-formatting path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35MoeGroupPlanKeyRef<'a> {
    n_layers: usize,
    layer_types: &'a [LayerType],
    num_experts: usize,
    dim: usize,
    moe_intermediate_size: usize,
}

impl<'a> Qwen35MoeGroupPlanKeyRef<'a> {
    fn from_config(cfg: &'a Qwen35Config) -> Self {
        Self {
            n_layers: cfg.n_layers,
            layer_types: &cfg.layer_types,
            num_experts: cfg.num_experts,
            dim: cfg.dim,
            moe_intermediate_size: cfg.moe_intermediate_size,
        }
    }
}

impl PartialEq<Qwen35MoeGroupPlanKeyRef<'_>> for Qwen35MoeGroupPlanKey {
    /// Complete five-field identity comparison against a borrowed request.
    /// The `layer_types` comparison is a slice comparison over the ORIGINAL
    /// sequence — O(n_layers), no allocation.
    fn eq(&self, other: &Qwen35MoeGroupPlanKeyRef<'_>) -> bool {
        self.n_layers == other.n_layers
            && self.layer_types.as_slice() == other.layer_types
            && self.num_experts == other.num_experts
            && self.dim == other.dim
            && self.moe_intermediate_size == other.moe_intermediate_size
    }
}

/// Test-only instrumentation for the plan-key construction seam: proves the
/// initialized same-key lookup never constructs the OWNED key again (which
/// would clone `layer_types`). Gated so unrelated tests never observe it;
/// zero production overhead (`cfg(test)` only).
#[cfg(test)]
pub(crate) mod plan_key_seam {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Mutex, MutexGuard};

    /// Serializes seam tests so their delta assertions cannot observe each
    /// other's increments.
    pub static LOCK: Mutex<()> = Mutex::new(());
    /// When set, every [`super::Qwen35MoeGroupPlanKey::from_config`] call
    /// increments [`CONSTRUCTIONS`].
    pub static INSTRUMENT: AtomicBool = AtomicBool::new(false);
    /// Number of owned-key constructions performed while instrumented.
    pub static CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);

    /// Reset the instrumentation state (call while holding [`LOCK`]).
    pub fn reset() {
        INSTRUMENT.store(false, Ordering::Relaxed);
        CONSTRUCTIONS.store(0, Ordering::Relaxed);
    }

    /// RAII guard: enables the counter for its lifetime (holding [`LOCK`]
    /// so delta assertions are race-free) and restores it on drop.
    pub struct SeamGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl SeamGuard {
        pub fn on() -> Self {
            let _lock = LOCK.lock().unwrap();
            reset();
            INSTRUMENT.store(true, Ordering::Relaxed);
            SeamGuard { _lock }
        }
    }

    impl Drop for SeamGuard {
        fn drop(&mut self) {
            reset();
        }
    }
}

/// One model-owned plan-cache entry: the config identity the plans were
/// resolved under plus the cached success/failure result. The cell is
/// write-once (thread-safe first-wins); a later access with a DIFFERENT
/// config identity returns an explicit mismatch WITHOUT replacing or
/// recomputing — never silent stale reuse. Failures are cached too
/// (reported once, replayed verbatim for the same identity).
pub(crate) struct Qwen35MoeGroupPlansCacheEntry {
    key: Qwen35MoeGroupPlanKey,
    result: Result<Qwen35MoeGroupPlans, String>,
}

/// Model-owned, immutable per-layer expert-group plans (STEP-002 Task 9,
/// Phase 3 remediation).
///
/// Resolved EXACTLY ONCE through the validated manifest authority
/// ([`hipfire_runtime::weight_manifest::resolve_expert_group_plans`] over
/// [`crate::arch::Qwen35::weight_manifest`] + the Single-policy declaration)
/// and borrowed by layer during decode/prefill — never reconstructed per
/// token. The decode/prefill hot paths resolve through
/// [`Qwen35Weights::moe_group_plans`] once per forward, then borrow
/// [`by_layer`](Self::by_layer).
///
/// The cache is keyed by the config identity consumed by the resolution
/// authority ([`Qwen35MoeGroupPlanKey`]): the same identity borrows the
/// cached success or failure; a different identity is refused explicitly.
/// The cell is a `OnceLock` because the config-less `load_weights`
/// orchestrator path cannot resolve at construction; every construction
/// site initializes the cell empty and the first forward resolves through
/// the validated authority (never a fabricated plan). The state is
/// per-model (not a global cache) and immutable after resolution.
#[derive(Clone, Debug)]
pub(crate) struct Qwen35MoeGroupPlans {
    plans: Vec<hipfire_runtime::weight_manifest::ExpertGroupPlan>,
}

impl Qwen35MoeGroupPlans {
    /// Resolve one validated plan per MoE layer through the manifest
    /// authority. Dense (non-MoE) configs resolve an empty set.
    pub(crate) fn resolve(config: &Qwen35Config) -> Result<Self, String> {
        if config.num_experts == 0 {
            return Ok(Self { plans: Vec::new() });
        }
        let policy = MoEExecutionPolicy::single();
        let specs = qwen35_moe_expert_group_specs(config, &policy);
        let manifest =
            <crate::arch::Qwen35 as hipfire_runtime::arch::Architecture>::weight_manifest(config);
        let plans =
            hipfire_runtime::weight_manifest::resolve_expert_group_plans(&specs, &manifest, 1)?;
        Ok(Self { plans })
    }

    /// Borrow the immutable plan for one MoE layer (plans are indexed by
    /// global layer — every layer is MoE when `num_experts > 0`).
    pub(crate) fn by_layer(
        &self,
        layer: usize,
    ) -> &hipfire_runtime::weight_manifest::ExpertGroupPlan {
        &self.plans[layer]
    }

    /// Number of resolved plans (== number of MoE layers; 0 for dense).
    pub(crate) fn len(&self) -> usize {
        self.plans.len()
    }
}

/// The Qwen35 Single expert-group manifest declaration (STEP-002 Task 9).
///
/// One layer-local group per MoE layer (every layer is MoE when
/// `num_experts > 0`) with stable semantic identities matching the actual
/// dispatch builder plans:
/// - `router_identity = "softmax_topk"` — the decode builder's
///   [`Step::MoeSoftmaxTopK`] routing;
/// - `allowed_executions = [indexed_quantized, grouped_quantized]` — decode
///   and prefill Path 0/1 build indexed steps; grouped prefill Path 2 builds
///   a grouped program. Both are declared; the CPU fallback is not declared
///   (it lives outside lowering).
///
/// TP/EP policies resolve ZERO groups: Qwen35 has no parallel expert-group
/// admission, and refusing here happens before any program construction.
pub(crate) fn qwen35_moe_expert_group_specs(
    cfg: &Qwen35Config,
    policy: &hipfire_runtime::moe_plan::MoEExecutionPolicy,
) -> Vec<hipfire_runtime::weight_manifest::ExpertGroupSpec> {
    use hipfire_runtime::moe_plan::MoEExecutionKind;
    use hipfire_runtime::weight_manifest::{
        ExpertExecutionIdentity, ExpertGroupSpec, ExpertParallelism, ExpertResourceRequirements,
        ExpertSourceLayout,
    };
    if cfg.num_experts == 0 || policy.kind() != MoEExecutionKind::Single {
        return Vec::new();
    }
    (0..cfg.n_layers)
        .map(|layer| ExpertGroupSpec {
            group: format!("qwen35_moe_layer_{layer}"),
            layer: Some(layer),
            n_experts: cfg.num_experts,
            parallelism: ExpertParallelism::Single,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PerExpertFused {
                gate_up: (0..cfg.num_experts)
                    .map(|e| format!("expert.{e}.gate_up"))
                    .collect(),
                down: (0..cfg.num_experts)
                    .map(|e| format!("expert.{e}.down"))
                    .collect(),
                sidecars: Vec::new(),
            },
            // Full f16 sizes of gate_up + down, the admission requirement.
            resources: ExpertResourceRequirements {
                bytes_per_expert: 3 * cfg.moe_intermediate_size * cfg.dim * 2,
                alignment: 64,
            },
            router: "router".to_owned(),
            router_identity: "softmax_topk".to_owned(),
            allowed_executions: vec![
                ExpertExecutionIdentity::IndexedQuantized,
                ExpertExecutionIdentity::GroupedQuantized,
            ],
        })
        .collect()
}

pub(crate) fn validate_frozen_moe_dispatch(
    config: &Qwen35Config,
    snapshot: &MoeDtypeSnapshot,
    per_expert_gate_up: &[DType],
    per_expert_down: &[DType],
    has_paro_shared: bool,
    routed_down_has_awq: bool,
    is_wave32: bool,
    has_wmma: bool,
    has_deltanet: bool,
) -> Result<(), String> {
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    // ── 1. Universal constraints ────────────────────────────────────
    // Frozen MoE requires k=8 with indexed GPU-top-K decode.  No CPU
    // routed-expert fallback exists in the Frozen path.
    if k != 8 {
        return Err(format!(
            "Frozen MoE requires num_experts_per_tok == 8, got {k}"
        ));
    }
    if !(8..=1024).contains(&n_exp) {
        return Err(format!(
            "Frozen MoE requires 8 <= num_experts <= 1024, got {n_exp}"
        ));
    }

    // ── 2. Build MoeDtypes from snapshot + per-expert info ──────────
    // Follows the same mapping as `moe_dtypes_from_view` so
    // MoeResolution::resolve_arch produces the same eligibility.

    let per_gu_tiers = if n_exp > 0 {
        mixed_tier_table(per_expert_gate_up.to_vec())
    } else {
        None
    };
    let per_dn_tiers = if n_exp > 0 {
        mixed_tier_table(per_expert_down.to_vec())
    } else {
        None
    };

    // Compute `experts_all_gate_up_mq4` from per-expert actual dtypes
    // (not just the representative — handle the mixed case).
    let experts_all_gate_up_mq4 =
        n_exp > 0 && per_expert_gate_up.iter().all(|&dt| dt == DType::MQ4G256);

    let dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: snapshot.router,
        shared_gate: snapshot.shared_expert_scalar_gate,
        shared_expert_gate: snapshot.shared_gate,
        shared_expert_up: snapshot.shared_up,
        shared_expert_down: snapshot.shared_down,
        experts_all_gate_up_mq4,
        routed_gate_up: snapshot.expert_gate_up,
        routed_down: snapshot.expert_down,
        routed_has_mixed_experts: snapshot.expert_dtype_tags_present,
        has_paro_shared,
        gate_side_has_awq: snapshot.gate_side_has_awq,
        routed_down_has_awq,
        per_expert_gate_up: per_gu_tiers,
        per_expert_down: per_dn_tiers,
    };

    let res = hipfire_dispatch::families::moe::MoeResolution::resolve_arch(&dtypes, k, has_wmma);

    // ── 3. GPU top-k required (no CPU routed-expert fallback) ───────
    if !res.use_gpu_topk {
        return Err(format!(
            "Frozen MoE routed-dtype combination is not indexable on this arch: \
             gate_up={:?} down={:?} mixed={} paro={}",
            snapshot.expert_gate_up,
            snapshot.expert_down,
            snapshot.expert_dtype_tags_present,
            has_paro_shared,
        ));
    }

    // ── 4. Additional arch guards (resolver over-broad cases) ───────
    if res.routed_indexable_mq5 && !is_wave32 {
        return Err("MQ5 routed experts require wave32 architecture (gfx11/gfx12)".into());
    }
    if res.routed_indexable_mq6 && !is_wave32 {
        return Err("MQ6 routed experts require wave32 architecture (gfx11/gfx12)".into());
    }
    if res.routed_indexable_mixed_gu4_dn6 && !is_wave32 {
        return Err(
            "MQ4/MQ6 mixed routed experts require wave32 architecture (gfx11/gfx12)".into(),
        );
    }
    if res.routed_indexable_mq2lloyd && !is_wave32 {
        return Err("MQ2-Lloyd routed experts require wave32 architecture (gfx11/gfx12)".into());
    }
    if res.routed_indexable_mq3lloyd && !is_wave32 {
        return Err("MQ3-Lloyd routed experts require wave32 architecture (gfx11/gfx12)".into());
    }
    if res.mixed && !is_wave32 {
        return Err(
            "Mixed-precision routed experts require wave32 architecture (gfx11/gfx12)".into(),
        );
    }

    // ── 5. Tag coherence (both directions) ───────────────────────────
    // Use the actual per-expert dtype vectors to determine whether the
    // tag table state (present/absent) matches the actual per-expert
    // variation.  Also enforce the special MQ4-gate-up constraint for
    // (MQ4, other) pairs.
    if n_exp > 0 {
        let n = per_expert_gate_up.len().min(per_expert_down.len());

        // Determine if pairs actually vary across experts.
        let first_pair = (per_expert_gate_up[0], per_expert_down[0]);
        let pairs_vary = n > 1
            && per_expert_gate_up[1..]
                .iter()
                .zip(per_expert_down[1..].iter())
                .any(|(gu, dn)| *gu != first_pair.0 || *dn != first_pair.1);

        // Both directions: tags-present+identical OR tags-absent+varying.
        if snapshot.expert_dtype_tags_present && !pairs_vary {
            return Err(format!(
                "dtype_tags present but all {n} experts have identical pair; \
                 tags should be absent"
            ));
        }
        if !snapshot.expert_dtype_tags_present && pairs_vary {
            return Err(
                "dtype_tags absent but experts have varying pairs; tags should be present"
                    .to_string(),
            );
        }

        // When pairs vary (mixed/graded), validate each distinct pair is
        // in the supported tag table via fallible_dtype_tag.  Uniform pairs
        // skip this — they are indexable without tags.
        if pairs_vary {
            for i in 0..n {
                let gu = per_expert_gate_up[i];
                let dn = per_expert_down[i];
                crate::store::fallible_dtype_tag(gu, dn)
                    .map_err(|msg| format!("expert.{i}: {msg}"))?;
            }

            // Special (MQ4, other) pairs: require ALL gate-up projections
            // to be uniformly MQ4.  The (MQ4, MQ6), (MQ4, MQ2Lloyd),
            // (MQ4, MQ3Lloyd), (MQ4, MFP3E8), (MQ4, MFP2E8) variants are
            // only valid when gate_up stays MQ4 across all experts.
            if per_expert_gate_up.iter().any(|&gu| gu != DType::MQ4G256)
                && per_expert_gate_up
                    .iter()
                    .zip(per_expert_down.iter())
                    .any(|(gu, dn)| *gu == DType::MQ4G256 && *dn != DType::MQ4G256)
            {
                return Err(
                    "special (MQ4, down-other) pairs require all gate_up to be uniformly MQ4"
                        .into(),
                );
            }
        }
    }

    // ── 6. Gate/router/shared projection dtype sanity ────────────────
    // Verify every projection dtype has a known rotation/GEMV plan.
    // Non-F32 projections must have a rotation plan beyond None, or be
    // F32/F16/Q8_0/HFQ-family (Plain GEMV without rotation).
    let check_proj = |label: &str, dt: DType| -> Result<(), String> {
        let plan = dtype_rotation_plan(dt);
        match plan {
            RotationPlan::None => {
                // Known types that use Plain GEMV without rotation.
                if !matches!(
                    dt,
                    DType::F32
                        | DType::F16
                        | DType::BF16
                        | DType::Q8_0
                        | DType::HFQ4G256
                        | DType::HFQ3G256
                        | DType::HFQ6G256
                        | DType::ParoQ4G128
                ) {
                    return Err(format!(
                        "{label}: dtype {dt:?} has RotationPlan::None but is not \
                         a known non-rotated type (expected F32/F16/BF16/Q8/HFQ/Paro)"
                    ));
                }
                // Paro requires Givens rotation; if plan is None for Paro
                // something is off (ParoQ4G128 maps to Givens below).
            }
            RotationPlan::FwhtG256
            | RotationPlan::FwhtG128
            | RotationPlan::Mq8Internal
            | RotationPlan::Givens => {
                // Known rotated types are fine — they have a rotation path.
            }
        }
        // Verify the post-rotation variant is concrete (not just Plain
        // for an unknown type — every MQ/MFP family maps to Prerotated;
        // Paro maps to Plain post-Givens).
        if plan == RotationPlan::None
            && dt != DType::F32
            && !matches!(
                dt,
                DType::F16
                    | DType::BF16
                    | DType::Q8_0
                    | DType::HFQ4G256
                    | DType::HFQ3G256
                    | DType::HFQ6G256
                    | DType::ParoQ4G128
            )
        {
            return Err(format!(
                "{label}: dtype {dt:?} has no rotation plan and is not a known \
                 unrotated weight type"
            ));
        }
        Ok(())
    };

    check_proj("router", snapshot.router)?;
    check_proj("shared_expert_gate", snapshot.shared_expert_scalar_gate)?;
    check_proj("shared_gate_proj", snapshot.shared_gate)?;
    check_proj("shared_up_proj", snapshot.shared_up)?;
    check_proj("shared_down_proj", snapshot.shared_down)?;

    // ── 7. Shared-down non-MQ4 requires compiled DeltaNet path ──────
    // The non-MQ4 shared-down variant requires compiled DeltaNet
    // (the gated_residual_delta_net path).  Reject if unavailable.
    if snapshot.shared_down != DType::MQ4G256 && !has_deltanet {
        return Err(format!(
            "shared_down dtype {dt:?} requires the DeltaNet feature but it is not enabled",
            dt = snapshot.shared_down
        ));
    }

    // ── 8. AWQ constraints ──────────────────────────────────────────
    // Gate-side AWQ disables fused gate execution (MoeResolution already
    // handles this via gate_fusable / gate_side_mq4).  No additional
    // rejection needed — the snapshot.gate_side_has_awq flag correctly
    // disabled gate_side_mq4, so MoeResolution::gate_fusable is false,
    // and the forward path uses individual WeightRef paths.

    // Routed-down AWQ: the store planner already validates all-or-none
    // coverage; the forward path handles it via expert_down_awq_ptrs.
    // Routed gate-up AWQ: rejected upstream by is_routed_gate_up_awq.
    // No duplicate rejection needed here.

    // ── 9. Prefill eligibility (decode-eligible publication only) ───
    // Batched prefill MAY be ineligible for certain dtype combinations
    // (e.g. non-MQ4 shared-down without batching env gated on).  This is
    // NOT a freeze rejection — the per-token indexed decode path remains
    // eligible.  We flag soft but do not reject.
    //
    // Routed-down AWQ must force Path2 (grouped) eligibility false, but
    // individual indexed paths remain allowed.  The planner sets
    // HAS_AWQ_DOWN flag on the resident metadata so the forward path
    // can gate Path2 selection.  No rejection here.

    Ok(())
}

impl MmqScreenable for Qwen35Weights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.output, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            match layer {
                LayerWeights::DeltaNet(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttn(weights) => {
                    for weight in [
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                // Routed and shared experts live outside ordinary WeightTensor
                // storage in paged/EP modes. Screen the resident attention and
                // dense router weights here; expert screening is separate work.
                LayerWeights::DeltaNetMoe(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                    // Routed expert storage is screened on the Legacy path
                    // only; Frozen storage is owned by the resident.
                    if let Some(ffn) = weights.ffn.as_legacy() {
                        screen_weight_tensor(&ffn.router, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttnMoe(weights) => {
                    for weight in [&weights.wq, &weights.wk, &weights.wv, &weights.wo] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                    if let Some(ffn) = weights.ffn.as_legacy() {
                        screen_weight_tensor(&ffn.router, gpu, &mut safe, &mut unsafe_count);
                    }
                }
            }
        }
        (safe, unsafe_count)
    }
}

/// Free one MoE layer's FFN storage: Legacy owned weights are fully freed;
/// Frozen markers own nothing (the resident frees its allocations when freed
/// separately).
fn free_moe_storage(gpu: &mut Gpu, storage: MoeFfnStorage) {
    match storage {
        MoeFfnStorage::Legacy(ffn) => free_moe_ffn(gpu, ffn),
        MoeFfnStorage::Frozen => {}
    }
}

fn free_moe_ffn(gpu: &mut Gpu, ffn: MoeFfnWeights) {
    ffn.router.free_all(gpu);
    ffn.shared_expert_gate.free_all(gpu);
    ffn.shared_expert.gate.free_all(gpu);
    ffn.shared_expert.up.free_all(gpu);
    ffn.shared_expert.down.free_all(gpu);
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
    // Non-owning pointer table — free the buffer only; the per-expert scales it
    // points into are owned by `experts[i].down.awq_scale` and freed below via
    // `e.down.free_all`.
    if let Some(t) = ffn.expert_down_awq_ptrs {
        let _ = gpu.free_tensor(t);
    }
    // Owned device buffer (built from per-expert gpu_dtype). Free it.
    if let Some(t) = ffn.expert_dtype_tags {
        let _ = gpu.free_tensor(t);
    }
    if let Some(owners) = ffn.packed_expert_owners {
        // Packed expert WeightTensors are non-owning views. Free only metadata
        // that remains individually owned, then return each layer blob once.
        for e in ffn.experts {
            free_weight_metadata_only(gpu, e.gate_up);
            free_weight_metadata_only(gpu, e.down);
        }
        let _ = gpu.free_tensor(owners.gate_up);
        let _ = gpu.free_tensor(owners.down);
    } else {
        for e in ffn.experts {
            e.gate_up.free_all(gpu);
            e.down.free_all(gpu);
        }
    }
    // ParoQuant MoE: free the owning shared sidecars (per-expert `paro` fields
    // alias these and must NOT be freed separately — they're non-owning views).
    if let Some(s) = ffn.paro_shared {
        let _ = gpu.free_tensor(s.gate_up_pairs);
        let _ = gpu.free_tensor(s.gate_up_theta);
        let _ = gpu.free_tensor(s.gate_up_channel_scales);
        let _ = gpu.free_tensor(s.down_pairs);
        let _ = gpu.free_tensor(s.down_theta);
        let _ = gpu.free_tensor(s.down_channel_scales);
    }
    for d in ffn.ep_dummy_buffers {
        let _ = gpu.free_tensor(d);
    }
}
/// Continue-and-retain helper for `MoeFfnWeights` (Legacy path).
/// Frees every tensor in the MoE FFN struct, retaining on failure.
/// Mirrors the unchecked [`free_moe_ffn`] owner set exactly: shared
/// projections, pointer tables, AWQ/dtype-tag tables, packed-owner blobs
/// (expert views freed metadata-only), Paro shared sidecars, and the
/// mainline EP dummy buffers.  `global_expert_dtypes` is CPU-side and
/// carries no free authority.
fn free_moe_ffn_checked(
    label: &str,
    ffn: MoeFfnWeights,
    gpu: &mut Gpu,
    failures: &mut Vec<RetainedGpuTensor>,
) {
    free_weight_all_checked(&format!("{label}.router"), ffn.router, gpu, failures);
    free_weight_all_checked(
        &format!("{label}.shared_expert_gate"),
        ffn.shared_expert_gate,
        gpu,
        failures,
    );
    free_weight_all_checked(
        &format!("{label}.shared_expert.gate"),
        ffn.shared_expert.gate,
        gpu,
        failures,
    );
    free_weight_all_checked(
        &format!("{label}.shared_expert.up"),
        ffn.shared_expert.up,
        gpu,
        failures,
    );
    free_weight_all_checked(
        &format!("{label}.shared_expert.down"),
        ffn.shared_expert.down,
        gpu,
        failures,
    );

    free_tensor_retained(
        format!("{label}.expert_gate_up_ptrs"),
        ffn.expert_gate_up_ptrs,
        gpu,
        failures,
    );
    free_tensor_retained(
        format!("{label}.expert_down_ptrs"),
        ffn.expert_down_ptrs,
        gpu,
        failures,
    );

    if let Some(t) = ffn.expert_down_awq_ptrs {
        free_tensor_retained(format!("{label}.expert_down_awq_ptrs"), t, gpu, failures);
    }
    if let Some(t) = ffn.expert_dtype_tags {
        free_tensor_retained(format!("{label}.expert_dtype_tags"), t, gpu, failures);
    }

    if let Some(owners) = ffn.packed_expert_owners {
        // Packed expert WeightTensors are NON-OWNING views (interior
        // pointers into the two layer blobs). Free only the metadata that
        // remains individually owned (awq_scale/paro sidecars — the checked
        // equivalent of the unchecked free_weight_metadata_only), then
        // return each layer blob ONCE. NEVER free the view bufs: pooling an
        // interior pointer aliases the live blob and orphans the owners.
        for (i, e) in ffn.experts.into_iter().enumerate() {
            free_weight_sidecars_checked(
                &format!("{label}.experts[{i}].gate_up"),
                e.gate_up,
                gpu,
                failures,
            );
            free_weight_sidecars_checked(
                &format!("{label}.experts[{i}].down"),
                e.down,
                gpu,
                failures,
            );
        }
        free_tensor_retained(
            format!("{label}.packed_expert_owners.gate_up"),
            owners.gate_up,
            gpu,
            failures,
        );
        free_tensor_retained(
            format!("{label}.packed_expert_owners.down"),
            owners.down,
            gpu,
            failures,
        );
    } else {
        for (i, e) in ffn.experts.into_iter().enumerate() {
            free_weight_all_checked(
                &format!("{label}.experts[{i}].gate_up"),
                e.gate_up,
                gpu,
                failures,
            );
            free_weight_all_checked(&format!("{label}.experts[{i}].down"), e.down, gpu, failures);
        }
    }

    if let Some(s) = ffn.paro_shared {
        free_tensor_retained(
            format!("{label}.paro_shared.gate_up_pairs"),
            s.gate_up_pairs,
            gpu,
            failures,
        );
        free_tensor_retained(
            format!("{label}.paro_shared.gate_up_theta"),
            s.gate_up_theta,
            gpu,
            failures,
        );
        free_tensor_retained(
            format!("{label}.paro_shared.gate_up_channel_scales"),
            s.gate_up_channel_scales,
            gpu,
            failures,
        );
        free_tensor_retained(
            format!("{label}.paro_shared.down_pairs"),
            s.down_pairs,
            gpu,
            failures,
        );
        free_tensor_retained(
            format!("{label}.paro_shared.down_theta"),
            s.down_theta,
            gpu,
            failures,
        );
        free_tensor_retained(
            format!("{label}.paro_shared.down_channel_scales"),
            s.down_channel_scales,
            gpu,
            failures,
        );
    }

    // Mainline EP streaming dummies: one owned zero buffer per distinct
    // non-owned storage layout, reclaimed like the unchecked path.
    for (i, d) in ffn.ep_dummy_buffers.into_iter().enumerate() {
        free_tensor_retained(format!("{label}.ep_dummy_buffers[{i}]"), d, gpu, failures);
    }
}

/// Free a [`WeightTensor`]'s owning sidecars without freeing its weight buffer.
/// Used only for non-owning views into [`PackedExpertOwners`].
fn free_weight_metadata_only(gpu: &mut Gpu, weight: WeightTensor) {
    if let Some(paro) = weight.paro {
        if !paro.is_alias {
            let _ = gpu.free_tensor(paro.pairs);
            let _ = gpu.free_tensor(paro.theta);
            let _ = gpu.free_tensor(paro.channel_scales);
        }
    }
    if let Some(awq) = weight.awq_scale {
        let _ = gpu.free_tensor(awq);
    }
}

// ─── State ──────────────────────────────────────────────────────────────

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Per-element f16 error-feedback residual for Q8 state requant (sigma-delta
    /// noise-shaping). Empty unless Q8 + `HIPFIRE_DN_STATE_EF`. Same element count
    /// as `s_matrices`; carries the previous step's quant error so the next
    /// requant cancels it — DeltaNet's contractive decay damps the shaped noise,
    /// yielding ~FP32-grade state at Q8's byte container.
    pub s_ef_residual: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    /// EF residual for a delta-layer, if error-feedback is active (Q8 + flag).
    /// `None` ⇒ callers pass null ⇒ kernel uses the legacy stochastic-rounding requant.
    #[inline]
    pub fn ef_residual(&self, idx: usize) -> Option<&GpuTensor> {
        self.s_ef_residual.get(idx)
    }

    /// Non-owning single-lane view into state allocated by
    /// [`Self::new_batched_with_quant`]. Used only to seed prompts through the
    /// existing sequential prefill path. The returned view must not be freed.
    pub(crate) fn q8_lane_view(
        &self,
        config: &Qwen35Config,
        lane: usize,
        batch: usize,
    ) -> HipResult<Self> {
        if self.quant != StateQuant::Q8 || lane >= batch {
            return Err(HipError::new(
                0,
                "DeltaNet q8_lane_view requires Q8 state and a valid lane",
            ));
        }
        let n_heads = config.linear_num_value_heads;
        let hd = config.linear_value_head_dim;
        let s_elems = n_heads * hd * hd;
        let scale_elems = n_heads * hd;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_elems = conv_channels * (config.conv_kernel_dim - 1);

        let byte_view = |t: &GpuTensor, off: usize, bytes: usize, dtype: DType| {
            let ptr = unsafe { (t.buf.as_ptr() as *mut u8).add(off) as *mut std::ffi::c_void };
            GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
                shape: vec![bytes / dtype.size()],
                dtype,
            }
        };
        Ok(Self {
            s_matrices: self
                .s_matrices
                .iter()
                .map(|t| byte_view(t, lane * s_elems, s_elems, DType::Raw))
                .collect(),
            s_scales: self
                .s_scales
                .iter()
                .map(|t| byte_view(t, lane * scale_elems * 4, scale_elems * 4, DType::F32))
                .collect(),
            conv_states: self
                .conv_states
                .iter()
                .map(|t| byte_view(t, lane * conv_elems * 4, conv_elems * 4, DType::F32))
                .collect(),
            s_ef_residual: self
                .s_ef_residual
                .iter()
                .map(|t| byte_view(t, lane * s_elems * 2, s_elems * 2, DType::F16))
                .collect(),
            quant: StateQuant::Q8,
        })
    }

    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, StateQuant::Q8)
    }

    pub fn new_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<Self> {
        Self::new_batched_with_quant(gpu, config, quant, 1)
    }

    /// Allocate lane-major recurrent state for independent-sequence decode.
    ///
    /// The ordinary state has an implicit batch of one.  This variant keeps
    /// the same per-layer vectors, but every tensor is laid out as
    /// `[batch, ...single-lane shape...]`.  It is intentionally consumed only
    /// by [`Qwen35DecodeBatchState`]: passing it to sequential prefill would
    /// advance lane 0 and leave the other lanes stale.
    pub fn new_batched_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
        batch: usize,
    ) -> HipResult<Self> {
        assert!(batch > 0, "DeltaNetState batch must be non-zero");
        let n_delta_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size_per_lane = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144
        let s_size = batch * s_size_per_lane;

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = batch * conv_channels * (config.conv_kernel_dim - 1);

        // Error-feedback (sigma-delta) requant for Q8 state — DEFAULT ON as of
        // 2026-06-08. q8_ef ≈ FP32 coherence at −0.7% decode vs FP32's −4.5% (best
        // spec-decode τ too), and far better than stochastic Q8 — DFlash 27b-prose
        // unique_ratio 0.625 vs 0.555, max_freq 0.055 vs 0.078. Also makes the DN
        // state DETERMINISTIC (no stochastic dither). Opt OUT with
        // HIPFIRE_DN_STATE_EF=0. Q8-only (FP32 has no requant; Q4 EF is future
        // work; the multi-GPU band split is still stochastic — new_with_quant_multi
        // leaves s_ef_residual empty). Residual is f16 per-element.
        let ef_enabled = quant == StateQuant::Q8
            && hipfire_config::developer_var("HIPFIRE_DN_STATE_EF")
                .map(|v| v != "0")
                .unwrap_or(true);

        // GpuTensor has no freeing Drop (free needs &mut Gpu). Mirror
        // alloc_k_v_vmm_filtered: on any mid-loop failure free every tensor
        // already pushed before propagating.
        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        let mut s_ef_residual = Vec::with_capacity(if ef_enabled { n_delta_layers } else { 0 });
        let result = (|| -> HipResult<()> {
            for _ in 0..n_delta_layers {
                match quant {
                    StateQuant::FP32 => {
                        s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                        s_scales.push(gpu.zeros(&[batch * n_heads], DType::F32)?);
                    }
                    StateQuant::Q8 => {
                        // int8 state: s_size bytes (1 byte each), per-row scales
                        let buf = gpu.hip.malloc(s_size)?;
                        if let Err(e) = gpu.hip.memset(&buf, 0, s_size) {
                            let _ = gpu.hip.free(buf);
                            return Err(e);
                        }
                        s_matrices.push(GpuTensor {
                            buf,
                            shape: vec![s_size],
                            dtype: DType::F32,
                        });
                        s_scales.push(gpu.zeros(&[batch * n_heads * s_dim], DType::F32)?);
                    }
                    StateQuant::Q4 => {
                        // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                        let buf = gpu.hip.malloc(s_size / 2)?;
                        if let Err(e) = gpu.hip.memset(&buf, 0, s_size / 2) {
                            let _ = gpu.hip.free(buf);
                            return Err(e);
                        }
                        s_matrices.push(GpuTensor {
                            buf,
                            shape: vec![s_size / 2],
                            dtype: DType::F32,
                        });
                        s_scales.push(gpu.zeros(&[batch * n_heads * s_dim], DType::F32)?);
                    }
                }
                if ef_enabled {
                    s_ef_residual.push(gpu.zeros(&[s_size], DType::F16)?);
                }
                conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in s_matrices
                .drain(..)
                .chain(s_scales.drain(..))
                .chain(conv_states.drain(..))
                .chain(s_ef_residual.drain(..))
            {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(err);
        }
        Ok(Self {
            s_matrices,
            s_scales,
            conv_states,
            s_ef_residual,
            quant,
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.s_matrices {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_scales {
            let _ = gpu.free_tensor(t);
        }
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_ef_residual {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Reset all DeltaNet recurrent buffers to zero in place. Lets callers
    /// reuse a single `DeltaNetState` across independent chunks/sequences
    /// without allocating per chunk (which leaks since DeltaNetState has no
    /// Drop). Mirrors `ModelSlot::reset_state` in speculative.rs.
    ///
    /// Returns `Err` on the first HIP memset/memset_async failure so production
    /// rollback can attest `rolled_back:false`.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                for s in &self.s_matrices {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.s_scales {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.conv_states {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.s_ef_residual {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
            }
            None => {
                for s in &self.s_matrices {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.s_scales {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.conv_states {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.s_ef_residual {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
            }
        }
        Ok(())
    }

    /// Multi-GPU companion to `new_with_quant`. Each LA-layer's state is
    /// allocated on the device that owns the layer in the multi-GPU band
    /// split: `gpus.devices[gpus.device_for_layer(orig_layer_idx)]` for the
    /// `orig_layer_idx` of the LA-layer. Returns the state alongside the
    /// `la_to_device` mapping the daemon needs to route reset memsets to
    /// the correct device.
    pub fn new_with_quant_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<(Self, Vec<u8>)> {
        let s_dim = config.linear_key_head_dim;
        let n_heads = config.linear_num_value_heads;
        let s_size = n_heads * s_dim * s_dim;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        let mut s_matrices = Vec::new();
        let mut s_scales = Vec::new();
        let mut conv_states = Vec::new();
        let mut la_to_device: Vec<u8> = Vec::new();

        for (orig_layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != LayerType::LinearAttention {
                continue;
            }
            let dev_idx = gpus.device_for_layer(orig_layer_idx);
            la_to_device.push(dev_idx as u8);
            let g = &mut gpus.devices[dev_idx];
            // g.hip.malloc/memset bypass the Stage 2 bind_thread audit
            // (HipRuntime methods don't carry a device id). Bind explicitly
            // before any raw HIP ops so allocations land on the right device.
            g.bind_thread()?;
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(g.zeros(&[s_size], DType::F32)?);
                    s_scales.push(g.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    let buf = g.hip.malloc(s_size)?;
                    g.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    let buf = g.hip.malloc(s_size / 2)?;
                    g.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size / 2],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            conv_states.push(g.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok((
            Self {
                s_matrices,
                s_scales,
                conv_states,
                // EF residual not wired for the multi-GPU band split (would need
                // per-device residual alloc routed by device_for_layer); empty ⇒
                // ef_residual() returns None ⇒ kernel uses the stochastic path.
                s_ef_residual: Vec::new(),
                quant,
            },
            la_to_device,
        ))
    }

    /// Free per-LA-layer tensors on the devices listed in `la_to_device`
    /// (the second tuple element returned by `new_with_quant_multi`).
    pub fn free_gpu_multi(self, gpus: &mut Gpus, la_to_device: &[u8]) {
        for (i, t) in self.s_matrices.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.s_scales.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.conv_states.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        // Empty today (multi-GPU EF not wired); free if/when residuals land.
        for (i, t) in self.s_ef_residual.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
    }
}

impl DeltaNetState {
    /// Checked GPU cleanup: attempts every tensor independently, retains
    /// every allocation that could not be freed for retry.
    ///
    /// ## Ownership semantics
    ///
    /// Every tensor is attempted even after prior failures.  On success all
    /// resources are consumed (`Ok(())`).  On failure the returned
    /// `Vec<RetainedGpuTensor>` carries the exact original tensors that
    /// could not be freed, ready for retry.
    ///
    /// ## GPU evidence limitation
    ///
    /// `free_tensor_checked` only fails on `bind_thread` errors (see
    /// `GpuCleanupFailure` notes).  Full retry requires a real HIP device.
    pub fn abort_checked(self, gpu: &mut Gpu) -> Result<(), Vec<RetainedGpuTensor>> {
        let mut failures: Vec<RetainedGpuTensor> = Vec::new();

        for (i, t) in self.s_matrices.into_iter().enumerate() {
            free_tensor_retained(
                format!("DeltaNetState.s_matrices[{i}]"),
                t,
                gpu,
                &mut failures,
            );
        }
        for (i, t) in self.s_scales.into_iter().enumerate() {
            free_tensor_retained(
                format!("DeltaNetState.s_scales[{i}]"),
                t,
                gpu,
                &mut failures,
            );
        }
        for (i, t) in self.conv_states.into_iter().enumerate() {
            free_tensor_retained(
                format!("DeltaNetState.conv_states[{i}]"),
                t,
                gpu,
                &mut failures,
            );
        }
        for (i, t) in self.s_ef_residual.into_iter().enumerate() {
            free_tensor_retained(
                format!("DeltaNetState.s_ef_residual[{i}]"),
                t,
                gpu,
                &mut failures,
            );
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

impl Qwen35Scratch {
    /// Checked GPU cleanup: attempts every tensor independently, retains
    /// every allocation that could not be freed for retry.
    ///
    /// ## Ownership semantics
    ///
    /// Every tensor is attempted even after prior failures.  On success all
    /// resources are consumed (`Ok(())`).  On failure the returned
    /// `Vec<RetainedGpuTensor>` carries the exact original tensors that
    /// could not be freed, ready for retry.
    ///
    /// `pos_buf` / `pos_buf3` are raw [`hip_bridge::DeviceBuffer`]s; each is
    /// represented as a `GpuTensor` with `shape=[]` / `DType::Raw` — the
    /// honest description of a bare allocation, not a fabrication.
    pub fn abort_checked(self, gpu: &mut Gpu) -> Result<(), Vec<RetainedGpuTensor>> {
        let mut failures: Vec<RetainedGpuTensor> = Vec::new();

        free_tensor_retained("Qwen35Scratch.x", self.x, gpu, &mut failures);
        free_tensor_retained("Qwen35Scratch.tmp", self.tmp, gpu, &mut failures);
        // pos_buf: raw DeviceBuffer → honest GpuTensor wrapper (Raw dtype).
        free_tensor_retained(
            "Qwen35Scratch.pos_buf",
            GpuTensor {
                buf: self.pos_buf,
                shape: vec![],
                dtype: DType::Raw,
            },
            gpu,
            &mut failures,
        );
        // pos_buf3: raw DeviceBuffer (3D mrope positions) → same wrapper.
        free_tensor_retained(
            "Qwen35Scratch.pos_buf3",
            GpuTensor {
                buf: self.pos_buf3,
                shape: vec![],
                dtype: DType::Raw,
            },
            gpu,
            &mut failures,
        );
        for (label, t) in [
            ("Qwen35Scratch.dn_qkv", self.dn_qkv),
            ("Qwen35Scratch.dn_z", self.dn_z),
            ("Qwen35Scratch.dn_alpha", self.dn_alpha),
            ("Qwen35Scratch.dn_beta", self.dn_beta),
            ("Qwen35Scratch.dn_conv_out", self.dn_conv_out),
            ("Qwen35Scratch.dn_q", self.dn_q),
            ("Qwen35Scratch.dn_k", self.dn_k),
            ("Qwen35Scratch.dn_v", self.dn_v),
            ("Qwen35Scratch.dn_q_raw", self.dn_q_raw),
            ("Qwen35Scratch.dn_k_raw", self.dn_k_raw),
            ("Qwen35Scratch.dn_attn_out", self.dn_attn_out),
            ("Qwen35Scratch.dn_normed", self.dn_normed),
            ("Qwen35Scratch.fa_q_full", self.fa_q_full),
            ("Qwen35Scratch.fa_q", self.fa_q),
            ("Qwen35Scratch.fa_gate", self.fa_gate),
            ("Qwen35Scratch.fa_k", self.fa_k),
            ("Qwen35Scratch.fa_v", self.fa_v),
            ("Qwen35Scratch.fa_attn_out", self.fa_attn_out),
            ("Qwen35Scratch.o", self.o),
            ("Qwen35Scratch.gate_ffn", self.gate_ffn),
            ("Qwen35Scratch.up", self.up),
            ("Qwen35Scratch.ffn_hidden", self.ffn_hidden),
            ("Qwen35Scratch.ffn_out", self.ffn_out),
            ("Qwen35Scratch.logits", self.logits),
            ("Qwen35Scratch.sample_buf", self.sample_buf),
            ("Qwen35Scratch.repeat_buf", self.repeat_buf),
            ("Qwen35Scratch.x_rot", self.x_rot),
            ("Qwen35Scratch.flash_partials", self.flash_partials),
        ] {
            free_tensor_retained(label, t, gpu, &mut failures);
        }
        // MoE scratch — only present for MoE configs; skip None.
        for (label, t) in [
            ("Qwen35Scratch.moe_router_logits", self.moe_router_logits),
            ("Qwen35Scratch.moe_scalar_buf", self.moe_scalar_buf),
            ("Qwen35Scratch.moe_x_rot", self.moe_x_rot),
            ("Qwen35Scratch.moe_gate_up_buf", self.moe_gate_up_buf),
            ("Qwen35Scratch.moe_gate_buf", self.moe_gate_buf),
            ("Qwen35Scratch.moe_up_buf", self.moe_up_buf),
            ("Qwen35Scratch.moe_ffn_hidden", self.moe_ffn_hidden),
            ("Qwen35Scratch.moe_ffn_out", self.moe_ffn_out),
            ("Qwen35Scratch.moe_gate_batch", self.moe_gate_batch),
            ("Qwen35Scratch.moe_up_batch", self.moe_up_batch),
            ("Qwen35Scratch.moe_rot_batch", self.moe_rot_batch),
            ("Qwen35Scratch.moe_topk_indices", self.moe_topk_indices),
            ("Qwen35Scratch.moe_topk_weights", self.moe_topk_weights),
            ("Qwen35Scratch.moe_down_expanded", self.moe_down_expanded),
        ] {
            if let Some(t) = t {
                free_tensor_retained(label, t, gpu, &mut failures);
            }
        }
        // Optional batched-prefill scratch — delegate to its own checked abort.
        if let Some(pbs) = self.prefill_batch {
            if let Err(pbs_failures) = pbs.abort_checked(gpu) {
                failures.extend(pbs_failures);
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

impl PrefillBatchScratch {
    /// Checked GPU cleanup: attempts every tensor independently, retains
    /// every allocation that could not be freed for retry.
    ///
    /// ## Ownership semantics
    ///
    /// Every tensor is attempted even after prior failures.  On success all
    /// resources are consumed (`Ok(())`).  On failure the returned
    /// `Vec<RetainedGpuTensor>` carries the exact original tensors that
    /// could not be freed, ready for retry.
    pub fn abort_checked(self, gpu: &mut Gpu) -> Result<(), Vec<RetainedGpuTensor>> {
        let mut failures: Vec<RetainedGpuTensor> = Vec::new();

        for (label, t) in [
            ("PrefillBatchScratch.x_batch", self.x_batch),
            ("PrefillBatchScratch.x_rot_batch", self.x_rot_batch),
            ("PrefillBatchScratch.x_norm_batch", self.x_norm_batch),
            ("PrefillBatchScratch.dn_qkv_batch", self.dn_qkv_batch),
            ("PrefillBatchScratch.dn_z_batch", self.dn_z_batch),
            ("PrefillBatchScratch.dn_alpha_batch", self.dn_alpha_batch),
            ("PrefillBatchScratch.dn_beta_batch", self.dn_beta_batch),
            ("PrefillBatchScratch.dn_q_raw_batch", self.dn_q_raw_batch),
            ("PrefillBatchScratch.dn_k_raw_batch", self.dn_k_raw_batch),
            ("PrefillBatchScratch.dn_v_batch", self.dn_v_batch),
            ("PrefillBatchScratch.dn_q_batch", self.dn_q_batch),
            ("PrefillBatchScratch.dn_k_batch", self.dn_k_batch),
            (
                "PrefillBatchScratch.dn_attn_out_batch",
                self.dn_attn_out_batch,
            ),
            ("PrefillBatchScratch.dn_normed_batch", self.dn_normed_batch),
            ("PrefillBatchScratch.gate_ffn_batch", self.gate_ffn_batch),
            ("PrefillBatchScratch.up_batch", self.up_batch),
            (
                "PrefillBatchScratch.ffn_hidden_batch",
                self.ffn_hidden_batch,
            ),
            (
                "PrefillBatchScratch.dn_normed_rot_batch",
                self.dn_normed_rot_batch,
            ),
            ("PrefillBatchScratch.positions", self.positions),
            ("PrefillBatchScratch.rope_positions", self.rope_positions),
            ("PrefillBatchScratch.tokens", self.tokens),
            ("PrefillBatchScratch.fa_q_full_batch", self.fa_q_full_batch),
            ("PrefillBatchScratch.fa_q_batch", self.fa_q_batch),
            ("PrefillBatchScratch.fa_gate_batch", self.fa_gate_batch),
            ("PrefillBatchScratch.fa_k_batch", self.fa_k_batch),
            ("PrefillBatchScratch.fa_v_batch", self.fa_v_batch),
            (
                "PrefillBatchScratch.fa_attn_out_batch",
                self.fa_attn_out_batch,
            ),
            (
                "PrefillBatchScratch.fa_attn_out_rot_batch",
                self.fa_attn_out_rot_batch,
            ),
        ] {
            free_tensor_retained(label, t, gpu, &mut failures);
        }
        // Optional MoE/tree-verify scratch — only present for MoE/LA configs.
        for (label, t) in [
            (
                "PrefillBatchScratch.moe_router_logits_batch",
                self.moe_router_logits_batch,
            ),
            (
                "PrefillBatchScratch.moe_shared_scalar_batch",
                self.moe_shared_scalar_batch,
            ),
            (
                "PrefillBatchScratch.moe_shared_gate_batch",
                self.moe_shared_gate_batch,
            ),
            (
                "PrefillBatchScratch.moe_shared_up_batch",
                self.moe_shared_up_batch,
            ),
            (
                "PrefillBatchScratch.moe_shared_rot_batch",
                self.moe_shared_rot_batch,
            ),
            (
                "PrefillBatchScratch.moe_topk_indices_batch",
                self.moe_topk_indices_batch,
            ),
            (
                "PrefillBatchScratch.moe_topk_weights_batch",
                self.moe_topk_weights_batch,
            ),
            ("PrefillBatchScratch.moe_gate_batch", self.moe_gate_batch),
            ("PrefillBatchScratch.moe_up_batch", self.moe_up_batch),
            ("PrefillBatchScratch.moe_rot_batch", self.moe_rot_batch),
            (
                "PrefillBatchScratch.moe_down_expanded_batch",
                self.moe_down_expanded_batch,
            ),
            (
                "PrefillBatchScratch.moe_expert_token_counts",
                self.moe_expert_token_counts,
            ),
            (
                "PrefillBatchScratch.moe_expert_offsets",
                self.moe_expert_offsets,
            ),
            (
                "PrefillBatchScratch.moe_sorted_slot_index",
                self.moe_sorted_slot_index,
            ),
            (
                "PrefillBatchScratch.moe_inverse_perm",
                self.moe_inverse_perm,
            ),
            (
                "PrefillBatchScratch.moe_expert_tile_ids",
                self.moe_expert_tile_ids,
            ),
            (
                "PrefillBatchScratch.moe_y_gate_up_grouped",
                self.moe_y_gate_up_grouped,
            ),
            (
                "PrefillBatchScratch.moe_y_down_grouped",
                self.moe_y_down_grouped,
            ),
            ("PrefillBatchScratch.dn_s_tape_q8", self.dn_s_tape_q8),
            (
                "PrefillBatchScratch.dn_s_tape_scales",
                self.dn_s_tape_scales,
            ),
            ("PrefillBatchScratch.dn_s_tape_f32", self.dn_s_tape_f32),
        ] {
            if let Some(t) = t {
                free_tensor_retained(label, t, gpu, &mut failures);
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType;

    // ── SP2 — per-expert mixed-tier table builder (CPU-pure) ──────────────
    // `mixed_tier_table` is the testable core of `per_expert_tier_tables`:
    // empty/uniform columns collapse to None (uniform fast path), only a
    // genuinely multi-tier column yields Some(table).
    #[test]
    fn mixed_tier_table_empty_is_none() {
        // Paged mode: no resident experts → uniform fast path.
        assert_eq!(mixed_tier_table(Vec::new()), None);
    }

    #[test]
    fn mixed_tier_table_uniform_is_none() {
        // The common case: every expert one tier → None → byte-identical
        // uniform path, no allocation surfaced to MoeDtypes.
        let tiers = vec![DType::MQ4G256; 4];
        assert_eq!(mixed_tier_table(tiers), None);
        // Single-expert uniform column is also None.
        assert_eq!(mixed_tier_table(vec![DType::MQ6G256]), None);
    }

    #[test]
    fn mixed_tier_table_mixed_is_some_preserving_order() {
        // A re-quant overlay bumped experts 1 and 3 to MQ6 → Some, and the
        // table preserves per-expert order/dtype so dispatch buckets correctly.
        let tiers = vec![
            DType::MQ4G256,
            DType::MQ6G256,
            DType::MQ4G256,
            DType::MQ6G256,
        ];
        assert_eq!(mixed_tier_table(tiers.clone()), Some(tiers));
    }

    #[test]
    fn mixed_tier_table_mixed_first_differs() {
        // Guard against an off-by-one where only expert[0] is compared:
        // here every later expert differs from expert[0].
        let tiers = vec![DType::MQ4G256, DType::MQ6G256, DType::MQ6G256];
        assert_eq!(mixed_tier_table(tiers.clone()), Some(tiers));
    }

    #[test]
    fn dtype_from_quant_type_neutral_v2_one_to_one() {
        // Each qt maps one-to-one to its V2 DType and exact block bytes;
        // V2 DTypes are distinct from legacy MQ2/3/5/6.
        assert_eq!(dtype_from_quant_type(47).unwrap(), DType::MQ6G256V2);
        assert_eq!(dtype_from_quant_type(48).unwrap(), DType::MQ5G256V2);
        assert_eq!(dtype_from_quant_type(49).unwrap(), DType::MQ3G256V2);
        assert_eq!(dtype_from_quant_type(50).unwrap(), DType::MQ2G256V2);
        // Legacy unchanged.
        assert_eq!(dtype_from_quant_type(15).unwrap(), DType::MQ6G256);
        assert_ne!(dtype_from_quant_type(47).unwrap(), DType::MQ6G256);
        assert_ne!(dtype_from_quant_type(49).unwrap(), DType::MQ3G256);
        // Bad qt fails closed.
        assert!(dtype_from_quant_type(99).is_err());
        // qt44/45 still map to their V2/C DTypes.
        assert_eq!(dtype_from_quant_type(44).unwrap(), DType::MQ4G256V2);
        assert_eq!(dtype_from_quant_type(45).unwrap(), DType::MQ4CG256);
    }

    // ── STEP-002 Task 9: model-owned expert-group plans + plan cache ──────

    use hipfire_runtime::llama::EmbeddingFormat;
    use hipfire_runtime::moe_plan::MoEExecutionPolicy;

    /// Build a minimal WeightTensor with a given dtype and zero-sized buffer.
    fn dummy_wt(dtype: DType) -> WeightTensor {
        let mut buf = GpuTensor::null_for_test();
        buf.shape = vec![1, 1];
        WeightTensor {
            buf,
            gpu_dtype: dtype,
            m: 1,
            k: 1,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    }

    /// A minimal MoE Qwen35Config with dims matching the plan machinery.
    fn task9_moe_config() -> Qwen35Config {
        let inner = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": 1000,
            "layer_types": ["full_attention"],
            "num_experts": 8,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 96,
        });
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": inner}).to_string())
            .unwrap()
    }

    /// A minimal dense (non-MoE) Qwen35Config for the no-group resolution.
    fn task9_dense_config() -> Qwen35Config {
        let inner = serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": 1000,
            "layer_types": ["full_attention"],
        });
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": inner}).to_string())
            .unwrap()
    }

    /// A test weights object with an empty plan cache cell.
    fn plan_cache_weights() -> Qwen35Weights {
        let nt = || GpuTensor::null_for_test();
        Qwen35Weights {
            token_embd: nt(),
            embd_format: EmbeddingFormat::F32,
            output_norm: nt(),
            output: dummy_wt(DType::F32),
            layers: vec![],
            moe_has_mq6: false,
            pager: None,
            lm_head_aliases_embd: false,
            moe_resident: None,
            moe_group_plans: std::sync::OnceLock::new(),
            ep_shard: None,
        }
    }

    /// A task-9-shaped MoE config differing from [`task9_moe_config`] in
    /// exactly one key dimension (the named field), valid in isolation.
    fn key_variant_config(field: &str) -> Qwen35Config {
        let inner = match field {
            "n_layers" => serde_json::json!({
                "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                "vocab_size": 1000, "layer_types": ["full_attention", "full_attention"],
                "num_experts": 8, "num_experts_per_tok": 8,
                "moe_intermediate_size": 128, "shared_expert_intermediate_size": 96,
            }),
            "layer_types" => serde_json::json!({
                "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
                "vocab_size": 1000, "layer_types": ["linear_attention"],
                "num_experts": 8, "num_experts_per_tok": 8,
                "moe_intermediate_size": 128, "shared_expert_intermediate_size": 96,
            }),
            "num_experts" => serde_json::json!({
                "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
                "vocab_size": 1000, "layer_types": ["full_attention"],
                "num_experts": 16, "num_experts_per_tok": 8,
                "moe_intermediate_size": 128, "shared_expert_intermediate_size": 96,
            }),
            "dim" => serde_json::json!({
                "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
                "vocab_size": 1000, "layer_types": ["full_attention"],
                "num_experts": 8, "num_experts_per_tok": 8,
                "moe_intermediate_size": 128, "shared_expert_intermediate_size": 96,
            }),
            "moe_intermediate_size" => serde_json::json!({
                "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
                "vocab_size": 1000, "layer_types": ["full_attention"],
                "num_experts": 8, "num_experts_per_tok": 8,
                "moe_intermediate_size": 256, "shared_expert_intermediate_size": 96,
            }),
            // Resolution FAILURE fixture: dim=0 makes the manifest router
            // entry zero-dimensional; the plan validator refuses it.
            "dim_zero" => serde_json::json!({
                "hidden_size": 0, "num_hidden_layers": 1, "num_attention_heads": 4,
                "vocab_size": 1000, "layer_types": ["full_attention"],
                "num_experts": 8, "num_experts_per_tok": 8,
                "moe_intermediate_size": 128, "shared_expert_intermediate_size": 96,
            }),
            other => panic!("unknown key variant: {other}"),
        };
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": inner}).to_string())
            .unwrap()
    }

    #[test]
    fn qwen35_model_owned_plans_resolve_through_manifest_authority() {
        let cfg = task9_moe_config();
        let plans = Qwen35MoeGroupPlans::resolve(&cfg).unwrap();
        assert_eq!(plans.len(), 1, "one plan per MoE layer");
        // Independent resolution through the manifest authority must agree
        // field-for-field with the model-owned plan.
        let policy = MoEExecutionPolicy::single();
        let specs = qwen35_moe_expert_group_specs(&cfg, &policy);
        let manifest =
            <crate::arch::Qwen35 as hipfire_runtime::arch::Architecture>::weight_manifest(&cfg);
        let resolved =
            hipfire_runtime::weight_manifest::resolve_expert_group_plans(&specs, &manifest, 1)
                .unwrap();
        let owned = plans.by_layer(0);
        assert_eq!(owned.group, resolved[0].group);
        assert_eq!(owned.layer, resolved[0].layer);
        assert_eq!(owned.n_experts, resolved[0].n_experts);
        assert_eq!(owned.group_size, resolved[0].group_size);
        assert_eq!(owned.parallelism, resolved[0].parallelism);
        assert_eq!(owned.assignment, resolved[0].assignment);
        assert_eq!(owned.experts, resolved[0].experts);
        assert_eq!(owned.source_layout, resolved[0].source_layout);
        assert_eq!(owned.resources, resolved[0].resources);
        assert_eq!(owned.router, resolved[0].router);
        assert_eq!(owned.router_identity, resolved[0].router_identity);
        assert_eq!(owned.allowed_executions, resolved[0].allowed_executions);
        assert_eq!(owned.collective, resolved[0].collective);
        // Dense configs resolve no groups at all.
        let dense = task9_dense_config();
        assert_eq!(Qwen35MoeGroupPlans::resolve(&dense).unwrap().len(), 0);
    }

    #[test]
    fn qwen35_model_owned_plans_are_reused_not_rebuilt() {
        let cfg = task9_moe_config();
        let plans = Qwen35MoeGroupPlans::resolve(&cfg).unwrap();
        let a = plans.by_layer(0) as *const hipfire_runtime::weight_manifest::ExpertGroupPlan;
        let b = plans.by_layer(0) as *const hipfire_runtime::weight_manifest::ExpertGroupPlan;
        assert!(
            std::ptr::eq(a, b),
            "by_layer must borrow the same immutable plan, never rebuild it"
        );
        let weights = plan_cache_weights();
        let p1 = weights.moe_group_plans(&cfg).unwrap();
        let p2 = weights.moe_group_plans(&cfg).unwrap();
        assert!(
            std::ptr::eq(
                p1 as *const Qwen35MoeGroupPlans,
                p2 as *const Qwen35MoeGroupPlans
            ),
            "weights.moe_group_plans must resolve once and reuse the same state"
        );
        assert_eq!(p1.len(), 1);
        assert_eq!(p1.by_layer(0).group, "qwen35_moe_layer_0");
    }

    #[test]
    fn qwen35_plan_cache_refuses_config_identity_mismatch() {
        let weights = plan_cache_weights();
        let base = task9_moe_config();
        let p0 = weights
            .moe_group_plans(&base)
            .expect("the base config resolves through the authority");
        for field in [
            "n_layers",
            "layer_types",
            "num_experts",
            "dim",
            "moe_intermediate_size",
        ] {
            let variant = key_variant_config(field);
            let independent = Qwen35MoeGroupPlans::resolve(&variant)
                .expect("variant resolves in isolation through the authority");
            assert!(
                independent.len() >= 1,
                "variant {field} must resolve a non-empty plan set in isolation"
            );
            let err = weights.moe_group_plans(&variant).expect_err(
                "a different config identity must be refused, never served stale plans",
            );
            assert!(
                err.contains("config identity mismatch"),
                "variant {field}: expected the cache identity mismatch, got: {err}"
            );
        }
        assert!(
            std::ptr::eq(
                p0 as *const Qwen35MoeGroupPlans,
                weights.moe_group_plans(&base).unwrap() as *const Qwen35MoeGroupPlans
            ),
            "the refused mismatch must not replace or invalidate the cached entry"
        );
    }

    #[test]
    fn qwen35_plan_cache_replays_cached_failure_for_same_key() {
        let weights = plan_cache_weights();
        let failing = key_variant_config("dim_zero");
        let e1 = weights
            .moe_group_plans(&failing)
            .expect_err("dim=0 must fail plan resolution");
        let entry_a =
            weights.moe_group_plans.get().unwrap() as *const Qwen35MoeGroupPlansCacheEntry;
        let e2 = weights
            .moe_group_plans(&failing)
            .expect_err("the cached failure must replay, not re-resolve");
        assert_eq!(e1, e2, "the cached failure is replayed verbatim");
        let entry_b =
            weights.moe_group_plans.get().unwrap() as *const Qwen35MoeGroupPlansCacheEntry;
        assert!(
            std::ptr::eq(entry_a, entry_b),
            "repeated same-key failure calls must reuse the same cache-entry address"
        );
        let base = task9_moe_config();
        let err = weights
            .moe_group_plans(&base)
            .expect_err("a different identity after a cached failure must be refused");
        assert!(
            err.contains("config identity mismatch"),
            "expected the cache identity mismatch, got: {err}"
        );
    }

    #[test]
    fn qwen35_plan_cache_same_key_lookup_constructs_no_owned_key() {
        let _seam = plan_key_seam::SeamGuard::on();
        let weights = plan_cache_weights();
        let cfg = task9_moe_config();
        let _ = weights
            .moe_group_plans(&cfg)
            .expect("the base config resolves through the authority");
        plan_key_seam::reset();
        plan_key_seam::INSTRUMENT.store(true, std::sync::atomic::Ordering::Relaxed);
        for _ in 0..4 {
            let plans = weights
                .moe_group_plans(&cfg)
                .expect("same-key lookup borrows the cached entry");
            assert_eq!(plans.len(), 1);
        }
        let constructions = plan_key_seam::CONSTRUCTIONS.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            constructions, 0,
            "initialized same-key lookups must not construct the owned key (no layer_types clone)"
        );
    }

    /// Build a shared-expert with all three projections at the given dtype.
    fn dummy_shared(dtype: DType) -> SharedExpertWeights {
        SharedExpertWeights {
            gate: dummy_wt(dtype),
            up: dummy_wt(dtype),
            down: dummy_wt(dtype),
        }
    }

    /// Build a minimal MoeFfnWeights suitable for dtype-level predicate
    /// testing. The caller sets the expert slice to control per-expert dtypes.
    fn dummy_moe_ffn(
        router_dtype: DType,
        shared_expert_gate_dtype: DType,
        shared: SharedExpertWeights,
        experts: Vec<ExpertWeights>,
        dtype_tags: Option<GpuTensor>,
    ) -> MoeFfnWeights {
        let n_exp = experts.len();
        MoeFfnWeights {
            router: dummy_wt(router_dtype),
            experts,
            shared_expert: shared,
            shared_expert_gate: dummy_wt(shared_expert_gate_dtype),
            expert_gate_up_ptrs: {
                let mut t = GpuTensor::null_for_test();
                t.shape = vec![2 * n_exp.max(1)];
                t
            },
            expert_down_ptrs: {
                let mut t = GpuTensor::null_for_test();
                t.shape = vec![2 * n_exp.max(1)];
                t
            },
            expert_down_awq_ptrs: None,
            expert_dtype_tags: dtype_tags,
            layer_idx: 0,
            expert_shape: None,
            paro_shared: None,
            packed_expert_owners: None,
            global_expert_dtypes: None,
            ep_dummy_buffers: Vec::new(),
        }
    }

    /// A one-layer MoE weights object with the given FFN storage, null tensor
    /// buffers (never GPU-touched) and an EMPTY plan cache cell.
    fn view_probe_weights(storage: MoeFfnStorage) -> Qwen35Weights {
        let nt = || GpuTensor::null_for_test();
        let wt = dummy_wt;
        Qwen35Weights {
            token_embd: nt(),
            embd_format: EmbeddingFormat::F32,
            output_norm: nt(),
            output: wt(DType::F32),
            layers: vec![LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                attn_norm: nt(),
                wqkv: wt(DType::MQ4G256),
                wz: wt(DType::MQ4G256),
                w_alpha: wt(DType::MQ4G256),
                w_beta: wt(DType::MQ4G256),
                a_log: nt(),
                dt_bias: nt(),
                conv_weight: nt(),
                norm_weight: nt(),
                wo: wt(DType::MQ4G256),
                ffn_norm: nt(),
                ffn: storage,
            })],
            moe_has_mq6: false,
            pager: None,
            lm_head_aliases_embd: false,
            moe_resident: None,
            moe_group_plans: std::sync::OnceLock::new(),
            ep_shard: None,
        }
    }

    #[test]
    fn qwen35_moe_ffn_view_frozen_reachability_refusal_cell() {
        // The default-on lowered decode path (Qwen35Bindings::run_moe)
        // resolves MoE views through `Qwen35Weights::moe_ffn_view` — the same
        // authority the hand decode arms use — so Frozen storage reachability
        // is pinned by THIS seam, not by a bindings-side storage match.
        //
        // 1. Frozen marker WITHOUT a resident → the explicit binding error
        //    (refusal cell): never a panic, never a fabricated Legacy view.
        let frozen = view_probe_weights(MoeFfnStorage::Frozen);
        let err = match frozen.moe_ffn_view(0) {
            Ok(_) => panic!("Frozen storage without a resident must refuse the view bind"),
            Err(e) => e,
        };
        assert!(
            matches!(
                err,
                Qwen35MoeBindError::TensorLookup(ref name, _) if name == "moe_resident"
            ),
            "the Frozen-without-resident refusal must name the missing resident, got: {err:?}"
        );
        // 2. Legacy storage → Legacy view binds (the bindings' sealed path).
        let legacy = view_probe_weights(MoeFfnStorage::Legacy(dummy_moe_ffn(
            DType::MQ4G256,
            DType::MQ4G256,
            dummy_shared(DType::MQ4G256),
            vec![],
            None,
        )));
        assert!(
            matches!(legacy.moe_ffn_view(0), Ok(MoeFfnView::Legacy(_))),
            "Legacy storage must bind a Legacy view through the same seam"
        );
        // 3. Dense/non-MoE layer index → LayerOutOfRange (unchanged refusal).
        let dense = Qwen35Weights {
            layers: vec![],
            ..view_probe_weights(MoeFfnStorage::Frozen)
        };
        assert!(matches!(
            dense.moe_ffn_view(0),
            Err(Qwen35MoeBindError::LayerOutOfRange {
                requested: 0,
                count: 0
            })
        ));
    }

    #[test]
    fn qwen35_plan_cache_resolves_exactly_once_under_contention() {
        let weights = plan_cache_weights();
        let cfg = task9_moe_config();
        let cell_ref: &std::sync::OnceLock<Qwen35MoeGroupPlansCacheEntry> =
            &weights.moe_group_plans;
        let cfg_ref = &cfg;
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let count = std::sync::Arc::clone(&count);
                let barrier = std::sync::Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    let _ = Qwen35Weights::moe_group_plans_with(cell_ref, cfg_ref, |cfg| {
                        count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Qwen35MoeGroupPlans::resolve(cfg)
                    });
                });
            }
        });
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "exactly one resolution under concurrent first calls"
        );
        let plans = weights
            .moe_group_plans(&cfg)
            .expect("the winning config identity still borrows the entry");
        assert_eq!(plans.len(), 1);
    }
}
