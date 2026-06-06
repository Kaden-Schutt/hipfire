// SPDX-License-Identifier: MIT OR Apache-2.0
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
    pub shared_gate: DType,          // ffn.shared_expert_gate
    pub shared_expert_gate: DType,   // ffn.shared_expert.gate
    pub shared_expert_up: DType,     // ffn.shared_expert.up
    pub experts_all_gate_up_mq4: bool,
    pub routed_gate_up: DType,       // ffn.experts[0].gate_up
    pub routed_down: DType,          // ffn.experts[0].down
    pub has_paro_shared: bool,       // ffn.paro_shared.is_some()
}

/// Resolved fused-vs-fallback eligibility for one MoE decode layer. This IS the
/// routing-config logic, relocated from `moe_ffn_decode_impl` into one typed,
/// testable place (review finding #1). Pure function of `MoeDtypes` + k.
#[derive(Clone, Copy, Debug)]
pub struct MoeResolution {
    pub gate_side_mq4: bool,
    pub routed_indexable_mq4: bool,
    pub routed_indexable_mq6: bool,
    pub routed_indexable_paro: bool,
    pub use_gpu_topk: bool,
    pub needs_x_rot_local: bool,
}

impl MoeResolution {
    pub fn resolve(d: &MoeDtypes, k: usize) -> Self {
        use DType::*;
        let gate_side_mq4 = d.router == MQ4G256
            && d.shared_gate == MQ4G256
            && d.shared_expert_gate == MQ4G256
            && d.shared_expert_up == MQ4G256
            && d.experts_all_gate_up_mq4;

        let routed_gate_up_mq4 = d.routed_gate_up == MQ4G256;
        let routed_gate_up_mq6 = d.routed_gate_up == MQ6G256;
        let routed_gate_up_paro = d.routed_gate_up == ParoQ4G128 && d.has_paro_shared;

        let routed_indexable_mq4 = (d.routed_down == MQ4G256) && routed_gate_up_mq4;
        let routed_indexable_mq6 = (d.routed_down == MQ6G256) && routed_gate_up_mq6;
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;

        let routed_dtype_indexable =
            routed_indexable_mq4 || routed_indexable_mq6 || routed_indexable_paro;

        let use_gpu_topk = k == 8 && routed_dtype_indexable;
        let needs_x_rot_local = gate_side_mq4
            || routed_gate_up_mq4
            || routed_gate_up_mq6
            || routed_gate_up_paro;

        Self {
            gate_side_mq4,
            routed_indexable_mq4,
            routed_indexable_mq6,
            routed_indexable_paro,
            use_gpu_topk,
            needs_x_rot_local,
        }
    }

    pub fn routed_indexable(&self) -> bool {
        self.routed_indexable_mq4 || self.routed_indexable_mq6 || self.routed_indexable_paro
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
    // activations / residual
    pub x_norm: &'a GpuTensor,
    pub x_residual: &'a GpuTensor,
    // gate-side weights
    pub router: WeightRef<'a>,
    pub shared_expert_gate: WeightRef<'a>,
    pub shared_gate_w: WeightRef<'a>,
    pub shared_up_w: WeightRef<'a>,
    pub shared_down_w: WeightRef<'a>,
    // routed expert pointer tables + dims
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
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
    pub scores: &'a GpuTensor,    // post-sqrt_softplus gate·x (weights use these)
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
    Hash { tid2eid: &'a GpuTensor, tokens: &'a GpuTensor },
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
    pub scores: &'a GpuTensor,       // post-sqrt_softplus moe_scores_batch [B, n_exp]
    pub topk_indices: &'a GpuTensor, // [B, k_top] (routing out, expert in)
    pub topk_weights: &'a GpuTensor, // [B, k_top]
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    // activation / residual
    pub x_rot: &'a GpuTensor,        // ffn_x_rot_batch [B, hidden]
    pub ffn_out: &'a GpuTensor,      // ffn_out_batch [B, hidden] (accumulate target)
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
        // MQ6 grouped-WMMA (`gemm_hfq6g256_moe_grouped_wmma`) is gfx12-only
        // (no gfx11 variant yet). Fall back to Path 1 (indexed batched GEMV)
        // on gfx11 to avoid the gfx12-only kernel panic. Path 1 MQ6 indexed
        // kernels exist on all WMMA archs.
        let mq6_on_non_gfx12 = d.routed_gate_up == DType::MQ6G256
            && !(arch.is_gfx1200() || arch.is_gfx1201());
        let use_path2 = use_path2 && !mq6_on_non_gfx12;
        // Path 0: gfx9* wave64 archs (gfx906/gfx908/gfx94x) — cheap HBM
        // atomics make the atomic GEMV pattern competitive vs expanded scratch.
        let down_path0 = arch.is_gcn5() || arch.is_cdna1() || arch.is_cdna3();
        let is_gfx1151 = arch.is_gfx1151();
        let use_paro_i8 = paro_mode && use_path2 && is_gfx1151
            && flags.moe_paro_i8.unwrap_or(true);
        let use_paro_i8_k8 = use_paro_i8
            && flags.moe_paro_i8_k8.unwrap_or(true);
        Self { use_path2, down_path0, use_paro_i8, use_paro_i8_k8, paro_mode }
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
        registry.validate().expect("moe kernel table has empty entries");
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
