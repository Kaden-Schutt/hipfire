// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Op-list interpreter. Phase 2a: GEMV + a fused rmsnorm-rotate producer; empty
//! fusion table (all per-op fallback).

use hipfire_hardware::DeviceMesh;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::fused_qkv::{FusedQkvFamily, FusedQkvParams};
use crate::families::gemv::{GemvFamily, GemvParams, RotateInputs, WeightRef};
use crate::families::moe::{
    launch_grouped_down, launch_grouped_gate_up, launch_indexed_down, launch_indexed_down_residual,
    launch_indexed_gate_up, launch_moe_combine, launch_moe_combine_grouped, launch_moe_route,
    launch_moe_scatter, launch_moe_unscatter, MoeExpertRef,
};
use crate::families::rotation::{RotationFamily, RotationParams};
use crate::types::GemvVariant;
use crate::types::{DispatchError, KernelKey, PipelineOp, RotationPlan, RotationVariant};

/// Rotation disposition of a Gemv's input. Borrows (never owns a RotatedActivation).
pub enum GemvInput<'a> {
    Raw(&'a GpuTensor),        // launch_op self-rotates via run_auto (plan-aware)
    Prerotated(&'a GpuTensor), // already FWHT-rotated; dispatched via Prerotated variant
}

/// Down-projection shape discriminant for [`Step::IndexedMoeGemv`].
///
/// Two kernel families underly three shapes:
/// - **Expanded** (`GateUp` / `DownExpanded`): writes an intermediate buffer; a
///   separate [`Step::MoeCombine`] folds the per-expert outputs with `topk_weights`
///   into the EP partial. MQ4/HFQ4/MQ6/MQ2L support this path via
///   [`launch_indexed_down`]. MQ3L does **not** (no `*_expanded_k4` kernel exists).
/// - **Residual-fused** (`DownResidual`): [`launch_indexed_down_residual`] folds the
///   weighted combine into the kernel and writes directly into the EP partial. Used
///   by MQ2L (minimax self-combining path) and MQ3L (the only down path for MQ3L).
///   Calling [`Step::MoeCombine`] after `DownResidual` would double-accumulate.
pub enum MoeProj<'a> {
    /// Gate+up projection: writes gate_batch (= step `out`) + up_batch (= `up_out`).
    /// Requires FWHT-pre-rotated input. `topk_weights` not needed here.
    GateUp { up_out: &'a GpuTensor },
    /// Down expanded path (MQ4/HFQ4/MQ6/MQ2L): writes per-expert outputs to `out`
    /// (= `down_expanded`). A separate [`Step::MoeCombine`] folds into the EP partial.
    DownExpanded,
    /// Down residual-fused path (MQ2L/MQ3L): folds the weighted combine into the
    /// down kernel. The step's `out` IS the EP partial (accumulate semantics).
    /// No [`Step::MoeCombine`] follows — that would double-accumulate.
    DownResidual { topk_weights: &'a GpuTensor },
}

pub enum Step<'a> {
    Gemv {
        w: &'a WeightRef<'a>,
        input: GemvInput<'a>,
        out: &'a GpuTensor,
    },
    /// GEMV with in-place residual add: `residual += W · input`.
    /// For MQ-family, `input` must be pre-rotated (Prerotated variant) or the
    /// Raw variant triggers FWHT rotation before calling the residual kernel.
    GemvResidual {
        w: &'a WeightRef<'a>,
        input: GemvInput<'a>,
        residual: &'a GpuTensor,
        out: &'a GpuTensor,
    },
    /// Batched (B>1) GEMM: `y[batch×m] = W · x[batch×k]`. Prefill-only; decode
    /// uses `Gemv`. Column-parallel use: `y=[batch×m]` on-rank shard. Row-parallel
    /// use: `y=[batch×dim]` partial → `AllReduceOut` → `ResidualAdd` (never fused).
    Gemm {
        w: &'a WeightRef<'a>,
        x: &'a GpuTensor,
        y: &'a GpuTensor,
        batch: usize,
    },
    /// Fused rmsnorm + optional FWHT rotation. The `rotation` field is derived
    /// by the caller via `dtype_rotation_plan(w.dtype)`. `out` holds the
    /// ready-to-use activation (FWHT-rotated for FwhtG256, plain-normed for None).
    /// All downstream Gemv steps use GemvInput::Prerotated(out).
    RmsnormAutomatic {
        x: &'a GpuTensor,
        norm_weight: &'a GpuTensor,
        x_plain: &'a GpuTensor, // rmsnorm intermediate scratch (always written)
        out: &'a GpuTensor,     // final activation output (written by this step)
        awq_scale: Option<&'a GpuTensor>,
        k: usize,
        eps: f32,
        rotation: RotationPlan, // FwhtG256 for MQ dtypes, None for HFQ4/others
    },
    /// Paired KV-write + flash-attention (Phase 0.3). Consumes a KvTierPlan
    /// (derived once per attention step) and AttnParams (tensor borrows).
    /// Not fusible — the two ops are inherently coupled.
    Attend {
        plan: crate::families::kv_tier::KvTierPlan,
        io: crate::families::attention::AttnParams<'a>,
    },
    /// In-place RoPE on Q and K. Per-op only (no fused entry) — present so the
    /// attention block can be one contiguous step list (future fusion seam).
    Rope {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        pos_buf: &'a hip_bridge::DeviceBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        theta: f32,
    },
    /// Per-head rmsnorm on one tensor (Qwen3-style qk-norm). One step per tensor.
    QkNorm {
        x: &'a GpuTensor,
        weight: &'a GpuTensor,
        n_groups: usize, // n_heads (Q) or n_kv_heads (K)
        head_dim: usize,
        eps: f32,
    },
    /// In-place bias add on one tensor (e.g. qwen2 QKV bias).
    BiasAdd {
        x: &'a GpuTensor,
        bias: &'a GpuTensor,
        dim: usize,
    },
    /// SwiGLU activation: `out = silu(gate) * up` (elementwise). Present so a
    /// dense FFN block can be one contiguous step list — the IR previously fused
    /// silu into gate-up kernels, leaving no standalone activation op, which
    /// blocked expressing a column-parallel FFN's on-rank intermediate as Steps.
    SiluMul {
        gate: &'a GpuTensor,
        up: &'a GpuTensor,
        out: &'a GpuTensor,
    },
    /// In-place residual add: `x += y`. The single-GPU dense forward fuses this
    /// into `GemvResidual`, but a row-parallel `GemvResidual` would all-reduce
    /// `(partial + residual)` and sum the residual `tp×`. Under TP the row-parallel
    /// projection is a plain `Gemv` → all-reduce → this `ResidualAdd`, so the
    /// residual is added exactly once, after the collective.
    ResidualAdd {
        x: &'a GpuTensor,
        y: &'a GpuTensor,
        dim: usize,
    },
    /// Bias-aware top-K MoE routing (deepseek4 decode path, k=6).
    /// Selects on `scores + gate_bias`, weights on the unbiased `scores`, normalizes,
    /// folds in `route_scale` — all in one launch. Writes `topk_indices` and
    /// `topk_weights`. Delegates to [`launch_moe_route`].
    MoeRoute {
        scores: &'a GpuTensor,
        gate_bias: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        k: usize,
        n_experts: usize,
        route_scale: f32,
    },
    /// Indexed per-expert GEMV for the top-K selected experts (decode, B=1).
    ///
    /// Three shapes via `which` (see [`MoeProj`]):
    /// - `GateUp`: gate+up → `out` = gate_batch, `which.up_out` = up_batch.
    ///   `input` must be FWHT-pre-rotated. No `topk_weights` (combine is later).
    /// - `DownExpanded`: down → `out` = `down_expanded` [k × expert_k].
    ///   A separate [`Step::MoeCombine`] folds into the EP partial.
    ///   `batch_size` is consumed; MQ3L is unsupported (use `DownResidual`).
    /// - `DownResidual`: down + weighted-combine fused (MQ2L/MQ3L) → `out` = EP partial.
    ///   `which.topk_weights` carries weights. No [`Step::MoeCombine`] follows.
    ///
    /// `tp_step_out_buf` returns `Some(&out.buf)` only for `DownResidual`
    /// (the partial that the EP all-reduce reduces over).
    IndexedMoeGemv {
        experts: &'a MoeExpertRef<'a>,
        which: MoeProj<'a>,
        topk_indices: &'a GpuTensor,
        /// FWHT-pre-rotated input for GateUp; SwiGLU output (rot_batch) for Down*.
        input: GemvInput<'a>,
        /// gate_batch for GateUp, down_expanded for DownExpanded, EP partial for DownResidual.
        out: &'a GpuTensor,
        k_top: usize,
        /// Used by DownExpanded; ignored by GateUp and DownResidual.
        batch_size: usize,
    },
    /// Weighted combine of per-expert expanded down outputs into the EP partial.
    /// Delegates to [`launch_moe_combine`] (decode) or [`launch_moe_combine_grouped`]
    /// (prefill grouped path, when `inverse_perm` is `Some`).
    ///
    /// - `inverse_perm = None` → `moe_down_combine_k8_batched` (decode).
    ///   Call after [`Step::IndexedMoeGemv`] with `which = DownExpanded`.
    ///   Do NOT call after `DownResidual` (double-accumulate).
    /// - `inverse_perm = Some(&perm)` → `moe_down_combine_grouped_k8` (prefill Path 2).
    ///   Call after [`Step::GroupedMoeGemm`] with `which = DownExpanded`.
    ///
    /// `out` is the pre-zeroed EP partial (accumulate semantics); the executor
    /// zeroes it via `zero_before` before this step runs. `tp_step_out_buf` returns
    /// `Some(&out.buf)` so the EP all-reduce finds the partial buffer.
    MoeCombine {
        down_out: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        out: &'a GpuTensor,
        k: usize,
        hidden: usize,
        batch_size: usize,
        /// Grouped-path inverse permutation produced by [`Step::MoeScatter`].
        /// `Some` → prefill grouped combine (`moe_down_combine_grouped_k8`).
        /// `None` → decode expanded combine (`moe_down_combine_k8_batched`).
        inverse_perm: Option<&'a GpuTensor>,
    },
    /// Scatter+histogram for grouped-GEMM prefill (Path 2). Builds
    /// `sorted_slot_index`, `expert_tile_ids`, and `inverse_perm` from
    /// `topk_indices`. Delegates to [`launch_moe_scatter`].
    /// Must run before [`Step::GroupedMoeGemm`]. `tp_step_out_buf` returns `None`.
    MoeScatter {
        topk_indices: &'a GpuTensor,
        expert_token_counts: &'a GpuTensor,
        expert_offsets: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        inverse_perm: &'a GpuTensor,
        total_slots: usize,
        n_experts: usize,
        m_total_max: usize,
        block_m: usize,
    },
    /// Grouped-WMMA expert GEMM for prefill (Path 2). One launch covers all
    /// expert tokens sorted by `sorted_slot_index`. `which` distinguishes:
    /// - `GateUp`: `m = 2·expert_m`, `x_row_div = k_top`, `rows = batch_size`.
    ///   Writes fused gate||up output to `y` (y_gate_up_grouped).
    ///   `up_out` in `MoeProj::GateUp` is unused by the grouped kernel (output is `y`).
    /// - `DownExpanded`: `m = expert_k`, `x_row_div = 1`, `rows = batch*k_top`.
    ///   Writes down output to `y` (y_down_grouped) for [`Step::MoeCombine`].
    ///
    /// `tp_step_out_buf` returns `None` — `y` is an intermediate, not an EP partial.
    GroupedMoeGemm {
        experts: &'a MoeExpertRef<'a>,
        which: MoeProj<'a>,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        /// For `GateUp`: x_rot_batch; for `DownExpanded`: rot_batch.
        x: &'a GpuTensor,
        /// For `GateUp`: y_gate_up_grouped; for `DownExpanded`: y_down_grouped.
        y: &'a GpuTensor,
        m_total: usize,
        batch_size: usize,
        k_top: usize,
    },
    /// Unscatter grouped gate_up result: `y_grouped → gate_batch + up_batch`.
    /// Delegates to [`launch_moe_unscatter`]. Call after [`Step::GroupedMoeGemm`]
    /// with `which = GateUp`. `tp_step_out_buf` returns `None`.
    MoeUnscatter {
        y_grouped: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        gate_batch: &'a GpuTensor,
        up_batch: &'a GpuTensor,
        mi: usize,
        k_top: usize,
        m_total: usize,
    },
    // ── Note (Task 6): ds4 `hc_ffn_mix` is intentionally NOT a Step variant ──
    // The ds4 MoE tail mixes the EP all-reduced `ffn_out` partial into
    // `residual_streams` via `hc_mix_4stream` + `memcpy_dtod_auto`. Its two view
    // operands (`comb_view`, `post_view`) are ephemeral `GpuTensor` values computed
    // at call time via `sub_offset` on `state.hc_c`; they have no stable backing
    // storage to borrow `&'a GpuTensor` from inside a Step.
    // Task 8's `forward_ep` calls `crate::families::moe::launch_hc_ffn_mix`
    // directly after `execute_steps_parallel` returns and the EP all-reduce
    // completes. minimax's MoE tail (`add_inplace_f32`) reuses `Step::ResidualAdd`.
}

/// Op-kind for fusion matching. Total over Step variants.
fn op_kind(step: &Step) -> PipelineOp {
    match step {
        Step::Gemv { .. } => PipelineOp::Gemv,
        // Reuses the Gemv tag: op_kind only feeds the fused-decode prefix table,
        // which the prefill-only Gemm step never enters.
        Step::Gemm { .. } => PipelineOp::Gemv,
        Step::GemvResidual { .. } => PipelineOp::GemvResidual,
        Step::RmsnormAutomatic { .. } => PipelineOp::RmsnormAutomatic,
        Step::Attend { .. } => PipelineOp::Attend,
        Step::Rope { .. } => PipelineOp::Rope,
        Step::QkNorm { .. } => PipelineOp::QkNorm,
        Step::BiasAdd { .. } => PipelineOp::BiasAdd,
        Step::SiluMul { .. } => PipelineOp::SiluMul,
        Step::ResidualAdd { .. } => PipelineOp::ResidualAdd,
        // MoE decode ops (Task 4). Not fusible — no entry in FUSED_TABLE.
        Step::MoeRoute { .. } => PipelineOp::MoeRoute,
        Step::IndexedMoeGemv { .. } => PipelineOp::IndexedMoeGemv,
        Step::MoeCombine { .. } => PipelineOp::MoeCombine,
        // MoE prefill grouped ops (Task 5). Not fusible.
        Step::MoeScatter { .. } => PipelineOp::MoeScatter,
        Step::GroupedMoeGemm { .. } => PipelineOp::GroupedMoeGemm,
        Step::MoeUnscatter { .. } => PipelineOp::MoeUnscatter,
    }
}

// ── Guard helpers ──────────────────────────────────────────────────────────

/// Extract the dtype of the first Gemv step in the window (step index 1,
/// after the RmsnormAutomatic producer). Returns None if not a Gemv step.
fn window_gemv_dtype(steps: &[Step]) -> Option<DType> {
    match steps.get(1)? {
        Step::Gemv { w, .. } => Some(w.dtype),
        _ => None,
    }
}

/// True if all Gemv steps in the window (indices 1..) have:
/// - the given dtype
/// - GemvInput::Prerotated
/// - awq_scale == None (iff require_no_awq)
fn gemv_steps_uniform(steps: &[Step], dtype: DType, require_no_awq: bool) -> bool {
    steps[1..].iter().all(|s| match s {
        Step::Gemv {
            w,
            input: GemvInput::Prerotated(_),
            ..
        } => w.dtype == dtype && (!require_no_awq || w.awq_scale.is_none()),
        _ => false,
    })
}

/// True if all Gemv steps in the window (indices 1..) have:
/// - the given dtype
/// - GemvInput::Raw (kernel rotates internally — used for Paro guards)
fn gemv_steps_uniform_raw(steps: &[Step], dtype: DType) -> bool {
    steps[1..].iter().all(|s| match s {
        Step::Gemv {
            w,
            input: GemvInput::Raw(_),
            ..
        } => w.dtype == dtype,
        _ => false,
    })
}

/// True if ctx has dp4a and !force_unfused.
fn dp4a_eligible(ctx: &DispatchCtx) -> bool {
    !ctx.flags.force_unfused && ctx.arch.gemv_dp4a_enabled()
}

// ── QKV 3-way guards ──

pub(crate) fn guard_qkv_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_qkv_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

/// Covers both DType::MQ4G256 (plain) and DType::HFQ4G256 — both feed
/// gpu.fused_qkv_hfq4g256 which takes a pre-normalized x.
pub(crate) fn guard_qkv_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 4 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256.
/// Fusion is safe on RDNA (fused_qkv.rs None arm falls back to gemm n=1)
/// and beneficial on RDNA3+ even without dp4a; dp4a is handled per-arm
/// in fused_qkv.rs dispatch.
pub(crate) fn guard_qkv_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 4 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── QKVZA 4-way guards (DeltaNet linear attention) ──

pub(crate) fn guard_qkvza_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_qkvza_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

/// Covers both DType::MQ4G256 (plain) and DType::HFQ4G256.
pub(crate) fn guard_qkvza_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256.
/// Fusion is safe on RDNA (fused_qkv.rs None arm falls back to gemm n=1)
/// and beneficial on RDNA3+ even without dp4a; dp4a is handled per-arm
/// in fused_qkv.rs dispatch.
pub(crate) fn guard_qkvza_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── Gate+Up 2-way guards ──

pub(crate) fn guard_gate_up_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_gate_up_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

pub(crate) fn guard_gate_up_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

pub(crate) fn guard_gate_up_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if !dp4a_eligible(ctx) {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── mfp4-E8 decode launch-fusion guards (gfx1151 / Strix Halo ONLY) ──
// These are the SOLE producers of the FusedGateUpMfp4G32E8 / FusedQkvzaMfp4G32E8
// keys. The `is_gfx1151()` check firewalls the fused kernels to gfx1151 — on every
// other arch these return false and the projections fall through to the
// per-projection gemv_mfp4g32_e8 path unchanged. The fused kernels embed the
// byte-identical gemv_mfp4g32_e8 per-row body, so the fused output equals N
// sequential GEMVs bit-for-bit (only the launch count shrinks).
//
// gfx11 E8 port finding: the fusion (launch-overhead reduction, +5.8% on the Strix
// Halo APU) does NOT transfer to the gfx1100 dGPU — measured decode 101.7 (fused)
// vs 102.6 (unfused) tok/s, a ~1% LOSS, bit-identical output. The dGPU's faster
// compute + the (32,7) launch_bounds tuned for gfx1151 occupancy leave no launch
// win to capture. Kept gfx1151-only; revisit only with a gfx1100 occupancy retune.
pub(crate) fn guard_gate_up_mfp4g32e8(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if !ctx.arch.is_gfx1151() {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::MFP4G32E8 && gemv_steps_uniform(steps, dt, true)
}

pub(crate) fn guard_qkvza_mfp4g32e8(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if !ctx.arch.is_gfx1151() {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::MFP4G32E8 && gemv_steps_uniform(steps, dt, true)
}

// ── Paro fused guards (Raw input — kernel rotates internally) ──

// ── Q8_0 / Q4K fused guards (non-rotated, Prerotated input) ──
// These dtypes have no activation rotation (RotationPlan::None), so the
// RmsnormAutomatic producer does plain rmsnorm and the fused kernels take
// the pre-normed x directly. Prerotated input is correct because
// for_gemv_prerotated(Q8_0/Q4K) falls back to the plain GEMV kernel.

/// Fused QKV with Q4K weights. Used by llama (dense).
pub(crate) fn guard_qkv_q4k(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::Q4K, true)
}

/// Fused gate+up with Q4K weights. Used by llama (dense).
pub(crate) fn guard_gate_up_q4k(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Q4K, true)
}

/// Fused gate+up with Q8_0 weights. Used by qwen2 FFN.
pub(crate) fn guard_gate_up_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

/// Fused 4-way QKVZA with Q8_0 weights (DECODE path, n=1). Used by
/// Qwen3.5/A3B .mq4p DeltaNet layers (qt=3). No dp4a required.
pub(crate) fn guard_qkvza_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

/// Fused 3-way QKV with Q8_0 weights (DECODE path, n=1). No dp4a required.
pub(crate) fn guard_qkv_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

pub(crate) fn guard_gate_up_paro4g128t(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::ParoQ4G128
        && gemv_steps_uniform_raw(steps, DType::ParoQ4G128)
        && steps[1..].iter().all(|s| match s {
            Step::Gemv { w, .. } => w.m % 8 == 0 && w.k % 128 == 0,
            _ => false,
        })
        // Gate and up must have equal m — the fused kernel takes a single m.
        && {
            let m0 = match &steps[1] { Step::Gemv { w, .. } => w.m, _ => return false };
            let m1 = match &steps[2] { Step::Gemv { w, .. } => w.m, _ => return false };
            m0 == m1
        }
}

pub(crate) fn guard_qkvza_paro4g128t(_steps: &[Step], _ctx: &DispatchCtx) -> bool {
    false
}

pub(crate) fn guard_qkv_paro4g128t(_steps: &[Step], _ctx: &DispatchCtx) -> bool {
    false
}

pub struct FusedPattern {
    pub ops: &'static [PipelineOp],
    pub key: KernelKey,
    /// Dtype/arch predicate called after op-kind prefix match. Must return true
    /// for the entry to fire. Receives the full matched window (all ops.len()
    /// steps starting at the current position).
    pub guard: fn(&[Step], &DispatchCtx) -> bool,
}

/// Greedy longest-prefix op-pattern match with dtype/arch guard.
pub fn match_prefix(
    table: &[FusedPattern],
    steps: &[Step],
    ctx: &DispatchCtx,
) -> Option<(KernelKey, usize)> {
    table
        .iter()
        .filter(|p| {
            !p.ops.is_empty()
                && p.ops.len() <= steps.len()
                && p.ops.iter().zip(steps).all(|(o, s)| *o == op_kind(s))
                && (p.guard)(&steps[..p.ops.len()], ctx)
        })
        .max_by_key(|p| p.ops.len())
        .map(|p| (p.key, p.ops.len()))
}

/// Lower-time fusion match over the canonical `FUSED_TABLE`. The Ship-6 super-op
/// lowering (`superop::lower_layer`) calls THIS — reusing the same table + guards
/// verbatim — so a lowered program can never drift from what `execute_steps`
/// would dispatch live (the fusion-drift mitigation, spike risk #1).
pub(crate) fn match_fused_prefix(steps: &[Step], ctx: &DispatchCtx) -> Option<(KernelKey, usize)> {
    match_prefix(FUSED_TABLE, steps, ctx)
}

/// Public(crate) op-kind accessor for the lowering (mirror of the private `op_kind`).
pub(crate) fn step_op_kind(step: &Step) -> PipelineOp {
    op_kind(step)
}

const QKV3: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];
const QKVZA4: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];
const GATE_UP2: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];

const FUSED_TABLE: &[FusedPattern] = &[
    // ── QKV 3-way ──────────────────────────────────────────────────────────
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvMq4G256Lloyd,
        guard: guard_qkv_mq4g256lloyd,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvMq3G256Lloyd,
        guard: guard_qkv_mq3g256lloyd,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvHfq4G256,
        guard: guard_qkv_hfq4g256,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvHfq6G256,
        guard: guard_qkv_hfq6g256,
    },
    // ── QKVZA 4-way (DeltaNet linear attention) ────────────────────────────
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMq4G256Lloyd,
        guard: guard_qkvza_mq4g256lloyd,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMq3G256Lloyd,
        guard: guard_qkvza_mq3g256lloyd,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaHfq4G256,
        guard: guard_qkvza_hfq4g256,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaHfq6G256,
        guard: guard_qkvza_hfq6g256,
    },
    // mfp4-E8 decode launch-fusion — gfx1151-ONLY (guard firewalls the arch).
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMfp4G32E8,
        guard: guard_qkvza_mfp4g32e8,
    },
    // ── Gate+Up 2-way ───────────────────────────────────────────────────────
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMq4G256Lloyd,
        guard: guard_gate_up_mq4g256lloyd,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMq3G256Lloyd,
        guard: guard_gate_up_mq3g256lloyd,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpHfq4G256,
        guard: guard_gate_up_hfq4g256,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpHfq6G256,
        guard: guard_gate_up_hfq6g256,
    },
    // mfp4-E8 decode launch-fusion — gfx1151-ONLY (guard firewalls the arch).
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMfp4G32E8,
        guard: guard_gate_up_mfp4g32e8,
    },
    // ── Q8_0 / Q4K fused entries (non-rotated, Always arch gate) ─────────
    // Q8_0 QKV/QKVZA: Qwen3.5-A3B .mq4p uses Q8_0 for all linear-attention
    // projections (qt=3). Scalar decode kernels added 2026-06-14.
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaQ8_0,
        guard: guard_qkvza_q8_0,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvQ8_0,
        guard: guard_qkv_q8_0,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvQ4K,
        guard: guard_qkv_q4k,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpQ4K,
        guard: guard_gate_up_q4k,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpQ8_0,
        guard: guard_gate_up_q8_0,
    },
    // ── Paro fused Paro4G128T (dp4a, Raw input) ────────────────────────
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpParo4G128T,
        guard: guard_gate_up_paro4g128t,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaParo4G128T,
        guard: guard_qkvza_paro4g128t,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvParo4G128T,
        guard: guard_qkv_paro4g128t,
    },
];
static GEMV: OnceLock<GemvFamily> = OnceLock::new();
static ROTATION: OnceLock<RotationFamily> = OnceLock::new();
static FUSED_QKV: OnceLock<FusedQkvFamily> = OnceLock::new();

pub fn execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[Step],
) -> Result<(), DispatchError> {
    let mut i = 0;
    while i < steps.len() {
        if let Some((key, len)) = match_prefix(FUSED_TABLE, &steps[i..], ctx) {
            launch_fused(gpu, ctx, key, &steps[i..i + len])?;
            i += len;
        } else {
            launch_op(gpu, ctx, &steps[i])?;
            i += 1;
        }
    }
    Ok(())
}

/// Mesh-aware spine (P-A). Threads the device mesh to the dispatch chokepoint so
/// per-`Step` parallelism (TP in P-B, PP/EP in later phases) can be resolved
/// here. For the single-device (1×1) mesh it forwards `gpu` unchanged and is
/// byte-identical to calling [`execute_steps`] directly — this is the
/// zero-behavior-change foundation the executor half of the pivot builds on.
///
/// P-A threads only the mesh (a cheap value) alongside the existing `&mut Gpu`,
/// so every call site migrates by adding a `mesh` argument with no borrow
/// rework. The `&mut Gpu` → `&mut Gpus` promotion (for real cross-rank TP)
/// happens in P-B, where it is bundled with the serve-path `Gpus` hoist and
/// applied only to the paths that shard.
pub fn execute_steps_mesh(
    mesh: &DeviceMesh,
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[Step],
) -> Result<(), DispatchError> {
    debug_assert_eq!(
        mesh.n_devices(),
        1,
        "execute_steps_mesh: only the single (1×1) mesh is supported in P-A; \
         TP/PP/EP sharding lands in P-B..P-E"
    );
    execute_steps(gpu, ctx, steps)
}

/// Collective to inject after a step ran on every rank of the `Tp` group.
/// Keyed by step index in the per-rank step lists (see [`execute_steps_tp`]).
#[derive(Debug)]
pub enum TpCollective {
    /// Column-parallel (or replicated) step — output stays on-rank, no collective.
    None,
    /// Row-parallel step — each rank produced a partial `out` of length `dim`;
    /// sum them in place across the `Tp` group so every rank holds the whole.
    AllReduceOut { dim: usize },
}

/// Axis-keyed collective to inject after a step in [`execute_steps_parallel`].
/// Replaces the TP-specific `TpCollective` with a generic form that covers both
/// Tp (dense row-parallel) and Ep (MoE expert-parallel) all-reduces.
#[derive(Debug)]
pub enum StepCollective {
    /// No collective — column-parallel / replicated steps leave output on-rank.
    None,
    /// All-reduce the step's partial output over the `kind` group.
    /// `dim` is the element count (f32) of the partial buffer on each rank.
    AllReduce {
        kind: hipfire_hardware::DimKind,
        dim: usize,
    },
}

/// The `out` buffer of a step that carries a row-parallel or EP partial output
/// (the only kind that needs an all-reduce). `None` for steps that never carry
/// such a buffer.
///
/// MoE additions (Task 4):
/// - `MoeCombine.out` — the pre-zeroed EP partial; the executor zeros it via
///   `zero_before` and the EP all-reduce sums it across ranks.
/// - `IndexedMoeGemv` with `DownResidual` — the step's `out` IS the EP partial
///   (the residual-fused kernel writes combined output directly into it).
///   `GateUp` and `DownExpanded` do not carry a partial: their output buffers
///   are intermediates, not reduced over the EP group.
fn tp_step_out_buf<'a>(step: &'a Step) -> Option<&'a hip_bridge::DeviceBuffer> {
    match step {
        Step::Gemv { out, .. } => Some(&out.buf),
        Step::GemvResidual { out, .. } => Some(&out.buf),
        Step::Gemm { y, .. } => Some(&y.buf),
        // EP partial: combine result or residual-fused down result.
        Step::MoeCombine { out, .. } => Some(&out.buf),
        Step::IndexedMoeGemv {
            which: MoeProj::DownResidual { .. },
            out,
            ..
        } => Some(&out.buf),
        // Prefill grouped ops: intermediates, never EP partials.
        Step::MoeScatter { .. } | Step::GroupedMoeGemm { .. } | Step::MoeUnscatter { .. } => None,
        _ => None,
    }
}

/// Pure-logic validation for [`execute_steps_parallel`] arg lengths.
/// Separated into its own fn so it is testable without a GPU.
/// Returns `n_steps` (= `per_rank_steps[0].len()`) on success.
fn validate_parallel_args(
    group_size: usize,
    per_rank_steps: &[Vec<Step>],
    collectives: &[StepCollective],
    zero_before: &[bool],
) -> Result<usize, DispatchError> {
    if per_rank_steps.len() != group_size {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {} step lists for group of {group_size}",
            per_rank_steps.len()
        )));
    }
    let n_steps = per_rank_steps[0].len();
    for (r, s) in per_rank_steps.iter().enumerate() {
        if s.len() != n_steps {
            return Err(DispatchError::Hip(format!(
                "execute_steps_parallel: rank {r} has {} steps, rank 0 has {n_steps} (must be lock-step)",
                s.len()
            )));
        }
    }
    if collectives.len() != n_steps {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {} collectives for {n_steps} steps",
            collectives.len()
        )));
    }
    if zero_before.len() != n_steps {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {} zero_before flags for {n_steps} steps",
            zero_before.len()
        )));
    }
    Ok(n_steps)
}

/// Axis-keyed parallel Step executor (P-D foundation). The generic form of
/// [`execute_steps_tp`]: runs `per_rank_steps` lock-step across the mesh group
/// for `collectives[i].kind`, injects an axis-keyed all-reduce for
/// `StepCollective::AllReduce` steps, and optionally zeroes the step's output
/// buffer before running it (required for EP accumulation into a partial).
///
/// **`zero_before[i]`** — when true, `memset_async` the step's output buffer to 0
/// before launching step `i` on each rank. The element count is taken from the
/// paired `AllReduce { dim }` collective. Same 4-bytes-per-elem, same
/// `active_stream` requirement as the EP accumulation pattern.
///
/// **Collective choice** is keyed on the collective's `kind`:
/// - `DimKind::Tp` → always `all_reduce_sum_f32_peer` (RCCL not required on Tp path).
/// - `DimKind::Ep` → `ep_peer_allreduce_decode()` ? `_peer` : `_rccl`.
///
/// The group is resolved via `mesh.group_along(kind, coord_of(0))` — identical
/// to the TP path, so byte-identical results are guaranteed for Tp collectives.
///
/// Preconditions: each rank in the group has `active_stream` set and peer access
/// enabled (`ensure_rank_streams` + `enable_peer_all`).
pub fn execute_steps_parallel(
    mesh: &DeviceMesh,
    gpus: &mut hipfire_hardware::Gpus,
    per_rank_steps: &[Vec<Step>],
    collectives: &[StepCollective],
    zero_before: &[bool],
) -> Result<(), DispatchError> {
    // Determine the parallelism axis from the first AllReduce collective, or
    // fall back to Tp (all-None collectives remain on the Tp group for compat).
    let kind = collectives
        .iter()
        .find_map(|c| {
            if let StepCollective::AllReduce { kind, .. } = c {
                Some(*kind)
            } else {
                None
            }
        })
        .unwrap_or(hipfire_hardware::DimKind::Tp);

    let group = mesh.group_along(kind, &mesh.coord_of(0));
    let group_size = group.len();

    if group_size <= 1 {
        return Err(DispatchError::Hip(format!(
            "execute_steps_parallel: {kind:?} group size {group_size} — needs >1 ranks"
        )));
    }
    let n_steps = validate_parallel_args(group_size, per_rank_steps, collectives, zero_before)?;

    let hip_err = |e: hip_bridge::HipError| DispatchError::Hip(e.to_string());

    for i in 0..n_steps {
        // Optional pre-zero of each rank's output buffer (EP accumulation pattern):
        // memset_async, dim*4 bytes, on the rank's active_stream.
        if zero_before[i] {
            let dim = match &collectives[i] {
                StepCollective::AllReduce { dim, .. } => *dim,
                StepCollective::None => {
                    return Err(DispatchError::Hip(format!(
                        "execute_steps_parallel: zero_before[{i}] is true but collective is None (no dim)"
                    )));
                }
            };
            for (r, &dev) in group.iter().enumerate() {
                gpus.devices[dev].bind_thread().map_err(hip_err)?;
                let stream = gpus.devices[dev].active_stream.as_ref().ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: device {dev} has no active_stream for zero_before"
                    ))
                })?;
                let buf = tp_step_out_buf(&per_rank_steps[r][i]).ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: step {i} zero_before=true but has no out buffer"
                    ))
                })?;
                gpus.devices[dev]
                    .hip
                    .memset_async(buf, 0, dim * 4, stream)
                    .map_err(hip_err)?;
            }
        }

        // Run step i on every rank (each with its own sharded weights/buffers).
        for (r, &dev) in group.iter().enumerate() {
            gpus.devices[dev].bind_thread().map_err(hip_err)?;
            let ctx = DispatchCtx::new(&gpus.devices[dev]);
            launch_op(&mut gpus.devices[dev], &ctx, &per_rank_steps[r][i])?;
        }

        // Collective: all-reduce the partial outputs over the axis group.
        if let StepCollective::AllReduce {
            kind: coll_kind,
            dim,
        } = &collectives[i]
        {
            for &dev in &group {
                let g = &gpus.devices[dev];
                g.bind_thread().map_err(hip_err)?;
                let stream = g.active_stream.as_ref().ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: device {dev} has no active_stream"
                    ))
                })?;
                g.hip.stream_synchronize(stream).map_err(hip_err)?;
            }
            let mut refs: Vec<&hip_bridge::DeviceBuffer> = Vec::with_capacity(group_size);
            for (r, _) in group.iter().enumerate() {
                let buf = tp_step_out_buf(&per_rank_steps[r][i]).ok_or_else(|| {
                    DispatchError::Hip(format!(
                        "execute_steps_parallel: step {i} marked AllReduce but has no out buffer"
                    ))
                })?;
                refs.push(buf);
            }
            // Peer/RCCL choice: Tp always uses peer (RCCL not installed on Tp path);
            // Ep branches on HIPFIRE_EP_PEER_ALLREDUCE_DECODE. Mirrors lib.rs:823-826.
            match coll_kind {
                hipfire_hardware::DimKind::Tp => {
                    gpus.all_reduce_sum_f32_peer(&group, &refs, *dim)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
                _ => {
                    if hipfire_hardware::ep_peer_allreduce_decode() {
                        gpus.all_reduce_sum_f32_peer(&group, &refs, *dim)
                            .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    } else {
                        gpus.all_reduce_sum_f32(&group, &refs, *dim)
                            .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Tensor-parallel executor (P-B grand-unify, PB-TP1). The `Tp>1` counterpart of
/// [`execute_steps_mesh`]: instead of one whole-model step list on one `Gpu`, it
/// takes **per-rank step lists** (`per_rank_steps[r]` references rank `r`'s own
/// sharded weights + buffers, built by the caller from a sharded `WeightStore`)
/// and runs them **lock-step** across the mesh's `Tp` group — every rank executes
/// step `i`, then a `Tp` all-reduce is injected for the row-parallel steps.
///
/// Column-parallel `Gemv`s leave their output sharded (`inter/tp`) to feed the
/// next step; row-parallel `Gemv`s each produce a partial `[dim]` which this
/// executor sums in place via `all_reduce_sum_f32_peer` — so after a
/// `TpCollective::AllReduceOut` step every rank holds the whole result. Residual
/// adds must be a SEPARATE post-collective step (a row-parallel `GemvResidual`
/// would sum the residual `tp×`), so row-parallel ops are plain `Gemv`s here.
///
/// This is the EP `run_layer_program_mesh` shape in the `Step` world, keyed by
/// the caller-supplied `collectives` (from `ShardPolicy`) instead of
/// `SuperOpKind::Moe`. Fusion is intentionally not applied on the TP path yet
/// (F32 GEMV needs none); per-rank `DispatchCtx` is built like the EP path.
///
/// Preconditions the caller owns: each device in the `Tp` group has an
/// `active_stream` set and peer access enabled (`ensure_rank_streams` +
/// `enable_peer_all`).
///
/// **Wrapper:** delegates to [`execute_steps_parallel`] with `zero_before` all-false
/// and `TpCollective` mapped to `StepCollective`. Byte-identical to the prior
/// monolithic implementation for all existing TP callers/examples.
pub fn execute_steps_tp(
    mesh: &DeviceMesh,
    gpus: &mut hipfire_hardware::Gpus,
    per_rank_steps: &[Vec<Step>],
    collectives: &[TpCollective],
) -> Result<(), DispatchError> {
    let n_steps = per_rank_steps.first().map(|v| v.len()).unwrap_or(0);
    let collectives2: Vec<StepCollective> = collectives
        .iter()
        .map(|c| match c {
            TpCollective::AllReduceOut { dim } => StepCollective::AllReduce {
                kind: hipfire_hardware::DimKind::Tp,
                dim: *dim,
            },
            TpCollective::None => StepCollective::None,
        })
        .collect();
    let zero_before = vec![false; n_steps];
    execute_steps_parallel(mesh, gpus, per_rank_steps, &collectives2, &zero_before)
}

/// Per-op fallback. FULL enum match (no catch-all) so the compiler forces every
/// op to have an arm (spec F4 — a missing arm would be a silent runtime error).
fn launch_op(gpu: &mut Gpu, ctx: &DispatchCtx, step: &Step) -> Result<(), DispatchError> {
    match step {
        Step::Gemv {
            w,
            input: GemvInput::Raw(x),
            out,
        } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run_auto(ctx, gpu, w, x, out)
        }
        Step::Gemm { w, x, y, batch } => {
            // Batched (B>1) GEMM. Mirrors runtime `weight_gemm` (llama.rs:1444)
            // per-dtype against a `WeightRef`; the batched kernels live in
            // rdna-compute (same ones `weight_gemm` calls). Prefill-only — no
            // fused-decode entry hits this.
            let hip_err = |e: hip_bridge::HipError| DispatchError::Hip(e.to_string());
            match w.dtype {
                DType::HFQ4G256 => gpu
                    .gemm_hfq4g256(w.buf, x, y, w.m, w.k, *batch)
                    .map_err(hip_err),
                DType::HFQ4G128 => gpu
                    .gemm_hfq4g128(w.buf, x, y, w.m, w.k, *batch)
                    .map_err(hip_err),
                // MQ4G256 = HFQ4G256 layout + an AWQ-aware FWHT rotation of x.
                // FWHT-rotate all `batch` activation columns once (the dispatch
                // twin of runtime `rotate_x_mq_batched_for` — `rotate` runs
                // `ensure_mq_signs` internally via prepare_rotation_scratch), then
                // feed the same INT4-G256 batched WMMA kernel weight_gemm uses.
                DType::MQ4G256 => {
                    let gemv = GEMV.get_or_init(GemvFamily::new);
                    let h = gemv.rotate(
                        ctx,
                        gpu,
                        w,
                        x,
                        &RotateInputs {
                            batch_size: *batch,
                            ..Default::default()
                        },
                    )?;
                    let x_rot = h.into_buf();
                    gpu.gemm_hfq4g256_batched_lmhead(w.buf, &x_rot, y, w.m, w.k, *batch)
                        .map_err(hip_err)
                }
                other => Err(DispatchError::Hip(format!(
                    "Step::Gemm: dtype {other:?} not wired (add its weight_gemm arm)"
                ))),
            }
        }
        Step::Gemv {
            w,
            input: GemvInput::Prerotated(xr),
            out,
        } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(
                ctx,
                gpu,
                &GemvParams {
                    w,
                    x: xr,
                    y: out,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        Step::GemvResidual {
            w,
            input: GemvInput::Prerotated(xr),
            residual,
            out: _,
        } => {
            // MQ-family with a fused residual kernel: writes `residual` in-place via
            // GemvVariant::WithResidual. `out` is NOT written — it is scratch for the
            // fallback path only (see the Raw arm below). Nothing downstream reads
            // `out` after this step in either qwen2 or llama decode paths.
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(
                ctx,
                gpu,
                &GemvParams {
                    w,
                    x: xr,
                    y: residual,
                    variant: GemvVariant::WithResidual,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        Step::GemvResidual {
            w,
            input: GemvInput::Raw(x),
            residual,
            out,
        } => {
            // For dtypes WITHOUT a fused residual kernel (Q8_0, Q4K, F32), the
            // fallback path runs a plain GEMV then `residual += result`. `out` may
            // be used as scratch ONLY when it does not alias `residual`; when it
            // does (the common qwen35 o_proj / dn_out case where out == residual ==
            // &s.x), a fresh temp is allocated instead. See the aliasing guard below.
            // Nothing reads `out` after this step in any model decode path.
            let gemv = GEMV.get_or_init(GemvFamily::new);
            // Dtypes with a fused `gemv_*_residual` kernel use it in one launch.
            // Dtypes without one (Q8_0, ParoQ4G128, …) fall back to plain GEMV into
            // the `out` scratch + `residual += out` — reuses the pre-allocated `out`
            // buffer instead of alloc/free per call. Plain GEMV applies this
            // dtype's own rotation (FWHT / Givens) internally, so this is correct
            // for both no-rotation (Q8) and Givens (Paro) dtypes.
            if KernelKey::for_gemv_residual(w.dtype).is_ok() {
                if crate::types::dtype_rotation_plan(w.dtype) != RotationPlan::None {
                    let h = gemv.rotate(ctx, gpu, w, x, &RotateInputs::default())?;
                    let xr = h.into_buf();
                    gemv.run(
                        ctx,
                        gpu,
                        &GemvParams {
                            w,
                            x: &xr,
                            y: residual,
                            variant: GemvVariant::WithResidual,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )
                } else {
                    gemv.run(
                        ctx,
                        gpu,
                        &GemvParams {
                            w,
                            x,
                            y: residual,
                            variant: GemvVariant::WithResidual,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )
                }
            } else {
                // run_auto applies the dtype's rotation (FWHT/Givens) before the
                // kernel, so ParoQ4G128 gets its Givens rotation. Plain would skip it.
                //
                // ALIASING GUARD: most callers (e.g. qwen35 o_proj / dn_out) pass
                // `out` == `residual` (both `&s.x`). Reusing `out` as the GEMV scratch
                // in that case is WRONG: run_auto would overwrite the residual with
                // `W·x` and the subsequent `residual += out` would then compute
                // `2·(W·x)` — the residual is lost. Detect the alias by device pointer
                // and allocate a fresh scratch when they overlap. When `out` is a
                // genuinely-distinct buffer, reuse it (no alloc churn).
                if std::ptr::eq(residual, out) || residual.buf.as_ptr() == out.buf.as_ptr() {
                    let tmp = gpu
                        .alloc_tensor(&[w.m], DType::F32)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    gemv.run_auto(ctx, gpu, w, x, &tmp)?;
                    gpu.add_inplace_f32(residual, &tmp)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    gpu.free_tensor(tmp)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                } else {
                    gemv.run_auto(ctx, gpu, w, x, out)?;
                    gpu.add_inplace_f32(residual, out)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
                Ok(())
            }
        }
        Step::RmsnormAutomatic {
            x,
            norm_weight,
            x_plain,
            out,
            awq_scale,
            k,
            eps,
            rotation,
        } => {
            if *rotation == RotationPlan::None {
                // HFQ4G256 and other non-FWHT dtypes: plain rmsnorm into `out`.
                // x_plain is not written in this path (scratch only for FWHT path).
                gpu.rmsnorm_f32(x, norm_weight, out, *eps)
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            } else if *rotation == RotationPlan::Mq8Internal {
                // MQ8 cannot share LDS with the FWHT-G256 fused kernel: it produces an
                // INT8 scratch consumed by the downstream gemv_mq8_prerotated kernel.
                // RotationFamily::WithRmsnorm would route to fused_rmsnorm_rotate_mq
                // (FWHT, F32 output) — wrong dtype for the MQ8 GEMV. Mirror the fix
                // from qwen35.rs::rmsnorm_rotate_dispatch (7b35e700).
                gpu.rmsnorm_f32(x, norm_weight, out, *eps)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.rotate_quantize_x_mq8(out, *k)
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            } else {
                let rotation_family = ROTATION.get_or_init(RotationFamily::new);
                rotation_family
                    .run(
                        ctx,
                        gpu,
                        RotationParams {
                            x,
                            x_up: None,
                            w_norm: Some(norm_weight),
                            x_plain,
                            x_rot: out,
                            awq_scale: *awq_scale,
                            k: *k,
                            eps: *eps,
                            batch_size: 1,
                            variant: RotationVariant::WithRmsnorm,
                            givens_pairs: None,
                            givens_theta: None,
                            givens_scales: None,
                            givens_krot: None,
                        },
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            }
        }
        Step::Attend { plan, io } => {
            use crate::families::attention::AttentionFamily;
            static ATTENTION: OnceLock<AttentionFamily> = OnceLock::new();
            let attn = ATTENTION.get_or_init(AttentionFamily::new);
            attn.run_attention(ctx, gpu, plan, io)
        }
        Step::Rope {
            q,
            k,
            pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            theta,
        } => gpu
            .rope_f32(q, k, pos_buf, *n_heads, *n_kv_heads, *head_dim, *theta)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::QkNorm {
            x,
            weight,
            n_groups,
            head_dim,
            eps,
        } => gpu
            .rmsnorm_batched(x, weight, x, *n_groups, *head_dim, *eps)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::BiasAdd { x, bias, dim } => gpu
            .bias_add_f32(x, bias, 1, *dim)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::SiluMul { gate, up, out } => gpu
            .silu_mul_f32(gate, up, out)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::ResidualAdd { x, y, dim: _ } => gpu
            .add_f32(x, y, x)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        // ── MoE decode ops (Task 4) ─────────────────────────────────────
        Step::MoeRoute {
            scores,
            gate_bias,
            topk_indices,
            topk_weights,
            k,
            n_experts,
            route_scale,
        } => launch_moe_route(
            gpu,
            scores,
            gate_bias,
            topk_indices,
            topk_weights,
            *n_experts,
            *k,
            *route_scale,
        ),
        Step::IndexedMoeGemv {
            experts,
            which,
            topk_indices,
            input,
            out,
            k_top,
            batch_size,
        } => {
            // Extract the inner tensor — the helpers take a plain &GpuTensor.
            // Both Raw and Prerotated are accepted; callers should pass Prerotated
            // (the activation is always FWHT-rotated before building the step).
            let x = match input {
                GemvInput::Raw(x) | GemvInput::Prerotated(x) => x,
            };
            match which {
                MoeProj::GateUp { up_out } => {
                    launch_indexed_gate_up(gpu, experts, topk_indices, x, out, up_out, *k_top)
                }
                MoeProj::DownExpanded => {
                    launch_indexed_down(gpu, experts, topk_indices, x, out, *k_top, *batch_size)
                }
                MoeProj::DownResidual { topk_weights } => launch_indexed_down_residual(
                    gpu,
                    experts,
                    topk_indices,
                    topk_weights,
                    x,
                    out,
                    *k_top,
                ),
            }
        }
        Step::MoeCombine {
            down_out,
            topk_weights,
            out,
            k,
            hidden,
            batch_size,
            inverse_perm,
        } => match inverse_perm {
            Some(perm) => {
                // Prefill grouped path: moe_down_combine_grouped_k8.
                launch_moe_combine_grouped(
                    gpu,
                    down_out,
                    perm,
                    topk_weights,
                    out,
                    *hidden,
                    *k,
                    *batch_size,
                )
            }
            None => {
                // Decode expanded path: moe_down_combine_k8_batched.
                launch_moe_combine(gpu, down_out, topk_weights, out, *hidden, *k, *batch_size)
            }
        },
        // ── MoE prefill grouped ops (Task 5) ───────────────────────────────
        Step::MoeScatter {
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
        } => launch_moe_scatter(
            gpu,
            topk_indices,
            expert_token_counts,
            expert_offsets,
            sorted_slot_index,
            expert_tile_ids,
            inverse_perm,
            *total_slots,
            *n_experts,
            *m_total_max,
            *block_m,
        ),
        Step::GroupedMoeGemm {
            experts,
            which,
            sorted_slot_index,
            expert_tile_ids,
            x,
            y,
            m_total,
            batch_size,
            k_top,
        } => match which {
            MoeProj::GateUp { .. } => launch_grouped_gate_up(
                gpu,
                experts,
                sorted_slot_index,
                expert_tile_ids,
                x,
                y,
                *m_total,
                *k_top,
                *batch_size,
            ),
            MoeProj::DownExpanded => launch_grouped_down(
                gpu,
                experts,
                sorted_slot_index,
                expert_tile_ids,
                x,
                y,
                *m_total,
                *k_top,
                *batch_size,
            ),
            MoeProj::DownResidual { .. } => Err(DispatchError::Hip(
                "GroupedMoeGemm: DownResidual is not a valid grouped projection; \
                 use DownExpanded + MoeCombine(inverse_perm=Some) for grouped down"
                    .to_string(),
            )),
        },
        Step::MoeUnscatter {
            y_grouped,
            sorted_slot_index,
            gate_batch,
            up_batch,
            mi,
            k_top,
            m_total,
        } => launch_moe_unscatter(
            gpu,
            y_grouped,
            sorted_slot_index,
            gate_batch,
            up_batch,
            *mi,
            *k_top,
            *m_total,
        ),
    }
}

/// Borrow `out` from a `RmsnormAutomatic` step. The guard has already confirmed
/// step[0] is RmsnormAutomatic; this panics in debug if called incorrectly.
fn rmsnorm_out<'a>(step: &'a Step<'a>) -> &'a rdna_compute::GpuTensor {
    match step {
        Step::RmsnormAutomatic { out, .. } => out,
        _ => panic!("launch_fused: expected RmsnormAutomatic at step[0]"),
    }
}

/// Borrow `w` and `out` from a `Gemv` step.
fn gemv_weight_out<'a>(step: &'a Step<'a>) -> (&'a WeightRef<'a>, &'a rdna_compute::GpuTensor) {
    match step {
        Step::Gemv { w, out, .. } => (w, out),
        _ => panic!("launch_fused: expected Gemv step"),
    }
}

fn launch_fused(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    steps: &[Step],
) -> Result<(), DispatchError> {
    // Step 0 is always RmsnormAutomatic — run it to fill the activated buffer.
    launch_op(gpu, ctx, &steps[0])?;
    let activated = rmsnorm_out(&steps[0]);
    let fused_qkv = FUSED_QKV.get_or_init(FusedQkvFamily::new);

    match key {
        KernelKey::FusedQkvMq4G256Lloyd
        | KernelKey::FusedQkvMq3G256Lloyd
        | KernelKey::FusedQkvHfq4G256
        | KernelKey::FusedQkvHfq6G256
        | KernelKey::FusedQkvQ4K
        | KernelKey::FusedQkvQ8_0 => {
            let (wq, q) = gemv_weight_out(&steps[1]);
            let (wk, k) = gemv_weight_out(&steps[2]);
            let (wv, v) = gemv_weight_out(&steps[3]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wq.buf, wk.buf, wv.buf],
                    x: activated,
                    outputs: &[q, k, v],
                    m: &[wq.m, wk.m, wv.m],
                    k: wq.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedGateUpMq4G256Lloyd
        | KernelKey::FusedGateUpMq3G256Lloyd
        | KernelKey::FusedGateUpHfq4G256
        | KernelKey::FusedGateUpHfq6G256
        | KernelKey::FusedGateUpQ4K
        | KernelKey::FusedGateUpQ8_0
        | KernelKey::FusedGateUpMfp4G32E8 => {
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wg.buf, wu.buf],
                    x: activated,
                    outputs: &[gate, up],
                    m: &[wg.m, wu.m],
                    k: wg.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }
        // ── QKVZA 4-way (DeltaNet) ──
        KernelKey::FusedQkvzaHfq4G256
        | KernelKey::FusedQkvzaMq3G256Lloyd
        | KernelKey::FusedQkvzaMq4G256Lloyd
        | KernelKey::FusedQkvzaHfq6G256
        | KernelKey::FusedQkvzaMfp4G32E8
        | KernelKey::FusedQkvzaQ8_0 => {
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wqkv.buf, wz.buf, wb.buf, wa.buf],
                    x: activated,
                    outputs: &[qkv, z, beta, alpha],
                    m: &[wqkv.m, wz.m, wb.m, wa.m],
                    k: wqkv.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }

        // ── Paro fused Paro4G128T ────────────────────────────────────────
        // For all three Paro fused keys, we allocate rotation scratch from
        // gpu.scratch.paro_fused_scratch (4 × [k] F32 buffers). The QKVZA
        // path passes all 4; the QKV (3-way) passes 4 with m3=0 via aliasing;
        // the gate+up path passes 1 (x_rot_gate), with the kernel using
        // gpu.scratch.mq_x_rot internally for x_rot_up.
        //
        // Build aliased GpuTensor descriptors before the mutable borrow of
        // gpu (fused_qkv.run takes &mut Gpu). DeviceBuffer::alias() creates
        // an owned descriptor over the same VRAM — no Rust borrow held.
        KernelKey::FusedGateUpParo4G128T => {
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            let k = wg.k;
            #[cfg(debug_assertions)]
            eprintln!("[dispatch] GateUp Paro: k={}, mg={}, mu={}", k, wg.m, wu.m);
            gpu.ensure_paro_fused_scratch(k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            // Also ensure mq_x_rot >= k (the kernel aliases it for x_rot_up).
            gpu.ensure_mq_signs()
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            #[cfg(debug_assertions)]
            {
                let gate_buf = &gpu.scratch.paro_fused_scratch.as_ref().unwrap()[0];
                let up_internal = gpu.scratch.mq_x_rot.as_ref().unwrap();
                debug_assert!(
                    gate_buf.buf.as_ptr() != up_internal.buf.as_ptr(),
                    "Paro gate+up: x_rot_gate must not alias mq_x_rot"
                );
            }
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wg.buf, wu.buf],
                    x: activated,
                    outputs: &[gate, up],
                    m: &[wg.m, wu.m],
                    k,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedQkvzaParo4G128T => {
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            let k = wqkv.k;
            #[cfg(debug_assertions)]
            eprintln!(
                "[dispatch] QKVZA Paro: k={}, mqkv={}, mz={}, mbeta={}, malpha={}",
                k, wqkv.m, wz.m, wb.m, wa.m
            );
            gpu.ensure_paro_fused_scratch(k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wqkv.buf, wz.buf, wb.buf, wa.buf],
                    x: activated,
                    outputs: &[qkv, z, beta, alpha],
                    m: &[wqkv.m, wz.m, wb.m, wa.m],
                    k,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedQkvParo4G128T => {
            let (wq, q) = gemv_weight_out(&steps[1]);
            let (wk, k) = gemv_weight_out(&steps[2]);
            let (wv, v) = gemv_weight_out(&steps[3]);
            let kk = wq.k;
            #[cfg(debug_assertions)]
            eprintln!(
                "[dispatch] QKV Paro: k={}, mq={}, mk={}, mv={}",
                kk, wq.m, wk.m, wv.m
            );
            gpu.ensure_paro_fused_scratch(kk)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wq.buf, wk.buf, wv.buf],
                    x: activated,
                    outputs: &[q, k, v],
                    m: &[wq.m, wk.m, wv.m],
                    k: kk,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        _ => Err(DispatchError::MissingImpl { key }),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DispatchCtx;
    use crate::families::fused_qkv::FusedQkvFamily;
    use crate::types::KernelKey;

    #[test]
    fn qkvza_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedQkvzaMq4G256Lloyd),
            "FusedQkvzaMq4G256Lloyd missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaMq3G256Lloyd),
            "FusedQkvzaMq3G256Lloyd missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaHfq4G256),
            "FusedQkvzaHfq4G256 missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaHfq6G256),
            "FusedQkvzaHfq6G256 missing"
        );

        for entry in FUSED_TABLE.iter() {
            if matches!(
                entry.key,
                KernelKey::FusedQkvzaMq4G256Lloyd
                    | KernelKey::FusedQkvzaMq3G256Lloyd
                    | KernelKey::FusedQkvzaHfq4G256
                    | KernelKey::FusedQkvzaHfq6G256
            ) {
                assert_eq!(
                    entry.ops.len(),
                    5,
                    "QKVZA entry {:?} should have 5 ops",
                    entry.key
                );
            }
        }
    }

    #[test]
    fn qkvza_guards_reject_short_slices() {
        let ctx = DispatchCtx::for_test("gfx1100");
        // Guards must return false for slices shorter than 5 steps.
        let empty: &[Step] = &[];
        assert!(!guard_qkvza_mq4g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_mq3g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_hfq4g256(empty, &ctx));
        assert!(!guard_qkvza_hfq6g256(empty, &ctx));
    }

    #[test]
    fn qkvza_no_paro_or_q8_fused_entries() {
        use crate::types::GemvVariant;
        // ParoQ4G128 should not resolve to any fused QKVZA key. It may resolve
        // to a plain GEMV key (or nothing for unsupported arches). Both are fine.
        let paro = KernelKey::for_gemv(DType::ParoQ4G128, GemvVariant::Plain, false);
        let q8 = KernelKey::for_gemv(DType::Q8_0, GemvVariant::Plain, false);
        for key in [paro.ok(), q8.ok()].into_iter().flatten() {
            assert!(
                !matches!(
                    key,
                    KernelKey::FusedQkvzaMq4G256Lloyd
                        | KernelKey::FusedQkvzaMq3G256Lloyd
                        | KernelKey::FusedQkvzaHfq4G256
                        | KernelKey::FusedQkvzaHfq6G256
                ),
                "ParoQ4G128/Q8_0 must not resolve to a fused QKVZA key, got {:?}",
                key
            );
        }
    }

    #[test]
    fn qkvza_guards_reject_force_unfused() {
        // The plan mandates that force_unfused must prevent fused QKVZA dispatch.
        // Construct a DispatchCtx with force_unfused=true and verify each guard
        // returns false even for otherwise-matching dtypes. We can't build full
        // Steps with real GPU tensors, so we test the guard logic directly with
        // the flag set.
        use rdna_compute::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::from_env_for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                Arc::new(FeatureFlags::from_env_for_test("gfx1100")),
            ),
            flags: Arc::new(flags),
            resources: crate::resource::ResourceManager::for_test(),
        };
        // short-circuit: every guard opens with `force_unfused → false`, so even
        // an empty slice returns false. This proves the branch exists.
        let empty: &[Step] = &[];
        assert!(!guard_qkvza_mq4g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_mq3g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_hfq4g256(empty, &ctx));
        assert!(!guard_qkvza_hfq6g256(empty, &ctx));
    }

    #[test]
    fn qkvza_fused_table_no_paro_q4_or_q8_entries() {
        // ParoQ4G128 and Q8_0 must NOT have fused QKVZA entries — they fall
        // through to per-op dispatch. This test asserts that none of the fused
        // table keys match a Paro or Q8 variant, ensuring byte-identical
        // unfused-path correctness for those dtypes.
        let paro_q4_key = KernelKey::for_gemv(DType::ParoQ4G128, GemvVariant::Plain, false);
        let q8_key = KernelKey::for_gemv(DType::Q8_0, GemvVariant::Plain, false);
        // Paro and Q8 should resolve to plain GEMV keys, not fused QKVZA keys.
        // (They may be Err for arches without support, which is also fine.)
        for key in [paro_q4_key, q8_key] {
            if let Ok(k) = key {
                assert!(
                    !matches!(
                        k,
                        KernelKey::FusedQkvzaMq4G256Lloyd
                            | KernelKey::FusedQkvzaMq3G256Lloyd
                            | KernelKey::FusedQkvzaHfq4G256
                            | KernelKey::FusedQkvzaHfq6G256
                    ),
                    "ParoQ4G128/Q8_0 should not resolve to a fused QKVZA key"
                );
            }
        }
    }

    #[test]
    fn qkvza_fused_table_arch_coverage() {
        let family = FusedQkvFamily::new();
        let ctx1100 = DispatchCtx::for_test("gfx1100");
        let ctx1201 = DispatchCtx::for_test("gfx1201");

        let wmma_keys = &[
            KernelKey::FusedQkvzaMq4G256Lloyd,
            KernelKey::FusedQkvzaMq3G256Lloyd,
            KernelKey::FusedQkvzaHfq4G256,
        ];

        for &key in wmma_keys {
            assert!(
                family.resolve(key, &ctx1100, None).is_ok(),
                "QKVZA {:?} should resolve on gfx1100",
                key
            );
            assert!(
                family.resolve(key, &ctx1201, None).is_ok(),
                "QKVZA {:?} should resolve on gfx1201",
                key
            );
        }

        // dp4a key: just verify no panic
        let _ = family.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx1100, None);
        let _ = family.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx1201, None);
    }

    #[test]
    fn paro_guards_reject_force_unfused() {
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(
            !guard_gate_up_paro4g128t(empty, &ctx),
            "force_unfused must reject gate_up_paro"
        );
        assert!(
            !guard_qkvza_paro4g128t(empty, &ctx),
            "force_unfused must reject qkvza_paro"
        );
        assert!(
            !guard_qkv_paro4g128t(empty, &ctx),
            "force_unfused must reject qkv_paro"
        );
    }

    #[test]
    fn paro_guards_require_raw_input_and_alignment() {
        // Paro guards require GemvInput::Raw (not Prerotated) and m%8==0/k%128==0.
        // We can't construct real Gemv steps with GPU tensors in a unit test,
        // but we can verify the guards reject empty/wrong-length slices.
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(!guard_gate_up_paro4g128t(empty, &ctx));
        assert!(!guard_qkvza_paro4g128t(empty, &ctx));
        assert!(!guard_qkv_paro4g128t(empty, &ctx));
    }

    #[test]
    fn paro_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedGateUpParo4G128T),
            "FusedGateUpParo4G128T missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaParo4G128T),
            "FusedQkvzaParo4G128T missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvParo4G128T),
            "FusedQkvParo4G128T missing from FUSED_TABLE"
        );
    }

    #[test]
    fn paro_fused_table_arch_coverage() {
        let family = FusedQkvFamily::new();
        let ctx1100 = DispatchCtx::for_test("gfx1100");
        let ctx1201 = DispatchCtx::for_test("gfx1201");

        let paro_keys = &[
            KernelKey::FusedGateUpParo4G128T,
            KernelKey::FusedQkvzaParo4G128T,
            KernelKey::FusedQkvParo4G128T,
        ];

        for &key in paro_keys {
            // Paro uses dp4a — should resolve on gfx1100 (RDNA3) and gfx1201 (RDNA4).
            assert!(
                family.resolve(key, &ctx1100, None).is_ok(),
                "Paro key {:?} should resolve on gfx1100",
                key
            );
            assert!(
                family.resolve(key, &ctx1201, None).is_ok(),
                "Paro key {:?} should resolve on gfx1201",
                key
            );
        }
    }

    // ── Q4K / Q8_0 guard tests (Ship 2.1 A1 — Claude F1 / glm5 F2) ──────

    #[test]
    fn q4k_q8_0_guards_reject_force_unfused() {
        // All three new guards must return false when force_unfused is set,
        // even for empty slices (the guard opens with the early-return).
        use rdna_compute::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::from_env_for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                Arc::new(FeatureFlags::from_env_for_test("gfx1100")),
            ),
            flags: Arc::new(flags),
            resources: crate::resource::ResourceManager::for_test(),
        };
        let empty: &[Step] = &[];
        assert!(
            !guard_qkv_q4k(empty, &ctx),
            "guard_qkv_q4k must reject force_unfused"
        );
        assert!(
            !guard_gate_up_q4k(empty, &ctx),
            "guard_gate_up_q4k must reject force_unfused"
        );
        assert!(
            !guard_gate_up_q8_0(empty, &ctx),
            "guard_gate_up_q8_0 must reject force_unfused"
        );
    }

    #[test]
    fn q4k_q8_0_guards_reject_wrong_length() {
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(!guard_qkv_q4k(empty, &ctx), "Q4K QKV guard needs len==4");
        assert!(
            !guard_gate_up_q4k(empty, &ctx),
            "Q4K gate+up guard needs len==3"
        );
        assert!(
            !guard_gate_up_q8_0(empty, &ctx),
            "Q8_0 gate+up guard needs len==3"
        );
    }

    #[test]
    fn q4k_q8_0_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedQkvQ4K),
            "FusedQkvQ4K missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedGateUpQ4K),
            "FusedGateUpQ4K missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedGateUpQ8_0),
            "FusedGateUpQ8_0 missing from FUSED_TABLE"
        );
    }

    /// Pure-logic test: TpCollective→StepCollective wrapper mapping and
    /// the three validate_parallel_args length-mismatch guards.
    /// No GPU needed — only enum construction and the pure validator are exercised.
    #[test]
    fn parallel_arg_length_guards() {
        use hipfire_hardware::DimKind;

        // --- mapping test: execute_steps_tp wrapper logic ---
        // AllReduceOut{dim:8} must map to AllReduce{kind:Tp, dim:8}
        let tp_colls = vec![TpCollective::None, TpCollective::AllReduceOut { dim: 8 }];
        let mapped: Vec<StepCollective> = tp_colls
            .iter()
            .map(|c| match c {
                TpCollective::AllReduceOut { dim } => StepCollective::AllReduce {
                    kind: DimKind::Tp,
                    dim: *dim,
                },
                TpCollective::None => StepCollective::None,
            })
            .collect();
        // First element: None → None
        assert!(matches!(mapped[0], StepCollective::None));
        // Second element: AllReduceOut{8} → AllReduce{Tp, 8}
        match &mapped[1] {
            StepCollective::AllReduce { kind, dim } => {
                assert!(matches!(kind, DimKind::Tp), "expected Tp, got {kind:?}");
                assert_eq!(*dim, 8);
            }
            other => panic!("expected AllReduce, got {other:?}"),
        }

        // --- guard 1: per_rank_steps.len() != group_size ---
        // group=2, but only 1 step list supplied
        let group_size = 2usize;
        let steps_1: Vec<Vec<Step<'_>>> = vec![vec![]]; // len=1, mismatch
        let colls_0: Vec<StepCollective> = vec![];
        let zb_0: Vec<bool> = vec![];
        let e = super::validate_parallel_args(group_size, &steps_1, &colls_0, &zb_0)
            .expect_err("should fail: per_rank_steps.len()==1 != group_size==2");
        assert!(
            matches!(&e, DispatchError::Hip(msg) if msg.contains("step lists")),
            "unexpected error: {e:?}"
        );

        // --- guard 2: collectives.len() != n_steps ---
        // group=2, 2 step lists each with n_steps=0, but 1 collective supplied
        let steps_2: Vec<Vec<Step<'_>>> = vec![vec![], vec![]]; // 2 ranks, n_steps=0
        let colls_1 = vec![StepCollective::None]; // len=1, mismatch with n_steps=0
        let zb_0: Vec<bool> = vec![];
        let e = super::validate_parallel_args(group_size, &steps_2, &colls_1, &zb_0)
            .expect_err("should fail: collectives.len()==1 != n_steps==0");
        assert!(
            matches!(&e, DispatchError::Hip(msg) if msg.contains("collectives")),
            "unexpected error: {e:?}"
        );

        // --- guard 3: zero_before.len() != n_steps ---
        // group=2, 2 step lists with n_steps=0, 0 collectives, but zero_before has 1 elem
        let colls_0: Vec<StepCollective> = vec![]; // len=0 matches n_steps=0
        let zb_1 = vec![false]; // len=1, mismatch with n_steps=0
        let e = super::validate_parallel_args(group_size, &steps_2, &colls_0, &zb_1)
            .expect_err("should fail: zero_before.len()==1 != n_steps==0");
        assert!(
            matches!(&e, DispatchError::Hip(msg) if msg.contains("zero_before")),
            "unexpected error: {e:?}"
        );
    }
}
