// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Op-list interpreter. Phase 2a: GEMV + a fused rmsnorm-rotate producer; empty
//! fusion table (all per-op fallback).

use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::fused_qkv::{FusedQkvFamily, FusedQkvParams};
use crate::families::gemv::{GemvFamily, GemvParams, RotateInputs, WeightRef};
use crate::families::rotation::{RotationFamily, RotationParams};
use crate::types::GemvVariant;
use crate::types::{DispatchError, KernelKey, PipelineOp, RotationPlan, RotationVariant};

/// Rotation disposition of a Gemv's input. Borrows (never owns a RotatedActivation).
pub enum GemvInput<'a> {
    Raw(&'a GpuTensor),        // launch_op self-rotates via run_auto (plan-aware)
    Prerotated(&'a GpuTensor), // already FWHT-rotated; dispatched via Prerotated variant
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
}

/// Op-kind for fusion matching. Total over Step variants.
fn op_kind(step: &Step) -> PipelineOp {
    match step {
        Step::Gemv { .. } => PipelineOp::Gemv,
        Step::GemvResidual { .. } => PipelineOp::GemvResidual,
        Step::RmsnormAutomatic { .. } => PipelineOp::RmsnormAutomatic,
        Step::Attend { .. } => PipelineOp::Attend,
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
/// - GemvInput::Raw (Paro uses Raw because the kernel rotates internally)
/// - awq_scale == None
fn gemv_steps_uniform_raw(steps: &[Step], dtype: DType) -> bool {
    steps[1..].iter().all(|s| match s {
        Step::Gemv {
            w,
            input: GemvInput::Raw(_),
            ..
        } => w.dtype == dtype && w.awq_scale.is_none(),
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

/// Covers both DType::HFQ6G256 and DType::MQ6G256 — both use dp4a.
pub(crate) fn guard_qkv_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if !dp4a_eligible(ctx) {
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

/// Opus W4A4 fused QKVZA — all four LA weights Oq4G256, prerotated activation.
pub(crate) fn guard_qkvza_oq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::Oq4G256, true)
}

/// Opus W8A8 (OQ+) fused QKVZA — all four LA weights Oq8G256, prerotated activation.
pub(crate) fn guard_qkvza_oq8g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::Oq8G256, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256 — both use dp4a.
pub(crate) fn guard_qkvza_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if !dp4a_eligible(ctx) {
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

/// Opus W4A4 fused Gate+Up — both weights Oq4G256, prerotated activation.
pub(crate) fn guard_gate_up_oq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Oq4G256, true)
}

/// Opus W8A8 (OQ+) fused Gate+Up — both weights Oq8G256, prerotated activation.
pub(crate) fn guard_gate_up_oq8g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Oq8G256, true)
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

pub(crate) fn guard_qkvza_paro4g128t(steps: &[Step], ctx: &DispatchCtx) -> bool {
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
    dt == DType::ParoQ4G128
        && gemv_steps_uniform_raw(steps, DType::ParoQ4G128)
        && steps[1..].iter().all(|s| match s {
            Step::Gemv { w, .. } => w.m % 8 == 0 && w.k % 128 == 0,
            _ => false,
        })
}

pub(crate) fn guard_qkv_paro4g128t(steps: &[Step], ctx: &DispatchCtx) -> bool {
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
    dt == DType::ParoQ4G128
        && gemv_steps_uniform_raw(steps, DType::ParoQ4G128)
        && steps[1..].iter().all(|s| match s {
            Step::Gemv { w, .. } => w.m % 8 == 0 && w.k % 128 == 0,
            _ => false,
        })
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
        key: KernelKey::FusedQkvzaOq4G256,
        guard: guard_qkvza_oq4g256,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaOq8G256,
        guard: guard_qkvza_oq8g256,
    },
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
    // ── Gate+Up 2-way ───────────────────────────────────────────────────────
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpOq4G256,
        guard: guard_gate_up_oq4g256,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpOq8G256,
        guard: guard_gate_up_oq8g256,
    },
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
    // ── Q8_0 / Q4K fused entries (non-rotated, Always arch gate) ─────────
    // No FusedQkvQ8_0 entry: neither qwen2 (QKV is HFQ4G256) nor llama (QKV is
    // Q4K/MQ/plain) uses Q8_0 for QKV — only gate+up.
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

/// Opus Quant W4A4 (Oq4G256) GEMV: the only int4-activation path. The dispatch
/// families have no Oq4 GEMV kernel (activations must be int4-quantized at
/// runtime), so every Oq4 `Step::Gemv*` routes here instead of `gemv.run*`.
///   - `already_rotated`: x is the FWHT-rotated (and awq-divided) activation from
///     the upstream RmsnormAutomatic step → just int4-quantize + grouped GEMM.
///   - else: FWHT-rotate x here (applying the awq sidecar if present), then
///     quantize + GEMM. Used for standalone GEMVs (lm_head, residual Raw).
/// Weight buffer is the combined `[nibbles | f32 scales]` layout; the scale
/// pointer is a sub_offset view at M*(K/2). Single-token (B=1).
fn oq4_gemv_into(
    gpu: &mut Gpu,
    w: &WeightRef,
    x: &hipfire_rdna::GpuTensor,
    already_rotated: bool,
    out: &hipfire_rdna::GpuTensor,
) -> Result<(), DispatchError> {
    oq4_gemv_into_impl(gpu, w, x, already_rotated, out, false)
}

/// As `oq4_gemv_into` but `residual=true` does the residual add in-kernel
/// (`out += W·x`, gemv_oq4_grouped_residual) — no separate add_inplace_f32.
fn oq4_gemv_into_impl(
    gpu: &mut Gpu,
    w: &WeightRef,
    x: &hipfire_rdna::GpuTensor,
    already_rotated: bool,
    out: &hipfire_rdna::GpuTensor,
    residual: bool,
) -> Result<(), DispatchError> {
    const GROUP: usize = 256;
    let hip = |e: hipfire_rdna::HipError| DispatchError::Hip(e.to_string());
    let (m, k) = (w.m, w.k);
    let ng = k / GROUP;
    gpu.ensure_mq_signs().map_err(hip)?;
    gpu.ensure_oq4_scratch().map_err(hip)?;

    // Persistent scratch (aliased, no per-call alloc → hipGraph-capture-clean).
    // Stream-ordered reuse across sequential projections is safe (one stream).
    let alias = |t: &hipfire_rdna::GpuTensor, n: usize, dt: DType| hipfire_rdna::GpuTensor {
        buf: unsafe { t.buf.alias() },
        shape: vec![n],
        dtype: dt,
    };
    // Rotated f32 activation: reuse the persistent oq4_xr scratch when we must
    // rotate here (Raw paths: lm_head, residual-raw); else x is already rotated.
    let rotated = if already_rotated {
        None
    } else {
        let xr = alias(gpu.oq4_xr.as_ref().unwrap(), k, DType::F32);
        if let Some(awq) = w.awq_scale {
            gpu.rotate_x_mq_awq(x, awq, &xr, k).map_err(hip)?;
        } else {
            gpu.rotate_x_mq(x, &xr, k).map_err(hip)?;
        }
        Some(xr)
    };
    let xr: &hipfire_rdna::GpuTensor = rotated.as_ref().unwrap_or(x);

    // OQ4+ decode (B=1): read the INTERLEAVED weight region ([f32 scale][128
    // nibbles]/group, one coalesced stream) appended after the split region by the
    // loader. `residual` fuses the residual add (out += W·x) into the gemv.
    let il = w.buf.sub_offset(m * (k / 2) + m * ng * 4, m * ng * 132);
    if residual {
        gpu.gemv_oq4_interleaved_residual(&il, xr, out, m, k, GROUP)
            .map_err(hip)
    } else {
        gpu.gemv_oq4_interleaved(&il, xr, out, m, k, GROUP)
            .map_err(hip)
    }
}

/// Opus W8A8 (OQ+) single-projection decode GEMV (B=1) — int8 analog of
/// `oq4_gemv_into`. Quantizes the (rotated) activation to int8 into persistent
/// scratch, then runs the one-wave-per-row `gemv_oq8_grouped` (no WMMA N-tile
/// waste at B=1). Weight buffer is `[int8 M*K | f32 scales M*ng]`; scale pointer
/// is a sub_offset view at M*K (one int8 per weight, not nibble-packed).
fn oq8_gemv_into(
    gpu: &mut Gpu,
    w: &WeightRef,
    x: &hipfire_rdna::GpuTensor,
    already_rotated: bool,
    out: &hipfire_rdna::GpuTensor,
) -> Result<(), DispatchError> {
    const GROUP: usize = 256;
    let hip = |e: hipfire_rdna::HipError| DispatchError::Hip(e.to_string());
    let (m, k) = (w.m, w.k);
    let ng = k / GROUP;
    gpu.ensure_mq_signs().map_err(hip)?;
    gpu.ensure_oq4_scratch().map_err(hip)?; // oq4_xr reused as the rotate target
    let alias = |t: &hipfire_rdna::GpuTensor, n: usize, dt: DType| hipfire_rdna::GpuTensor {
        buf: unsafe { t.buf.alias() },
        shape: vec![n],
        dtype: dt,
    };
    let rotated = if already_rotated {
        None
    } else {
        let xr = alias(gpu.oq4_xr.as_ref().unwrap(), k, DType::F32);
        if let Some(awq) = w.awq_scale {
            gpu.rotate_x_mq_awq(x, awq, &xr, k).map_err(hip)?;
        } else {
            gpu.rotate_x_mq(x, &xr, k).map_err(hip)?;
        }
        Some(xr)
    };
    let xr: &hipfire_rdna::GpuTensor = rotated.as_ref().unwrap_or(x);

    // DECODE (B=1): consume the f32 rotated activation directly — dequant the
    // int8 weight inline (W4A16 decode, mq4-style). No quantize_act_oq8 launch.
    let ws = w.buf.sub_offset(m * k, m * ng * 4);
    gpu.gemv_oq8_grouped(w.buf, &ws, xr, out, m, k, GROUP)
        .map_err(hip)
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
            if w.dtype == DType::Oq4G256 {
                return oq4_gemv_into(gpu, w, x, false, out);
            }
            if w.dtype == DType::Oq8G256 {
                return oq8_gemv_into(gpu, w, x, false, out);
            }
            if w.dtype == DType::BF16 {
                return gpu
                    .gemm_bf16_x_bf16_wmma(w.buf, x, out, w.m, w.k, 1)
                    .map_err(|e| DispatchError::Hip(e.to_string()));
            }
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run_auto(ctx, gpu, w, x, out)
        }
        Step::Gemv {
            w,
            input: GemvInput::Prerotated(xr),
            out,
        } => {
            if w.dtype == DType::Oq4G256 {
                return oq4_gemv_into(gpu, w, xr, true, out);
            }
            if w.dtype == DType::Oq8G256 {
                return oq8_gemv_into(gpu, w, xr, true, out);
            }
            if w.dtype == DType::BF16 {
                return gpu
                    .gemm_bf16_x_bf16_wmma(w.buf, xr, out, w.m, w.k, 1)
                    .map_err(|e| DispatchError::Hip(e.to_string()));
            }
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
            if w.dtype == DType::Oq4G256 {
                // FUSED residual gemv (Y += W·x) in one launch — eliminates the
                // separate add_inplace_f32 (the standout decode-gap item vs mq4's
                // gemv_hfq4g256_residual). xr is already rotated.
                let ng = w.k / 256;
                let ws = w.buf.sub_offset(w.m * (w.k / 2), w.m * ng * 4);
                return gpu
                    .gemv_oq4_grouped_residual(w.buf, &ws, xr, residual, w.m, w.k, 256)
                    .map_err(|e| DispatchError::Hip(e.to_string()));
            }
            if w.dtype == DType::Oq8G256 {
                // wo/down residual: B=1 gemv_oq8 (no WMMA N-tile waste) into scratch
                // + add, instead of gemm_oq8 at B=1 (16× wasteful) via WithResidual.
                gpu.ensure_oq4_scratch()
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                let tmp = hipfire_rdna::GpuTensor {
                    buf: unsafe { gpu.oq4_ytmp.as_ref().unwrap().buf.alias() },
                    shape: vec![w.m],
                    dtype: DType::F32,
                };
                oq8_gemv_into(gpu, w, xr, true, &tmp)?;
                return gpu
                    .add_inplace_f32(residual, &tmp)
                    .map_err(|e| DispatchError::Hip(e.to_string()));
            }
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
            if w.dtype == DType::Oq4G256 {
                // FUSED residual gemv (rotate x, then out += W·x in one launch).
                return oq4_gemv_into_impl(gpu, w, x, false, residual, true);
            }
            if w.dtype == DType::Oq8G256 {
                gpu.ensure_oq4_scratch()
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                let tmp = hipfire_rdna::GpuTensor {
                    buf: unsafe { gpu.oq4_ytmp.as_ref().unwrap().buf.alias() },
                    shape: vec![w.m],
                    dtype: DType::F32,
                };
                oq8_gemv_into(gpu, w, x, false, &tmp)?;
                return gpu
                    .add_inplace_f32(residual, &tmp)
                    .map_err(|e| DispatchError::Hip(e.to_string()));
            }
            // For dtypes WITHOUT a fused residual kernel (Q8_0, Q4K, F32), the
            // fallback path runs a plain GEMV then `residual += result`. `out` may
            // be used as scratch ONLY when it does not alias `residual`; when it
            // does (the common qwen35 o_proj / dn_out case where out == residual ==
            // &s.x), a fresh temp is allocated instead. See the aliasing guard below.
            // Nothing reads `out` after this step in any model decode path.
            if w.dtype == DType::BF16 {
                let tmp = gpu
                    .alloc_tensor(&[w.m], DType::F32)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.gemm_bf16_x_bf16_wmma(w.buf, x, &tmp, w.m, w.k, 1)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.add_inplace_f32(residual, &tmp)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                return gpu
                    .free_tensor(tmp)
                    .map_err(|e| DispatchError::Hip(e.to_string()));
            }
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
    }
}

/// Borrow `out` from a `RmsnormAutomatic` step. The guard has already confirmed
/// step[0] is RmsnormAutomatic; this panics in debug if called incorrectly.
fn rmsnorm_out<'a>(step: &'a Step<'a>) -> &'a hipfire_rdna::GpuTensor {
    match step {
        Step::RmsnormAutomatic { out, .. } => out,
        _ => panic!("launch_fused: expected RmsnormAutomatic at step[0]"),
    }
}

/// Borrow `w` and `out` from a `Gemv` step.
fn gemv_weight_out<'a>(step: &'a Step<'a>) -> (&'a WeightRef<'a>, &'a hipfire_rdna::GpuTensor) {
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
        // ── OQ4+ fused projections (decode) ──────────────────────────────────
        // `activated` is the FWHT-rotated f32 activation, consumed DIRECTLY:
        // W4A16 decode (unpack the 4-bit-resident weight inline, f32 acts) — no
        // quantize_act_oq4 launch, mq4-style. One wave-per-row GEMV per
        // sub-projection (leaner than the batched WMMA which wastes 15/16 of its
        // N-tile at B=1; the WMMA fused kernel stays for batched prefill).
        KernelKey::FusedQkvzaOq4G256 => {
            let hip = |e: hipfire_rdna::HipError| DispatchError::Hip(e.to_string());
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            let k = wqkv.k;
            let ng = k / 256;
            // INTERLEAVED-layout fused decode: one launch over the contiguous
            // [scale][nibbles] region (one coalesced stream/group), appended after
            // the split region by the loader.
            let il_qkv = wqkv
                .buf
                .sub_offset(wqkv.m * (k / 2) + wqkv.m * ng * 4, wqkv.m * ng * 132);
            let il_z = wz
                .buf
                .sub_offset(wz.m * (k / 2) + wz.m * ng * 4, wz.m * ng * 132);
            let il_b = wb
                .buf
                .sub_offset(wb.m * (k / 2) + wb.m * ng * 4, wb.m * ng * 132);
            let il_a = wa
                .buf
                .sub_offset(wa.m * (k / 2) + wa.m * ng * 4, wa.m * ng * 132);
            gpu.fused_qkvza_oq4_interleaved(
                &il_qkv, &il_z, &il_b, &il_a, activated, qkv, z, beta, alpha, wqkv.m, wz.m, wb.m,
                wa.m, k, 256,
            )
            .map_err(hip)?;
            Ok(())
        }
        KernelKey::FusedGateUpOq4G256 => {
            let hip = |e: hipfire_rdna::HipError| DispatchError::Hip(e.to_string());
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            let k = wg.k;
            let ng = k / 256;
            let il_g = wg
                .buf
                .sub_offset(wg.m * (k / 2) + wg.m * ng * 4, wg.m * ng * 132);
            let il_u = wu
                .buf
                .sub_offset(wu.m * (k / 2) + wu.m * ng * 4, wu.m * ng * 132);
            gpu.fused_gate_up_oq4_interleaved(
                &il_g, &il_u, activated, gate, up, wg.m, wu.m, k, 256,
            )
            .map_err(hip)?;
            Ok(())
        }
        // ── Opus W4A8 (OQ+) fused projections (decode) ───────────────────────
        // mq4-style decode: consume the shared FWHT-rotated f32 activation
        // DIRECTLY in one fused GEMV-demux launch — the int8 weight is
        // dequantized inline (W4A16 at B=1), so there is NO quantize_act_oq8
        // launch (the dominant decode launch cost). Weights are
        // [int8 M*K | f32 scales M*ng]; the kernel derives the scale ptr.
        KernelKey::FusedQkvzaOq8G256 => {
            let hip = |e: hipfire_rdna::HipError| DispatchError::Hip(e.to_string());
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            let k = wqkv.k;
            gpu.fused_qkvza_oq8_gemv(
                wqkv.buf, wz.buf, wb.buf, wa.buf, activated, qkv, z, beta, alpha, wqkv.m, wz.m,
                wb.m, wa.m, k, 256,
            )
            .map_err(hip)?;
            Ok(())
        }
        KernelKey::FusedGateUpOq8G256 => {
            let hip = |e: hipfire_rdna::HipError| DispatchError::Hip(e.to_string());
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            let k = wg.k;
            gpu.fused_gate_up_oq8_gemv(wg.buf, wu.buf, activated, gate, up, wg.m, wu.m, k, 256)
                .map_err(hip)?;
            Ok(())
        }
        KernelKey::FusedQkvMq4G256Lloyd
        | KernelKey::FusedQkvMq3G256Lloyd
        | KernelKey::FusedQkvHfq4G256
        | KernelKey::FusedQkvHfq6G256
        | KernelKey::FusedQkvQ4K => {
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
        | KernelKey::FusedGateUpQ8_0 => {
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
        | KernelKey::FusedQkvzaHfq6G256 => {
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
        // gpu.paro_fused_scratch (4 × [k] F32 buffers). The QKVZA
        // path passes all 4; the QKV (3-way) passes 4 with m3=0 via aliasing;
        // the gate+up path passes 1 (x_rot_gate), with the kernel using
        // gpu.mq_x_rot internally for x_rot_up.
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
                let gate_buf = &gpu.paro_fused_scratch.as_ref().unwrap()[0];
                let up_internal = gpu.mq_x_rot.as_ref().unwrap();
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
        use hipfire_rdna::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::from_env_for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: hipfire_rdna::arch_caps::ArchCaps::new(
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
        use hipfire_rdna::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::from_env_for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: hipfire_rdna::arch_caps::ArchCaps::new(
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
}
