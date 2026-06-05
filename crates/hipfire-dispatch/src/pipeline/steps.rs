// SPDX-License-Identifier: MIT OR Apache-2.0
//! Op-list interpreter. Phase 2a: GEMV + a fused rmsnorm-rotate producer; empty
//! fusion table (all per-op fallback).

use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, GemvParams, RotateInputs, WeightRef};
use crate::types::GemvVariant;
use crate::families::fused_qkv::{FusedQkvFamily, FusedQkvParams};
use crate::families::rotation::{RotationFamily, RotationParams};
use crate::types::{DispatchError, KernelKey, PipelineOp, RotationPlan, RotationVariant};

/// Rotation disposition of a Gemv's input. Borrows (never owns a RotatedActivation).
pub enum GemvInput<'a> {
    Raw(&'a GpuTensor),         // launch_op self-rotates via run_auto (plan-aware)
    Prerotated(&'a GpuTensor),  // already FWHT-rotated; dispatched via Prerotated variant
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
        x_plain: &'a GpuTensor,   // rmsnorm intermediate scratch (always written)
        out: &'a GpuTensor,       // final activation output (written by this step)
        awq_scale: Option<&'a GpuTensor>,
        k: usize,
        eps: f32,
        rotation: RotationPlan,   // FwhtG256 for MQ dtypes, None for HFQ4/others
    },
}

/// Op-kind for fusion matching. Total over Step variants.
fn op_kind(step: &Step) -> PipelineOp {
    match step {
        Step::Gemv { .. } => PipelineOp::Gemv,
        Step::GemvResidual { .. } => PipelineOp::GemvResidual,
        Step::RmsnormAutomatic { .. } => PipelineOp::RmsnormAutomatic,
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
        Step::Gemv { w, input: GemvInput::Prerotated(_), .. } => {
            w.dtype == dtype && (!require_no_awq || w.awq_scale.is_none())
        }
        _ => false,
    })
}

/// True if ctx has dp4a and !force_unfused.
fn dp4a_eligible(ctx: &DispatchCtx) -> bool {
    !ctx.flags.force_unfused && ctx.arch.gemv_dp4a_enabled()
}

// ── QKV 3-way guards ──

pub(crate) fn guard_qkv_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused { return false; }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_qkv_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused { return false; }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

/// Covers both DType::MQ4G256 (plain) and DType::HFQ4G256 — both feed
/// gpu.fused_qkv_hfq4g256 which takes a pre-normalized x.
pub(crate) fn guard_qkv_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused { return false; }
    if steps.len() != 4 { return false; }
    let dt = match window_gemv_dtype(steps) { Some(d) => d, None => return false };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256)
        && gemv_steps_uniform(steps, dt, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256 — both use dp4a.
pub(crate) fn guard_qkv_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if !dp4a_eligible(ctx) { return false; }
    if steps.len() != 4 { return false; }
    let dt = match window_gemv_dtype(steps) { Some(d) => d, None => return false };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256)
        && gemv_steps_uniform(steps, dt, true)
}

// ── Gate+Up 2-way guards ──

pub(crate) fn guard_gate_up_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused { return false; }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

pub(crate) fn guard_gate_up_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused { return false; }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

pub(crate) fn guard_gate_up_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused { return false; }
    if steps.len() != 3 { return false; }
    let dt = match window_gemv_dtype(steps) { Some(d) => d, None => return false };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256)
        && gemv_steps_uniform(steps, dt, true)
}

pub(crate) fn guard_gate_up_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if !dp4a_eligible(ctx) { return false; }
    if steps.len() != 3 { return false; }
    let dt = match window_gemv_dtype(steps) { Some(d) => d, None => return false };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256)
        && gemv_steps_uniform(steps, dt, true)
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

const QKV3: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv, PipelineOp::Gemv, PipelineOp::Gemv,
];
const GATE_UP2: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv, PipelineOp::Gemv,
];

const FUSED_TABLE: &[FusedPattern] = &[
    // ── QKV 3-way ──────────────────────────────────────────────────────────
    FusedPattern { ops: QKV3, key: KernelKey::FusedQkvMq4G256Lloyd,  guard: guard_qkv_mq4g256lloyd  },
    FusedPattern { ops: QKV3, key: KernelKey::FusedQkvMq3G256Lloyd,  guard: guard_qkv_mq3g256lloyd  },
    FusedPattern { ops: QKV3, key: KernelKey::FusedQkvHfq4G256,      guard: guard_qkv_hfq4g256      },
    FusedPattern { ops: QKV3, key: KernelKey::FusedQkvHfq6G256,      guard: guard_qkv_hfq6g256      },
    // ── Gate+Up 2-way ───────────────────────────────────────────────────────
    FusedPattern { ops: GATE_UP2, key: KernelKey::FusedGateUpMq4G256Lloyd, guard: guard_gate_up_mq4g256lloyd },
    FusedPattern { ops: GATE_UP2, key: KernelKey::FusedGateUpMq3G256Lloyd, guard: guard_gate_up_mq3g256lloyd },
    FusedPattern { ops: GATE_UP2, key: KernelKey::FusedGateUpHfq4G256,     guard: guard_gate_up_hfq4g256     },
    FusedPattern { ops: GATE_UP2, key: KernelKey::FusedGateUpHfq6G256,     guard: guard_gate_up_hfq6g256     },
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

/// Per-op fallback. FULL enum match (no catch-all) so the compiler forces every
/// op to have an arm (spec F4 — a missing arm would be a silent runtime error).
fn launch_op(gpu: &mut Gpu, ctx: &DispatchCtx, step: &Step) -> Result<(), DispatchError> {
    match step {
        Step::Gemv { w, input: GemvInput::Raw(x), out } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run_auto(ctx, gpu, w, x, out)
        }
        Step::Gemv { w, input: GemvInput::Prerotated(xr), out } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(ctx, gpu, &GemvParams {
                w, x: xr, y: out, variant: GemvVariant::Prerotated,
                residual: None, gate: None, up: None,
            })
        }
        Step::GemvResidual { w, input: GemvInput::Prerotated(xr), residual, out: _ } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(ctx, gpu, &GemvParams {
                w, x: xr, y: residual, variant: GemvVariant::WithResidual,
                residual: None, gate: None, up: None,
            })
        }
        Step::GemvResidual { w, input: GemvInput::Raw(x), residual, out: _ } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            // Dtypes with a fused `gemv_*_residual` kernel use it in one launch.
            // Dtypes without one (Q8_0, ParoQ4G128, …) fall back to plain GEMV into
            // a scratch temp + `residual += tmp` — the same two-launch path the
            // legacy `weight_gemv_residual` `_` arm uses. Plain GEMV applies this
            // dtype's own rotation (FWHT / Givens) internally, so this is correct
            // for both no-rotation (Q8) and Givens (Paro) dtypes.
            if KernelKey::for_gemv_residual(w.dtype).is_ok() {
                if crate::types::dtype_rotation_plan(w.dtype) != RotationPlan::None {
                    let h = gemv.rotate(ctx, gpu, w, x, &RotateInputs::default())?;
                    let xr = h.into_buf();
                    gemv.run(ctx, gpu, &GemvParams {
                        w, x: &xr, y: residual, variant: GemvVariant::WithResidual,
                        residual: None, gate: None, up: None,
                    })
                } else {
                    gemv.run(ctx, gpu, &GemvParams {
                        w, x, y: residual, variant: GemvVariant::WithResidual,
                        residual: None, gate: None, up: None,
                    })
                }
            } else {
                // run_auto applies the dtype's rotation (FWHT/Givens) before the
                // kernel, so ParoQ4G128 gets its Givens rotation. Plain would skip it.
                let tmp = gpu.alloc_tensor(&[w.m], DType::F32)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gemv.run_auto(ctx, gpu, w, x, &tmp)?;
                gpu.add_inplace_f32(residual, &tmp)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.free_tensor(tmp)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                Ok(())
            }
        }
        Step::RmsnormAutomatic { x, norm_weight, x_plain, out, awq_scale, k, eps, rotation } => {
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
                rotation_family.run(ctx, gpu, RotationParams {
                    x, x_up: None, w_norm: Some(norm_weight),
                    x_plain, x_rot: out, awq_scale: *awq_scale,
                    k: *k, eps: *eps, batch_size: 1,
                    variant: RotationVariant::WithRmsnorm,
                    givens_pairs: None, givens_theta: None,
                    givens_scales: None, givens_krot: None,
                }).map_err(|e| DispatchError::Hip(e.to_string()))
            }
        }
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
        | KernelKey::FusedQkvHfq6G256 => {
            let (wq, q) = gemv_weight_out(&steps[1]);
            let (wk, k) = gemv_weight_out(&steps[2]);
            let (wv, v) = gemv_weight_out(&steps[3]);
            fused_qkv.run(ctx, gpu, &FusedQkvParams {
                kind: key,
                weights: &[wq.buf, wk.buf, wv.buf],
                x: activated,
                outputs: &[q, k, v],
                m: &[wq.m, wk.m, wv.m],
                k: wq.k,
            })
        }
        KernelKey::FusedGateUpMq4G256Lloyd
        | KernelKey::FusedGateUpMq3G256Lloyd
        | KernelKey::FusedGateUpHfq4G256
        | KernelKey::FusedGateUpHfq6G256 => {
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up)   = gemv_weight_out(&steps[2]);
            fused_qkv.run(ctx, gpu, &FusedQkvParams {
                kind: key,
                weights: &[wg.buf, wu.buf],
                x: activated,
                outputs: &[gate, up],
                m: &[wg.m, wu.m],
                k: wg.k,
            })
        }
        _ => Err(DispatchError::MissingImpl { key }),
    }
}
