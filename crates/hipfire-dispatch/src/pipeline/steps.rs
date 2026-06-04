// SPDX-License-Identifier: MIT OR Apache-2.0
//! Op-list interpreter. Phase 2a: GEMV + a fused rmsnorm-rotate producer; empty
//! fusion table (all per-op fallback).

use rdna_compute::{Gpu, GpuTensor};
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, GemvParams, RotateInputs, WeightRef};
use crate::types::GemvVariant;
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
    /// Fused rmsnorm + FWHT producer (mirrors rmsnorm_rotate_dispatch's MQ branch
    /// → RotationFamily.run(WithRmsnorm)). Writes the rotated activation to `out`.
    RmsnormRotateMq {
        x: &'a GpuTensor,
        norm_weight: &'a GpuTensor,
        x_plain: &'a GpuTensor,           // = tmp (rmsnorm intermediate)
        out: &'a GpuTensor,               // = x_rot scratch
        awq_scale: Option<&'a GpuTensor>,
        k: usize,
        eps: f32,
    },
}

/// Op-kind for fusion matching. Total over Step variants.
fn op_kind(step: &Step) -> PipelineOp {
    match step {
        Step::Gemv { .. } => PipelineOp::Gemv,
        Step::GemvResidual { .. } => PipelineOp::GemvResidual,
        Step::RmsnormRotateMq { .. } => PipelineOp::RmsnormRotateMq,
    }
}

pub struct FusedPattern {
    pub ops: &'static [PipelineOp],
    pub key: KernelKey,
}

/// Greedy longest-prefix op-pattern match. Op-pattern only (Phase 1/2a: empty table).
pub fn match_prefix(table: &[FusedPattern], steps: &[Step]) -> Option<(KernelKey, usize)> {
    table
        .iter()
        .filter(|p| {
            !p.ops.is_empty()
                && p.ops.len() <= steps.len()
                && p.ops.iter().zip(steps).all(|(o, s)| *o == op_kind(s))
        })
        .max_by_key(|p| p.ops.len())
        .map(|p| (p.key, p.ops.len()))
}

const FUSED_TABLE: &[FusedPattern] = &[];
static GEMV: OnceLock<GemvFamily> = OnceLock::new();
static ROTATION: OnceLock<RotationFamily> = OnceLock::new();

pub fn execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[Step],
) -> Result<(), DispatchError> {
    let mut i = 0;
    while i < steps.len() {
        if let Some((key, len)) = match_prefix(FUSED_TABLE, &steps[i..]) {
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
        }
        Step::RmsnormRotateMq { x, norm_weight, x_plain, out, awq_scale, k, eps } => {
            let rotation = ROTATION.get_or_init(RotationFamily::new);
            rotation.run(ctx, gpu, RotationParams {
                x, x_up: None, w_norm: Some(norm_weight),
                x_plain, x_rot: out, awq_scale: *awq_scale, k: *k, eps: *eps,
                batch_size: 1, variant: RotationVariant::WithRmsnorm,
                givens_pairs: None, givens_theta: None, givens_scales: None, givens_krot: None,
            }).map_err(|e| DispatchError::Hip(e.to_string()))
        }
    }
}

fn launch_fused(
    _gpu: &mut Gpu, _ctx: &DispatchCtx, key: KernelKey, _steps: &[Step],
) -> Result<(), DispatchError> {
    Err(DispatchError::MissingImpl { key })
}
