// SPDX-License-Identifier: MIT OR Apache-2.0
//! Op-list interpreter: a `&[Step]` is walked, fusing runs of ops into single
//! kernels where a matcher entry covers them, else launching each op via its
//! per-op fallback. Phase 1: GEMV-only, empty fusion table (all fallback).

use rdna_compute::{Gpu, GpuTensor};
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, WeightRef};
use crate::types::{DispatchError, KernelKey, PipelineOp};

/// A single op plus its operand bindings. References borrow into model-owned
/// tensors and the GPU scratch arena; the interpreter owns no buffers.
pub struct Step<'a> {
    pub op: PipelineOp,
    /// QKV = 3 weights; a plain Gemv = 1.
    pub weights: &'a [&'a WeightRef<'a>],
    /// Shared input (QKV/gate-up share one `x`).
    pub input: &'a GpuTensor,
    pub outputs: &'a [&'a GpuTensor],
}

/// A fusion table entry: an op-pattern that collapses to one kernel.
/// Phase 1 ships an empty table; Phase 2b populates it (with a full operand
/// guard layered on top of this op-pattern match).
/// Entries must have a non-empty `ops` (a zero-length pattern never matches).
pub struct FusedPattern {
    pub ops: &'static [PipelineOp],
    pub key: KernelKey,
}

/// Greedy longest-prefix op-pattern match. Returns `(key, consumed_len)` for the
/// longest entry whose op-sequence is a prefix of `steps`. **Op-pattern only** —
/// the full operand guard (shared input, dtype/awq/row_stride homogeneity) is a
/// Phase-2b concern; in Phase 1 the table is empty so this never fires.
pub fn match_prefix(table: &[FusedPattern], steps: &[Step]) -> Option<(KernelKey, usize)> {
    table
        .iter()
        .filter(|p| {
            !p.ops.is_empty()
                && p.ops.len() <= steps.len()
                && p.ops.iter().zip(steps).all(|(o, s)| *o == s.op)
        })
        .max_by_key(|p| p.ops.len())
        .map(|p| (p.key, p.ops.len()))
}

// ── Executor ────────────────────────────────────────────────────────────────

/// Phase 1: empty. Phase 2b adds `[Gemv,Gemv,Gemv]→FusedQkv*` etc.
const FUSED_TABLE: &[FusedPattern] = &[];

static GEMV: OnceLock<GemvFamily> = OnceLock::new();

/// Walk a step list. At each position, greedily fuse the longest matching run
/// (Phase 1: never matches — table empty), else launch the single op.
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

/// Per-op fallback: the always-correct discrete path. Phase 1 = `Gemv` only.
fn launch_op(gpu: &mut Gpu, ctx: &DispatchCtx, step: &Step) -> Result<(), DispatchError> {
    match step.op {
        PipelineOp::Gemv => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run_auto(ctx, gpu, step.weights[0], step.input, step.outputs[0])
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "execute_steps",
            variant: "op-not-in-phase1",
            arch: "",
            quant: "",
        }),
    }
}

/// Launch one fused kernel for a consumed run. Phase 1 has no entries.
fn launch_fused(
    _gpu: &mut Gpu,
    _ctx: &DispatchCtx,
    key: KernelKey,
    _steps: &[Step],
) -> Result<(), DispatchError> {
    Err(DispatchError::MissingImpl { key })
}
