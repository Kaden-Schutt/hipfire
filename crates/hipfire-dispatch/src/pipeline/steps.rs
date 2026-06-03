// SPDX-License-Identifier: MIT OR Apache-2.0
//! Op-list interpreter: a `&[Step]` is walked, fusing runs of ops into single
//! kernels where a matcher entry covers them, else launching each op via its
//! per-op fallback. Phase 1: GEMV-only, empty fusion table (all fallback).

use rdna_compute::GpuTensor;

use crate::families::gemv::WeightRef;
use crate::types::{KernelKey, PipelineOp};

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
#[allow(dead_code)]
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
