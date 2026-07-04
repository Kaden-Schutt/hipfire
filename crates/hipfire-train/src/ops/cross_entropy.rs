// SPDX-License-Identifier: Apache-2.0
//! Fused cross-entropy (logsoftmax + NLL) with `ignore_index` masking.
//!
//! Forward and backward are one call: per-row `loss` and `d_logits = softmax −
//! onehot` (both zero for ignored rows). `d_logits` is a SUM-reduction gradient
//! — divide by the valid-token count for a mean loss (matching `sft.py`).

use hipfire_rdna::{Gpu, GpuTensor, HipResult};

/// `targets` are integer-valued f32 (class id, or `ignore_index`). Writes
/// `loss` `[rows]` and `d_logits` `[rows*v]`.
#[allow(clippy::too_many_arguments)]
pub fn cross_entropy(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    targets: &GpuTensor,
    loss: &GpuTensor,
    d_logits: &GpuTensor,
    rows: usize,
    v: usize,
    ignore_index: i32,
) -> HipResult<()> {
    gpu.cross_entropy_train(logits, targets, loss, d_logits, rows, v, ignore_index)
}
