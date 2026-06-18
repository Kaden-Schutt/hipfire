// SPDX-License-Identifier: Apache-2.0
//! RMSNorm: `y[t,i] = x[t,i] / sqrt(mean_i x² + eps) · w[i]`.
//!
//! Thin wrappers over the `rmsnorm_train_{fwd,bwd}` kernels. The forward saves
//! `1/r` per row into `rinv`, which the backward consumes (no recompute). `dw`
//! is atomic-accumulated, so zero it before calling backward.

use rdna_compute::{Gpu, GpuTensor, HipResult};

/// Forward. `x`,`y`: `[rows*h]`; `w`: `[h]`; `rinv`: `[rows]` (output, saved
/// for backward).
pub fn rmsnorm_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    y: &GpuTensor,
    rinv: &GpuTensor,
    rows: usize,
    h: usize,
    eps: f32,
) -> HipResult<()> {
    gpu.rmsnorm_train_fwd(x, w, y, rinv, rows, h, eps)
}

/// Backward. Writes `dx` `[rows*h]`; atomic-accumulates `dw` `[h]` (zero first).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_backward(
    gpu: &mut Gpu,
    dy: &GpuTensor,
    x: &GpuTensor,
    w: &GpuTensor,
    rinv: &GpuTensor,
    dx: &GpuTensor,
    dw: &GpuTensor,
    rows: usize,
    h: usize,
) -> HipResult<()> {
    gpu.rmsnorm_train_bwd(dy, x, w, rinv, dx, dw, rows, h)
}
