// SPDX-License-Identifier: Apache-2.0
//! Row-softmax: `p = softmax(s)` along the last dim.
//!
//! Forward writes `p` into `y`; backward reuses it. `rows` = number of
//! independent softmax rows (e.g. q_heads × seq for attention scores), `n` =
//! row length (e.g. key length).

use hipfire_rdna::{Gpu, GpuTensor, HipResult};

pub fn softmax_forward(
    gpu: &mut Gpu,
    s: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    n: usize,
) -> HipResult<()> {
    gpu.softmax_train_fwd(s, y, rows, n)
}

/// `ds_i = p_i (dy_i − Σ_j dy_j p_j)`. `p` is the saved forward output.
pub fn softmax_backward(
    gpu: &mut Gpu,
    dy: &GpuTensor,
    p: &GpuTensor,
    ds: &GpuTensor,
    rows: usize,
    n: usize,
) -> HipResult<()> {
    gpu.softmax_train_bwd(dy, p, ds, rows, n)
}
