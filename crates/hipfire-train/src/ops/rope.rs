// SPDX-License-Identifier: Apache-2.0
//! RoPE (HF-Llama half-split). Parameter-free, so backward only produces the
//! input gradient (rotation by −angle). `x`/`out`: `[rows*d]` with
//! rows = seq*n_heads; `pos`: `[seq]`.

use rdna_compute::{Gpu, GpuTensor, HipResult};

pub fn rope_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    out: &GpuTensor,
    pos: &GpuTensor,
    rows: usize,
    n_heads: usize,
    d: usize,
    base: f32,
) -> HipResult<()> {
    gpu.rope_train_fwd(x, out, pos, rows, n_heads, d, base)
}

pub fn rope_backward(
    gpu: &mut Gpu,
    d_out: &GpuTensor,
    dx: &GpuTensor,
    pos: &GpuTensor,
    rows: usize,
    n_heads: usize,
    d: usize,
    base: f32,
) -> HipResult<()> {
    gpu.rope_train_bwd(d_out, dx, pos, rows, n_heads, d, base)
}
