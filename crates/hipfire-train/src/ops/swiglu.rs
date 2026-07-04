// SPDX-License-Identifier: Apache-2.0
//! SwiGLU: `out = silu(gate) * up`. Elementwise; `n` total elements.

use hipfire_rdna::{Gpu, GpuTensor, HipResult};

pub fn swiglu_forward(
    gpu: &mut Gpu,
    gate: &GpuTensor,
    up: &GpuTensor,
    out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    gpu.swiglu_train_fwd(gate, up, out, n)
}

/// `d_up = d_out·silu(gate)`, `d_gate = d_out·up·silu'(gate)`.
pub fn swiglu_backward(
    gpu: &mut Gpu,
    d_out: &GpuTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    d_gate: &GpuTensor,
    d_up: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    gpu.swiglu_train_bwd(d_out, gate, up, d_gate, d_up, n)
}
