// SPDX-License-Identifier: Apache-2.0
//! Elementwise sigmoid (fp32 training twin). `out = 1/(1+e^-x)`; backward
//! `d_x = d_out·out·(1-out)` consumes the saved forward output. The GLA-lite
//! forget gate `g = sigmoid(g_pre)`.

use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

/// Forward: `x` `[n]` → `out` `[n]` (allocated here).
pub fn sigmoid_forward(gpu: &mut Gpu, x: &GpuTensor, n: usize) -> HipResult<GpuTensor> {
    let out = gpu.zeros(&[n], DType::F32)?;
    gpu.sigmoid_train_fwd(x, &out, n)?;
    Ok(out)
}

/// Backward: `d_out`,`out` `[n]` → `d_x` `[n]` (allocated here). `out` is the
/// saved forward output.
pub fn sigmoid_backward(
    gpu: &mut Gpu,
    d_out: &GpuTensor,
    out: &GpuTensor,
    n: usize,
) -> HipResult<GpuTensor> {
    let d_x = gpu.zeros(&[n], DType::F32)?;
    gpu.sigmoid_train_bwd(d_out, out, &d_x, n)?;
    Ok(d_x)
}
