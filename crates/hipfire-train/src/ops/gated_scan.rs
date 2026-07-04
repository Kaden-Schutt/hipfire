// SPDX-License-Identifier: Apache-2.0
//! Gated linear-recurrence scan (fp32 training twin) — the token-mixer for the
//! GLA-lite / minimal-selective-SSM PFlash drafter.
//!
//! `h[t] = g[t]*h[t-1] + (1-g[t])*u[t]`, `h[-1]=0`. `g` is an input-dependent
//! forget gate in (0,1) (the selectivity); `u` the input projection. Per-channel
//! diagonal recurrence — no shared memory (gfx1103 LDS kernels wedge the GPU).
//! Tensors are time-major `[seq*D]` (index `t*D+c`).

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

/// Forward: `g`,`u` `[seq*D]` → `h_out` `[seq*D]` (allocated here).
pub fn gated_scan_forward(
    gpu: &mut Gpu,
    g: &GpuTensor,
    u: &GpuTensor,
    seq: usize,
    d: usize,
) -> HipResult<GpuTensor> {
    let h_out = gpu.zeros(&[seq * d], DType::F32)?;
    gpu.gated_scan_fwd(g, u, &h_out, seq, d)?;
    Ok(h_out)
}

/// Backward: given `d_hout` `[seq*D]` (dL/dh[t]), produce `(d_g, d_u)` `[seq*D]`.
/// `h_out` is the forward output (needed for `h[t-1]`). Allocates both grads.
pub fn gated_scan_backward(
    gpu: &mut Gpu,
    g: &GpuTensor,
    u: &GpuTensor,
    h_out: &GpuTensor,
    d_hout: &GpuTensor,
    seq: usize,
    d: usize,
) -> HipResult<(GpuTensor, GpuTensor)> {
    let d_g = gpu.zeros(&[seq * d], DType::F32)?;
    let d_u = gpu.zeros(&[seq * d], DType::F32)?;
    gpu.gated_scan_bwd(g, u, h_out, d_hout, &d_g, &d_u, seq, d)?;
    Ok((d_g, d_u))
}
