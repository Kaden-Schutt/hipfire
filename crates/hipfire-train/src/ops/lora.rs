// SPDX-License-Identifier: Apache-2.0
//! LoRA-adapted linear: `y = x·Wᵀ + scale·(x·Aᵀ)·Bᵀ`.
//!
//! Base `W:[n,k]` is FROZEN (no dW). Trainable adapters `A:[r,k]`, `B:[n,r]`,
//! `scale = alpha/r`. Composed entirely from the verified `linear` op — no new
//! kernel. The forward saves the low-rank activation `h = x·Aᵀ` `[m,r]` for the
//! backward.
//!
//! Backward (given dy):
//!   dyl = scale·dy
//!   dB  = dylᵀ·h          dh = dyl·B
//!   dA  = dhᵀ·x
//!   dx  = dy·W + dh·A      (base contributes to input grad even though frozen)

use super::linear::{linear_backward_w, linear_backward_x, linear_forward};
use hipfire_rdna::{Gpu, GpuTensor, HipResult};

/// Forward. `x:[m*k]`, `w:[n*k]`, `a:[r*k]`, `b:[n*r]`; outputs `y:[m*n]` and
/// saves `h:[m*r]` for the backward.
#[allow(clippy::too_many_arguments)]
pub fn lora_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    a: &GpuTensor,
    b: &GpuTensor,
    h: &GpuTensor,    // out scratch [m*r]
    lora: &GpuTensor, // out scratch [m*n]
    y: &GpuTensor,    // out [m*n]
    m: usize,
    k: usize,
    n: usize,
    r: usize,
    scale: f32,
) -> HipResult<()> {
    linear_forward(gpu, x, w, y, m, k, n)?; // base = x·Wᵀ
    linear_forward(gpu, x, a, h, m, k, r)?; // h = x·Aᵀ
    linear_forward(gpu, h, b, lora, m, r, n)?; // lora = h·Bᵀ
    gpu.scale_f32(lora, scale)?;
    gpu.add_inplace_f32(y, lora)?; // y = base + scale·lora
    Ok(())
}

/// Backward. Writes `da:[r*k]`, `db:[n*r]`, `dx:[m*k]`. `dyl` is a scratch
/// `[m*n]` used to hold `scale·dy`.
#[allow(clippy::too_many_arguments)]
pub fn lora_backward(
    gpu: &mut Gpu,
    dy: &GpuTensor,
    x: &GpuTensor,
    w: &GpuTensor,
    a: &GpuTensor,
    b: &GpuTensor,
    h: &GpuTensor,
    dyl: &GpuTensor, // scratch [m*n]
    dh: &GpuTensor,  // scratch [m*r]
    da: &GpuTensor,  // out [r*k]
    db: &GpuTensor,  // out [n*r]
    dx: &GpuTensor,  // out [m*k]
    m: usize,
    k: usize,
    n: usize,
    r: usize,
    scale: f32,
    accumulate_dx: bool, // add into dx instead of overwriting (q/k/v fan-in)
) -> HipResult<()> {
    // dyl = scale·dy
    gpu.memcpy_dtod_auto(&dyl.buf, &dy.buf, m * n * 4)?;
    gpu.scale_f32(dyl, scale)?;

    // dB = dylᵀ·h  [n,r];  dh = dyl·B  [m,r]
    linear_backward_w(gpu, dyl, h, db, m, r, n, false)?;
    linear_backward_x(gpu, dyl, b, dh, m, r, n, false)?;

    // dA = dhᵀ·x  [r,k]
    linear_backward_w(gpu, dh, x, da, m, k, r, false)?;

    // dx += dy·W (base) + dh·A (lora). The base term honors accumulate_dx; the
    // lora term always accumulates on top of it.
    linear_backward_x(gpu, dy, w, dx, m, k, n, accumulate_dx)?;
    linear_backward_x(gpu, dh, a, dx, m, k, r, true)?;
    Ok(())
}
