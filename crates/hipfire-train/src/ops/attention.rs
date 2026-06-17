// SPDX-License-Identifier: Apache-2.0
//! Single-head causal scaled-dot-product attention (fp32).
//!
//! Composed from the verified `gemm_f32_train` + `softmax` ops plus the
//! causal-mask kernel. `Q`,`K`,`V`: `[seq*d]`. The GQA multi-head wrapper
//! (head loop + kv-head broadcast + grad accumulation) builds on this.
//!
//! fwd:  scores = (Q·Kᵀ)·scale → causal-mask → P = softmax(scores) → ctx = P·V
//! bwd:  dP = d_ctx·Vᵀ; dV = Pᵀ·d_ctx; dscores = softmax_bwd(dP,P)·scale;
//!       dQ = dscores·K; dK = dscoresᵀ·Q

use super::softmax::{softmax_backward, softmax_forward};
use rdna_compute::{Gpu, GpuTensor, HipResult};

/// Forward. `scale` is typically `1/sqrt(d)`. Scratch `scores`,`p`: `[seq*seq]`;
/// `ctx`: `[seq*d]` out. `p` (softmax output) is saved for the backward.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_forward(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    scores: &GpuTensor, // scratch [seq*seq]
    p: &GpuTensor,      // out (= softmax) [seq*seq]
    ctx: &GpuTensor,    // out [seq*d]
    seq: usize,
    d: usize,
    scale: f32,
) -> HipResult<()> {
    // scores = Q·Kᵀ  [seq,seq]
    gpu.gemm_f32_train(q, k, scores, seq, seq, d, d, d, false, true)?;
    gpu.scale_f32(scores, scale)?;
    gpu.causal_mask_train(scores, seq, seq)?;
    softmax_forward(gpu, scores, p, seq, seq)?;
    // ctx = P·V  [seq,d]
    gpu.gemm_f32_train(p, v, ctx, seq, d, seq, seq, d, false, false)?;
    Ok(())
}

/// Backward. Writes `dq`,`dk`,`dv`: `[seq*d]`. Scratch `dp`,`dscores`:
/// `[seq*seq]`. `p` is the saved forward softmax.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_backward(
    gpu: &mut Gpu,
    d_ctx: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    p: &GpuTensor,
    dp: &GpuTensor,      // scratch [seq*seq]
    dscores: &GpuTensor, // scratch [seq*seq]
    dq: &GpuTensor,
    dk: &GpuTensor,
    dv: &GpuTensor,
    seq: usize,
    d: usize,
    scale: f32,
) -> HipResult<()> {
    // dP = d_ctx·Vᵀ  [seq,seq]
    gpu.gemm_f32_train(d_ctx, v, dp, seq, seq, d, d, d, false, true)?;
    // dV = Pᵀ·d_ctx  [seq,d]
    gpu.gemm_f32_train(p, d_ctx, dv, seq, d, seq, seq, d, true, false)?;
    // dscores = softmax_bwd(dP, P) · scale
    softmax_backward(gpu, dp, p, dscores, seq, seq)?;
    gpu.scale_f32(dscores, scale)?;
    // dQ = dscores·K  [seq,d]
    gpu.gemm_f32_train(dscores, k, dq, seq, d, seq, seq, d, false, false)?;
    // dK = dscoresᵀ·Q  [seq,d]
    gpu.gemm_f32_train(dscores, q, dk, seq, d, seq, seq, d, true, false)?;
    Ok(())
}
