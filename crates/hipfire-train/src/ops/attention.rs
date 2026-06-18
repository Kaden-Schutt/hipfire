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
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

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

// ─── GQA multi-head wrapper ──────────────────────────────────────────────────
//
// q: [seq, n_heads*d] (q_dim), k/v: [seq, n_kv*d] (kv_dim). Each kv head serves
// `group = n_heads / n_kv` query heads. Reuses the verified single-head sdpa via
// gather/scatter: extract each head to contiguous scratch, run sdpa, scatter the
// result back; on backward, dQ scatters (disjoint) while dK/dV scatter-ADD into
// their shared kv head. `p_all` ([n_heads*seq*seq]) saves every head's softmax
// for the backward.

/// GQA forward. Writes `ctx` `[seq*q_dim]` and saves `p_all`
/// `[n_heads*seq*seq]`. `scale` is typically `1/sqrt(d)`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_forward(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    p_all: &GpuTensor,
    ctx: &GpuTensor,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    d: usize,
    scale: f32,
) -> HipResult<()> {
    let q_dim = n_heads * d;
    let kv_dim = n_kv * d;
    let group = n_heads / n_kv;

    let qh = gpu.zeros(&[seq * d], DType::F32)?;
    let kh = gpu.zeros(&[seq * d], DType::F32)?;
    let vh = gpu.zeros(&[seq * d], DType::F32)?;
    let scores = gpu.zeros(&[seq * seq], DType::F32)?;
    let ph = gpu.zeros(&[seq * seq], DType::F32)?;
    let ctxh = gpu.zeros(&[seq * d], DType::F32)?;

    for h in 0..n_heads {
        let kvh = h / group;
        gpu.strided_copy_2d(q, h * d, q_dim, &qh, 0, d, seq, d, false)?;
        gpu.strided_copy_2d(k, kvh * d, kv_dim, &kh, 0, d, seq, d, false)?;
        gpu.strided_copy_2d(v, kvh * d, kv_dim, &vh, 0, d, seq, d, false)?;

        sdpa_forward(gpu, &qh, &kh, &vh, &scores, &ph, &ctxh, seq, d, scale)?;

        // save p_h → p_all[h], scatter ctx_h → ctx[:, h]
        gpu.strided_copy_2d(&ph, 0, seq, p_all, h * seq * seq, seq, seq, seq, false)?;
        gpu.strided_copy_2d(&ctxh, 0, d, ctx, h * d, q_dim, seq, d, false)?;
    }
    Ok(())
}

/// GQA backward. Writes `dq` `[seq*q_dim]`, `dk`/`dv` `[seq*kv_dim]`.
/// **`dk` and `dv` must be zero-initialized** (heads scatter-add into them).
#[allow(clippy::too_many_arguments)]
pub fn gqa_backward(
    gpu: &mut Gpu,
    d_ctx: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    p_all: &GpuTensor,
    dq: &GpuTensor,
    dk: &GpuTensor,
    dv: &GpuTensor,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    d: usize,
    scale: f32,
) -> HipResult<()> {
    let q_dim = n_heads * d;
    let kv_dim = n_kv * d;
    let group = n_heads / n_kv;

    let qh = gpu.zeros(&[seq * d], DType::F32)?;
    let kh = gpu.zeros(&[seq * d], DType::F32)?;
    let vh = gpu.zeros(&[seq * d], DType::F32)?;
    let ph = gpu.zeros(&[seq * seq], DType::F32)?;
    let dctxh = gpu.zeros(&[seq * d], DType::F32)?;
    let dp = gpu.zeros(&[seq * seq], DType::F32)?;
    let dsc = gpu.zeros(&[seq * seq], DType::F32)?;
    let dqh = gpu.zeros(&[seq * d], DType::F32)?;
    let dkh = gpu.zeros(&[seq * d], DType::F32)?;
    let dvh = gpu.zeros(&[seq * d], DType::F32)?;

    for h in 0..n_heads {
        let kvh = h / group;
        gpu.strided_copy_2d(q, h * d, q_dim, &qh, 0, d, seq, d, false)?;
        gpu.strided_copy_2d(k, kvh * d, kv_dim, &kh, 0, d, seq, d, false)?;
        gpu.strided_copy_2d(v, kvh * d, kv_dim, &vh, 0, d, seq, d, false)?;
        gpu.strided_copy_2d(p_all, h * seq * seq, seq, &ph, 0, seq, seq, seq, false)?;
        gpu.strided_copy_2d(d_ctx, h * d, q_dim, &dctxh, 0, d, seq, d, false)?;

        sdpa_backward(
            gpu, &dctxh, &qh, &kh, &vh, &ph, &dp, &dsc, &dqh, &dkh, &dvh, seq, d, scale,
        )?;

        // dq: disjoint write; dk/dv: scatter-ADD into shared kv head
        gpu.strided_copy_2d(&dqh, 0, d, dq, h * d, q_dim, seq, d, false)?;
        gpu.strided_copy_2d(&dkh, 0, d, dk, kvh * d, kv_dim, seq, d, true)?;
        gpu.strided_copy_2d(&dvh, 0, d, dv, kvh * d, kv_dim, seq, d, true)?;
    }
    Ok(())
}
