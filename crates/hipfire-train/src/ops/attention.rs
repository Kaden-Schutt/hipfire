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
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

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

// ─── Masked / bidirectional variant (additive bias) ─────────────────────────
//
// Same math as `sdpa_forward`/`sdpa_backward` but (a) supports rectangular
// scores `[seq_q, seq_k]` (q rows = seq_q, kv rows = seq_k) and (b) replaces the
// causal 0/−inf in-place mask with a caller-supplied *additive* bias
// `[seq_q*seq_k]` (0 to keep, −inf to drop) added to the scaled scores before
// softmax. `bias == None` is a fully bidirectional attention over all keys.
// This is the primitive DSpark block attention needs: bidirectional over
// `[context_KV ++ block_KV]`, which the causal path cannot express.
//
// The bias is a constant (no learnable grad): softmax_backward already zeroes
// grads for masked (p≈0) entries, so backward is the plain rectangular SDPA
// backward with no masking step.

/// Masked/bidirectional forward. `scores`,`p`: scratch/out `[seq_q*seq_k]`;
/// `ctx`: out `[seq_q*d]`. `bias` (`Some([seq_q*seq_k])`) is added to the scaled
/// scores before softmax; `None` = fully bidirectional. `p` is saved for bwd.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_forward_masked(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    scores: &GpuTensor, // scratch [seq_q*seq_k]
    p: &GpuTensor,      // out (= softmax) [seq_q*seq_k]
    ctx: &GpuTensor,    // out [seq_q*d]
    seq_q: usize,
    seq_k: usize,
    d: usize,
    scale: f32,
    bias: Option<&GpuTensor>, // [seq_q*seq_k] additive (0 or −inf), shared across heads
) -> HipResult<()> {
    // scores = Q·Kᵀ  [seq_q,seq_k]
    gpu.gemm_f32_train(q, k, scores, seq_q, seq_k, d, d, d, false, true)?;
    gpu.scale_f32(scores, scale)?;
    if let Some(b) = bias {
        // scores += bias (in-place; each thread reads+writes its own element)
        gpu.add_f32(scores, b, scores)?;
    }
    softmax_forward(gpu, scores, p, seq_q, seq_k)?;
    // ctx = P·V  [seq_q,d]
    gpu.gemm_f32_train(p, v, ctx, seq_q, d, seq_k, seq_k, d, false, false)?;
    Ok(())
}

/// Masked/bidirectional backward. Writes `dq` `[seq_q*d]`, `dk`/`dv`
/// `[seq_k*d]`. Scratch `dp`,`dscores`: `[seq_q*seq_k]`. `p` is the saved
/// forward softmax. The bias has no grad; masked (p≈0) entries carry zero grad.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_backward_masked(
    gpu: &mut Gpu,
    d_ctx: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    p: &GpuTensor,
    dp: &GpuTensor,      // scratch [seq_q*seq_k]
    dscores: &GpuTensor, // scratch [seq_q*seq_k]
    dq: &GpuTensor,      // [seq_q*d]
    dk: &GpuTensor,      // [seq_k*d]
    dv: &GpuTensor,      // [seq_k*d]
    seq_q: usize,
    seq_k: usize,
    d: usize,
    scale: f32,
) -> HipResult<()> {
    // dP = d_ctx·Vᵀ  [seq_q,seq_k]
    gpu.gemm_f32_train(d_ctx, v, dp, seq_q, seq_k, d, d, d, false, true)?;
    // dV = Pᵀ·d_ctx  [seq_k,d]
    gpu.gemm_f32_train(p, d_ctx, dv, seq_k, d, seq_q, seq_k, d, true, false)?;
    // dscores = softmax_bwd(dP, P) · scale
    softmax_backward(gpu, dp, p, dscores, seq_q, seq_k)?;
    gpu.scale_f32(dscores, scale)?;
    // dQ = dscores·K  [seq_q,d]
    gpu.gemm_f32_train(dscores, k, dq, seq_q, d, seq_k, seq_k, d, false, false)?;
    // dK = dscoresᵀ·Q  [seq_k,d]
    gpu.gemm_f32_train(dscores, q, dk, seq_k, d, seq_q, seq_k, d, true, false)?;
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
    // per-head scratch (reused across the loop) → back to the pool; no Drop.
    for t in [qh, kh, vh, scores, ph, ctxh] {
        gpu.free_tensor(t)?;
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
    for t in [qh, kh, vh, ph, dctxh, dp, dsc, dqh, dkh, dvh] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

// ─── GQA masked/bidirectional wrappers ───────────────────────────────────────
//
// Mirror `gqa_forward`/`gqa_backward` but call the masked SDPA and thread a
// single `bias` `[seq_q*seq_k]` shared across all heads. q: `[seq_q, n_heads*d]`
// (q_dim), k/v: `[seq_k, n_kv*d]` (kv_dim). `p_all`: `[n_heads*seq_q*seq_k]`.

/// GQA masked forward. Writes `ctx` `[seq_q*q_dim]` and saves `p_all`
/// `[n_heads*seq_q*seq_k]`. `bias` (`Some([seq_q*seq_k])`) is shared across
/// heads; `None` = fully bidirectional.
#[allow(clippy::too_many_arguments)]
pub fn gqa_forward_masked(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    p_all: &GpuTensor,
    ctx: &GpuTensor,
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    n_kv: usize,
    d: usize,
    scale: f32,
    bias: Option<&GpuTensor>, // [seq_q*seq_k], shared across heads
) -> HipResult<()> {
    let q_dim = n_heads * d;
    let kv_dim = n_kv * d;
    let group = n_heads / n_kv;

    let qh = gpu.zeros(&[seq_q * d], DType::F32)?;
    let kh = gpu.zeros(&[seq_k * d], DType::F32)?;
    let vh = gpu.zeros(&[seq_k * d], DType::F32)?;
    let scores = gpu.zeros(&[seq_q * seq_k], DType::F32)?;
    let ph = gpu.zeros(&[seq_q * seq_k], DType::F32)?;
    let ctxh = gpu.zeros(&[seq_q * d], DType::F32)?;

    for h in 0..n_heads {
        let kvh = h / group;
        gpu.strided_copy_2d(q, h * d, q_dim, &qh, 0, d, seq_q, d, false)?;
        gpu.strided_copy_2d(k, kvh * d, kv_dim, &kh, 0, d, seq_k, d, false)?;
        gpu.strided_copy_2d(v, kvh * d, kv_dim, &vh, 0, d, seq_k, d, false)?;

        sdpa_forward_masked(
            gpu, &qh, &kh, &vh, &scores, &ph, &ctxh, seq_q, seq_k, d, scale, bias,
        )?;

        // save p_h → p_all[h], scatter ctx_h → ctx[:, h]
        gpu.strided_copy_2d(
            &ph,
            0,
            seq_k,
            p_all,
            h * seq_q * seq_k,
            seq_k,
            seq_q,
            seq_k,
            false,
        )?;
        gpu.strided_copy_2d(&ctxh, 0, d, ctx, h * d, q_dim, seq_q, d, false)?;
    }
    for t in [qh, kh, vh, scores, ph, ctxh] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// GQA masked backward. Writes `dq` `[seq_q*q_dim]`, `dk`/`dv` `[seq_k*kv_dim]`.
/// **`dk` and `dv` must be zero-initialized** (heads scatter-add into them).
#[allow(clippy::too_many_arguments)]
pub fn gqa_backward_masked(
    gpu: &mut Gpu,
    d_ctx: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    p_all: &GpuTensor,
    dq: &GpuTensor,
    dk: &GpuTensor,
    dv: &GpuTensor,
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    n_kv: usize,
    d: usize,
    scale: f32,
) -> HipResult<()> {
    let q_dim = n_heads * d;
    let kv_dim = n_kv * d;
    let group = n_heads / n_kv;

    let qh = gpu.zeros(&[seq_q * d], DType::F32)?;
    let kh = gpu.zeros(&[seq_k * d], DType::F32)?;
    let vh = gpu.zeros(&[seq_k * d], DType::F32)?;
    let ph = gpu.zeros(&[seq_q * seq_k], DType::F32)?;
    let dctxh = gpu.zeros(&[seq_q * d], DType::F32)?;
    let dp = gpu.zeros(&[seq_q * seq_k], DType::F32)?;
    let dsc = gpu.zeros(&[seq_q * seq_k], DType::F32)?;
    let dqh = gpu.zeros(&[seq_q * d], DType::F32)?;
    let dkh = gpu.zeros(&[seq_k * d], DType::F32)?;
    let dvh = gpu.zeros(&[seq_k * d], DType::F32)?;

    for h in 0..n_heads {
        let kvh = h / group;
        gpu.strided_copy_2d(q, h * d, q_dim, &qh, 0, d, seq_q, d, false)?;
        gpu.strided_copy_2d(k, kvh * d, kv_dim, &kh, 0, d, seq_k, d, false)?;
        gpu.strided_copy_2d(v, kvh * d, kv_dim, &vh, 0, d, seq_k, d, false)?;
        gpu.strided_copy_2d(
            p_all,
            h * seq_q * seq_k,
            seq_k,
            &ph,
            0,
            seq_k,
            seq_q,
            seq_k,
            false,
        )?;
        gpu.strided_copy_2d(d_ctx, h * d, q_dim, &dctxh, 0, d, seq_q, d, false)?;

        sdpa_backward_masked(
            gpu, &dctxh, &qh, &kh, &vh, &ph, &dp, &dsc, &dqh, &dkh, &dvh, seq_q, seq_k, d, scale,
        )?;

        // dq: disjoint write; dk/dv: scatter-ADD into shared kv head
        gpu.strided_copy_2d(&dqh, 0, d, dq, h * d, q_dim, seq_q, d, false)?;
        gpu.strided_copy_2d(&dkh, 0, d, dk, kvh * d, kv_dim, seq_k, d, true)?;
        gpu.strided_copy_2d(&dvh, 0, d, dv, kvh * d, kv_dim, seq_k, d, true)?;
    }
    for t in [qh, kh, vh, ph, dctxh, dp, dsc, dqh, dkh, dvh] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}
