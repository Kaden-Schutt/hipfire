// SPDX-License-Identifier: Apache-2.0
//! One pre-norm LLaMA transformer block, fp32, with LoRA on q_proj and v_proj.
//!
//! Forward saves every activation the backward needs into `BlockActivations`;
//! backward consumes them and produces LoRA gradients + the input gradient.
//! Base weights are FROZEN (no weight gradients except the LoRA adapters).
//!
//! fwd:  xn1 = rmsnorm(x); q = lora_q(xn1); k = lin(xn1); v = lora_v(xn1);
//!       q,k = rope(q),rope(k); ctx = gqa(q,k,v); attn = lin_o(ctx);
//!       x_mid = x + attn; xn2 = rmsnorm(x_mid);
//!       act = swiglu(lin_g(xn2), lin_u(xn2)); mlp = lin_d(act); x_out = x_mid + mlp

use crate::ops::attention::{gqa_backward, gqa_forward};
use crate::ops::linear::{linear_backward_w, linear_backward_x, linear_forward};
use crate::ops::lora::{lora_backward, lora_forward};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use crate::ops::rope::{rope_backward, rope_forward};
use crate::ops::swiglu::{swiglu_backward, swiglu_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

#[derive(Clone, Copy)]
pub struct BlockDims {
    pub seq: usize,
    pub h: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub lora_scale: f32,
    pub lora_rank: usize,
}

impl BlockDims {
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Frozen base weights for one block (HF row-major `[out, in]`).
pub struct BlockWeights<'a> {
    pub norm1: &'a GpuTensor,
    pub wq: &'a GpuTensor,
    pub wk: &'a GpuTensor,
    pub wv: &'a GpuTensor,
    pub wo: &'a GpuTensor,
    pub norm2: &'a GpuTensor,
    pub wgate: &'a GpuTensor,
    pub wup: &'a GpuTensor,
    pub wdown: &'a GpuTensor,
}

/// Trainable LoRA adapters (q_proj, v_proj).
pub struct BlockLora<'a> {
    pub aq: &'a GpuTensor,
    pub bq: &'a GpuTensor,
    pub av: &'a GpuTensor,
    pub bv: &'a GpuTensor,
}

/// Gradients of the trainable block params: LoRA adapters + the two RMSNorm
/// weights (the layernorms QTIP recovery FT tunes). Base linears stay frozen.
pub struct BlockLoraGrad {
    pub daq: GpuTensor,
    pub dbq: GpuTensor,
    pub dav: GpuTensor,
    pub dbv: GpuTensor,
    pub dnorm1: GpuTensor, // [h]
    pub dnorm2: GpuTensor, // [h]
}

/// Saved forward activations needed by the backward pass.
pub struct BlockActivations {
    pub xn1: GpuTensor,
    pub rinv1: GpuTensor,
    pub hq: GpuTensor,
    pub hv: GpuTensor,
    pub q_r: GpuTensor,
    pub k_r: GpuTensor,
    pub v: GpuTensor,
    pub p_all: GpuTensor,
    pub ctx: GpuTensor,
    pub x_mid: GpuTensor,
    pub xn2: GpuTensor,
    pub rinv2: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub act: GpuTensor,
    pub pos: GpuTensor,
}

/// Forward. Returns `x_out` `[seq*h]` and the saved activations.
pub fn block_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    dims: &BlockDims,
    pos_host: &[f32],
) -> HipResult<(GpuTensor, BlockActivations)> {
    let (seq, h, inter) = (dims.seq, dims.h, dims.inter);
    let (qd, kvd, r) = (dims.q_dim(), dims.kv_dim(), dims.lora_rank);

    let xn1 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv1 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, x, w.norm1, &xn1, &rinv1, seq, h, dims.eps)?;

    // q = lora_q(xn1), k = lin(xn1), v = lora_v(xn1)
    let hq = gpu.zeros(&[seq * r], DType::F32)?;
    let loraq_s = gpu.zeros(&[seq * qd], DType::F32)?;
    let q = gpu.zeros(&[seq * qd], DType::F32)?;
    lora_forward(
        gpu,
        &xn1,
        w.wq,
        lora.aq,
        lora.bq,
        &hq,
        &loraq_s,
        &q,
        seq,
        h,
        qd,
        r,
        dims.lora_scale,
    )?;
    let k = gpu.zeros(&[seq * kvd], DType::F32)?;
    linear_forward(gpu, &xn1, w.wk, &k, seq, h, kvd)?;
    let hv = gpu.zeros(&[seq * r], DType::F32)?;
    let lorav_s = gpu.zeros(&[seq * kvd], DType::F32)?;
    let v = gpu.zeros(&[seq * kvd], DType::F32)?;
    lora_forward(
        gpu,
        &xn1,
        w.wv,
        lora.av,
        lora.bv,
        &hv,
        &lorav_s,
        &v,
        seq,
        h,
        kvd,
        r,
        dims.lora_scale,
    )?;

    // rope(q), rope(k)
    let pos = gpu.upload_f32(pos_host, &[seq])?;
    let q_r = gpu.zeros(&[seq * qd], DType::F32)?;
    rope_forward(
        gpu,
        &q,
        &q_r,
        &pos,
        seq * dims.n_heads,
        dims.n_heads,
        dims.head_dim,
        dims.rope_base,
    )?;
    let k_r = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_forward(
        gpu,
        &k,
        &k_r,
        &pos,
        seq * dims.n_kv,
        dims.n_kv,
        dims.head_dim,
        dims.rope_base,
    )?;

    // KV-compression sim-noise (KVarN-4bit + CASK merge) on post-RoPE K and V,
    // forward-only (STE) — no-op unless HIPFIRE_KVNOISE=1. See crate::kv_noise.
    let (k_r, v) = crate::kv_noise::maybe_compress_kv(
        gpu,
        k_r,
        v,
        crate::kv_noise::cfg_from_env(),
        seq,
        kvd,
        dims.head_dim,
    )?;

    // attention
    let p_all = gpu.zeros(&[dims.n_heads * seq * seq], DType::F32)?;
    let ctx = gpu.zeros(&[seq * qd], DType::F32)?;
    gqa_forward(
        gpu,
        &q_r,
        &k_r,
        &v,
        &p_all,
        &ctx,
        seq,
        dims.n_heads,
        dims.n_kv,
        dims.head_dim,
        dims.attn_scale(),
    )?;

    // o_proj + residual
    let attn = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &ctx, w.wo, &attn, seq, qd, h)?;
    let x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(x, &attn, &x_mid)?;

    // norm2 + mlp + residual
    let xn2 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv2 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, &x_mid, w.norm2, &xn2, &rinv2, seq, h, dims.eps)?;
    let gate = gpu.zeros(&[seq * inter], DType::F32)?;
    linear_forward(gpu, &xn2, w.wgate, &gate, seq, h, inter)?;
    let up = gpu.zeros(&[seq * inter], DType::F32)?;
    linear_forward(gpu, &xn2, w.wup, &up, seq, h, inter)?;
    let act = gpu.zeros(&[seq * inter], DType::F32)?;
    swiglu_forward(gpu, &gate, &up, &act, seq * inter)?;
    let mlp = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &act, w.wdown, &mlp, seq, inter, h)?;
    let x_out = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(&x_mid, &mlp, &x_out)?;

    // Return forward scratch the backward never reads (pre-rope q/k, lora
    // pre-scale, attn/mlp pre-residual) to the pool — GpuTensor has no Drop, so
    // without this each block_forward leaks ~5 MB and many-step training OOMs.
    for t in [loraq_s, q, k, lorav_s, attn, mlp] {
        gpu.free_tensor(t)?;
    }

    Ok((
        x_out,
        BlockActivations {
            xn1,
            rinv1,
            hq,
            hv,
            q_r,
            k_r,
            v,
            p_all,
            ctx,
            x_mid,
            xn2,
            rinv2,
            gate,
            up,
            act,
            pos,
        },
    ))
}

/// Backward. `d_x_out` `[seq*h]` upstream; writes LoRA grads and returns the
/// input gradient `d_x` `[seq*h]`. `x` is the original block input (for norm1).
#[allow(clippy::too_many_arguments)]
/// Gradients of the FROZEN base block linears — only produced by
/// `block_backward_full` (for from-scratch training, e.g. the PFlash drafter).
/// Recovery FT leaves the base frozen and never asks for these.
pub struct BlockWeightGrad {
    pub dwq: GpuTensor,    // [q_dim, h]
    pub dwk: GpuTensor,    // [kv_dim, h]
    pub dwv: GpuTensor,    // [kv_dim, h]
    pub dwo: GpuTensor,    // [h, q_dim]
    pub dwgate: GpuTensor, // [inter, h]
    pub dwup: GpuTensor,   // [inter, h]
    pub dwdown: GpuTensor, // [h, inter]
}

/// Return one block's saved activations to the pool (GpuTensor has no Drop).
pub fn free_block_acts(gpu: &mut Gpu, b: BlockActivations) -> HipResult<()> {
    let BlockActivations {
        xn1, rinv1, hq, hv, q_r, k_r, v, p_all, ctx, x_mid, xn2, rinv2, gate, up, act, pos,
    } = b;
    for t in [xn1, rinv1, hq, hv, q_r, k_r, v, p_all, ctx, x_mid, xn2, rinv2, gate, up, act, pos] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Recovery-FT backward: base frozen, returns LoRA + norm grads only.
pub fn block_backward(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
) -> HipResult<(GpuTensor, BlockLoraGrad)> {
    let (d_x, lora_g, _) = block_backward_inner(gpu, d_x_out, x, w, lora, acts, dims, false)?;
    Ok((d_x, lora_g))
}

/// From-scratch backward: also returns gradients for the base linears.
pub fn block_backward_full(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
) -> HipResult<(GpuTensor, BlockLoraGrad, BlockWeightGrad)> {
    let (d_x, lora_g, wg) = block_backward_inner(gpu, d_x_out, x, w, lora, acts, dims, true)?;
    Ok((d_x, lora_g, wg.expect("want_w=true ⇒ Some")))
}

#[allow(clippy::too_many_arguments)]
fn block_backward_inner(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x: &GpuTensor,
    w: &BlockWeights,
    lora: &BlockLora,
    acts: &BlockActivations,
    dims: &BlockDims,
    want_w: bool,
) -> HipResult<(GpuTensor, BlockLoraGrad, Option<BlockWeightGrad>)> {
    let (seq, h, inter) = (dims.seq, dims.h, dims.inter);
    let (qd, kvd, r) = (dims.q_dim(), dims.kv_dim(), dims.lora_rank);
    let dnorm1 = gpu.zeros(&[h], DType::F32)?; // trainable RMSNorm grads
    let dnorm2 = gpu.zeros(&[h], DType::F32)?;

    // ── MLP branch ──────────────────────────────────────────────────────────
    // x_out = x_mid + mlp  ⇒ d_mlp = d_x_out, and d_x_mid starts = d_x_out.
    let d_act = gpu.zeros(&[seq * inter], DType::F32)?;
    linear_backward_x(gpu, d_x_out, w.wdown, &d_act, seq, inter, h, false)?;
    let d_gate = gpu.zeros(&[seq * inter], DType::F32)?;
    let d_up = gpu.zeros(&[seq * inter], DType::F32)?;
    swiglu_backward(
        gpu,
        &d_act,
        &acts.gate,
        &acts.up,
        &d_gate,
        &d_up,
        seq * inter,
    )?;
    let d_xn2 = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_gate, w.wgate, &d_xn2, seq, h, inter, false)?;
    linear_backward_x(gpu, &d_up, w.wup, &d_xn2, seq, h, inter, true)?;
    // norm2 backward → adds into d_x_mid; dnorm2 is the trainable weight grad
    let d_x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x_mid.buf, &d_x_out.buf, seq * h * 4)?; // residual
    let d_xmid_norm = gpu.zeros(&[seq * h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xn2,
        &acts.x_mid,
        w.norm2,
        &acts.rinv2,
        &d_xmid_norm,
        &dnorm2,
        seq,
        h,
    )?;
    gpu.add_inplace_f32(&d_x_mid, &d_xmid_norm)?;

    // ── Attention branch ──────────────────────────────────────────────────────
    // x_mid = x + attn ⇒ d_attn = d_x_mid, d_x starts = d_x_mid.
    let d_ctx = gpu.zeros(&[seq * qd], DType::F32)?;
    linear_backward_x(gpu, &d_x_mid, w.wo, &d_ctx, seq, qd, h, false)?;
    let d_q_r = gpu.zeros(&[seq * qd], DType::F32)?;
    let d_k_r = gpu.zeros(&[seq * kvd], DType::F32)?;
    let d_v = gpu.zeros(&[seq * kvd], DType::F32)?;
    gqa_backward(
        gpu,
        &d_ctx,
        &acts.q_r,
        &acts.k_r,
        &acts.v,
        &acts.p_all,
        &d_q_r,
        &d_k_r,
        &d_v,
        seq,
        dims.n_heads,
        dims.n_kv,
        dims.head_dim,
        dims.attn_scale(),
    )?;
    // rope backward
    let d_q = gpu.zeros(&[seq * qd], DType::F32)?;
    rope_backward(
        gpu,
        &d_q_r,
        &d_q,
        &acts.pos,
        seq * dims.n_heads,
        dims.n_heads,
        dims.head_dim,
        dims.rope_base,
    )?;
    let d_k = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_backward(
        gpu,
        &d_k_r,
        &d_k,
        &acts.pos,
        seq * dims.n_kv,
        dims.n_kv,
        dims.head_dim,
        dims.rope_base,
    )?;

    // q/k/v projection backward → accumulate into d_xn1, produce LoRA grads
    let d_xn1 = gpu.zeros(&[seq * h], DType::F32)?;
    let daq = gpu.zeros(&[r * h], DType::F32)?;
    let dbq = gpu.zeros(&[qd * r], DType::F32)?;
    let dav = gpu.zeros(&[r * h], DType::F32)?;
    let dbv = gpu.zeros(&[kvd * r], DType::F32)?;
    let dyl_q = gpu.zeros(&[seq * qd], DType::F32)?;
    let dh_q = gpu.zeros(&[seq * r], DType::F32)?;
    lora_backward(
        gpu,
        &d_q,
        &acts.xn1,
        w.wq,
        lora.aq,
        lora.bq,
        &acts.hq,
        &dyl_q,
        &dh_q,
        &daq,
        &dbq,
        &d_xn1,
        seq,
        h,
        qd,
        r,
        dims.lora_scale,
        true,
    )?;
    linear_backward_x(gpu, &d_k, w.wk, &d_xn1, seq, h, kvd, true)?;
    let dyl_v = gpu.zeros(&[seq * kvd], DType::F32)?;
    let dh_v = gpu.zeros(&[seq * r], DType::F32)?;
    lora_backward(
        gpu,
        &d_v,
        &acts.xn1,
        w.wv,
        lora.av,
        lora.bv,
        &acts.hv,
        &dyl_v,
        &dh_v,
        &dav,
        &dbv,
        &d_xn1,
        seq,
        h,
        kvd,
        r,
        dims.lora_scale,
        true,
    )?;

    // norm1 backward → adds into d_x; dnorm1 is the trainable weight grad
    let d_x = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x.buf, &d_x_mid.buf, seq * h * 4)?; // residual
    let d_x_norm = gpu.zeros(&[seq * h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xn1,
        x,
        w.norm1,
        &acts.rinv1,
        &d_x_norm,
        &dnorm1,
        seq,
        h,
    )?;
    gpu.add_inplace_f32(&d_x, &d_x_norm)?;

    // Base-linear weight grads (from-scratch training only). dw = dyᵀ·x; every
    // (dy, input) pair below is already materialised above.
    let wg = if want_w {
        let dwq = gpu.zeros(&[qd * h], DType::F32)?;
        let dwk = gpu.zeros(&[kvd * h], DType::F32)?;
        let dwv = gpu.zeros(&[kvd * h], DType::F32)?;
        let dwo = gpu.zeros(&[h * qd], DType::F32)?;
        let dwgate = gpu.zeros(&[inter * h], DType::F32)?;
        let dwup = gpu.zeros(&[inter * h], DType::F32)?;
        let dwdown = gpu.zeros(&[h * inter], DType::F32)?;
        // attention projections (input = acts.xn1)
        linear_backward_w(gpu, &d_q, &acts.xn1, &dwq, seq, h, qd, false)?;
        linear_backward_w(gpu, &d_k, &acts.xn1, &dwk, seq, h, kvd, false)?;
        linear_backward_w(gpu, &d_v, &acts.xn1, &dwv, seq, h, kvd, false)?;
        // output projection (input = acts.ctx)
        linear_backward_w(gpu, &d_x_mid, &acts.ctx, &dwo, seq, qd, h, false)?;
        // MLP (gate/up input = acts.xn2; down input = acts.act)
        linear_backward_w(gpu, &d_gate, &acts.xn2, &dwgate, seq, h, inter, false)?;
        linear_backward_w(gpu, &d_up, &acts.xn2, &dwup, seq, h, inter, false)?;
        linear_backward_w(gpu, d_x_out, &acts.act, &dwdown, seq, inter, h, false)?;
        Some(BlockWeightGrad { dwq, dwk, dwv, dwo, dwgate, dwup, dwdown })
    } else {
        None
    };

    // Return internal temporaries to the pool — GpuTensor has no Drop, so without
    // this the per-step training graph climbs ~50 MB/layer and OOMs. Only the
    // returned grads (d_x, BlockLoraGrad, BlockWeightGrad) survive.
    for t in [
        d_act, d_gate, d_up, d_xn2, d_x_mid, d_xmid_norm, d_ctx, d_q_r, d_k_r, d_v, d_q, d_k,
        d_xn1, d_x_norm, dyl_q, dh_q, dyl_v, dh_v,
    ] {
        gpu.free_tensor(t)?;
    }

    Ok((
        d_x,
        BlockLoraGrad {
            daq,
            dbq,
            dav,
            dbv,
            dnorm1,
            dnorm2,
        },
        wg,
    ))
}
