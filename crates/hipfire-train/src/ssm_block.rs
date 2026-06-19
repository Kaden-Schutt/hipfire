// SPDX-License-Identifier: Apache-2.0
//! One GLA-lite / minimal-selective-SSM block, fp32. The token-mixer is a gated
//! linear recurrence (`gated_scan`) instead of attention — sharing the qwen3.5
//! target's SSM inductive bias (P5: the attention drafter ceilings at +0.47 on
//! the SSM-driven target). All base weights are trainable (from-scratch).
//!
//! fwd:  xn1 = rmsnorm(x); u = lin_u(xn1); g = sigmoid(lin_g(xn1));
//!       hs = gated_scan(g, u); mix = lin_o(hs); x_mid = x + mix;
//!       xn2 = rmsnorm(x_mid); act = swiglu(lin_gate(xn2), lin_up(xn2));
//!       x_out = x_mid + lin_down(act)

use crate::ops::gated_scan::{gated_scan_backward, gated_scan_forward};
use crate::ops::linear::{linear_backward_w, linear_backward_x, linear_forward};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use crate::ops::sigmoid::{sigmoid_backward, sigmoid_forward};
use crate::ops::swiglu::{swiglu_backward, swiglu_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

#[derive(Clone, Copy)]
pub struct SsmBlockDims {
    pub seq: usize,
    pub h: usize,
    pub inter: usize,
    pub eps: f32,
}

/// Trainable weights for one GLA-lite block (HF row-major `[out, in]`).
pub struct SsmBlockWeights<'a> {
    pub norm1: &'a GpuTensor,  // [h]
    pub w_u: &'a GpuTensor,    // [h, h]   input projection
    pub w_g: &'a GpuTensor,    // [h, h]   gate projection
    pub w_o: &'a GpuTensor,    // [h, h]   output projection (post-scan)
    pub norm2: &'a GpuTensor,  // [h]
    pub wgate: &'a GpuTensor,  // [inter, h]
    pub wup: &'a GpuTensor,    // [inter, h]
    pub wdown: &'a GpuTensor,  // [h, inter]
}

/// Gradients for every trainable weight of one GLA-lite block.
pub struct SsmBlockGrad {
    pub dnorm1: GpuTensor,
    pub dw_u: GpuTensor,
    pub dw_g: GpuTensor,
    pub dw_o: GpuTensor,
    pub dnorm2: GpuTensor,
    pub dwgate: GpuTensor,
    pub dwup: GpuTensor,
    pub dwdown: GpuTensor,
}

/// Saved forward activations the backward needs.
pub struct SsmBlockActivations {
    pub xn1: GpuTensor,   // [seq*h]
    pub rinv1: GpuTensor, // [seq]
    pub u: GpuTensor,     // [seq*h]
    pub g: GpuTensor,     // [seq*h]  sigmoid output
    pub hs: GpuTensor,    // [seq*h]  scan output
    pub x_mid: GpuTensor, // [seq*h]
    pub xn2: GpuTensor,   // [seq*h]
    pub rinv2: GpuTensor, // [seq]
    pub gate: GpuTensor,  // [seq*inter]
    pub up: GpuTensor,    // [seq*inter]
    pub act: GpuTensor,   // [seq*inter]
}

/// Forward. Returns `x_out` `[seq*h]` and the saved activations.
pub fn ssm_block_forward(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &SsmBlockWeights,
    dims: &SsmBlockDims,
) -> HipResult<(GpuTensor, SsmBlockActivations)> {
    let (seq, h, inter) = (dims.seq, dims.h, dims.inter);

    // token mixer: rmsnorm → u, g → scan → o_proj → residual
    let xn1 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv1 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, x, w.norm1, &xn1, &rinv1, seq, h, dims.eps)?;

    let u = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &xn1, w.w_u, &u, seq, h, h)?;
    let gpre = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &xn1, w.w_g, &gpre, seq, h, h)?;
    let g = sigmoid_forward(gpu, &gpre, seq * h)?;
    gpu.free_tensor(gpre)?;

    let hs = gated_scan_forward(gpu, &g, &u, seq, h)?;
    let mix = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &hs, w.w_o, &mix, seq, h, h)?;

    let x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(x, &mix, &x_mid)?;
    gpu.free_tensor(mix)?;

    // MLP: rmsnorm → swiglu → down → residual
    let xn2 = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv2 = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, &x_mid, w.norm2, &xn2, &rinv2, seq, h, dims.eps)?;

    let gate = gpu.zeros(&[seq * inter], DType::F32)?;
    linear_forward(gpu, &xn2, w.wgate, &gate, seq, h, inter)?;
    let up = gpu.zeros(&[seq * inter], DType::F32)?;
    linear_forward(gpu, &xn2, w.wup, &up, seq, h, inter)?;
    let act = gpu.zeros(&[seq * inter], DType::F32)?;
    swiglu_forward(gpu, &gate, &up, &act, seq * inter)?;
    let down = gpu.zeros(&[seq * h], DType::F32)?;
    linear_forward(gpu, &act, w.wdown, &down, seq, inter, h)?;

    let x_out = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.add_f32(&x_mid, &down, &x_out)?;
    gpu.free_tensor(down)?;

    let acts = SsmBlockActivations { xn1, rinv1, u, g, hs, x_mid, xn2, rinv2, gate, up, act };
    Ok((x_out, acts))
}

/// Backward. `d_x_out` `[seq*h]` → input grad `d_x` + all weight grads.
pub fn ssm_block_backward(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x_in: &GpuTensor,
    w: &SsmBlockWeights,
    acts: &SsmBlockActivations,
    dims: &SsmBlockDims,
) -> HipResult<(GpuTensor, SsmBlockGrad)> {
    let (seq, h, inter) = (dims.seq, dims.h, dims.inter);

    // x_out = x_mid + down ⇒ d_down = d_x_out, d_x_mid starts = d_x_out.
    let d_x_mid = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x_mid.buf, &d_x_out.buf, seq * h * 4)?;

    // down = lin_down(act): d_act = d_down·wdown, dwdown = d_downᵀ·act.
    let d_act = gpu.zeros(&[seq * inter], DType::F32)?;
    linear_backward_x(gpu, d_x_out, w.wdown, &d_act, seq, inter, h, false)?;
    let dwdown = gpu.zeros(&[h * inter], DType::F32)?;
    linear_backward_w(gpu, d_x_out, &acts.act, &dwdown, seq, inter, h, false)?;

    // swiglu bwd: d_act → d_gate, d_up.
    let d_gate = gpu.zeros(&[seq * inter], DType::F32)?;
    let d_up = gpu.zeros(&[seq * inter], DType::F32)?;
    swiglu_backward(gpu, &d_act, &acts.gate, &acts.up, &d_gate, &d_up, seq * inter)?;
    gpu.free_tensor(d_act)?;

    // wgate, wup grads + d_xn2 (two contributions, accumulate).
    let dwgate = gpu.zeros(&[inter * h], DType::F32)?;
    linear_backward_w(gpu, &d_gate, &acts.xn2, &dwgate, seq, h, inter, false)?;
    let dwup = gpu.zeros(&[inter * h], DType::F32)?;
    linear_backward_w(gpu, &d_up, &acts.xn2, &dwup, seq, h, inter, false)?;
    let d_xn2 = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_gate, w.wgate, &d_xn2, seq, h, inter, false)?;
    linear_backward_x(gpu, &d_up, w.wup, &d_xn2, seq, h, inter, true)?;
    gpu.free_tensor(d_gate)?;
    gpu.free_tensor(d_up)?;

    // rmsnorm2 bwd → d_xmid_norm (+dnorm2); add into d_x_mid.
    let d_xmid_norm = gpu.zeros(&[seq * h], DType::F32)?;
    let dnorm2 = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(gpu, &d_xn2, &acts.x_mid, w.norm2, &acts.rinv2, &d_xmid_norm, &dnorm2, seq, h)?;
    gpu.free_tensor(d_xn2)?;
    gpu.add_inplace_f32(&d_x_mid, &d_xmid_norm)?;
    gpu.free_tensor(d_xmid_norm)?;

    // x_mid = x + mix ⇒ d_mix = d_x_mid, d_x starts = d_x_mid.
    let d_x = gpu.zeros(&[seq * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x.buf, &d_x_mid.buf, seq * h * 4)?;

    // mix = lin_o(hs): d_hs = d_mix·w_o, dw_o = d_mixᵀ·hs.
    let d_hs = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_x_mid, w.w_o, &d_hs, seq, h, h, false)?;
    let dw_o = gpu.zeros(&[h * h], DType::F32)?;
    linear_backward_w(gpu, &d_x_mid, &acts.hs, &dw_o, seq, h, h, false)?;
    gpu.free_tensor(d_x_mid)?;

    // gated_scan bwd: d_hs → d_g, d_u.
    let (d_g, d_u) = gated_scan_backward(gpu, &acts.g, &acts.u, &acts.hs, &d_hs, seq, h)?;
    gpu.free_tensor(d_hs)?;

    // sigmoid bwd: d_g → d_gpre.
    let d_gpre = sigmoid_backward(gpu, &d_g, &acts.g, seq * h)?;
    gpu.free_tensor(d_g)?;

    // w_u, w_g grads + d_xn1 (two contributions, accumulate).
    let dw_u = gpu.zeros(&[h * h], DType::F32)?;
    linear_backward_w(gpu, &d_u, &acts.xn1, &dw_u, seq, h, h, false)?;
    let dw_g = gpu.zeros(&[h * h], DType::F32)?;
    linear_backward_w(gpu, &d_gpre, &acts.xn1, &dw_g, seq, h, h, false)?;
    let d_xn1 = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_u, w.w_u, &d_xn1, seq, h, h, false)?;
    linear_backward_x(gpu, &d_gpre, w.w_g, &d_xn1, seq, h, h, true)?;
    gpu.free_tensor(d_u)?;
    gpu.free_tensor(d_gpre)?;

    // rmsnorm1 bwd → d_xnorm (+dnorm1); add into d_x.
    let d_xnorm = gpu.zeros(&[seq * h], DType::F32)?;
    let dnorm1 = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(gpu, &d_xn1, x_in, w.norm1, &acts.rinv1, &d_xnorm, &dnorm1, seq, h)?;
    gpu.free_tensor(d_xn1)?;
    gpu.add_inplace_f32(&d_x, &d_xnorm)?;
    gpu.free_tensor(d_xnorm)?;

    let grad = SsmBlockGrad { dnorm1, dw_u, dw_g, dw_o, dnorm2, dwgate, dwup, dwdown };
    Ok((d_x, grad))
}

/// Return a forward's saved activations to the pool (GpuTensor has no Drop).
pub fn free_ssm_block_acts(gpu: &mut Gpu, a: SsmBlockActivations) -> HipResult<()> {
    let SsmBlockActivations { xn1, rinv1, u, g, hs, x_mid, xn2, rinv2, gate, up, act } = a;
    for t in [xn1, rinv1, u, g, hs, x_mid, xn2, rinv2, gate, up, act] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Return a backward's weight grads to the pool after the optimizer step.
pub fn free_ssm_block_grad(gpu: &mut Gpu, g: SsmBlockGrad) -> HipResult<()> {
    let SsmBlockGrad { dnorm1, dw_u, dw_g, dw_o, dnorm2, dwgate, dwup, dwdown } = g;
    for t in [dnorm1, dw_u, dw_g, dw_o, dnorm2, dwgate, dwup, dwdown] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}
