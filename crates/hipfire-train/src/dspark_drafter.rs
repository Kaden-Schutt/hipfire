// SPDX-License-Identifier: Apache-2.0
//! DSpark drafter BODY — un-fused fp32 forward + matching backward (T2).
//!
//! The drafter is a 5-layer dense-GQA transformer whose `block` query positions
//! attend **bidirectionally** over `[context_KV ++ block_KV]`. It is the native
//! trainer fork of the inference arch in
//! `crates/hipfire-arch-llama/src/dspark_body.rs` (`dspark_qwen3_block_forward`)
//! and the DeepSpec reference `third_party/dspark/deepspec/modeling/dspark/qwen3`
//! (`Qwen3DSparkModel._forward_backbone`, `Qwen3DSparkAttention.forward`).
//!
//! This module owns only the transformer body + context ingest + their
//! backward. The markov / confidence heads, the lm-head, and the loss are
//! SEPARATE tasks; the forward stops at `x_head = out_norm(last_block_out)`
//! (pre-lm-head hidden states) and the backward starts from `d_x_head`.
//!
//! Fork of `block.rs` (`block_forward`/`block_backward_full`). The only
//! structural differences from a causal LLaMA block are:
//!   (a) **QK-norm** — a per-head rmsnorm on q and on the concatenated k BEFORE
//!       RoPE (reuses `rmsnorm_forward/backward` with `rows = tokens*heads`,
//!       `h = head_dim`).
//!   (b) **masked bidirectional attention over `[ctx_KV ++ block_KV]`** via
//!       `gqa_forward_masked`/`gqa_backward_masked` (`seq_q = block`,
//!       `seq_k = ctx_len + block`, additive `bias`). Context K/V for each layer
//!       are `k_proj(main_x)` / `v_proj(main_x)` where `main_x` is shared across
//!       all layers; block K/V are `k_proj(xn1)` / `v_proj(xn1)`.
//!
//! Per-layer op sequence (reference modeling.py:99–116, 181–198):
//! ```text
//! xn1  = input_layernorm(x_block)                       [rmsnorm]
//! qp   = q_proj(xn1)                                    [linear]
//! qn   = q_norm(qp, per-head)                           [rmsnorm, BEFORE rope]
//! q_r  = rope(qn, block_positions)
//! k_ctx= k_proj(main_x); k_blk = k_proj(xn1)
//! kcat = [k_ctx ++ k_blk]
//! kn   = k_norm(kcat, per-head)                         [rmsnorm, BEFORE rope]
//! k_r  = rope(kn, [ctx_positions ++ block_positions])
//! v    = [v_proj(main_x) ++ v_proj(xn1)]
//! ctx  = gqa_masked(q_r, k_r, v, bias)                  [bidirectional]
//! attn = o_proj(ctx); x_mid = x_block + attn
//! xn2  = post_attention_layernorm(x_mid)
//! mlp  = down(swiglu(gate(xn2), up(xn2))); x_out = x_mid + mlp
//! ```
//! Context ingest: `main_x = hidden_norm(fc(main_hidden))`, where `main_hidden`
//! is the concat of target hidden states at `target_layer_ids`
//! (`[ctx_len, n_targets*h] -fc-> [ctx_len, h] -hidden_norm-> [ctx_len, h]`).

use crate::ops::attention::{gqa_backward_masked, gqa_forward_masked};
use crate::ops::linear::{linear_backward_w, linear_backward_x, linear_forward};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use crate::ops::rope::{rope_backward, rope_forward};
use crate::ops::sigmoid::{sigmoid_backward, sigmoid_forward};
use crate::ops::swiglu::{swiglu_backward, swiglu_forward};
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

// ── Config + dims ─────────────────────────────────────────────────────────────

/// Drafter architecture hyperparameters.
#[derive(Clone, Copy)]
pub struct DsparkDrafterConfig {
    pub h: usize,        // hidden dim
    pub n_layers: usize, // 5 for the qwen3 drafter
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub inter: usize, // MLP intermediate dim
    pub rope_base: f32,
    pub eps: f32,
    pub block_size: usize, // block query positions per anchor
    pub n_targets: usize,  // len(target_layer_ids) — fc input = n_targets*h
    pub qk_norm: bool,     // true for the qwen3 drafter (per-head q/k rmsnorm)
    pub vocab: usize,      // vocab (embed / lm_head are target-shared, external)
}

impl DsparkDrafterConfig {
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
    pub fn dims(&self) -> DsparkDims {
        DsparkDims {
            h: self.h,
            n_heads: self.n_heads,
            n_kv: self.n_kv,
            head_dim: self.head_dim,
            inter: self.inter,
            rope_base: self.rope_base,
            eps: self.eps,
            qk_norm: self.qk_norm,
        }
    }
}

/// Shape/size params threaded into the block fwd/bwd (block & ctx lengths are
/// derived from the tensors so one dims value serves any window size).
#[derive(Clone, Copy)]
pub struct DsparkDims {
    pub h: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub qk_norm: bool,
}

impl DsparkDims {
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

fn clone_tensor(gpu: &mut Gpu, t: &GpuTensor) -> HipResult<GpuTensor> {
    let n: usize = t.shape.iter().product();
    let c = gpu.zeros(&t.shape, t.dtype)?;
    gpu.memcpy_dtod_auto(&c.buf, &t.buf, n * 4)?;
    Ok(c)
}

// ── Weights ───────────────────────────────────────────────────────────────────

/// One drafter layer's owned fp32 weights (HF row-major `[out, in]`).
pub struct DsparkLayerWeights {
    pub input_ln: GpuTensor, // [h]
    pub wq: GpuTensor,       // [q_dim, h]
    pub wk: GpuTensor,       // [kv_dim, h]
    pub wv: GpuTensor,       // [kv_dim, h]
    pub wo: GpuTensor,       // [h, q_dim]
    pub q_norm: GpuTensor,   // [head_dim]
    pub k_norm: GpuTensor,   // [head_dim]
    pub post_ln: GpuTensor,  // [h]
    pub wgate: GpuTensor,    // [inter, h]
    pub wup: GpuTensor,      // [inter, h]
    pub wdown: GpuTensor,    // [h, inter]
}

/// Full drafter body weights (globals + per-layer). Embedding + lm-head are
/// target-shared and live OUTSIDE this struct.
pub struct DsparkDrafterWeights {
    pub fc: GpuTensor,          // main_proj [h, n_targets*h]
    pub hidden_norm: GpuTensor, // main_norm [h]
    pub layers: Vec<DsparkLayerWeights>,
    pub out_norm: GpuTensor, // final norm [h]
}

impl DsparkDrafterWeights {
    /// Trainable params in a fixed order (matches `DsparkDrafterGrads::flat`):
    /// `fc, hidden_norm`, then per layer
    /// `[wq, wk, wv, wo, wgate, wup, wdown, input_ln, post_ln, q_norm, k_norm]`,
    /// then `out_norm`. Order mirrors `optim.rs`'s AdamW `params/step` contract.
    pub fn params(&self) -> Vec<&GpuTensor> {
        let mut v = vec![&self.fc, &self.hidden_norm];
        for l in &self.layers {
            v.push(&l.wq);
            v.push(&l.wk);
            v.push(&l.wv);
            v.push(&l.wo);
            v.push(&l.wgate);
            v.push(&l.wup);
            v.push(&l.wdown);
            v.push(&l.input_ln);
            v.push(&l.post_ln);
            v.push(&l.q_norm);
            v.push(&l.k_norm);
        }
        v.push(&self.out_norm);
        v
    }

    pub fn param_sizes(&self) -> Vec<usize> {
        self.params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }
}

/// Borrowed view of one layer's weights for the block fwd/bwd.
pub struct DsparkBlockWeights<'a> {
    pub input_ln: &'a GpuTensor,
    pub wq: &'a GpuTensor,
    pub wk: &'a GpuTensor,
    pub wv: &'a GpuTensor,
    pub wo: &'a GpuTensor,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub post_ln: &'a GpuTensor,
    pub wgate: &'a GpuTensor,
    pub wup: &'a GpuTensor,
    pub wdown: &'a GpuTensor,
}

impl DsparkLayerWeights {
    pub fn view(&self) -> DsparkBlockWeights<'_> {
        DsparkBlockWeights {
            input_ln: &self.input_ln,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            post_ln: &self.post_ln,
            wgate: &self.wgate,
            wup: &self.wup,
            wdown: &self.wdown,
        }
    }
}

// ── Gradients ─────────────────────────────────────────────────────────────────

/// One layer's weight grads (same field set as `DsparkLayerWeights`).
pub struct DsparkBlockWeightGrad {
    pub dwq: GpuTensor,
    pub dwk: GpuTensor,
    pub dwv: GpuTensor,
    pub dwo: GpuTensor,
    pub dwgate: GpuTensor,
    pub dwup: GpuTensor,
    pub dwdown: GpuTensor,
    pub dinput_ln: GpuTensor,
    pub dpost_ln: GpuTensor,
    pub dq_norm: GpuTensor,
    pub dk_norm: GpuTensor,
}

/// Ingest (fc + hidden_norm) grads.
pub struct DsparkIngestGrad {
    pub d_fc: GpuTensor,          // [h, n_targets*h]
    pub d_hidden_norm: GpuTensor, // [h]
}

/// Full drafter body grads. `flat()` returns them in `params()` order for the
/// AdamW step; `d_main_hidden` is the grad w.r.t. the (frozen, external) target
/// hidden input — NOT a param, exposed only for gradchecking the ingest input.
pub struct DsparkDrafterGrads {
    pub ingest: DsparkIngestGrad,
    pub layers: Vec<DsparkBlockWeightGrad>,
    pub d_out_norm: GpuTensor,
    pub d_main_hidden: GpuTensor, // [ctx_len * n_targets * h] — not in flat()
}

impl DsparkDrafterGrads {
    /// Flatten in the SAME fixed order as `DsparkDrafterWeights::params()`.
    pub fn flat(&self) -> Vec<&GpuTensor> {
        let mut v = vec![&self.ingest.d_fc, &self.ingest.d_hidden_norm];
        for g in &self.layers {
            v.push(&g.dwq);
            v.push(&g.dwk);
            v.push(&g.dwv);
            v.push(&g.dwo);
            v.push(&g.dwgate);
            v.push(&g.dwup);
            v.push(&g.dwdown);
            v.push(&g.dinput_ln);
            v.push(&g.dpost_ln);
            v.push(&g.dq_norm);
            v.push(&g.dk_norm);
        }
        v.push(&self.d_out_norm);
        v
    }
}

// ── Saved activations ─────────────────────────────────────────────────────────

/// Backward-needed activations for one drafter block.
///
/// Saved: the two block-norm outputs + their `rinv` (`xn1`,`rinv1`,`xn2`,
/// `rinv2`); the q-projection output `qp` and per-head `q_rinv` (q_norm bwd
/// inputs); the concatenated pre-norm K `kcat` and per-head `k_rinv` (k_norm bwd
/// inputs); the post-RoPE `q_rope`/`k_rope` and `all_v` + `p_all` softmax (gqa
/// bwd inputs); the attention context `ctx_attn` (wo weight grad); the residual
/// point `x_mid` (post_ln bwd input); the MLP `gate`/`up`/`act`; and the RoPE
/// position tensors `q_pos`/`k_pos` (rope bwd). `ctx_len` is stashed to split
/// the ctx/block halves of the K/V grads.
pub struct DsparkBlockActivations {
    pub ctx_len: usize,
    pub xn1: GpuTensor,
    pub rinv1: GpuTensor,
    pub qp: GpuTensor,
    pub q_rinv: GpuTensor,
    pub q_rope: GpuTensor,
    pub kcat: GpuTensor,
    pub k_rinv: GpuTensor,
    pub k_rope: GpuTensor,
    pub all_v: GpuTensor,
    pub p_all: GpuTensor,
    pub ctx_attn: GpuTensor,
    pub x_mid: GpuTensor,
    pub xn2: GpuTensor,
    pub rinv2: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub act: GpuTensor,
    pub q_pos: GpuTensor,
    pub k_pos: GpuTensor,
}

/// Return one block's saved activations to the pool (GpuTensor has no Drop).
pub fn free_dspark_block_acts(gpu: &mut Gpu, a: DsparkBlockActivations) -> HipResult<()> {
    let DsparkBlockActivations {
        ctx_len: _,
        xn1,
        rinv1,
        qp,
        q_rinv,
        q_rope,
        kcat,
        k_rinv,
        k_rope,
        all_v,
        p_all,
        ctx_attn,
        x_mid,
        xn2,
        rinv2,
        gate,
        up,
        act,
        q_pos,
        k_pos,
    } = a;
    for t in [
        xn1, rinv1, qp, q_rinv, q_rope, kcat, k_rinv, k_rope, all_v, p_all, ctx_attn, x_mid, xn2,
        rinv2, gate, up, act, q_pos, k_pos,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

// ── Block forward ─────────────────────────────────────────────────────────────

/// One drafter block. `x_block` `[block*h]`; `ctx` (`main_x`) `[ctx_len*h]`
/// (shared context, same tensor every layer). `q_pos_host` `[block]`,
/// `k_pos_host` `[ctx_len+block]` are the RoPE positions (Q at block positions;
/// K at `[ctx_positions ++ block_positions]`). `bias` (`Some([block*(ctx_len+
/// block)])`) is the additive attention mask (bidirectional/valid); `None` =
/// fully bidirectional over all `[ctx ++ block]` keys.
///
/// Returns `x_out` `[block*h]` and the saved activations.
#[allow(clippy::too_many_arguments)]
pub fn dspark_block_forward(
    gpu: &mut Gpu,
    x_block: &GpuTensor,
    ctx: &GpuTensor,
    w: &DsparkBlockWeights,
    dims: &DsparkDims,
    q_pos_host: &[f32],
    k_pos_host: &[f32],
    bias: Option<&GpuTensor>,
) -> HipResult<(GpuTensor, DsparkBlockActivations)> {
    let (h, inter) = (dims.h, dims.inter);
    let (qd, kvd) = (dims.q_dim(), dims.kv_dim());
    let (nh, nkv, hd) = (dims.n_heads, dims.n_kv, dims.head_dim);
    let block = x_block.shape.iter().product::<usize>() / h;
    let ctx_len = ctx.shape.iter().product::<usize>() / h;
    let kv_rows = ctx_len + block;
    debug_assert_eq!(q_pos_host.len(), block);
    debug_assert_eq!(k_pos_host.len(), kv_rows);

    // 1. xn1 = input_layernorm(x_block)
    let xn1 = gpu.zeros(&[block * h], DType::F32)?;
    let rinv1 = gpu.zeros(&[block], DType::F32)?;
    rmsnorm_forward(gpu, x_block, w.input_ln, &xn1, &rinv1, block, h, dims.eps)?;

    // 2. qp = q_proj(xn1)
    let qp = gpu.zeros(&[block * qd], DType::F32)?;
    linear_forward(gpu, &xn1, w.wq, &qp, block, h, qd)?;

    // 3. qn = q_norm(qp) per head (rows = block*n_heads, h = head_dim), BEFORE rope
    let qn = gpu.zeros(&[block * qd], DType::F32)?;
    let q_rinv = gpu.zeros(&[block * nh], DType::F32)?;
    if dims.qk_norm {
        rmsnorm_forward(gpu, &qp, w.q_norm, &qn, &q_rinv, block * nh, hd, dims.eps)?;
    } else {
        gpu.memcpy_dtod_auto(&qn.buf, &qp.buf, block * qd * 4)?;
    }

    // 4. q_rope = rope(qn, block_positions)
    let q_pos = gpu.upload_f32(q_pos_host, &[block])?;
    let q_rope = gpu.zeros(&[block * qd], DType::F32)?;
    rope_forward(
        gpu,
        &qn,
        &q_rope,
        &q_pos,
        block * nh,
        nh,
        hd,
        dims.rope_base,
    )?;

    // 5. kcat = [k_proj(ctx) ++ k_proj(xn1)]  (ctx rows 0..ctx_len, block rows after)
    let kcat = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    {
        let k_ctx = kcat.sub_offset(0, ctx_len * kvd);
        linear_forward(gpu, ctx, w.wk, &k_ctx, ctx_len, h, kvd)?;
        let k_blk = kcat.sub_offset(ctx_len * kvd, block * kvd);
        linear_forward(gpu, &xn1, w.wk, &k_blk, block, h, kvd)?;
    }

    // 6. kn = k_norm(kcat) per head (rows = kv_rows*n_kv, h = head_dim), BEFORE rope
    let kn = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    let k_rinv = gpu.zeros(&[kv_rows * nkv], DType::F32)?;
    if dims.qk_norm {
        rmsnorm_forward(
            gpu,
            &kcat,
            w.k_norm,
            &kn,
            &k_rinv,
            kv_rows * nkv,
            hd,
            dims.eps,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&kn.buf, &kcat.buf, kv_rows * kvd * 4)?;
    }

    // 7. k_rope = rope(kn, [ctx_positions ++ block_positions])
    let k_pos = gpu.upload_f32(k_pos_host, &[kv_rows])?;
    let k_rope = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    rope_forward(
        gpu,
        &kn,
        &k_rope,
        &k_pos,
        kv_rows * nkv,
        nkv,
        hd,
        dims.rope_base,
    )?;

    // 8. all_v = [v_proj(ctx) ++ v_proj(xn1)]  (no norm, no rope)
    let all_v = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    {
        let v_ctx = all_v.sub_offset(0, ctx_len * kvd);
        linear_forward(gpu, ctx, w.wv, &v_ctx, ctx_len, h, kvd)?;
        let v_blk = all_v.sub_offset(ctx_len * kvd, block * kvd);
        linear_forward(gpu, &xn1, w.wv, &v_blk, block, h, kvd)?;
    }

    // 9. bidirectional masked GQA over [ctx ++ block]
    let p_all = gpu.zeros(&[nh * block * kv_rows], DType::F32)?;
    let ctx_attn = gpu.zeros(&[block * qd], DType::F32)?;
    gqa_forward_masked(
        gpu,
        &q_rope,
        &k_rope,
        &all_v,
        &p_all,
        &ctx_attn,
        block,
        kv_rows,
        nh,
        nkv,
        hd,
        dims.attn_scale(),
        bias,
    )?;

    // 10. attn = o_proj(ctx_attn); x_mid = x_block + attn
    let attn = gpu.zeros(&[block * h], DType::F32)?;
    linear_forward(gpu, &ctx_attn, w.wo, &attn, block, qd, h)?;
    let x_mid = gpu.zeros(&[block * h], DType::F32)?;
    gpu.add_f32(x_block, &attn, &x_mid)?;

    // 11. xn2 = post_attention_layernorm(x_mid); MLP; residual
    let xn2 = gpu.zeros(&[block * h], DType::F32)?;
    let rinv2 = gpu.zeros(&[block], DType::F32)?;
    rmsnorm_forward(gpu, &x_mid, w.post_ln, &xn2, &rinv2, block, h, dims.eps)?;
    let gate = gpu.zeros(&[block * inter], DType::F32)?;
    linear_forward(gpu, &xn2, w.wgate, &gate, block, h, inter)?;
    let up = gpu.zeros(&[block * inter], DType::F32)?;
    linear_forward(gpu, &xn2, w.wup, &up, block, h, inter)?;
    let act = gpu.zeros(&[block * inter], DType::F32)?;
    swiglu_forward(gpu, &gate, &up, &act, block * inter)?;
    let mlp = gpu.zeros(&[block * h], DType::F32)?;
    linear_forward(gpu, &act, w.wdown, &mlp, block, inter, h)?;
    let x_out = gpu.zeros(&[block * h], DType::F32)?;
    gpu.add_f32(&x_mid, &mlp, &x_out)?;

    // Transients the backward never reads → back to the pool (no Drop).
    for t in [qn, kn, attn, mlp] {
        gpu.free_tensor(t)?;
    }

    Ok((
        x_out,
        DsparkBlockActivations {
            ctx_len,
            xn1,
            rinv1,
            qp,
            q_rinv,
            q_rope,
            kcat,
            k_rinv,
            k_rope,
            all_v,
            p_all,
            ctx_attn,
            x_mid,
            xn2,
            rinv2,
            gate,
            up,
            act,
            q_pos,
            k_pos,
        },
    ))
}

// ── Block backward ────────────────────────────────────────────────────────────

/// One drafter block backward. `d_x_out` `[block*h]` upstream. Returns
/// `(d_x_block [block*h], d_ctx [ctx_len*h], weight grads)`. `d_ctx` is this
/// layer's contribution to the shared context grad (the caller accumulates it
/// across layers). `x_block`/`ctx` are the same tensors passed to the forward.
#[allow(clippy::too_many_arguments)]
pub fn dspark_block_backward(
    gpu: &mut Gpu,
    d_x_out: &GpuTensor,
    x_block: &GpuTensor,
    ctx: &GpuTensor,
    w: &DsparkBlockWeights,
    acts: &DsparkBlockActivations,
    dims: &DsparkDims,
) -> HipResult<(GpuTensor, GpuTensor, DsparkBlockWeightGrad)> {
    let (h, inter) = (dims.h, dims.inter);
    let (qd, kvd) = (dims.q_dim(), dims.kv_dim());
    let (nh, nkv, hd) = (dims.n_heads, dims.n_kv, dims.head_dim);
    let ctx_len = acts.ctx_len;
    let block = x_block.shape.iter().product::<usize>() / h;
    let kv_rows = ctx_len + block;

    // Trainable norm grads (rmsnorm_backward atomic-accumulates → zero first).
    let dinput_ln = gpu.zeros(&[h], DType::F32)?;
    let dpost_ln = gpu.zeros(&[h], DType::F32)?;
    let dq_norm = gpu.zeros(&[hd], DType::F32)?;
    let dk_norm = gpu.zeros(&[hd], DType::F32)?;

    // ── MLP branch: x_out = x_mid + mlp ⇒ d_mlp = d_x_out, d_x_mid starts = d_x_out.
    let d_act = gpu.zeros(&[block * inter], DType::F32)?;
    linear_backward_x(gpu, d_x_out, w.wdown, &d_act, block, inter, h, false)?;
    let d_gate = gpu.zeros(&[block * inter], DType::F32)?;
    let d_up = gpu.zeros(&[block * inter], DType::F32)?;
    swiglu_backward(
        gpu,
        &d_act,
        &acts.gate,
        &acts.up,
        &d_gate,
        &d_up,
        block * inter,
    )?;
    let d_xn2 = gpu.zeros(&[block * h], DType::F32)?;
    linear_backward_x(gpu, &d_gate, w.wgate, &d_xn2, block, h, inter, false)?;
    linear_backward_x(gpu, &d_up, w.wup, &d_xn2, block, h, inter, true)?;
    let d_x_mid = gpu.zeros(&[block * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x_mid.buf, &d_x_out.buf, block * h * 4)?; // residual
    let d_xmid_norm = gpu.zeros(&[block * h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xn2,
        &acts.x_mid,
        w.post_ln,
        &acts.rinv2,
        &d_xmid_norm,
        &dpost_ln,
        block,
        h,
    )?;
    gpu.add_inplace_f32(&d_x_mid, &d_xmid_norm)?;

    // ── Attention branch: x_mid = x_block + attn ⇒ d_attn = d_x_mid.
    let d_ctx_attn = gpu.zeros(&[block * qd], DType::F32)?;
    linear_backward_x(gpu, &d_x_mid, w.wo, &d_ctx_attn, block, qd, h, false)?;
    let d_q_rope = gpu.zeros(&[block * qd], DType::F32)?;
    let d_k_rope = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    let d_all_v = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    gqa_backward_masked(
        gpu,
        &d_ctx_attn,
        &acts.q_rope,
        &acts.k_rope,
        &acts.all_v,
        &acts.p_all,
        &d_q_rope,
        &d_k_rope,
        &d_all_v,
        block,
        kv_rows,
        nh,
        nkv,
        hd,
        dims.attn_scale(),
    )?;

    // d_xn1 accumulates q/k-block/v-block contributions; d_ctx accumulates
    // k-ctx/v-ctx contributions.
    let d_xn1 = gpu.zeros(&[block * h], DType::F32)?;
    let d_ctx = gpu.zeros(&[ctx_len * h], DType::F32)?;

    // ── Q path: rope⁻¹ → q_norm⁻¹ → q_proj⁻¹
    let d_qn = gpu.zeros(&[block * qd], DType::F32)?;
    rope_backward(
        gpu,
        &d_q_rope,
        &d_qn,
        &acts.q_pos,
        block * nh,
        nh,
        hd,
        dims.rope_base,
    )?;
    let d_qp = gpu.zeros(&[block * qd], DType::F32)?;
    if dims.qk_norm {
        rmsnorm_backward(
            gpu,
            &d_qn,
            &acts.qp,
            w.q_norm,
            &acts.q_rinv,
            &d_qp,
            &dq_norm,
            block * nh,
            hd,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&d_qp.buf, &d_qn.buf, block * qd * 4)?;
    }
    let dwq = gpu.zeros(&[qd * h], DType::F32)?;
    linear_backward_w(gpu, &d_qp, &acts.xn1, &dwq, block, h, qd, false)?;
    linear_backward_x(gpu, &d_qp, w.wq, &d_xn1, block, h, qd, false)?; // first writer → overwrite

    // ── K path: rope⁻¹ → k_norm⁻¹ → split ctx/block → k_proj⁻¹
    let d_kn = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    rope_backward(
        gpu,
        &d_k_rope,
        &d_kn,
        &acts.k_pos,
        kv_rows * nkv,
        nkv,
        hd,
        dims.rope_base,
    )?;
    let d_kcat = gpu.zeros(&[kv_rows * kvd], DType::F32)?;
    if dims.qk_norm {
        rmsnorm_backward(
            gpu,
            &d_kn,
            &acts.kcat,
            w.k_norm,
            &acts.k_rinv,
            &d_kcat,
            &dk_norm,
            kv_rows * nkv,
            hd,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&d_kcat.buf, &d_kn.buf, kv_rows * kvd * 4)?;
    }
    let d_k_ctx = d_kcat.sub_offset(0, ctx_len * kvd);
    let d_k_blk = d_kcat.sub_offset(ctx_len * kvd, block * kvd);
    let dwk = gpu.zeros(&[kvd * h], DType::F32)?;
    linear_backward_w(gpu, &d_k_ctx, ctx, &dwk, ctx_len, h, kvd, false)?;
    linear_backward_w(gpu, &d_k_blk, &acts.xn1, &dwk, block, h, kvd, true)?;
    linear_backward_x(gpu, &d_k_ctx, w.wk, &d_ctx, ctx_len, h, kvd, false)?; // first writer → overwrite
    linear_backward_x(gpu, &d_k_blk, w.wk, &d_xn1, block, h, kvd, true)?;

    // ── V path: split ctx/block → v_proj⁻¹
    let d_v_ctx = d_all_v.sub_offset(0, ctx_len * kvd);
    let d_v_blk = d_all_v.sub_offset(ctx_len * kvd, block * kvd);
    let dwv = gpu.zeros(&[kvd * h], DType::F32)?;
    linear_backward_w(gpu, &d_v_ctx, ctx, &dwv, ctx_len, h, kvd, false)?;
    linear_backward_w(gpu, &d_v_blk, &acts.xn1, &dwv, block, h, kvd, true)?;
    linear_backward_x(gpu, &d_v_ctx, w.wv, &d_ctx, ctx_len, h, kvd, true)?;
    linear_backward_x(gpu, &d_v_blk, w.wv, &d_xn1, block, h, kvd, true)?;

    // ── input_ln backward: x_mid = x_block + attn ⇒ d_x_block residual = d_x_mid.
    let d_x_block = gpu.zeros(&[block * h], DType::F32)?;
    gpu.memcpy_dtod_auto(&d_x_block.buf, &d_x_mid.buf, block * h * 4)?; // residual
    let d_x_norm = gpu.zeros(&[block * h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xn1,
        x_block,
        w.input_ln,
        &acts.rinv1,
        &d_x_norm,
        &dinput_ln,
        block,
        h,
    )?;
    gpu.add_inplace_f32(&d_x_block, &d_x_norm)?;

    // ── weight grads for o/gate/up/down (single-input linears)
    let dwo = gpu.zeros(&[h * qd], DType::F32)?;
    linear_backward_w(gpu, &d_x_mid, &acts.ctx_attn, &dwo, block, qd, h, false)?;
    let dwgate = gpu.zeros(&[inter * h], DType::F32)?;
    linear_backward_w(gpu, &d_gate, &acts.xn2, &dwgate, block, h, inter, false)?;
    let dwup = gpu.zeros(&[inter * h], DType::F32)?;
    linear_backward_w(gpu, &d_up, &acts.xn2, &dwup, block, h, inter, false)?;
    let dwdown = gpu.zeros(&[h * inter], DType::F32)?;
    linear_backward_w(gpu, d_x_out, &acts.act, &dwdown, block, inter, h, false)?;

    // Return internal temporaries to the pool (no Drop). Only the returned grads
    // (d_x_block, d_ctx, DsparkBlockWeightGrad) survive.
    for t in [
        d_act,
        d_gate,
        d_up,
        d_xn2,
        d_x_mid,
        d_xmid_norm,
        d_ctx_attn,
        d_q_rope,
        d_k_rope,
        d_all_v,
        d_qn,
        d_qp,
        d_kn,
        d_kcat,
        d_xn1,
        d_x_norm,
    ] {
        gpu.free_tensor(t)?;
    }

    Ok((
        d_x_block,
        d_ctx,
        DsparkBlockWeightGrad {
            dwq,
            dwk,
            dwv,
            dwo,
            dwgate,
            dwup,
            dwdown,
            dinput_ln,
            dpost_ln,
            dq_norm,
            dk_norm,
        },
    ))
}

// ── Context ingest ────────────────────────────────────────────────────────────

/// Ingest forward: `main_x = hidden_norm(fc(main_hidden))`.
/// `main_hidden` `[ctx_len*(n_targets*h)]` → `fc_out` `[ctx_len*h]` → `main_x`.
/// Returns `(main_x, fc_out, rinv_hn)` — `fc_out`/`rinv_hn` are the ingest bwd
/// inputs (`hidden_norm` backward reads `fc_out` + `rinv_hn`).
pub fn dspark_ingest_forward(
    gpu: &mut Gpu,
    fc: &GpuTensor,
    hidden_norm: &GpuTensor,
    main_hidden: &GpuTensor,
    dims: &DsparkDims,
    n_targets: usize,
) -> HipResult<(GpuTensor, GpuTensor, GpuTensor)> {
    let h = dims.h;
    let fin = n_targets * h;
    let ctx_len = main_hidden.shape.iter().product::<usize>() / fin;
    let fc_out = gpu.zeros(&[ctx_len * h], DType::F32)?;
    linear_forward(gpu, main_hidden, fc, &fc_out, ctx_len, fin, h)?;
    let main_x = gpu.zeros(&[ctx_len * h], DType::F32)?;
    let rinv_hn = gpu.zeros(&[ctx_len], DType::F32)?;
    rmsnorm_forward(
        gpu,
        &fc_out,
        hidden_norm,
        &main_x,
        &rinv_hn,
        ctx_len,
        h,
        dims.eps,
    )?;
    Ok((main_x, fc_out, rinv_hn))
}

/// Ingest backward from `d_main_x` `[ctx_len*h]`. Returns
/// `(d_main_hidden [ctx_len*(n_targets*h)], DsparkIngestGrad)`. `d_main_hidden`
/// is grad w.r.t. the (frozen) target hidden — exposed for gradchecking only.
#[allow(clippy::too_many_arguments)]
pub fn dspark_ingest_backward(
    gpu: &mut Gpu,
    d_main_x: &GpuTensor,
    fc: &GpuTensor,
    hidden_norm: &GpuTensor,
    main_hidden: &GpuTensor,
    fc_out: &GpuTensor,
    rinv_hn: &GpuTensor,
    dims: &DsparkDims,
    n_targets: usize,
) -> HipResult<(GpuTensor, DsparkIngestGrad)> {
    let h = dims.h;
    let fin = n_targets * h;
    let ctx_len = fc_out.shape.iter().product::<usize>() / h;

    // hidden_norm backward → d_fc_out, d_hidden_norm
    let d_fc_out = gpu.zeros(&[ctx_len * h], DType::F32)?;
    let d_hidden_norm = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        d_main_x,
        fc_out,
        hidden_norm,
        rinv_hn,
        &d_fc_out,
        &d_hidden_norm,
        ctx_len,
        h,
    )?;

    // fc backward → d_fc weight grad + d_main_hidden
    let d_fc = gpu.zeros(&[h * fin], DType::F32)?;
    linear_backward_w(gpu, &d_fc_out, main_hidden, &d_fc, ctx_len, fin, h, false)?;
    let d_main_hidden = gpu.zeros(&[ctx_len * fin], DType::F32)?;
    linear_backward_x(gpu, &d_fc_out, fc, &d_main_hidden, ctx_len, fin, h, false)?;

    gpu.free_tensor(d_fc_out)?;
    Ok((
        d_main_hidden,
        DsparkIngestGrad {
            d_fc,
            d_hidden_norm,
        },
    ))
}

// ── Full drafter body ─────────────────────────────────────────────────────────

/// Saved activations for the full drafter body training forward.
pub struct DsparkDrafterActs {
    pub fc_out: GpuTensor,            // ingest: fc output [ctx_len*h]
    pub rinv_hn: GpuTensor,           // ingest: hidden_norm rinv [ctx_len]
    pub main_x: GpuTensor,            // ingest: context [ctx_len*h]
    pub layer_inputs: Vec<GpuTensor>, // input to each block [block*h]
    pub layer_acts: Vec<DsparkBlockActivations>,
    pub x_last: GpuTensor,   // last block output [block*h]
    pub xn_out: GpuTensor,   // x_head = out_norm(x_last) [block*h]
    pub rinv_out: GpuTensor, // out_norm rinv [block]
}

impl DsparkDrafterActs {
    /// The pre-lm-head hidden states (`x_head`) produced by the forward.
    pub fn x_head(&self) -> &GpuTensor {
        &self.xn_out
    }
}

/// Return a forward's saved activations to the pool.
pub fn free_dspark_drafter_acts(gpu: &mut Gpu, a: DsparkDrafterActs) -> HipResult<()> {
    let DsparkDrafterActs {
        fc_out,
        rinv_hn,
        main_x,
        layer_inputs,
        layer_acts,
        x_last,
        xn_out,
        rinv_out,
    } = a;
    for t in layer_inputs {
        gpu.free_tensor(t)?;
    }
    for b in layer_acts {
        free_dspark_block_acts(gpu, b)?;
    }
    for t in [fc_out, rinv_hn, main_x, x_last, xn_out, rinv_out] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Return a backward's grads to the pool after the optimizer step.
pub fn free_dspark_drafter_grads(gpu: &mut Gpu, g: DsparkDrafterGrads) -> HipResult<()> {
    let DsparkDrafterGrads {
        ingest,
        layers,
        d_out_norm,
        d_main_hidden,
    } = g;
    for t in [ingest.d_fc, ingest.d_hidden_norm, d_out_norm, d_main_hidden] {
        gpu.free_tensor(t)?;
    }
    for lg in layers {
        let DsparkBlockWeightGrad {
            dwq,
            dwk,
            dwv,
            dwo,
            dwgate,
            dwup,
            dwdown,
            dinput_ln,
            dpost_ln,
            dq_norm,
            dk_norm,
        } = lg;
        for t in [
            dwq, dwk, dwv, dwo, dwgate, dwup, dwdown, dinput_ln, dpost_ln, dq_norm, dk_norm,
        ] {
            gpu.free_tensor(t)?;
        }
    }
    Ok(())
}

/// Full drafter body training forward: ingest → N blocks → out_norm.
///
/// * `main_hidden`       — `[ctx_len*(n_targets*h)]` concat of target hidden
///                         states at `target_layer_ids`.
/// * `block_embeds`      — `[block*h]` noise/seed block token embeddings
///                         (looked up from the target-shared embedding by the
///                         caller; embed is external to this module).
/// * `ctx_positions`     — `[ctx_len]` RoPE positions for the context rows.
/// * `block_positions`   — `[block]` RoPE positions for the block query rows.
/// * `bias`              — optional additive attention mask `[block*(ctx_len+
///                         block)]` (bidirectional/valid); `None` = all keys.
///
/// Produces `x_head = out_norm(last_block_out)` `[block*h]` (via `acts.x_head()`).
#[allow(clippy::too_many_arguments)]
pub fn dspark_drafter_forward_train(
    gpu: &mut Gpu,
    weights: &DsparkDrafterWeights,
    cfg: &DsparkDrafterConfig,
    main_hidden: &GpuTensor,
    block_embeds: &GpuTensor,
    ctx_positions: &[f32],
    block_positions: &[f32],
    bias: Option<&GpuTensor>,
) -> HipResult<DsparkDrafterActs> {
    let dims = cfg.dims();
    let h = cfg.h;

    // ingest
    let (main_x, fc_out, rinv_hn) = dspark_ingest_forward(
        gpu,
        &weights.fc,
        &weights.hidden_norm,
        main_hidden,
        &dims,
        cfg.n_targets,
    )?;

    // RoPE positions: q at block positions, k at [ctx ++ block].
    let q_pos = block_positions.to_vec();
    let mut k_pos = ctx_positions.to_vec();
    k_pos.extend_from_slice(block_positions);

    // blocks
    let mut layer_inputs = Vec::with_capacity(cfg.n_layers);
    let mut layer_acts = Vec::with_capacity(cfg.n_layers);
    let mut x = clone_tensor(gpu, block_embeds)?;
    for l in &weights.layers {
        layer_inputs.push(clone_tensor(gpu, &x)?);
        let bw = l.view();
        let (x_out, a) = dspark_block_forward(gpu, &x, &main_x, &bw, &dims, &q_pos, &k_pos, bias)?;
        gpu.free_tensor(x)?;
        layer_acts.push(a);
        x = x_out;
    }
    let x_last = x;

    // out_norm → x_head
    let xn_out = gpu.zeros(&[block_positions.len() * h], DType::F32)?;
    let rinv_out = gpu.zeros(&[block_positions.len()], DType::F32)?;
    rmsnorm_forward(
        gpu,
        &x_last,
        &weights.out_norm,
        &xn_out,
        &rinv_out,
        block_positions.len(),
        h,
        dims.eps,
    )?;

    Ok(DsparkDrafterActs {
        fc_out,
        rinv_hn,
        main_x,
        layer_inputs,
        layer_acts,
        x_last,
        xn_out,
        rinv_out,
    })
}

/// Full drafter body backward from `d_x_head` `[block*h]` (grad of the loss
/// w.r.t. the pre-lm-head hidden states). Produces all param grads (in
/// `params()` order via `flat()`) plus `d_main_hidden`.
pub fn dspark_drafter_backward(
    gpu: &mut Gpu,
    weights: &DsparkDrafterWeights,
    cfg: &DsparkDrafterConfig,
    main_hidden: &GpuTensor,
    acts: &DsparkDrafterActs,
    d_x_head: &GpuTensor,
) -> HipResult<DsparkDrafterGrads> {
    let dims = cfg.dims();
    let h = cfg.h;
    let block = d_x_head.shape.iter().product::<usize>() / h;
    let ctx_len = acts.main_x.shape.iter().product::<usize>() / h;

    // out_norm backward → d_x_last
    let d_x_last = gpu.zeros(&[block * h], DType::F32)?;
    let d_out_norm = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        d_x_head,
        &acts.x_last,
        &weights.out_norm,
        &acts.rinv_out,
        &d_x_last,
        &d_out_norm,
        block,
        h,
    )?;

    // blocks in reverse; accumulate the shared context grad.
    let d_main_x = gpu.zeros(&[ctx_len * h], DType::F32)?;
    let mut layer_grads: Vec<DsparkBlockWeightGrad> = Vec::with_capacity(cfg.n_layers);
    let mut d_x = d_x_last;
    for i in (0..cfg.n_layers).rev() {
        let bw = weights.layers[i].view();
        let (d_in, d_ctx, wg) = dspark_block_backward(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &acts.main_x,
            &bw,
            &acts.layer_acts[i],
            &dims,
        )?;
        gpu.free_tensor(d_x)?;
        gpu.add_inplace_f32(&d_main_x, &d_ctx)?;
        gpu.free_tensor(d_ctx)?;
        layer_grads.push(wg);
        d_x = d_in;
    }
    layer_grads.reverse();
    gpu.free_tensor(d_x)?; // grad w.r.t. block_embeds (external) — dropped

    // ingest backward
    let (d_main_hidden, ingest) = dspark_ingest_backward(
        gpu,
        &d_main_x,
        &weights.fc,
        &weights.hidden_norm,
        main_hidden,
        &acts.fc_out,
        &acts.rinv_hn,
        &dims,
        cfg.n_targets,
    )?;
    gpu.free_tensor(d_main_x)?;

    Ok(DsparkDrafterGrads {
        ingest,
        layers: layer_grads,
        d_out_norm,
        d_main_hidden,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// DSpark drafter HEADS (T2b) — lm-head + VanillaMarkov + AcceptRatePredictor.
// ═══════════════════════════════════════════════════════════════════════════
//
// Consumes the body's `x_head` `[block, h]` and produces:
//   * `draft_logits [block, vocab]` = `x_head @ lm_headᵀ + markov_bias`
//   * `confidence_pred [block]`     = `sigmoid(proj(concat(x_head, markov_latent)))`
//
// References:
//   * `third_party/dspark/deepspec/modeling/dspark/markov_head.py` (`VanillaMarkov`):
//       `markov_w1 = Embedding(vocab, rank)`, `markov_w2 = Linear(rank, vocab, bias=False)`.
//       `markov_bias = markov_w2(markov_w1[prev_token])`, added per block position with
//       that position's `prev_token`.
//   * `third_party/dspark/deepspec/modeling/dspark/common.py` (`AcceptRatePredictor`):
//       `proj = Linear(input_dim, 1)`; with `confidence_head_with_markov=True`,
//       `input_dim = h + markov_rank` and `features = concat(x_head, markov_latent)`;
//       `confidence_pred = sigmoid(logit)`. Inference side
//       `crates/hipfire-arch-llama/src/dspark_body.rs` confirms
//       `confidence_proj:[1, h+rank]`, `confidence_bias:[1]`.
//
// The lm-head weight is TARGET-SHARED / external (borrowed, frozen — no grad).
// `markov_w1` gather/scatter reuses the RoughQuant `rq_gather_f32` /
// `rq_scatter_add_f32` element movers with a flattened row index
// (`idx[b*rank + j] = prev[b]*rank + j`), so gather and scatter share one index
// and are exact inverses.

/// Head hyperparameters (kept separate from `DsparkDrafterConfig` so the body
/// forward/backward and their gradchecks are untouched). `markov_rank` is the
/// low-rank width of the VanillaMarkov head.
#[derive(Clone, Copy)]
pub struct DsparkHeadsConfig {
    pub h: usize,
    pub vocab: usize,
    pub markov_rank: usize,
}

impl DsparkHeadsConfig {
    /// Derive the head config from the body config plus the markov rank.
    pub fn from_drafter(cfg: &DsparkDrafterConfig, markov_rank: usize) -> Self {
        Self {
            h: cfg.h,
            vocab: cfg.vocab,
            markov_rank,
        }
    }
    /// Confidence proj input width = `h + markov_rank`.
    pub fn conf_in(&self) -> usize {
        self.h + self.markov_rank
    }
}

// ── Head weights / grads ────────────────────────────────────────────────────

/// Trainable head weights. The lm-head is NOT here (target-shared, borrowed).
pub struct DsparkHeadsWeights {
    pub markov_w1: GpuTensor,       // [vocab, rank]  — Embedding(vocab, rank)
    pub markov_w2: GpuTensor,       // [vocab, rank]  — Linear(rank→vocab), HF [out, in]
    pub confidence_proj: GpuTensor, // [1, h+rank]    — AcceptRatePredictor proj
    pub confidence_bias: GpuTensor, // [1]            — AcceptRatePredictor bias
}

impl DsparkHeadsWeights {
    /// Head params in a fixed order (matches `DsparkHeadsGrads::flat`):
    /// `[markov_w1, markov_w2, confidence_proj, confidence_bias]`.
    pub fn params(&self) -> Vec<&GpuTensor> {
        vec![
            &self.markov_w1,
            &self.markov_w2,
            &self.confidence_proj,
            &self.confidence_bias,
        ]
    }
    pub fn param_sizes(&self) -> Vec<usize> {
        self.params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }
}

/// Head weight grads (same field set / order as `DsparkHeadsWeights`).
pub struct DsparkHeadsGrads {
    pub d_markov_w1: GpuTensor,       // [vocab, rank]
    pub d_markov_w2: GpuTensor,       // [vocab, rank]
    pub d_confidence_proj: GpuTensor, // [1, h+rank]
    pub d_confidence_bias: GpuTensor, // [1]
}

impl DsparkHeadsGrads {
    /// Flatten in the SAME fixed order as `DsparkHeadsWeights::params()`.
    pub fn flat(&self) -> Vec<&GpuTensor> {
        vec![
            &self.d_markov_w1,
            &self.d_markov_w2,
            &self.d_confidence_proj,
            &self.d_confidence_bias,
        ]
    }
}

/// Backward-needed head activations. `markov_idx` is the flattened i32 gather
/// index reused by the scatter in backward (so `prev_tokens` need not be
/// threaded into `dspark_heads_backward`).
pub struct DsparkHeadsActs {
    pub draft_logits: GpuTensor,      // [block, vocab]  — head output
    pub markov_latent: GpuTensor,     // [block, rank]   — markov_w1[prev]
    pub confidence_logit: GpuTensor,  // [block]         — pre-sigmoid
    pub confidence_pred: GpuTensor,   // [block]         — sigmoid(logit)
    pub markov_idx: GpuTensor,        // [block*rank] i32 (Raw) — gather/scatter index
    pub markov_onehot_idx: GpuTensor, // [block] i32 (Raw) — dest `b*vocab + prev[b]`
}

/// Return head activations to the pool (GpuTensor has no Drop).
pub fn free_dspark_heads_acts(gpu: &mut Gpu, a: DsparkHeadsActs) -> HipResult<()> {
    let DsparkHeadsActs {
        draft_logits,
        markov_latent,
        confidence_logit,
        confidence_pred,
        markov_idx,
        markov_onehot_idx,
    } = a;
    for t in [
        draft_logits,
        markov_latent,
        confidence_logit,
        confidence_pred,
        markov_idx,
        markov_onehot_idx,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Return head grads to the pool after the optimizer step.
pub fn free_dspark_heads_grads(gpu: &mut Gpu, g: DsparkHeadsGrads) -> HipResult<()> {
    let DsparkHeadsGrads {
        d_markov_w1,
        d_markov_w2,
        d_confidence_proj,
        d_confidence_bias,
    } = g;
    for t in [
        d_markov_w1,
        d_markov_w2,
        d_confidence_proj,
        d_confidence_bias,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Upload a host i32 index as a Raw device tensor (rq gather/scatter read the
/// buffer as `i32*`; the dtype tag is unused by those kernels).
fn upload_idx_i32(gpu: &Gpu, data: &[i32], n: usize) -> HipResult<GpuTensor> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.upload_raw(bytes, &[n])
}

/// Row-concat `[x_head[b] ++ markov_latent[b]]` → `feat [block, h+rank]`.
fn concat_feat(
    gpu: &mut Gpu,
    x_head: &GpuTensor,
    markov_latent: &GpuTensor,
    block: usize,
    h: usize,
    r: usize,
) -> HipResult<GpuTensor> {
    let feat = gpu.zeros(&[block * (h + r)], DType::F32)?;
    for b in 0..block {
        gpu.memcpy_dtod_at_auto(&feat.buf, b * (h + r) * 4, &x_head.buf, b * h * 4, h * 4)?;
        gpu.memcpy_dtod_at_auto(
            &feat.buf,
            (b * (h + r) + h) * 4,
            &markov_latent.buf,
            b * r * 4,
            r * 4,
        )?;
    }
    Ok(feat)
}

// ── Heads forward ───────────────────────────────────────────────────────────

/// Head forward. `x_head` `[block*h]` (body output); `prev_tokens` `[block]` are
/// the previous-token ids per block position (VanillaMarkov step token); `lm_head`
/// `[vocab*h]` is the frozen target-shared lm-head weight (HF `[out, in]`).
///
/// Produces `draft_logits [block*vocab]`, `markov_latent [block*rank]`,
/// `confidence_logit`/`confidence_pred [block]` (saved for backward).
pub fn dspark_heads_forward(
    gpu: &mut Gpu,
    x_head: &GpuTensor,
    prev_tokens: &[u32],
    lm_head: &GpuTensor,
    w: &DsparkHeadsWeights,
    cfg: &DsparkHeadsConfig,
) -> HipResult<DsparkHeadsActs> {
    let (h, v, r) = (cfg.h, cfg.vocab, cfg.markov_rank);
    let block = x_head.shape.iter().product::<usize>() / h;
    debug_assert_eq!(prev_tokens.len(), block);

    // Flattened row index for markov_w1 gather: idx[b*r+j] = prev[b]*r + j.
    let mut idx_host = vec![0i32; block * r];
    // Per-block one-hot destination for the race-free backward: distinct index
    // `b*vocab + prev[b]` (used to build onehot[block, vocab] in dspark_heads_backward).
    let mut onehot_idx_host = vec![0i32; block];
    for b in 0..block {
        let base = prev_tokens[b] as i32 * r as i32;
        for j in 0..r {
            idx_host[b * r + j] = base + j as i32;
        }
        onehot_idx_host[b] = (b * v) as i32 + prev_tokens[b] as i32;
    }
    let markov_idx = upload_idx_i32(gpu, &idx_host, block * r)?;
    let markov_onehot_idx = upload_idx_i32(gpu, &onehot_idx_host, block)?;

    // markov_latent = markov_w1[prev]  (row gather of rank-wide rows).
    let markov_latent = gpu.zeros(&[block * r], DType::F32)?;
    gpu.rq_gather_f32(
        &w.markov_w1,
        &markov_idx,
        &markov_latent,
        block * r,
        block * r,
    )?;

    // markov_bias = markov_latent @ markov_w2ᵀ  [block, vocab].
    let markov_bias = gpu.zeros(&[block * v], DType::F32)?;
    linear_forward(gpu, &markov_latent, &w.markov_w2, &markov_bias, block, r, v)?;

    // base_logits = x_head @ lm_headᵀ  [block, vocab]  (lm_head frozen).
    let base_logits = gpu.zeros(&[block * v], DType::F32)?;
    linear_forward(gpu, x_head, lm_head, &base_logits, block, h, v)?;

    // draft_logits = base_logits + markov_bias.
    let draft_logits = gpu.zeros(&[block * v], DType::F32)?;
    gpu.add_f32(&base_logits, &markov_bias, &draft_logits)?;

    // confidence: features = concat(x_head, markov_latent) → proj + bias → sigmoid.
    let feat = concat_feat(gpu, x_head, &markov_latent, block, h, r)?;
    let confidence_logit = gpu.zeros(&[block], DType::F32)?;
    linear_forward(
        gpu,
        &feat,
        &w.confidence_proj,
        &confidence_logit,
        block,
        h + r,
        1,
    )?;
    gpu.bias_add_f32(&confidence_logit, &w.confidence_bias, block, 1)?;
    let confidence_pred = sigmoid_forward(gpu, &confidence_logit, block)?;

    for t in [markov_bias, base_logits, feat] {
        gpu.free_tensor(t)?;
    }

    Ok(DsparkHeadsActs {
        draft_logits,
        markov_latent,
        confidence_logit,
        confidence_pred,
        markov_idx,
        markov_onehot_idx,
    })
}

// ── Heads backward ──────────────────────────────────────────────────────────

/// Head backward. `d_draft_logits` `[block*vocab]` and `d_confidence_pred`
/// `[block]` are the two upstream seeds (either may be all-zero). Returns
/// `(d_x_head [block*h], DsparkHeadsGrads)`. `d_x_head` sums the lm-head path
/// and the confidence-feature path; `x_head`/`lm_head`/`w` are the same tensors
/// passed to the forward.
#[allow(clippy::too_many_arguments)]
pub fn dspark_heads_backward(
    gpu: &mut Gpu,
    d_draft_logits: &GpuTensor,
    d_confidence_pred: &GpuTensor,
    acts: &DsparkHeadsActs,
    x_head: &GpuTensor,
    lm_head: &GpuTensor,
    w: &DsparkHeadsWeights,
    cfg: &DsparkHeadsConfig,
) -> HipResult<(GpuTensor, DsparkHeadsGrads)> {
    let (h, v, r) = (cfg.h, cfg.vocab, cfg.markov_rank);
    let block = x_head.shape.iter().product::<usize>() / h;

    // d_x_head and d_markov_latent accumulate contributions from both heads.
    let d_x_head = gpu.zeros(&[block * h], DType::F32)?;
    let d_markov_latent = gpu.zeros(&[block * r], DType::F32)?;

    // ── draft_logits path: draft = base + markov_bias.
    //   base = x_head @ lm_headᵀ ⇒ d_x_head = d_draft @ lm_head (lm_head frozen).
    linear_backward_x(gpu, d_draft_logits, lm_head, &d_x_head, block, h, v, false)?; // first writer
                                                                                     //   markov_bias = markov_latent @ markov_w2ᵀ.
    let d_markov_w2 = gpu.zeros(&[v, r], DType::F32)?;
    linear_backward_w(
        gpu,
        d_draft_logits,
        &acts.markov_latent,
        &d_markov_w2,
        block,
        r,
        v,
        false,
    )?;
    linear_backward_x(
        gpu,
        d_draft_logits,
        &w.markov_w2,
        &d_markov_latent,
        block,
        r,
        v,
        false,
    )?; // first writer

    // ── confidence path: pred = sigmoid(proj(feat) + bias), feat=concat(x_head, markov_latent).
    let d_logit = sigmoid_backward(gpu, d_confidence_pred, &acts.confidence_pred, block)?;
    let feat = concat_feat(gpu, x_head, &acts.markov_latent, block, h, r)?;
    let d_confidence_proj = gpu.zeros(&[1, h + r], DType::F32)?;
    linear_backward_w(
        gpu,
        &d_logit,
        &feat,
        &d_confidence_proj,
        block,
        h + r,
        1,
        false,
    )?;
    let d_feat = gpu.zeros(&[block * (h + r)], DType::F32)?;
    linear_backward_x(
        gpu,
        &d_logit,
        &w.confidence_proj,
        &d_feat,
        block,
        h + r,
        1,
        false,
    )?;
    // bias grad = Σ_b d_logit[b]  (via dyᵀ·ones).
    let ones = gpu.full_f32(&[block], 1.0)?;
    let d_confidence_bias = gpu.zeros(&[1], DType::F32)?;
    linear_backward_w(gpu, &d_logit, &ones, &d_confidence_bias, block, 1, 1, false)?;

    // Split d_feat rows → d_x_head (cols 0..h) and d_markov_latent (cols h..h+r),
    // accumulating onto the draft-path contributions.
    let d_xh_conf = gpu.zeros(&[block * h], DType::F32)?;
    let d_ml_conf = gpu.zeros(&[block * r], DType::F32)?;
    for b in 0..block {
        gpu.memcpy_dtod_at_auto(
            &d_xh_conf.buf,
            b * h * 4,
            &d_feat.buf,
            b * (h + r) * 4,
            h * 4,
        )?;
        gpu.memcpy_dtod_at_auto(
            &d_ml_conf.buf,
            b * r * 4,
            &d_feat.buf,
            (b * (h + r) + h) * 4,
            r * 4,
        )?;
    }
    gpu.add_inplace_f32(&d_x_head, &d_xh_conf)?;
    gpu.add_inplace_f32(&d_markov_latent, &d_ml_conf)?;

    // ── markov_w1: forward is a row gather markov_latent[b,:] = markov_w1[prev[b],:],
    // so d_markov_w1[i,:] = Σ_{b: prev[b]=i} d_markov_latent[b,:] = onehotᵀ @ d_markov_latent.
    // A non-atomic scatter-add races when two blocks share prev_token, dropping a
    // contribution; the GEMM sums duplicates correctly. Build onehot[block, vocab]
    // via a scatter into DISTINCT destinations (b*vocab + prev[b]) — race-free —
    // then dw = onehotᵀ·d_markov_latent through the training GEMM (dyᵀ·x).
    let onehot = gpu.zeros(&[block * v], DType::F32)?;
    let onehot_ones = gpu.full_f32(&[block], 1.0)?;
    gpu.rq_scatter_add_f32(&onehot, &acts.markov_onehot_idx, &onehot_ones, block)?;
    let d_markov_w1 = gpu.zeros(&[v, r], DType::F32)?;
    linear_backward_w(
        gpu,
        &onehot,
        &d_markov_latent,
        &d_markov_w1,
        block,
        r,
        v,
        false,
    )?;
    for t in [onehot, onehot_ones] {
        gpu.free_tensor(t)?;
    }

    for t in [
        d_logit,
        feat,
        d_feat,
        ones,
        d_xh_conf,
        d_ml_conf,
        d_markov_latent,
    ] {
        gpu.free_tensor(t)?;
    }

    Ok((
        d_x_head,
        DsparkHeadsGrads {
            d_markov_w1,
            d_markov_w2,
            d_confidence_proj,
            d_confidence_bias,
        },
    ))
}

// ── Full drafter (body → heads) ─────────────────────────────────────────────

/// Combined body + head weights. `params()` appends the head params AFTER the
/// body params, preserving the body's fixed order (so an AdamW state built for
/// the body remains a prefix of the full state).
pub struct DsparkFullWeights {
    pub body: DsparkDrafterWeights,
    pub heads: DsparkHeadsWeights,
}

impl DsparkFullWeights {
    /// Body params (in `DsparkDrafterWeights::params()` order) then head params
    /// (in `DsparkHeadsWeights::params()` order).
    pub fn params(&self) -> Vec<&GpuTensor> {
        let mut v = self.body.params();
        v.extend(self.heads.params());
        v
    }
    pub fn param_sizes(&self) -> Vec<usize> {
        self.params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }
}

/// Combined grads; `flat()` mirrors `DsparkFullWeights::params()` (body then heads).
pub struct DsparkFullGrads {
    pub body: DsparkDrafterGrads,
    pub heads: DsparkHeadsGrads,
}

impl DsparkFullGrads {
    pub fn flat(&self) -> Vec<&GpuTensor> {
        let mut v = self.body.flat();
        v.extend(self.heads.flat());
        v
    }
}

/// Saved activations for the full body → heads forward.
pub struct DsparkFullActs {
    pub body: DsparkDrafterActs,
    pub heads: DsparkHeadsActs,
}

/// Borrowed view of the head outputs a train loop consumes.
pub struct DsparkForwardOutput<'a> {
    pub draft_logits: &'a GpuTensor,    // [block, vocab]
    pub confidence_pred: &'a GpuTensor, // [block]
    pub markov_latent: &'a GpuTensor,   // [block, rank]
}

impl DsparkFullActs {
    pub fn output(&self) -> DsparkForwardOutput<'_> {
        DsparkForwardOutput {
            draft_logits: &self.heads.draft_logits,
            confidence_pred: &self.heads.confidence_pred,
            markov_latent: &self.heads.markov_latent,
        }
    }
}

/// Return a full forward's activations to the pool.
pub fn free_dspark_full_acts(gpu: &mut Gpu, a: DsparkFullActs) -> HipResult<()> {
    free_dspark_drafter_acts(gpu, a.body)?;
    free_dspark_heads_acts(gpu, a.heads)?;
    Ok(())
}

/// Full training forward: body (`dspark_drafter_forward_train`) → heads.
#[allow(clippy::too_many_arguments)]
pub fn dspark_drafter_forward_full(
    gpu: &mut Gpu,
    weights: &DsparkFullWeights,
    cfg: &DsparkDrafterConfig,
    heads_cfg: &DsparkHeadsConfig,
    main_hidden: &GpuTensor,
    block_embeds: &GpuTensor,
    prev_tokens: &[u32],
    lm_head: &GpuTensor,
    ctx_positions: &[f32],
    block_positions: &[f32],
    bias: Option<&GpuTensor>,
) -> HipResult<DsparkFullActs> {
    let body = dspark_drafter_forward_train(
        gpu,
        &weights.body,
        cfg,
        main_hidden,
        block_embeds,
        ctx_positions,
        block_positions,
        bias,
    )?;
    let heads = dspark_heads_forward(
        gpu,
        body.x_head(),
        prev_tokens,
        lm_head,
        &weights.heads,
        heads_cfg,
    )?;
    Ok(DsparkFullActs { body, heads })
}

/// Full training backward: heads (from `d_draft_logits` + `d_confidence_pred`)
/// → body (from the summed `d_x_head`). `d_main_hidden` rides inside `body`.
#[allow(clippy::too_many_arguments)]
pub fn dspark_drafter_backward_full(
    gpu: &mut Gpu,
    weights: &DsparkFullWeights,
    cfg: &DsparkDrafterConfig,
    heads_cfg: &DsparkHeadsConfig,
    main_hidden: &GpuTensor,
    lm_head: &GpuTensor,
    acts: &DsparkFullActs,
    d_draft_logits: &GpuTensor,
    d_confidence_pred: &GpuTensor,
) -> HipResult<DsparkFullGrads> {
    let (d_x_head, heads) = dspark_heads_backward(
        gpu,
        d_draft_logits,
        d_confidence_pred,
        &acts.heads,
        acts.body.x_head(),
        lm_head,
        &weights.heads,
        heads_cfg,
    )?;
    let body =
        dspark_drafter_backward(gpu, &weights.body, cfg, main_hidden, &acts.body, &d_x_head)?;
    gpu.free_tensor(d_x_head)?;
    Ok(DsparkFullGrads { body, heads })
}
