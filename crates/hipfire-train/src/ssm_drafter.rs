// SPDX-License-Identifier: Apache-2.0
//! PFlash importance-scorer drafter with a GLA-lite / minimal-selective-SSM body.
//!
//! Same shape as [`crate::drafter::Drafter`] — shared frozen target embedding →
//! input projection → N small blocks → out_norm → K-head → rope → cosine score —
//! but the blocks are gated-recurrence ([`crate::ssm_block`]) instead of attention.
//! Motivated by P5: the attention drafter ceilings at +0.47 on the SSM-driven
//! qwen3.5 target (tuning-resistant); an SSM body shares the target's bias.
//!
//! The scoring K-head (`wk_score` → rope) and PFlash cosine consumer are unchanged
//! from the attention drafter, so this is a drop-in body swap for training.

use crate::ops::linear::{linear_backward_w, linear_backward_x, linear_forward};
use crate::ops::pflash_score::pflash_score_backward;
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use crate::ops::rope::{rope_backward, rope_forward};
use crate::ssm_block::{
    free_ssm_block_acts, ssm_block_backward, ssm_block_forward, SsmBlockActivations, SsmBlockDims,
    SsmBlockGrad, SsmBlockWeights,
};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

fn clone_tensor(gpu: &mut Gpu, t: &GpuTensor) -> HipResult<GpuTensor> {
    let n: usize = t.shape.iter().product();
    let c = gpu.zeros(&t.shape, t.dtype)?;
    gpu.memcpy_dtod_auto(&c.buf, &t.buf, n * 4)?;
    Ok(c)
}

/// Shape hyperparameters for the SSM drafter body (independent of the target).
#[derive(Clone, Copy)]
pub struct SsmDrafterConfig {
    pub h_draft: usize, // body hidden width (≪ target)
    pub n_layers: usize,
    pub inter: usize,   // MLP intermediate
    pub n_kv: usize,    // scoring K-head geometry (rope)
    pub head_dim: usize,
    pub rope_base: f32,
    pub eps: f32,
}

impl SsmDrafterConfig {
    /// Tiny default: h=512, 3 layers, MLP 2×, scoring K-head 4×64.
    pub fn tiny(rope_base: f32, eps: f32) -> Self {
        SsmDrafterConfig { h_draft: 512, n_layers: 3, inter: 1024, n_kv: 4, head_dim: 64, rope_base, eps }
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
}

/// Trainable weights for one SSM layer (owned, row-major `[out, in]`).
pub struct SsmLayerWeights {
    pub norm1: GpuTensor,
    pub w_u: GpuTensor,
    pub w_g: GpuTensor,
    pub w_o: GpuTensor,
    pub norm2: GpuTensor,
    pub wgate: GpuTensor,
    pub wup: GpuTensor,
    pub wdown: GpuTensor,
}

pub struct SsmDrafter {
    pub embed: GpuTensor, // shared target embedding [vocab, h_t], FROZEN
    pub h_t: usize,
    pub vocab: usize,
    pub in_proj: GpuTensor, // [h_draft, h_t]
    pub layers: Vec<SsmLayerWeights>,
    pub out_norm: GpuTensor, // [h_draft]
    pub wk_score: GpuTensor, // [kv_dim, h_draft]
    pub dims: SsmBlockDims,
    pub cfg: SsmDrafterConfig,
    pub seq: usize,
}

/// Deterministic LCG pseudo-random fill in [-scale, scale).
fn rand_fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32; // ~[0,2)
            (u - 1.0) * scale
        })
        .collect()
}

impl SsmDrafter {
    /// Build a randomly-initialised SSM drafter sharing `embed` (moved in, frozen).
    pub fn new(
        gpu: &mut Gpu,
        embed: GpuTensor,
        h_t: usize,
        vocab: usize,
        cfg: SsmDrafterConfig,
        seq: usize,
    ) -> HipResult<Self> {
        let hd = cfg.h_draft;
        let kvd = cfg.kv_dim();
        let lin = |gpu: &mut Gpu, out: usize, inn: usize, seed: u64| -> HipResult<GpuTensor> {
            let scale = 1.0 / (inn as f32).sqrt();
            gpu.upload_f32(&rand_fill(out * inn, seed, scale), &[out, inn])
        };
        let ones = |gpu: &mut Gpu, n: usize| -> HipResult<GpuTensor> {
            gpu.upload_f32(&vec![1.0f32; n], &[n])
        };

        let in_proj = lin(gpu, hd, h_t, 0xA11CE)?;
        let out_norm = ones(gpu, hd)?;
        let wk_score = lin(gpu, kvd, hd, 0x5C03E)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let s = 0x2000 * (li as u64 + 1);
            layers.push(SsmLayerWeights {
                norm1: ones(gpu, hd)?,
                w_u: lin(gpu, hd, hd, s + 1)?,
                w_g: lin(gpu, hd, hd, s + 2)?,
                w_o: lin(gpu, hd, hd, s + 3)?,
                norm2: ones(gpu, hd)?,
                wgate: lin(gpu, cfg.inter, hd, s + 4)?,
                wup: lin(gpu, cfg.inter, hd, s + 5)?,
                wdown: lin(gpu, hd, cfg.inter, s + 6)?,
            });
        }

        let dims = SsmBlockDims { seq, h: hd, inter: cfg.inter, eps: cfg.eps };
        Ok(SsmDrafter { embed, h_t, vocab, in_proj, layers, out_norm, wk_score, dims, cfg, seq })
    }

    fn layer_views<'a>(w: &'a SsmLayerWeights) -> SsmBlockWeights<'a> {
        SsmBlockWeights {
            norm1: &w.norm1, w_u: &w.w_u, w_g: &w.w_g, w_o: &w.w_o,
            norm2: &w.norm2, wgate: &w.wgate, wup: &w.wup, wdown: &w.wdown,
        }
    }

    /// Trainable params in a fixed order (matches `SsmDrafterGrads::flat`):
    /// in_proj, per layer [w_u,w_g,w_o,wgate,wup,wdown,norm1,norm2], out_norm,
    /// wk_score. (Embedding frozen.)
    pub fn params(&self) -> Vec<&GpuTensor> {
        let mut v = vec![&self.in_proj];
        for w in &self.layers {
            v.push(&w.w_u);
            v.push(&w.w_g);
            v.push(&w.w_o);
            v.push(&w.wgate);
            v.push(&w.wup);
            v.push(&w.wdown);
            v.push(&w.norm1);
            v.push(&w.norm2);
        }
        v.push(&self.out_norm);
        v.push(&self.wk_score);
        v
    }

    pub fn param_sizes(&self) -> Vec<usize> {
        self.params().iter().map(|t| t.shape.iter().product()).collect()
    }
}

/// Saved activations for the SSM drafter backward.
pub struct SsmDrafterActs {
    pub emb: GpuTensor,
    pub layer_inputs: Vec<GpuTensor>,
    pub layer_acts: Vec<SsmBlockActivations>,
    pub x_last: GpuTensor,
    pub xn_out: GpuTensor,
    pub rinv_out: GpuTensor,
    pub score_k: GpuTensor,
    pub pos: GpuTensor,
}

/// Grads for every trainable param, in `params()` order.
pub struct SsmDrafterGrads {
    pub d_in_proj: GpuTensor,
    pub layers: Vec<SsmBlockGrad>,
    pub d_out_norm: GpuTensor,
    pub d_wk_score: GpuTensor,
}

impl SsmDrafterGrads {
    pub fn flat(&self) -> Vec<&GpuTensor> {
        let mut v = vec![&self.d_in_proj];
        for g in &self.layers {
            v.push(&g.dw_u);
            v.push(&g.dw_g);
            v.push(&g.dw_o);
            v.push(&g.dwgate);
            v.push(&g.dwup);
            v.push(&g.dwdown);
            v.push(&g.dnorm1);
            v.push(&g.dnorm2);
        }
        v.push(&self.d_out_norm);
        v.push(&self.d_wk_score);
        v
    }
}

pub fn free_ssm_drafter_acts(gpu: &mut Gpu, a: SsmDrafterActs) -> HipResult<()> {
    let SsmDrafterActs { emb, layer_inputs, layer_acts, x_last, xn_out, rinv_out, score_k, pos } = a;
    for t in layer_inputs {
        gpu.free_tensor(t)?;
    }
    for b in layer_acts {
        free_ssm_block_acts(gpu, b)?;
    }
    for t in [emb, x_last, xn_out, rinv_out, score_k, pos] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

pub fn free_ssm_drafter_grads(gpu: &mut Gpu, g: SsmDrafterGrads) -> HipResult<()> {
    let SsmDrafterGrads { d_in_proj, layers, d_out_norm, d_wk_score } = g;
    gpu.free_tensor(d_in_proj)?;
    for lg in layers {
        crate::ssm_block::free_ssm_block_grad(gpu, lg)?;
    }
    gpu.free_tensor(d_out_norm)?;
    gpu.free_tensor(d_wk_score)?;
    Ok(())
}

/// Training forward: embed → in_proj → SSM blocks → out_norm → K-head → rope.
pub fn ssm_drafter_forward_train(
    gpu: &mut Gpu,
    d: &SsmDrafter,
    token_ids: &[u32],
    pos_host: &[f32],
) -> HipResult<SsmDrafterActs> {
    let (seq, hd, h_t) = (d.seq, d.dims.h, d.h_t);
    let (kvd, n_kv, head_dim) = (d.cfg.kv_dim(), d.cfg.n_kv, d.cfg.head_dim);
    assert_eq!(token_ids.len(), seq);

    let emb = gpu.zeros(&[seq * h_t], DType::F32)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        gpu.strided_copy_2d(&d.embed, tok as usize * h_t, h_t, &emb, t * h_t, h_t, 1, h_t, false)?;
    }
    let x0 = gpu.zeros(&[seq * hd], DType::F32)?;
    linear_forward(gpu, &emb, &d.in_proj, &x0, seq, h_t, hd)?;

    let mut layer_inputs = Vec::with_capacity(d.layers.len());
    let mut layer_acts = Vec::with_capacity(d.layers.len());
    let mut x = x0;
    for w in &d.layers {
        layer_inputs.push(clone_tensor(gpu, &x)?);
        let bw = SsmDrafter::layer_views(w);
        let (x_out, acts) = ssm_block_forward(gpu, &x, &bw, &d.dims)?;
        gpu.free_tensor(x)?;
        layer_acts.push(acts);
        x = x_out;
    }
    let x_last = x;

    let xn_out = gpu.zeros(&[seq * hd], DType::F32)?;
    let rinv_out = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(gpu, &x_last, &d.out_norm, &xn_out, &rinv_out, seq, hd, d.dims.eps)?;

    let ks = gpu.zeros(&[seq * kvd], DType::F32)?;
    linear_forward(gpu, &xn_out, &d.wk_score, &ks, seq, hd, kvd)?;
    let pos = gpu.upload_f32(pos_host, &[seq])?;
    let score_k = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_forward(gpu, &ks, &score_k, &pos, seq * n_kv, n_kv, head_dim, d.cfg.rope_base)?;
    gpu.free_tensor(ks)?;

    Ok(SsmDrafterActs { emb, layer_inputs, layer_acts, x_last, xn_out, rinv_out, score_k, pos })
}

/// Training backward: `dscores` `[n_blocks]` → all param grads.
pub fn ssm_drafter_backward(
    gpu: &mut Gpu,
    d: &SsmDrafter,
    acts: &SsmDrafterActs,
    dscores: &GpuTensor,
    block_size: usize,
    n_blocks: usize,
    last_pos: usize,
) -> HipResult<SsmDrafterGrads> {
    let (seq, hd, h_t) = (d.seq, d.dims.h, d.h_t);
    let (kvd, n_kv, head_dim) = (d.cfg.kv_dim(), d.cfg.n_kv, d.cfg.head_dim);

    // score head: dscores → d(score_k) → d(ks) (derope) → wk_score grad + d(xn_out)
    let d_score_k =
        pflash_score_backward(gpu, &acts.score_k, dscores, seq, kvd, block_size, n_blocks, last_pos)?;
    let d_ks = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_backward(gpu, &d_score_k, &d_ks, &acts.pos, seq * n_kv, n_kv, head_dim, d.cfg.rope_base)?;
    let d_wk_score = gpu.zeros(&[kvd * hd], DType::F32)?;
    linear_backward_w(gpu, &d_ks, &acts.xn_out, &d_wk_score, seq, hd, kvd, false)?;
    let d_xn_out = gpu.zeros(&[seq * hd], DType::F32)?;
    linear_backward_x(gpu, &d_ks, &d.wk_score, &d_xn_out, seq, hd, kvd, false)?;
    gpu.free_tensor(d_score_k)?;
    gpu.free_tensor(d_ks)?;

    // out_norm backward → d(x_last)
    let d_x_last = gpu.zeros(&[seq * hd], DType::F32)?;
    let d_out_norm = gpu.zeros(&[hd], DType::F32)?;
    rmsnorm_backward(gpu, &d_xn_out, &acts.x_last, &d.out_norm, &acts.rinv_out, &d_x_last, &d_out_norm, seq, hd)?;
    gpu.free_tensor(d_xn_out)?;

    // SSM blocks in reverse
    let mut layer_grads: Vec<SsmBlockGrad> = Vec::with_capacity(d.layers.len());
    let mut d_x = d_x_last;
    for i in (0..d.layers.len()).rev() {
        let bw = SsmDrafter::layer_views(&d.layers[i]);
        let (d_in, wg) = ssm_block_backward(gpu, &d_x, &acts.layer_inputs[i], &bw, &acts.layer_acts[i], &d.dims)?;
        gpu.free_tensor(d_x)?;
        layer_grads.push(wg);
        d_x = d_in;
    }
    layer_grads.reverse();

    // in_proj backward (embed frozen)
    let d_in_proj = gpu.zeros(&[hd * h_t], DType::F32)?;
    linear_backward_w(gpu, &d_x, &acts.emb, &d_in_proj, seq, h_t, hd, false)?;
    gpu.free_tensor(d_x)?;

    Ok(SsmDrafterGrads { d_in_proj, layers: layer_grads, d_out_norm, d_wk_score })
}

/// Inference forward: return the LAST block's-body scoring K (`[seq*kv_dim]`,
/// post-rope) that PFlash scores. (Mirrors `drafter_forward`.)
pub fn ssm_drafter_forward(
    gpu: &mut Gpu,
    d: &SsmDrafter,
    token_ids: &[u32],
    pos_host: &[f32],
) -> HipResult<GpuTensor> {
    let acts = ssm_drafter_forward_train(gpu, d, token_ids, pos_host)?;
    let score_k = clone_tensor(gpu, &acts.score_k)?;
    free_ssm_drafter_acts(gpu, acts)?;
    Ok(score_k)
}
