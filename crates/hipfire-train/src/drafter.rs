// SPDX-License-Identifier: Apache-2.0
//! PFlash importance-scorer drafter (P2 scaffold).
//!
//! A tiny model that reuses the TARGET's token embedding (shared, frozen,
//! already resident) and adds a narrow input projection + a few small
//! attention+MLP blocks, then emits per-token K from its last block. PFlash's
//! existing `cosine(block_mean_K, last_token_K)` scoring consumes that K
//! unchanged (the "drop-in" training target chosen 2026-06-18).
//!
//! Design rationale lives in docs/plans/2026-06-18-pflash-qat-drafter.md:
//! attention is non-negotiable (importance is contextual — M0b needle); the
//! shared embedding is what makes "tiny" possible at a 248K vocab; width is
//! `h_draft ≪ h_target` via the learned input projection.
//!
//! This module is FORWARD-only for now (P2). Training (P3) backprops a listwise
//! ranking loss from the drafter's block-cosine scores toward the target's
//! mid-layer block-cosine ranking, reusing `block_backward`.

use crate::block::{
    block_backward_full, block_forward, BlockActivations, BlockDims, BlockLora, BlockWeightGrad,
    BlockWeights,
};
use crate::model::{LayerLora, LayerWeights};
use crate::ops::linear::{linear_backward_w, linear_backward_x, linear_forward};
use crate::ops::pflash_score::pflash_score_backward;
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use crate::ops::rope::{rope_backward, rope_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

fn clone_tensor(gpu: &mut Gpu, t: &GpuTensor) -> HipResult<GpuTensor> {
    let n: usize = t.shape.iter().product();
    let c = gpu.zeros(&t.shape, t.dtype)?;
    gpu.memcpy_dtod_auto(&c.buf, &t.buf, n * 4)?;
    Ok(c)
}

/// Shape/size hyperparameters for the drafter body (independent of the target).
#[derive(Clone, Copy)]
pub struct DrafterConfig {
    pub h_draft: usize,  // drafter hidden width (≪ target)
    pub n_layers: usize, // small (2–4)
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub rope_base: f32,
    pub eps: f32,
}

impl DrafterConfig {
    /// Sensible tiny default: h=512, 3 layers, GQA 8/4 heads × 64, MLP 2×.
    pub fn tiny(rope_base: f32, eps: f32) -> Self {
        DrafterConfig {
            h_draft: 512,
            n_layers: 3,
            n_heads: 8,
            n_kv: 4,
            head_dim: 64,
            inter: 1024,
            rope_base,
            eps,
        }
    }
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
}

pub struct Drafter {
    pub embed: GpuTensor, // shared target embedding [vocab, h_t], FROZEN
    pub h_t: usize,
    pub vocab: usize,
    pub in_proj: GpuTensor, // [h_draft, h_t]
    pub layers: Vec<(LayerWeights, LayerLora)>,
    pub out_norm: GpuTensor, // [h_draft] final RMSNorm before the score K-head
    pub wk_score: GpuTensor, // [kv_dim, h_draft] scoring K projection (post-rope)
    pub dims: BlockDims,     // h = h_draft
}

/// Deterministic LCG pseudo-random fill in [-scale, scale).
fn rand_fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32; // ~[0,2)
            (u - 1.0) * scale
        })
        .collect()
}

impl Drafter {
    /// Build a randomly-initialised drafter that shares `embed` (moved in,
    /// frozen). `h_t` is the target/embedding width; `vocab` its row count.
    /// `seq` fixes `BlockDims.seq`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &mut Gpu,
        embed: GpuTensor,
        h_t: usize,
        vocab: usize,
        cfg: DrafterConfig,
        seq: usize,
    ) -> HipResult<Self> {
        let (hd, qd, kvd) = (cfg.h_draft, cfg.q_dim(), cfg.kv_dim());
        // Kaiming-ish: U(-1,1)/sqrt(fan_in).
        let lin = |gpu: &mut Gpu, out: usize, inn: usize, seed: u64| -> HipResult<GpuTensor> {
            let scale = 1.0 / (inn as f32).sqrt();
            gpu.upload_f32(&rand_fill(out * inn, seed, scale), &[out, inn])
        };
        let ones = |gpu: &mut Gpu, n: usize| -> HipResult<GpuTensor> {
            gpu.upload_f32(&vec![1.0f32; n], &[n])
        };

        let in_proj = lin(gpu, hd, h_t, 0xA11CE)?;
        let out_norm = ones(gpu, hd)?;
        let wk_score = lin(gpu, kvd, hd, 0x5C03E)?; // scoring K-head [kv_dim, h]

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let s = 0x1000 * (li as u64 + 1);
            let weights = LayerWeights {
                norm1: ones(gpu, hd)?,
                wq: lin(gpu, qd, hd, s + 1)?,
                wk: lin(gpu, kvd, hd, s + 2)?,
                wv: lin(gpu, kvd, hd, s + 3)?,
                wo: lin(gpu, hd, qd, s + 4)?,
                norm2: ones(gpu, hd)?,
                wgate: lin(gpu, cfg.inter, hd, s + 5)?,
                wup: lin(gpu, cfg.inter, hd, s + 6)?,
                wdown: lin(gpu, hd, cfg.inter, s + 7)?,
            };
            // Zero LoRA (rank 4): block_forward applies it as a no-op until P3
            // makes these the trainable adapters (or we train the base directly).
            let r = 4;
            let lora = LayerLora {
                aq: gpu.upload_f32(
                    &rand_fill(r * hd, s + 8, 1.0 / (hd as f32).sqrt()),
                    &[r * hd],
                )?,
                bq: gpu.zeros(&[qd * r], DType::F32)?,
                av: gpu.upload_f32(
                    &rand_fill(r * hd, s + 9, 1.0 / (hd as f32).sqrt()),
                    &[r * hd],
                )?,
                bv: gpu.zeros(&[kvd * r], DType::F32)?,
            };
            layers.push((weights, lora));
        }

        let dims = BlockDims {
            seq,
            h: hd,
            n_heads: cfg.n_heads,
            n_kv: cfg.n_kv,
            head_dim: cfg.head_dim,
            inter: cfg.inter,
            rope_base: cfg.rope_base,
            eps: cfg.eps,
            lora_scale: 1.0 / 4.0,
            lora_rank: 4,
        };

        Ok(Drafter {
            embed,
            h_t,
            vocab,
            in_proj,
            layers,
            out_norm,
            wk_score,
            dims,
        })
    }

    /// Trainable params in a fixed order (matches `DrafterGrads::flat`):
    /// in_proj, then per layer [wq,wk,wv,wo,wgate,wup,wdown,norm1,norm2], then
    /// out_norm, wk_score. (Embedding is frozen; LoRA is unused/no-op.)
    pub fn params(&self) -> Vec<&GpuTensor> {
        let mut v = vec![&self.in_proj];
        for (w, _) in &self.layers {
            v.push(&w.wq);
            v.push(&w.wk);
            v.push(&w.wv);
            v.push(&w.wo);
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
        self.params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }
}

/// Saved activations for the training backward pass.
pub struct DrafterActs {
    pub emb: GpuTensor,               // [seq*h_t] frozen-embedding lookup
    pub layer_inputs: Vec<GpuTensor>, // input to each block
    pub layer_acts: Vec<BlockActivations>,
    pub x_last: GpuTensor,   // last block output [seq*h]
    pub xn_out: GpuTensor,   // out_norm output [seq*h]
    pub rinv_out: GpuTensor, // [seq]
    pub score_k: GpuTensor,  // post-rope scoring K [seq*kv_dim]
    pub pos: GpuTensor,      // [seq] positions (rope bwd)
}

/// Per-layer base grads + the two norm grads (norm grads come from the LoRA-grad
/// struct's dnorm fields; LoRA adapters themselves are unused).
pub struct DrafterGrads {
    pub d_in_proj: GpuTensor,
    pub layers: Vec<(BlockWeightGrad, GpuTensor, GpuTensor)>, // (base, dnorm1, dnorm2)
    pub d_out_norm: GpuTensor,
    pub d_wk_score: GpuTensor,
}

impl DrafterGrads {
    /// Flatten in the SAME order as `Drafter::params()`.
    pub fn flat(&self) -> Vec<&GpuTensor> {
        let mut v = vec![&self.d_in_proj];
        for (wg, dn1, dn2) in &self.layers {
            v.push(&wg.dwq);
            v.push(&wg.dwk);
            v.push(&wg.dwv);
            v.push(&wg.dwo);
            v.push(&wg.dwgate);
            v.push(&wg.dwup);
            v.push(&wg.dwdown);
            v.push(dn1);
            v.push(dn2);
        }
        v.push(&self.d_out_norm);
        v.push(&self.d_wk_score);
        v
    }
}

/// Return a forward's saved activations to the pool (no Drop on GpuTensor).
pub fn free_drafter_acts(gpu: &mut Gpu, a: DrafterActs) -> HipResult<()> {
    let DrafterActs {
        emb,
        layer_inputs,
        layer_acts,
        x_last,
        xn_out,
        rinv_out,
        score_k,
        pos,
    } = a;
    for t in layer_inputs {
        gpu.free_tensor(t)?;
    }
    for b in layer_acts {
        let BlockActivations {
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
        } = b;
        for t in [
            xn1, rinv1, hq, hv, q_r, k_r, v, p_all, ctx, x_mid, xn2, rinv2, gate, up, act, pos,
        ] {
            gpu.free_tensor(t)?;
        }
    }
    for t in [emb, x_last, xn_out, rinv_out, score_k, pos] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Return a backward's grads to the pool after the optimizer step.
pub fn free_drafter_grads(gpu: &mut Gpu, g: DrafterGrads) -> HipResult<()> {
    let DrafterGrads {
        d_in_proj,
        layers,
        d_out_norm,
        d_wk_score,
    } = g;
    gpu.free_tensor(d_in_proj)?;
    for (wg, dn1, dn2) in layers {
        let BlockWeightGrad {
            dwq,
            dwk,
            dwv,
            dwo,
            dwgate,
            dwup,
            dwdown,
        } = wg;
        for t in [dwq, dwk, dwv, dwo, dwgate, dwup, dwdown, dn1, dn2] {
            gpu.free_tensor(t)?;
        }
    }
    gpu.free_tensor(d_out_norm)?;
    gpu.free_tensor(d_wk_score)?;
    Ok(())
}

fn block_views<'a>(lw: &'a LayerWeights, ll: &'a LayerLora) -> (BlockWeights<'a>, BlockLora<'a>) {
    (
        BlockWeights {
            norm1: &lw.norm1,
            wq: &lw.wq,
            wk: &lw.wk,
            wv: &lw.wv,
            wo: &lw.wo,
            norm2: &lw.norm2,
            wgate: &lw.wgate,
            wup: &lw.wup,
            wdown: &lw.wdown,
        },
        BlockLora {
            aq: &ll.aq,
            bq: &ll.bq,
            av: &ll.av,
            bv: &ll.bv,
        },
    )
}

/// Training forward: embed → in_proj → blocks → out_norm → K-head → rope, saving
/// everything the backward needs. The scoring K is `acts.score_k`.
pub fn drafter_forward_train(
    gpu: &mut Gpu,
    d: &Drafter,
    token_ids: &[u32],
    pos_host: &[f32],
) -> HipResult<DrafterActs> {
    let (seq, hd, h_t) = (d.dims.seq, d.dims.h, d.h_t);
    let (kvd, n_kv, head_dim) = (d.dims.kv_dim(), d.dims.n_kv, d.dims.head_dim);
    assert_eq!(token_ids.len(), seq);

    let emb = gpu.zeros(&[seq * h_t], DType::F32)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        gpu.strided_copy_2d(
            &d.embed,
            tok as usize * h_t,
            h_t,
            &emb,
            t * h_t,
            h_t,
            1,
            h_t,
            false,
        )?;
    }
    let x0 = gpu.zeros(&[seq * hd], DType::F32)?;
    linear_forward(gpu, &emb, &d.in_proj, &x0, seq, h_t, hd)?;

    let mut layer_inputs = Vec::with_capacity(d.layers.len());
    let mut layer_acts = Vec::with_capacity(d.layers.len());
    let mut x = x0;
    for (lw, ll) in &d.layers {
        layer_inputs.push(clone_tensor(gpu, &x)?);
        let (bw, bl) = block_views(lw, ll);
        let (x_out, acts) = block_forward(gpu, &x, &bw, &bl, &d.dims, pos_host)?;
        layer_acts.push(acts);
        x = x_out;
    }
    let x_last = x;

    let xn_out = gpu.zeros(&[seq * hd], DType::F32)?;
    let rinv_out = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(
        gpu,
        &x_last,
        &d.out_norm,
        &xn_out,
        &rinv_out,
        seq,
        hd,
        d.dims.eps,
    )?;

    let ks = gpu.zeros(&[seq * kvd], DType::F32)?;
    linear_forward(gpu, &xn_out, &d.wk_score, &ks, seq, hd, kvd)?;
    let pos = gpu.upload_f32(pos_host, &[seq])?;
    let score_k = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_forward(
        gpu,
        &ks,
        &score_k,
        &pos,
        seq * n_kv,
        n_kv,
        head_dim,
        d.dims.rope_base,
    )?;

    Ok(DrafterActs {
        emb,
        layer_inputs,
        layer_acts,
        x_last,
        xn_out,
        rinv_out,
        score_k,
        pos,
    })
}

/// Training backward: `dscores` `[n_blocks]` (grad of loss w.r.t. the block
/// cosine scores) → all drafter param grads. `block_size`/`last_pos` must match
/// the forward scoring call.
pub fn drafter_backward(
    gpu: &mut Gpu,
    d: &Drafter,
    acts: &DrafterActs,
    dscores: &GpuTensor,
    block_size: usize,
    n_blocks: usize,
    last_pos: usize,
) -> HipResult<DrafterGrads> {
    let (seq, hd, h_t) = (d.dims.seq, d.dims.h, d.h_t);
    let (kvd, n_kv, head_dim) = (d.dims.kv_dim(), d.dims.n_kv, d.dims.head_dim);

    // score head: dscores → d(score_k) → d(ks) (derope) → wk_score grad + d(xn_out)
    let d_score_k = pflash_score_backward(
        gpu,
        &acts.score_k,
        dscores,
        seq,
        kvd,
        block_size,
        n_blocks,
        last_pos,
    )?;
    let d_ks = gpu.zeros(&[seq * kvd], DType::F32)?;
    rope_backward(
        gpu,
        &d_score_k,
        &d_ks,
        &acts.pos,
        seq * n_kv,
        n_kv,
        head_dim,
        d.dims.rope_base,
    )?;
    let d_wk_score = gpu.zeros(&[kvd * hd], DType::F32)?;
    linear_backward_w(gpu, &d_ks, &acts.xn_out, &d_wk_score, seq, hd, kvd, false)?;
    let d_xn_out = gpu.zeros(&[seq * hd], DType::F32)?;
    linear_backward_x(gpu, &d_ks, &d.wk_score, &d_xn_out, seq, hd, kvd, false)?;
    gpu.free_tensor(d_score_k)?;
    gpu.free_tensor(d_ks)?;

    // out_norm backward → d(x_last)
    let d_x_last = gpu.zeros(&[seq * hd], DType::F32)?;
    let d_out_norm = gpu.zeros(&[hd], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xn_out,
        &acts.x_last,
        &d.out_norm,
        &acts.rinv_out,
        &d_x_last,
        &d_out_norm,
        seq,
        hd,
    )?;
    gpu.free_tensor(d_xn_out)?;

    // blocks in reverse (full base grads)
    let mut layer_grads: Vec<(BlockWeightGrad, GpuTensor, GpuTensor)> =
        Vec::with_capacity(d.layers.len());
    let mut d_x = d_x_last;
    for i in (0..d.layers.len()).rev() {
        let (lw, ll) = &d.layers[i];
        let (bw, bl) = block_views(lw, ll);
        let (d_in, lora_g, wg) = block_backward_full(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &bw,
            &bl,
            &acts.layer_acts[i],
            &d.dims,
        )?;
        gpu.free_tensor(d_x)?; // consumed; free before reassigning
                               // we only train norms (dnorm1/dnorm2); LoRA grads are unused → free.
        let crate::block::BlockLoraGrad {
            daq,
            dbq,
            dav,
            dbv,
            dnorm1,
            dnorm2,
        } = lora_g;
        for t in [daq, dbq, dav, dbv] {
            gpu.free_tensor(t)?;
        }
        layer_grads.push((wg, dnorm1, dnorm2));
        d_x = d_in;
    }
    layer_grads.reverse();

    // in_proj backward: d_x is grad w.r.t. in_proj output [seq*hd]; embed frozen.
    let d_in_proj = gpu.zeros(&[hd * h_t], DType::F32)?;
    linear_backward_w(gpu, &d_x, &acts.emb, &d_in_proj, seq, h_t, hd, false)?;
    gpu.free_tensor(d_x)?;

    Ok(DrafterGrads {
        d_in_proj,
        layers: layer_grads,
        d_out_norm,
        d_wk_score,
    })
}

/// Forward the drafter and return the LAST block's post-rope K (`[seq*kv_dim]`),
/// which PFlash scores via cosine(block_mean_K, last_token_K).
pub fn drafter_forward(
    gpu: &mut Gpu,
    d: &Drafter,
    token_ids: &[u32],
    pos_host: &[f32],
) -> HipResult<GpuTensor> {
    let (seq, hd, h_t) = (d.dims.seq, d.dims.h, d.h_t);
    assert_eq!(token_ids.len(), seq);

    // embedding lookup at target width → [seq*h_t]
    let emb = gpu.zeros(&[seq * h_t], DType::F32)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        gpu.strided_copy_2d(
            &d.embed,
            tok as usize * h_t,
            h_t,
            &emb,
            t * h_t,
            h_t,
            1,
            h_t,
            false,
        )?;
    }
    // input projection h_t → h_draft
    let mut x = gpu.zeros(&[seq * hd], DType::F32)?;
    linear_forward(gpu, &emb, &d.in_proj, &x, seq, h_t, hd)?;

    // small blocks; keep last block's K
    let mut last_k: Option<GpuTensor> = None;
    for (lw, ll) in &d.layers {
        let bw = BlockWeights {
            norm1: &lw.norm1,
            wq: &lw.wq,
            wk: &lw.wk,
            wv: &lw.wv,
            wo: &lw.wo,
            norm2: &lw.norm2,
            wgate: &lw.wgate,
            wup: &lw.wup,
            wdown: &lw.wdown,
        };
        let bl = BlockLora {
            aq: &ll.aq,
            bq: &ll.bq,
            av: &ll.av,
            bv: &ll.bv,
        };
        let (x_out, acts) = block_forward(gpu, &x, &bw, &bl, &d.dims, pos_host)?;
        last_k = Some(acts.k_r);
        x = x_out;
    }
    Ok(last_k.expect("drafter must have ≥1 layer"))
}
