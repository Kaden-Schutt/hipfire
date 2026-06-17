// SPDX-License-Identifier: Apache-2.0
//! Full dense LLaMA model (fp32) for training: embedding → N pre-norm blocks →
//! final RMSNorm → tied-logit projection → cross-entropy. LoRA on every layer's
//! q_proj/v_proj; all base weights (incl. embedding / tied lm_head) frozen.
//!
//! Built on the gradchecked `block` module — this layer is the embed/head/loss
//! bookends plus the block loop and its reverse.

use crate::block::{
    block_backward, block_forward, BlockActivations, BlockDims, BlockLora, BlockLoraGrad,
    BlockWeights,
};
use crate::config::LlamaConfig;
use crate::loader::LlamaWeightsF32;
use crate::ops::cross_entropy::cross_entropy;
use crate::ops::distill::distill_kl;
use crate::ops::linear::{linear_backward_x, linear_forward};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

/// Owned frozen weights for one layer.
pub struct LayerWeights {
    pub norm1: GpuTensor,
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
    pub norm2: GpuTensor,
    pub wgate: GpuTensor,
    pub wup: GpuTensor,
    pub wdown: GpuTensor,
}

impl LayerWeights {
    fn as_block(&self) -> BlockWeights<'_> {
        BlockWeights {
            norm1: &self.norm1,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            norm2: &self.norm2,
            wgate: &self.wgate,
            wup: &self.wup,
            wdown: &self.wdown,
        }
    }
}

/// Owned trainable LoRA params for one layer.
pub struct LayerLora {
    pub aq: GpuTensor,
    pub bq: GpuTensor,
    pub av: GpuTensor,
    pub bv: GpuTensor,
}

impl LayerLora {
    fn as_block(&self) -> BlockLora<'_> {
        BlockLora {
            aq: &self.aq,
            bq: &self.bq,
            av: &self.av,
            bv: &self.bv,
        }
    }
}

pub struct LlamaModel {
    pub embed: GpuTensor,      // [vocab, h] (also tied lm_head)
    pub final_norm: GpuTensor, // [h]
    pub layers: Vec<(LayerWeights, LayerLora)>,
    pub dims: BlockDims,
    pub vocab: usize,
}

impl LlamaModel {
    /// Build a trainable model from loaded fp32 base weights. Base weights are
    /// frozen; LoRA adapters are created on q_proj/v_proj of every layer with
    /// `A` small-random and `B = 0` (so the initial LoRA contribution is zero
    /// and the model starts exactly at the base). `seq` is the (fixed) training
    /// sequence length used to populate `BlockDims`. Requires tied embeddings.
    pub fn from_f32_weights(
        gpu: &mut Gpu,
        cfg: &LlamaConfig,
        w: LlamaWeightsF32,
        seq: usize,
        lora_rank: usize,
        lora_alpha: f32,
    ) -> HipResult<Self> {
        assert!(
            w.lm_head.is_none(),
            "from_f32_weights requires tied embeddings"
        );
        let h = cfg.hidden_size;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let r = lora_rank;

        // Deterministic small-random A init (Kaiming-ish, scaled); B = 0.
        let a_init = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x =
                        ((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32 / 1000.0;
                    (x - 0.5) * (2.0 / h as f32).sqrt()
                })
                .collect()
        };

        let mut layers = Vec::with_capacity(w.layers.len());
        for (li, lw) in w.layers.into_iter().enumerate() {
            let weights = LayerWeights {
                norm1: lw.input_layernorm,
                wq: lw.q_proj,
                wk: lw.k_proj,
                wv: lw.v_proj,
                wo: lw.o_proj,
                norm2: lw.post_attention_layernorm,
                wgate: lw.gate_proj,
                wup: lw.up_proj,
                wdown: lw.down_proj,
            };
            let lora = LayerLora {
                aq: gpu.upload_f32(&a_init(r * h, li * 7 + 1), &[r * h])?,
                bq: gpu.zeros(&[qd * r], DType::F32)?,
                av: gpu.upload_f32(&a_init(r * h, li * 7 + 3), &[r * h])?,
                bv: gpu.zeros(&[kvd * r], DType::F32)?,
            };
            layers.push((weights, lora));
        }

        let dims = BlockDims {
            seq,
            h,
            n_heads: cfg.num_attention_heads,
            n_kv: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            inter: cfg.intermediate_size,
            rope_base: cfg.rope_theta,
            eps: cfg.rms_norm_eps,
            lora_scale: lora_alpha / r as f32,
            lora_rank: r,
        };

        Ok(Self {
            embed: w.embed_tokens,
            final_norm: w.final_norm,
            layers,
            dims,
            vocab: cfg.vocab_size,
        })
    }

    /// Flat list of trainable LoRA params, layer-major `[aq,bq,av,bv]` — the
    /// order the optimizer and `model_loss_backward`'s grads use.
    pub fn lora_params(&self) -> Vec<&GpuTensor> {
        let mut v = Vec::with_capacity(self.layers.len() * 4);
        for (_, l) in &self.layers {
            v.push(&l.aq);
            v.push(&l.bq);
            v.push(&l.av);
            v.push(&l.bv);
        }
        v
    }

    /// Element counts of `lora_params()`, for `AdamW::new`.
    pub fn lora_param_sizes(&self) -> Vec<usize> {
        self.lora_params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }

    /// Trainable params for QTIP recovery FT: per layer `[aq,bq,av,bv,norm1,
    /// norm2]`, then the final norm. Matches `flatten_recovery_grads` order.
    pub fn recovery_params(&self) -> Vec<&GpuTensor> {
        let mut v = Vec::with_capacity(self.layers.len() * 6 + 1);
        for (w, l) in &self.layers {
            v.push(&l.aq);
            v.push(&l.bq);
            v.push(&l.av);
            v.push(&l.bv);
            v.push(&w.norm1);
            v.push(&w.norm2);
        }
        v.push(&self.final_norm);
        v
    }

    pub fn recovery_param_sizes(&self) -> Vec<usize> {
        self.recovery_params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }
}

/// Flatten recovery grads to match `recovery_params()`: per layer
/// `[daq,dbq,dav,dbv,dnorm1,dnorm2]`, then `d_final_norm`.
pub fn flatten_recovery_grads<'a>(
    grads: &'a [BlockLoraGrad],
    d_final_norm: &'a GpuTensor,
) -> Vec<&'a GpuTensor> {
    let mut v = Vec::with_capacity(grads.len() * 6 + 1);
    for g in grads {
        v.push(&g.daq);
        v.push(&g.dbq);
        v.push(&g.dav);
        v.push(&g.dbv);
        v.push(&g.dnorm1);
        v.push(&g.dnorm2);
    }
    v.push(d_final_norm);
    v
}

/// Saved forward state for the backward pass.
pub struct ModelActivations {
    pub layer_inputs: Vec<GpuTensor>, // input to each block (layer_inputs[0] = embedding)
    pub layer_acts: Vec<BlockActivations>,
    pub x_last: GpuTensor, // output of last block (input to final norm)
    pub rinv_final: GpuTensor,
    pub xf: GpuTensor,     // final-norm output (logit input)
    pub logits: GpuTensor, // [seq, vocab]
}

/// Forward through logits (no loss). `token_ids.len()` must equal `dims.seq`.
pub fn model_forward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    token_ids: &[u32],
    pos_host: &[f32],
) -> HipResult<ModelActivations> {
    let (seq, h) = (model.dims.seq, model.dims.h);
    assert_eq!(token_ids.len(), seq);

    // Embedding lookup: gather row token_ids[t] of embed into x0[t].
    let x0 = gpu.zeros(&[seq * h], DType::F32)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        gpu.strided_copy_2d(
            &model.embed,
            tok as usize * h,
            h,
            &x0,
            t * h,
            h,
            1,
            h,
            false,
        )?;
    }

    let mut layer_inputs = Vec::with_capacity(model.layers.len());
    let mut layer_acts = Vec::with_capacity(model.layers.len());
    let mut x = x0;
    for (lw, ll) in &model.layers {
        layer_inputs.push(clone_tensor(gpu, &x)?);
        let (x_out, acts) = block_forward(
            gpu,
            &x,
            &lw.as_block(),
            &ll.as_block(),
            &model.dims,
            pos_host,
        )?;
        layer_acts.push(acts);
        x = x_out;
    }
    let x_last = x;

    // final norm → logits = xf · embedᵀ (tied)
    let xf = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv_final = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(
        gpu,
        &x_last,
        &model.final_norm,
        &xf,
        &rinv_final,
        seq,
        h,
        model.dims.eps,
    )?;
    let logits = gpu.zeros(&[seq * model.vocab], DType::F32)?;
    linear_forward(gpu, &xf, &model.embed, &logits, seq, h, model.vocab)?;

    Ok(ModelActivations {
        layer_inputs,
        layer_acts,
        x_last,
        rinv_final,
        xf,
        logits,
    })
}

/// Cross-entropy loss (summed over non-ignored tokens) + full backward.
/// Returns the summed loss and per-layer LoRA gradients. `targets` are
/// integer-valued f32 (or `ignore_index`).
pub fn model_loss_backward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    targets: &[f32],
    ignore_index: i32,
) -> HipResult<(f32, Vec<BlockLoraGrad>)> {
    let (seq, h, vocab) = (model.dims.seq, model.dims.h, model.vocab);

    // Loss + d_logits (sum-reduction).
    let tgt = gpu.upload_f32(targets, &[seq])?;
    let loss = gpu.zeros(&[seq], DType::F32)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32)?;
    cross_entropy(
        gpu,
        &acts.logits,
        &tgt,
        &loss,
        &d_logits,
        seq,
        vocab,
        ignore_index,
    )?;
    let loss_sum: f32 = gpu.download_f32(&loss)?.iter().sum();

    // d_xf = d_logits · embed  (tied lm_head frozen)
    let d_xf = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, &model.embed, &d_xf, seq, h, vocab, false)?;
    // final norm backward → d_x_last
    let d_x_last = gpu.zeros(&[seq * h], DType::F32)?;
    let dw_dummy = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xf,
        &acts.x_last,
        &model.final_norm,
        &acts.rinv_final,
        &d_x_last,
        &dw_dummy,
        seq,
        h,
    )?;

    // Walk blocks in reverse.
    let mut grads: Vec<BlockLoraGrad> = Vec::with_capacity(model.layers.len());
    let mut d_x = d_x_last;
    for i in (0..model.layers.len()).rev() {
        let (lw, ll) = &model.layers[i];
        let (d_in, g) = block_backward(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &lw.as_block(),
            &ll.as_block(),
            &acts.layer_acts[i],
            &model.dims,
        )?;
        grads.push(g);
        d_x = d_in;
    }
    grads.reverse(); // align with layer order
    Ok((loss_sum, grads))
}

/// Flatten per-layer LoRA grads into the same order as `LlamaModel::lora_params`.
pub fn flatten_lora_grads(grads: &[BlockLoraGrad]) -> Vec<&GpuTensor> {
    let mut v = Vec::with_capacity(grads.len() * 4);
    for g in grads {
        v.push(&g.daq);
        v.push(&g.dbq);
        v.push(&g.dav);
        v.push(&g.dbv);
    }
    v
}

/// Distillation forward-loss + backward against a teacher distribution
/// `teacher_p` `[seq*vocab]` (probabilities). Returns the summed KL, per-layer
/// grads (LoRA + layernorms), and the final-norm grad. The student's base
/// linears stay frozen.
pub fn model_distill_backward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    teacher_p: &GpuTensor,
) -> HipResult<(f32, Vec<BlockLoraGrad>, GpuTensor)> {
    let (seq, h, vocab) = (model.dims.seq, model.dims.h, model.vocab);

    let loss = gpu.zeros(&[seq], DType::F32)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32)?;
    distill_kl(gpu, &acts.logits, teacher_p, &loss, &d_logits, seq, vocab)?;
    let loss_sum: f32 = gpu.download_f32(&loss)?.iter().sum();

    let d_xf = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, &model.embed, &d_xf, seq, h, vocab, false)?;
    let d_x_last = gpu.zeros(&[seq * h], DType::F32)?;
    let d_final_norm = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xf,
        &acts.x_last,
        &model.final_norm,
        &acts.rinv_final,
        &d_x_last,
        &d_final_norm,
        seq,
        h,
    )?;

    let mut grads: Vec<BlockLoraGrad> = Vec::with_capacity(model.layers.len());
    let mut d_x = d_x_last;
    for i in (0..model.layers.len()).rev() {
        let (lw, ll) = &model.layers[i];
        let (d_in, g) = block_backward(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &lw.as_block(),
            &ll.as_block(),
            &acts.layer_acts[i],
            &model.dims,
        )?;
        grads.push(g);
        d_x = d_in;
    }
    grads.reverse();
    Ok((loss_sum, grads, d_final_norm))
}

fn clone_tensor(gpu: &mut Gpu, t: &GpuTensor) -> HipResult<GpuTensor> {
    let out = gpu.zeros(&t.shape, DType::F32)?;
    let bytes: usize = t.shape.iter().product::<usize>() * 4;
    gpu.memcpy_dtod_auto(&out.buf, &t.buf, bytes)?;
    Ok(out)
}
