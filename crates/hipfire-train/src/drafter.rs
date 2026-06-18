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

use crate::block::{block_forward, BlockDims, BlockLora, BlockWeights};
use crate::model::{LayerLora, LayerWeights};
use crate::ops::linear::linear_forward;
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

/// Shape/size hyperparameters for the drafter body (independent of the target).
#[derive(Clone, Copy)]
pub struct DrafterConfig {
    pub h_draft: usize,   // drafter hidden width (≪ target)
    pub n_layers: usize,  // small (2–4)
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
    pub dims: BlockDims, // h = h_draft
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
                aq: gpu.upload_f32(&rand_fill(r * hd, s + 8, 1.0 / (hd as f32).sqrt()), &[r * hd])?,
                bq: gpu.zeros(&[qd * r], DType::F32)?,
                av: gpu.upload_f32(&rand_fill(r * hd, s + 9, 1.0 / (hd as f32).sqrt()), &[r * hd])?,
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

        Ok(Drafter { embed, h_t, vocab, in_proj, layers, dims })
    }
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
        gpu.strided_copy_2d(&d.embed, tok as usize * h_t, h_t, &emb, t * h_t, h_t, 1, h_t, false)?;
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
        let bl = BlockLora { aq: &ll.aq, bq: &ll.bq, av: &ll.av, bv: &ll.bv };
        let (x_out, acts) = block_forward(gpu, &x, &bw, &bl, &d.dims, pos_host)?;
        last_k = Some(acts.k_r);
        x = x_out;
    }
    Ok(last_k.expect("drafter must have ≥1 layer"))
}
