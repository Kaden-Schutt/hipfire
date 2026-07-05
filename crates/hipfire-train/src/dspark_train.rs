// SPDX-License-Identifier: Apache-2.0
//! DSpark drafter TRAINING LOOP (T5a) — ties the landed pieces together.
//!
//! Consumes a `DSLB v1` label cache (written by `examples/dspark_labels.rs`) and
//! trains a 5-layer DSpark drafter (body + markov/confidence heads) against the
//! frozen target's soft logits + hard next tokens. The moving parts, all already
//! landed and gradchecked, are wired here:
//!
//!   * body + heads fwd/bwd — [`crate::dspark_drafter`]
//!     (`dspark_drafter_forward_full` / `dspark_drafter_backward_full`).
//!   * loss — [`crate::dspark_loss`] (`dspark_loss_forward_backward`).
//!   * optimizer — [`crate::optim::AdamW`].
//!   * frozen shared embed + lm-head — [`crate::loader::load_llama_from_hfq`]:
//!     the block-token embeddings the drafter consumes are gathered from the
//!     target's `embed_tokens`; the head's base logits use the target's
//!     `lm_head` (or `embed_tokens` when tied). Neither receives a gradient.
//!
//! Export to `.dspark.hfq` is a SEPARATE follow-up; this module stops at a
//! resumable `DSCK` checkpoint (drafter weights + AdamW moments).
//!
//! Per training micro-step (one window) the tensor lifetimes are hand-managed —
//! `GpuTensor` has no `Drop`, so every activation / grad / uploaded input is
//! returned to the pool each step or the run OOMs. `main_hidden` and the target
//! `lm_head` are the only inputs kept alive across the forward→backward boundary
//! (both are read again by the backward).

use crate::config::LlamaConfig;
use crate::dspark_drafter::{
    dspark_drafter_backward_full, dspark_drafter_forward_full, free_dspark_drafter_grads,
    free_dspark_full_acts, free_dspark_heads_grads, DsparkDrafterConfig, DsparkDrafterWeights,
    DsparkFullActs, DsparkFullWeights, DsparkHeadsConfig, DsparkHeadsWeights, DsparkLayerWeights,
};
use crate::dspark_loss::{dspark_loss_forward_backward, DsparkLossCfg, DsparkLossOut};
use crate::loader::LlamaWeightsF32;
use crate::optim::AdamW;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use std::io::{self, Read, Write};

// ═══════════════════════════════════════════════════════════════════════════
// Random init helpers (deterministic LCG — mirrors src/drafter.rs)
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic LCG pseudo-random fill in `[-scale, scale)`.
fn rand_fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
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

/// Kaiming-ish linear `[out, in]` weight: `U(-1,1)/sqrt(fan_in)`.
fn lin(gpu: &mut Gpu, out: usize, inn: usize, seed: u64) -> HipResult<GpuTensor> {
    let scale = 1.0 / (inn as f32).sqrt();
    gpu.upload_f32(&rand_fill(out * inn, seed, scale), &[out, inn])
}

/// All-ones vector `[n]` (rmsnorm gains initialise to 1).
fn ones(gpu: &mut Gpu, n: usize) -> HipResult<GpuTensor> {
    gpu.upload_f32(&vec![1.0f32; n], &[n])
}

/// Upload a host i32 index as a Raw device tensor (`rq_gather_f32` reads the
/// buffer as `i32*`; the dtype tag is unused by that kernel). Mirrors the
/// private helper in `dspark_drafter.rs`.
fn upload_idx_i32(gpu: &Gpu, data: &[i32], n: usize) -> HipResult<GpuTensor> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.upload_raw(bytes, &[n])
}

/// Gather the frozen block-token embeddings `[block*h]` from `embed` `[vocab,h]`
/// via the RoughQuant row-gather element mover (`idx[b*h+j] = tok[b]*h + j`).
fn embed_block_tokens(
    gpu: &mut Gpu,
    embed: &GpuTensor,
    block_tokens: &[u32],
    h: usize,
) -> HipResult<GpuTensor> {
    let block = block_tokens.len();
    let mut idx = vec![0i32; block * h];
    for b in 0..block {
        let base = block_tokens[b] as i32 * h as i32;
        for j in 0..h {
            idx[b * h + j] = base + j as i32;
        }
    }
    let idx_t = upload_idx_i32(gpu, &idx, block * h)?;
    let out = gpu.zeros(&[block * h], DType::F32)?;
    gpu.rq_gather_f32(embed, &idx_t, &out, block * h, block * h)?;
    gpu.free_tensor(idx_t)?;
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Drafter model (weights + geometry) and its random init
// ═══════════════════════════════════════════════════════════════════════════

/// Overridable drafter hyperparameters not fixed by the target geometry.
#[derive(Clone, Copy)]
pub struct DsparkDrafterCfg {
    /// Drafter depth (DSpark qwen3 drafter is 5).
    pub n_layers: usize,
    /// VanillaMarkov low-rank width.
    pub markov_rank: usize,
    /// Per-head q/k rmsnorm before RoPE (true for the qwen3 drafter).
    pub qk_norm: bool,
    /// Init seed.
    pub seed: u64,
}

impl Default for DsparkDrafterCfg {
    fn default() -> Self {
        Self {
            n_layers: 5,
            markov_rank: 256,
            qk_norm: true,
            seed: 0xD5_9A_12_34,
        }
    }
}

/// A trainable drafter: full weights + the two configs the fwd/bwd need. The
/// drafter hidden width equals the target hidden dim (it ingests target hidden
/// states and shares the target embed/lm-head, both `[vocab, h]`).
pub struct DsparkModel {
    pub weights: DsparkFullWeights,
    pub body_cfg: DsparkDrafterConfig,
    pub heads_cfg: DsparkHeadsConfig,
}

impl DsparkModel {
    /// AdamW param element counts, in `weights.params()` order (body then heads).
    pub fn param_sizes(&self) -> Vec<usize> {
        self.weights.param_sizes()
    }
}

/// Random-init a drafter at the target's geometry. `n_targets` / `block_size`
/// come from the DSLB header; `markov_rank` / depth from `dcfg`.
pub fn init_dspark_model(
    gpu: &mut Gpu,
    target_cfg: &LlamaConfig,
    n_targets: usize,
    block_size: usize,
    dcfg: &DsparkDrafterCfg,
) -> HipResult<DsparkModel> {
    let h = target_cfg.hidden_size;
    let n_heads = target_cfg.num_attention_heads;
    let n_kv = target_cfg.num_key_value_heads;
    let head_dim = target_cfg.head_dim;
    let inter = target_cfg.intermediate_size;
    let vocab = target_cfg.vocab_size;
    let qd = n_heads * head_dim;
    let kvd = n_kv * head_dim;

    let body_cfg = DsparkDrafterConfig {
        h,
        n_layers: dcfg.n_layers,
        n_heads,
        n_kv,
        head_dim,
        inter,
        rope_base: target_cfg.rope_theta,
        eps: target_cfg.rms_norm_eps,
        block_size,
        n_targets,
        qk_norm: dcfg.qk_norm,
        vocab,
    };
    let heads_cfg = DsparkHeadsConfig::from_drafter(&body_cfg, dcfg.markov_rank);

    // ── body ────────────────────────────────────────────────────────────────
    let fin = n_targets * h;
    let fc = lin(gpu, h, fin, dcfg.seed ^ 0xF00D)?; // main_proj [h, n_targets*h]
    let hidden_norm = ones(gpu, h)?;
    let out_norm = ones(gpu, h)?;
    let mut layers = Vec::with_capacity(dcfg.n_layers);
    for li in 0..dcfg.n_layers {
        let s = dcfg.seed ^ (0x1000u64.wrapping_mul(li as u64 + 1));
        layers.push(DsparkLayerWeights {
            input_ln: ones(gpu, h)?,
            wq: lin(gpu, qd, h, s + 1)?,
            wk: lin(gpu, kvd, h, s + 2)?,
            wv: lin(gpu, kvd, h, s + 3)?,
            wo: lin(gpu, h, qd, s + 4)?,
            q_norm: ones(gpu, head_dim)?,
            k_norm: ones(gpu, head_dim)?,
            post_ln: ones(gpu, h)?,
            wgate: lin(gpu, inter, h, s + 5)?,
            wup: lin(gpu, inter, h, s + 6)?,
            wdown: lin(gpu, h, inter, s + 7)?,
        });
    }
    let body = DsparkDrafterWeights {
        fc,
        hidden_norm,
        layers,
        out_norm,
    };

    // ── heads ─────────────────────────────────────────────────────────────────
    // markov weights small (do not dominate the shared-lm-head base logits);
    // confidence proj small, confidence bias zero.
    let r = dcfg.markov_rank;
    let markov_w1 = gpu.upload_f32(&rand_fill(vocab * r, dcfg.seed ^ 0xA0, 0.02), &[vocab, r])?;
    let markov_w2 = gpu.upload_f32(&rand_fill(vocab * r, dcfg.seed ^ 0xB0, 0.02), &[vocab, r])?;
    let confidence_proj = lin(gpu, 1, h + r, dcfg.seed ^ 0xC0)?;
    let confidence_bias = gpu.zeros(&[1], DType::F32)?;
    let heads = DsparkHeadsWeights {
        markov_w1,
        markov_w2,
        confidence_proj,
        confidence_bias,
    };

    Ok(DsparkModel {
        weights: DsparkFullWeights { body, heads },
        body_cfg,
        heads_cfg,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// DSLB v1 label-cache loader
// ═══════════════════════════════════════════════════════════════════════════

/// One training window from the DSLB cache (host side; uploaded per step).
pub struct DsparkWindow {
    pub main_hidden: Vec<f32>,   // [ctx_len * n_targets * dim]
    pub target_logits: Vec<f32>, // [block * vocab]
    pub next_tokens: Vec<i32>,   // [block] (-100 = invalid / ignore_index)
    pub block_tokens: Vec<u32>,  // [block]
    pub prev_tokens: Vec<u32>,   // [block]
    pub eval_mask: Vec<u8>,      // [block]
}

/// Parsed DSLB v1 label cache (see `examples/dspark_labels.rs` for the format).
/// Frozen embed + lm-head are NOT stored here — reload them from `target_path`.
pub struct DsparkLabelCache {
    pub vocab: usize,
    pub dim: usize,
    pub n_targets: usize,
    pub block: usize,
    pub ctx_len: usize,
    pub target_layer_ids: Vec<usize>,
    pub target_path: String,
    pub windows: Vec<DsparkWindow>,
}

impl DsparkLabelCache {
    pub fn n_windows(&self) -> usize {
        self.windows.len()
    }
    pub fn window(&self, i: usize) -> &DsparkWindow {
        &self.windows[i]
    }
    /// Keep only the first `k` windows (used by `--overfit`).
    pub fn truncate(&mut self, k: usize) {
        self.windows.truncate(k);
    }
}

/// Little-endian byte cursor over the whole cache (read once into memory — DSLB
/// caches for overfit / small training runs are modest, and a single read keeps
/// the parser simple and obviously correct).
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cur<'a> {
    fn u32(&mut self) -> io::Result<u32> {
        let e = self.p + 4;
        let s = self
            .b
            .get(self.p..e)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "DSLB truncated"))?;
        self.p = e;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let e = self.p + n;
        let s = self
            .b
            .get(self.p..e)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "DSLB truncated"))?;
        self.p = e;
        Ok(s)
    }
    fn f32s(&mut self, n: usize) -> io::Result<Vec<f32>> {
        let raw = self.bytes(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    fn i32s(&mut self, n: usize) -> io::Result<Vec<i32>> {
        let raw = self.bytes(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    fn u32s(&mut self, n: usize) -> io::Result<Vec<u32>> {
        let raw = self.bytes(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// Load a `DSLB v1` label cache.
pub fn load_dslb(path: &str) -> io::Result<DsparkLabelCache> {
    let bytes = std::fs::read(path)?;
    let mut c = Cur { b: &bytes, p: 0 };
    let magic = c.bytes(4)?;
    if magic != b"DSLB" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad DSLB magic"));
    }
    let version = c.u32()?;
    if version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported DSLB version {version} (want 1)"),
        ));
    }
    let vocab = c.u32()? as usize;
    let dim = c.u32()? as usize;
    let n_targets = c.u32()? as usize;
    let block = c.u32()? as usize;
    let ctx_len = c.u32()? as usize;
    let _flags = c.u32()?;
    let n_windows = c.u32()? as usize;
    let k = c.u32()? as usize; // target_layer_ids count
    let target_layer_ids: Vec<usize> = (0..k)
        .map(|_| c.u32().map(|x| x as usize))
        .collect::<io::Result<_>>()?;
    let path_len = c.u32()? as usize;
    let target_path = String::from_utf8_lossy(c.bytes(path_len)?).into_owned();

    let main_len = ctx_len * n_targets * dim;
    let logits_len = block * vocab;
    let mut windows = Vec::with_capacity(n_windows);
    for _ in 0..n_windows {
        let main_hidden = c.f32s(main_len)?;
        let target_logits = c.f32s(logits_len)?;
        let next_tokens = c.i32s(block)?;
        let block_tokens = c.u32s(block)?;
        let prev_tokens = c.u32s(block)?;
        let eval_mask = c.bytes(block)?.to_vec();
        windows.push(DsparkWindow {
            main_hidden,
            target_logits,
            next_tokens,
            block_tokens,
            prev_tokens,
            eval_mask,
        });
    }

    Ok(DsparkLabelCache {
        vocab,
        dim,
        n_targets,
        block,
        ctx_len,
        target_layer_ids,
        target_path,
        windows,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Training config / report
// ═══════════════════════════════════════════════════════════════════════════

/// DSpark training hyperparameters.
#[derive(Clone, Copy)]
pub struct DsparkTrainCfg {
    pub epochs: usize,
    pub lr: f32,
    pub weight_decay: f32,
    /// Draft positions per block (must equal the cache block size).
    pub block_size: usize,
    /// Loss weights + positional decay.
    pub loss: DsparkLossCfg,
    /// Fraction of the (tail) windows held out for eval.
    pub eval_frac: f32,
    /// Write a checkpoint on best-eval epochs whose index is a multiple of this.
    pub checkpoint_every: usize,
    pub seed: u64,
}

impl Default for DsparkTrainCfg {
    fn default() -> Self {
        Self {
            epochs: 100,
            lr: 1e-3,
            weight_decay: 0.0,
            block_size: 7,
            loss: DsparkLossCfg::default(),
            eval_frac: 0.1,
            checkpoint_every: 10,
            seed: 0,
        }
    }
}

/// Outcome of a training run.
pub struct DsparkTrainReport {
    /// Lowest mean eval loss seen.
    pub best_eval_loss: f32,
    pub best_epoch: usize,
    /// Acceptance-rate proxy (`1 - 0.5*mean_l1`) at the best-eval epoch.
    pub best_accept: f32,
    pub final_train_loss: f32,
    pub final_eval_loss: f32,
    /// Host snapshot of `weights.params()` (in order) at the best-eval epoch.
    pub best_weights: Vec<Vec<f32>>,
}

// ═══════════════════════════════════════════════════════════════════════════
// One window: forward + loss (shared by train & eval)
// ═══════════════════════════════════════════════════════════════════════════

/// Live tensors of one window's forward+loss. `main_hidden` and the loss grads
/// are kept alive for the backward; everything is returned to the pool by
/// `free_step` (eval) or after the backward+step (train).
struct Step {
    acts: DsparkFullActs,
    loss: DsparkLossOut,
    main_hidden: GpuTensor,
    target_logits: GpuTensor,
    next_tokens: GpuTensor,
    eval_mask: GpuTensor,
}

#[allow(clippy::too_many_arguments)]
fn forward_loss(
    gpu: &mut Gpu,
    model: &DsparkModel,
    embed: &GpuTensor,
    lm_head: &GpuTensor,
    ctx_pos: &[f32],
    block_pos: &[f32],
    win: &DsparkWindow,
    cache: &DsparkLabelCache,
    loss_cfg: &DsparkLossCfg,
) -> HipResult<Step> {
    let h = model.body_cfg.h;
    let block = cache.block;
    let vocab = cache.vocab;
    let main_len = cache.ctx_len * cache.n_targets * cache.dim;

    let main_hidden = gpu.upload_f32(&win.main_hidden, &[main_len])?;
    let block_embeds = embed_block_tokens(gpu, embed, &win.block_tokens, h)?;

    let acts = dspark_drafter_forward_full(
        gpu,
        &model.weights,
        &model.body_cfg,
        &model.heads_cfg,
        &main_hidden,
        &block_embeds,
        &win.prev_tokens,
        lm_head,
        ctx_pos,
        block_pos,
        None,
    )?;
    // block_embeds is cloned inside the forward → safe to reclaim now.
    gpu.free_tensor(block_embeds)?;

    let target_logits = gpu.upload_f32(&win.target_logits, &[block * vocab])?;
    let next_f32: Vec<f32> = win.next_tokens.iter().map(|&t| t as f32).collect();
    let next_tokens = gpu.upload_f32(&next_f32, &[block])?;
    let mask_f32: Vec<f32> = win.eval_mask.iter().map(|&m| m as f32).collect();
    let eval_mask = gpu.upload_f32(&mask_f32, &[block])?;

    let out = acts.output();
    // NOTE: the loss operates on the confidence LOGIT (BCE-with-logits) and
    // returns d/d(logit); the head's confidence path is linear from that logit.
    let loss = dspark_loss_forward_backward(
        gpu,
        out.draft_logits,
        &acts.heads.confidence_logit,
        &target_logits,
        &next_tokens,
        &eval_mask,
        loss_cfg,
    )?;

    Ok(Step {
        acts,
        loss,
        main_hidden,
        target_logits,
        next_tokens,
        eval_mask,
    })
}

/// Return a step's tensors to the pool WITHOUT running the backward (eval path).
fn free_step(gpu: &mut Gpu, step: Step) -> HipResult<()> {
    let Step {
        acts,
        loss,
        main_hidden,
        target_logits,
        next_tokens,
        eval_mask,
    } = step;
    free_dspark_full_acts(gpu, acts)?;
    for t in [
        loss.d_draft_logits,
        loss.d_confidence_logit,
        main_hidden,
        target_logits,
        next_tokens,
        eval_mask,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Training loop
// ═══════════════════════════════════════════════════════════════════════════

/// Train a DSpark drafter against a DSLB cache. `target` supplies the frozen
/// shared embed + lm-head (lm-head falls back to the tied embedding). The eval
/// split is the LAST `eval_frac` windows (caller is expected to have ordered /
/// shuffled the cache). `on_epoch(epoch, train_loss, eval_loss, best_eval,
/// best_epoch, accept)` fires every epoch. When `ckpt_path` is set, a `DSCK`
/// checkpoint is written on best-eval epochs at the `checkpoint_every` cadence.
///
/// The drafter weights are left at their FINAL state; the returned
/// `best_weights` host snapshot is the generalizing model to persist.
#[allow(clippy::too_many_arguments)]
pub fn train_dspark_loop(
    gpu: &mut Gpu,
    model: &DsparkModel,
    target: &LlamaWeightsF32,
    cache: &DsparkLabelCache,
    opt: &mut AdamW,
    cfg: &DsparkTrainCfg,
    ckpt_path: Option<&str>,
    mut on_epoch: impl FnMut(usize, f32, f32, f32, usize, f32),
) -> HipResult<DsparkTrainReport> {
    let embed = &target.embed_tokens;
    let lm_head = target.lm_head.as_ref().unwrap_or(&target.embed_tokens);

    let n = cache.n_windows();
    assert!(n > 0, "dspark train: empty label cache");
    let n_eval = ((n as f32) * cfg.eval_frac).round() as usize;
    let n_eval = n_eval.min(n.saturating_sub(1)); // always keep >=1 train window
    let n_train = n - n_eval;

    // RoPE positions: ctx = 0..ctx_len, block = ctx_len..ctx_len+block.
    let ctx_pos: Vec<f32> = (0..cache.ctx_len).map(|p| p as f32).collect();
    let block_pos: Vec<f32> = (0..cache.block)
        .map(|p| (cache.ctx_len + p) as f32)
        .collect();

    // Loss cfg with the cache's true block size (positional decay depends on it).
    let mut loss_cfg = cfg.loss;
    loss_cfg.block_size = cache.block;

    let mut best_eval_loss = f32::INFINITY;
    let mut best_epoch = 0usize;
    let mut best_accept = 0.0f32;
    let mut final_train_loss = 0.0f32;
    let mut final_eval_loss = 0.0f32;
    let mut best_weights: Vec<Vec<f32>> = Vec::new();

    for ep in 0..cfg.epochs {
        // ── train split ─────────────────────────────────────────────────────
        let mut train_loss = 0.0f32;
        for i in 0..n_train {
            let step = forward_loss(
                gpu,
                model,
                embed,
                lm_head,
                &ctx_pos,
                &block_pos,
                cache.window(i),
                cache,
                &loss_cfg,
            )?;
            train_loss += step.loss.total;

            // backward from the two loss seeds → grads → AdamW step.
            let grads = dspark_drafter_backward_full(
                gpu,
                &model.weights,
                &model.body_cfg,
                &model.heads_cfg,
                &step.main_hidden,
                lm_head,
                &step.acts,
                &step.loss.d_draft_logits,
                &step.loss.d_confidence_logit,
            )?;
            opt.step(gpu, &model.weights.params(), &grads.flat())?;
            free_dspark_drafter_grads(gpu, grads.body)?;
            free_dspark_heads_grads(gpu, grads.heads)?;
            free_step(gpu, step)?;
        }
        let train_loss = train_loss / n_train as f32;
        final_train_loss = train_loss;

        // ── eval split (fallback to train windows when eval_frac==0) ─────────
        let (estart, eend) = if n_eval > 0 {
            (n_train, n)
        } else {
            (0, n_train)
        };
        let mut eval_loss = 0.0f32;
        let mut eval_l1 = 0.0f32;
        for i in estart..eend {
            let step = forward_loss(
                gpu,
                model,
                embed,
                lm_head,
                &ctx_pos,
                &block_pos,
                cache.window(i),
                cache,
                &loss_cfg,
            )?;
            eval_loss += step.loss.total;
            eval_l1 += step.loss.l1;
            free_step(gpu, step)?;
        }
        let ecount = (eend - estart).max(1) as f32;
        let eval_loss = eval_loss / ecount;
        let accept = (1.0 - 0.5 * (eval_l1 / ecount)).clamp(0.0, 1.0);
        final_eval_loss = eval_loss;

        if eval_loss < best_eval_loss {
            best_eval_loss = eval_loss;
            best_epoch = ep;
            best_accept = accept;
            best_weights = model
                .weights
                .params()
                .iter()
                .map(|t| gpu.download_f32(t))
                .collect::<HipResult<_>>()?;
            if let Some(path) = ckpt_path {
                let every = cfg.checkpoint_every.max(1);
                if ep % every == 0 {
                    save_dspark_ckpt(gpu, path, model, opt, ep as u32)?;
                }
            }
        }

        on_epoch(
            ep,
            train_loss,
            eval_loss,
            best_eval_loss,
            best_epoch,
            accept,
        );
    }

    Ok(DsparkTrainReport {
        best_eval_loss,
        best_epoch,
        best_accept,
        final_train_loss,
        final_eval_loss,
        best_weights,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Checkpoint (DSCK) — weights + AdamW moments, for resume
// ═══════════════════════════════════════════════════════════════════════════

const DSCK_MAGIC: &[u8; 4] = b"DSCK";

fn wu32(w: &mut impl Write, x: u32) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn wi32(w: &mut impl Write, x: i32) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn wvec(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    wu32(w, v.len() as u32)?;
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    w.write_all(bytes)
}
fn ru32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn ri32(r: &mut impl Read) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn rvec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let n = ru32(r)? as usize;
    let mut buf = vec![0u8; n * 4];
    r.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn io_err(e: io::Error) -> hipfire_rdna::HipError {
    hipfire_rdna::HipError {
        code: u32::MAX,
        message: format!("dspark checkpoint io: {e}"),
    }
}

/// Save drafter weights (`params()` order) + AdamW moments + epoch. Atomic
/// (`.tmp` then rename), mirroring the `PFDC` drafter checkpoint.
pub fn save_dspark_ckpt(
    gpu: &mut Gpu,
    path: &str,
    model: &DsparkModel,
    opt: &AdamW,
    epoch: u32,
) -> HipResult<()> {
    let params = model.weights.params();
    let weights: Vec<Vec<f32>> = params
        .iter()
        .map(|t| gpu.download_f32(t))
        .collect::<HipResult<_>>()?;
    let (m, v, t) = opt.save_state(gpu)?;
    let tmp = format!("{path}.tmp");
    let mut f = io::BufWriter::new(std::fs::File::create(&tmp).map_err(io_err)?);
    (|| -> io::Result<()> {
        f.write_all(DSCK_MAGIC)?;
        wu32(&mut f, 1)?; // version
        wu32(&mut f, epoch)?;
        wu32(&mut f, weights.len() as u32)?;
        for w in &weights {
            wvec(&mut f, w)?;
        }
        wi32(&mut f, t)?;
        for x in &m {
            wvec(&mut f, x)?;
        }
        for x in &v {
            wvec(&mut f, x)?;
        }
        f.flush()
    })()
    .map_err(io_err)?;
    std::fs::rename(&tmp, path).map_err(io_err)?;
    Ok(())
}

/// Load weights + AdamW state into an already-constructed model/optimizer of the
/// SAME geometry. Returns the saved epoch, or `None` if the file is absent.
pub fn load_dspark_ckpt(
    gpu: &mut Gpu,
    path: &str,
    model: &DsparkModel,
    opt: &mut AdamW,
) -> HipResult<Option<u32>> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut f = io::BufReader::new(file);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(io_err)?;
    if &magic != DSCK_MAGIC {
        return Err(io_err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad DSCK magic",
        )));
    }
    let _ver = ru32(&mut f).map_err(io_err)?;
    let epoch = ru32(&mut f).map_err(io_err)?;
    let np = ru32(&mut f).map_err(io_err)? as usize;
    let weights: Vec<Vec<f32>> = (0..np)
        .map(|_| rvec(&mut f))
        .collect::<io::Result<_>>()
        .map_err(io_err)?;
    let t = ri32(&mut f).map_err(io_err)?;
    let m: Vec<Vec<f32>> = (0..np)
        .map(|_| rvec(&mut f))
        .collect::<io::Result<_>>()
        .map_err(io_err)?;
    let v: Vec<Vec<f32>> = (0..np)
        .map(|_| rvec(&mut f))
        .collect::<io::Result<_>>()
        .map_err(io_err)?;

    let params = model.weights.params();
    let sizes = model.weights.param_sizes();
    assert_eq!(weights.len(), params.len(), "DSCK param count mismatch");
    for (i, w) in weights.iter().enumerate() {
        assert_eq!(w.len(), sizes[i], "DSCK param[{i}] size mismatch");
        gpu.memcpy_htod_auto(&params[i].buf, bytemuck_f32(w))?;
    }
    opt.load_state(gpu, &m, &v, t)?;
    Ok(Some(epoch))
}

/// Upload a host param snapshot (from `DsparkTrainReport::best_weights`) into the
/// live model params, in `params()` order. Used to persist the BEST model.
pub fn load_weights_into(
    gpu: &mut Gpu,
    model: &DsparkModel,
    weights: &[Vec<f32>],
) -> HipResult<()> {
    let params = model.weights.params();
    let sizes = model.weights.param_sizes();
    assert_eq!(
        weights.len(),
        params.len(),
        "best-weights param count mismatch"
    );
    for (i, w) in weights.iter().enumerate() {
        assert_eq!(w.len(), sizes[i], "best-weights param[{i}] size mismatch");
        gpu.memcpy_htod_auto(&params[i].buf, bytemuck_f32(w))?;
    }
    Ok(())
}
