// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2 DFlash target/draft bridge.
//!
//! This module owns the reusable pieces proven by the LFM2 seed smoke:
//! validating the target/draft shape contract, staging the target embedding
//! block, running the generic DFlash draft forward, and applying the LFM2 target
//! lm_head over draft rows `1..B`, verifying against the target, and replaying
//! the accepted prefix.
//!
//! It is not the daemon admission path yet: trained sidecars, quality gates, and
//! serving-core integration remain separate work.

use crate::config::Lfm2MoeConfig;
use crate::forward::{prefill_batch, prefill_batch_with_hidden_logits, Lfm2HiddenCapture};
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use hip_bridge::{DeviceBuffer, HipResult};
use hipfire_runtime::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::weights::weight_gemm;
use hipfire_rdna::{DType, Gpu, GpuTensor};

/// LFM2 DFlash is still experimental on the F16 draft GEMM path. Default to the
/// stable F16-on-disk -> F32-on-GPU lift unless explicitly opted in.
pub fn lfm2_dflash_use_f16_weights() -> bool {
    std::env::var("HIPFIRE_LFM2_DFLASH_F16").ok().as_deref() == Some("1")
}

/// Synchronize after each draft GEMM by default for LFM2 DFlash bring-up.
/// Set `HIPFIRE_LFM2_DFLASH_SYNC_GEMM=0` only when validating the async path.
pub fn lfm2_dflash_sync_gemm() -> bool {
    std::env::var("HIPFIRE_LFM2_DFLASH_SYNC_GEMM")
        .ok()
        .as_deref()
        != Some("0")
}

/// Host logits produced by one LFM2 DFlash draft block.
pub struct Lfm2DflashDraftLogits {
    /// Row-major `[batch, vocab_size]`, where `batch = block_size - 1`.
    pub logits: Vec<f32>,
    pub batch: usize,
    pub vocab_size: usize,
}

/// Target verification result for one LFM2 DFlash block.
pub struct Lfm2DflashVerifyOutput {
    /// Row-major `[batch, vocab_size]` target logits after each verified token.
    pub logits_per_pos: Vec<f32>,
    /// Greedy target argmax for each verified position.
    pub argmax_per_pos: Vec<u32>,
    /// Row-major `[batch][extract_layer][hidden]` target-hidden rows captured
    /// from the verified block. Append the accepted prefix of this buffer to
    /// the drafter context after rollback/replay.
    pub target_hidden_rows: Vec<f32>,
    pub batch: usize,
    pub vocab_size: usize,
}

/// Result of one greedy LFM2 DFlash speculative step.
#[derive(Debug, Clone)]
pub struct Lfm2DflashSpecStepResult {
    /// Number of draft tokens accepted, excluding the seed token.
    pub accepted: usize,
    /// Target's prediction at the first rejection point, or after all drafted
    /// tokens if every draft row was accepted.
    pub bonus_token: u32,
    /// Full verified block: `[seed, draft_0, draft_1, ...]`.
    pub drafted: Vec<u32>,
    /// Emitted/committed sequence: `[seed, accepted drafts..., bonus]`.
    pub committed: Vec<u32>,
    /// Number of target positions replayed or retained after verify. This is
    /// `accepted + 1`: the seed plus accepted draft prefix. The bonus is
    /// emitted but becomes the next cycle's seed and is not replayed here.
    pub advance: usize,
    /// Greedy target argmax rows from the batched verify pass.
    pub target_argmax_per_pos: Vec<u32>,
}

/// Rewind point for an LFM2 DFlash target verify.
///
/// LFM2 has no DeltaNet recurrent state, but it does have attention KV rows,
/// short-conv rolling state, and single-token scratch/logits that are mutated
/// by batched verify. This snapshot lets a speculative step restore to the
/// pre-verify position, then replay only the accepted prefix.
pub struct Lfm2DflashTargetSnapshot {
    max_rows: usize,
    saved_n_tokens: usize,
    saved_compact_offset: usize,
    saved_graph_warmed_up: bool,
    kv_start: usize,
    kv_rows: usize,
    kv_k: Vec<DeviceBuffer>,
    kv_v: Vec<DeviceBuffer>,
    kv_k_scales: Vec<DeviceBuffer>,
    kv_v_scales: Vec<DeviceBuffer>,
    conv_states: Vec<DeviceBuffer>,
    h: DeviceBuffer,
    final_norm_buf: DeviceBuffer,
    logits: DeviceBuffer,
}

impl Lfm2DflashTargetSnapshot {
    pub fn new_for(gpu: &mut Gpu, state: &Lfm2MoeState, max_rows: usize) -> HipResult<Self> {
        let kv_k = alloc_kv_rows(gpu, &state.kv.k_gpu, state.kv.physical_cap, max_rows)?;
        let kv_v = alloc_kv_rows(gpu, &state.kv.v_gpu, state.kv.physical_cap, max_rows)?;
        let kv_k_scales = alloc_kv_rows(gpu, &state.kv.k_scales, state.kv.physical_cap, max_rows)?;
        let kv_v_scales = alloc_kv_rows(gpu, &state.kv.v_scales, state.kv.physical_cap, max_rows)?;
        let conv_states = alloc_like_tensors(gpu, &state.conv_states)?;
        Ok(Self {
            max_rows,
            saved_n_tokens: 0,
            saved_compact_offset: 0,
            saved_graph_warmed_up: false,
            kv_start: 0,
            kv_rows: 0,
            kv_k,
            kv_v,
            kv_k_scales,
            kv_v_scales,
            conv_states,
            h: gpu.hip.malloc(state.h.buf.size())?,
            final_norm_buf: gpu.hip.malloc(state.final_norm_buf.buf.size())?,
            logits: gpu.hip.malloc(state.logits.buf.size())?,
        })
    }

    pub fn save_from(&mut self, gpu: &mut Gpu, state: &Lfm2MoeState) -> HipResult<()> {
        self.saved_n_tokens = state.n_tokens;
        self.saved_compact_offset = state.kv.compact_offset;
        self.saved_graph_warmed_up = state.graph_warmed_up;
        self.kv_start = state.n_tokens;
        self.kv_rows = self
            .max_rows
            .min(state.kv.physical_cap.saturating_sub(self.kv_start));

        save_kv_rows(
            gpu,
            &self.kv_k,
            &state.kv.k_gpu,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        save_kv_rows(
            gpu,
            &self.kv_v,
            &state.kv.v_gpu,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        save_kv_rows(
            gpu,
            &self.kv_k_scales,
            &state.kv.k_scales,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        save_kv_rows(
            gpu,
            &self.kv_v_scales,
            &state.kv.v_scales,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        for (dst, src) in self.conv_states.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        gpu.hip
            .memcpy_dtod(&self.h, &state.h.buf, state.h.buf.size())?;
        gpu.hip.memcpy_dtod(
            &self.final_norm_buf,
            &state.final_norm_buf.buf,
            state.final_norm_buf.buf.size(),
        )?;
        gpu.hip
            .memcpy_dtod(&self.logits, &state.logits.buf, state.logits.buf.size())?;
        Ok(())
    }

    pub fn restore_to(&self, gpu: &mut Gpu, state: &mut Lfm2MoeState) -> HipResult<()> {
        restore_kv_rows(
            gpu,
            &self.kv_k,
            &state.kv.k_gpu,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        restore_kv_rows(
            gpu,
            &self.kv_v,
            &state.kv.v_gpu,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        restore_kv_rows(
            gpu,
            &self.kv_k_scales,
            &state.kv.k_scales,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        restore_kv_rows(
            gpu,
            &self.kv_v_scales,
            &state.kv.v_scales,
            state.kv.physical_cap,
            self.kv_start,
            self.kv_rows,
        )?;
        for (src, dst) in self.conv_states.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        gpu.hip.memcpy_dtod(&state.h.buf, &self.h, self.h.size())?;
        gpu.hip.memcpy_dtod(
            &state.final_norm_buf.buf,
            &self.final_norm_buf,
            self.final_norm_buf.size(),
        )?;
        gpu.hip
            .memcpy_dtod(&state.logits.buf, &self.logits, self.logits.size())?;
        state.n_tokens = self.saved_n_tokens;
        state.kv.compact_offset = self.saved_compact_offset;
        state.graph_warmed_up = self.saved_graph_warmed_up;
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for buf in self
            .kv_k
            .into_iter()
            .chain(self.kv_v)
            .chain(self.kv_k_scales)
            .chain(self.kv_v_scales)
            .chain(self.conv_states)
            .chain([self.h, self.final_norm_buf, self.logits])
        {
            let _ = gpu.hip.free(buf);
        }
    }
}

/// Validate the LFM2 target <-> DFlash sidecar contract before allocating
/// heavyweight draft state.
pub fn validate_dflash_contract(
    target_cfg: &Lfm2MoeConfig,
    draft_cfg: &DflashConfig,
) -> Result<(), String> {
    if draft_cfg.hidden != target_cfg.hidden_size {
        return Err(format!(
            "lfm2 dflash: draft hidden {} != target hidden {}",
            draft_cfg.hidden, target_cfg.hidden_size
        ));
    }
    if draft_cfg.vocab_size != target_cfg.vocab_size {
        return Err(format!(
            "lfm2 dflash: draft vocab {} != target vocab {}",
            draft_cfg.vocab_size, target_cfg.vocab_size
        ));
    }
    if draft_cfg.num_target_layers != target_cfg.num_hidden_layers {
        return Err(format!(
            "lfm2 dflash: draft num_target_layers {} != target layers {}",
            draft_cfg.num_target_layers, target_cfg.num_hidden_layers
        ));
    }
    if draft_cfg.target_layer_ids.is_empty() {
        return Err("lfm2 dflash: draft target_layer_ids is empty".to_string());
    }
    let mut seen = std::collections::BTreeSet::new();
    for &layer in &draft_cfg.target_layer_ids {
        if layer >= target_cfg.num_hidden_layers {
            return Err(format!(
                "lfm2 dflash: target layer {layer} out of range 0..{}",
                target_cfg.num_hidden_layers
            ));
        }
        if !seen.insert(layer) {
            return Err(format!(
                "lfm2 dflash: duplicate target layer {layer} in draft config"
            ));
        }
    }
    if draft_cfg.mask_token_id >= target_cfg.vocab_size as u32 {
        return Err(format!(
            "lfm2 dflash: draft mask token {} outside target vocab {}",
            draft_cfg.mask_token_id, target_cfg.vocab_size
        ));
    }
    Ok(())
}

/// Run one LFM2 DFlash draft block and return target-lm_head logits for draft
/// rows `1..B`.
///
/// `target_hidden_host` is the cumulative target-hidden prefix in DFlash
/// layout: `[position][extract_layer][hidden]`. `position` is the logical
/// sequence position where the draft block begins, so `target_hidden_host` must
/// contain at least `position` rows. `ctx_slice` can restrict the context window
/// passed to the draft; `None` uses the full prefix.
///
/// When `noise_embedding` is `None`, the input block is staged from LFM2 target
/// embeddings as `[seed_token, mask, mask, ...]`. A host `noise_embedding`
/// override is kept for diagnostic smokes and must be `[block_size, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn run_dflash_draft_for_logits(
    gpu: &mut Gpu,
    target_weights: &Lfm2MoeWeights,
    target_cfg: &Lfm2MoeConfig,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    target_hidden_host: &[f32],
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    block_size: usize,
    noise_embedding: Option<&[f32]>,
) -> Result<Lfm2DflashDraftLogits, String> {
    validate_dflash_contract(target_cfg, draft_cfg)?;
    if block_size < 2 {
        return Err("lfm2 dflash: block_size must be >= 2".to_string());
    }
    if block_size > draft_scratch.max_block_size {
        return Err(format!(
            "lfm2 dflash: block_size {block_size} > scratch max {}",
            draft_scratch.max_block_size
        ));
    }
    if position == 0 {
        return Err("lfm2 dflash: position must be > 0".to_string());
    }
    if seed_token >= target_cfg.vocab_size as u32 {
        return Err(format!(
            "lfm2 dflash: seed token {seed_token} outside target vocab {}",
            target_cfg.vocab_size
        ));
    }

    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let row_f32 = ne * h;
    let required_prefix = position
        .checked_mul(row_f32)
        .ok_or_else(|| "lfm2 dflash: target_hidden size overflow".to_string())?;
    if target_hidden_host.len() < required_prefix {
        return Err(format!(
            "lfm2 dflash: target_hidden has {} floats, need at least {} for position {}",
            target_hidden_host.len(),
            required_prefix,
            position
        ));
    }

    if let Some(noise) = noise_embedding {
        if noise.len() != block_size * h {
            return Err(format!(
                "lfm2 dflash: noise_embedding has {} floats, expected {}",
                noise.len(),
                block_size * h
            ));
        }
    } else {
        for row in 0..block_size {
            let tok = if row == 0 {
                seed_token
            } else {
                draft_cfg.mask_token_id
            };
            let dst = draft_scratch.x.sub_offset(row * h, h);
            gpu.embedding_lookup_q8(&target_weights.embed, &dst, tok, h)
                .map_err(|e| format!("lfm2 dflash: target embedding row {row}: {e:?}"))?;
        }
    }

    let effective_ctx_len = ctx_slice.unwrap_or(position).min(position);
    if effective_ctx_len == 0 {
        return Err("lfm2 dflash: effective context length is zero".to_string());
    }
    if effective_ctx_len > draft_scratch.max_ctx_len {
        return Err(format!(
            "lfm2 dflash: effective ctx {} > scratch max {}",
            effective_ctx_len, draft_scratch.max_ctx_len
        ));
    }
    let ctx_start = position - effective_ctx_len;
    let th_offset = ctx_start * row_f32;
    let th_end = th_offset + effective_ctx_len * row_f32;
    let target_hidden_slice = &target_hidden_host[th_offset..th_end];

    let positions_q: Vec<i32> = (position as i32..(position + block_size) as i32).collect();
    let positions_k: Vec<i32> = (ctx_start as i32..(position + block_size) as i32).collect();

    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        noise_embedding,
        Some(target_hidden_slice),
        &positions_q,
        &positions_k,
        block_size,
        effective_ctx_len,
        draft_scratch,
    )
    .map_err(|e| format!("lfm2 dflash: draft_forward: {e:?}"))?;

    let batch = block_size - 1;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    let logits_batch = gpu
        .alloc_tensor(&[batch * target_cfg.vocab_size], DType::F32)
        .map_err(|e| format!("lfm2 dflash: alloc logits batch: {e:?}"))?;
    let result = (|| {
        weight_gemm(
            gpu,
            &target_weights.lm_head,
            &hidden_rows,
            &logits_batch,
            batch,
        )
        .map_err(|e| format!("lfm2 dflash: target lm_head: {e:?}"))?;
        if let Some(logit_bias) = draft_weights.logit_bias.as_ref() {
            gpu.bias_add_f32(&logits_batch, logit_bias, batch, target_cfg.vocab_size)
                .map_err(|e| format!("lfm2 dflash: logit bias add: {e:?}"))?;
        }
        gpu.download_f32(&logits_batch)
            .map_err(|e| format!("lfm2 dflash: download logits: {e:?}"))
    })();
    let _ = gpu.free_tensor(logits_batch);
    let logits = result?;
    Ok(Lfm2DflashDraftLogits {
        logits,
        batch,
        vocab_size: target_cfg.vocab_size,
    })
}

/// Verify a DFlash block against the LFM2 target with batched prefill.
///
/// `tokens` should include the seed token at row 0 followed by `B-1` draft
/// proposals. The function advances `target_state` by `tokens.len()` and
/// returns target logits/argmax for every row plus DFlash target-hidden rows.
/// Callers doing speculative acceptance should save an
/// [`Lfm2DflashTargetSnapshot`] before this call, restore it after comparing
/// draft vs target, then replay the accepted prefix into the target state.
pub fn verify_dflash_tokens(
    gpu: &mut Gpu,
    target_weights: &Lfm2MoeWeights,
    target_cfg: &Lfm2MoeConfig,
    target_state: &mut Lfm2MoeState,
    draft_cfg: &DflashConfig,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Lfm2DflashVerifyOutput, String> {
    validate_dflash_contract(target_cfg, draft_cfg)?;
    if tokens.is_empty() {
        return Err("lfm2 dflash verify: token block is empty".to_string());
    }
    if target_state.n_tokens != start_pos {
        return Err(format!(
            "lfm2 dflash verify: state position {} != start_pos {}",
            target_state.n_tokens, start_pos
        ));
    }
    for (i, &tok) in tokens.iter().enumerate() {
        if tok >= target_cfg.vocab_size as u32 {
            return Err(format!(
                "lfm2 dflash verify: token row {i} id {tok} outside target vocab {}",
                target_cfg.vocab_size
            ));
        }
    }

    let mut capture = Lfm2HiddenCapture::new(
        target_cfg.num_hidden_layers,
        target_cfg.hidden_size,
        draft_cfg.target_layer_ids.clone(),
    )?;
    let logits_per_pos = prefill_batch_with_hidden_logits(
        target_cfg,
        target_weights,
        target_state,
        gpu,
        tokens,
        &mut capture,
    )?;
    let expected = tokens.len() * target_cfg.vocab_size;
    if logits_per_pos.len() != expected {
        return Err(format!(
            "lfm2 dflash verify: got {} logits, expected {}",
            logits_per_pos.len(),
            expected
        ));
    }

    let argmax_per_pos = logits_per_pos
        .chunks_exact(target_cfg.vocab_size)
        .map(argmax_u32)
        .collect();
    Ok(Lfm2DflashVerifyOutput {
        logits_per_pos,
        argmax_per_pos,
        target_hidden_rows: capture.take_rows(),
        batch: tokens.len(),
        vocab_size: target_cfg.vocab_size,
    })
}

/// One greedy LFM2 DFlash speculative iteration.
///
/// This mirrors the conservative Qwen3.5 chain-mode invariant:
/// - draft a block from the current seed,
/// - verify `[seed + drafts]` in one LFM2 batched prefill,
/// - accept the longest greedy prefix,
/// - restore the pre-verify target state when verify over-ran,
/// - replay only `[seed + accepted drafts]`.
///
/// The `bonus_token` is included in `committed` for the caller to emit, but it
/// is intentionally not replayed; the next cycle forwards it as its seed.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_dflash(
    gpu: &mut Gpu,
    target_weights: &Lfm2MoeWeights,
    target_cfg: &Lfm2MoeConfig,
    target_state: &mut Lfm2MoeState,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut Lfm2DflashTargetSnapshot,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    block_size_override: Option<usize>,
) -> Result<Lfm2DflashSpecStepResult, String> {
    validate_dflash_contract(target_cfg, draft_cfg)?;
    if target_state.n_tokens != position {
        return Err(format!(
            "lfm2 dflash step: target state position {} != step position {}",
            target_state.n_tokens, position
        ));
    }
    let block_size = block_size_override.unwrap_or(draft_cfg.block_size);
    if block_size < 2 {
        return Err("lfm2 dflash step: block_size must be >= 2".to_string());
    }
    if block_size > target_snap.max_rows {
        return Err(format!(
            "lfm2 dflash step: block_size {block_size} > snapshot max_rows {}",
            target_snap.max_rows
        ));
    }
    let row_floats = draft_cfg.num_extract() * draft_cfg.hidden;
    let expected_hidden = position
        .checked_mul(row_floats)
        .ok_or_else(|| "lfm2 dflash step: target_hidden size overflow".to_string())?;
    if target_hidden_host.len() != expected_hidden {
        return Err(format!(
            "lfm2 dflash step: target_hidden has {} floats, expected {} for position {}",
            target_hidden_host.len(),
            expected_hidden,
            position
        ));
    }

    let draft_logits = run_dflash_draft_for_logits(
        gpu,
        target_weights,
        target_cfg,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        position,
        seed_token,
        ctx_slice,
        block_size,
        None,
    )?;
    let draft_tokens: Vec<u32> = draft_logits
        .logits
        .chunks_exact(draft_logits.vocab_size)
        .map(argmax_u32)
        .collect();
    if draft_tokens.len() != block_size - 1 {
        return Err(format!(
            "lfm2 dflash step: draft returned {} rows, expected {}",
            draft_tokens.len(),
            block_size - 1
        ));
    }
    let mut verify_tokens = Vec::with_capacity(block_size);
    verify_tokens.push(seed_token);
    verify_tokens.extend_from_slice(&draft_tokens);

    target_snap
        .save_from(gpu, target_state)
        .map_err(|e| format!("lfm2 dflash step: snapshot save: {e:?}"))?;
    let verify = verify_dflash_tokens(
        gpu,
        target_weights,
        target_cfg,
        target_state,
        draft_cfg,
        &verify_tokens,
        position,
    )?;
    let (accepted, bonus_token, committed) =
        greedy_acceptance(seed_token, &draft_tokens, &verify.argmax_per_pos)?;
    let rows_to_keep = accepted + 1;
    let hidden_keep = rows_to_keep * row_floats;
    if verify.target_hidden_rows.len() < hidden_keep {
        return Err(format!(
            "lfm2 dflash step: verify hidden has {} floats, need {}",
            verify.target_hidden_rows.len(),
            hidden_keep
        ));
    }

    if rows_to_keep < block_size {
        target_snap
            .restore_to(gpu, target_state)
            .map_err(|e| format!("lfm2 dflash step: snapshot restore: {e:?}"))?;
        let replay_tokens = &committed[..rows_to_keep];
        prefill_batch(target_cfg, target_weights, target_state, gpu, replay_tokens)?;
    }
    if target_state.n_tokens != position + rows_to_keep {
        return Err(format!(
            "lfm2 dflash step: target replay ended at {}, expected {}",
            target_state.n_tokens,
            position + rows_to_keep
        ));
    }

    target_hidden_host.extend_from_slice(&verify.target_hidden_rows[..hidden_keep]);

    Ok(Lfm2DflashSpecStepResult {
        accepted,
        bonus_token,
        drafted: verify_tokens,
        committed,
        advance: rows_to_keep,
        target_argmax_per_pos: verify.argmax_per_pos,
    })
}

fn argmax_u32(row: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &value) in row.iter().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn greedy_acceptance(
    seed_token: u32,
    draft_tokens: &[u32],
    target_argmax_per_pos: &[u32],
) -> Result<(usize, u32, Vec<u32>), String> {
    if target_argmax_per_pos.len() < draft_tokens.len() + 1 {
        return Err(format!(
            "lfm2 dflash step: target argmax rows {} < required {}",
            target_argmax_per_pos.len(),
            draft_tokens.len() + 1
        ));
    }
    let mut accepted = 0usize;
    for (i, &draft) in draft_tokens.iter().enumerate() {
        if target_argmax_per_pos[i] == draft {
            accepted += 1;
        } else {
            break;
        }
    }
    let bonus_token = target_argmax_per_pos[accepted];
    let mut committed = Vec::with_capacity(accepted + 2);
    committed.push(seed_token);
    committed.extend_from_slice(&draft_tokens[..accepted]);
    committed.push(bonus_token);
    Ok((accepted, bonus_token, committed))
}

fn alloc_like_tensors(gpu: &mut Gpu, tensors: &[GpuTensor]) -> HipResult<Vec<DeviceBuffer>> {
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        out.push(gpu.hip.malloc(tensor.buf.size())?);
    }
    Ok(out)
}

fn alloc_kv_rows(
    gpu: &mut Gpu,
    tensors: &[GpuTensor],
    physical_cap: usize,
    max_rows: usize,
) -> HipResult<Vec<DeviceBuffer>> {
    if physical_cap == 0 || max_rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let row_bytes = tensor.buf.size() / physical_cap;
        out.push(gpu.hip.malloc(row_bytes.saturating_mul(max_rows))?);
    }
    Ok(out)
}

fn save_kv_rows(
    gpu: &mut Gpu,
    snapshots: &[DeviceBuffer],
    tensors: &[GpuTensor],
    physical_cap: usize,
    start_pos: usize,
    rows: usize,
) -> HipResult<()> {
    if physical_cap == 0 || rows == 0 {
        return Ok(());
    }
    for (snapshot, tensor) in snapshots.iter().zip(tensors.iter()) {
        let row_bytes = tensor.buf.size() / physical_cap;
        gpu.hip.memcpy_dtod_at(
            snapshot,
            0,
            &tensor.buf,
            start_pos * row_bytes,
            rows * row_bytes,
        )?;
    }
    Ok(())
}

fn restore_kv_rows(
    gpu: &mut Gpu,
    snapshots: &[DeviceBuffer],
    tensors: &[GpuTensor],
    physical_cap: usize,
    start_pos: usize,
    rows: usize,
) -> HipResult<()> {
    if physical_cap == 0 || rows == 0 {
        return Ok(());
    }
    for (snapshot, tensor) in snapshots.iter().zip(tensors.iter()) {
        let row_bytes = tensor.buf.size() / physical_cap;
        gpu.hip.memcpy_dtod_at(
            &tensor.buf,
            start_pos * row_bytes,
            snapshot,
            0,
            rows * row_bytes,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_cfg() -> Lfm2MoeConfig {
        Lfm2MoeConfig::from_config_value(&serde_json::json!({
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "conv_L_cache": 3,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_dense_layers": 1,
            "rope_theta": 1000000.0,
            "norm_eps": 1e-5,
            "max_position_embeddings": 1024,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"]
        }))
        .unwrap()
    }

    fn draft_cfg() -> DflashConfig {
        DflashConfig {
            n_layers: 1,
            hidden: 64,
            intermediate: 128,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            vocab_size: 128,
            norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            block_size: 8,
            mask_token_id: 127,
            target_layer_ids: vec![1, 3],
            num_target_layers: 4,
        }
    }

    #[test]
    fn dflash_contract_accepts_matching_lfm2_shape() {
        validate_dflash_contract(&target_cfg(), &draft_cfg()).unwrap();
    }

    #[test]
    fn dflash_contract_rejects_bad_target_layers() {
        let target = target_cfg();
        let mut draft = draft_cfg();
        draft.target_layer_ids = vec![1, 4];
        assert!(validate_dflash_contract(&target, &draft)
            .unwrap_err()
            .contains("out of range"));
        draft.target_layer_ids = vec![1, 1];
        assert!(validate_dflash_contract(&target, &draft)
            .unwrap_err()
            .contains("duplicate"));
    }

    #[test]
    fn dflash_contract_rejects_hidden_vocab_and_mask_mismatch() {
        let target = target_cfg();
        let mut draft = draft_cfg();
        draft.hidden = 128;
        assert!(validate_dflash_contract(&target, &draft)
            .unwrap_err()
            .contains("hidden"));
        draft = draft_cfg();
        draft.vocab_size = 129;
        assert!(validate_dflash_contract(&target, &draft)
            .unwrap_err()
            .contains("vocab"));
        draft = draft_cfg();
        draft.mask_token_id = 128;
        assert!(validate_dflash_contract(&target, &draft)
            .unwrap_err()
            .contains("mask token"));
    }

    #[test]
    fn greedy_acceptance_rejects_first_draft_and_uses_row0_bonus() {
        let (accepted, bonus, committed) = greedy_acceptance(10, &[20, 30], &[99, 30, 40]).unwrap();
        assert_eq!(accepted, 0);
        assert_eq!(bonus, 99);
        assert_eq!(committed, vec![10, 99]);
    }

    #[test]
    fn greedy_acceptance_accepts_prefix_and_uses_rejection_row_bonus() {
        let (accepted, bonus, committed) =
            greedy_acceptance(10, &[20, 30, 40], &[20, 30, 77, 88]).unwrap();
        assert_eq!(accepted, 2);
        assert_eq!(bonus, 77);
        assert_eq!(committed, vec![10, 20, 30, 77]);
    }

    #[test]
    fn greedy_acceptance_full_accept_uses_tail_row_bonus() {
        let (accepted, bonus, committed) =
            greedy_acceptance(10, &[20, 30, 40], &[20, 30, 40, 88]).unwrap();
        assert_eq!(accepted, 3);
        assert_eq!(bonus, 88);
        assert_eq!(committed, vec![10, 20, 30, 40, 88]);
    }
}
