//! Qualcomm-style training-free MTP probe (arXiv 2603.17942).
//!
//! v1 — engine-surface validation: single mask token per forward, no tree,
//! greedy lossless verify. Goal is to determine whether mask-token embedding
//! injection produces a useful logit signal on hipfire-quantized Qwen3.5,
//! before committing to native-MTP-head extraction + Q8/MQ4 head quant.
//!
//! ## Algorithm (Qualcomm 2603.17942)
//!
//! - §3.1 soft init: mask embedding initialized as the mean of the prompt
//!   token embeddings (computed by looking up each prompt token via the
//!   target's embedding table and averaging on the host).
//! - Eq 4 dynamic update: after each committed token `t`, update
//!   `mask <- (1 - λ) * mask + λ * embed(t)` (default λ=0.1).
//! - §3.3 verify: run a batched forward with `[last_committed,
//!   pending_candidate?, MASK_SENTINEL]` and apply `MaskEmbedOverride` at
//!   the mask slot to substitute the prompt-mean embedding. Greedy argmax
//!   at slot 0 is always committed; if a pending candidate is present and
//!   it matches argmax_at_slot_0, the bonus argmax_at_slot_1 is also
//!   committed. The mask-slot argmax becomes next cycle's pending candidate.
//!
//! τ = (real + speculative) / cycles. Baseline (no MTP gain) is τ=1.0;
//! perfect mask hits give τ=2.0.
//!
//! ## Forward path
//!
//! Uses `qwen35::forward_prefill_batch_with_pbs` directly (NOT the verify
//! path's snapshot/restore — MTP probe always commits its forward output;
//! the model state is monotonically advanced). The batched lm_head dispatch
//! is replicated here per-dtype (mirroring `verify_dflash_block_inner`'s
//! pattern at speculative.rs:2080+) since this isn't a verify call.
//!
//! ## Scope
//!
//! - k=1 mask token per cycle (no tree, no multi-mask).
//! - Greedy lossless: every committed token is what greedy decode would
//!   produce — bit-identical to AR within numerical noise.
//! - No KV rollback: forward always advances 2 or 3 positions.
//! - HIPFIRE_VERIFY_GRAPH path bypassed: mask_override semantics conflict
//!   with the captured embedding-lookup kernel (the override happens via
//!   uncaptured memcpy_htod between embed-lookup and layer 0; under capture
//!   the memcpy would be re-recorded with stale source bytes every replay).

use crate::qwen35::{self, MaskEmbedOverride, PrefillBatchScratch, Qwen35Config, Qwen35Weights};
use crate::speculative::ModelSlot;
use hip_bridge::HipResult;
use hipfire_runtime::llama::{self, EmbeddingFormat};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Maximum batch size per MTP probe cycle: 1 last_committed + 1
/// pending_candidate + 1 mask. v1 never exceeds this.
const MTP_PROBE_MAX_BATCH: usize = 3;

/// Mask-token id placeholder. The `MaskEmbedOverride` overwrites this slot's
/// embedding bytes between the embedding-lookup kernel and the first layer's
/// read of `pbs.x_batch`, so the actual id is purely positional — it controls
/// which row gets looked up by the embedding kernel, but that lookup result is
/// immediately stomped. Picking 0 keeps the lookup cheap (any in-vocab id
/// works; out-of-vocab would risk a bounds check). On Qwen3.5 the id 0 maps
/// to a real token in the vocab so lookups never fault.
pub const MASK_SENTINEL: u32 = 0;

/// State for the Qualcomm MTP probe across one generation.
///
/// `mask_embed` is the prompt-mean (initialized) and dynamically updated
/// (Eq 4) F32 mask-token embedding. It is uploaded into `pbs.x_batch`'s
/// mask slot via `MaskEmbedOverride` on every cycle. `last_committed` is
/// always the most-recently emitted token (drives slot 0 of the next
/// cycle's batch). `pending_candidate` is the previous cycle's mask-slot
/// argmax, or `None` on cycle 0 / immediately after a non-acceptance.
pub struct MtpProbeState {
    /// CPU-side mask embedding, length `dim`. Updated by `update_mask`
    /// (Eq 4) after each committed token; uploaded to GPU each cycle.
    pub mask_embed: Vec<f32>,
    /// Eq 4 EMA factor. Default 0.1 per the paper.
    pub lambda: f32,
    /// Most-recent committed token. `None` only before the first cycle.
    pub last_committed: Option<u32>,
    /// Previous cycle's mask-slot argmax. `None` on cycle 0 / after EOS
    /// / after the speculative slot was rejected (this cycle's slot-0
    /// argmax did NOT match the previous mask prediction).
    pub pending_candidate: Option<u32>,
    // ── Per-call GPU scratch (allocated once in `new_for_prompt`) ──
    /// Scratch for single-token embedding lookups during init + Eq 4
    /// updates. Shape `[dim]` F32. Reused every call.
    embed_tmp: GpuTensor,
    /// Post-output-norm hidden state for all 2-3 batch positions, written
    /// by `forward_prefill_batch_with_pbs(per_token_hidden_out=Some)`.
    /// Shape `[max_n × dim]` F32.
    final_hidden: GpuTensor,
    /// Batched lm_head output, shape `[max_n × vocab]` F32.
    logits: GpuTensor,
    /// FWHT-rotated hidden for MQ4 lm_head path, shape `[max_n × dim]`.
    /// Allocated unconditionally; unused on non-MQ targets.
    rot: GpuTensor,
    /// Persistent prefill batch scratch sized to `max_n` (=3 for v1).
    pbs: PrefillBatchScratch,
    /// Maximum batch size this state was sized for (3 for v1).
    max_n: usize,
}

impl MtpProbeState {
    /// Allocate state and compute the prompt-mean mask embedding.
    ///
    /// O(prompt_len) `memcpy_dtoh`s but only runs once per generation; for
    /// any non-trivial decode the cost is negligible.
    pub fn new_for_prompt(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        prompt_tokens: &[u32],
    ) -> HipResult<Self> {
        assert!(!prompt_tokens.is_empty(), "MTP probe requires non-empty prompt");
        let dim = config.dim;
        let max_n = MTP_PROBE_MAX_BATCH;
        let vocab = config.vocab_size;

        let embed_tmp = gpu.alloc_tensor(&[dim], DType::F32)?;
        let final_hidden = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let logits = gpu.alloc_tensor(&[max_n * vocab], DType::F32)?;
        let rot = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let pbs = PrefillBatchScratch::new(gpu, config, max_n)?;

        // Prompt-mean init (§3.1).
        let mut mean = vec![0.0f32; dim];
        for &tok in prompt_tokens {
            embed_lookup_to_scratch(gpu, weights, &embed_tmp, tok, dim)?;
            let row = gpu.download_f32(&embed_tmp)?;
            debug_assert_eq!(row.len(), dim);
            for (m, v) in mean.iter_mut().zip(row.iter()) {
                *m += *v;
            }
        }
        let inv_n = 1.0f32 / (prompt_tokens.len() as f32);
        for m in mean.iter_mut() {
            *m *= inv_n;
        }

        Ok(Self {
            mask_embed: mean,
            lambda: 0.1,
            last_committed: None,
            pending_candidate: None,
            embed_tmp,
            final_hidden,
            logits,
            rot,
            pbs,
            max_n,
        })
    }

    /// Eq 4: `mask <- (1 - λ) * mask + λ * just_committed_embed`.
    pub fn update_mask(&mut self, just_committed_embed: &[f32]) {
        debug_assert_eq!(just_committed_embed.len(), self.mask_embed.len());
        let lambda = self.lambda;
        let one_minus = 1.0 - lambda;
        for (m, &v) in self.mask_embed.iter_mut().zip(just_committed_embed.iter()) {
            *m = one_minus * *m + lambda * v;
        }
    }

    /// Free GPU buffers. Safe to call once at end of generation.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tmp);
        let _ = gpu.free_tensor(self.final_hidden);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.rot);
        self.pbs.free_gpu(gpu);
    }
}

/// Per-cycle counters. τ = `(committed_real + committed_speculative) / cycles`.
/// Baseline (no MTP gain) is τ=1.0; perfect mask-slot hit rate gives τ=2.0.
#[derive(Default, Debug, Clone, Copy)]
pub struct MtpProbeStats {
    pub cycles: usize,
    pub committed_real: usize,
    pub committed_speculative: usize,
    pub mask_proposed: usize,
    pub eos_hit: bool,
}

impl MtpProbeStats {
    pub fn tau(&self) -> f32 {
        if self.cycles == 0 {
            return 0.0;
        }
        (self.committed_real + self.committed_speculative) as f32 / self.cycles as f32
    }
}

/// Run one MTP probe cycle.
///
/// Returns `(committed_tokens, eos_hit)` where `committed_tokens` is 1 or 2
/// new tokens this cycle (1 = no speculation acceptance, 2 = mask-slot
/// argmax from last cycle matched this cycle's slot-0 argmax). `eos_hit` is
/// true if any committed token is `eos_token`; the caller should stop
/// generation once this fires.
///
/// `cur_pos` is the absolute KV position where slot 0 of this cycle's batch
/// lands. Caller MUST NOT advance KV between calls — this function advances
/// `target.kv_cache` and `target.dn_state` by exactly `batch.len()` positions
/// (2 when no pending candidate, 3 when one is present), regardless of
/// whether the candidate was accepted. The caller MUST advance `cur_pos` by
/// the same amount before the next call:
/// `cur_pos += 2 + state.pending_candidate.is_some() as usize` BEFORE the
/// call (so the next cycle's slot 0 lands at the correct position). Note
/// that `committed_tokens.len()` (1 or 2) is NOT the KV advance — using it
/// to step `cur_pos` will silently corrupt subsequent positions on every
/// rejected candidate.
///
/// State updates:
/// - `state.last_committed` is set to the LAST committed token of this cycle.
/// - `state.pending_candidate` is set to this cycle's mask-slot argmax.
/// - `state.mask_embed` is updated via Eq 4 once per committed token.
pub fn mtp_probe_step(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    state: &mut MtpProbeState,
    cur_pos: usize,
    eos_token: u32,
) -> HipResult<(Vec<u32>, bool)> {
    // Slot 0 always carries the most-recent committed token. Cycle 0 must
    // have been preceded by prompt-prefill, which committed the final
    // prompt token; the caller seeds `state.last_committed` from that.
    let slot0_tok = state
        .last_committed
        .expect("mtp_probe_step requires last_committed to be seeded \
                 from the prompt prefill's final token");

    // Build the batch. Always 2 or 3 slots: [slot0, candidate?, mask].
    let (batch, mask_slot, candidate_slot): (Vec<u32>, usize, Option<usize>) =
        match state.pending_candidate {
            None => (vec![slot0_tok, MASK_SENTINEL], 1, None),
            Some(cand) => (vec![slot0_tok, cand, MASK_SENTINEL], 2, Some(1)),
        };
    debug_assert!(batch.len() <= state.max_n);

    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let n = batch.len();

    // ── Forward (with mask-embed override at the mask slot) ────────────
    //
    // Bypass HIPFIRE_VERIFY_GRAPH-style capture entirely; mask_override
    // semantics conflict with replay (memcpy_htod source bytes change every
    // cycle, but the captured graph would re-record the original pointer).
    let final_hidden_view = state.final_hidden.sub_offset(0, n * dim);
    qwen35::forward_prefill_batch_with_pbs(
        gpu,
        &target.weights,
        &target.config,
        &batch,
        cur_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None,                       // hidden_rb: MTP probe doesn't drive a draft model
        Some(&final_hidden_view),    // post-output-norm hidden for all N rows
        None,                       // gdn_tape
        None,                       // tree_verify
        Some(&state.pbs),
        Some(MaskEmbedOverride {
            slot: mask_slot,
            embed: &state.mask_embed,
        }),
        None, // max_layer: MTP probe runs full stack
    )?;

    // ── Per-position lm_head GEMM ───────────────────────────────────────
    //
    // Mirrors verify_dflash_block_inner's batched dispatch (speculative.rs
    // ~2080+). Greedy-only (no temperature) so we use GPU batched argmax
    // for the single-row D2H per slot rather than full B×vocab download.
    let w_out = &target.weights.output;
    let logits_batch = state.logits.sub_offset(0, n * vocab);

    match w_out.gpu_dtype {
        DType::Q8_0 => {
            // gemm_q8_0_batched has a hard MAX_BATCH=64 (speculative.rs
            // Q8_LM_MAX); probe's n <= MTP_PROBE_MAX_BATCH = 3 always
            // satisfies this. If MTP_PROBE_MAX_BATCH grows past 64, copy the
            // chunking loop from verify_dflash_block_inner. Do NOT route
            // through gemm_qkv_q8_0_wmma — would break greedy-parity with
            // GEMV.
            gpu.gemm_q8_0_batched(
                &w_out.buf, &final_hidden_view, &logits_batch,
                w_out.m, w_out.k, n,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &final_hidden_view, &logits_batch,
                w_out.m, w_out.k, n,
            )?;
        }
        DType::MQ4G256 => {
            // MQ4 needs FWHT-rotated x first; reuse `state.rot` as scratch.
            let rot_view = state.rot.sub_offset(0, n * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &final_hidden_view, &rot_view, w_out.k, n)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot_view, &logits_batch,
                w_out.m, w_out.k, n,
            )?;
        }
        DType::MQ3G256 => {
            let rot_view = state.rot.sub_offset(0, n * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &final_hidden_view, &rot_view, w_out.k, n)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot_view, &logits_batch,
                w_out.m, w_out.k, n,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &final_hidden_view, &logits_batch,
                w_out.m, w_out.k, n,
            )?;
        }
        DType::MQ6G256 => {
            let rot_view = state.rot.sub_offset(0, n * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &final_hidden_view, &rot_view, w_out.k, n)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot_view, &logits_batch,
                w_out.m, w_out.k, n,
            )?;
        }
        // Fallback for less-common lm_head dtypes: per-position GEMV.
        _ => {
            for i in 0..n {
                let hidden_row = final_hidden_view.sub_offset(i * dim, dim);
                let logits_row = logits_batch.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &hidden_row, &logits_row)?;
            }
        }
    }

    // Greedy argmax per slot. Download full logits once — cost is small
    // since n ≤ 3 (3 × vocab × 4 B at vocab=152K ≈ 1.7 MB).
    let host_logits = gpu.download_f32(&logits_batch)?;
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let row = &host_logits[i * vocab..(i + 1) * vocab];
        argmax_per_pos.push(argmax_u32(row));
    }

    // ── Decide commits ─────────────────────────────────────────────────
    //
    // Slot 0 argmax = the model's true greedy next token at position
    // `cur_pos + 1` (i.e. the token that follows `slot0_tok`). Always
    // commit. If we had a pending candidate at `candidate_slot` AND
    // argmax_at_slot_0 == that candidate, then last cycle's mask correctly
    // predicted this token → also commit argmax at the candidate slot
    // (which is the model's prediction for one position FURTHER along, made
    // under the assumption that the candidate would land where it did).
    let mut committed: Vec<u32> = Vec::with_capacity(2);
    let real_token = argmax_per_pos[0];
    committed.push(real_token);

    if let Some(cand_slot) = candidate_slot {
        let candidate = state.pending_candidate.unwrap();
        if real_token == candidate {
            // Mask hit: the candidate matched the model's greedy prediction.
            // The argmax at the candidate slot is our bonus token.
            let bonus = argmax_per_pos[cand_slot];
            committed.push(bonus);
        }
    }

    // ── Update state for next cycle ────────────────────────────────────
    //
    // Eq 4 update: lookup each newly-committed token's embedding, dtoh,
    // apply the EMA. Done in commit order; the final committed token is
    // also stored as `last_committed`.
    for &tok in &committed {
        embed_lookup_to_scratch(gpu, &target.weights, &state.embed_tmp, tok, dim)?;
        let row = gpu.download_f32(&state.embed_tmp)?;
        state.update_mask(&row);
    }
    state.last_committed = Some(*committed.last().unwrap());

    let eos_hit = committed.iter().any(|&t| t == eos_token);

    // EOS resets pending_candidate so a downstream caller that ignores the
    // eos_hit signal and keeps stepping won't carry stale speculation across
    // a logical generation boundary.
    state.pending_candidate = if eos_hit {
        None
    } else {
        Some(argmax_per_pos[mask_slot])
    };

    Ok((committed, eos_hit))
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Per-format embedding-lookup dispatch. Writes one row of the embedding
/// table into `out` (shape `[dim]` F32). Mirrors the format-switch pattern
/// in `qwen35::forward_scratch` (qwen35.rs:3035+).
fn embed_lookup_to_scratch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    out: &GpuTensor,
    token: u32,
    dim: usize,
) -> HipResult<()> {
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, out, token, dim)
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, out, token, dim)
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, out, token, dim)
        }
        // Q4K is not currently produced by qwen35.rs embedding-format assignment;
        // kept defensively in case it lands later — verify call convention then.
        EmbeddingFormat::Q4K => {
            gpu.embedding_lookup_q4k(&weights.token_embd, out, token, dim)
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, out, token, dim)
        }
    }
}

/// Single-pass argmax. Keeps `mtp_probe.rs` self-contained (the equivalent
/// helper in `speculative.rs` is private).
#[inline]
fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}
