//! MTP-only speculative decode: standalone trunk + native MTP head loop.
//!
//! The MTP head ([`crate::mtp_head`], Task 9) is a single transformer-decoder
//! block that takes (last-committed-token, trunk-post-norm-hidden) and emits
//! a single-token next-prediction. This module composes that head into a
//! greedy lossless spec-decode cycle:
//!
//! ```text
//! cycle:
//!   1. Snapshot trunk DN state (for rollback after batched verify).
//!   2. MTP-chain `max_n` candidates by feeding each step's `t_mtp_out` and
//!      its predicted token back into mtp_head_forward.
//!   3. Run trunk verify on `[last_committed, c1, ..., c_max_n]`
//!      (max_n + 1 tokens). Capture per-position post-output-norm hidden via
//!      `per_token_hidden_out`.
//!   4. Trunk lm_head over all positions → batched argmax → accept-prefix
//!      where MTP candidate matches trunk argmax.
//!   5. bonus_token = trunk argmax at slot `accept_count` (always commits
//!      unless we early-stopped on EOS inside the accepted prefix).
//!   6. Update `state.prev_hidden` from `verify_hidden[advance - 1]`.
//!   7. Roll back trunk DN state to snapshot, replay accepted committed
//!      tokens (= `verify_tokens[..advance]`) so trunk dn/kv lands at
//!      position `cur_pos + advance` for the next cycle.
//!   8. MTP head KV cache: NO explicit rollback. Stale slots from rejected
//!      candidates get overwritten in-order by next cycle's MTP chain
//!      because each next-cycle MTP step writes slot `pos` BEFORE attending
//!      `[0..pos+1)`.
//! ```
//!
//! Greedy-only (temp=0). No DDTree, no PLD, no rejection sampling — that's
//! Task 11 territory.

use crate::mtp_head::{
    self, Qwen35MtpHead, Qwen35MtpHeadKvCache, Qwen35MtpHeadScratch,
};
use crate::qwen35::{self, Qwen35Weights};
use crate::speculative::{DeltaNetSnapshot, ModelSlot};
use hip_bridge::HipResult;
use hipfire_runtime::llama;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Public state ────────────────────────────────────────────────────────

/// All persistent buffers needed by [`spec_step_mtp`]. Allocate once per
/// generation; reuse across cycles. Frees at end of generation via
/// [`MtpSpecState::free_gpu`].
pub struct MtpSpecState {
    /// Persistent post-output-norm hidden snapshot of the trunk at the
    /// position of the most-recently-committed token. Fed as `prev_hidden`
    /// into the MTP head's first step each cycle. Shape: `[dim]` F32.
    pub prev_hidden: GpuTensor,

    /// Per-cycle scratch: post-output-norm hidden states from the trunk
    /// verify, one row per verify position. Shape: `[(max_n + 1) × dim]` F32.
    pub verify_hidden: GpuTensor,

    /// Per-cycle scratch: trunk lm_head output over all verify positions.
    /// Shape: `[(max_n + 1) × vocab]` F32.
    pub verify_logits: GpuTensor,

    /// Per-cycle scratch: FWHT-rotated `verify_hidden` for MQ-family
    /// lm_head. Shape: `[(max_n + 1) × dim]` F32. Allocated unconditionally;
    /// unused for non-MQ lm_head dtypes.
    pub verify_rot: GpuTensor,

    /// GPU-side argmax buffer for batched argmax over verify positions.
    /// Shape: `[(max_n + 1)]` i32 stored as F32 slots.
    pub verify_argmax: GpuTensor,

    /// Trunk DN snapshot for rollback before replay.
    pub trunk_snap: DeltaNetSnapshot,

    /// Persistent batch scratch for the trunk's batched verify forward.
    /// Sized to `max_n + 1` so verify always fits in one chunk.
    pub trunk_pbs: qwen35::PrefillBatchScratch,

    /// MTP head per-call scratch (Task 9). One per slot; reused across cycles.
    pub mtp_scratch: Qwen35MtpHeadScratch,

    /// MTP head KV cache (Task 9). Single-layer F32, rolling overwrite.
    pub mtp_kv: Qwen35MtpHeadKvCache,

    /// Per-step `t_mtp_out` capture buffer for the K-step batched lm_head
    /// chain (Task 10b). Shape `[max_n, n_embd]` row-major. After the
    /// chain runs, the end-of-chain batched lm_head consumes this whole
    /// buffer in one GEMM call. Step k writes its `t_mtp_out` into row k
    /// (offset `k * n_embd * 4` bytes); step k+1 reads row k as its
    /// "next-token embedding override" + `prev_hidden`.
    pub mtp_t_outs: GpuTensor,

    /// Per-step batched-rmsnorm output for the K-step batched lm_head.
    /// Shape `[max_n, n_embd]`. Caller of `mtp_head_apply_lm_head_batched`
    /// owns this scratch.
    pub mtp_lm_tmp: GpuTensor,

    /// Per-step FWHT-rotated x scratch for MagnumQuant lm_heads (MQ4/MQ3/MQ6).
    /// Shape `[max_n, n_embd]`. Unused for non-MQ lm_head dtypes.
    pub mtp_lm_rot: GpuTensor,

    /// Per-step batched lm_head logits output. Shape `[max_n, vocab]`.
    /// Caller `argmax_f32_batched` reads this for the K predicted tokens.
    pub mtp_lm_logits: GpuTensor,

    /// GPU-side argmax destination for the K-step batched argmax over
    /// `mtp_lm_logits`. Shape `[max_n]` i32 stored as F32 slots.
    pub mtp_lm_argmax: GpuTensor,

    /// Optional FastMTP-style compressed batched-logits output, shape
    /// `[max_n * compressed_vocab_size]`. Populated by
    /// [`spec_step_mtp_compressed`] when the head ships a sidecar.
    /// Allocated lazily via [`MtpSpecState::ensure_compressed_lm_logits`].
    pub mtp_lm_logits_compressed: Option<GpuTensor>,

    /// Maximum number of MTP candidates per cycle (chain depth).
    pub max_n: usize,
}

impl MtpSpecState {
    /// Allocate all per-generation buffers, sizing the trunk DN snapshot
    /// against the live `target.dn_state`.
    pub fn new_for_slot(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_n: usize,
    ) -> HipResult<Self> {
        assert!(
            max_n >= 1,
            "MtpSpecState::new_for_slot: max_n must be ≥ 1 (chain depth)"
        );
        let dim = target.config.dim;
        let vocab = target.config.vocab_size;
        assert_eq!(
            head.config.n_embd, dim,
            "MtpSpecState: trunk dim={dim} but head n_embd={}",
            head.config.n_embd,
        );
        assert_eq!(
            head.config.vocab_size, vocab,
            "MtpSpecState: trunk vocab={vocab} but head vocab={}",
            head.config.vocab_size,
        );

        let prev_hidden = gpu.alloc_tensor(&[dim], DType::F32)?;
        let verify_hidden = gpu.alloc_tensor(&[(max_n + 1) * dim], DType::F32)?;
        let verify_logits = gpu.alloc_tensor(&[(max_n + 1) * vocab], DType::F32)?;
        let verify_rot = gpu.alloc_tensor(&[(max_n + 1) * dim], DType::F32)?;
        let verify_argmax = gpu.alloc_tensor(&[max_n + 1], DType::F32)?;
        let trunk_snap = DeltaNetSnapshot::new_for(gpu, &target.dn_state)?;
        let trunk_pbs = qwen35::PrefillBatchScratch::new(gpu, &target.config, max_n + 1)?;
        let mtp_scratch = Qwen35MtpHeadScratch::new(gpu, &head.config)?;
        let mtp_kv = Qwen35MtpHeadKvCache::new(gpu, &head.config)?;

        // Per-step batched-lm_head scratch (Task 10b).
        let mtp_t_outs = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_tmp = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_rot = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_logits = gpu.alloc_tensor(&[max_n * vocab], DType::F32)?;
        let mtp_lm_argmax = gpu.alloc_tensor(&[max_n], DType::F32)?;

        Ok(Self {
            prev_hidden,
            verify_hidden,
            verify_logits,
            verify_rot,
            verify_argmax,
            trunk_snap,
            trunk_pbs,
            mtp_scratch,
            mtp_kv,
            mtp_t_outs,
            mtp_lm_tmp,
            mtp_lm_rot,
            mtp_lm_logits,
            mtp_lm_argmax,
            mtp_lm_logits_compressed: None,
            max_n,
        })
    }

    /// Allocate `mtp_lm_logits_compressed` shape `[max_n * cvs]` for the
    /// compressed batched-lm_head dispatch path. Idempotent — reallocates
    /// only if the existing tensor has the wrong size. Call once after
    /// loading a head with a sidecar (cvs = `head.weights.compressed_vocab_size`).
    pub fn ensure_compressed_lm_logits(
        &mut self, gpu: &mut Gpu, cvs: usize,
    ) -> HipResult<()> {
        let needed = self.max_n * cvs;
        if let Some(existing) = self.mtp_lm_logits_compressed.as_ref() {
            if existing.numel() == needed {
                return Ok(());
            }
            if let Some(old) = self.mtp_lm_logits_compressed.take() {
                let _ = gpu.free_tensor(old);
            }
        }
        self.mtp_lm_logits_compressed = Some(
            gpu.alloc_tensor(&[needed], DType::F32)?,
        );
        Ok(())
    }

    /// Capture the trunk's post-output-norm hidden after a `forward_scratch`
    /// call. Source is `target.scratch.tmp` (same buffer lm_head reads from).
    /// Call once after prefill's last `forward_scratch` to seed the cycle.
    pub fn capture_prev_hidden_from_scratch_tmp(
        &self,
        gpu: &Gpu,
        target_scratch_tmp: &GpuTensor,
        dim: usize,
    ) -> HipResult<()> {
        gpu.hip.memcpy_dtod_at(
            &self.prev_hidden.buf, 0,
            &target_scratch_tmp.buf, 0,
            dim * 4,
        )
    }

    /// Capture the trunk's post-output-norm hidden from a row of
    /// `verify_hidden` (per-position output of the batched verify forward).
    /// Used after spec_step accepts to set up next cycle's MTP `prev_hidden`
    /// from the verify slot corresponding to the last committed token,
    /// avoiding the cost of a separate single-token forward.
    pub fn capture_prev_hidden_from_verify_row(
        &self,
        gpu: &Gpu,
        row: usize,
        dim: usize,
    ) -> HipResult<()> {
        gpu.hip.memcpy_dtod_at(
            &self.prev_hidden.buf, 0,
            &self.verify_hidden.buf, row * dim * 4,
            dim * 4,
        )
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.prev_hidden);
        let _ = gpu.free_tensor(self.verify_hidden);
        let _ = gpu.free_tensor(self.verify_logits);
        let _ = gpu.free_tensor(self.verify_rot);
        let _ = gpu.free_tensor(self.verify_argmax);
        let _ = gpu.free_tensor(self.mtp_t_outs);
        let _ = gpu.free_tensor(self.mtp_lm_tmp);
        let _ = gpu.free_tensor(self.mtp_lm_rot);
        let _ = gpu.free_tensor(self.mtp_lm_logits);
        let _ = gpu.free_tensor(self.mtp_lm_argmax);
        if let Some(lc) = self.mtp_lm_logits_compressed {
            let _ = gpu.free_tensor(lc);
        }
        // DeltaNetSnapshot's DeviceBuffers free on drop.
        drop(self.trunk_snap);
        self.trunk_pbs.free_gpu(gpu);
        self.mtp_scratch.free_gpu(gpu);
        self.mtp_kv.free_gpu(gpu);
    }
}

// ─── One spec-decode cycle ───────────────────────────────────────────────

/// Result of one MTP spec-decode cycle.
#[derive(Debug, Clone)]
pub struct MtpSpecResult {
    /// Tokens committed THIS cycle (excludes the seed `last_committed` from
    /// the previous cycle; includes the bonus when no early EOS).
    /// Length == `advance`.
    pub committed: Vec<u32>,
    /// Number of MTP candidates that matched trunk argmax (not counting the
    /// bonus). 0 = MTP wasted, max_n = full acceptance.
    pub accept_count: usize,
    /// True iff any committed token equals `eos_token_id` (caller stops).
    pub hit_eos: bool,
    /// Position advance (= committed.len()). Caller's `cur_pos += advance`.
    pub advance: usize,
}

/// One MTP-only spec-decode cycle (greedy, temp=0).
///
/// Preconditions:
/// - Trunk has been forwarded through position `cur_pos - 1`. Trunk's
///   `dn_state` is post-position-`cur_pos - 1` ready to accept the next
///   token at position `cur_pos`. `target.kv_cache` has K/V written at
///   positions `[0..cur_pos)`. `target.scratch.tmp` is unused / available.
/// - `state.prev_hidden` holds the trunk's post-output-norm hidden at
///   position `cur_pos - 1` (i.e., the position whose argmax produced
///   `last_committed`).
/// - MTP head KV cache is "rolling": stale slots from prior rejected
///   chains may exist at slots `>= cur_pos`; they will be overwritten in
///   place by this cycle's chain.
/// - `last_committed` is the token id at logical position `cur_pos - 1`.
///   It IS verified again as block[0] of this cycle's verify input — its
///   K/V row in trunk's cache stays consistent (idempotent re-write at
///   position cur_pos - 1? No — at position cur_pos - 1 the token was
///   written at the PREVIOUS cycle's last verify or replay. block[0]
///   here is at position cur_pos: trunk's verify will write trunk KV
///   slot cur_pos with last_committed's K/V. NOTE: this means
///   `last_committed`'s "logical" position for THIS cycle's verify is
///   cur_pos, NOT cur_pos - 1.
///
/// So the precise semantics: `cur_pos` is the position where this cycle's
/// verify block[0] (= `last_committed` re-emit) lands. The PREVIOUS cycle
/// committed `last_committed` at position `cur_pos - 1`? Or at position
/// `cur_pos`?
///
/// Actually re-reading spec_step_dflash @ 2560-2583: the seed token is
/// re-fed at position `position` (the `position` parameter), and verify
/// runs `block` at positions `[position..position + B)`. So the seed is
/// at `position`, NOT `position - 1`. This is intentional: each cycle
/// re-runs the seed forward to land trunk K/V in slot `position` (because
/// after the previous cycle's replay, trunk advanced to `position` but
/// did NOT yet write a K/V slot at `position` for the bonus_token — the
/// bonus is "deferred" to be written here as block[0]).
///
/// We mirror that convention. Caller supplies `cur_pos` = position where
/// `last_committed` (= prev cycle's bonus) sits. After cycle, advance by
/// `result.advance`. NEW commits include indices [cur_pos+1..cur_pos+advance]
/// PLUS we also "re-confirm" cur_pos itself via block[0]. The caller's
/// emitted-token list should NOT include `last_committed` again.
///
/// Postconditions:
/// - Trunk dn_state advanced to `cur_pos + advance` (state ready for next
///   cycle's verify at position `cur_pos + advance`).
/// - Trunk kv_cache has K/V at positions `[0..cur_pos + advance)`.
/// - `state.prev_hidden` updated to trunk's post-output-norm hidden at
///   position `cur_pos + advance - 1` (= the LAST committed token).
/// - MTP KV cache may have stale slots at `>= cur_pos + advance`; OK.
/// - `result.hit_eos`: true iff any committed token equals `eos_token_id`.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Ensure stream exists for any async memset (downstream batched kernels
    // assume it). Mirrors spec_step_dflash convention.
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. MTP candidate chain (Approach B: feature-only K-step) ────────
    //
    // ARCHITECTURALLY LOSSY: each step k (k > 0) feeds the previous step's
    // `t_mtp_out` directly as both the "embedding of the predicted token"
    // (bypassing the embedding table — see `mtp_head_forward_block_only`'s
    // `next_token_embed` doc comment) AND as `prev_hidden`. This lets us
    // chain all K block forwards back-to-back WITHOUT running lm_head per
    // step, then collapse the K argmaxes into a single batched lm_head GEMM.
    //
    // Trade-off (vs the earlier discrete-token-roundtrip path that called
    // `mtp_head_forward` K times): saves K-1 separate GEMV launches into
    // the trunk's vocab×n_embd lm_head matrix. Each saved launch is ~12 ms
    // for the 27B-3.5 lm_head (vocab=152K). On gfx1100 K=3 the projected
    // win is ~22 ms/cycle = ~37%-1.5× lift if τ holds.
    //
    // Lossless guarantee preserved at trunk-verify: any incorrect MTP
    // candidate (from the lossy chain or otherwise) is rejected by trunk
    // argmax check; the cycle just degenerates to AR + bonus token. τ may
    // drop relative to the discrete-token-roundtrip path because the head
    // was trained with `embed[token]`, not raw `t_mtp_out`, as its
    // step-input distribution.
    //
    // MTP step k writes its own KV slot at position cur_pos + k (same as
    // before — a per-step single-layer transformer cache).
    let dim_bytes = dim * 4;
    for k in 0..max_n {
        // Step 0 reads from state.prev_hidden (trunk's last post-norm
        // hidden, snapshot from the previous cycle / prefill) and embeds
        // last_committed via the trunk's table. Step k > 0 reads from row
        // (k-1) of mtp_t_outs and uses that same row as the embedding
        // override (lossy substitute for embed[predicted_token_{k-1}]).
        if k == 0 {
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                last_committed,
                &state.prev_hidden,
                None,
                cur_pos + k,
                trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                0,                  // sentinel: ignored when next_token_embed is Some(_)
                &prev_row,          // prev_hidden = step k-1's t_mtp_out
                Some(&prev_row),    // embed override = step k-1's t_mtp_out (lossy)
                cur_pos + k,
                trunk_weights,
            )?;
        }
        // Capture this step's t_mtp_out into row k of mtp_t_outs so the
        // next step can read it AND so the end-of-chain batched lm_head
        // can ingest the whole stack.
        gpu.hip.memcpy_dtod_at(
            &state.mtp_t_outs.buf, k * dim_bytes,
            &state.mtp_scratch.t_mtp_out.buf, 0,
            dim_bytes,
        )?;
    }

    // ── 1b. Batched lm_head over all K t_mtp_outs ────────────────────────
    //
    // Single GEMM (N=max_n) replacing K separate GEMV calls. shared_head_norm
    // is applied row-wise via rmsnorm_batched inside the helper. MQ4
    // dispatch path includes the FWHT rotation pre-pass (rotate_x_mq_batched).
    let t_outs_view = state.mtp_t_outs.sub_offset(0, max_n * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, max_n * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, max_n * dim);
    let lm_logits_view = state.mtp_lm_logits.sub_offset(0, max_n * vocab);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu,
        head,
        &trunk_weights.output,
        &t_outs_view,
        &lm_tmp_view,
        &lm_rot_view,
        &lm_logits_view,
        max_n,
    )?;

    // Batched argmax over (max_n) rows → single D2H of K i32 ids.
    let lm_argmax_view = state.mtp_lm_argmax.sub_offset(0, max_n);
    gpu.argmax_f32_batched(&lm_logits_view, &lm_argmax_view, vocab, max_n)?;
    let mut argmax_host: Vec<i32> = vec![0; max_n];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_host.as_mut_ptr() as *mut u8, max_n * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &lm_argmax_view.buf)?;
    }
    let candidates: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // ── 2. Build trunk verify input: [last_committed, c1, ..., c_max_n] ──
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    debug_assert_eq!(verify_tokens.len(), max_n + 1);

    // ── 3. Snapshot trunk DN state (rollback target before verify) ───────
    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    // ── 4. Trunk batched verify, capturing per-position hidden ───────────
    //
    // `forward_prefill_batch_with_pbs` writes per-position post-output-norm
    // hidden into `state.verify_hidden` when per_token_hidden_out is Some.
    // Trunk KV cache and dn_state advance by max_n+1 (= verify_tokens.len()).
    qwen35::forward_prefill_batch_with_pbs(
        gpu,
        trunk_weights,
        &target.config,
        &verify_tokens,
        cur_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None, // hidden_rb: not used here (DFlash drafter not in the loop)
        Some(&state.verify_hidden),
        None, // gdn_tape
        None, // tree_verify
        Some(&state.trunk_pbs),
        None, // mask_override
    )?;

    // ── 5. Per-position lm_head + batched argmax ─────────────────────────
    //
    // Mirrors verify_dflash_block's batched lm_head dispatch by dtype.
    // Sized to (max_n + 1). Greedy-only, so we use GPU-side batched argmax
    // to avoid the (max_n + 1) × vocab D2H.
    let n_verify = max_n + 1;
    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            // gemm_q8_0_batched supports up to 64 rows in one launch.
            gpu.gemm_q8_0_batched(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        _ => {
            // Fallback: per-row weight_gemv. Same path verify_dflash_block uses.
            for i in 0..n_verify {
                let row = state.verify_hidden.sub_offset(i * dim, dim);
                let logits_row = state.verify_logits.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &row, &logits_row)?;
            }
        }
    }

    // GPU-side batched argmax over (max_n + 1) rows.
    let argmax_view = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_view, vocab, n_verify)?;
    let mut argmax_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_host.as_mut_ptr() as *mut u8, n_verify * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // ── 6. Accept-prefix ────────────────────────────────────────────────
    //
    // For k in [0..max_n): if MTP candidate `candidates[k]` matches trunk's
    // argmax at slot k (= trunk's prediction of token at position
    // cur_pos + k + 1), accept. Stop on first miss OR EOS.
    //
    // bonus_token = argmax_per_pos[accept_count] (always commits unless
    // an accepted MTP candidate was already EOS).
    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(max_n + 1);

    for k in 0..max_n {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        // Bonus: trunk's argmax at the first rejection slot (or after the
        // full accepted chain). Always commits one extra token.
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }

    let advance = committed.len();
    debug_assert!(advance >= 1, "spec_step_mtp must commit at least the bonus");
    debug_assert!(advance <= max_n + 1);

    // ── 7. Capture prev_hidden for next cycle ────────────────────────────
    //
    // Next cycle's MTP step 0 needs trunk's post-output-norm hidden whose
    // argmax PRODUCED the next cycle's last_committed (= this cycle's
    // last committed token). Verify-slot `i` holds the post-norm hidden at
    // trunk position `cur_pos + i`, which produces argmax_per_pos[i] for
    // position `cur_pos + i + 1`. The last committed token sits at:
    //
    //   - If hit_eos triggered inside the MTP-accepted chain (no bonus):
    //     advance == accept_count, last token at position cur_pos + advance
    //     (= cur_pos + accept_count), which is the i = advance - 1 slot.
    //   - Otherwise: last token = bonus at position cur_pos + accept_count + 1,
    //     advance == accept_count + 1, slot = advance - 1.
    //
    // Both cases reduce to slot `advance - 1`.
    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, prev_hidden_row, dim)?;

    // ── 8. Roll back trunk DN state + replay accepted committed tokens ───
    //
    // After verify, trunk dn_state is at position cur_pos + max_n + 1.
    // We need it at cur_pos + advance for next cycle. Restore snapshot
    // (back to cur_pos), then replay first `advance` of `verify_tokens`.
    // Replay re-writes trunk KV slots [cur_pos..cur_pos + advance) with
    // identical K/V (idempotent — same inputs, same positions), and
    // re-runs DN recurrence.
    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if advance >= 2 {
        let replay = &verify_tokens[..advance];
        qwen35::forward_prefill_batch(
            gpu,
            trunk_weights,
            &target.config,
            replay,
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None, None, None, None,
        )?;
    } else {
        // advance == 1: only the seed (last_committed) is "replayed". This
        // happens when EOS hits with accept_count==0 AND bonus==eos
        // (very rare), or when the bonus was committed alone (also implies
        // accept_count==0, which is the common AR-degraded case).
        qwen35::forward_scratch(
            gpu,
            trunk_weights,
            &target.config,
            verify_tokens[0],
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
        )?;
    }

    // ── 9. MTP KV cache rollback: NOT NEEDED ────────────────────────────
    //
    // Argument: MTP step k of the original chain wrote slot cur_pos + k
    // with K/V derived from input (k==0 ? last_committed : candidates[k-1]).
    // For k in [0..accept_count], the actual committed token at position
    // cur_pos + k matches that input (last_committed for k=0, candidates[k-1]
    // = committed[k-1] for k≥1 because c_k was accepted), so those slots
    // hold correct K/V. For k in [accept_count+1..max_n), the chain wrote
    // K/V from rejected candidates — STALE. But the next cycle starts at
    // cur_pos_new = cur_pos + advance = cur_pos + accept_count + 1
    // (or cur_pos + accept_count if EOS-no-bonus), and step 0 of that
    // next cycle WRITES slot cur_pos_new BEFORE attending [0..cur_pos_new+1).
    // Stale slots [cur_pos + advance, cur_pos + max_n) are at-or-beyond
    // cur_pos_new = cur_pos + advance, so step 0 next cycle writes the
    // first stale slot before reading it. Step k next cycle writes the
    // (k+1)-th formerly-stale slot before its attention reads it. The
    // chain progresses left-to-right, overwriting stale slots in-order
    // exactly as needed. NO explicit rollback required.

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
    })
}

// ─── FastMTP-style compressed spec step (K = max_n) ───────────────────────
//
// Drop-in alternative to [`spec_step_mtp`] when the loaded MTP head ships a
// compressed `lm_head_draft` sidecar. Chains K = state.max_n MTP block
// forwards (k=0 normal, k>0 with lossy embedding override = previous step's
// `t_mtp_out`), then runs ONE batched compressed lm_head dispatch over the K
// stacked outputs (vs K full-vocab dispatches in the naive K-step path).
//
// Per-cycle BW (27B-3.5, K=3):
//   K full-vocab lm_head GEMVs: K * 635 MB ≈ 1.9 GB MQ4 streamed in
//   1 batched compressed lm_head GEMM: 84 MB MQ4 + K * 5120 * 4 B activations ≈ 84 MB
//   Saved: ~1.8 GB BW per cycle; trunk verify still runs full lm_head (n=K+1 batched).
//
// K=1 retains the original K=1 fast path; K>1 chains lossily (same OOD risk
// the existing `spec_step_mtp` warns about — τ may degrade vs discrete-token
// roundtrip). Lossless greedy preserved at the trunk-verify layer regardless.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Precondition: head must have compressed sidecar loaded + caller must
    // have allocated the compressed batched-logits scratch.
    let lm_head_draft = head.weights.lm_head_draft.as_ref()
        .expect("spec_step_mtp_compressed requires head loaded with --vocab-sidecar");
    let vocab_map = head.weights.lm_head_draft_vocab_map.as_ref()
        .expect("compressed head missing vocab_map");
    let cvs = head.weights.compressed_vocab_size
        .expect("compressed head missing compressed_vocab_size");
    let logits_c_batched = state.mtp_lm_logits_compressed.as_ref()
        .expect("MtpSpecState::mtp_lm_logits_compressed not allocated; \
                 call ensure_compressed_lm_logits(gpu, cvs) after head load");
    assert!(
        logits_c_batched.numel() >= max_n * cvs,
        "mtp_lm_logits_compressed too small: {} < max_n*cvs ({})",
        logits_c_batched.numel(), max_n * cvs,
    );

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. Chain K MTP block forwards (lossy embedding override for k>0) ──
    //
    // Identical structure to spec_step_mtp's K-step lossy chain. Step 0
    // reads from state.prev_hidden + embeds last_committed via trunk's
    // token table. Step k>0 reads from row (k-1) of mtp_t_outs and uses
    // that row as the embedding override (continuous t_mtp_out as a lossy
    // substitute for embed[predicted_token_{k-1}]) so we don't need to
    // know the chain's interim tokens before running step k+1.
    //
    // The chain's ARCHITECTURAL LOSS comes from this OOD-for-the-head
    // continuous embedding — the MTP head was trained with discrete-token
    // roundtrips. τ may degrade. Trunk verify catches any wrong candidate
    // (lossless greedy preserved), but acceptance rate suffers.
    let dim_bytes = dim * 4;
    for k in 0..max_n {
        if k == 0 {
            mtp_head::mtp_head_forward_block_only(
                gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                last_committed, &state.prev_hidden, None, cur_pos + k,
                trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                0,                  // ignored when next_token_embed = Some
                &prev_row,
                Some(&prev_row),    // lossy embed override
                cur_pos + k,
                trunk_weights,
            )?;
        }
        gpu.hip.memcpy_dtod_at(
            &state.mtp_t_outs.buf, k * dim_bytes,
            &state.mtp_scratch.t_mtp_out.buf, 0,
            dim_bytes,
        )?;
    }

    // ── 1b. Batched COMPRESSED lm_head over all K t_mtp_outs ──────────────
    //
    // Single GEMM with M=cvs (vs M=vocab=248K in the plain path) — this is
    // the FastMTP-style ~7.5x BW reduction lever. shared_head_norm applied
    // row-wise inside the helper. MQ4 path includes the FWHT pre-pass.
    let t_outs_view = state.mtp_t_outs.sub_offset(0, max_n * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, max_n * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, max_n * dim);
    let lm_logits_view = logits_c_batched.sub_offset(0, max_n * cvs);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu, head, lm_head_draft,
        &t_outs_view, &lm_tmp_view, &lm_rot_view, &lm_logits_view,
        max_n,
    )?;

    // Batched argmax over K compressed logit rows → K i32 ids → remap.
    let lm_argmax_view = state.mtp_lm_argmax.sub_offset(0, max_n);
    gpu.argmax_f32_batched(&lm_logits_view, &lm_argmax_view, cvs, max_n)?;
    let mut argmax_host: Vec<i32> = vec![0; max_n];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_host.as_mut_ptr() as *mut u8, max_n * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &lm_argmax_view.buf)?;
    }
    let mut candidates: Vec<u32> = Vec::with_capacity(max_n);
    for raw in argmax_host {
        let draft_idx = raw as usize;
        assert!(
            draft_idx < cvs,
            "draft argmax {draft_idx} out of compressed vocab {cvs}"
        );
        candidates.push(vocab_map[draft_idx]);
    }

    // ── 2. Trunk verify on [last_committed, c1, ..., c_max_n] ─────────────
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    let n_verify = verify_tokens.len();

    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    qwen35::forward_prefill_batch_with_pbs(
        gpu,
        trunk_weights,
        &target.config,
        &verify_tokens,
        cur_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None,
        Some(&state.verify_hidden),
        None,
        None,
        Some(&state.trunk_pbs),
        None,
    )?;

    // ── 3. Trunk batched lm_head over verify positions ─────────────────────
    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        _ => {
            for i in 0..n_verify {
                let row = state.verify_hidden.sub_offset(i * dim, dim);
                let logits_row = state.verify_logits.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &row, &logits_row)?;
            }
        }
    }

    let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
    let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_v_host.as_mut_ptr() as *mut u8, n_verify * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_v_host.into_iter().map(|x| x as u32).collect();

    // ── 4. Accept-prefix (K-general) ──────────────────────────────────────
    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(max_n + 1);
    for k in 0..max_n {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }
    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= max_n + 1);

    // ── 5. Capture prev_hidden from verify slot advance-1 ─────────────────
    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, prev_hidden_row, dim)?;

    // ── 6. Roll back trunk DN state + replay accepted ─────────────────────
    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if advance >= 2 {
        let replay = &verify_tokens[..advance];
        qwen35::forward_prefill_batch(
            gpu,
            trunk_weights,
            &target.config,
            replay,
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None, None, None, None,
        )?;
    } else {
        qwen35::forward_scratch(
            gpu,
            trunk_weights,
            &target.config,
            verify_tokens[0],
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
        )?;
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
    })
}

// ─── FastMTP-style compressed SERIAL spec step (K serial roundtrips) ──────
//
// Variant of [`spec_step_mtp_compressed`] that runs K MTP steps as
// DISCRETE-TOKEN ROUNDTRIPS (not lossy chain): each step argmaxes its
// compressed logits, remaps via vocab_map to a full-vocab token id, and
// uses that token id (embedded via trunk's `token_embd` table) as the
// next step's input. Mirrors the original Task 10 "serial lm_head" path
// — but with COMPRESSED lm_head dispatches (~7.5x BW saving each).
//
// Per-cycle BW (27B-3.5, K=3, projection):
//   3 full lm_head GEMVs:        3 × 635 MB = 1.9 GB MQ4  (Task 10 baseline)
//   3 compressed lm_head GEMVs:  3 × 84 MB  = 252 MB MQ4  (this path)
//   Saved: ~1.65 GB BW per cycle → ~21-27 ms per cycle on gfx1100.
//
// Task 10 plain serial measured 39.68 tok/s τ=3.08; this path projects
// ~50-58 tok/s at the same τ (gate is 60). A trained sidecar (hiptrx
// trunk-argmax distillation) lifting τ from 3.08 → 3.5+ should clear.
//
// Trade-off vs the batched lossy chain: K serial dispatches launch K
// times instead of 1 batched, costing extra launch overhead and per-step
// argmax+D2H. On a 248K-vocab model this is dwarfed by the BW savings.
// On smaller-vocab models the lossy batched path may still be faster.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed_serial(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Sidecar must be loaded.
    let _ = head.weights.lm_head_draft.as_ref()
        .expect("spec_step_mtp_compressed_serial requires --vocab-sidecar");
    let vocab_map = head.weights.lm_head_draft_vocab_map.as_ref()
        .expect("compressed head missing vocab_map");
    let cvs = head.weights.compressed_vocab_size
        .expect("compressed head missing compressed_vocab_size");
    let logits_c_single = state.mtp_scratch.logits_compressed.as_ref()
        .expect("Qwen35MtpHeadScratch::logits_compressed not allocated; \
                 call mtp_scratch.ensure_compressed_logits(gpu, cvs) after head load");

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    let dim_bytes = dim * 4;
    let mut candidates: Vec<u32> = Vec::with_capacity(max_n);
    let argmax_view = state.mtp_lm_argmax.sub_offset(0, 1);

    // ── 1. K serial discrete-token roundtrips ─────────────────────────────
    for k in 0..max_n {
        let next_tok = if k == 0 { last_committed } else { candidates[k - 1] };
        if k == 0 {
            mtp_head::mtp_head_forward_compressed(
                gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                next_tok, &state.prev_hidden, cur_pos + k, trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_compressed(
                gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                next_tok, &prev_row, cur_pos + k, trunk_weights,
            )?;
        }

        gpu.argmax_f32_batched(logits_c_single, &argmax_view, cvs, 1)?;
        let mut argmax_host: [i32; 1] = [0];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    argmax_host.as_mut_ptr() as *mut u8, 4,
                )
            };
            gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
        }
        let draft_idx = argmax_host[0] as usize;
        assert!(draft_idx < cvs, "draft argmax {draft_idx} out of cvs {cvs}");
        candidates.push(vocab_map[draft_idx]);

        if k + 1 < max_n {
            gpu.hip.memcpy_dtod_at(
                &state.mtp_t_outs.buf, k * dim_bytes,
                &state.mtp_scratch.t_mtp_out.buf, 0,
                dim_bytes,
            )?;
        }
    }

    // ── 2. Trunk verify ───────────────────────────────────────────────────
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    let n_verify = verify_tokens.len();

    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    qwen35::forward_prefill_batch_with_pbs(
        gpu, trunk_weights, &target.config, &verify_tokens, cur_pos,
        &mut target.kv_cache, &mut target.dn_state, &target.scratch,
        None, Some(&state.verify_hidden), None, None,
        Some(&state.trunk_pbs), None,
    )?;

    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            gpu.rotate_x_mq_batched(&state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        _ => {
            for i in 0..n_verify {
                let row = state.verify_hidden.sub_offset(i * dim, dim);
                let logits_row = state.verify_logits.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &row, &logits_row)?;
            }
        }
    }

    let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
    let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_v_host.as_mut_ptr() as *mut u8, n_verify * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_v_host.into_iter().map(|x| x as u32).collect();

    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(max_n + 1);
    for k in 0..max_n {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }
    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= max_n + 1);

    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, prev_hidden_row, dim)?;

    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if advance >= 2 {
        let replay = &verify_tokens[..advance];
        qwen35::forward_prefill_batch(
            gpu, trunk_weights, &target.config, replay, cur_pos,
            &mut target.kv_cache, &mut target.dn_state, &target.scratch,
            None, None, None, None,
        )?;
    } else {
        qwen35::forward_scratch(
            gpu, trunk_weights, &target.config, verify_tokens[0], cur_pos,
            &mut target.kv_cache, &mut target.dn_state, &target.scratch,
        )?;
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
    })
}
