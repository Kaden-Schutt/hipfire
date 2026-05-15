//! DFlash + MTP linear-chain composition (Task 11).
//!
//! Per-cycle:
//!   1. Run dflash drafter (B-1 candidates + seed).
//!   2. Run K MTP block-only forwards in chain. Step 0's `prev_hidden` is
//!      the LAST drafter hidden (post-final-norm at slot B-1) which lives
//!      in `draft_scratch.x`. `next_token` for step 0 is `drafted[B-1]`
//!      (the last dflash candidate). Steps 1..K-1 chain feature-only
//!      (lossy) per Task 10b's `mtp_head_apply_lm_head_batched` pattern.
//!   3. Single batched lm_head over K t_mtp_outs → K MTP candidates.
//!   4. Build composite chain `[seed, c_1, ..., c_{B-1}, m_1, ..., m_K]`
//!      (length B+K) and run a single trunk verify on it.
//!   5. Greedy accept-prefix: longest i with composite[i+1] == argmax_per_pos[i].
//!   6. bonus_token = argmax_per_pos[accept_len]. committed = composite tokens
//!      up through accept_len plus bonus. Roll back trunk DN state +
//!      replay accepted tokens like spec_step_dflash does.
//!
//! ## KV management
//!
//! Drafter KV (`target.kv_cache` for dflash uses target's cache via verify):
//! identical to spec_step_dflash — verify writes B+K positions then snapshot
//! restore + replay rewinds to (cur_pos + accept_len + 1).
//!
//! MTP head KV (private cache `mtp_kv`): each MTP step k writes slot
//! `cur_pos + B - 1 + k`. After verify, slots beyond accepted range are
//! stale but get overwritten in next cycle (by either MTP fanout or are
//! beyond the next cycle's writes — same pattern as `mtp_spec.rs`).
//!
//! KEY CAVEAT: MTP attention will see HOLES at trunk-only positions
//! (positions cur_pos..cur_pos+B-2 between cycles). Per the existing
//! `mtp_spec.rs` design, this degrades MTP candidate quality but does NOT
//! break correctness — trunk verify rejects bad MTP candidates and the
//! system falls back to dflash + bonus.
//!
//! ## Why drafter hidden as MTP prev_hidden
//!
//! MTP head was trained on trunk's post-output-norm hidden states. The
//! drafter's post-final-norm hidden at slot B-1 is `dim`-dimensional
//! (matched drafter has `cfg.hidden == trunk.dim`) and is trained to
//! mimic trunk's hidden by the dflash distillation objective. Lossy
//! substitution acceptable since trunk verify is the correctness gate.
//!
//! ## Why this might be a net loss
//!
//! Trunk verify cost grows linearly with B+K (vs B for dflash baseline).
//! MTP candidates only contribute when dflash full-accepts (every cycle's
//! `accept_dflash == B - 1`). For τ_dflash ≈ 10 with B=16, full-accept
//! cycles are uncommon, so MTP slot work is wasted compute most cycles.
//! This module is a research artifact to MEASURE the actual lift.

use crate::mtp_head::{self, Qwen35MtpHead, Qwen35MtpHeadKvCache, Qwen35MtpHeadScratch};
use crate::qwen35::{self, Qwen35Weights};
use crate::speculative::{
    self, DeltaNetSnapshot, DflashVerifyOutput, GdnTape, HiddenStateRingBuffer, ModelSlot,
    VerifyScratch,
};
use hip_bridge::HipResult;
use hipfire_runtime::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::llama;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Public state ────────────────────────────────────────────────────────

/// All per-generation buffers needed by [`spec_step_dflash_mtp`]. Holds the
/// MTP-side scratch; dflash-side scratch (DflashScratch, HiddenStateRingBuffer,
/// VerifyScratch, DeltaNetSnapshot, GdnTape) is owned by the caller.
pub struct MtpComposeState {
    /// MTP head per-call scratch.
    pub mtp_scratch: Qwen35MtpHeadScratch,
    /// MTP head's private KV cache.
    pub mtp_kv: Qwen35MtpHeadKvCache,
    /// Per-step `t_mtp_out` capture buffer for the K-step chain. Shape
    /// `[max_k, n_embd]` row-major.
    pub mtp_t_outs: GpuTensor,
    /// Batched-rmsnorm scratch for the end-of-chain lm_head. Shape
    /// `[max_k, n_embd]`.
    pub mtp_lm_tmp: GpuTensor,
    /// FWHT-rotated x scratch for MQ-family lm_heads. Shape
    /// `[max_k, n_embd]`. Unused for non-MQ.
    pub mtp_lm_rot: GpuTensor,
    /// Batched MTP candidate logits. Shape `[max_k, vocab]`.
    pub mtp_lm_logits: GpuTensor,
    /// GPU-side argmax destination over `mtp_lm_logits`. Shape `[max_k]`.
    pub mtp_lm_argmax: GpuTensor,
    /// Maximum K candidates per cycle.
    pub max_k: usize,
}

impl MtpComposeState {
    /// Allocate per-generation MTP buffers. Caller still allocates and owns
    /// dflash-side scratch (DflashScratch, hidden ring buffer, verify scratch).
    pub fn new(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_k: usize,
    ) -> HipResult<Self> {
        assert!(max_k >= 1, "MtpComposeState: max_k must be >= 1");
        let dim = target.config.dim;
        let vocab = target.config.vocab_size;
        assert_eq!(
            head.config.n_embd, dim,
            "MtpComposeState: trunk dim={dim} but head n_embd={}",
            head.config.n_embd,
        );
        assert_eq!(
            head.config.vocab_size, vocab,
            "MtpComposeState: trunk vocab={vocab} but head vocab={}",
            head.config.vocab_size,
        );

        let mtp_scratch = Qwen35MtpHeadScratch::new(gpu, &head.config)?;
        let mtp_kv = Qwen35MtpHeadKvCache::new(gpu, &head.config)?;
        let mtp_t_outs = gpu.alloc_tensor(&[max_k * dim], DType::F32)?;
        let mtp_lm_tmp = gpu.alloc_tensor(&[max_k * dim], DType::F32)?;
        let mtp_lm_rot = gpu.alloc_tensor(&[max_k * dim], DType::F32)?;
        let mtp_lm_logits = gpu.alloc_tensor(&[max_k * vocab], DType::F32)?;
        let mtp_lm_argmax = gpu.alloc_tensor(&[max_k], DType::F32)?;

        Ok(Self {
            mtp_scratch,
            mtp_kv,
            mtp_t_outs,
            mtp_lm_tmp,
            mtp_lm_rot,
            mtp_lm_logits,
            mtp_lm_argmax,
            max_k,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.mtp_t_outs);
        let _ = gpu.free_tensor(self.mtp_lm_tmp);
        let _ = gpu.free_tensor(self.mtp_lm_rot);
        let _ = gpu.free_tensor(self.mtp_lm_logits);
        let _ = gpu.free_tensor(self.mtp_lm_argmax);
        self.mtp_scratch.free_gpu(gpu);
        self.mtp_kv.free_gpu(gpu);
    }
}

// ─── Result ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MtpComposeResult {
    /// Number of dflash candidates accepted (0..=B-1).
    pub accept_dflash: usize,
    /// Number of MTP candidates accepted (0..=K). 0 unless dflash accepted
    /// the FULL B-1 chain (otherwise the verify accept-prefix stops before
    /// MTP slots).
    pub accept_mtp: usize,
    /// The bonus token (target's argmax at the first rejection point).
    pub bonus_token: u32,
    /// All B+K drafted tokens (`[c_1..c_{B-1}, m_1..m_K]` after the seed).
    pub drafted: Vec<u32>,
    /// Tokens committed THIS cycle. Includes the seed re-confirm at slot 0
    /// just like `SpecStepResult.committed` (= [seed, accepted, bonus]).
    pub committed: Vec<u32>,
}

// ─── One spec step ───────────────────────────────────────────────────────

/// One DFlash + MTP composition cycle. Greedy / temp=0 only.
///
/// Mirrors the call surface of `spec_step_dflash` but uses a stripped-down
/// arg list — caller uses this directly, not via the dflash demo's full
/// adaptive-B + PLD + n-gram + repeat-penalty knobs (those compose later
/// if MTP shows a win).
#[allow(clippy::too_many_arguments)]
pub fn spec_step_dflash_mtp(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_snap: &mut DeltaNetSnapshot,
    verify_scratch: &VerifyScratch,
    gdn_tape: Option<&mut GdnTape>,
    head: &Qwen35MtpHead,
    state: &mut MtpComposeState,
    position: usize,
    seed_token: u32,
    dflash_b: Option<usize>,
    mtp_k: usize,
) -> HipResult<MtpComposeResult> {
    let trunk_weights: &Qwen35Weights = &target.weights;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;

    let b = dflash_b.unwrap_or(draft_cfg.block_size);
    assert!(b >= 2, "dflash block size must be >= 2");
    assert!(mtp_k >= 1, "mtp_k must be >= 1");
    assert!(
        mtp_k <= state.max_k,
        "spec_step_dflash_mtp: mtp_k={mtp_k} > max_k={}",
        state.max_k,
    );

    let h = draft_cfg.hidden;
    assert_eq!(
        h, dim,
        "spec_step_dflash_mtp: drafter hidden ({}) must match trunk dim ({}) — \
         use a matched drafter (not a different-size one)",
        h, dim,
    );
    let _ne = draft_cfg.num_extract();
    let mask_token = draft_cfg.mask_token_id;

    // Stream sanity, mirrors spec_step_dflash.
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. DFlash drafter (inline copy, simpler than calling spec_step_dflash
    // and immediately discarding its verify) ─────────────────────────────
    //
    // Build [seed, mask, mask, ...] block.
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // D2D embed each block slot via target's embedding table.
    let dim_bytes = dim * 4;
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("dflash_mtp: unsupported target embedding format"),
        }
    }

    // Positions (no eviction support — this is a v1 path, no FlashCASK).
    let effective_ctx_len = draft_scratch.target_hidden_abs_positions.len().min(position);
    let co = target.kv_cache.compact_offset as i32;
    let positions_q: Vec<i32> =
        ((position as i32 + co)..(position as i32 + b as i32 + co)).collect();
    let positions_k: Vec<i32> = {
        let mut v = Vec::with_capacity(effective_ctx_len + b);
        let th_abs = &draft_scratch.target_hidden_abs_positions;
        let start_idx = th_abs.len().saturating_sub(effective_ctx_len);
        v.extend_from_slice(&th_abs[start_idx..]);
        for p in 0..b {
            v.push(position as i32 + p as i32 + co);
        }
        v
    };

    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        None,
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // Drafter lm_head via target's output to extract drafted candidates.
    let w_out = &target.weights.output;
    let mut drafted: Vec<u32> = vec![seed_token];
    {
        let batch = b - 1;
        assert!(
            batch <= verify_scratch.max_n,
            "verify_scratch max_n {} < draft batch {}",
            verify_scratch.max_n, batch,
        );
        let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
        let logits_batch = verify_scratch.logits.sub_offset(0, batch * vocab);
        match w_out.gpu_dtype {
            DType::Q8_0 => {
                gpu.gemm_q8_0_batched(
                    &w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            DType::HFQ4G256 => {
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            DType::MQ4G256 => {
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            DType::MQ3G256 => {
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq3g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            DType::HFQ6G256 => {
                gpu.gemm_hfq6g256_batched_lmhead(
                    &w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            DType::MQ6G256 => {
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq6g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            _ => {
                // Fallback per-row gemv.
                for i in 1..b {
                    let hidden_row = draft_scratch.x.sub_offset(i * h, h);
                    llama::weight_gemv(
                        gpu, w_out, &hidden_row, &target.scratch.logits,
                    )?;
                    let logits = gpu.download_f32(&target.scratch.logits)?;
                    drafted.push(argmax_u32(&logits));
                }
            }
        }
        // Use GPU-batched argmax (saves the per-row D2H of (b-1) × vocab).
        if drafted.len() == 1 {
            let argmax_buf = verify_scratch.argmax.sub_offset(0, batch);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, batch)?;
            let mut host_idx = vec![0i32; batch];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, batch * 4)
                };
                gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
            }
            for &idx in &host_idx {
                drafted.push(idx as u32);
            }
        }
    }
    debug_assert_eq!(drafted.len(), b);

    // Reflect drafted into block (positions 1..b are the drafter's argmax).
    for i in 1..b {
        block[i] = drafted[i];
    }

    // ── 2. MTP fanout (K-step chain) ────────────────────────────────────
    //
    // prev_hidden for step 0 = drafter's post-final-norm hidden at slot B-1
    // (drafter's predicted hidden for position cur_pos + B - 1, which is
    // where drafted[B-1] would land).
    //
    // next_token for step 0 = drafted[B-1] (the drafter's argmax token at
    // that slot — what the drafter believes the next token is).
    //
    // Steps 1..K-1: feature-only chain (same lossy pattern as
    // mtp_head_apply_lm_head_batched + mtp_spec.rs Approach B).
    //
    // KV writes: step k writes MTP slot `position + b - 1 + k`. Bound check:
    // position + b - 1 + (K - 1) < kv.max_seq.
    let drafter_hidden_last = draft_scratch.x.sub_offset((b - 1) * h, h);
    let mtp_pos_base = position + b - 1;
    assert!(
        mtp_pos_base + mtp_k <= state.mtp_kv.max_seq,
        "mtp_pos_base + mtp_k ({}) > mtp_kv.max_seq ({})",
        mtp_pos_base + mtp_k, state.mtp_kv.max_seq,
    );

    for k in 0..mtp_k {
        if k == 0 {
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                drafted[b - 1],
                &drafter_hidden_last,
                None,
                mtp_pos_base + k,
                trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                0,
                &prev_row,
                Some(&prev_row),
                mtp_pos_base + k,
                trunk_weights,
            )?;
        }
        gpu.hip.memcpy_dtod_at(
            &state.mtp_t_outs.buf,
            k * dim_bytes,
            &state.mtp_scratch.t_mtp_out.buf,
            0,
            dim_bytes,
        )?;
    }

    // ── 3. Batched MTP lm_head (K rows → K logits) ──────────────────────
    let t_outs_view = state.mtp_t_outs.sub_offset(0, mtp_k * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, mtp_k * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, mtp_k * dim);
    let lm_logits_view = state.mtp_lm_logits.sub_offset(0, mtp_k * vocab);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu,
        head,
        &trunk_weights.output,
        &t_outs_view,
        &lm_tmp_view,
        &lm_rot_view,
        &lm_logits_view,
        mtp_k,
    )?;

    let lm_argmax_view = state.mtp_lm_argmax.sub_offset(0, mtp_k);
    gpu.argmax_f32_batched(&lm_logits_view, &lm_argmax_view, vocab, mtp_k)?;
    let mut argmax_host: Vec<i32> = vec![0; mtp_k];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, mtp_k * 4)
        };
        gpu.hip.memcpy_dtoh(bytes, &lm_argmax_view.buf)?;
    }
    let mtp_candidates: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // ── 4. Composite verify chain: [seed, c_1..c_{B-1}, m_1..m_K] ────────
    let n_verify = b + mtp_k;
    let mut composite: Vec<u32> = Vec::with_capacity(n_verify);
    composite.extend_from_slice(&block);
    composite.extend_from_slice(&mtp_candidates);
    debug_assert_eq!(composite.len(), n_verify);

    // Snapshot trunk DN state for rollback.
    target_snap.save_from(&target.dn_state, gpu)?;

    // MoE-aware: tape capture is lossy on MoE per the spec_step_dflash
    // comment block. Preserve the same gate.
    let mut gdn_tape_opt = gdn_tape;
    let target_has_moe = target.weights.layers.iter().any(|lw| matches!(
        lw,
        qwen35::LayerWeights::DeltaNetMoe(_) | qwen35::LayerWeights::FullAttnMoe(_),
    ));
    if target_has_moe {
        gdn_tape_opt = None;
    }

    let verify_out: DflashVerifyOutput = speculative::verify_dflash_block(
        gpu,
        target,
        &composite,
        position,
        hidden_rb,
        gdn_tape_opt.as_deref_mut(),
        false, // greedy / temp=0
        verify_scratch,
    )?;

    // ── 5. Greedy accept-prefix over composite ──────────────────────────
    //
    // For each i in 0..n_verify-1: argmax_per_pos[i] is trunk's prediction
    // for position position + i + 1 given inputs composite[0..=i]. Accept
    // composite[i+1] if it matches.
    let argmax = &verify_out.argmax_per_pos;
    debug_assert_eq!(argmax.len(), n_verify);

    let mut accept_len = 0usize;
    for i in 0..n_verify - 1 {
        if argmax[i] == composite[i + 1] {
            accept_len += 1;
        } else {
            break;
        }
    }
    let bonus_token = argmax[accept_len];

    // Decompose accept_len into dflash + MTP portions.
    let dflash_max_accept = b - 1; // candidates after seed
    let accept_dflash = accept_len.min(dflash_max_accept);
    let accept_mtp = accept_len.saturating_sub(dflash_max_accept);

    // ── 6. Build committed = [seed, accepted..., bonus] ─────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for i in 0..accept_len {
        committed.push(composite[i + 1]);
    }
    committed.push(bonus_token);

    // ── 7. Append accepted target hidden rows to draft_scratch.target_hidden ─
    //
    // Same pattern as spec_step_dflash. Verify wrote n_verify rows into
    // hidden_rb; we keep the first `accept_dflash + 1` rows (positions
    // [position..position + accept_dflash + 1)) for the next cycle's draft
    // forward. MTP slots are NOT scattered — they're past the dflash chain
    // and don't feed back into draft_forward (the drafter only attends
    // positions through its own context).
    //
    // Actually we want to keep all accepted positions so the drafter can use
    // them next cycle, INCLUDING any MTP-accepted slots. The drafter's
    // attention reads target_hidden up to the current cycle's start, which
    // for next cycle is `position + accept_len + 1`. So scatter
    // accept_len + 1 rows.
    //
    // BUT: hidden_rb only holds the FIRST B rows used by dflash verify.
    // verify_dflash_block writes B+K rows; check whether hidden_rb is sized
    // for that. The caller should pre-size with max_block_size = B + max_K.
    let rows_to_keep = accept_len + 1;
    speculative::scatter_hidden_block_to_interleaved(
        gpu,
        hidden_rb,
        &draft_scratch.target_hidden,
        position,
        n_verify,
        rows_to_keep,
    )?;
    draft_scratch.uploaded_target_hidden_rows = position + rows_to_keep;
    let co = target.kv_cache.compact_offset as i32;
    for p in 0..rows_to_keep {
        draft_scratch
            .target_hidden_abs_positions
            .push(position as i32 + p as i32 + co);
    }

    // ── 8. Rollback trunk DN state + replay accepted committed tokens ────
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    if let Some(tape) = gdn_tape_opt.as_deref() {
        tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
    } else {
        let replay_tokens = &committed[..accept_len + 1];
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            replay_tokens,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None,
            None,
            None,
            None,
        )?;
    }

    Ok(MtpComposeResult {
        accept_dflash,
        accept_mtp,
        bonus_token,
        drafted: composite,
        committed,
    })
}

// ─── Helpers (module-private) ────────────────────────────────────────────

fn argmax_u32(v: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i as u32;
        }
    }
    best
}
