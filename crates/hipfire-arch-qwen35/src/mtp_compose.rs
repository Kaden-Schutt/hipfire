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

use crate::mtp_head::{
    self, Qwen35MtpHead, Qwen35MtpHeadBatchedScratch, Qwen35MtpHeadKvCache, Qwen35MtpHeadScratch,
};
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
        // Qwen35MtpHeadKvCache::free_gpu does `drop(inner)` which does not
        // release GPU memory (llama::KvCache has no Drop). Call the inner
        // KvCache's own free_gpu directly to properly hipFree each tensor.
        self.mtp_kv.inner.free_gpu(gpu);
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
                llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            DType::MQ3G256 => {
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
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
                llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
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

// ─── Per-slot tree composition (Task 11b) ────────────────────────────────

/// Per-generation buffers for [`spec_step_dflash_mtp_tree`]. Holds:
/// - Batched MTP head scratch sized for `B × K` MTP forwards per cycle
/// - MTP head's private KV cache
/// - Pre-allocated per-cycle host buffers for tree construction
/// - Batched lm_head + argmax destinations
pub struct MtpComposeTreeState {
    /// MTP head batched scratch sized for `max_b * max_k` slots per cycle.
    pub mtp_scratch: Qwen35MtpHeadBatchedScratch,
    /// MTP head's private KV cache (single-layer).
    pub mtp_kv: Qwen35MtpHeadKvCache,
    /// Stacked drafter hiddens [max_b * max_k, n_embd] used as MTP prev_hidden
    /// (each MTP child gets its parent dflash slot's hidden replicated K times).
    pub prev_hiddens_stacked: GpuTensor,
    /// Per-MTP-call rotated-x scratch for MQ4 weights inside batched gemm.
    /// Sized to the widest k across MTP head 2D weights × `max_b * max_k`.
    pub gemm_rotate_scratch: GpuTensor,
    /// Batched-rmsnorm temp for the end-of-call lm_head. [max_b*max_k, n_embd].
    pub mtp_lm_tmp: GpuTensor,
    /// FWHT-rotated x for MQ-family lm_heads. [max_b*max_k, n_embd]. Unused
    /// for non-MQ.
    pub mtp_lm_rot: GpuTensor,
    /// Batched MTP candidate logits. [max_b*max_k, vocab].
    pub mtp_lm_logits: GpuTensor,
    /// GPU-side argmax destination. [max_b*max_k] f32 (i32 alias).
    pub mtp_lm_argmax: GpuTensor,
    pub max_b: usize,
    pub max_k: usize,
}

impl MtpComposeTreeState {
    pub fn new(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_b: usize,
        max_k: usize,
    ) -> HipResult<Self> {
        assert!(max_b >= 2, "MtpComposeTreeState: max_b must be >= 2");
        assert!(max_k >= 1, "MtpComposeTreeState: max_k must be >= 1");
        let dim = target.config.dim;
        let vocab = target.config.vocab_size;
        assert_eq!(head.config.n_embd, dim,
            "MtpComposeTreeState: trunk dim={dim} but head n_embd={}", head.config.n_embd);
        assert_eq!(head.config.vocab_size, vocab,
            "MtpComposeTreeState: trunk vocab={vocab} but head vocab={}", head.config.vocab_size);

        let max_n = max_b * max_k;
        let mtp_scratch = Qwen35MtpHeadBatchedScratch::new(gpu, &head.config, max_n)?;
        let mtp_kv = Qwen35MtpHeadKvCache::new(gpu, &head.config)?;

        // Widest k over MTP head's 2D weights = max(eh_proj.k=2*dim, n_ff, dim, n_head*head_dim).
        let widest_k = (2 * dim)
            .max(head.config.n_ff)
            .max(dim)
            .max(head.config.n_head * head.config.head_dim);

        Ok(Self {
            mtp_scratch,
            mtp_kv,
            prev_hiddens_stacked: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            gemm_rotate_scratch: gpu.alloc_tensor(&[max_n * widest_k], DType::F32)?,
            mtp_lm_tmp: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            mtp_lm_rot: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            mtp_lm_logits: gpu.alloc_tensor(&[max_n * vocab], DType::F32)?,
            mtp_lm_argmax: gpu.alloc_tensor(&[max_n], DType::F32)?,
            max_b,
            max_k,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.prev_hiddens_stacked);
        let _ = gpu.free_tensor(self.gemm_rotate_scratch);
        let _ = gpu.free_tensor(self.mtp_lm_tmp);
        let _ = gpu.free_tensor(self.mtp_lm_rot);
        let _ = gpu.free_tensor(self.mtp_lm_logits);
        let _ = gpu.free_tensor(self.mtp_lm_argmax);
        self.mtp_scratch.free_gpu(gpu);
        // Qwen35MtpHeadKvCache::free_gpu does `drop(inner)` which does not
        // release GPU memory (llama::KvCache has no Drop). Call the inner
        // KvCache's own free_gpu directly to properly hipFree each tensor.
        self.mtp_kv.inner.free_gpu(gpu);
    }
}

/// Result of one [`spec_step_dflash_mtp_tree`] cycle.
#[derive(Debug, Clone)]
pub struct MtpComposeTreeResult {
    /// Number of dflash candidates accepted along the committed path
    /// (0..=B-1).
    pub accept_dflash: usize,
    /// Number of MTP candidates accepted along the committed path (0..=K).
    /// Non-zero whenever the trunk's argmax at the LAST accepted dflash slot
    /// matches one of that slot's K MTP children.
    pub accept_mtp: usize,
    /// The bonus token (target's argmax at the first rejection point).
    pub bonus_token: u32,
    /// Tokens committed THIS cycle. Includes seed at index 0:
    /// `[seed, accepted..., bonus]`.
    pub committed: Vec<u32>,
    /// Total tree node count (1 root + B-1 dflash + (B-1)*K MTP children).
    pub tree_nodes: usize,
}

/// One DFlash + per-slot MTP tree spec-decode cycle (Task 11b).
///
/// Tree shape (for B=16, K=2):
/// ```text
///   root (seed @ position)
///    ├── dflash_c1 @ position+1
///    │    ├── mtp_c1_a @ position+2
///    │    └── mtp_c1_b @ position+2
///    ├── dflash_c2 @ position+2
///    │    ├── mtp_c2_a @ position+3
///    │    └── mtp_c2_b @ position+3
///    ...
///    └── dflash_c{B-1} @ position+B-1
///         ├── mtp_c{B-1}_a @ position+B
///         └── mtp_c{B-1}_b @ position+B
/// ```
///
/// Total nodes: `1 + (B-1) + (B-1)*K`. For B=16, K=2: `1 + 15 + 30 = 46`.
///
/// Linearization order (matches positions for tree mask construction):
/// - slot 0: root (seed)
/// - slots 1..B: dflash_c1..dflash_c{B-1} (positions+1..position+B-1)
/// - slots B..B + (B-1)*K: MTP children, grouped by parent dflash slot
///
/// Tree-attention mask: each MTP child sees root + its parent dflash slot ONLY.
/// Each dflash slot sees root + previous dflash slots (causal chain).
///
/// ## Why this might cross the +20% gate
///
/// Linear-chain composition wasted MTP compute on cycles where dflash dropped
/// out before reaching MTP slots (~75% of cycles for τ_dflash=10, B=16). Per-slot
/// tree gives the LAST accepted dflash slot's MTP children a chance to provide
/// extra accept-prefix length, even when dflash drops out early.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_dflash_mtp_tree(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_snap: &mut DeltaNetSnapshot,
    verify_scratch: &VerifyScratch,
    ddtree_scratch: &speculative::DdtreeScratch,
    gdn_tape: Option<&mut GdnTape>,
    head: &Qwen35MtpHead,
    state: &mut MtpComposeTreeState,
    position: usize,
    seed_token: u32,
    dflash_b: Option<usize>,
    mtp_k: usize,
) -> HipResult<MtpComposeTreeResult> {
    let trunk_weights: &Qwen35Weights = &target.weights;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;

    let b = dflash_b.unwrap_or(draft_cfg.block_size);
    assert!(b >= 2, "dflash block size must be >= 2");
    assert!(mtp_k >= 1, "mtp_k must be >= 1");
    assert!(b <= state.max_b, "b={b} > state.max_b={}", state.max_b);
    assert!(mtp_k <= state.max_k, "mtp_k={mtp_k} > state.max_k={}", state.max_k);

    let h = draft_cfg.hidden;
    assert_eq!(h, dim, "drafter hidden ({h}) must match trunk dim ({dim})");
    let mask_token = draft_cfg.mask_token_id;

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. DFlash drafter (inline, mirrors spec_step_dflash_mtp linear) ─
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

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
            _ => panic!("dflash_mtp_tree: unsupported target embedding format"),
        }
    }

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
        gpu, draft_weights, draft_cfg, None, None,
        &positions_q, &positions_k, b, effective_ctx_len, draft_scratch,
    )?;

    // Drafter lm_head over slots 1..B → drafted candidates.
    let w_out = &target.weights.output;
    let mut drafted: Vec<u32> = vec![seed_token];
    {
        let batch = b - 1;
        assert!(batch <= verify_scratch.max_n,
            "verify_scratch max_n {} < draft batch {}", verify_scratch.max_n, batch);
        let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
        let logits_batch = verify_scratch.logits.sub_offset(0, batch * vocab);
        match w_out.gpu_dtype {
            DType::Q8_0 => gpu.gemm_q8_0_batched(
                &w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)?,
            DType::HFQ4G256 => gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)?,
            DType::MQ4G256 => {
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch)?;
            }
            DType::MQ3G256 => {
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq3g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch)?;
            }
            DType::HFQ6G256 => gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)?,
            DType::MQ6G256 => {
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq6g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch)?;
            }
            _ => panic!("dflash_mtp_tree: unsupported drafter lm_head dtype"),
        }
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
    debug_assert_eq!(drafted.len(), b);
    for i in 1..b {
        block[i] = drafted[i];
    }

    // ── 2. MTP fanout: B-1 slots × K children, batched in ONE call ───────
    //
    // For each dflash slot `i` in 1..B, we want K MTP children. Their
    // `prev_hidden` is the drafter's hidden at slot i (for i in 1..B), and
    // their `next_token` is `drafted[i]` for the FIRST of K (k=0). For k>0
    // we still feed `drafted[i]` since v1 can't easily get K different
    // children without an additional logits pass — instead we'll take the
    // top-K from the trunk's argmax at slot i. But we don't have the trunk
    // posterior yet.
    //
    // Simplification: K=1 means one MTP child per dflash slot, fed with
    // drafted[i] as next_token. K>1: we replicate the same MTP forward but
    // in v1 just take top-K from MTP head's logits (which we get cheaply
    // since we run lm_head batched anyway). For K=1 the children are the
    // greedy MTP prediction; for K=2 we add the runner-up; for K=3 the next.
    //
    // To get top-K cleanly, we run the MTP head ONCE per dflash slot (so n =
    // B-1 forwards, NOT B-1 * K), then take top-K from the resulting logits.
    // That yields K candidates per slot at the cost of N=B-1 MTP forwards
    // (much cheaper than N=(B-1)*K) — and is a strict superset of K=1's
    // information.
    let n_mtp_forwards = b - 1;
    let mtp_pos_base = position + 1;

    // Stack drafter hiddens for slots 1..B as MTP prev_hidden inputs.
    // draft_scratch.x[i*h..] is slot i's hidden; we need rows 1..B.
    gpu.hip.memcpy_dtod_at(
        &state.prev_hiddens_stacked.buf, 0,
        &draft_scratch.x.buf, dim_bytes,
        n_mtp_forwards * dim_bytes,
    )?;

    // Per-MTP-slot positions: mtp_c{i} sits at position + i + 1 (one beyond
    // the dflash slot it attaches to, which lives at position + i).
    let mtp_positions: Vec<i32> =
        (0..n_mtp_forwards).map(|i| (mtp_pos_base + i) as i32).collect();
    let mtp_next_tokens: Vec<u32> =
        (0..n_mtp_forwards).map(|i| drafted[i + 1]).collect();

    // Bound check vs MTP cache.
    assert!(
        mtp_pos_base + n_mtp_forwards <= state.mtp_kv.max_seq,
        "MTP cache too small: pos_base + n = {} > max_seq {}",
        mtp_pos_base + n_mtp_forwards, state.mtp_kv.max_seq,
    );

    mtp_head::mtp_head_forward_block_batched(
        gpu,
        head,
        &mut state.mtp_scratch,
        &mut state.mtp_kv,
        &mtp_next_tokens,
        &state.prev_hiddens_stacked,
        &mtp_positions,
        n_mtp_forwards,
        trunk_weights,
        Some(&state.gemm_rotate_scratch),
    )?;

    // ── 3. Batched MTP lm_head + top-K extraction ─────────────────────────
    let t_outs_view = state.mtp_scratch.t_mtp_outs.sub_offset(0, n_mtp_forwards * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, n_mtp_forwards * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, n_mtp_forwards * dim);
    let lm_logits_view = state.mtp_lm_logits.sub_offset(0, n_mtp_forwards * vocab);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu, head, &trunk_weights.output,
        &t_outs_view, &lm_tmp_view, &lm_rot_view, &lm_logits_view,
        n_mtp_forwards,
    )?;

    // Download MTP logits to host for top-K extraction. Cost: (B-1) × vocab × 4B.
    // For 27B vocab=151936, B=16 → 15 × 152K × 4 = 9.1 MB D2H. Acceptable
    // (single PCIe round-trip, no per-row launches).
    let mtp_logits_host = gpu.download_f32(&lm_logits_view)?;
    debug_assert_eq!(mtp_logits_host.len(), n_mtp_forwards * vocab);

    // Per-slot top-K for MTP children. For each MTP forward i (i=0..B-1), the
    // top-K tokens become the K children of dflash slot i+1 in the tree.
    let mut mtp_children: Vec<Vec<u32>> = Vec::with_capacity(n_mtp_forwards);
    for i in 0..n_mtp_forwards {
        let row = &mtp_logits_host[i * vocab..(i + 1) * vocab];
        mtp_children.push(topk_indices(row, mtp_k));
    }

    // ── 4. Build linearized tree ──────────────────────────────────────────
    //
    // Total nodes = 1 (root) + (B-1) dflash + (B-1)*K MTP children.
    // Layout in linearization:
    //   slot 0:               root (seed)
    //   slots 1..B:           dflash_c1..dflash_c{B-1}
    //   slots B..B+(B-1)*K:   mtp_c1_0..mtp_c1_{K-1}, mtp_c2_0..mtp_c2_{K-1}, ...
    let n_dflash = b - 1;
    let n_mtp_total = n_dflash * mtp_k;
    let n_total = 1 + n_dflash + n_mtp_total;

    let mut tree_tokens: Vec<u32> = Vec::with_capacity(n_total);
    let mut tree_positions: Vec<i32> = Vec::with_capacity(n_total);
    let mut parent_indices: Vec<i32> = Vec::with_capacity(n_total);

    // Slot 0: root (seed).
    tree_tokens.push(seed_token);
    tree_positions.push((position + co as usize) as i32);
    parent_indices.push(-1);

    // Slots 1..B: dflash chain (causal, each parent = previous dflash or root).
    for i in 0..n_dflash {
        tree_tokens.push(drafted[i + 1]);
        tree_positions.push((position + i + 1 + co as usize) as i32);
        // dflash_c1 parent = root (slot 0); dflash_c2 parent = slot 1; etc.
        parent_indices.push(i as i32);
    }

    // Slots B..B+(B-1)*K: MTP children grouped per dflash slot.
    for i in 0..n_dflash {
        let parent_dflash_slot = (i + 1) as i32; // dflash_c{i+1} is at linearization slot i+1
        let mtp_pos = (position + i + 2 + co as usize) as i32;
        for k in 0..mtp_k {
            tree_tokens.push(mtp_children[i][k]);
            tree_positions.push(mtp_pos);
            parent_indices.push(parent_dflash_slot);
        }
    }
    debug_assert_eq!(tree_tokens.len(), n_total);

    // Build tree-attention mask: -inf except where j is an ancestor of i.
    // Row i, col j: 0.0 if j is on the ancestor chain of i (incl. i and root),
    // else -inf.
    let mut mask_host: Vec<f32> = vec![f32::NEG_INFINITY; n_total * n_total];
    // For each node, walk its ancestor chain using parent_indices to mark
    // visibility 0.0 along that chain.
    for i in 0..n_total {
        // Self always visible.
        mask_host[i * n_total + i] = 0.0;
        let mut cur = parent_indices[i];
        while cur >= 0 {
            mask_host[i * n_total + cur as usize] = 0.0;
            cur = parent_indices[cur as usize];
        }
    }

    // Upload mask + parent_indices into the DdtreeScratch buffers.
    assert!(n_total <= ddtree_scratch.max_n,
        "tree size {} exceeds ddtree_scratch.max_n {}", n_total, ddtree_scratch.max_n);
    {
        let mask_bytes = unsafe {
            std::slice::from_raw_parts(mask_host.as_ptr() as *const u8, mask_host.len() * 4)
        };
        gpu.hip.memcpy_htod(&ddtree_scratch.attn_bias.buf, mask_bytes)?;
    }
    let use_tree_la = std::env::var("HIPFIRE_DDTREE_TREE_LA").ok().as_deref() != Some("0");
    if use_tree_la {
        let parent_bytes = unsafe {
            std::slice::from_raw_parts(parent_indices.as_ptr() as *const u8, parent_indices.len() * 4)
        };
        gpu.hip.memcpy_htod(&ddtree_scratch.parent_indices.buf, parent_bytes)?;
    }

    // ── 5. Tree verify ────────────────────────────────────────────────────
    target_snap.save_from(&target.dn_state, gpu)?;

    // MoE-aware tape gate (mirrors spec_step_dflash).
    let mut gdn_tape_opt = gdn_tape;
    let target_has_moe = target.weights.layers.iter().any(|lw| matches!(
        lw,
        qwen35::LayerWeights::DeltaNetMoe(_) | qwen35::LayerWeights::FullAttnMoe(_),
    ));
    if target_has_moe {
        gdn_tape_opt = None;
    }

    let attn_bias_view = ddtree_scratch.attn_bias.sub_offset(0, n_total * n_total);
    let parent_view = ddtree_scratch.parent_indices.sub_offset(0, n_total * 4);
    let ctx = qwen35::TreeVerifyCtx {
        positions: &tree_positions,
        attn_bias: &attn_bias_view,
        parent_indices: if use_tree_la { Some(&parent_view) } else { None },
        pre_rope_k_capture: None,
    };

    let verify_out: DflashVerifyOutput = speculative::verify_dflash_block_tree(
        gpu, target, &tree_tokens, position, hidden_rb, gdn_tape_opt.as_deref_mut(),
        false, // greedy / temp=0
        ctx,
        verify_scratch,
    )?;

    let posterior = &verify_out.argmax_per_pos;
    debug_assert_eq!(posterior.len(), n_total);

    // ── 6. Greedy walk: longest accepted path through the tree ────────────
    //
    // Start at root (slot 0). For each step, look at posterior[current_slot]
    // (target's argmax at that position) and find a CHILD of current_slot
    // whose token == posterior[current_slot]. If found, accept it and recurse.
    // Otherwise stop.
    //
    // Children of slot s: for the dflash chain, slot s+1 (if s+1 < B). For
    // each dflash slot s in 1..B, also slots B + (s-1)*K .. B + s*K (its K
    // MTP children).
    let mut accepted_slots: Vec<usize> = Vec::new();
    let mut current = 0usize; // root slot
    loop {
        let target_argmax = posterior[current];
        // Find children of `current`.
        let mut chosen: Option<usize> = None;
        // dflash child: only the root has a dflash child as an "implicit"
        // direct successor (slot 1), and each dflash slot s has slot s+1 as
        // its dflash continuation. MTP children attach to dflash slots
        // (current >= 1 and current < B).
        if current == 0 && n_dflash >= 1 {
            // root → dflash_c1 (slot 1)
            if tree_tokens[1] == target_argmax {
                chosen = Some(1);
            }
        } else if current >= 1 && current < b {
            // dflash slot → next dflash slot (if any)
            if current + 1 < b && tree_tokens[current + 1] == target_argmax {
                chosen = Some(current + 1);
            } else {
                // Try MTP children of this dflash slot.
                let mtp_base = b + (current - 1) * mtp_k;
                for k in 0..mtp_k {
                    let mtp_slot = mtp_base + k;
                    if mtp_slot < n_total && tree_tokens[mtp_slot] == target_argmax {
                        chosen = Some(mtp_slot);
                        break;
                    }
                }
            }
        }
        // MTP slots (current >= B) have NO children in v1 — they're leaves.
        match chosen {
            Some(next) => {
                accepted_slots.push(next);
                current = next;
            }
            None => break,
        }
    }
    let bonus_token = posterior[current];

    // Accept_dflash / accept_mtp split.
    let accept_dflash = accepted_slots.iter().filter(|&&s| s >= 1 && s < b).count();
    let accept_mtp = accepted_slots.iter().filter(|&&s| s >= b).count();
    debug_assert_eq!(accept_dflash + accept_mtp, accepted_slots.len());

    // ── 7. Build committed = [seed, accepted..., bonus] ──────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accepted_slots.len() + 2);
    committed.push(seed_token);
    for &s in &accepted_slots {
        committed.push(tree_tokens[s]);
    }
    committed.push(bonus_token);

    // ── 8. Append accepted-prefix target hidden rows to draft_scratch ────
    //
    // verify wrote n_total rows of target hidden into hidden_rb in linearization
    // order. The committed-path slots are [0, accepted_slots[0], accepted_slots[1], ...].
    // For dflash chain accepts these are contiguous: 0, 1, 2, ..., accept_dflash.
    // The drafter will only consume up to position+accept_dflash+1 next cycle,
    // so we copy that many rows as a contiguous prefix.
    //
    // MTP-accepted rows are valid target hiddens too, but they correspond to
    // tree positions BEYOND the dflash chain — we just keep them as additional
    // context. Same scatter pattern as spec_step_dflash_mtp's accept_len+1.
    let rows_to_keep = accept_dflash + 1; // committed dflash prefix only (drafter doesn't see MTP rows yet)
    speculative::scatter_hidden_block_to_interleaved(
        gpu, hidden_rb, &draft_scratch.target_hidden,
        position, n_total, rows_to_keep,
    )?;
    draft_scratch.uploaded_target_hidden_rows = position + rows_to_keep;
    for p in 0..rows_to_keep {
        draft_scratch
            .target_hidden_abs_positions
            .push(position as i32 + p as i32 + co);
    }

    // ── 9. Rollback trunk DN state + replay accepted committed tokens ────
    //
    // For dflash-only accepts, replay the dflash chain prefix. For MTP
    // accepts, the MTP token represents a position that's `accept_dflash + 1`
    // beyond seed. The replay tokens mirror this: just `committed[..accept_dflash + accept_mtp + 1]`.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    let n_replay = accept_dflash + accept_mtp + 1; // committed up to (but not including) bonus
    if let Some(tape) = gdn_tape_opt.as_deref() {
        // Tape was captured in linearization order; replay only works for
        // contiguous dflash prefix (slots 0..=accept_dflash). MTP accepted
        // slots aren't on the dflash linear chain so we re-prefill if any
        // MTP was accepted.
        if accept_mtp == 0 {
            tape.replay_gdn(
                gpu, &target.weights, &target.config, &mut target.dn_state,
                accept_dflash + 1,
            )?;
        } else {
            // Replay via prefill_batch (one extra forward but correct).
            let replay_tokens = &committed[..n_replay];
            qwen35::forward_prefill_batch(
                gpu, &target.weights, &target.config, replay_tokens, position,
                &mut target.kv_cache, &mut target.dn_state, &target.scratch,
                None, None, None, None,
            )?;
        }
    } else {
        let replay_tokens = &committed[..n_replay];
        qwen35::forward_prefill_batch(
            gpu, &target.weights, &target.config, replay_tokens, position,
            &mut target.kv_cache, &mut target.dn_state, &target.scratch,
            None, None, None, None,
        )?;
    }

    Ok(MtpComposeTreeResult {
        accept_dflash,
        accept_mtp,
        bonus_token,
        committed,
        tree_nodes: n_total,
    })
}

/// Top-K argmax indices from a logits row, in descending log-prob order.
/// Returns up to `k` indices; if `k > vocab` the result is truncated.
fn topk_indices(logits: &[f32], k: usize) -> Vec<u32> {
    let vocab = logits.len();
    let k = k.min(vocab);
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![argmax_u32(logits)];
    }
    // Min-heap of size k.
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;
    #[derive(Copy, Clone, PartialEq)]
    struct Item(f32, u32);
    impl Eq for Item {}
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
                .then(self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut heap: BinaryHeap<std::cmp::Reverse<Item>> = BinaryHeap::with_capacity(k + 1);
    for (i, &v) in logits.iter().enumerate() {
        heap.push(std::cmp::Reverse(Item(v, i as u32)));
        if heap.len() > k {
            heap.pop();
        }
    }
    let mut items: Vec<Item> = heap.into_iter().map(|r| r.0).collect();
    items.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    items.into_iter().map(|it| it.1).collect()
}
