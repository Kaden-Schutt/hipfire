// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 speculative decoding via the built-in MTP head (DeepSeek V3 §4
//! Multi-Token Prediction).
//!
//! Pipeline:
//! 1. Generate K candidate tokens by iterating `mtp_forward` K times,
//!    seeding each step with the previous step's predicted token and
//!    the previous step's MTP-layer hidden state.
//! 2. Run the main DeepSeek V4 forward as `forward_prefill_batch_chunk` at B=K
//!    on `[committed_token, draft_1, draft_2, ..., draft_{K-1}]`.
//! 3. Compare main's top-1 at each verify position against the next
//!    draft's predicted token; accept the longest matching prefix.
//! 4. Return `accepted_tokens` + the main model's preferred token at
//!    the divergence position (to keep generation moving forward even
//!    on a rejected suffix).
//!
//! When all K drafts are accepted, the next call starts from
//! `accepted_tokens[K-1]` and the previously-cached hidden state.
//!
//! ## Status
//!
//! Skeleton only. The MTP forward (`forward::mtp_forward`) is currently
//! a stub that returns an error until the standard layer block runs
//! against `weights.mtp_layer` (tracked as M3 in
//! `docs/plans/deepseek4-mtp-requant-2026-05-20.md`). This module compiles
//! and exposes the public API but errors out at the first MTP step
//! until M3 lands.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::{self};
use crate::grammar;
use rdna_compute::Gpu;

/// One acceptance window of speculative decoding.
#[derive(Debug, Clone)]
pub struct SpecStepResult {
    /// Tokens accepted this window (in emission order). At minimum
    /// always contains the verifier's preferred token at the
    /// divergence position; on full acceptance contains all K drafts.
    pub accepted_tokens: Vec<u32>,
    /// How many of the K drafts were accepted (longest matching
    /// prefix between drafts and main-model top-1 logits).
    pub n_accepted: usize,
    /// How many draft tokens were proposed (= K).
    pub n_proposed: usize,
}

/// Caller-owned grammar state for tool-call constrained speculative decode.
///
/// The daemon owns the tokenizer/decoded-vocab cache and DSML matcher. Passing
/// them here lets the MTP draft path and the verifier path apply the same
/// structural mask used by plain DeepSeek4 decoding, without making this module
/// depend on the daemon's request machinery.
pub struct SpecGrammar<'a> {
    pub matcher: &'a mut grammar::Matcher,
    pub decoded_vocab: &'a [String],
    pub mask: &'a mut Vec<bool>,
}

/// Run one speculative-decode acceptance window.
///
/// Inputs:
/// - `cfg`/`weights`/`state`/`gpu`: standard DeepSeek V4 runtime
/// - `last_token`: the most-recently-committed token (position N)
/// - `last_hidden`: optional cached hidden state at position N
///   (populated from the prior main forward); if `None`, the function
///   will run a 1-token main forward to materialize it
/// - `k`: number of draft tokens to propose
///
/// Returns the acceptance result.
///
/// Stub status: returns an error from the first `mtp_forward` call
/// until M3 lands.
/// Same as [`speculative_decode_step`] but takes a caller-owned PBS scratch
/// instead of allocating one per call. Allocating PBS internally (~30 small
/// GpuTensor allocations) costs measurable milliseconds at small K — caching
/// it once at session setup and passing it in eliminates that.
#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step_with_pbs(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &forward::PrefillBatchScratch,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&rdna_compute::GpuTensor>,
    k: usize,
) -> Result<SpecStepResult, String> {
    speculative_decode_impl(
        cfg,
        weights,
        state,
        gpu,
        Some(pbs),
        last_token,
        last_position,
        last_hidden,
        k,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step_with_pbs_grammar(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &forward::PrefillBatchScratch,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&rdna_compute::GpuTensor>,
    k: usize,
    matcher: &mut grammar::Matcher,
    decoded_vocab: &[String],
    grammar_mask: &mut Vec<bool>,
) -> Result<SpecStepResult, String> {
    speculative_decode_impl(
        cfg,
        weights,
        state,
        gpu,
        Some(pbs),
        last_token,
        last_position,
        last_hidden,
        k,
        Some(SpecGrammar {
            matcher,
            decoded_vocab,
            mask: grammar_mask,
        }),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&rdna_compute::GpuTensor>,
    k: usize,
) -> Result<SpecStepResult, String> {
    speculative_decode_impl(
        cfg,
        weights,
        state,
        gpu,
        None,
        last_token,
        last_position,
        last_hidden,
        k,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn speculative_decode_impl(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    cached_pbs: Option<&forward::PrefillBatchScratch>,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&rdna_compute::GpuTensor>,
    k: usize,
    mut grammar: Option<SpecGrammar<'_>>,
) -> Result<SpecStepResult, String> {
    if k == 0 {
        return Err("speculative_decode_step: k must be > 0".to_string());
    }
    if cfg.num_nextn_predict_layers == 0 || weights.mtp_layer.is_none() {
        return Err("speculative_decode_step: MTP layer not loaded — \
            quantize with deepseek4-q8-mtp + addon, or set HIPFIRE_DEEPSEEK4_LOAD_MTP=1"
            .to_string());
    }

    // ── 1. Pick the initial hidden state h_n ───────────────────────────
    // V3 §4: the first MTP step takes the post-layer-block hidden at
    // position N (= `last_position`). Caller can supply it via
    // `last_hidden` (faster — from a prior main forward); else we fall
    // back to `state.mtp_last_hidden`, populated by every `decode_step`
    // / `mtp_forward` call.
    //
    // Stored as a raw pointer so subsequent iterations can borrow
    // `state` mutably without re-aliasing this initial hidden. SAFETY:
    // both candidate sources live in stable VRAM allocations for the
    // duration of this function; mtp_forward writes ONLY to scratch
    // buffers + mtp_last_hidden (which is what step 1+ reads), never
    // to the `last_hidden` caller-passed tensor.
    let h_n_ptr: *const rdna_compute::GpuTensor = match last_hidden {
        Some(h) => h as *const _,
        None => {
            let h = state.mtp_last_hidden.as_ref().ok_or_else(|| {
                "speculative_decode_step: no hidden state available — \
                 run decode_step(last_token, last_position) first or \
                 pass last_hidden explicitly"
                    .to_string()
            })?;
            h as *const _
        }
    };

    // ── 2. K draft iterations of mtp_forward ──────────────────────────
    // Each call writes the post-layer-block stream-0 hidden back to
    // `state.mtp_last_hidden`, so step k+1 chains from step k's output.
    //
    // **state.n_tokens bookkeeping** (subtle): `attn_stub` (called by
    // mtp_forward via the standard layer block) reads `state.n_tokens`
    // to pick the SWA ring slot. After the caller's most recent
    // decode_step at position N, `state.n_tokens == N+1`. For MTP step
    // s at position N+1+s, we need `state.n_tokens == N+1+s` so the
    // SWA write lands in the right MTP-layer ring slot. We increment
    // between iterations and restore based on `n_accept` at the end.
    //
    // (forward_prefill_batch_chunk uses an explicit start_pos parameter
    // and doesn't touch state.n_tokens — verified.)
    let initial_n_tokens = state.n_tokens;
    let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
    let mut draft_matcher = grammar.as_ref().map(|g| (*g.matcher).clone());
    for step in 0..k {
        let next_token = if step == 0 {
            last_token
        } else {
            draft_tokens[step - 1]
        };
        // V3 paper §4: h_i^k = M_k @ Concat(norm(h_i^{k-1}), norm(e_{i+k})).
        // The MTP transformer block operates at position i (not i+1).
        // For step k=0 predicting T_{N+1}: i = N-1 = last_position.
        // For step k=s predicting T_{N+1+s}: i = N-1+s = last_position+s.
        // So position passed to mtp_forward (which sets RoPE phase + SWA
        // slot) is `last_position + step`, NOT `last_position + 1 + step`.
        // The off-by-one earlier was causing MTP attn_stub to write the
        // wrong SWA slot and RoPE to encode the wrong phase — accepted
        // rate measured at ~50% K=2 with the bug; fix is being tested.
        let position = last_position + step as u32;
        state.n_tokens = position as u64;

        // For step 0 we use h_n_ptr; for step k>0 we point at the
        // freshly-written state.mtp_last_hidden. Both go through a raw
        // pointer to decouple from state's borrow. SAFETY: as above,
        // these GpuTensors live in stable allocations and are only
        // READ by the GEMV chain inside mtp_forward (which writes to
        // distinct scratch + state.mtp_last_hidden each iteration).
        let hidden_ptr: *const rdna_compute::GpuTensor = if step == 0 {
            h_n_ptr
        } else {
            state.mtp_last_hidden.as_ref().ok_or_else(|| {
                format!(
                    "spec_decode: mtp_last_hidden missing after step {}",
                    step - 1
                )
            })? as *const _
        };
        let hidden: &rdna_compute::GpuTensor = unsafe { &*hidden_ptr };
        let mut logits =
            forward::mtp_forward(cfg, weights, state, gpu, hidden, next_token, position)?;
        if let (Some(g), Some(matcher)) = (grammar.as_mut(), draft_matcher.as_ref()) {
            apply_grammar_mask(matcher, g.decoded_vocab, g.mask, &mut logits);
        }
        let argmax = logits_argmax(&logits) as u32;
        draft_tokens.push(argmax);
        if let (Some(g), Some(matcher)) = (grammar.as_ref(), draft_matcher.as_mut()) {
            advance_matcher_token(matcher, g.decoded_vocab, argmax);
        }
    }

    // ── 3. Single B=K main verify pass ────────────────────────────────
    // Tokens to feed the verifier: the last committed token plus the
    // first K-1 drafts. The verifier outputs logits at K positions,
    // each predicting "what comes after my input token" — these are
    // the predictions we compare to the drafts.
    //
    //   verify_tokens[0] = last_token   → predicts pos N+1's token (= draft[0]'s target)
    //   verify_tokens[1] = draft[0]     → predicts pos N+2's token (= draft[1]'s target)
    //   ...
    //   verify_tokens[K-1] = draft[K-2] → predicts pos N+K  's token
    let verify_tokens: Vec<u32> = std::iter::once(last_token)
        .chain(draft_tokens.iter().take(k - 1).copied())
        .collect();
    debug_assert_eq!(verify_tokens.len(), k);

    // Use the caller-provided PBS if available; otherwise allocate one.
    // The owned variant exists so single-shot callers / tests still work
    // without threading a PBS through; the cached variant is the perf-
    // critical path used by tight spec-decode loops.
    let owned_pbs: Option<forward::PrefillBatchScratch> = match cached_pbs {
        Some(_) => None,
        None => Some(forward::PrefillBatchScratch::new(gpu, cfg, k)?),
    };
    let pbs: &forward::PrefillBatchScratch =
        cached_pbs.unwrap_or_else(|| owned_pbs.as_ref().unwrap());
    if pbs.max_batch < k {
        return Err(format!(
            "spec_decode: cached PBS max_batch ({}) < k ({})",
            pbs.max_batch, k
        ));
    }
    forward::forward_prefill_batch_chunk(
        cfg,
        weights,
        state,
        gpu,
        pbs,
        &verify_tokens,
        last_position + 1,
    )?;

    // ── 4. Per-position top-1 from the verifier ───────────────────────
    let all_logits = forward::final_norm_and_head_all_batched(cfg, weights, state, pbs, gpu, k)?;
    let mut verify_matcher = grammar.as_ref().map(|g| (*g.matcher).clone());

    // ── 5. Longest matching prefix → acceptance ────────────────────────
    //
    // In tool-call mode the verifier's preferred token is chosen after the
    // same DSML grammar mask as the non-spec decode path. The verifier matcher
    // advances only along the actually accepted prefix; at divergence, the
    // appended verifier token is legal for the grammar state reached by that
    // prefix.
    let mut accepted_tokens: Vec<u32> = Vec::with_capacity(k);
    let mut n_accept = 0usize;
    for (idx, &draft) in draft_tokens.iter().enumerate() {
        let main = match (grammar.as_mut(), verify_matcher.as_ref()) {
            (Some(g), Some(matcher)) => {
                let mut logits = all_logits[idx].clone();
                apply_grammar_mask(matcher, g.decoded_vocab, g.mask, &mut logits);
                logits_argmax(&logits) as u32
            }
            _ => logits_argmax(&all_logits[idx]) as u32,
        };
        if draft == main {
            accepted_tokens.push(draft);
            n_accept += 1;
            if let (Some(g), Some(matcher)) = (grammar.as_ref(), verify_matcher.as_mut()) {
                advance_matcher_token(matcher, g.decoded_vocab, draft);
            }
        } else {
            accepted_tokens.push(main);
            if let (Some(g), Some(matcher)) = (grammar.as_ref(), verify_matcher.as_mut()) {
                advance_matcher_token(matcher, g.decoded_vocab, main);
            }
            break;
        }
    }

    if let Some(g) = grammar.as_mut() {
        for &tok in &accepted_tokens {
            advance_matcher_token(g.matcher, g.decoded_vocab, tok);
        }
    }

    // ── 6. Refresh state.mtp_last_hidden from the verify pass ──────────
    // Capture the FULL [hc_mult, hidden] residual stream of
    // pbs.streams_batch[accepted_tokens.len() - 1, :, :]. Matches the
    // antirez/ds4 reference MTP HC plumbing (see project memory entry
    // `project_deepseek4_mtp_hc_plumbing_gap`). Stream-0-only capture was what
    // discarded 75% of HC signal and pinned K=2 accept at ~50%.
    {
        let last_idx = accepted_tokens.len() - 1;
        let stream_len = cfg.hc_mult * cfg.hidden_size;
        let off = last_idx * stream_len;
        let last_full = pbs.streams_batch.sub_offset(off, stream_len);
        let need_realloc = state
            .mtp_last_hidden
            .as_ref()
            .map(|t| t.numel() != stream_len)
            .unwrap_or(true);
        if need_realloc {
            state.mtp_last_hidden = Some(
                gpu.alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], rdna_compute::DType::F32)
                    .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?,
            );
        }
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &last_full.buf, stream_len * 4)
            .map_err(|e| format!("capture verify-pass full HC streams: {e:?}"))?;
    }

    // ── 7. Restore state.n_tokens to the post-accept position ─────────
    // Caller's next forward expects `state.n_tokens` == (next position
    // to be processed). We emitted `accepted_tokens.len()` tokens
    // starting at position last_position+1, so the next free position
    // is last_position + 1 + accepted_tokens.len().
    //
    // Why this isn't simply `initial_n_tokens + accepted_tokens.len()`:
    // initial_n_tokens (== last_position + 1) is the position of the
    // FIRST emitted token. After emitting all accepted_tokens, next
    // position is initial_n_tokens + accepted_tokens.len(). Same thing.
    //
    // Stale-cache caveat: MTP layer's SWA cache has writes at positions
    // [N+1 .. N+K] from the draft loop; the main layers' SWA caches
    // have writes at the same positions from the verify pass. Positions
    // BEYOND n_accept were computed using rejected draft tokens (input
    // mismatch with what the caller will treat as committed). Those
    // entries get naturally invalidated when the caller's next forward
    // overwrites them via ring buffer. Bug only manifests when a
    // forward READS those stale slots before overwriting — happens
    // only in narrow windows and is documented as a production-hardening
    // follow-up.
    state.n_tokens = initial_n_tokens + accepted_tokens.len() as u64;

    Ok(SpecStepResult {
        accepted_tokens,
        n_accepted: n_accept,
        n_proposed: k,
    })
}

/// Standalone helper: compute argmax of a [vocab] logits vector.
/// Used by `speculative_decode_step` to pick the verifier's preferred
/// token at the divergence position.
#[inline]
pub fn logits_argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = logits[0];
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

fn apply_grammar_mask(
    matcher: &grammar::Matcher,
    decoded_vocab: &[String],
    mask: &mut Vec<bool>,
    logits: &mut [f32],
) {
    if matcher.is_free() || decoded_vocab.is_empty() {
        return;
    }
    if mask.len() < decoded_vocab.len() {
        mask.resize(decoded_vocab.len(), true);
    }
    matcher.token_mask(decoded_vocab, mask);
    grammar::Matcher::apply_mask_to_logits(mask, logits);
}

fn advance_matcher_token(matcher: &mut grammar::Matcher, decoded_vocab: &[String], token: u32) {
    if let Some(text) = decoded_vocab.get(token as usize) {
        matcher.advance(text);
    }
}
