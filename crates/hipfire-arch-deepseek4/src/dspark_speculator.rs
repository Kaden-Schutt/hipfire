// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 **DSpark** `MtpDrafter` impl — the DSpark draft module wired
//! into the unified MTP spec-decode core ([`hipfire_runtime::spec::MtpDrafter`]
//! + [`MtpSpeculator`]), mirroring [`crate::mtp_speculator::Deepseek4MtpDrafter`].
//!
//! The ONLY difference from the MTP drafter is the DRAFT SOURCE: instead of K
//! iterations of `mtp_forward`, one [`crate::forward::dspark_forward`] call
//! produces all `block_size` draft tokens in a single block-batched pass. The
//! VERIFY + ACCEPT machinery is the shared trunk-forward + `accept_greedy_prefix`
//! used by [`crate::spec_decode`].
//!
//! ## main_hidden bookkeeping (the crux)
//!
//! `dspark_forward(main_hidden@P, prev_token=token@P, position=P)` drafts the
//! tokens at positions `P+1 ..= P+block`. So before drafting at the seed
//! position `P` we need the trunk's captured `[40,41,42]` main_hidden FOR the
//! seed token at `P`. The seed token is freshly committed (it has never been
//! forwarded through the trunk), so we materialize its main_hidden with a single
//! 1-token capture-armed trunk forward, caching its position in
//! `self.main_hidden_pos`. The window's K+1 verify forward also captures, but it
//! captures the *seed + drafts* positions — NOT the next seed (the bonus), which
//! is a brand-new token. Hence the bootstrap forward fires once per window.
//! (Warming the DSpark stage KV rings during prefill — a τ optimisation — is a
//! TODO; see `mtp_prefill`.)

use crate::forward::{self, dspark_assemble_main_hidden, dspark_forward, PrefillBatchScratch};
use crate::mtp_speculator::Deepseek4SpecGrammar;
use crate::spec_decode::logits_argmax;
use crate::spec_impl::Deepseek4Bundle;
use hipfire_runtime::spec::{
    accept_greedy_prefix, MtpDrafter, MtpSpeculator, MtpWindow, SpecGrammar, SpecTarget, Speculator,
};
use rdna_compute::Gpu;

/// DeepSeek V4 DSpark drafter. Holds its own trunk-sized `PrefillBatchScratch`
/// (the verify + bootstrap forwards run through it) allocated lazily on the
/// first `mtp_prefill`. `main_hidden_pos` tracks which absolute position the
/// seed's main_hidden currently in `state.dspark_main_hidden` belongs to, so a
/// window can skip the bootstrap forward when it's already in sync (it never is
/// today — each window's next seed is a fresh token — but the guard keeps the
/// contract explicit and makes a future fold cheap).
pub struct Deepseek4DsparkDrafter {
    pbs: Option<PrefillBatchScratch>,
    /// Absolute position of the seed token whose main_hidden lives in
    /// `state.dspark_main_hidden`. `None` ⇒ must bootstrap.
    main_hidden_pos: Option<usize>,
    block: usize,
    ctx_capacity: usize,
    /// Confidence-truncation threshold (survival sigmoid cutoff). Resolved once
    /// at build time as env > CLI param > 0.5 — see `build_deepseek4_dspark_speculator`.
    conf_threshold: f32,
}

impl Deepseek4DsparkDrafter {
    pub fn new(block: usize, ctx_capacity: usize, conf_threshold: f32) -> Self {
        Self {
            pbs: None,
            main_hidden_pos: None,
            block: block.clamp(1, 8),
            ctx_capacity,
            conf_threshold,
        }
    }

    fn bundle(target: &mut dyn SpecTarget) -> Result<&mut Deepseek4Bundle, String> {
        target
            .as_any_mut()
            .downcast_mut::<Deepseek4Bundle>()
            .ok_or_else(|| "Deepseek4DsparkDrafter: target is not a Deepseek4Bundle".to_string())
    }

    fn pbs_max_batch() -> usize {
        std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024)
    }
}

impl MtpDrafter for Deepseek4DsparkDrafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
    ) -> Result<u32, String> {
        if !cache_hit {
            target.reset_recurrent(gpu);
            self.main_hidden_pos = None;
        }

        if self.pbs.is_none() {
            let bundle = Self::bundle(target)?;
            self.pbs = Some(
                PrefillBatchScratch::new(gpu, &bundle.config, Self::pbs_max_batch())
                    .map_err(|e| format!("Deepseek4DsparkDrafter: alloc PBS: {e}"))?,
            );
        }

        let bundle = Self::bundle(target)?;
        let Deepseek4Bundle {
            config,
            weights,
            state,
            ..
        } = bundle;

        if weights.dspark.is_none() {
            return Err("Deepseek4DsparkDrafter: weights.dspark is None".into());
        }

        // Arm the [40,41,42] target-hidden capture for the prefill forward.
        state.dspark_target_layers = weights
            .dspark
            .as_ref()
            .unwrap()
            .cfg
            .target_layer_ids
            .clone();
        state.dspark_capture_active = true;

        let pbs = self.pbs.as_ref().expect("just built");
        // Strict batched-only trunk prefill with capture armed (same path the
        // validated dspark_forward_smoke uses). Returns the LAST position's
        // trunk logits; their argmax is the AR seed.
        let last_logits = forward::forward_prefill_batch_chunked(
            config,
            weights,
            state,
            gpu,
            fill_tokens,
            start_pos as u32,
            pbs,
        )
        .map_err(|e| format!("dspark prefill: {e}"))?;

        // NOTE: we deliberately do NOT warm the DSpark stage main_kv rings over
        // the prompt here. The reference forward_spec(start_pos==0) does prime
        // them, and an experimental `forward::dspark_warm_rings` reproduces that
        // (phase-correct, sharing `dspark_stage_main_kv_to_ring` with decode).
        // But measured on this MQ2-Lloyd build it is a consistent LOSS: on a
        // 96-token code prompt, priming dropped τ 3.24→2.55 and decode 13.8→10.8
        // tok/s (deterministic, interleaved A/B). The trained draft attends
        // BETTER to a sparse recent-decode window than to the full committed
        // prompt history — injecting the prompt's main_kv misaligns the draft
        // from the target under this quant. Left unwired pending a full-precision
        // model where the reference's priming actually pays. See the branch's
        // bench notes / dspark-v4-deepseek4-port memory entry.
        //
        // The seed sits one position past the last prefilled position; its
        // main_hidden is materialised on the first mtp_step's bootstrap forward,
        // so we leave main_hidden_pos = None.
        self.main_hidden_pos = None;

        Ok(logits_argmax(&last_logits) as u32)
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        // DSpark drafts the whole block at once; the in-step grammar (tool-call)
        // path is not yet wired for DSpark — downcast to surface a wrong pairing
        // loudly rather than silently dropping the mask, but otherwise ignore it
        // (the daemon's post-hoc emission-layer grammar still applies).
        if let Some(g) = grammar {
            let _ = g
                .as_any_mut()
                .downcast_mut::<Deepseek4SpecGrammar>()
                .ok_or("Deepseek4DsparkDrafter: grammar handle is not a Deepseek4SpecGrammar")?;
        }

        let bundle = Self::bundle(target)?;
        // Detach config (small, Clone) so it doesn't pin `&bundle`. The remaining
        // accesses go through disjoint field paths (`bundle.weights.*` immutable,
        // `bundle.state` mutable) which the borrow checker allows.
        let config = bundle.config.clone();

        if bundle.weights.dspark.is_none() {
            return Err("Deepseek4DsparkDrafter: weights.dspark is None".into());
        }
        let target_layers = bundle
            .weights
            .dspark
            .as_ref()
            .unwrap()
            .cfg
            .target_layer_ids
            .clone();
        let block = bundle
            .weights
            .dspark
            .as_ref()
            .unwrap()
            .cfg
            .block_size
            .min(k)
            .max(1);

        // Trunk tensors (embedding / lm_head / output norm). shallow_clone()
        // detaches them from the `weights` borrow so they can coexist with the
        // `&mut state` the forwards take.
        let token_embd = bundle
            .weights
            .token_embd
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: weights.token_embd is None")?
            .shallow_clone();
        let head = bundle
            .weights
            .head
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: weights.head is None")?
            .shallow_clone();
        let output_norm = bundle
            .weights
            .output_norm
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: weights.output_norm is None")?
            .shallow_clone();

        // Read the in-sync guard before borrowing pbs (which pins `self`); all
        // writes to `self.main_hidden_pos` happen after pbs's last use (step 5).
        let need_bootstrap = self.main_hidden_pos != Some(position);
        let pbs = self
            .pbs
            .as_ref()
            .ok_or("Deepseek4DsparkDrafter: mtp_step before mtp_prefill")?;

        // ── 1. Ensure main_hidden@position for the seed ─────────────────────
        // The seed is a fresh token; materialise its captured [40,41,42] hidden
        // with a single 1-token capture-armed trunk forward. (Guard lets a
        // future verify-fold skip this when already in sync.)
        if need_bootstrap {
            bundle.state.dspark_target_layers = target_layers.clone();
            bundle.state.dspark_capture_active = true;
            forward::forward_prefill_batch_chunk(
                &config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                pbs,
                &[seed],
                position as u32,
            )
            .map_err(|e| format!("dspark bootstrap forward: {e}"))?;
            dspark_assemble_main_hidden(&mut bundle.state, gpu, &config, 0)
                .map_err(|e| format!("dspark assemble bootstrap main_hidden: {e}"))?;
        }

        // ── 2. Draft the block with DSpark ──────────────────────────────────
        let main_hidden = bundle
            .state
            .dspark_main_hidden
            .as_ref()
            .ok_or("dspark: main_hidden missing after bootstrap")?
            .shallow_clone();
        let draft = dspark_forward(
            &config,
            bundle.weights.dspark.as_ref().unwrap(),
            &mut bundle.state,
            gpu,
            &main_hidden,
            &token_embd,
            &head,
            &output_norm,
            seed,
            position as u32,
        )
        .map_err(|e| format!("dspark draft: {e}"))?;
        let mut drafts: Vec<u32> = draft.tokens.into_iter().take(block).collect();

        // ── 2a. Confidence-threshold draft truncation (DSpark's own adaptive
        // draft-length mechanism — reference deepspec `_confident_prefix_length`).
        //
        // `dspark_forward`'s confidence head emits a per-slot confidence LOGIT
        // (pre-sigmoid). Survival of slot i is `sigmoid(confidence[i])`; the
        // reference truncates the proposal at the FIRST slot whose survival drops
        // below a threshold. Truncating BEFORE the verify forward means the heavy
        // 43-layer/256-expert trunk runs over fewer positions on uncertain (prose)
        // drafts — cheaper windows — while the full block survives where the model
        // is confident (code). This does NOT change which tokens get committed
        // when the model is confident: a slot below threshold is one the draft is
        // unsure of and likely-to-be-rejected anyway, so cutting it trades a sliver
        // of potential acceptance for a strictly cheaper verify. The committed
        // stream remains target-verified greedy ⇒ coherence is preserved.
        //
        // Threshold default 0.5: survival 0.5 ⇔ confidence logit 0.0, i.e. keep a
        // slot iff the head's confidence logit is non-negative. This is the natural
        // decision boundary of a sigmoid gate and matches the reference default.
        // Resolved once at build (env > `--dspark-conf-threshold` > 0.5).
        let conf_threshold = self.conf_threshold;
        let confident_len = {
            // First slot below threshold cuts the proposal there; always keep ≥1.
            let mut l = drafts.len();
            for (i, &c) in draft.confidence.iter().enumerate().take(drafts.len()) {
                let survival = 1.0f32 / (1.0 + (-c).exp());
                if survival < conf_threshold {
                    l = i;
                    break;
                }
            }
            l.max(1)
        };
        drafts.truncate(confident_len);
        let n_proposed = drafts.len();

        // ── 3. Verify: trunk forward [seed, draft0..draft_{n-1}] ────────────
        // Placed at their TRUE trunk positions (seed@position, drafts at
        // position+1..). Capture armed so the verify pass also refreshes the
        // captures, though the next seed (bonus) is a fresh token captured by the
        // next window's bootstrap forward.
        let verify_tokens: Vec<u32> = std::iter::once(seed)
            .chain(drafts.iter().copied())
            .collect();
        if pbs.max_batch < verify_tokens.len() {
            return Err(format!(
                "dspark verify: PBS max_batch ({}) < verify len ({})",
                pbs.max_batch,
                verify_tokens.len()
            ));
        }
        bundle.state.dspark_target_layers = target_layers.clone();
        bundle.state.dspark_capture_active = true;
        forward::forward_prefill_batch_chunk(
            &config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            pbs,
            &verify_tokens,
            position as u32,
        )
        .map_err(|e| format!("dspark verify forward: {e}"))?;

        // ── 4. Greedy accept (shared core). target_pick[i] = argmax at verify
        //    slot i = the trunk's prediction for position+i+1. Argmax runs
        //    ON GPU and downloads only the K+1 token ids — not the K+1 × 200k
        //    logits the host-argmax path used to. EOS-aware so an accepted EOS
        //    draft stops the window without a stale bonus. ─────────────────────
        let target_pick = forward::final_norm_and_argmax_all_batched(
            &config,
            &bundle.weights,
            &mut bundle.state,
            pbs,
            gpu,
            verify_tokens.len(),
        )
        .map_err(|e| format!("dspark verify head+argmax: {e}"))?;
        let acc = accept_greedy_prefix(&drafts, &target_pick, Some(eos));
        let committed = acc.committed;
        let n_accepted = acc.accepted;

        // ── 5. Advance trunk position + invalidate the next seed's main_hidden.
        // The verify forward wrote ring slots position..position+n_proposed using
        // (possibly rejected) drafts; only the first committed.len() are real.
        // The next window's bootstrap forward overwrites the next-seed slot.
        //
        // NOTE: the next seed (committed.last() = the verifier's bonus) is the one
        // token never run through the trunk, so its [40,41,42] main_hidden is NOT
        // in the capture buffer — we force a fresh 1-token bootstrap forward next
        // window. We tried folding this away (Lever 2): emit only the accepted
        // prefix and seed the next window from the last accepted token, whose
        // main_hidden IS captured (dspark_caps[a]), re-proposing the bonus as the
        // next window's first draft. It is correct and coherent but a measured
        // LOSS (short prompt 10.6→8.8, code 13.8→12.9 tok/s): DSpark's verify is a
        // full 43-layer MoE trunk pass that dominates the cheap 1-token bootstrap,
        // so dropping the free bonus (≈ −1 token/window ⇒ ~1.4× more expensive
        // verify forwards) costs more than the bootstrap saves. The free bonus is
        // worth more than the bootstrap is expensive — so we keep the 2-forward
        // shape. See the branch bench notes.
        bundle.state.n_tokens = (position + committed.len()) as u64;
        self.main_hidden_pos = None;

        Ok(MtpWindow {
            committed,
            accepted: n_accepted,
            drafts_generated: n_proposed,
        })
    }

    fn mtp_reset(&mut self, _gpu: &mut Gpu) {
        // No drafter-local conversation state beyond `pbs` (scratch). The target
        // bundle's recurrent reset is the daemon's job. Invalidate the cached
        // main_hidden position so the next prefill re-bootstraps cleanly.
        self.main_hidden_pos = None;
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(pbs) = self.pbs {
            pbs.free_gpu(gpu);
        }
    }

    fn k(&self) -> usize {
        self.block
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        true
    }
}

/// Build the deepseek4 DSpark speculator (the boxed `dyn Speculator` the loader
/// returns when a `-dspark` sidecar is present). The trunk-sized
/// `PrefillBatchScratch` is allocated lazily on the first `mtp_prefill`.
///
/// `conf_threshold` is the CLI-forwarded confidence-truncation cutoff (`None` =
/// loader default 0.5). Ladder: env `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD`
/// wins, else the CLI param, else 0.5. Clamped to `[0, 1]` here — it is a
/// survival-sigmoid cutoff, so the env/JSON paths (which bypass the CLI's TS
/// validation) can't push it out of range and silently degrade (`>1` ⇒ block
/// always trims to 1 ≈ AR; `<0` ⇒ truncation never fires).
pub fn build_deepseek4_dspark_speculator(
    block: usize,
    ctx_capacity: usize,
    conf_threshold: Option<f32>,
) -> Box<dyn Speculator> {
    let conf_threshold = std::env::var("HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .or(conf_threshold)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    Box::new(MtpSpeculator::new(Deepseek4DsparkDrafter::new(
        block,
        ctx_capacity,
        conf_threshold,
    )))
}
