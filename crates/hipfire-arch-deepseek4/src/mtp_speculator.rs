// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 MTP `MtpDrafter` impl — the arch-specific half of the unified
//! MTP spec-decode core ([`hipfire_runtime::spec::MtpDrafter`] +
//! [`MtpSpeculator`]).
//!
//! Owns a lazily-allocated [`PrefillBatchScratch`] (the same scratch that
//! the non-spec prefill path uses; sized to `HIPFIRE_DEEPSEEK4_PP_BATCH`,
//! default 1024) and drives the fused `speculative_decode_step_with_pbs` per
//! acceptance window. Downcasts the generic `&mut dyn SpecTarget` to the
//! concrete [`Deepseek4Bundle`] — exactly as `DflashSpeculator` does for
//! qwen35. The arch-INvariant adaptation (prefill→`PrefillOutcome`,
//! window→`SpecStep`) lives in `MtpSpeculator<A>`; here we only implement the
//! four fused operations.
//!
//! `last_hidden` is NOT a drafter field — it lives on
//! `bundle.state.mtp_last_hidden` and is updated by the step function. We
//! read it via a raw pointer to dodge the borrow conflict (state is borrowed
//! `&mut` for the call while we read `mtp_last_hidden` as `&`). See the
//! `mtp_step` implementation for the safety argument.

use crate::forward::PrefillBatchScratch;
use crate::grammar::{Matcher, ToolSchema};
use crate::spec_decode::{
    logits_argmax, speculative_decode_step_with_pbs, speculative_decode_step_with_pbs_grammar,
};
use crate::spec_impl::Deepseek4Bundle;
use hipfire_runtime::spec::{
    MtpDrafter, MtpSpeculator, MtpWindow, SpecGrammar, SpecTarget, Speculator,
};
use rdna_compute::Gpu;
use std::sync::Arc;

/// Concrete in-step grammar handle for deepseek4 MTP spec-decode. Owns the
/// tool-call [`Matcher`], a shared decoded-vocab table (Arc-cloned from the
/// daemon's cache so it never borrows the model), and a reusable per-step token
/// mask. The daemon's `Deepseek4Emit::grammar()` hands this in erased as
/// `&mut dyn SpecGrammar`; [`Deepseek4MtpDrafter::mtp_step`] downcasts it back.
/// The matcher advances INSIDE `speculative_decode_step_with_pbs_grammar` ONLY
/// (single-advance invariant — the emitter must not re-advance it).
pub struct Deepseek4SpecGrammar {
    pub matcher: Matcher,
    pub decoded_vocab: Arc<Vec<String>>,
    pub grammar_mask: Vec<bool>,
}

impl Deepseek4SpecGrammar {
    /// Build from tool schemas + the daemon's cached decoded-vocab. `grammar_mask`
    /// is sized to the vocab; `Matcher::token_mask` refills it each fused step.
    pub fn new(tools: Vec<ToolSchema>, decoded_vocab: Arc<Vec<String>>) -> Self {
        let grammar_mask = vec![true; decoded_vocab.len()];
        Self {
            matcher: Matcher::new(tools),
            decoded_vocab,
            grammar_mask,
        }
    }
}

impl SpecGrammar for Deepseek4SpecGrammar {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// DeepSeek V4 MTP drafter. `pbs` is allocated on the first `mtp_prefill`
/// (it needs `&Gpu` + the concrete config, neither available at construction).
/// Greedy-only: deepseek4 MTP verification uses argmax.
pub struct Deepseek4MtpDrafter {
    pbs: Option<PrefillBatchScratch>,
    max_n: usize,
    ctx_capacity: usize,
}

impl Deepseek4MtpDrafter {
    pub fn new(max_n: usize, ctx_capacity: usize) -> Self {
        Self {
            pbs: None,
            max_n: max_n.clamp(1, 8),
            ctx_capacity,
        }
    }

    /// Downcast the generic target to a deepseek4 `Deepseek4Bundle`.
    fn bundle(target: &mut dyn SpecTarget) -> Result<&mut Deepseek4Bundle, String> {
        target
            .as_any_mut()
            .downcast_mut::<Deepseek4Bundle>()
            .ok_or_else(|| "Deepseek4MtpDrafter: target is not a Deepseek4Bundle".to_string())
    }
}

impl Deepseek4MtpDrafter {
    fn mtp_prefill_native(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<Option<u32>, String> {
        let bundle = Self::bundle(target)?;
        self.mtp_prefill_with_abort(gpu, bundle, fill_tokens, start_pos, cache_hit, abort)
    }
}

impl Deepseek4MtpDrafter {
    fn mtp_prefill_with_abort(
        &mut self,
        gpu: &mut Gpu,
        bundle: &mut Deepseek4Bundle,
        fill_tokens: &[u32],
        start_pos: usize,
        _cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<Option<u32>, String> {
        if abort() {
            return Ok(None);
        }
        let Deepseek4Bundle {
            config,
            weights,
            state,
            ..
        } = bundle;
        if abort() {
            return Ok(None);
        }

        // Lazily build the PBS. Sized identically to the loader's deepseek4_pbs
        // (carriers.rs:645-649): HIPFIRE_DEEPSEEK4_PP_BATCH, default 1024.
        if self.pbs.is_none() {
            let pbs_max_batch: usize = std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1024);
            self.pbs = Some(
                PrefillBatchScratch::new(gpu, config, pbs_max_batch)
                    .map_err(|e| format!("Deepseek4MtpDrafter: alloc PrefillBatchScratch: {e}"))?,
            );
        }
        let pbs = self.pbs.as_ref().expect("just built");
        if abort() {
            return Ok(None);
        }

        let logits = crate::forward::prefill_with_mtp_fill_abortable(
            config,
            weights,
            state,
            gpu,
            pbs,
            fill_tokens,
            start_pos as u32,
            abort,
        )
        .map_err(|e| format!("mtp prefill: {e}"))?;
        let Some(logits) = logits else {
            return Ok(None);
        };
        if abort() {
            return Ok(None);
        }

        let first_token = logits_argmax(&logits) as u32;
        if abort() {
            Ok(None)
        } else {
            Ok(Some(first_token))
        }
    }
}

impl MtpDrafter for Deepseek4MtpDrafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
    ) -> Result<u32, String> {
        fn never() -> bool {
            false
        }
        self.mtp_prefill_native(gpu, target, fill_tokens, start_pos, cache_hit, &never)
            .map(|token| token.expect("non-abortable deepseek4 MTP prefill aborted"))
    }

    fn mtp_prefill_abortable(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: Option<&dyn Fn() -> bool>,
    ) -> Result<Option<u32>, String> {
        fn never() -> bool {
            false
        }
        let abort = abort.unwrap_or(&never);
        self.mtp_prefill_native(gpu, target, fill_tokens, start_pos, cache_hit, abort)
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        _eos: u32,
        grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        let pbs = self
            .pbs
            .as_ref()
            .ok_or("Deepseek4MtpDrafter: mtp_step called before mtp_prefill")?;

        // In-step grammar: downcast the erased handle to the concrete ds4 grammar
        // (matcher + decoded-vocab + reusable mask). `None` ⇒ the plain fused step.
        // The matcher advances inside the grammar step ONLY; the daemon's
        // Deepseek4Emit::observe must not re-advance it (single-advance invariant).
        let grammar = match grammar {
            Some(g) => Some(
                g.as_any_mut()
                    .downcast_mut::<Deepseek4SpecGrammar>()
                    .ok_or("Deepseek4MtpDrafter: grammar handle is not a Deepseek4SpecGrammar")?,
            ),
            None => None,
        };

        let bundle = Self::bundle(target)?;
        let Deepseek4Bundle {
            config,
            weights,
            state,
            ..
        } = bundle;

        // SAFETY: `last_hidden_ref` is a raw pointer to `state.mtp_last_hidden`'s
        // allocation, which lives in stable VRAM for the duration of the call.
        // We need to read `state.mtp_last_hidden` (as `&`) while `state` is
        // simultaneously borrowed `&mut` by `speculative_decode_step_with_pbs`.
        // The step function only WRITES to `state.mtp_last_hidden` (refreshing
        // it from the verify pass), never frees or reallocates the backing
        // buffer — so the pointer remains valid across the call. This mirrors
        // the raw-pointer pattern in daemon.rs generate_deepseek4 (line 9683/9708).
        let last_hidden_ref: Option<*const rdna_compute::GpuTensor> =
            state.mtp_last_hidden.as_ref().map(|t| t as *const _);

        let lh: Option<&rdna_compute::GpuTensor> =
            unsafe { last_hidden_ref.and_then(|p| (p as *const rdna_compute::GpuTensor).as_ref()) };

        let r = match grammar {
            Some(g) => speculative_decode_step_with_pbs_grammar(
                config,
                weights,
                state,
                gpu,
                pbs,
                seed,
                position as u32,
                lh,
                k,
                &mut g.matcher,
                &g.decoded_vocab[..],
                &mut g.grammar_mask,
            ),
            None => speculative_decode_step_with_pbs(
                config,
                weights,
                state,
                gpu,
                pbs,
                seed,
                position as u32,
                lh,
                k,
            ),
        }
        .map_err(|e| format!("mtp step: {e}"))?;

        Ok(MtpWindow {
            committed: r.accepted_tokens,
            accepted: r.n_accepted,
            drafts_generated: r.n_proposed,
        })
    }

    fn mtp_reset(&mut self, _gpu: &mut Gpu) {
        // deepseek4's drafter has no drafter-local GPU state besides `pbs`
        // (scratch, not conversation state). The target bundle's request reset
        // is owned by LoadedModel::reset_context.
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(pbs) = self.pbs {
            pbs.free_gpu(gpu);
        }
        // The Deepseek4Bundle (target) is NOT owned by the drafter — do not free it.
    }

    fn k(&self) -> usize {
        self.max_n
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        true
    }
}

/// Build the deepseek4 MTP speculator (the boxed `dyn Speculator` the loader's
/// `build_speculator` returns). The `PrefillBatchScratch` is allocated lazily
/// on the first `mtp_prefill`.
pub fn build_deepseek4_mtp_speculator(max_n: usize, ctx_capacity: usize) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(Deepseek4MtpDrafter::new(
        max_n,
        ctx_capacity,
    )))
}

// ── Send-bound assertions ──────────────────────────────────────────────
#[cfg(test)]
mod send_assertions {
    use hipfire_runtime::spec::MtpSpeculator;

    fn _assert_send<T: Send>() {}

    #[test]
    fn mtp_speculator_deepseek4_is_send() {
        _assert_send::<MtpSpeculator<super::Deepseek4MtpDrafter>>();
    }
}
