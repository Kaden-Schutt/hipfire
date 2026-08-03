// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Qwen3.5/3.6 MTP `MtpDrafter` impl — the arch-specific half of the unified
//! MTP spec-decode core ([`hipfire_runtime::spec::MtpDrafter`] +
//! [`MtpSpeculator`]).
//!
//! This owns the loaded [`Qwen35MtpHead`] + a lazily-allocated [`MtpSpecState`]
//! and drives the full-vocab compressed-serial spec step
//! ([`spec_step_mtp_compressed_serial`]) per acceptance window. It downcasts the
//! generic `&mut dyn SpecTarget` to the concrete [`ModelSlot`] — exactly as
//! `DflashSpeculator` does — so the daemon never sees a qwen35 type. The
//! arch-INvariant adaptation (prefill→`PrefillOutcome`, window→`SpecStep`) lives
//! in `MtpSpeculator<A>`; here we only implement the four fused operations.

use crate::mtp_head::{MtpKvMode, Qwen35MtpHead};
use crate::mtp_spec::{
    prefill_trunk_and_mtp_cache_abortable, spec_step_mtp_compressed_serial, MtpSamplingConfig,
    MtpSpecState,
};
use crate::speculative::ModelSlot;
use hipfire_runtime::spec::{
    MtpDrafter, MtpSpeculator, MtpWindow, SpecGrammar, SpecTarget, Speculator,
};
use rdna_compute::Gpu;

/// qwen35 MTP drafter. `state` is allocated on the first `mtp_prefill` (it needs
/// `&mut Gpu` + the concrete `ModelSlot`, neither available at load-time
/// construction). Greedy-only: `p_min` is forced to 0 and sampling kept at the
/// greedy default.
pub struct Qwen35MtpDrafter {
    head: Qwen35MtpHead,
    state: Option<MtpSpecState>,
    max_n: usize,
    ctx_capacity: usize,
}

impl Qwen35MtpDrafter {
    pub fn new(head: Qwen35MtpHead, max_n: usize, ctx_capacity: usize) -> Self {
        Self {
            head,
            state: None,
            max_n: max_n.clamp(1, 8),
            ctx_capacity,
        }
    }

    /// Downcast the generic target to a qwen35 `ModelSlot` (same as DflashSpeculator).
    fn slot(target: &mut dyn SpecTarget) -> Result<&mut ModelSlot, String> {
        target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or_else(|| "Qwen35MtpDrafter: target is not a Qwen3.5 ModelSlot".to_string())
    }

    /// Allocate `state` against `slot` on first use (greedy config).
    fn ensure_state(&mut self, gpu: &mut Gpu, slot: &ModelSlot) -> Result<(), String> {
        if self.state.is_none() {
            let mut st = MtpSpecState::new_for_slot_with_kv_mode(
                gpu,
                slot,
                &self.head,
                self.max_n,
                MtpKvMode::Q8,
            )
            .map_err(|e| format!("alloc MtpSpecState: {e}"))?;
            // Greedy: disable the p_min chain early-exit and keep the greedy
            // sampling default (the daemon only routes here at temp≈0).
            st.set_p_min(0.0);
            st.set_sampling(MtpSamplingConfig::default(), 42);
            self.state = Some(st);
        }
        Ok(())
    }
}

impl MtpDrafter for Qwen35MtpDrafter {
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
        self.mtp_prefill_with_abort(gpu, target, fill_tokens, start_pos, cache_hit, &never)
            .map(|token| token.expect("non-abortable qwen35 MTP prefill aborted"))
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
        self.mtp_prefill_with_abort(gpu, target, fill_tokens, start_pos, cache_hit, abort)
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        // qwen35 enforces tool-call grammar post-hoc in the emission layer; the
        // in-step grammar handle is unused here. `k` is fixed at build time
        // (== state.max_n), so the step reads it from the state.
        debug_assert_eq!(k, self.max_n, "qwen35 MTP k must match build-time max_n");
        let slot = Self::slot(target)?;
        let state = self
            .state
            .as_mut()
            .ok_or("Qwen35MtpDrafter: mtp_step before mtp_prefill")?;
        let r = spec_step_mtp_compressed_serial(gpu, slot, &self.head, state, position, seed, eos)
            .map_err(|e| e.to_string())?;
        Ok(MtpWindow {
            committed: r.committed,
            accepted: r.accept_count,
            drafts_generated: r.drafts_generated,
        })
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) {
        let _ = self.mtp_reset_checked(gpu);
    }

    fn mtp_reset_checked(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35 MTP reset: {e}"))?;
        }
        Ok(())
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(state) = self.state {
            state.free_gpu(gpu);
        }
        self.head.free_gpu(gpu);
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

impl Qwen35MtpDrafter {
    fn mtp_prefill_with_abort(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        _cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<Option<u32>, String> {
        if abort() {
            return Ok(None);
        }
        if abort() {
            return Ok(None);
        }
        let slot = Self::slot(target)?;
        self.ensure_state(gpu, slot)?;
        let state = self.state.as_mut().expect("ensure_state set it");

        if abort() {
            return Ok(None);
        }
        let prefill = prefill_trunk_and_mtp_cache_abortable(
            gpu,
            slot,
            &self.head,
            state,
            fill_tokens,
            start_pos,
            abort,
        )
        .map_err(|e| format!("mtp prefill: {e}"))?;
        if prefill.is_none() {
            return Ok(None);
        }

        // Seed = greedy argmax of the trunk logits at the last prefilled
        // position (`prefill_trunk_and_mtp_cache` leaves them in scratch.logits).
        if abort() {
            return Ok(None);
        }
        let logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| format!("download seed logits: {e}"))?;
        if abort() {
            return Ok(None);
        }
        let first_token = logits
            .iter()
            .enumerate()
            .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i as u32, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        Ok((!abort()).then_some(first_token))
    }
}

/// Build the qwen35 MTP speculator (the boxed `dyn Speculator` the loader's
/// `build_speculator` returns). The `MtpSpecState` is allocated lazily on the
/// first `prefill`.
pub fn build_qwen35_mtp_speculator(
    head: Qwen35MtpHead,
    max_n: usize,
    ctx_capacity: usize,
) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(Qwen35MtpDrafter::new(
        head,
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
    fn mtp_speculator_qwen35_is_send() {
        _assert_send::<MtpSpeculator<super::Qwen35MtpDrafter>>();
    }
}
