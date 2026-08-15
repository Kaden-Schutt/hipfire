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
    prefill_trunk_and_mtp_cache, spec_step_mtp_compressed_serial_with_k, MtpSamplingConfig,
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
        // Cold start: reset the trunk recurrent state (target owns it) BEFORE
        // borrowing the concrete slot, so positions start at 0.
        if !cache_hit {
            target.reset_recurrent(gpu)?;
        }
        let slot = Self::slot(target)?;
        self.ensure_state(gpu, slot)?;
        let state = self.state.as_mut().expect("ensure_state set it");
        if !cache_hit {
            // Clear the head KV so its absolute positions start clean too.
            state
                .reset(gpu)
                .map_err(|e| format!("mtp state reset: {e}"))?;
        }

        prefill_trunk_and_mtp_cache(gpu, slot, &self.head, state, fill_tokens, start_pos)
            .map_err(|e| format!("mtp prefill: {e}"))?;

        // Seed = greedy argmax of the trunk logits at the last prefilled
        // position (`prefill_trunk_and_mtp_cache` leaves them in scratch.logits).
        let logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| format!("download seed logits: {e}"))?;
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
        Ok(first_token)
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
        // in-step grammar handle is unused here. `k` is the per-call budget from
        // MtpSpeculator (already clamped by remaining max_emit); the fixed-window
        // compressed-serial core must honor it — never draft state.max_n blindly.
        let k = k.min(self.max_n);
        let slot = Self::slot(target)?;
        let state = self
            .state
            .as_mut()
            .ok_or("Qwen35MtpDrafter: mtp_step before mtp_prefill")?;
        let r = spec_step_mtp_compressed_serial_with_k(
            gpu, slot, &self.head, state, position, seed, eos, k,
        )
        .map_err(|e| e.to_string())?;
        debug_assert!(
            r.committed.len() <= k + 1,
            "qwen35 MTP committed past k budget"
        );
        Ok(MtpWindow {
            committed: r.committed,
            accepted: r.accept_count,
            drafts_generated: r.drafts_generated,
        })
    }

    fn mtp_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        // Forced tokens must land in BOTH the trunk and the MTP head KV /
        // prev_hidden. Plain spec_advance only moves trunk state, leaving an
        // unwritten hole in the head that poisons later draft steps.
        if tokens.is_empty() {
            return Ok(true);
        }
        if abort() {
            return Ok(true);
        }
        let slot = Self::slot(target)?;
        self.ensure_state(gpu, slot)?;
        let state = self.state.as_mut().expect("ensure_state set it");
        prefill_trunk_and_mtp_cache(gpu, slot, &self.head, state, tokens, start_pos)
            .map_err(|e| format!("qwen35 MTP forced advance: {e}"))?;
        Ok(true)
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35-mtp drafter reset: {e}"))?;
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
