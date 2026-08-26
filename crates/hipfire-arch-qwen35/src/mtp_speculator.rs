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
    prefill_trunk_and_mtp_cache, sample_from_logits, spec_step_mtp_compressed_serial_with_k,
    spec_step_mtp_compressed_serial_with_takeover_candidates, MtpSamplingConfig, MtpSpecState,
};
use crate::speculative::ModelSlot;
use hipfire_runtime::ngram_mod::{NgramModConfig, NgramModPool};
use hipfire_runtime::spec::{
    MtpDrafter, MtpRequestStats, MtpSpeculator, MtpWindow, SpecGrammar, SpecRequestConfig,
    SpecTarget, Speculator,
};
use rdna_compute::Gpu;

/// Env `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,N_MAX}` with production defaults.
/// `None` when the triple is invalid (max>64, zero match/max, or min>max).
fn ngram_mod_env_config() -> Option<NgramModConfig> {
    let n_match: usize = std::env::var("HIPFIRE_NGRAM_MOD_N_MATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let n_min: usize = std::env::var("HIPFIRE_NGRAM_MOD_N_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let n_max: usize = std::env::var("HIPFIRE_NGRAM_MOD_N_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    if n_max <= 64 && n_max >= 1 && n_match >= 1 && n_min <= n_max {
        Some(NgramModConfig {
            capacity: 1 << 22,
            n_match,
            n_min,
            n_max,
        })
    } else {
        None
    }
}

/// qwen35 MTP drafter. `state` is allocated on the first `mtp_prefill` (it needs
/// `&mut Gpu` + the concrete `ModelSlot`, neither available at load-time
/// construction). Sampling is installed once per request via
/// [`MtpDrafter::configure_request`] and persisted across warm LCP hits.
///
/// Optional n-gram-mod composition is owned here (no Arc/Mutex): the pool may
/// survive requests; request context/retirement reset per `configure_request`.
pub struct Qwen35MtpDrafter {
    head: Qwen35MtpHead,
    state: Option<MtpSpecState>,
    max_n: usize,
    ctx_capacity: usize,
    request: SpecRequestConfig,
    ngram_pool: Option<NgramModPool>,
    ngram_active: bool,
    ngram_context: Vec<u32>,
    ngram_indexed_until: usize,
    /// Number of host-emitted request tokens already appended to `ngram_context`.
    ngram_emitted_len: usize,
    ngram_retired: bool,
    /// Request-local wire counters (reset on configure_request).
    stats: MtpRequestStats,
}

impl Qwen35MtpDrafter {
    pub fn new(head: Qwen35MtpHead, max_n: usize, ctx_capacity: usize) -> Self {
        Self {
            head,
            state: None,
            max_n: max_n.clamp(1, 8),
            ctx_capacity,
            request: SpecRequestConfig::default(),
            ngram_pool: None,
            ngram_active: false,
            ngram_context: Vec::new(),
            ngram_indexed_until: 0,
            ngram_emitted_len: 0,
            ngram_retired: false,
            stats: MtpRequestStats::default(),
        }
    }

    /// Downcast the generic target to a qwen35 `ModelSlot` (same as DflashSpeculator).
    fn slot(target: &mut dyn SpecTarget) -> Result<&mut ModelSlot, String> {
        target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or_else(|| "Qwen35MtpDrafter: target is not a Qwen3.5 ModelSlot".to_string())
    }

    /// Install sampling without changing the independent MTP draft-confidence
    /// cutoff initialized by `MtpSpecState` from its arch/env default.
    fn apply_request(state: &mut MtpSpecState, cfg: SpecRequestConfig) {
        let top_p = if cfg.top_p > 0.0 { cfg.top_p.min(1.0) } else { 1.0 };
        state.set_sampling(
            MtpSamplingConfig {
                temp: cfg.temp,
                top_k: cfg.top_k,
                top_p,
                min_p: cfg.min_p,
            },
            cfg.rng_seed,
        );
    }

    fn reset_ngram_request(&mut self) {
        self.ngram_context.clear();
        self.ngram_indexed_until = 0;
        self.ngram_emitted_len = 0;
        self.ngram_retired = false;
        self.stats = MtpRequestStats::default();
    }

    fn ngram_learn_tokens(&mut self, tokens: &[u32]) {
        if !self.ngram_active || tokens.is_empty() {
            return;
        }
        self.ngram_context.extend_from_slice(tokens);
        if let Some(pool) = self.ngram_pool.as_mut() {
            pool.insert_range(&self.ngram_context, self.ngram_indexed_until);
        }
        self.ngram_indexed_until = self.ngram_context.len();
    }

    /// Learn only tokens the emitter retained. GPU verify may commit a longer
    /// speculative tail, so `mtp_step` results are never authoritative history.
    fn ngram_sync_emitted(&mut self, emitted: &[u32]) {
        if !self.ngram_active {
            return;
        }
        debug_assert!(
            emitted.len() >= self.ngram_emitted_len,
            "qwen35 MTP emitted history must grow monotonically"
        );
        if emitted.len() <= self.ngram_emitted_len {
            return;
        }
        self.ngram_learn_tokens(&emitted[self.ngram_emitted_len..]);
        self.ngram_emitted_len = emitted.len();
    }

    /// Allocate `state` against `slot` on first use, plus both compressed
    /// sidecar scratches when the head ships a compressed vocab, then apply
    /// the current request. Subsequent calls are a no-op (warm hits persist).
    ///
    /// Verify capacity expands to the configured n-gram ceiling only when the
    /// process explicitly enables the modifier. This preserves normal MTP's
    /// smaller allocation while allowing a later eligible request to arm
    /// n-gram-mod without reallocating/destroying warm prefix state.
    fn ensure_state(&mut self, gpu: &mut Gpu, slot: &ModelSlot) -> Result<(), String> {
        if self.state.is_none() {
            let verify_capacity =
                if std::env::var("HIPFIRE_MTP_NGRAM").ok().as_deref() == Some("1") {
                    ngram_mod_env_config()
                        .map(|cfg| self.max_n.max(cfg.n_max))
                        .unwrap_or(self.max_n)
                } else {
                    self.max_n
                };
            let mut st = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
                gpu,
                slot,
                &self.head,
                self.max_n,
                verify_capacity,
                MtpKvMode::Q8,
            )
            .map_err(|e| format!("alloc MtpSpecState: {e}"))?;
            if let Some(cvs) = self.head.weights.compressed_vocab_size {
                let compressed_result = st
                    .mtp_scratch
                    .ensure_compressed_logits(gpu, cvs)
                    .map_err(|e| format!("alloc logits_compressed: {e}"))
                    .and_then(|_| {
                        st.ensure_compressed_lm_logits(gpu, cvs)
                            .map_err(|e| format!("alloc mtp_lm_logits_compressed: {e}"))
                    });
                if let Err(error) = compressed_result {
                    st.free_gpu(gpu);
                    return Err(error);
                }
            }
            Self::apply_request(&mut st, self.request);
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
        prompt_tokens: &[u32],
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<u32, String> {
        if abort() {
            return Err("aborted".into());
        }
        // Cold start: reset the trunk recurrent state (target owns it) BEFORE
        // borrowing the concrete slot, so positions start at 0. Warm hit
        // preserves target + MTP cache for honest LCP suffix prefill.
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
        if abort() {
            return Err("aborted".into());
        }

        // Seed from the trunk logits at the last prefilled position
        // (`prefill_trunk_and_mtp_cache` leaves them in scratch.logits).
        // Greedy = argmax; temp>0 draws from the state's continuous RNG/config.
        let logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| format!("download seed logits: {e}"))?;
        let first_token = if state.sampling.is_greedy() {
            logits
                .iter()
                .enumerate()
                .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i as u32, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        } else {
            sample_from_logits(&logits, &state.sampling, &mut state.rng).0
        };

        // Seed request-local n-gram history exactly once. Prefix realignment
        // calls prefill again inside the same request; preserving this context
        // keeps retirement, counters, and already-emitted history intact.
        if self.ngram_active && self.ngram_context.is_empty() {
            self.ngram_context.extend_from_slice(prompt_tokens);
            self.ngram_context.push(first_token);
            if let Some(pool) = self.ngram_pool.as_mut() {
                let n_match = pool.config().n_match;
                pool.insert_range(&self.ngram_context, n_match);
            }
            self.ngram_indexed_until = self.ngram_context.len();
            self.ngram_emitted_len = 1;
        }
        Ok(first_token)
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        // qwen35 enforces tool-call grammar post-hoc in the emission layer; the
        // in-step grammar handle is unused here. `k` is the per-call budget from
        // MtpSpeculator (already clamped by remaining max_emit); the fixed-window
        // compressed-serial core must honor it — never draft state.max_n blindly.
        // Sampling lives on `state` from configure_request; no per-step reconfig.
        self.ngram_sync_emitted(emitted);
        if self.ngram_active && !emitted.is_empty() {
            debug_assert!(
                self.ngram_context.ends_with(emitted),
                "qwen35 MTP ngram context suffix must match emitted (ctx={}, emitted={})",
                self.ngram_context.len(),
                emitted.len()
            );
        }
        let ngram_cands = if self.ngram_active {
            self.ngram_pool
                .as_ref()
                .and_then(|pool| pool.draft(&self.ngram_context, k))
        } else {
            None
        };
        let used_ngram = ngram_cands.as_ref().is_some_and(|c| !c.is_empty());
        let retired = self.ngram_retired;
        let modifier = self.ngram_active;
        let native_k = k.min(self.max_n);

        let slot = Self::slot(target)?;
        let r = {
            let state = self
                .state
                .as_mut()
                .ok_or("Qwen35MtpDrafter: mtp_step before mtp_prefill")?;
            if used_ngram {
                let cands = ngram_cands.as_ref().expect("used_ngram implies candidates");
                spec_step_mtp_compressed_serial_with_takeover_candidates(
                    gpu, slot, &self.head, state, position, seed, eos, cands, retired,
                )
            } else if modifier && retired {
                spec_step_mtp_compressed_serial_with_k(
                    gpu, slot, &self.head, state, position, seed, eos, 0,
                )
            } else {
                spec_step_mtp_compressed_serial_with_k(
                    gpu, slot, &self.head, state, position, seed, eos, native_k,
                )
            }
            .map_err(|e| e.to_string())?
        };
        let budget = if used_ngram {
            k
        } else if modifier && retired {
            0
        } else {
            native_k
        };
        debug_assert!(
            r.committed.len() <= budget + 1,
            "qwen35 MTP committed past k budget"
        );

        if used_ngram {
            self.stats.ngram_mod_windows += 1;
            self.stats.ngram_mod_drafts += r.drafts_generated;
            self.stats.ngram_mod_accepted += r.accept_count;
            if let Some(pool) = self.ngram_pool.as_mut() {
                let _ = pool.record_draft_result(
                    r.drafts_generated as u32,
                    r.accept_count as u32,
                );
            }
            if r.accept_count > 0 {
                self.ngram_retired = true;
            }
        } else if modifier {
            if retired {
                self.stats.ar_windows += 1;
            } else {
                self.stats.mtp_windows += 1;
            }
        } else {
            self.stats.mtp_windows += 1;
        }
        if modifier {
            eprintln!(
                "[mtp-ngram] used={} drafted={} accepted={} retired={}",
                u8::from(used_ngram),
                r.drafts_generated,
                r.accept_count,
                u8::from(self.ngram_retired),
            );
        }

        self.stats.mtp_retired = self.ngram_retired;
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
        self.reset_ngram_request();
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35-mtp drafter reset: {e}"))?;
        }
        Ok(())
    }

    fn mtp_reset_for_realign(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(state) = self.state.as_mut() {
            state
                .reset(gpu)
                .map_err(|e| format!("qwen35-mtp drafter realign reset: {e}"))?;
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

    fn proposal_capacity(&self) -> usize {
        if self.ngram_active {
            self.ngram_pool
                .as_ref()
                .map(|p| self.max_n.max(p.config().n_max))
                .unwrap_or(self.max_n)
        } else {
            self.max_n
        }
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        false
    }

    fn supports_temp_verify(&self) -> bool {
        true
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
        if let Some(state) = self.state.as_mut() {
            Self::apply_request(state, cfg);
        }
        self.reset_ngram_request();
        self.ngram_active = false;
        if !cfg.allow_ngram_modifier || cfg.temp > 1e-6 {
            return;
        }
        let Some(new_cfg) = ngram_mod_env_config() else {
            return;
        };
        let reuse = self
            .ngram_pool
            .as_ref()
            .is_some_and(|p| p.config() == &new_cfg);
        if !reuse {
            match NgramModPool::new(new_cfg) {
                Ok(pool) => self.ngram_pool = Some(pool),
                Err(_) => return,
            }
        }
        self.ngram_active = self.ngram_pool.is_some();
        self.stats.mtp_ngram = self.ngram_active;
    }

    fn request_stats(&self) -> MtpRequestStats {
        let mut s = self.stats;
        s.mtp_ngram = self.ngram_active;
        s.mtp_retired = self.ngram_retired;
        s.ngram_mod_accept_rate = if s.ngram_mod_drafts > 0 {
            let rate = s.ngram_mod_accepted as f64 / s.ngram_mod_drafts as f64;
            (rate * 1000.0).round() / 1000.0
        } else {
            0.0
        };
        s
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
