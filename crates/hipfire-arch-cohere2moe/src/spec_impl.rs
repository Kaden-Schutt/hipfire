// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE implementation of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! `impl SpecTarget for Cohere2MoeBundle` lets the model-free `NgramSpeculator`
//! drive a Cohere2-MoE target (North-Mini-Code, arch_id=12) with no arch
//! knowledge. Pure GQA attention (no recurrent state), so `commit_prefix` is a
//! no-op: the accepted-prefix KV the verify wrote is already correct, and the
//! rejected tail is overwritten by the next verify. `kv_cache_mut` stays at its
//! `None` default — Cohere2MoeState holds its own KV, not the shared
//! `llama::KvCache`, so arch_id=12 has no FlashCASK eviction.
//!
//! VERIFY IS SEQUENTIAL: `verify_block` runs one `decode_step` per candidate
//! block token. This is the correct byte-identical baseline AND it sidesteps
//! the sliding-window batched-mask problem entirely — each sequential
//! `decode_step` naturally applies the right windowed or full-causal attention
//! mask for the current token's position. The "parallel block" structure of
//! Cohere2-MoE (single RMSNorm feeding BOTH attention AND FFN branches) does
//! not help here because `forward_batch` uses a different scratch layout for
//! causal-within-block verification. Cold speculative prefill is different:
//! it uses the same chunked `forward_batch` path as AR, selected by the
//! explicit cold-start signal, so the first speculative token remains
//! numerically parity-aligned.
//!
//! NOTE (follow-up): a block-parallel `forward_batch`-based verify could
//! reuse the existing `attention_flash_q8_0_windowed` path with a
//! `positions`-array carrying `[P, P+1, ..., P+B-1]` and `q8_windowed=true`.
//! The blocker is that `AttnQ8_0KvBatchedMaskedWindowed` (the
//! `q8_windowed` + batch path) writes KV at all B positions but attends
//! each query only to the PRIOR history and NOT the causal-within-block
//! positions — the same gap that blocked qwen2's first batched-verify
//! attempt. The fix for qwen2 was `attention_decode_batched_history`; the
//! equivalent for cohere2moe requires a windowed variant. Implement as a
//! separate PR with a byte-parity test against this sequential baseline.

use crate::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights};
use crate::config::Cohere2MoeConfig;
use crate::forward;
use hipfire_runtime::gpu_cleanup::{BundleTeardown, GpuCleanupFailure};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

/// Arch-crate bundle for Cohere2-MoE spec-decode. Mirrors `Qwen2Bundle` in
/// `hipfire-arch-qwen2`: the arch crate owns both the constituent types
/// (`Cohere2MoeConfig`, `Cohere2MoeWeights`, `Cohere2MoeState`) and the
/// `SpecTarget` impl, keeping the impl in the defining crate.
///
/// The loader's `hipfire-loader::Cohere2MoeBundle` (which stores `eos_tok`
/// alongside these fields and lives in the loader crate) cannot be the
/// `SpecTarget` impl host because the loader crate depends on this arch
/// crate — the orphan rule requires the impl to be in the crate that owns
/// either the trait or the type. Integration hook: the
/// `Cohere2MoeCarrier::spec_target_guard` in `carriers.rs` must downcast
/// the loader bundle into (or build) a `Cohere2MoeBundle` (arch-crate) and
/// wrap it in `InPlaceGuard { bundle }`.
pub struct Cohere2MoeBundle {
    pub config: Cohere2MoeConfig,
    pub weights: Cohere2MoeWeights,
    pub state: Cohere2MoeState,
    pub eos_tok: u32,
}

impl Cohere2MoeBundle {
    fn advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        cold_start: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String> {
        self.state.n_tokens = start_pos;
        if use_batched_cold_prefill(
            cold_start,
            tokens.len(),
            forward::forward_batch_supported(&self.weights),
        ) {
            let mut last_logits = Vec::new();
            let mut offset = 0;
            while offset < tokens.len() {
                if abort() {
                    return Ok(SpecAdvance::Aborted);
                }
                let end = (offset + 256).min(tokens.len());
                let pos = self.state.n_tokens;
                last_logits = forward::forward_batch(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    gpu,
                    &tokens[offset..end],
                    pos,
                )
                .map_err(|e| format!("{e:?}"))?;
                offset = end;
            }
            let last_argmax = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            return Ok(SpecAdvance::Ready {
                last_argmax,
                last_logits: Some(last_logits),
            });
        }
        for &tok in tokens {
            if abort() {
                return Ok(SpecAdvance::Aborted);
            }
            let pos = self.state.n_tokens as u32;
            forward::decode_step(&self.config, &self.weights, &mut self.state, gpu, tok, pos)
                .map_err(|e| format!("{e:?}"))?;
        }
        let last_argmax = gpu
            .argmax_f32(&self.state.logits, self.config.vocab_size)
            .map_err(|e| format!("{e:?}"))?;
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: None,
        })
    }
}

/// Cohere2-MoE verify scratch: nothing persistent. Pure attention — no
/// recurrent state to snapshot between windows. The verify reuses the
/// bundle's own `Cohere2MoeState` scratch (dense GQA + MoE, no DeltaNet).
pub struct Cohere2MoeSpecScratch;

impl SpecScratch for Cohere2MoeSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

impl SpecTarget for Cohere2MoeBundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Pure attention: no recurrent state to zero. Clear the KV cache and
        // rewind the position cursor so the next prefill writes from slot 0.
        // Mirrors the daemon's arch_id=12 reset handler.
        self.state
            .reset(gpu)
            .map_err(|e| format!("cohere2moe reset_recurrent: {e}"))
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Ok(Box::new(Cohere2MoeSpecScratch))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.reset_recurrent(gpu)
                .map_err(|e| format!("cohere2moe spec_advance reset: {e}"))?;
        }
        self.advance(gpu, tokens, start_pos, false, abort)
    }

    fn spec_advance_cold_start(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        cold_start: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        eprintln!(
            "[cohere-spec-prefill] mode={} cold_start={} token_count={}",
            if use_batched_cold_prefill(
                cold_start,
                tokens.len(),
                forward::forward_batch_supported(&self.weights),
            ) {
                "batched"
            } else {
                "per_token"
            },
            cold_start,
            tokens.len(),
        );
        self.advance(gpu, tokens, start_pos, cold_start, abort)
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        // Sequential verify: `decode_step(block[i])` at absolute position
        // `position + i` predicts the token AFTER block[i] — exactly
        // `argmax[i]`. The sliding-window flash path
        // (`attention_flash_q8_0_windowed`) is called naturally per-step with
        // the correct absolute position, so each token attends only the last
        // `sliding_window` keys (for Sliding layers) or the full causal context
        // (for Full/NoPE layers) — no batched-mask bookkeeping required.
        //
        // Position the cursor at `position` so writes land at the right
        // absolute KV slots (overwriting any rejected-tail KV from a prior
        // window).
        self.state.n_tokens = position;
        let mut out = Vec::with_capacity(block.len());
        for &tok in block {
            let pos = self.state.n_tokens as u32;
            forward::decode_step(&self.config, &self.weights, &mut self.state, gpu, tok, pos)
                .map_err(|e| format!("{e:?}"))?;
            // `decode_step` already advances `state.n_tokens` to `position + 1`
            // — no manual step (see spec_advance).
            out.push(
                gpu.argmax_f32(&self.state.logits, self.config.vocab_size)
                    .map_err(|e| format!("{e:?}"))?,
            );
        }
        Ok(out)
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Pure attention: verify's accepted-prefix KV is already correct, and
        // the rejected tail is overwritten by the next verify. Nothing to
        // rewind (no recurrent / compressed state). Mirrors qwen2.
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq
    }

    // kv_cache_mut: default None — Cohere2MoeState is not a `llama::KvCache`,
    // and arch_id=12 has no FlashCASK eviction (the daemon's eviction sites
    // are `if let Some(ev)`-gated, so this is never reached).
}

fn use_batched_cold_prefill(cold_start: bool, token_count: usize, supported: bool) -> bool {
    cold_start && token_count > 1 && supported
}

impl BundleTeardown for Cohere2MoeBundle {
    /// Generic checked teardown the loader dispatches on: frees weights then
    /// state, each with CHECKED frees. Owners that survive are carried in the
    /// returned [`GpuCleanupFailure`] for exact-retention retry — no
    /// best-effort free is used as a correctness mechanism.
    fn free_checked(self, gpu: &mut Gpu) -> Result<(), GpuCleanupFailure> {
        let Cohere2MoeBundle {
            config: _,
            weights,
            state,
            eos_tok: _,
        } = self;
        let mut cf = GpuCleanupFailure::empty();
        if let Err(f) = weights.free_checked(gpu) {
            cf.merge(f);
        }
        if let Err(f) = state.free_checked(gpu) {
            cf.merge(f);
        }
        if cf.is_empty() {
            Ok(())
        } else {
            Err(cf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::use_batched_cold_prefill;

    #[test]
    fn speculative_cold_prefill_matches_batched_ar_only_on_cold_multi_token_span() {
        assert!(use_batched_cold_prefill(true, 8, true));
        assert!(!use_batched_cold_prefill(false, 8, true));
        assert!(!use_batched_cold_prefill(true, 1, true));
        assert!(!use_batched_cold_prefill(true, 8, false));
    }
}
