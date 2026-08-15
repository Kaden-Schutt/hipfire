// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 implementation of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! `impl SpecTarget for MiniMaxBundle` lets the model-free `NgramSpeculator`
//! drive a MiniMax-M2 target (arch_id=10) with no arch knowledge. Pure GQA
//! attention with NO recurrent state, so `commit_prefix` is a no-op (the
//! accepted-prefix KV the verify wrote is already correct; the rejected tail is
//! overwritten by the next verify). Same structural pattern as qwen2 (arch_id=7).
//!
//! [`MiniMaxBundle`] is defined here (in the arch crate) so that `SpecTarget`
//! can be implemented for it without orphan-rule violations — exactly the
//! pattern used by `Deepseek4Bundle` in `hipfire-arch-deepseek4`.
//!
//! VERIFY IS SEQUENTIAL: [`SpecTarget::verify_block`] runs one `forward::decode_step`
//! per candidate token. This is the correct byte-identical baseline; a batched
//! verify kernel is an explicit follow-up.

use crate::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

/// Owned MiniMax-M2 model bundle — the `ModelState::Minimax` payload and the
/// spec-decode target. Bundles config + weights + state + eos so the daemon can
/// borrow it as `&mut dyn SpecTarget`.
pub struct MiniMaxBundle {
    pub config: MiniMaxConfig,
    pub weights: MiniMaxWeights,
    pub state: MiniMaxState,
    pub eos_tok: u32,
}

/// MiniMax verify scratch: nothing persistent. Pure attention → no recurrent
/// snapshot to carry between windows.
pub struct MiniMaxSpecScratch;

impl SpecScratch for MiniMaxSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

impl SpecTarget for MiniMaxBundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Pure attention: no recurrent state to zero. Rewind the position cursor
        // so the next prefill writes from slot 0. Mirrors the daemon's
        // arch_id=10 reset handler.
        self.state.reset();
        Ok(())
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Ok(Box::new(MiniMaxSpecScratch))
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
        // Pure attention: "reset" rewinds the position cursor; the per-token
        // prefill then overwrites KV at the absolute positions it writes.
        if reset {
            self.state.reset();
        }
        self.state.n_tokens = start_pos;
        let mut last_logits: Vec<f32> = Vec::new();
        for &tok in tokens {
            if abort() {
                self.state.reset();
                return Ok(SpecAdvance::Aborted);
            }
            let position = self.state.n_tokens as u32;
            // decode_step downloads + returns the full logits Vec<f32>.
            // decode_step_body also sets state.n_tokens = position + 1.
            last_logits = crate::forward::decode_step(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                tok,
                position,
            )
            .map_err(|e| format!("{e:?}"))?;
        }
        let last_argmax = last_logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: Some(last_logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        // Sequential verify: `decode_step(block[i])` predicts the token AFTER
        // block[i] (with block[0..i] already in the KV cache), which is exactly
        // `argmax[i]`. Position the cursor at `position` first so the writes land
        // at the right absolute slots (overwriting any rejected-tail KV from a
        // prior window).
        self.state.n_tokens = position;
        let mut out = Vec::with_capacity(block.len());
        for &tok in block {
            let pos = self.state.n_tokens as u32;
            let logits = crate::forward::decode_step(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                tok,
                pos,
            )
            .map_err(|e| format!("{e:?}"))?;
            // decode_step returns Vec<f32> (downloaded); argmax host-side.
            let argmax = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            out.push(argmax);
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
        // Pure attention: verify's accepted-prefix KV is already correct, and the
        // rejected tail is overwritten by the next verify. Nothing to rewind.
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq
    }

    // kv_cache_mut: defaulted to `None` — MiniMaxState uses its own KvCache
    // (not the shared llama::KvCache borrow the eviction sites expect), and
    // arch_id=10 has no FlashCASK eviction wired. The daemon's eviction sites are
    // `if let Some(ev)`-gated, so this is never reached for a non-evicting target.
}
