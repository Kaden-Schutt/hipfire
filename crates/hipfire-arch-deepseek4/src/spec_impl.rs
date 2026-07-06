// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 impl of the arch-generic `hipfire_runtime::spec::SpecTarget`.
//!
//! [`Deepseek4Bundle`] owns the model pieces the daemon + MTP drafter need
//! (config + weights + recurrent state + eos) so deepseek4 can be borrowed as a
//! `&mut dyn SpecTarget` exactly like the qwen35 `ModelSlot` — the prerequisite
//! for routing it through the unified spec loop. The MTP draft+verify itself is
//! the [`crate::spec_decode`] fused step, reached by downcasting this bundle in
//! the deepseek4 `MtpDrafter` impl; deepseek4 never pairs with the model-free
//! n-gram drafter, so the n-gram-verify primitives are intentional error stubs.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

/// Owned deepseek4 model state — the future `ModelState::Deepseek4` payload and
/// the spec-decode target. Bundles config + weights + recurrent state + eos so
/// the daemon can borrow it as `&mut dyn SpecTarget`.
pub struct Deepseek4Bundle {
    pub config: DeepseekV4Config,
    pub weights: DeepseekV4Weights,
    pub state: DeepseekV4State,
    pub eos_tok: u32,
}

const DS4_NO_NGRAM: &str =
    "deepseek4 does not support model-free n-gram drafting (it pairs only with MTP spec-decode)";

impl SpecTarget for Deepseek4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) {
        // n_tokens → 0 + mtp_last_hidden cleared; the position-indexed KV / SWA /
        // compressed-KV rings are overwritten by the next prefill, never read
        // beyond n_tokens (see `DeepseekV4State::reset`). No GPU work needed.
        self.state.reset();
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.config.max_position_embeddings
    }

    // ── n-gram-verify primitives (intentionally unsupported) ────────────────
    // deepseek4 only ever pairs with `MtpSpeculator`, whose `MtpDrafter` downcasts
    // this bundle and runs the fused `spec_decode` step — it never invokes these
    // model-free verify hooks. Erroring keeps a wrong pairing loud instead of
    // silently miscomputing.
    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Err(DS4_NO_NGRAM.into())
    }

    fn spec_advance(
        &mut self,
        _gpu: &mut Gpu,
        _tokens: &[u32],
        _start_pos: usize,
        _reset: bool,
        _abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String> {
        Err(DS4_NO_NGRAM.into())
    }

    fn verify_block(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<Vec<u32>, String> {
        Err(DS4_NO_NGRAM.into())
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        Err(DS4_NO_NGRAM.into())
    }
}
