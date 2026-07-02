// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serving seam for `lfm2-moe` (arch_id 11): `SimpleAr` + `ServingBackend` on
//! [`Lfm2Backend`], routing plain generation through the shared `run_simple_ar`
//! → prefill → `decode_loop` loop (the same dense-AR seam as `LlamaBackend` /
//! `ZayaModel` / `NemotronModel`).
//!
//! `prefill` drives the crate's existing batched [`forward::prefill_batch`]
//! (hybrid LIV short-conv + GQA attention → top-4 MoE FFN), priming the KV +
//! conv-state cache and leaving the last token's logits in `state.logits`; each
//! `decode_step` advances one token via [`forward::decode_step`].
//!
//! ## Migration note — DFlash / CASK are NOT yet on this seam
//!
//! LFM2's speculative-decode (DFlash) and TriAttention/CASK eviction paths still
//! ride the legacy `LoadedModel` + daemon `generate_*` plumbing. `run_simple_ar`
//! covers only plain greedy/sampled AR, so this backend deliberately advertises
//! no fast-path [`ArchCaps`] yet. Folding DFlash/CASK in means a bespoke
//! [`ServingBackend::serve`] override (gated by `caps().dflash` / a loaded
//! drafter) rather than delegating to `run_simple_ar` — see the prefill-seam
//! plan. Until that lands, the daemon keeps using the legacy path when a draft
//! or CASK sidecar is present.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use crate::{forward, ARCH_ID};
use hipfire_runtime::arch::{
    run_simple_ar, ArchCaps, GenerateCtx, ServeOutcome, ServingBackend, SimpleAr,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};

/// A loaded LFM2.5-MoE model with GPU-resident weights and per-decode state
/// (KV cache + short-conv rolling state + logits), so decode is O(1) per token.
pub struct Lfm2Backend {
    config: Lfm2MoeConfig,
    weights: Lfm2MoeWeights,
    state: Lfm2MoeState,
    eos: u32,
}

impl Lfm2Backend {
    /// Load weights from an HFQ onto the GPU and allocate the decode state.
    /// `max_seq` bounds the KV/conv-state window; `physical_cap` sizes the
    /// backing allocation. `eos` is the daemon-resolved end-of-turn token (LFM2
    /// resolves it from tokenizer/metadata at load time, not from the config).
    pub fn from_hfq(
        gpu: &mut Gpu,
        hfq: &mut HfqFile,
        max_seq: usize,
        physical_cap: usize,
        eos: u32,
    ) -> Result<Self, String> {
        let config = Lfm2MoeConfig::from_hfq(hfq)?;
        let weights = Lfm2MoeWeights::load(hfq, &config, gpu)?;
        let state = Lfm2MoeState::new_with_physical_cap(gpu, &config, max_seq, physical_cap)
            .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_physical_cap failed: {e}"))?;
        Ok(Self {
            config,
            weights,
            state,
            eos,
        })
    }

    pub fn config(&self) -> &Lfm2MoeConfig {
        &self.config
    }
}

impl SimpleAr for Lfm2Backend {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("lfm2 prefill: empty prompt".to_string());
        }
        // Batched hybrid prefill; leaves the last token's logits in
        // `state.logits` (the host Vec it also returns is unused on this path —
        // the daemon sampler reads the GPU tensor via `logits()`).
        forward::prefill_batch(&self.config, &self.weights, &mut self.state, gpu, tokens)
            .map(|_| ())
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        forward::decode_step(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            token,
            pos as u32,
        )
        .map(|_| ())
    }

    fn logits(&self) -> &GpuTensor {
        &self.state.logits
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

impl ServingBackend for Lfm2Backend {
    fn arch_id(&self) -> u32 {
        ARCH_ID
    }

    fn caps(&self) -> ArchCaps {
        // No fast-path caps on the seam yet — DFlash/CASK still ride the legacy
        // path (see module migration note). Plain AR via `run_simple_ar`.
        ArchCaps::default()
    }

    fn eos_token(&self) -> u32 {
        self.eos
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let eos = self.eos;
        run_simple_ar(gpu, self, tok, eos, ctx)
    }

    fn reset_session(&mut self, gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        self.state.reset(gpu)
    }

    fn unload(self: Box<Self>, _gpu: &mut Gpu) {
        // LFM2's weights/state expose no explicit GPU-free API (matching the
        // legacy load path, which also relied on model-swap reclamation); drop
        // the boxed backend. A dedicated `free_gpu` is tracked with the
        // DFlash/CASK seam migration.
        drop(self);
    }
}
