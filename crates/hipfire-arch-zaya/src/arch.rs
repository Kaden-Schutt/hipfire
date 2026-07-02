// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serving seam for `zaya` (arch_id 16): `SimpleAr` + `ServingBackend` on
//! [`ZayaModel`], routing through the shared `run_simple_ar` → prefill →
//! decode loop (the same dense-AR seam as Llama/Nemotron). `prefill` primes the
//! per-layer `ZayaDecodeState` (KV cache + conv ring + delayed value); each
//! `decode_step` advances one token in O(1), leaving the last-position logits in
//! `self.logits` for the daemon sampler.

use crate::gpu::{gpu_decode, gpu_forward_serve, ZayaDecodeState, ZayaGpuWeights};
use crate::ZayaConfig;
use hipfire_model::ARCH_ID_ZAYA;
use hipfire_runtime::arch::{
    run_simple_ar, ArchCaps, GenerateCtx, ServeOutcome, ServingBackend, SimpleAr,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};

/// A loaded ZAYA1 model with GPU-resident weights and per-layer decode state
/// (KV cache + conv ring + delayed value), so decode is O(1) per token.
pub struct ZayaModel {
    weights: ZayaGpuWeights,
    cfg: ZayaConfig,
    state: ZayaDecodeState,
    logits: GpuTensor,
}

impl ZayaModel {
    /// Load weights from an HFQ onto the GPU and allocate the decode state +
    /// logits buffer. `max_seq` bounds the KV cache.
    pub fn from_hfq(
        gpu: &mut Gpu,
        hfq: &HfqFile,
        cfg: ZayaConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let weights = ZayaGpuWeights::load(hfq, gpu, &cfg)?;
        let state = ZayaDecodeState::new(gpu, &cfg, max_seq)?;
        let logits = gpu
            .zeros(&[cfg.vocab_size], DType::F32)
            .map_err(|e| format!("zaya logits alloc: {e:?}"))?;
        Ok(Self {
            weights,
            cfg,
            state,
            logits,
        })
    }

    pub fn config(&self) -> &ZayaConfig {
        &self.cfg
    }

    /// GPU-resident weights — exposed for the daemon calibration seam
    /// ([`hipfire_runtime::calibration::CalibratableBackend`]), which runs the
    /// capturing forward over the already-loaded weights (no second load).
    pub fn weights(&self) -> &ZayaGpuWeights {
        &self.weights
    }
}

impl SimpleAr for ZayaModel {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("zaya prefill: empty prompt".to_string());
        }
        gpu_forward_serve(
            gpu,
            &self.weights,
            &self.cfg,
            tokens,
            &mut self.state,
            &self.logits,
        )
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, _pos: usize) -> Result<(), String> {
        gpu_decode(
            gpu,
            &self.weights,
            &self.cfg,
            token,
            &mut self.state,
            &self.logits,
        )
    }

    fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
}

impl ServingBackend for ZayaModel {
    fn arch_id(&self) -> u32 {
        ARCH_ID_ZAYA
    }

    fn caps(&self) -> ArchCaps {
        ArchCaps::default()
    }

    fn eos_token(&self) -> u32 {
        self.cfg.eos_token_id
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let eos = self.cfg.eos_token_id;
        run_simple_ar(gpu, self, tok, eos, ctx)
    }

    fn reset_session(&mut self, _gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let m = *self;
        m.weights.free(gpu);
        m.state.free(gpu);
        let _ = gpu.free_tensor(m.logits);
    }
}
