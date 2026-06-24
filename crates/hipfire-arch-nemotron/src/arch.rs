// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serving seam for nemotron_h (arch_id 14): `SimpleAr` + `ServingBackend` on
//! [`NemotronModel`], routing through the shared `run_simple_ar` → prefill →
//! `decode_loop` loop (the same dense-AR seam as `LlamaBackend`). Mamba-2 /
//! attention / MLP are all decode-only single-token forwards, so prefill is a
//! per-token loop that builds the recurrent + KV state.

use crate::model::NemotronModel;
use hipfire_runtime::arch::{
    run_simple_ar, ArchCaps, GenerateCtx, ServeOutcome, ServingBackend, SimpleAr,
};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};

impl SimpleAr for NemotronModel {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("nemotron prefill: empty prompt".to_string());
        }
        // Fresh sequence: zero the recurrent (conv/SSM) state.
        self.reset(gpu)
            .map_err(|e| format!("nemotron reset: {e:?}"))?;
        if self.can_batched_prefill() {
            // N6 batched prefill (f32): one launch per recurrent kernel instead
            // of per token. Leaves the last token's logits in `self.logits`.
            return self
                .prefill_batched(gpu, tokens)
                .map_err(|e| format!("nemotron batched prefill: {e:?}"));
        }
        // Quantized models: per-token loop until the batched quant gemm lands.
        for (pos, &t) in tokens.iter().enumerate() {
            self.forward_gpu(gpu, t, pos)
                .map_err(|e| format!("nemotron prefill forward[{pos}]: {e:?}"))?;
        }
        Ok(())
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        self.forward_gpu(gpu, token, pos)
            .map_err(|e| format!("nemotron decode forward[{pos}]: {e:?}"))
    }

    fn logits(&self) -> &GpuTensor {
        self.logits_tensor()
    }

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
}

impl ServingBackend for NemotronModel {
    fn arch_id(&self) -> u32 {
        hipfire_model::ARCH_ID_NEMOTRON_H
    }

    fn caps(&self) -> ArchCaps {
        // Dense-AR bring-up: no DFlash/MTP/drafter/vision fast paths.
        ArchCaps::default()
    }

    fn eos_token(&self) -> u32 {
        self.config().eos_token_id
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let eos = self.config().eos_token_id;
        run_simple_ar(gpu, self, tok, eos, ctx)
    }

    fn reset_session(&mut self, gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        self.reset(gpu)
            .map_err(|e| format!("nemotron reset_session: {e:?}"))
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        (*self).free(gpu);
    }
}
