// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::spec_impl::DotsOcrBundle;

impl ArchModel for DotsOcrBundle {

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn dim(&self) -> usize {
        self.config.text.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.text.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "dots_ocr"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        // Mirrors the previous unload_model arm for dots_ocr_bundle verbatim:
        //   b.weights.free_gpu(gpu);
        //   b.state.free_gpu(gpu);
        // DotsOcrWeights::free_gpu frees both text and vision halves.
        // Order (weights then state) matches the old manual sequence.
        let DotsOcrBundle { config: _, weights, state } = *self;
        weights.free_gpu(gpu);
        state.free_gpu(gpu);
    }
}
