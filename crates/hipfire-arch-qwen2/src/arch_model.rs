// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::carrier::Qwen2Bundle;

impl ArchModel for Qwen2Bundle {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "qwen2"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let Qwen2Bundle { config: _, weights, state } = *self;
        // Mirror unload_model ModelState::Qwen2 arm:
        //   b.state.free_gpu(gpu);
        //   b.weights.free_gpu(gpu);
        state.free_gpu(gpu);
        weights.free_gpu(gpu);
    }
}
