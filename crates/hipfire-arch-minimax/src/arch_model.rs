// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::spec_impl::MiniMaxBundle;

impl ArchModel for MiniMaxBundle {
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
        "minimax"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        Some(&mut self.state.kv)
    }

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let MiniMaxBundle {
            config: _,
            weights,
            state,
            eos_tok: _,
        } = *self;
        state.free_gpu(gpu);
        weights.free_gpu(gpu);
    }
}
