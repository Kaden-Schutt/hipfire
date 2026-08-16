// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::carrier::LlamaBundle;

impl ArchModel for LlamaBundle {

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn dim(&self) -> usize {
        self.config.dim
    }

    fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "llama"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        Some(&mut self.kv)
    }

    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.kv.compact_offset = 0;
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let LlamaBundle {
            config: _,
            weights,
            scratch,
            kv,
            dflash_extract_layers: _,
            dspark_weights: _,
            dspark_assets: _,
        } = *self;
        // Mirror unload_model ModelState::Llama arm exactly (lib.rs:3041):
        //   b.scratch.free_gpu(gpu);
        //   b.weights.free_gpu(gpu);
        //   note(b.kv.free_gpu(gpu)…)
        // Ordering matters: scratch → weights → kv. dspark sidecars (when
        // present) are reclaimed via the speculator/spec scratch paths, not
        // here — matching the current unload_model which also does not handle
        // them in this arm.
        scratch.free_gpu(gpu);
        weights.free_gpu(gpu);
        let _ = kv.free_gpu(gpu);
    }
}
