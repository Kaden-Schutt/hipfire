// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::carrier::LlamaBundle;

impl ArchModel for LlamaBundle {
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
            manifest_plan: _,
            weight_store,
            weight_origin: _,
            mesh: _,
            dflash_extract_layers: _,
            dspark_weights: _,
            dspark_assets: _,
        } = *self;
        // Mirror the existing unload ordering: scratch → store/weights → kv.
        scratch.free_gpu(gpu);
        if let Some(store) = weight_store {
            // Attachment already checked the complete origin and created this
            // owner capability. There is no mismatch branch to leak the model:
            // an attached store can only be drained by this consuming owner.
            if let Err(error) = store.drain(gpu) {
                eprintln!("llama: failed to release attached weight store: {error}");
            }
        }
        weights.free_gpu(gpu);
        let _ = kv.free_gpu(gpu);
    }
}
