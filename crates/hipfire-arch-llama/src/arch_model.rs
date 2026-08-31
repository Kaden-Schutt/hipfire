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
            mesh,
            dflash_extract_layers: _,
            dspark_weights: _,
            dspark_assets: _,
        } = *self;
        // Mirror the existing unload ordering: scratch → store/weights → kv.
        // A committed store is only released here, through the ArchModel owner;
        // no store destructor or independent carrier free path exists.
        scratch.free_gpu(gpu);
        if let Some(store) = weight_store {
            if let Err((_, error)) = store.release_on_owner(&mesh, gpu) {
                eprintln!("llama: refusing weight-store release: {error}");
            }
        }
        weights.free_gpu(gpu);
        let _ = kv.free_gpu(gpu);
    }
}
