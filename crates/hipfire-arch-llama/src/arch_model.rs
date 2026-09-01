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

    fn validate_teardown_device(&self, device_id: i32) -> Result<(), String> {
        if let Some(store) = self.weight_store.as_ref() {
            store
                .validate_device(device_id)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let mut bundle = *self;
        // Preflight in `unload_model` already validated the device. If drain
        // still fails (e.g., hip free error), keep fail-closed quarantine
        // semantics but do not claim the failure is retryable.
        if let Some(store) = bundle.weight_store.take() {
            if let Err((_, error)) = store.drain(gpu) {
                eprintln!("llama: failed to release attached weight store: {error}");
                std::mem::forget(bundle);
                return;
            }
        }
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
        } = bundle;
        drop(weight_store);
        scratch.free_gpu(gpu);
        weights.free_gpu(gpu);
        let _ = kv.free_gpu(gpu);
    }
}
