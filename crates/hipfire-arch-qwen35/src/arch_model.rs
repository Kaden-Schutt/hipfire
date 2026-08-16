// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ArchModel` implementation for `Qwen35Bundle`.

use crate::carrier::Qwen35Bundle;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

impl ArchModel for Qwen35Bundle {
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
        "qwen35"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        Some(&mut self.kv_cache)
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Mirrors speculative::ModelSlot::reset_state and
        // spec_impl::SpecTarget::reset_recurrent for qwen35:
        //   dn_state.reset(gpu)?  (zeroes s_matrices / s_scales / conv_states / s_ef_residual,
        //                          stream-aware) + kv compact_offset rewind.
        self.dn_state
            .reset(gpu)
            .map_err(|e| format!("qwen35 reset_session_state: {e}"))?;
        self.kv_cache.compact_offset = 0;
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        // Mirrors unload_model single-GPU arm for ModelState::Qwen35 in
        // crates/hipfire-loader/src/lib.rs (lines ~3035-3040):
        //   note(b.kv_cache.free_gpu(gpu));
        //   b.scratch.free_gpu(gpu);
        //   b.weights.free_gpu(gpu);
        //   b.dn_state.free_gpu(gpu);
        // Order is kv → scratch → weights → dn. Errors are aggregated via
        // `note` in the loader but `ArchModel::free_gpu` is infallible, so
        // failures here are dropped (same as weight/dn frees which already
        // ignore errors).
        let Qwen35Bundle {
            config: _,
            weights,
            scratch,
            kv_cache,
            dn_state,
            kv_adaptive: _,
        } = *self;
        let _ = kv_cache.free_gpu(gpu);
        let _ = scratch.free_gpu(gpu);
        weights.free_gpu(gpu);
        dn_state.free_gpu(gpu);
    }
}
