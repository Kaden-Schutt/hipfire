// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ArchModel` implementations for DeepSeek V4 bundles.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::heterogeneous::DeepseekV4HeterogeneousModel;
use crate::spec_impl::Deepseek4Bundle;

// SAFETY: DeepseekV4HeterogeneousModel owns two `Gpu` handles (dense gfx1100
// + routed gfx1151). `Gpu` itself contains non-Send interior (AqlQueue with
// NonNull) and therefore is !Send by default. The model is accessed only as
// `&mut Self` on the single daemon thread that owns it, and its Drop
// explicitly binds the correct device before freeing. Moving the whole model
// between threads as `Box<dyn ArchModel>` is safe as long as no concurrent
// access occurs, which the `ArchModel: Send` contract guarantees via
// exclusive ownership. This mirrors the safety argument for HipRuntime
// itself (`hip-bridge/src/ffi.rs:328` — Send+Sync because HIP runtime is
// thread-safe for API calls).
unsafe impl Send for DeepseekV4HeterogeneousModel {}

// SAFETY: Deepseek4Bundle owns GpuTensors and DeepseekV4State which contains
// PrefillBatchScratch/ReplayController that is !Send due to NonNull queue
// pointers. As with the heterogeneous model, the bundle is always accessed
// as `&mut Self` on the single owning thread and only moved as
// `Box<dyn ArchModel>` with exclusive ownership, so cross-thread movement is
// safe.
unsafe impl Send for Deepseek4Bundle {}

/// ArchModel for the single-GPU DeepSeek V4 bundle.
///
/// Config mapping: `hidden_size` -> dim(), `num_hidden_layers` -> n_layers().
impl ArchModel for Deepseek4Bundle {
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
        "deepseek4"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Mirrors SpecTarget::reset_recurrent for Deepseek4Bundle:
        //   spec_impl.rs:706-713  self.state.reset(); self.state.zero_decode_caches(gpu);
        // This clears n_tokens + mtp_last_hidden + ar_forward_warmed_up and
        // zero-fills every position-indexed SWA / compressed-KV / indexer
        // cache so the next conversation cannot bleed prior-turn residue.
        // Pairs with daemon's gpu.invalidate_graph_state() after this call.
        self.state.reset();
        self.state.zero_decode_caches(gpu);
        Ok(())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        // Mirrors unload_model's former Deepseek4 arm plus the separate pbs free:
        //   state -> weights -> pbs. Config/eos are plain drops. pbs is now owned
        //   by the bundle (moved from LoadedModel) so it must be freed here.
        let Deepseek4Bundle {
            state,
            weights,
            pbs,
            ..
        } = *self;
        state.free_gpu(gpu);
        weights.free_gpu(gpu);
        if let Some(pbs) = pbs {
            pbs.free_gpu(gpu);
        }
    }
}
