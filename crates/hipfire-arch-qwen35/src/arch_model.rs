// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ArchModel` implementation for `Qwen35Bundle`.

use crate::carrier::Qwen35Bundle;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::gpu_cleanup::{enqueue_cleanup_failure, GpuCleanupFailure, RetainedGpuTensor};
use hipfire_runtime::llama::{KvCache, KvCacheExt};
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
        // Single-GPU unload teardown (unload_model's qwen35 arm reaches
        // this via `state.free_gpu(gpu)`): kv → scratch → weights → dn →
        // vision, with the decode-batch owner freed first.
        //
        // Every domain is freed with a CHECKED free: `kv_cache.free_checked`,
        // `scratch.abort_checked`, `weights.free_gpu_checked` (releases the
        // Frozen MoE resident — the single `SingleFrozenWeightStore` owner,
        // companions included — and the plain `weights.free_gpu` cannot be
        // used here, it hard-refuses a bundle that still carries
        // `moe_resident`), and `dn_state.abort_checked`. Whatever cannot be
        // freed is enqueued to the process retained-owner backlog (retried
        // at the next load/unload boundary) — NEVER dropped while allocated.
        //
        // `pp_scratch_set` is NOT freed here: it is `None` for every
        // single-GPU load (constructed as `None` in `load_bundle`) and for
        // pp>1 the set is freed via `free_gpu_multi(&mut Gpus)` in the
        // `if m.pp > 1` branch of `unload_model` BEFORE this single-GPU
        // `Box::new(state).free_gpu(gpu)` path is ever reached
        // (`if m.pp > 1 { return Ok(()) }` guards it). Double-free is
        // impossible because a bundle that took the pp>1 path never reaches
        // this `&mut Gpu` free; a leak is impossible because the pp>1 path
        // explicitly frees the set via `b.pp_scratch_set`.
        let Qwen35Bundle {
            config: _,
            weights,
            scratch,
            kv_cache,
            dn_state,
            kv_adaptive: _,
            pp_scratch_set,
            vision_config: _,
            vision_weights,
            qwen35_decode_batch,
        } = *self;
        debug_assert!(
            pp_scratch_set.is_none(),
            "Qwen35Bundle::free_gpu: pp_scratch_set must be None on single-GPU free (pp>1 sets are freed via free_gpu_multi)"
        );
        // Drop without freeing: the per-device set requires `&mut Gpus`
        // and this method only has `&mut Gpu`. The debug_assert above
        // guarantees this is `None` for every single-GPU bundle; a
        // pp>1 bundle that incorrectly reaches here would leak, not
        // double-free, and the assert surfaces the bug.
        let _ = pp_scratch_set;
        if let Some(batch) = qwen35_decode_batch {
            let _ = batch.free_gpu(gpu);
        }
        if let Err(failures) = kv_cache.free_checked(gpu) {
            let mut cf = GpuCleanupFailure::empty();
            for (label, tensor) in failures {
                cf.add_retained(RetainedGpuTensor {
                    label,
                    tensor,
                    last_error: "kv free_checked failed".into(),
                });
            }
            enqueue_cleanup_failure(cf);
        }
        if let Err(failures) = scratch.abort_checked(gpu) {
            let mut cf = GpuCleanupFailure::empty();
            for r in failures {
                cf.add_retained(r);
            }
            enqueue_cleanup_failure(cf);
        }
        if let Err(cf) = weights.free_gpu_checked(gpu) {
            enqueue_cleanup_failure(cf);
        }
        if let Err(failures) = dn_state.abort_checked(gpu) {
            let mut cf = GpuCleanupFailure::empty();
            for r in failures {
                cf.add_retained(r);
            }
            enqueue_cleanup_failure(cf);
        }
        if let Some(vw) = vision_weights {
            vw.free_gpu(gpu);
        }
    }
}
