// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA-family implementation of the arch-generic speculative-decode seam
//! (`hipfire_specdecode_dspark::spec`).
//!
//! `impl SpecTarget for LlamaBackend` lets the model-free speculators
//! (`NgramSpeculator`, the DSpark/DFlash drafter) drive a dense-attention target
//! with no arch knowledge. Pure attention makes this the *cheap* spec case:
//! `verify_block` runs ONE block-parallel batched forward
//! (`llama_spec::verify_block_argmax`), there is no recurrent state to snapshot,
//! and `commit_prefix` is a no-op — the accepted-prefix KV the verify wrote is
//! already correct and the rejected tail is overwritten by the next verify.
//!
//! The verify/advance/capture mechanics all live in the Phase-2 runtime helpers
//! (`hipfire_runtime::llama_spec::*` + `llama::forward_scratch_compute_capture`);
//! this file is the thin `SpecTarget` shell wiring `LlamaBackend`'s
//! `weights/config/scratch/kv_cache` into them.

use crate::arch::LlamaBackend;
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::llama::{self, KvCache, PrefillBatchScratch};
use hipfire_specdecode_dspark::spec::{SpecAdvance, SpecScratch, SpecTarget};

/// LLaMA target-verify scratch: just the per-block batched-forward scratch
/// (`PrefillBatchScratch`). No recurrent snapshot — pure attention.
pub struct LlamaSpecScratch {
    pbs: PrefillBatchScratch,
}

impl SpecScratch for LlamaSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        self.pbs.free_gpu(gpu);
    }
}

impl SpecTarget for LlamaBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) {
        // Pure attention: no recurrent state to zero. Drop the KV eviction offset
        // so the next conversation rotates from absolute 0.
        self.kv_cache.compact_offset = 0;
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        let block_size = block_size.max(2);
        // The tree-verify arm linearizes a DDTree of up to `dense_tree_verify_nodes`
        // slots into ONE batched forward, so the verify scratch must hold the larger
        // of the chain block and the (clamped) tree size — else a big
        // `HIPFIRE_DDTREE_BUDGET` overflows `pbs.max_batch` and panics.
        let tree_nodes = hipfire_runtime::llama_spec::dense_tree_verify_nodes(
            true,
            Some(hipfire_runtime::config::get().ddtree_budget),
        );
        let max_batch = block_size.max(tree_nodes);
        let pbs =
            PrefillBatchScratch::new(gpu, &self.config, max_batch, self.kv_cache.physical_cap)
                .map_err(|e| format!("LlamaSpecScratch PrefillBatchScratch: {e:?}"))?;
        Ok(Box::new(LlamaSpecScratch { pbs }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        // Pure attention: "reset" just rewinds the eviction offset; the prefill
        // forward overwrites KV at the absolute positions it writes.
        if reset {
            self.kv_cache.compact_offset = 0;
        }
        // DFlash hidden capture: only when the drafter configured extract layers
        // AND the caller passed a sink. The two together form the gate; either
        // missing → no capture.
        let extract = self.dflash_extract_layers.clone();
        let want_capture = !extract.is_empty() && hidden_out.is_some();
        // GREEDY-EQUIVALENCE: prefill the prompt token-by-token through the SAME
        // decode kernel AR uses (`forward_scratch_compute`), not the batched
        // prefill kernel. The batched and per-token forwards are not bitwise
        // equal; on a near-tie logit the batched path's KV can flip the verifier's
        // argmax off AR's greedy pick. Matching AR's per-token prefill restores
        // token-identical greedy spec decode. Per-token capture appends one
        // position's `num_extract × dim` residual per call (extract order).
        for (i, &tok) in tokens.iter().enumerate() {
            if abort() {
                self.kv_cache.compact_offset = 0;
                return Ok(SpecAdvance::Aborted);
            }
            let pos = start_pos + i;
            let mut sink = if want_capture {
                Some(llama::HiddenCaptureSink {
                    extract_layers: &extract,
                    hidden: hidden_out.as_deref_mut().unwrap(),
                    hidden_gpu: None,
                })
            } else {
                None
            };
            llama::forward_scratch_embed(gpu, &self.weights, &self.config, tok, pos, &self.scratch)
                .map_err(|e| format!("{e:?}"))?;
            llama::forward_scratch_compute_capture(
                gpu,
                &self.weights,
                &self.config,
                pos,
                &mut self.kv_cache,
                &self.scratch,
                sink.as_mut(),
            )
            .map_err(|e| format!("{e:?}"))?;
        }
        // The last per-token forward_scratch_compute leaves the final token's
        // logits in scratch.logits.
        let logits = gpu
            .download_f32(&self.scratch.logits)
            .map_err(|e| format!("{e:?}"))?;
        Ok(SpecAdvance::Ready {
            last_argmax: hipfire_runtime::sampler::argmax(&logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<LlamaSpecScratch>()
            .ok_or("verify_block: scratch is not LlamaSpecScratch")?;
        let extract = &self.dflash_extract_layers;
        let mut sink = match hidden_out {
            Some(h) if !extract.is_empty() => Some(llama::HiddenCaptureSink {
                extract_layers: extract,
                hidden: h,
                hidden_gpu: None,
            }),
            _ => None,
        };
        hipfire_runtime::llama_spec::verify_block_argmax(
            gpu,
            &self.weights,
            &self.config,
            block,
            position,
            &mut self.kv_cache,
            &self.scratch,
            &s.pbs,
            sink.as_mut(),
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn verify_block_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let extract = self.dflash_extract_layers.clone();
        let s = scratch
            .as_any_mut()
            .downcast_mut::<LlamaSpecScratch>()
            .ok_or("verify_block_capture_gpu: scratch is not LlamaSpecScratch")?;
        hipfire_runtime::llama_spec::verify_block_argmax_capture_gpu(
            gpu,
            &self.weights,
            &self.config,
            block,
            position,
            &mut self.kv_cache,
            &self.scratch,
            &s.pbs,
            &extract,
            hidden_gpu,
        )
        .map_err(|e| format!("{e:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_block_sampled_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        temp: f32,
        top_p: f32,
        top_k: usize,
        rng_state: &mut u64,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let extract = self.dflash_extract_layers.clone();
        let s = scratch
            .as_any_mut()
            .downcast_mut::<LlamaSpecScratch>()
            .ok_or("verify_block_sampled_capture_gpu: scratch is not LlamaSpecScratch")?;
        hipfire_runtime::llama_spec::verify_block_sampled_capture_gpu(
            gpu,
            &self.weights,
            &self.config,
            block,
            position,
            &mut self.kv_cache,
            &self.scratch,
            &s.pbs,
            &extract,
            hidden_gpu,
            temp,
            top_p,
            top_k,
            rng_state,
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn verify_block_logits(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<LlamaSpecScratch>()
            .ok_or("verify_block_logits: scratch is not LlamaSpecScratch")?;
        let extract = &self.dflash_extract_layers;
        let mut sink = match hidden_out {
            Some(h) if !extract.is_empty() => Some(llama::HiddenCaptureSink {
                extract_layers: extract,
                hidden: h,
                hidden_gpu: None,
            }),
            _ => None,
        };
        hipfire_runtime::llama_spec::verify_block_logits(
            gpu,
            &self.weights,
            &self.config,
            block,
            position,
            &mut self.kv_cache,
            &self.scratch,
            &s.pbs,
            sink.as_mut(),
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn verify_tree_logits(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        mask_block: &[f32],
        depth_positions: &[i32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<LlamaSpecScratch>()
            .ok_or("verify_tree_logits: scratch is not LlamaSpecScratch")?;
        let extract = &self.dflash_extract_layers;
        let mut sink = match hidden_out {
            Some(h) if !extract.is_empty() => Some(llama::HiddenCaptureSink {
                extract_layers: extract,
                hidden: h,
                hidden_gpu: None,
            }),
            _ => None,
        };
        hipfire_runtime::llama_spec::verify_tree_logits(
            gpu,
            &self.weights,
            &self.config,
            tokens,
            mask_block,
            depth_positions,
            position,
            &mut self.kv_cache,
            &self.scratch,
            &s.pbs,
            sink.as_mut(),
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Pure attention: verify's accepted-prefix KV is already correct, and the
        // rejected tail is overwritten by the next verify. Nothing to rewind.
        Ok(())
    }

    /// Apply the target's lm_head to `n` rows of pre-norm residual hidden states
    /// (`hidden_rows`: F32 GpuTensor, shape `[n × dim]`, row-major).
    ///
    /// Returns `n × vocab_size` host f32 logits (not argmax) so callers can do
    /// SWOR or any other sampling over the full distribution. Implemented by
    /// `hipfire_runtime::llama_spec::lm_head_logits_n_rows`, which mirrors the
    /// per-row `rmsnorm + weight_gemv(output)` loop already used in
    /// `verify_block_argmax`, sharing the same scratch buffers.
    fn lm_head_logits(
        &mut self,
        gpu: &mut Gpu,
        hidden_rows: &GpuTensor,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        hipfire_runtime::llama_spec::lm_head_logits_n_rows(
            gpu,
            &self.weights,
            &self.config,
            hidden_rows,
            n,
            &self.scratch,
        )
        .map_err(|e| format!("LlamaBackend::lm_head_logits: {e:?}"))
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token
    }

    fn ctx_capacity(&self) -> usize {
        self.kv_cache.physical_cap
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        Some(&mut self.kv_cache)
    }

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        if self.dflash_extract_layers.is_empty() {
            None
        } else {
            Some(&self.dflash_extract_layers)
        }
    }

    fn set_dflash_extract_layers(&mut self, layers: Vec<usize>) {
        // Delegate to the inherent setter (asserts ascending) so the generic
        // DFlash drafter can configure capture layers without naming the concrete
        // `LlamaBackend` type.
        LlamaBackend::set_dflash_extract_layers(self, layers);
    }

    /// Capture the target's residual hidden states at `layers` for a single
    /// seed token at `position`, returning the concatenated `[layers.len()*dim]` F32.
    ///
    /// Runs a 1-token capture-armed forward via `forward_scratch_embed` +
    /// `forward_scratch_compute_capture`, reusing the same `HiddenCaptureSink`
    /// path as `spec_advance`. The `layers` parameter comes from the generic
    /// drafter's `DsparkConfig::target_layer_ids` and may differ from
    /// `self.dflash_extract_layers` (e.g. when the bundle was not configured as a
    /// DFlash target), so we use the argument directly.
    fn capture_seed_main_hidden(
        &mut self,
        gpu: &mut Gpu,
        seed: u32,
        position: usize,
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        let mut hidden_out: Vec<f32> = Vec::new();
        let mut sink = llama::HiddenCaptureSink {
            extract_layers: layers,
            hidden: &mut hidden_out,
            hidden_gpu: None,
        };
        llama::forward_scratch_embed(
            gpu,
            &self.weights,
            &self.config,
            seed,
            position,
            &self.scratch,
        )
        .map_err(|e| format!("capture_seed_main_hidden embed: {e:?}"))?;
        llama::forward_scratch_compute_capture(
            gpu,
            &self.weights,
            &self.config,
            position,
            &mut self.kv_cache,
            &self.scratch,
            Some(&mut sink),
        )
        .map_err(|e| format!("capture_seed_main_hidden compute: {e:?}"))?;
        // hidden_out should now hold layers.len() * dim floats (one dim-vector per layer).
        Ok(hidden_out)
    }

    fn embed_row(&mut self, gpu: &mut Gpu, token_id: u32) -> Result<Vec<f32>, String> {
        // Look up one embedding row into the per-token scratch `x` (`[dim]` F32),
        // dispatching on the table's storage format, then download to host. Used
        // by the generic DFlash drafter to build its mask-token noise embedding.
        let dim = self.config.dim;
        let dst = self.scratch.x.sub_offset(0, dim);
        use hipfire_runtime::weights::EmbeddingFormat;
        match self.weights.embd_format {
            EmbeddingFormat::Q4K => {
                gpu.embedding_lookup_q4k(&self.weights.token_embd, &dst, token_id, dim)
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&self.weights.token_embd, &dst, token_id, dim)
            }
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&self.weights.token_embd, &dst, token_id, dim)
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&self.weights.token_embd, &dst, token_id, dim)
            }
            EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&self.weights.token_embd, &dst, token_id, dim)
            }
        }
        .map_err(|e| format!("LlamaBackend::embed_row: {e:?}"))?;
        gpu.download_f32(&dst)
            .map_err(|e| format!("LlamaBackend::embed_row download: {e:?}"))
    }
}
