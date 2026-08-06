// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 impl of the arch-generic `hipfire_runtime::spec::SpecTarget`.
//!
//! [`Deepseek4Bundle`] owns the model pieces the daemon + MTP drafter need
//! (config + weights + recurrent state + eos) so deepseek4 can be borrowed as a
//! `&mut dyn SpecTarget` exactly like the qwen35 `ModelSlot` — the prerequisite
//! for routing it through the unified spec loop. The MTP draft+verify itself is
//! the [`crate::spec_decode`] fused step, reached by downcasting this bundle in
//! the deepseek4 `MtpDrafter` impl; deepseek4 never pairs with the model-free
//! n-gram drafter, so the n-gram-verify primitives are intentional error stubs.
//!
//! The four DSpark-specific `SpecTarget` hooks (`new_spec_scratch`,
//! `verify_block`, `commit_prefix`, `capture_seed_main_hidden`) ARE
//! implemented here so the generic `DsparkDrafter` in `dspark_core` can
//! route verify + bootstrap through the trait without downcasting — the
//! byte-identical gate depends on these hitting the same kernel paths as
//! the old inline `Deepseek4DsparkDrafter`.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::{
    self, dspark_assemble_main_hidden, final_norm_and_argmax_all_batched,
    final_norm_and_argmax_all_batched_lazy, final_norm_and_sample_all_batched_lazy,
    forward_prefill_batch_chunk, PrefillBatchScratch,
};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::{Gpu, GpuTensor};

/// Owned deepseek4 model state — the future `ModelState::Deepseek4` payload and
/// the spec-decode target. Bundles config + weights + recurrent state + eos so
/// the daemon can borrow it as `&mut dyn SpecTarget`.
pub struct Deepseek4Bundle {
    pub config: DeepseekV4Config,
    pub weights: DeepseekV4Weights,
    pub state: DeepseekV4State,
    pub eos_tok: u32,
    /// Single-GPU chunked-prefill scratch, allocated once at load. (Relocated
    /// here from a loose LoadedModel field — god-struct collapse Increment 2.)
    pub pbs: crate::forward::PrefillBatchScratch,
}

/// Thin verify scratch for the DSpark `DsparkDrafter` path. DeepSeek V4's SWA
/// attention is stateless (no recurrent rewind needed between verify and
/// commit_prefix), so the scratch carries no GPU buffers — the PBS lives in
/// `state.dspark_verify_pbs` and is reused across windows.
pub struct Deepseek4DsparkScratch;

impl SpecScratch for Deepseek4DsparkScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn free(self: Box<Self>, _gpu: &mut Gpu) {
        // No GPU buffers owned by this scratch.
    }
}

/// Max batch for the trunk-side verify PBS (bootstrap 1-token + verify up to
/// block+1 tokens). Mirror of `Deepseek4DsparkDrafter::pbs_max_batch`.
fn dspark_verify_pbs_max_batch() -> usize {
    hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)
}

impl Deepseek4Bundle {
    /// Shared verify forward for `verify_block` / `verify_block_capture_gpu`:
    /// arms hidden capture into `state.dspark_caps`, runs one batched trunk
    /// forward over `block` at `position`, and leaves the head unapplied. When
    /// `refresh_layers`, re-reads the extract-layer ids from the sidecar (needed
    /// on steady-state windows that never called `capture_seed_main_hidden`).
    fn dspark_verify_forward(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        refresh_layers: bool,
    ) -> Result<(), String> {
        let n_verify = block.len();
        {
            let pbs_ref = self.state.dspark_verify_pbs.as_ref().ok_or(
                "Deepseek4Bundle::verify_block: dspark_verify_pbs not allocated (call new_spec_scratch first)",
            )?;
            if pbs_ref.max_batch < n_verify {
                return Err(format!(
                    "Deepseek4Bundle::verify_block: PBS max_batch ({}) < block len ({})",
                    pbs_ref.max_batch, n_verify
                ));
            }
        }
        if refresh_layers {
            if let Some(ref dspark) = self.weights.dspark {
                self.state.dspark_target_layers = dspark.cfg.target_layer_ids.clone();
            }
        }
        self.state.dspark_capture_active = true;
        // Take the PBS out of state to avoid immutable + mutable borrow collision.
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let fwd_result = forward_prefill_batch_chunk(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            &pbs,
            block,
            position as u32,
        );
        // Restore the PBS before propagating any error so state stays consistent.
        self.state.dspark_verify_pbs = Some(pbs);
        fwd_result.map_err(|e| format!("Deepseek4Bundle::verify_block forward: {e}"))
    }

    /// Apply the trunk final-norm + lm_head + per-position argmax over the
    /// `n_verify` hidden rows left in the verify PBS by `dspark_verify_forward`.
    fn dspark_verify_argmax(&mut self, gpu: &mut Gpu, n_verify: usize) -> Result<Vec<u32>, String> {
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let argmax_result = final_norm_and_argmax_all_batched(
            &self.config,
            &self.weights,
            &mut self.state,
            &pbs,
            gpu,
            n_verify,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        argmax_result.map_err(|e| format!("Deepseek4Bundle::verify_block head+argmax: {e}"))
    }

    /// LAZY twin of `dspark_verify_argmax`: greedy argmax per position with a
    /// prefix stop against the drafted `block` (skips heads after the first
    /// mismatch). Byte-identical committed output, fewer lm_head GEMVs.
    fn dspark_verify_argmax_lazy(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
    ) -> Result<Vec<u32>, String> {
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let res = final_norm_and_argmax_all_batched_lazy(
            &self.config,
            &self.weights,
            &mut self.state,
            &pbs,
            gpu,
            block,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        res.map_err(|e| format!("Deepseek4Bundle::verify_block head+argmax(lazy): {e}"))
    }

    /// temp>0 twin of `dspark_verify_argmax`: fused GPU sample per position with
    /// LAZY prefix stop against the drafted `block` (samples ~τ heads/window, not
    /// all n). Advances `rng_state`.
    fn dspark_verify_sample_lazy(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        temp: f32,
        top_p: f32,
        top_k: usize,
        cactus_delta: f32,
        rng_state: &mut u64,
    ) -> Result<Vec<u32>, String> {
        let result_buf = gpu
            .alloc_tensor(&[2], rdna_compute::DType::F32)
            .map_err(|e| format!("dspark_verify_sample_lazy result_buf: {e:?}"))?;
        let repeat_buf = gpu
            .alloc_tensor(&[1], rdna_compute::DType::F32)
            .map_err(|e| format!("dspark_verify_sample_lazy repeat_buf: {e:?}"))?;
        let mut rng32 = *rng_state as u32;
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let res = final_norm_and_sample_all_batched_lazy(
            &self.config,
            &self.weights,
            &mut self.state,
            &pbs,
            gpu,
            block,
            temp,
            top_p,
            top_k,
            cactus_delta,
            &mut rng32,
            &result_buf,
            &repeat_buf,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        *rng_state = rng32 as u64;
        let _ = gpu.free_tensor(result_buf);
        let _ = gpu.free_tensor(repeat_buf);
        res.map_err(|e| format!("Deepseek4Bundle::verify_block head+sample: {e}"))
    }
}

impl SpecTarget for Deepseek4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Host counters + MTP residual + graph-warmup flag, then zero every
        // position-indexed SWA/full/compressed/indexer cache so a fresh
        // conversation cannot bleed prior-turn residue (pairs with the
        // daemon's `gpu.invalidate_graph_state()` after this hook).
        self.state.reset();
        self.state.zero_decode_caches(gpu);
        Ok(())
    }

    fn retry_reset_eligible(&self) -> bool {
        // reset() + zero_decode_caches; daemon pairs invalidate_graph_state.
        true
    }
    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.config.max_position_embeddings
    }

    // ── n-gram-verify primitives (intentionally unsupported) ────────────────
    // deepseek4's MTP drafter downcasts this bundle and runs `spec_decode` —
    // those paths never hit these hooks. The DSpark drafter DOES use
    // `new_spec_scratch` / `verify_block` / `commit_prefix`; see below.

    /// Advance the trunk over `tokens` from `start_pos`, returning the greedy
    /// argmax at the last position. Used by `DsparkDrafter::mtp_prefill` to
    /// run the prompt through the trunk in a single pass.
    ///
    /// `reset` is always `false` here — the central model reset has already run
    /// on a cache miss. The abort callback is checked between prefill chunks and
    /// before/after the final head pass.
    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.reset_recurrent(gpu)
                .map_err(|e| format!("deepseek4 spec_advance reset: {e}"))?;
        }
        // Lazily allocate the trunk-sized PBS.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| format!("Deepseek4Bundle::spec_advance: alloc PBS: {e}"))?,
            );
        }
        // Take the PBS out of state to avoid a simultaneous immutable + mutable
        // borrow of self.state (the chunked forward path takes &mut state).
        // Restore it afterward (it is always Some after the lazy alloc above).
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let mut pos_cursor = start_pos;
        let mut remaining = tokens;
        let last_logits = loop {
            if abort() {
                self.state.dspark_verify_pbs = Some(pbs);
                return Ok(SpecAdvance::Aborted);
            }
            let take = remaining.len().min(pbs.max_batch);
            let chunk = &remaining[..take];
            let chunk_result = forward::forward_prefill_batch_chunk(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            &pbs,
                chunk,
                pos_cursor as u32,
        );
            if let Err(e) = chunk_result {
        self.state.dspark_verify_pbs = Some(pbs);
                return Err(format!("Deepseek4Bundle::spec_advance prefill: {e}"));
            }
            if take == remaining.len() {
                if abort() {
                    self.state.dspark_verify_pbs = Some(pbs);
                    return Ok(SpecAdvance::Aborted);
                }
                let logits_result = forward::final_norm_and_head_last_batched(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    &pbs,
                    gpu,
                    take,
                );
                let logits = match logits_result {
                    Ok(logits) => logits,
                    Err(e) => {
                        self.state.dspark_verify_pbs = Some(pbs);
                        return Err(format!("Deepseek4Bundle::spec_advance head: {e}"));
                    }
                };
                if abort() {
                    self.state.dspark_verify_pbs = Some(pbs);
                    return Ok(SpecAdvance::Aborted);
                }
                break logits;
            }
            pos_cursor += take;
            remaining = &remaining[take..];
        };
        let last_argmax = crate::spec_decode::logits_argmax(&last_logits) as u32;
        self.state.dspark_verify_pbs = Some(pbs);
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: Some(last_logits),
        })
    }

    // ── DSpark verify primitives ──────────────────────────────────────────
    //
    // The generic `DsparkDrafter` in `dspark_core` calls these three methods
    // to verify draft tokens against the trunk. They route to the IDENTICAL
    // kernel paths the old inline `Deepseek4DsparkDrafter` used —
    // `forward_prefill_batch_chunk` + `final_norm_and_argmax_all_batched` —
    // so the byte-identical gate passes without any numeric change.

    /// Allocate the thin DSpark verify scratch. The PBS lives in
    /// `state.dspark_verify_pbs` (lazily allocated here on first call);
    /// `Deepseek4DsparkScratch` itself carries no GPU buffers.
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // Lazily allocate the trunk-sized verify PBS if not yet present.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| format!("Deepseek4Bundle: alloc dspark_verify_pbs: {e}"))?,
            );
        }
        Ok(Box::new(Deepseek4DsparkScratch))
    }

    /// Run the trunk forward over `block` at absolute `position`, returning
    /// per-slot target argmaxes. Mirrors `Deepseek4DsparkDrafter::mtp_step`
    /// steps 3–4 exactly: capture armed, `forward_prefill_batch_chunk` then
    /// `final_norm_and_argmax_all_batched`.
    ///
    /// **Stage 3 hidden_out capture:** when `hidden_out` is `Some`, downloads
    /// the per-position captured main-hidden from `state.dspark_caps` and writes
    /// `n * n_targets * hidden` floats into it (row-major, one `n_targets * hidden`
    /// row per verified position). This is the multi-slot context the generic
    /// `DsparkDrafter` uses to skip bootstrap in steady-state windows.
    /// `dspark_target_layers` is set from the DSpark sidecar config before each
    /// capture so it remains valid even after the initial bootstrap.
    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let n_verify = block.len();
        // Arm capture + run the trunk forward. refresh_layers when hidden_out is
        // Some (steady-state windows re-read the sidecar extract layers).
        self.dspark_verify_forward(gpu, block, position, hidden_out.is_some())?;

        // ── Stage 3: download captured hidden for multi-slot context update ──
        // dspark_caps layout: [max_batch, n_targets, hidden] flat. Positions
        // 0..n_verify are contiguous at offset 0, so a single d2h suffices.
        if let Some(out) = hidden_out {
            let n_targets = self.state.dspark_target_layers.len();
            let hidden = self.config.hidden_size;
            if n_targets > 0 {
                let n_floats = n_verify * n_targets * hidden;
                let mut raw = vec![0.0f32; n_floats];
                if let Some(caps) = self.state.dspark_caps.as_ref() {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, n_floats * 4)
                    };
                    gpu.hip
                        .memcpy_dtoh(bytes, &caps.buf)
                        .map_err(|e| format!("Deepseek4Bundle::verify_block caps d2h: {e:?}"))?;
                }
                *out = raw;
            }
        }

        self.dspark_verify_argmax(gpu, n_verify)
    }

    /// GPU-resident variant of [`verify_block`]: captures the accepted-prefix
    /// hidden straight into the caller-owned `hidden_gpu` (GPU→GPU) instead of
    /// downloading it to a host `Vec` and re-uploading. deepseek4's batched
    /// forward captures every position, so `captured` is always true once the
    /// drafter has extract layers configured.
    fn verify_block_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let n_verify = block.len();
        self.dspark_verify_forward(gpu, block, position, true)?;

        let n_targets = self.state.dspark_target_layers.len();
        let hidden = self.config.hidden_size;
        let captured = n_targets > 0;
        if captured {
            let n_floats = n_verify * n_targets * hidden;
            if let Some(caps) = self.state.dspark_caps.as_ref() {
                gpu.memcpy_dtod_auto(&hidden_gpu.buf, &caps.buf, n_floats * 4)
                    .map_err(|e| {
                        format!("Deepseek4Bundle::verify_block_capture_gpu caps dtod: {e:?}")
                    })?;
            }
        }

        let picks = self.dspark_verify_argmax_lazy(gpu, block)?;
        Ok((picks, captured))
    }

    /// temp>0 twin of [`verify_block_capture_gpu`]: same batched forward + GPU
    /// hidden capture, but draws each position with the fused GPU sampler and
    /// LAZY prefix stop (distribution-identical to AR temp-T decoding).
    #[allow(clippy::too_many_arguments)]
    fn verify_block_sampled_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        temp: f32,
        top_p: f32,
        top_k: usize,
        cactus_delta: f32,
        rng_state: &mut u64,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let n_verify = block.len();
        self.dspark_verify_forward(gpu, block, position, true)?;

        let n_targets = self.state.dspark_target_layers.len();
        let hidden = self.config.hidden_size;
        let captured = n_targets > 0;
        if captured {
            let n_floats = n_verify * n_targets * hidden;
            if let Some(caps) = self.state.dspark_caps.as_ref() {
                gpu.memcpy_dtod_auto(&hidden_gpu.buf, &caps.buf, n_floats * 4)
                    .map_err(|e| {
                        format!(
                            "Deepseek4Bundle::verify_block_sampled_capture_gpu caps dtod: {e:?}"
                        )
                    })?;
            }
        }

        let picks = self.dspark_verify_sample_lazy(
            gpu,
            block,
            temp,
            top_p,
            top_k,
            cactus_delta,
            rng_state,
        )?;
        Ok((picks, captured))
    }

    /// Advance `state.n_tokens` to reflect the committed prefix. DeepSeek
    /// V4's SWA attention is stateless so no recurrent rewind is needed;
    /// the next verify forward simply overwrites the rejected tail slots.
    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        accept_len: usize,
        position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Mirrors the old inline drafter:
        // `bundle.state.n_tokens = (position + committed.len()) as u64`
        // where `committed.len() = accept_len + 1` (accepted drafts + bonus).
        self.state.n_tokens = (position + accept_len + 1) as u64;
        Ok(())
    }

    // ── DSpark bootstrap primitive ─────────────────────────────────────────

    /// Run a 1-token trunk forward with capture armed at `layers`, assemble
    /// the concatenated `[layers.len() * hidden]` main-hidden vector, and
    /// return it as a host-side `Vec<f32>`. Mirrors the bootstrap step of
    /// the old `Deepseek4DsparkDrafter::mtp_step` (steps 1a–1c) exactly.
    fn capture_seed_main_hidden(
        &mut self,
        gpu: &mut Gpu,
        seed: u32,
        position: usize,
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        // Lazily allocate the trunk-sized verify PBS if not yet present.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| {
                        format!("Deepseek4Bundle: alloc dspark_verify_pbs (bootstrap): {e}")
                    })?,
            );
        }

        self.state.dspark_target_layers = layers.to_vec();
        self.state.dspark_capture_active = true;
        // Take the PBS out of state to avoid immutable+mutable borrow conflict.
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let fwd_result = forward_prefill_batch_chunk(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            &pbs,
            &[seed],
            position as u32,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        fwd_result
            .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden forward: {e}"))?;

        dspark_assemble_main_hidden(&mut self.state, gpu, &self.config, 0)
            .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden assemble: {e}"))?;

        let n = layers.len() * self.config.hidden_size;
        let mut host = vec![0.0f32; n];
        {
            let main_hidden = self
                .state
                .dspark_main_hidden
                .as_ref()
                .ok_or("Deepseek4Bundle::capture_seed_main_hidden: dspark_main_hidden is None after assemble")?;
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, n * 4) };
            gpu.hip
                .memcpy_dtoh(bytes, &main_hidden.buf)
                .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden d2h: {e:?}"))?;
        }
        Ok(host)
    }
}
