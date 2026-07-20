// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM2.5-MoE (arch_id=11) implementation of the arch-generic
//! speculative-decode seam (`hipfire_runtime::spec`).
//!
//! LFM2.5-MoE is a HYBRID arch: ~18 short-conv (LIV) layers carry a recurrent
//! `[hidden,(K-1)]` rolling conv-state ring buffer each, interleaved with GQA
//! attention layers backed by the shared `llama::KvCache`. A speculative verify
//! over a B-token block advances BOTH: the conv-states roll forward by B, and
//! the attention KV is written at absolute positions `[position..position+B)`.
//!
//! So this is the *recurrent* shape of the seam (template: qwen35's DeltaNet
//! `ModelSlot` impl), NOT the stateless one (qwen2). The hard part is the
//! conv-state: on a PARTIAL accept we must roll the conv-state back to the
//! accepted prefix. We do this exactly like qwen35 snapshots its DeltaNet S/conv
//! state — by copying every conv-state ring buffer device-to-device into a
//! parallel set of snapshot buffers in [`verify_block`] BEFORE the forward
//! advances them, then restoring + replaying the accepted prefix in
//! [`commit_prefix`].
//!
//! The attention KV needs no explicit rewind: `decode_step` takes an absolute
//! `position` and writes KV there, so the accepted-prefix KV the verify wrote is
//! already correct and the rejected-tail KV is overwritten by the replay (and by
//! the next verify window). Only the conv-state — which advances *implicitly* as
//! a side effect of each conv layer and cannot be re-derived from position — must
//! be snapshotted.
//!
//! VERIFY IS SEQUENTIAL per-token (`decode_step` per block token). A batched
//! conv+attention verify kernel does not exist for this arch; the sequential
//! decode is the correct, coherence-bearing baseline (mirrors qwen2's legacy
//! sequential verify and the LFM2 decode hot path itself).

use crate::config::Lfm2MoeConfig;
use crate::forward::{
    decode_step as decode_step_impl, decode_step_device as decode_step_device_impl,
    decode_step_prefill as decode_step_prefill_impl,
    decode_step_speculative_device as decode_step_speculative_device_impl,
    forward_prefill_batch as forward_prefill_batch_impl, lfm2_decode_fusion_enabled,
    run_prepared_decode_layers_and_head, stage_decode_inputs, validate_lfm_retained_fixture,
};
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use crate::redline_plan::{
    DecodeExecutionMode, RetainedArtifactProvenance, RetainedFixtureEvidence,
};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Exact mutable-state image used by the daemon's retained replay oracle.
#[derive(Debug, PartialEq, Eq)]
pub struct Lfm2MoeRedlineSnapshot {
    pub logits: Vec<u8>,
    pub kv: Vec<u8>,
    pub recurrent: Vec<u8>,
    pub h: Vec<u8>,
    pub tmp: Vec<u8>,
    pub conv_bcx: Vec<u8>,
    pub ffn_x_rot: Vec<u8>,
    pub n_tokens: usize,
    pub compact_offset: usize,
}

fn append_device_buffer(
    gpu: &Gpu,
    output: &mut Vec<u8>,
    buffer: &hip_bridge::DeviceBuffer,
) -> Result<(), String> {
    let start = output.len();
    output.resize(start + buffer.size(), 0);
    gpu.hip
        .memcpy_dtoh(&mut output[start..], buffer)
        .map_err(|error| error.to_string())
}

/// Owned LFM2.5-MoE GPU bundle: the local type the arch-generic `SpecTarget`
/// seam is implemented for. Mirrors `hipfire_arch_qwen2::carrier::Qwen2Bundle`
/// (config + weights + state owned together in the arch crate) so the model-free
/// speculator can drive an LFM2 target with no arch knowledge.
///
/// The orphan rule requires `impl SpecTarget` to target a type local to THIS
/// crate, so the spec seam binds to this bundle rather than the loader-side
/// `hipfire_loader::Lfm2MoeBundle`. Integration: the loader's `Lfm2MoeCarrier`
/// should construct this type (or the loader bundle should re-export / wrap it)
/// so the spec-target guard can borrow it as `&mut dyn SpecTarget`.
pub struct Lfm2MoeBundle {
    config: Lfm2MoeConfig,
    weights: Lfm2MoeWeights,
    state: Lfm2MoeState,
    eos_tok: u32,
    retained_fixture_evidence: RetainedFixtureEvidence,
}

/// LFM2.5-MoE verify scratch: the pre-verify conv-state snapshot.
///
/// Owns one F32 GPU buffer per conv layer, each sized to match the corresponding
/// `state.conv_states[i]` (`[hidden*(K-1)]`). [`verify_block`] copies the live
/// conv-states INTO these before the block forward advances them, so a partial
/// accept can restore them in [`commit_prefix`]. The attention KV needs no
/// snapshot (absolute-position writes; see module docs), so nothing else is
/// carried between windows.
pub struct Lfm2MoeSpecScratch {
    /// `conv_snap[i]` is the saved copy of `state.conv_states[i]`.
    conv_snap: Vec<GpuTensor>,
}

impl SpecScratch for Lfm2MoeSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        // GpuTensor has no Drop — free every snapshot buffer explicitly or the
        // device memory orphans (see qwen35 Qwen35SpecScratch::free).
        let Lfm2MoeSpecScratch { conv_snap } = *self;
        for t in conv_snap {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl Lfm2MoeBundle {
    pub fn new(
        config: Lfm2MoeConfig,
        weights: Lfm2MoeWeights,
        state: Lfm2MoeState,
        eos_tok: u32,
        arch_id: u32,
        retained_artifact: RetainedArtifactProvenance,
    ) -> Self {
        let retained_fixture_evidence = RetainedFixtureEvidence::from_validation(
            validate_lfm_retained_fixture(&config, &weights, &state, arch_id).is_ok(),
            retained_artifact,
        );
        Self {
            config,
            weights,
            state,
            eos_tok,
            retained_fixture_evidence,
        }
    }

    #[inline]
    pub fn retained_fixture_evidence(&self) -> bool {
        self.retained_fixture_evidence.is_verified()
    }

    /// Device-side greedy sampling is confined to the same authenticated,
    /// lowered, non-graph exact-fusion route as decode.
    #[inline]
    pub fn device_greedy_sampling_eligible(&self, gpu: &Gpu) -> bool {
        lfm2_decode_fusion_enabled(self.retained_fixture_evidence, &gpu.arch)
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)
    }

    #[inline]
    pub fn n_tokens(&self) -> usize {
        self.state.n_tokens
    }

    #[inline]
    pub fn max_seq(&self) -> usize {
        self.state.max_seq
    }

    #[inline]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_tok
    }

    #[inline]
    pub fn model_dimensions(&self) -> (usize, usize, usize) {
        (
            self.config.hidden_size,
            self.config.num_hidden_layers,
            self.config.vocab_size,
        )
    }

    /// Return the bundle to a byte-deterministic state for retained-route
    /// AQL/PM4/blob/HIP shadow comparisons.
    pub fn redline_reset_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)?;
        self.state
            .kv
            .clear_gpu(gpu)
            .map_err(|error| error.to_string())?;
        self.state.kv.compact_offset = 0;
        self.state.graph_warmed_up = false;

        let state = &self.state;
        let buffers: &[&hip_bridge::DeviceBuffer] = &[
            &state.pos_buf,
            &state.token_buf.buf,
            &state.sample_buf.buf,
            &state.repeat_buf.buf,
            &state.h.buf,
            &state.tmp.buf,
            &state.fa_q.buf,
            &state.fa_k.buf,
            &state.fa_v.buf,
            &state.fa_attn_out.buf,
            &state.flash_partials.buf,
            &state.conv_bcx.buf,
            &state.conv_y.buf,
            &state.ffn_tmp.buf,
            &state.ffn_x_rot.buf,
            &state.dense_gate.buf,
            &state.dense_up.buf,
            &state.dense_act.buf,
            &state.router_logits.buf,
            &state.topk_indices.buf,
            &state.topk_weights.buf,
            &state.gate_batch.buf,
            &state.up_batch.buf,
            &state.rot_batch.buf,
            &state.down_expanded.buf,
            &state.final_norm_buf.buf,
            &state.logits.buf,
        ];
        for buffer in buffers {
            gpu.hip
                .memset(buffer, 0, buffer.size())
                .map_err(|error| error.to_string())?;
        }
        for buffer in [
            gpu.scratch.mq_x_rot.as_ref().map(|tensor| &tensor.buf),
            gpu.scratch.mq_x_rot_fp8.as_ref(),
            gpu.scratch.mq_x_q8.as_ref(),
            gpu.scratch.mq_x_scales.as_ref(),
            gpu.scratch.fp16_x_scratch.as_ref(),
            gpu.scratch.fp8_x_scratch.as_ref(),
            gpu.scratch.q8_1_mmq_x_scratch.as_ref(),
            gpu.scratch.ksplit_det_partials.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            gpu.hip
                .memset(buffer, 0, buffer.size())
                .map_err(|error| error.to_string())?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())
    }

    /// Prime deterministic sequential state outside the retained interval.
    pub fn redline_prime(&mut self, gpu: &mut Gpu, context: usize) -> Result<(), String> {
        if context > self.state.max_seq {
            return Err(format!(
                "lfm2moe: redline prime context {context} exceeds max_seq {}",
                self.state.max_seq
            ));
        }
        for position in 0..context {
            let token_id = 10 + (position as u32 % 1000);
            decode_step_prefill_impl(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                token_id,
                position as u32,
            )?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())
    }

    pub fn redline_snapshot(&self, gpu: &Gpu) -> Result<Lfm2MoeRedlineSnapshot, String> {
        let mut logits = Vec::new();
        append_device_buffer(gpu, &mut logits, &self.state.logits.buf)?;
        let mut kv = Vec::new();
        for tensor in self
            .state
            .kv
            .k_gpu
            .iter()
            .chain(self.state.kv.v_gpu.iter())
            .chain(self.state.kv.k_scales.iter())
            .chain(self.state.kv.v_scales.iter())
        {
            append_device_buffer(gpu, &mut kv, &tensor.buf)?;
        }
        let mut recurrent = Vec::new();
        for tensor in &self.state.conv_states {
            append_device_buffer(gpu, &mut recurrent, &tensor.buf)?;
        }
        let mut h = Vec::new();
        append_device_buffer(gpu, &mut h, &self.state.h.buf)?;
        let mut tmp = Vec::new();
        append_device_buffer(gpu, &mut tmp, &self.state.tmp.buf)?;
        let mut conv_bcx = Vec::new();
        append_device_buffer(gpu, &mut conv_bcx, &self.state.conv_bcx.buf)?;
        let mut ffn_x_rot = Vec::new();
        append_device_buffer(gpu, &mut ffn_x_rot, &self.state.ffn_x_rot.buf)?;
        Ok(Lfm2MoeRedlineSnapshot {
            logits,
            kv,
            recurrent,
            h,
            tmp,
            conv_bcx,
            ffn_x_rot,
            n_tokens: self.state.n_tokens,
            compact_offset: self.state.kv.compact_offset,
        })
    }

    pub fn redline_prepare_decode_inputs(
        &self,
        gpu: &mut Gpu,
        token_id: u32,
        position: u32,
    ) -> Result<(), String> {
        stage_decode_inputs(&self.state, gpu, token_id, position)
    }

    pub fn redline_run_prepared_hip(&mut self, gpu: &mut Gpu, position: u32) -> Result<(), String> {
        if self.state.n_tokens != position as usize {
            return Err(format!(
                "lfm2moe: redline HIP position {position} != cursor {}",
                self.state.n_tokens
            ));
        }
        let fusion_enabled = lfm2_decode_fusion_enabled(self.retained_fixture_evidence, &gpu.arch);
        run_prepared_decode_layers_and_head(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            position,
            true,
            fusion_enabled,
        )?;
        self.state.n_tokens = position as usize + 1;
        Ok(())
    }

    pub fn redline_commit_replayed_position(&mut self, position: u32) -> Result<(), String> {
        if self.state.n_tokens != position as usize {
            return Err(format!(
                "lfm2moe: replay position {position} != cursor {}",
                self.state.n_tokens
            ));
        }
        self.state.n_tokens = position as usize + 1;
        Ok(())
    }

    pub fn decode_step(
        &mut self,
        gpu: &mut Gpu,
        token_id: u32,
        position: u32,
        mode: DecodeExecutionMode,
    ) -> Result<Vec<f32>, String> {
        decode_step_impl(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            token_id,
            position,
            self.retained_fixture_evidence,
            mode,
        )
    }

    pub fn decode_step_device(
        &mut self,
        gpu: &mut Gpu,
        token_id: u32,
        position: u32,
        mode: DecodeExecutionMode,
    ) -> Result<(), String> {
        decode_step_device_impl(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            token_id,
            position,
            self.retained_fixture_evidence,
            mode,
        )
    }

    pub fn download_logits(&self, gpu: &Gpu) -> Result<Vec<f32>, String> {
        gpu.download_f32(&self.state.logits)
            .map_err(|error| format!("lfm2moe: download logits: {error:?}"))
    }

    fn argmax_logits(&self, gpu: &mut Gpu) -> Result<u32, String> {
        gpu.argmax_f32_batched(
            &self.state.logits,
            &self.state.sample_buf,
            self.config.vocab_size,
            1,
        )
        .map_err(|error| format!("lfm2moe: GPU argmax: {error:?}"))?;
        let mut token = 0i32;
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                (&mut token as *mut i32).cast::<u8>(),
                std::mem::size_of::<i32>(),
            )
        };
        gpu.hip
            .memcpy_dtoh(bytes, &self.state.sample_buf.buf)
            .map_err(|error| format!("lfm2moe: download argmax token: {error:?}"))?;
        Ok(token as u32)
    }

    /// Apply Hugging Face's once-per-unique-token repetition penalty on the
    /// device, then return the greedy token after downloading only its index.
    ///
    /// `unique_seen_tokens` must not contain duplicates; duplicate ids would
    /// compound the in-place penalty and diverge from the host policy.
    pub fn sample_device_greedy(
        &self,
        gpu: &mut Gpu,
        unique_seen_tokens: &[u32],
        repeat_penalty: f32,
    ) -> Result<u32, String> {
        let token_bytes = unique_seen_tokens
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "lfm2moe: repetition token byte-size overflow".to_string())?;
        if token_bytes > self.state.repeat_buf.buf.size() {
            return Err(format!(
                "lfm2moe: {} unique repetition tokens require {token_bytes} bytes, buffer has {}",
                unique_seen_tokens.len(),
                self.state.repeat_buf.buf.size(),
            ));
        }
        if repeat_penalty != 1.0 && !unique_seen_tokens.is_empty() {
            let bytes = unsafe {
                std::slice::from_raw_parts(unique_seen_tokens.as_ptr().cast::<u8>(), token_bytes)
            };
            gpu.hip
                .memcpy_htod(&self.state.repeat_buf.buf, bytes)
                .map_err(|error| format!("lfm2moe: upload repetition tokens: {error:?}"))?;
            gpu.apply_hf_repetition_penalty_f32(
                &self.state.logits,
                &self.state.repeat_buf,
                unique_seen_tokens.len(),
                self.config.vocab_size,
                repeat_penalty,
            )
            .map_err(|error| format!("lfm2moe: GPU repetition penalty: {error:?}"))?;
        }
        self.argmax_logits(gpu)
    }

    pub fn decode_step_prefill(
        &mut self,
        gpu: &mut Gpu,
        token_id: u32,
        position: u32,
    ) -> Result<(), String> {
        decode_step_prefill_impl(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            token_id,
            position,
        )
    }

    pub fn forward_prefill_batch(
        &mut self,
        gpu: &mut Gpu,
        token_ids: &[u32],
        start_pos: u32,
    ) -> Result<Vec<f32>, String> {
        forward_prefill_batch_impl(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            token_ids,
            start_pos,
        )
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Self { weights, state, .. } = self;
        state.free_gpu(gpu);
        weights.free_gpu(gpu);
    }

    /// Copy the live conv-states INTO the snapshot buffers (pre-verify).
    fn save_conv(&self, snap: &[GpuTensor], gpu: &mut Gpu) -> Result<(), String> {
        for (dst, src) in snap.iter().zip(self.state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
                .map_err(|e| format!("lfm2moe: save conv snapshot: {e:?}"))?;
        }
        Ok(())
    }

    /// Copy the snapshot buffers back INTO the live conv-states (pre-replay).
    fn restore_conv(&self, snap: &[GpuTensor], gpu: &mut Gpu) -> Result<(), String> {
        for (src, dst) in snap.iter().zip(self.state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
                .map_err(|e| format!("lfm2moe: restore conv snapshot: {e:?}"))?;
        }
        Ok(())
    }
}

impl SpecTarget for Lfm2MoeBundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) {
        // Zero every conv-state ring buffer + reset the token count (the daemon's
        // arch_id=11 reset path). KV is overwritten by absolute-position writes,
        // so there is no separate KV cursor to rewind here; drop the eviction
        // offset for symmetry with qwen35's reset. The trait is infallible, so a
        // GPU reset failure is terminal: panic rather than acknowledge a
        // partially-reset recurrent state as clean.
        if let Err(e) = self.state.reset(gpu) {
            panic!("lfm2moe: reset_recurrent failed: {e}");
        }
        self.state.kv.compact_offset = 0;
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // One snapshot buffer per conv-state, sized to match it exactly.
        let mut conv_snap = Vec::with_capacity(self.state.conv_states.len());
        for cs in &self.state.conv_states {
            conv_snap.push(
                gpu.alloc_tensor(&cs.shape, DType::F32)
                    .map_err(|e| format!("lfm2moe: alloc conv snapshot: {e:?}"))?,
            );
        }
        Ok(Box::new(Lfm2MoeSpecScratch { conv_snap }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        // Plain target advance: feed each token at its absolute position. `reset`
        // zeroes the conv-states + token count first (cache-miss prefill); on a
        // cache-hit suffix / replay we continue from current recurrent state.
        if reset {
            self.state
                .reset(gpu)
                .map_err(|e| format!("lfm2moe: spec_advance reset: {e}"))?;
            self.state.kv.compact_offset = 0;
        }
        let retained_fixture_evidence = self.retained_fixture_evidence;
        for (i, &token) in tokens.iter().enumerate() {
            if abort() {
                self.state
                    .reset(gpu)
                    .map_err(|e| format!("lfm2moe: spec_advance abort reset: {e}"))?;
                self.state.kv.compact_offset = 0;
                return Ok(SpecAdvance::Aborted);
            }
            decode_step_speculative_device_impl(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                token,
                (start_pos + i) as u32,
                retained_fixture_evidence,
            )?;
        }
        let last_argmax = if tokens.is_empty() {
            0
        } else {
            self.argmax_logits(gpu)?
        };
        Ok(SpecAdvance::Ready { last_argmax })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let scratch = scratch
            .as_any_mut()
            .downcast_mut::<Lfm2MoeSpecScratch>()
            .ok_or("verify_block: scratch is not Lfm2MoeSpecScratch")?;
        // Snapshot the recurrent conv-state before sequential direct-HIP verify.
        // Attention KV is written at absolute positions and needs no snapshot.
        self.save_conv(&scratch.conv_snap, gpu)?;

        let retained_fixture_evidence = self.retained_fixture_evidence;
        let mut picks = Vec::with_capacity(block.len());
        for (i, &token) in block.iter().enumerate() {
            decode_step_speculative_device_impl(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                token,
                (position + i) as u32,
                retained_fixture_evidence,
            )?;
            picks.push(self.argmax_logits(gpu)?);
        }
        Ok(picks)
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Full accept: verify already left both the conv-state and the KV at
        // exactly position+block.len(); the bonus token is the next seed (not yet
        // fed). Nothing to undo.
        let draft_len = block.len() - 1;
        if accept_len >= draft_len {
            return Ok(());
        }
        // Partial accept: restore the pre-verify conv-state, then replay the
        // committed prefix with the same sequential direct-HIP decode as verify.
        // Absolute-position KV writes overwrite the accepted prefix; the rejected
        // tail is overwritten by the next verify window before it is read.
        let scratch = scratch
            .as_any_mut()
            .downcast_mut::<Lfm2MoeSpecScratch>()
            .ok_or("commit_prefix: scratch is not Lfm2MoeSpecScratch")?;
        self.restore_conv(&scratch.conv_snap, gpu)?;

        let retained_fixture_evidence = self.retained_fixture_evidence;
        for (i, &token) in block[..accept_len + 1].iter().enumerate() {
            decode_step_speculative_device_impl(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                token,
                (position + i) as u32,
                retained_fixture_evidence,
            )?;
        }
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.state.kv.physical_cap
    }

    // kv_cache_mut: defaulted to `None`. Although LFM2 stores attention KV in the
    // shared `llama::KvCache`, FlashCASK eviction is UNSOUND on this hybrid arch:
    // evicting attention KV would desync the attention layers from the conv-state
    // layers (whose recurrent history cannot be evicted in lockstep). arch_id=11
    // therefore has no eviction; the daemon's eviction sites are
    // `if let Some(ev)`-gated so this is never reached.
}

