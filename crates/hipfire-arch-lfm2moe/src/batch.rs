// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Independent continuous-batch state for dense LFM2.5 models.
//!
//! Mirrors Qwen35DecodeBatchState but for dense LFM2.5 (arch_id 11, num_experts==0).
//! Each lane owns an independent KV slice and an independent conv ring buffer.
//! Steady-state forward uses batched kernels once per layer with batch=B, not a per-lane loop.

use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::KvCache;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::{Lfm2MoeConfig, MixerKind};
use crate::lfm2moe::Lfm2MoeWeights;

/// Dense LFM2.5 independent continuous-batch decode state.
///
/// Owns the batched KV cache, per-layer conv rings, batched scratch buffers,
/// and per-row sampling state. Every lane reset is range-checked and clears
/// only that lane's slices.
pub struct Lfm2DecodeBatchState {
    pub max_batch: usize,
    pub lane_capacity: usize,
    pub sample_repeat_capacity: usize,
    pub kv: KvCache,
    pub conv_states: Vec<GpuTensor>,

    // batched scratch
    pub x_batch: GpuTensor,
    pub tmp_batch: GpuTensor,
    pub conv_bcx_batch: GpuTensor,
    pub conv_y_batch: GpuTensor,
    pub fa_q_batch: GpuTensor,
    pub fa_k_batch: GpuTensor,
    pub fa_v_batch: GpuTensor,
    pub fa_attn_out_batch: GpuTensor,
    pub ffn_tmp_batch: GpuTensor,
    pub dense_gate_batch: GpuTensor,
    pub dense_up_batch: GpuTensor,
    pub dense_act_batch: GpuTensor,
    pub final_hidden: GpuTensor,
    pub logits: GpuTensor,
    pub lm_rot: GpuTensor,
    pub proj_rot: GpuTensor,
    pub sample_out: GpuTensor,
    pub sample_repeat_tokens: GpuTensor,
    pub sample_repeat_lengths: GpuTensor,
    pub sample_rng_states: GpuTensor,
    pub positions: GpuTensor,
    pub tokens: GpuTensor,
}

impl Lfm2DecodeBatchState {
    /// Create a new dense batch state. Fails closed on zero/overflow geometry,
    /// MoE configs, or lane capacity that exceeds the GPU's independent-
    /// attention LDS limit.
    pub fn new(
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        max_batch: usize,
        lane_capacity: usize,
        sample_repeat_capacity: usize,
    ) -> HipResult<Self> {
        if max_batch == 0 || lane_capacity == 0 || sample_repeat_capacity == 0 {
            return Err(HipError::new(
                0,
                "decode batch size, lane capacity, and repeat capacity must be non-zero",
            ));
        }
        if cfg.num_experts != 0 {
            return Err(HipError::new(
                0,
                "Lfm2DecodeBatchState: MoE configs not supported (dense only, num_experts==0)",
            ));
        }
        if cfg.hidden_size == 0
            || cfg.num_hidden_layers == 0
            || cfg.vocab_size == 0
            || cfg.head_dim == 0
            || cfg.num_attention_heads == 0
            || cfg.num_key_value_heads == 0
            || cfg.conv_kernel_size < 2
        {
            return Err(HipError::new(
                0,
                "Lfm2DecodeBatchState: zero or degenerate geometry",
            ));
        }
        // LDS check and full overflow validation BEFORE any GPU allocation.
        gpu.ensure_attention_q8_0_kv_independent_lds(lane_capacity, cfg.head_dim)?;

        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.intermediate_size;
        if dense_inter == 0 {
            return Err(HipError::new(
                0,
                "Lfm2DecodeBatchState: dense intermediate_size must be non-zero for dense model",
            ));
        }
        // All max_batch products that will be allocated as [N] tensors — fail fast before any GPU alloc.
        let total_capacity = max_batch
            .checked_mul(lane_capacity)
            .ok_or_else(|| HipError::new(0, "decode batch KV capacity multiplication overflow"))?;
        let repeat_tokens_len = max_batch
            .checked_mul(sample_repeat_capacity)
            .ok_or_else(|| {
                HipError::new(0, "decode batch repeat capacity multiplication overflow")
            })?;
        // Validate that layer counts are self-consistent; total_capacity already checked.
        let _ = total_capacity
            .checked_mul(cfg.num_key_value_heads)
            .ok_or_else(|| HipError::new(0, "KV head capacity overflow"))?;
        let _ = total_capacity
            .checked_mul(cfg.head_dim)
            .ok_or_else(|| HipError::new(0, "KV dim capacity overflow"))?;
        // Conv hist.
        let k = cfg.conv_kernel_size;
        let hist = hidden
            .checked_mul(k - 1)
            .ok_or_else(|| HipError::new(0, "conv hist size overflow"))?;
        let conv_hist_per_layer = max_batch
            .checked_mul(hist)
            .ok_or_else(|| HipError::new(0, "conv batch hist size overflow"))?;
        // All batched scratch products — checked before any allocation.
        let x_len = max_batch
            .checked_mul(hidden)
            .ok_or_else(|| HipError::new(0, "x_batch size overflow"))?;
        let bcx_len = max_batch
            .checked_mul(3)
            .and_then(|v| v.checked_mul(hidden))
            .ok_or_else(|| HipError::new(0, "conv_bcx size overflow"))?;
        let q_len = max_batch
            .checked_mul(q_dim)
            .ok_or_else(|| HipError::new(0, "fa_q size overflow"))?;
        let kv_len = max_batch
            .checked_mul(kv_dim)
            .ok_or_else(|| HipError::new(0, "fa_kv size overflow"))?;
        let gate_len = max_batch
            .checked_mul(dense_inter)
            .ok_or_else(|| HipError::new(0, "dense gate size overflow"))?;
        let logits_len = max_batch
            .checked_mul(cfg.vocab_size)
            .ok_or_else(|| HipError::new(0, "logits size overflow"))?;
        let sample_out_len = max_batch
            .checked_mul(2)
            .ok_or_else(|| HipError::new(0, "sample_out size overflow"))?;
        // Touch each computed len to silence unused warning and prove validation.
        let _ = (
            x_len,
            bcx_len,
            q_len,
            kv_len,
            gate_len,
            logits_len,
            sample_out_len,
        );
        let _ = conv_hist_per_layer;
        let _ = repeat_tokens_len;

        // `AttnWeights::kv_idx` is compact over attention layers only. Keep the
        // batch cache in that same index space; global mixer-layer placeholders
        // would let a compact index land on a one-element Conv placeholder.
        let is_kv_layer = batch_kv_slot_mask(&cfg.layer_types);

        let kv = match KvCache::new_gpu_q8_filtered(
            gpu,
            &is_kv_layer,
            cfg.num_key_value_heads,
            cfg.head_dim,
            total_capacity,
        ) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        // Conv states: per conv layer [conv_hist_per_layer] F32 zeroed (already validated).
        let n_conv = cfg.num_conv_layers();
        let mut conv_states: Vec<GpuTensor> = Vec::with_capacity(n_conv);
        for idx in 0..n_conv {
            let t = match gpu.zeros(&[conv_hist_per_layer], DType::F32) {
                Ok(v) => v,
                Err(e) => {
                    for prev in conv_states.drain(..) {
                        let _ = gpu.free_tensor(prev);
                    }
                    let _ = kv.free_gpu(gpu);
                    return Err(e);
                }
            };
            conv_states.push(t);
            let _ = idx;
        }
        // Allocation ledger for the remaining tensors (alias on error, drop on success).
        let mut ledger: Vec<GpuTensor> = Vec::with_capacity(24);
        macro_rules! alloc_zeros {
            ($shape:expr) => {
                match gpu.zeros($shape, DType::F32) {
                    Ok(t) => {
                        ledger.push(GpuTensor {
                            buf: unsafe { t.buf.alias() },
                            shape: t.shape.clone(),
                            dtype: t.dtype,
                        });
                        t
                    }
                    Err(e) => {
                        for prev in ledger.drain(..) {
                            let _ = gpu.free_tensor(prev);
                        }
                        for cs in conv_states.drain(..) {
                            let _ = gpu.free_tensor(cs);
                        }
                        let _ = kv.free_gpu(gpu);
                        return Err(e);
                    }
                }
            };
        }

        let x_batch = alloc_zeros!(&[max_batch * hidden]);
        let tmp_batch = alloc_zeros!(&[max_batch * hidden]);
        let conv_bcx_batch = alloc_zeros!(&[max_batch * 3 * hidden]);
        let conv_y_batch = alloc_zeros!(&[max_batch * hidden]);
        let fa_q_batch = alloc_zeros!(&[max_batch * q_dim]);
        let fa_k_batch = alloc_zeros!(&[max_batch * kv_dim]);
        let fa_v_batch = alloc_zeros!(&[max_batch * kv_dim]);
        let fa_attn_out_batch = alloc_zeros!(&[max_batch * q_dim]);
        let ffn_tmp_batch = alloc_zeros!(&[max_batch * hidden]);
        let dense_gate_batch = alloc_zeros!(&[max_batch * dense_inter]);
        let dense_up_batch = alloc_zeros!(&[max_batch * dense_inter]);
        let dense_act_batch = alloc_zeros!(&[max_batch * dense_inter]);
        let final_hidden = alloc_zeros!(&[max_batch * hidden]);
        let logits = alloc_zeros!(&[max_batch * cfg.vocab_size]);
        let lm_rot = alloc_zeros!(&[max_batch * hidden]);
        let max_proj_k = hidden.max(dense_inter);
        let proj_rot_len = max_batch
            .checked_mul(max_proj_k)
            .ok_or_else(|| HipError::new(0, "proj_rot size overflow"))?;
        let proj_rot = alloc_zeros!(&[proj_rot_len]);
        let sample_out = alloc_zeros!(&[max_batch * 2]);
        let sample_repeat_tokens = alloc_zeros!(&[repeat_tokens_len]);
        let sample_repeat_lengths = alloc_zeros!(&[max_batch]);
        let sample_rng_states = alloc_zeros!(&[max_batch]);
        let positions = alloc_zeros!(&[max_batch]);
        let tokens = alloc_zeros!(&[max_batch]);

        Ok(Self {
            max_batch,
            lane_capacity,
            sample_repeat_capacity,
            kv,
            conv_states,
            x_batch,
            tmp_batch,
            conv_bcx_batch,
            conv_y_batch,
            fa_q_batch,
            fa_k_batch,
            fa_v_batch,
            fa_attn_out_batch,
            ffn_tmp_batch,
            dense_gate_batch,
            dense_up_batch,
            dense_act_batch,
            final_hidden,
            logits,
            lm_rot,
            proj_rot,
            sample_out,
            sample_repeat_tokens,
            sample_repeat_lengths,
            sample_rng_states,
            positions,
            tokens,
        })
    }

    /// Clear all lanes and scratch.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        self.kv.clear_gpu(gpu)?;
        for cs in &self.conv_states {
            gpu.hip.memset(&cs.buf, 0, cs.buf.size())?;
        }
        for t in [
            &self.x_batch,
            &self.tmp_batch,
            &self.conv_bcx_batch,
            &self.conv_y_batch,
            &self.fa_q_batch,
            &self.fa_k_batch,
            &self.fa_v_batch,
            &self.fa_attn_out_batch,
            &self.ffn_tmp_batch,
            &self.dense_gate_batch,
            &self.dense_up_batch,
            &self.dense_act_batch,
            &self.final_hidden,
            &self.logits,
            &self.lm_rot,
            &self.proj_rot,
            &self.sample_out,
            &self.sample_repeat_tokens,
            &self.sample_repeat_lengths,
            &self.sample_rng_states,
            &self.positions,
            &self.tokens,
        ] {
            gpu.hip.memset(&t.buf, 0, t.buf.size())?;
        }
        Ok(())
    }

    /// Clear only one lane's KV, conv rings, and output row. Range-checked.
    pub fn reset_lane(&mut self, gpu: &mut Gpu, cfg: &Lfm2MoeConfig, lane: usize) -> HipResult<()> {
        if lane >= self.max_batch {
            return Err(HipError::new(0, "reset_lane: lane is out of range"));
        }
        if cfg.num_experts != 0 {
            return Err(HipError::new(
                0,
                "reset_lane: MoE not supported for dense batch",
            ));
        }
        let mut kv_lane = self.kv.q8_lane_view(lane, self.lane_capacity)?;
        kv_lane.clear_gpu(gpu)?;
        let hidden = cfg.hidden_size;
        let hist = hidden * (cfg.conv_kernel_size - 1);
        if hist > 0 {
            for cs in &self.conv_states {
                // cs is [max_batch* hist]; lane slice is contiguous.
                let lane_view = cs.sub_offset(lane * hist, hist);
                gpu.hip.memset(&lane_view.buf, 0, lane_view.buf.size())?;
            }
        }
        // zero per-row sampling/logits/norm buffers
        let lane_hidden = self.final_hidden.sub_offset(lane * hidden, hidden);
        gpu.hip
            .memset(&lane_hidden.buf, 0, lane_hidden.buf.size())?;
        let lane_logits = self
            .logits
            .sub_offset(lane * cfg.vocab_size, cfg.vocab_size);
        gpu.hip
            .memset(&lane_logits.buf, 0, lane_logits.buf.size())?;
        let lane_rot = self.lm_rot.sub_offset(lane * hidden, hidden);
        gpu.hip.memset(&lane_rot.buf, 0, lane_rot.buf.size())?;
        // proj_rot lane: max_k elements
        let max_proj_k = hidden.max(cfg.intermediate_size);
        let lane_proj = self.proj_rot.sub_offset(lane * max_proj_k, max_proj_k);
        gpu.hip.memset(&lane_proj.buf, 0, lane_proj.buf.size())?;

        // also clear x/tmp and attention scratch lane rows to avoid stale residues
        let lane_x = self.x_batch.sub_offset(lane * hidden, hidden);
        gpu.hip.memset(&lane_x.buf, 0, lane_x.buf.size())?;
        let lane_tmp = self.tmp_batch.sub_offset(lane * hidden, hidden);
        gpu.hip.memset(&lane_tmp.buf, 0, lane_tmp.buf.size())?;

        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let lane_fq = self.fa_q_batch.sub_offset(lane * q_dim, q_dim);
        gpu.hip.memset(&lane_fq.buf, 0, lane_fq.buf.size())?;
        let lane_fk = self.fa_k_batch.sub_offset(lane * kv_dim, kv_dim);
        gpu.hip.memset(&lane_fk.buf, 0, lane_fk.buf.size())?;
        let lane_fv = self.fa_v_batch.sub_offset(lane * kv_dim, kv_dim);
        gpu.hip.memset(&lane_fv.buf, 0, lane_fv.buf.size())?;
        let lane_fa_out = self.fa_attn_out_batch.sub_offset(lane * q_dim, q_dim);
        gpu.hip
            .memset(&lane_fa_out.buf, 0, lane_fa_out.buf.size())?;

        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv.free_gpu(gpu);
        for cs in self.conv_states {
            let _ = gpu.free_tensor(cs);
        }
        let _ = gpu.free_tensor(self.x_batch);
        let _ = gpu.free_tensor(self.tmp_batch);
        let _ = gpu.free_tensor(self.conv_bcx_batch);
        let _ = gpu.free_tensor(self.conv_y_batch);
        let _ = gpu.free_tensor(self.fa_q_batch);
        let _ = gpu.free_tensor(self.fa_k_batch);
        let _ = gpu.free_tensor(self.fa_v_batch);
        let _ = gpu.free_tensor(self.fa_attn_out_batch);
        let _ = gpu.free_tensor(self.ffn_tmp_batch);
        let _ = gpu.free_tensor(self.dense_gate_batch);
        let _ = gpu.free_tensor(self.dense_up_batch);
        let _ = gpu.free_tensor(self.dense_act_batch);
        let _ = gpu.free_tensor(self.final_hidden);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.lm_rot);
        let _ = gpu.free_tensor(self.proj_rot);
        let _ = gpu.free_tensor(self.sample_out);
        let _ = gpu.free_tensor(self.sample_repeat_tokens);
        let _ = gpu.free_tensor(self.sample_repeat_lengths);
        let _ = gpu.free_tensor(self.sample_rng_states);
        let _ = gpu.free_tensor(self.positions);
        let _ = gpu.free_tensor(self.tokens);
    }

    /// Seed one lane with a full prompt. Sequential per-lane prefill is
    /// allowed only during admission; steady-state decode remains batched.
    /// Returns Ok(true) if the full prompt was prefilled, Ok(false) if cancelled
    /// via `should_cancel` before sampling/logits use (lane left for daemon reset).
    pub fn prefill_lane_cancellable(
        &mut self,
        gpu: &mut Gpu,
        weights: &Lfm2MoeWeights,
        cfg: &Lfm2MoeConfig,
        lane: usize,
        tokens: &[u32],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> HipResult<bool> {
        if lane >= self.max_batch {
            return Err(HipError::new(0, "prefill_lane: lane out of range"));
        }
        if tokens.is_empty() || tokens.len() >= self.lane_capacity {
            return Err(HipError::new(
                0,
                "prefill_lane: token count invalid or leaves no decode capacity",
            ));
        }
        if cfg.num_experts != 0 {
            return Err(HipError::new(
                0,
                "prefill_lane: MoE not supported for dense batch",
            ));
        }
        // Authoritative weight format check (via lfm2moe helper to avoid cycle).
        crate::lfm2moe::batch_weight_formats_supported(weights)
            .map_err(|e| HipError::new(0, &e))?;
        // Need to clear lane before seeding (so KV/conv start clean).
        {
            let mut kv_lane = self.kv.q8_lane_view(lane, self.lane_capacity)?;
            kv_lane.clear_gpu(gpu)?;
            let hidden = cfg.hidden_size;
            let hist = hidden * (cfg.conv_kernel_size - 1);
            for cs in &self.conv_states {
                let v = cs.sub_offset(lane * hist, hist);
                gpu.hip.memset(&v.buf, 0, v.buf.size())?;
            }
        }

        // Sequential per-token forward for this lane using lane-isolated views.
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let eps = cfg.rms_norm_eps;

        let h_lane = self.x_batch.sub_offset(lane * hidden, hidden);
        let tmp_lane = self.tmp_batch.sub_offset(lane * hidden, hidden);
        let bcx_lane = self
            .conv_bcx_batch
            .sub_offset(lane * 3 * hidden, 3 * hidden);
        let y_lane = self.conv_y_batch.sub_offset(lane * hidden, hidden);
        let fq_lane = self.fa_q_batch.sub_offset(lane * q_dim, q_dim);
        let fk_lane = self.fa_k_batch.sub_offset(lane * kv_dim, kv_dim);
        let fv_lane = self.fa_v_batch.sub_offset(lane * kv_dim, kv_dim);
        let fo_lane = self.fa_attn_out_batch.sub_offset(lane * q_dim, q_dim);
        let ffn_tmp_lane = self.ffn_tmp_batch.sub_offset(lane * hidden, hidden);
        let gate_lane = self
            .dense_gate_batch
            .sub_offset(lane * cfg.intermediate_size, cfg.intermediate_size);
        let up_lane = self
            .dense_up_batch
            .sub_offset(lane * cfg.intermediate_size, cfg.intermediate_size);
        let act_lane = self
            .dense_act_batch
            .sub_offset(lane * cfg.intermediate_size, cfg.intermediate_size);
        let final_lane = self.final_hidden.sub_offset(lane * hidden, hidden);

        // Reuse the lane's positions slot as the scalar position buffer (no hipMalloc).
        let pos_buf = self.positions.sub_offset(lane, 1);

        for (pos, &tok) in tokens.iter().enumerate() {
            if should_cancel() {
                return Ok(false);
            }
            gpu.hip
                .memcpy_htod(&pos_buf.buf, &(pos as i32).to_ne_bytes())
                .map_err(|e| HipError::new(0, &format!("prefill_lane htod pos: {e:?}")))?;

            // embedding into h_lane (reuse same buffer as residual stream)
            match weights.embd_format {
                hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&weights.embed, &h_lane, tok, hidden)?
                }
                hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
                    gpu.embedding_lookup_hfq4g256(&weights.embed, &h_lane, tok, hidden)?
                }
                _ => {
                    return Err(HipError::new(
                        0,
                        "prefill_lane: unsupported embedding format",
                    ))
                }
            }

            // Per-layer stack (reuse existing single-lane helpers inline, operating on lane views).
            for (l, layer) in weights.layers.iter().enumerate() {
                gpu.rmsnorm_f32(&h_lane, &layer.operator_norm, &tmp_lane, eps)
                    .map_err(|e| {
                        HipError::new(0, &format!("prefill L{l} operator rmsnorm: {e:?}"))
                    })?;
                match &layer.mixer {
                    crate::lfm2moe::Mixer::Conv(c) => {
                        hipfire_runtime::llama::weight_gemv(gpu, &c.in_proj, &tmp_lane, &bcx_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} conv in_proj: {e:?}"))
                            })?;
                        let cs = &self.conv_states[c.conv_state_idx];
                        let hist = hidden * (cfg.conv_kernel_size - 1);
                        let cs_lane = cs.sub_offset(lane * hist, hist);
                        gpu.conv1d_gated_decode_f32(
                            &bcx_lane,
                            &cs_lane,
                            &c.conv_weight,
                            &y_lane,
                            1,
                            hidden,
                            cfg.conv_kernel_size,
                        )
                        .map_err(|e| {
                            HipError::new(0, &format!("prefill L{l} conv gated decode: {e:?}"))
                        })?;
                        hipfire_runtime::llama::weight_gemv_residual(
                            gpu,
                            &c.out_proj,
                            &y_lane,
                            &h_lane,
                        )
                        .map_err(|e| {
                            HipError::new(0, &format!("prefill L{l} conv out_proj: {e:?}"))
                        })?;
                    }
                    crate::lfm2moe::Mixer::Attention(a) => {
                        hipfire_runtime::llama::weight_gemv(gpu, &a.wq, &tmp_lane, &fq_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} q_proj: {e:?}"))
                            })?;
                        hipfire_runtime::llama::weight_gemv(gpu, &a.wk, &tmp_lane, &fk_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} k_proj: {e:?}"))
                            })?;
                        hipfire_runtime::llama::weight_gemv(gpu, &a.wv, &tmp_lane, &fv_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} v_proj: {e:?}"))
                            })?;
                        gpu.rmsnorm_batched(
                            &fq_lane,
                            &a.q_norm,
                            &fq_lane,
                            cfg.num_attention_heads,
                            cfg.head_dim,
                            eps,
                        )
                        .map_err(|e| HipError::new(0, &format!("prefill L{l} q_norm: {e:?}")))?;
                        gpu.rmsnorm_batched(
                            &fk_lane,
                            &a.k_norm,
                            &fk_lane,
                            cfg.num_key_value_heads,
                            cfg.head_dim,
                            eps,
                        )
                        .map_err(|e| HipError::new(0, &format!("prefill L{l} k_norm: {e:?}")))?;
                        gpu.rope_f32(
                            &fq_lane,
                            &fk_lane,
                            &pos_buf.buf,
                            cfg.num_attention_heads,
                            cfg.num_key_value_heads,
                            cfg.head_dim,
                            cfg.rope_theta,
                        )
                        .map_err(|e| HipError::new(0, &format!("prefill L{l} rope: {e:?}")))?;

                        let kv_idx = a.kv_idx;
                        let kv_lane = self.kv.q8_lane_view(lane, self.lane_capacity)?;
                        gpu.kv_cache_write_q8_0(
                            &kv_lane.k_gpu[kv_idx],
                            &fk_lane,
                            &pos_buf.buf,
                            cfg.num_key_value_heads,
                            cfg.head_dim,
                        )
                        .map_err(|e| {
                            HipError::new(0, &format!("prefill L{l} kv write k: {e:?}"))
                        })?;
                        gpu.kv_cache_write_q8_0(
                            &kv_lane.v_gpu[kv_idx],
                            &fv_lane,
                            &pos_buf.buf,
                            cfg.num_key_value_heads,
                            cfg.head_dim,
                        )
                        .map_err(|e| {
                            HipError::new(0, &format!("prefill L{l} kv write v: {e:?}"))
                        })?;

                        let seq_len = pos + 1;
                        gpu.attention_q8_0_kv(
                            &fq_lane,
                            &kv_lane.k_gpu[kv_idx],
                            &kv_lane.v_gpu[kv_idx],
                            &fo_lane,
                            &pos_buf.buf,
                            seq_len,
                            cfg.num_attention_heads,
                            cfg.num_key_value_heads,
                            cfg.head_dim,
                            self.lane_capacity,
                        )
                        .map_err(|e| HipError::new(0, &format!("prefill L{l} attention: {e:?}")))?;
                        hipfire_runtime::llama::weight_gemv_residual(gpu, &a.wo, &fo_lane, &h_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} out_proj: {e:?}"))
                            })?;
                    }
                }

                gpu.rmsnorm_f32(&h_lane, &layer.ffn_norm, &ffn_tmp_lane, eps)
                    .map_err(|e| HipError::new(0, &format!("prefill L{l} ffn rmsnorm: {e:?}")))?;
                match &layer.ffn {
                    crate::lfm2moe::Ffn::Dense(d) => {
                        hipfire_runtime::llama::weight_gemv(gpu, &d.w1, &ffn_tmp_lane, &gate_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} dense w1: {e:?}"))
                            })?;
                        hipfire_runtime::llama::weight_gemv(gpu, &d.w3, &ffn_tmp_lane, &up_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} dense w3: {e:?}"))
                            })?;
                        gpu.silu_mul_f32(&gate_lane, &up_lane, &act_lane)
                            .map_err(|e| {
                                HipError::new(0, &format!("prefill L{l} silu_mul: {e:?}"))
                            })?;
                        hipfire_runtime::llama::weight_gemv_residual(
                            gpu, &d.w2, &act_lane, &h_lane,
                        )
                        .map_err(|e| HipError::new(0, &format!("prefill L{l} dense w2: {e:?}")))?;
                    }
                    crate::lfm2moe::Ffn::Moe(_) => {
                        return Err(HipError::new(
                            0,
                            "prefill_lane: MoE FFN not supported in dense batch",
                        ))
                    }
                }
            }

            gpu.rmsnorm_f32(&h_lane, &weights.embedding_norm, &final_lane, eps)
                .map_err(|e| HipError::new(0, &format!("prefill final rmsnorm: {e:?}")))?;
            // Compute logits for this lane's final position; we keep overwriting lane logits
            // so after the loop the lane's row holds the last prompt token's logits.
            let logits_lane = self
                .logits
                .sub_offset(lane * cfg.vocab_size, cfg.vocab_size);
            hipfire_runtime::llama::weight_gemv(gpu, &weights.lm_head, &final_lane, &logits_lane)
                .map_err(|e| HipError::new(0, &format!("prefill lm_head: {e:?}")))?;
        }

        // Final cancellation check before logits are considered valid for sampling.
        if should_cancel() {
            return Ok(false);
        }
        Ok(true)
    }

    /// Legacy wrapper: prefill without cancellation (always completes).
    pub fn prefill_lane(
        &mut self,
        gpu: &mut Gpu,
        weights: &Lfm2MoeWeights,
        cfg: &Lfm2MoeConfig,
        lane: usize,
        tokens: &[u32],
    ) -> HipResult<()> {
        let ok = self.prefill_lane_cancellable(gpu, weights, cfg, lane, tokens, &mut || false)?;
        if !ok {
            return Err(HipError::new(0, "prefill_lane: cancelled"));
        }
        Ok(())
    }

    /// Pure geometry preflight for equal-length batched prefill.
    ///
    /// Validates `prompts` without touching GPU: `n` must be `2..=max_batch`,
    /// all prompts equal non-zero length and each `< lane_capacity`.
    /// Returns the common prompt length on success.
    pub fn batched_prefill_preflight(
        prompts: &[&[u32]],
        max_batch: usize,
        lane_capacity: usize,
    ) -> Result<usize, String> {
        let n = prompts.len();
        if n < 2 {
            return Err(format!(
                "batched prefill: requires at least 2 lanes, got {n}"
            ));
        }
        if n > max_batch {
            return Err(format!(
                "batched prefill: n={n} exceeds max_batch={max_batch}"
            ));
        }
        if lane_capacity == 0 {
            return Err("batched prefill: lane_capacity must be non-zero".to_string());
        }
        let first_len = prompts[0].len();
        if first_len == 0 {
            return Err("batched prefill: prompt length must be non-zero".to_string());
        }
        if first_len >= lane_capacity {
            return Err(format!(
                "batched prefill: prompt_len {first_len} >= lane_capacity {lane_capacity}"
            ));
        }
        for (i, p) in prompts.iter().enumerate() {
            if p.len() != first_len {
                return Err(format!(
                    "batched prefill: mixed prompt lengths: lane 0 has {first_len}, lane {i} has {}",
                    p.len()
                ));
            }
            if p.len() >= lane_capacity {
                return Err(format!(
                    "batched prefill: prompt_len {} >= lane_capacity {lane_capacity} at lane {i}",
                    p.len()
                ));
            }
            if p.is_empty() {
                return Err(format!("batched prefill: lane {i} has empty prompt"));
            }
        }
        Ok(first_len)
    }

    /// Batched equal-length prefill for lanes `0..n`.
    ///
    /// Takes a prefix-contiguous wave `prompts[0..n]` with `n>=2`, equal
    /// nonzero lengths but arbitrary token contents. Validates all geometry,
    /// formats, and lengths **before** any `reset_lane`/GPU work. On success,
    /// clears each lane `0..n` via `reset_lane`, then iterates prompt
    /// positions calling `forward_decode_batch_lfm` once per position with a
    /// dense `[n]` token row and equal positions. Final logits/KV/conv state
    /// equal sequential `prefill_lane`.
    pub fn prefill_lanes_batched(
        &mut self,
        gpu: &mut Gpu,
        weights: &Lfm2MoeWeights,
        cfg: &Lfm2MoeConfig,
        prompts: &[&[u32]],
    ) -> HipResult<()> {
        // ---- pure preflight (no GPU, no reset) ----
        let prompt_len =
            Self::batched_prefill_preflight(prompts, self.max_batch, self.lane_capacity)
                .map_err(|e| HipError::new(0, &e))?;
        // Config / format preflight before any GPU work
        if cfg.num_experts != 0 {
            return Err(HipError::new(
                0,
                "prefill_lanes_batched: MoE not supported for dense batch",
            ));
        }
        crate::lfm2moe::batch_weight_formats_supported(weights)
            .map_err(|e| HipError::new(0, &e))?;
        let n = prompts.len();
        // ---- reset lanes 0..n (no GPU work yet beyond memset) ----
        for lane in 0..n {
            self.reset_lane(gpu, cfg, lane)?;
        }
        // ---- batched forward per position: O(prompt_len) forwards, each B=n ----
        let mut tokens_row = vec![0u32; n];
        let mut positions_row = vec![0usize; n];
        for pos in 0..prompt_len {
            for (lane, prompt) in prompts.iter().enumerate() {
                tokens_row[lane] = prompt[pos];
                positions_row[lane] = pos;
            }
            crate::forward_batch::forward_decode_batch_lfm(
                gpu,
                weights,
                cfg,
                &tokens_row,
                &positions_row,
                self,
            )?;
        }
        Ok(())
    }

    /// Product sampler for a dense batch.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_product(
        &self,
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        batch_size: usize,
        repeat_tokens: &[u32],
        repeat_lengths: &[u32],
        rng_states: &[u32],
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: Option<f32>,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> HipResult<Vec<(u32, u32)>> {
        if batch_size == 0 || batch_size > self.max_batch {
            return Err(HipError::new(0, "sample_product: batch_size out of range"));
        }
        if repeat_tokens.len() != batch_size * self.sample_repeat_capacity
            || repeat_lengths.len() != batch_size
            || rng_states.len() != batch_size
        {
            return Err(HipError::new(
                0,
                "sample_product: inputs do not match active batch shape",
            ));
        }
        if cfg.num_experts != 0 {
            return Err(HipError::new(
                0,
                "sample_product: MoE not supported for dense batch",
            ));
        }
        let repeat_bytes = unsafe {
            std::slice::from_raw_parts(repeat_tokens.as_ptr() as *const u8, repeat_tokens.len() * 4)
        };
        let length_bytes = unsafe {
            std::slice::from_raw_parts(
                repeat_lengths.as_ptr() as *const u8,
                repeat_lengths.len() * 4,
            )
        };
        let rng_bytes = unsafe {
            std::slice::from_raw_parts(rng_states.as_ptr() as *const u8, rng_states.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&self.sample_repeat_tokens.buf, repeat_bytes)?;
        gpu.hip
            .memcpy_htod(&self.sample_repeat_lengths.buf, length_bytes)?;
        gpu.hip
            .memcpy_htod(&self.sample_rng_states.buf, rng_bytes)?;
        let logits = self.logits.sub_offset(0, batch_size * cfg.vocab_size);
        gpu.sample_rows_pf_f32(
            &logits,
            &self.sample_repeat_tokens,
            &self.sample_repeat_lengths,
            &self.sample_rng_states,
            &self.sample_out,
            batch_size,
            cfg.vocab_size,
            self.sample_repeat_capacity,
            temperature,
            top_p,
            repeat_penalty,
            presence_penalty,
            frequency_penalty,
            top_k,
            min_p,
        )
    }

    /// Single-row product sampler for refill (lane-replacement).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_lane_product(
        &self,
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        lane: usize,
        repeat_tokens: &[u32],
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: Option<f32>,
        rng_state: u32,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        if lane >= self.max_batch {
            return Err(HipError::new(0, "sample_lane_product: lane out of range"));
        }
        if repeat_tokens.len() > self.sample_repeat_capacity {
            return Err(HipError::new(
                0,
                "sample_lane_product: history exceeds repeat capacity",
            ));
        }
        let repeat_bytes = unsafe {
            std::slice::from_raw_parts(repeat_tokens.as_ptr() as *const u8, repeat_tokens.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&self.sample_repeat_tokens.buf, repeat_bytes)?;
        // zero the tail beyond len to avoid stale?
        let logits = self
            .logits
            .sub_offset(lane * cfg.vocab_size, cfg.vocab_size);
        gpu.sample_top_p_pf(
            &logits,
            &self.sample_out,
            &self.sample_repeat_tokens,
            cfg.vocab_size,
            temperature,
            top_p,
            rng_state,
            repeat_tokens.len(),
            repeat_penalty,
            presence_penalty,
            frequency_penalty,
            top_k,
            min_p,
        )
    }
}

fn batch_kv_slot_mask(layer_types: &[MixerKind]) -> Vec<bool> {
    let attention_slots = layer_types
        .iter()
        .filter(|&&kind| kind == MixerKind::Attention)
        .count()
        .max(1);
    vec![true; attention_slots]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_kv_slots_match_compact_attention_indices() {
        let layer_types = [
            MixerKind::Conv,
            MixerKind::Attention,
            MixerKind::Conv,
            MixerKind::Attention,
            MixerKind::Conv,
        ];
        let mask = batch_kv_slot_mask(&layer_types);
        assert_eq!(mask.len(), 2);
        assert!(mask.into_iter().all(|enabled| enabled));
    }

    #[test]
    fn batch_kv_slots_keep_zero_attention_shape_allocatable() {
        assert_eq!(batch_kv_slot_mask(&[MixerKind::Conv]), vec![true]);
    }

    #[test]
    fn batched_prefill_preflight_rejects_empty() {
        let err = Lfm2DecodeBatchState::batched_prefill_preflight(&[], 4, 2048).unwrap_err();
        assert!(err.contains("at least 2"), "{err}");
    }

    #[test]
    fn batched_prefill_preflight_rejects_single() {
        let a = vec![1u32, 2, 3];
        let prompts: Vec<&[u32]> = vec![&a];
        let err = Lfm2DecodeBatchState::batched_prefill_preflight(&prompts, 4, 2048).unwrap_err();
        assert!(err.contains("at least 2"), "{err}");
    }

    #[test]
    fn batched_prefill_preflight_rejects_mixed_length() {
        let a = vec![1u32, 2, 3];
        let b = vec![1u32, 2];
        let c = vec![1u32, 2, 3];
        let prompts: Vec<&[u32]> = vec![&a, &b, &c];
        let err = Lfm2DecodeBatchState::batched_prefill_preflight(&prompts, 8, 2048).unwrap_err();
        assert!(err.contains("mixed"), "{err}");
    }

    #[test]
    fn batched_prefill_preflight_rejects_over_capacity() {
        let a = vec![1u32; 2048];
        let b = vec![1u32; 2048];
        let prompts: Vec<&[u32]> = vec![&a, &b];
        let err = Lfm2DecodeBatchState::batched_prefill_preflight(&prompts, 8, 2048).unwrap_err();
        assert!(err.contains("lane_capacity"), "{err}");
    }

    #[test]
    fn batched_prefill_preflight_rejects_over_max_batch() {
        let a = vec![1u32, 2];
        let b = vec![3u32, 4];
        let c = vec![5u32, 6];
        let prompts: Vec<&[u32]> = vec![&a, &b, &c];
        let err = Lfm2DecodeBatchState::batched_prefill_preflight(&prompts, 2, 2048).unwrap_err();
        assert!(err.contains("max_batch"), "{err}");
    }

    #[test]
    fn batched_prefill_preflight_accepts_equal() {
        let a = vec![1u32, 2, 3];
        let b = vec![4u32, 5, 6];
        let prompts: Vec<&[u32]> = vec![&a, &b];
        let len = Lfm2DecodeBatchState::batched_prefill_preflight(&prompts, 8, 2048).unwrap();
        assert_eq!(len, 3);
    }

    #[test]
    fn batched_prefill_preflight_rejects_zero_prompt_len() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![];
        let prompts: Vec<&[u32]> = vec![&a, &b];
        let err = Lfm2DecodeBatchState::batched_prefill_preflight(&prompts, 8, 2048).unwrap_err();
        assert!(err.contains("non-zero") || err.contains("empty"), "{err}");
    }
}
