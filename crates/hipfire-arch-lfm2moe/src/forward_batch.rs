// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Batched decode forward path for dense LFM2.5 models.
//!
//! Provides `prepare_decode_batch_inputs_lfm`, `forward_decode_batch_lfm` and
//! `forward_decode_batch_prepared_lfm`. Each layer launches **one** batched kernel
//! with `batch = B`; the conv layer uses the existing `conv1d_gated_decode_f32`
//! with `batch = B` and a lane-isolated `[B,C,K-1]` state. No per-lane
//! full-forward loop exists in steady-state decode (prefill may be sequential).

use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu};

use crate::batch::Lfm2DecodeBatchState;
use crate::lfm2moe::Lfm2MoeWeights;

// ──────────────────────────────────────────────────────────────────────────
// helpers
// ──────────────────────────────────────────────────────────────────────────

fn check_dense(cfg: &crate::config::Lfm2MoeConfig) -> HipResult<()> {
    if cfg.num_experts != 0 {
        return Err(HipError::new(
            0,
            "batched decode: MoE configs not supported (dense only)",
        ));
    }
    if cfg.hidden_size == 0
        || cfg.vocab_size == 0
        || cfg.num_hidden_layers == 0
        || cfg.head_dim == 0
        || cfg.num_attention_heads == 0
        || cfg.num_key_value_heads == 0
        || cfg.intermediate_size == 0
    {
        return Err(HipError::new(0, "batched decode: zero geometry"));
    }
    if cfg.conv_kernel_size < 2 || cfg.conv_kernel_size > 8 {
        return Err(HipError::new(
            0,
            "batched decode: conv_kernel_size must be in [2,8]",
        ));
    }
    Ok(())
}

fn check_weight_formats(weights: &Lfm2MoeWeights) -> HipResult<()> {
    crate::lfm2moe::batch_weight_formats_supported(weights).map_err(|e| HipError::new(0, &e))
}

/// Authoritative batch format predicate — single source of truth for daemon attestation
/// and all forward paths. Dense projections: Q8_0/HFQ4G256/MQ4G256; embeddings: Q8/HFQ4;
/// lm_head: Q8/HFQ4/MQ4/HFQ6/MQ6/MQ3; MoE not admitted.
pub fn batch_weight_formats_supported(weights: &Lfm2MoeWeights) -> Result<(), String> {
    crate::lfm2moe::batch_weight_formats_supported(weights)
}

/// Single batched GEMM helper for dense projections.
/// Supports Q8_0, HFQ4G256 directly; MQ4G256 via FWHT rotation into caller-provided
/// scratch (must have capacity n*k). One GEMM per weight, no per-lane loop.
fn batched_proj(
    gpu: &mut Gpu,
    w: &hipfire_runtime::llama::WeightTensor,
    x: &rdna_compute::GpuTensor,
    y: &rdna_compute::GpuTensor,
    batch: usize,
    rot_scratch: &rdna_compute::GpuTensor,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(&w.buf, x, y, w.m, w.k, batch),
        DType::HFQ4G256 => gpu.gemm_hfq4g256(&w.buf, x, y, w.m, w.k, batch),
        DType::MQ4G256 => {
            let needed = batch.checked_mul(w.k).ok_or_else(|| {
                HipError::new(0, "batched_proj: batch*k overflow for MQ4 rotation")
            })?;
            if rot_scratch.numel() < needed {
                return Err(HipError::new(
                    0,
                    &format!(
                        "batched_proj: rot_scratch too small {} < {} (batch {batch} * k {})",
                        rot_scratch.numel(),
                        needed,
                        w.k
                    ),
                ));
            }
            let rot = rot_scratch.sub_offset(0, needed);
            hipfire_runtime::llama::rotate_x_mq_batched_for(gpu, w, x, &rot, w.k, batch)?;
            gpu.gemm_hfq4g256(&w.buf, &rot, y, w.m, w.k, batch)
        }
        other => Err(HipError::new(
            0,
            &format!("batched_proj: unsupported dtype {other:?} (expected Q8/HFQ4/MQ4)"),
        )),
    }
}

fn lm_head_batched_lfm(
    gpu: &mut Gpu,
    out_weight: &hipfire_runtime::llama::WeightTensor,
    hidden: &rdna_compute::GpuTensor,
    rot: &rdna_compute::GpuTensor,
    logits: &rdna_compute::GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    match out_weight.gpu_dtype {
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
            &out_weight.buf,
            hidden,
            logits,
            out_weight.m,
            out_weight.k,
            batch_size,
        ),
        DType::HFQ4G256 => gpu.gemm_hfq4g256_batched_lmhead(
            &out_weight.buf,
            hidden,
            logits,
            out_weight.m,
            out_weight.k,
            batch_size,
        ),
        DType::MQ4G256 => {
            hipfire_runtime::llama::rotate_x_mq_batched_for(
                gpu,
                out_weight,
                hidden,
                rot,
                out_weight.k,
                batch_size,
            )?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &out_weight.buf,
                rot,
                logits,
                out_weight.m,
                out_weight.k,
                batch_size,
            )
        }
        DType::HFQ6G256 => gpu.gemm_hfq6g256_batched_lmhead(
            &out_weight.buf,
            hidden,
            logits,
            out_weight.m,
            out_weight.k,
            batch_size,
        ),
        DType::MQ6G256 => {
            hipfire_runtime::llama::rotate_x_mq_batched_for(
                gpu,
                out_weight,
                hidden,
                rot,
                out_weight.k,
                batch_size,
            )?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &out_weight.buf,
                rot,
                logits,
                out_weight.m,
                out_weight.k,
                batch_size,
            )
        }
        DType::MQ3G256 => {
            hipfire_runtime::llama::rotate_x_mq_batched_for(
                gpu,
                out_weight,
                hidden,
                rot,
                out_weight.k,
                batch_size,
            )?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &out_weight.buf,
                rot,
                logits,
                out_weight.m,
                out_weight.k,
                batch_size,
            )
        }
        other => Err(HipError::new(
            0,
            &format!("lm_head_batched_lfm: unsupported dtype {other:?}"),
        )),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// public API
// ──────────────────────────────────────────────────────────────────────────

/// Populate batched embedding and position buffers. Returns the max position.
pub fn prepare_decode_batch_inputs_lfm(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    cfg: &crate::config::Lfm2MoeConfig,
    tokens: &[u32],
    positions: &[usize],
    state: &Lfm2DecodeBatchState,
) -> HipResult<usize> {
    let n = tokens.len();
    if n == 0 {
        return Err(HipError::new(
            0,
            "prepare_decode_batch_inputs_lfm: requires at least one lane",
        ));
    }
    if n > state.max_batch || positions.len() != n {
        return Err(HipError::new(
            0,
            "prepare_decode_batch_inputs_lfm: batch exceeds max_batch or positions length mismatch",
        ));
    }
    if positions.iter().any(|&p| p >= state.lane_capacity) {
        return Err(HipError::new(
            0,
            "prepare_decode_batch_inputs_lfm: position exceeds lane capacity",
        ));
    }
    check_dense(cfg)?;
    check_weight_formats(weights)?;

    // token ids -> device buffer [n] i32 stored as F32 slots (like Qwen)
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, tokens_host.len() * 4)
    };
    // copy only the active prefix; state.tokens is [max_batch] F32
    let tokens_active = state.tokens.sub_offset(0, n);
    gpu.hip.memcpy_htod(&tokens_active.buf, token_bytes)?;

    match weights.embd_format {
        hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
            let x_active = state.x_batch.sub_offset(0, n * cfg.hidden_size);
            gpu.embedding_lookup_hfq4g256_batched(
                &weights.embed,
                &x_active,
                &tokens_active,
                n,
                cfg.hidden_size,
            )?
        }
        hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
            let x_active = state.x_batch.sub_offset(0, n * cfg.hidden_size);
            gpu.embedding_lookup_q8_batched(
                &weights.embed,
                &x_active,
                &tokens_active,
                n,
                cfg.hidden_size,
            )?
        }
        _ => {
            return Err(HipError::new(
                0,
                "prepare_decode_batch_inputs_lfm: unsupported embedding format",
            ))
        }
    }

    let positions_host: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            positions_host.as_ptr() as *const u8,
            positions_host.len() * 4,
        )
    };
    let pos_active = state.positions.sub_offset(0, n);
    gpu.hip.memcpy_htod(&pos_active.buf, pos_bytes)?;

    Ok(positions.iter().copied().max().unwrap_or(0))
}

/// Batched decode: one launch per layer (no per-lane full-forward loop).
pub fn forward_decode_batch_lfm(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    cfg: &crate::config::Lfm2MoeConfig,
    tokens: &[u32],
    positions: &[usize],
    state: &mut Lfm2DecodeBatchState,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let position = prepare_decode_batch_inputs_lfm(gpu, weights, cfg, tokens, positions, state)?;
    forward_decode_batch_prepared_lfm(gpu, weights, cfg, tokens, positions, position, state)
}

/// Execute a batch whose embeddings/positions were already staged by
/// `prepare_decode_batch_inputs_lfm`. Validates that the staged buffers
/// match the declared shape.
pub fn forward_decode_batch_prepared_lfm(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    cfg: &crate::config::Lfm2MoeConfig,
    tokens: &[u32],
    positions: &[usize],
    position: usize,
    state: &mut Lfm2DecodeBatchState,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0
        || n > state.max_batch
        || positions.len() != n
        || positions.iter().any(|&p| p >= state.lane_capacity)
        || position != positions.iter().copied().max().unwrap_or(0)
    {
        return Err(HipError::new(
            0,
            "forward_decode_batch_prepared_lfm: prepared inputs do not match admitted shape",
        ));
    }
    check_dense(cfg)?;
    check_weight_formats(weights)?;

    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let dense_inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;

    // Active windows into the batched buffers (dense packing 0..n)
    let x = state.x_batch.sub_offset(0, n * hidden);
    let tmp = state.tmp_batch.sub_offset(0, n * hidden);
    let bcx = state.conv_bcx_batch.sub_offset(0, n * 3 * hidden);
    let y = state.conv_y_batch.sub_offset(0, n * hidden);
    let fq = state.fa_q_batch.sub_offset(0, n * q_dim);
    let fk = state.fa_k_batch.sub_offset(0, n * kv_dim);
    let fv = state.fa_v_batch.sub_offset(0, n * kv_dim);
    let fa_out = state.fa_attn_out_batch.sub_offset(0, n * q_dim);
    let ffn_tmp = state.ffn_tmp_batch.sub_offset(0, n * hidden);
    let gate = state.dense_gate_batch.sub_offset(0, n * dense_inter);
    let up = state.dense_up_batch.sub_offset(0, n * dense_inter);
    let act = state.dense_act_batch.sub_offset(0, n * dense_inter);
    let pos = state.positions.sub_offset(0, n);

    // x already holds embeddings after prepare. Now per-layer batched stack.
    for (l, layer) in weights.layers.iter().enumerate() {
        // pre-norm
        gpu.rmsnorm_batched(&x, &layer.operator_norm, &tmp, n, hidden, eps)
            .map_err(|e| HipError::new(0, &format!("lfm batch L{l} operator rmsnorm: {e:?}")))?;
        match &layer.mixer {
            crate::lfm2moe::Mixer::Conv(c) => {
                // in_proj batched: tmp [n, hidden] -> bcx [n, 3*hidden]
                batched_proj(gpu, &c.in_proj, &tmp, &bcx, n, &state.proj_rot).map_err(|e| {
                    HipError::new(0, &format!("lfm batch L{l} conv in_proj: {e:?}"))
                })?;

                // single batched conv launch with B=n
                let cs = &state.conv_states[c.conv_state_idx];
                let hist = hidden * (cfg.conv_kernel_size - 1);
                let cs_active = cs.sub_offset(0, n * hist);
                gpu.conv1d_gated_decode_f32(
                    &bcx,
                    &cs_active,
                    &c.conv_weight,
                    &y,
                    n,
                    hidden,
                    cfg.conv_kernel_size,
                )
                .map_err(|e| {
                    HipError::new(0, &format!("lfm batch L{l} conv gated decode: {e:?}"))
                })?;

                // out_proj + residual: y -> x   (one batched GEMM then add)
                // Use fa_out as temp for out_proj output (reusing attention scratch, not live in conv layer)
                let tmp_out = fa_out.sub_offset(0, n * hidden);
                batched_proj(gpu, &c.out_proj, &y, &tmp_out, n, &state.proj_rot).map_err(|e| {
                    HipError::new(0, &format!("lfm batch L{l} conv out_proj: {e:?}"))
                })?;
                gpu.add_inplace_f32(&x, &tmp_out).map_err(|e| {
                    HipError::new(0, &format!("lfm batch L{l} conv residual add: {e:?}"))
                })?;
            }
            crate::lfm2moe::Mixer::Attention(a) => {
                // q/k/v projections batched (Q8/HFQ4/MQ4 via one GEMM per weight)
                batched_proj(gpu, &a.wq, &tmp, &fq, n, &state.proj_rot)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} q_proj: {e:?}")))?;
                batched_proj(gpu, &a.wk, &tmp, &fk, n, &state.proj_rot)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} k_proj: {e:?}")))?;
                batched_proj(gpu, &a.wv, &tmp, &fv, n, &state.proj_rot)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} v_proj: {e:?}")))?;
                // per-head qk norm: batch = n * n_heads vectors
                gpu.rmsnorm_batched(
                    &fq,
                    &a.q_norm,
                    &fq,
                    n * cfg.num_attention_heads,
                    cfg.head_dim,
                    eps,
                )
                .map_err(|e| HipError::new(0, &format!("lfm batch L{l} q_norm: {e:?}")))?;
                gpu.rmsnorm_batched(
                    &fk,
                    &a.k_norm,
                    &fk,
                    n * cfg.num_key_value_heads,
                    cfg.head_dim,
                    eps,
                )
                .map_err(|e| HipError::new(0, &format!("lfm batch L{l} k_norm: {e:?}")))?;

                // rope batched
                gpu.rope_batched_f32(
                    &fq,
                    &fk,
                    &pos,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.rope_theta,
                    n,
                )
                .map_err(|e| HipError::new(0, &format!("lfm batch L{l} rope: {e:?}")))?;

                // kv write + attention independent (lane-isolated)
                let kv_idx = a.kv_idx;
                gpu.kv_cache_write_q8_0_independent(
                    &state.kv.k_gpu[kv_idx],
                    &fk,
                    &pos,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    n,
                    state.lane_capacity,
                )
                .map_err(|e| HipError::new(0, &format!("lfm batch L{l} kv write k: {e:?}")))?;
                gpu.kv_cache_write_q8_0_independent(
                    &state.kv.v_gpu[kv_idx],
                    &fv,
                    &pos,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    n,
                    state.lane_capacity,
                )
                .map_err(|e| HipError::new(0, &format!("lfm batch L{l} kv write v: {e:?}")))?;

                gpu.attention_q8_0_kv_independent(
                    &fq,
                    &state.kv.k_gpu[kv_idx],
                    &state.kv.v_gpu[kv_idx],
                    &fa_out,
                    &pos,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    state.lane_capacity,
                    state.lane_capacity,
                    n,
                )
                .map_err(|e| HipError::new(0, &format!("lfm batch L{l} attention: {e:?}")))?;

                // out_proj + residual
                let tmp_out = state.conv_y_batch.sub_offset(0, n * hidden);
                batched_proj(gpu, &a.wo, &fa_out, &tmp_out, n, &state.proj_rot)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} out_proj: {e:?}")))?;
                gpu.add_inplace_f32(&x, &tmp_out).map_err(|e| {
                    HipError::new(0, &format!("lfm batch L{l} attn residual: {e:?}"))
                })?;
            }
        }

        // FFN block (dense SwiGLU, pre-norm)
        gpu.rmsnorm_batched(&x, &layer.ffn_norm, &ffn_tmp, n, hidden, eps)
            .map_err(|e| HipError::new(0, &format!("lfm batch L{l} ffn rmsnorm: {e:?}")))?;
        match &layer.ffn {
            crate::lfm2moe::Ffn::Dense(d) => {
                batched_proj(gpu, &d.w1, &ffn_tmp, &gate, n, &state.proj_rot)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} dense w1: {e:?}")))?;
                batched_proj(gpu, &d.w3, &ffn_tmp, &up, n, &state.proj_rot)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} dense w3: {e:?}")))?;
                gpu.silu_mul_f32(&gate, &up, &act)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} silu_mul: {e:?}")))?;
                let tmp_out = fa_out.sub_offset(0, n * hidden);
                batched_proj(gpu, &d.w2, &act, &tmp_out, n, &state.proj_rot)
                    .map_err(|e| HipError::new(0, &format!("lfm batch L{l} dense w2: {e:?}")))?;
                gpu.add_inplace_f32(&x, &tmp_out).map_err(|e| {
                    HipError::new(0, &format!("lfm batch L{l} ffn residual: {e:?}"))
                })?;
            }
            crate::lfm2moe::Ffn::Moe(_) => {
                return Err(HipError::new(
                    0,
                    "forward_decode_batch_prepared_lfm: MoE FFN not supported in dense batch",
                ))
            }
        }
    }

    // final norm + lm_head batched
    let final_hidden = state.final_hidden.sub_offset(0, n * hidden);
    gpu.rmsnorm_batched(&x, &weights.embedding_norm, &final_hidden, n, hidden, eps)
        .map_err(|e| HipError::new(0, &format!("lfm batch final rmsnorm: {e:?}")))?;
    let logits = state.logits.sub_offset(0, n * cfg.vocab_size);
    let rot = state.lm_rot.sub_offset(0, n * hidden);
    lm_head_batched_lfm(gpu, &weights.lm_head, &final_hidden, &rot, &logits, n)
        .map_err(|e| HipError::new(0, &format!("lfm batch lm_head: {e:?}")))?;

    Ok(())
}
