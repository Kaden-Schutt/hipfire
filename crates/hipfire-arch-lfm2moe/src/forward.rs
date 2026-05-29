// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-MoE forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (pre-norm; mixer = conv OR attention, FFN = dense OR MoE):
//!   tmp = operator_norm(h)
//!   if conv:   h += out_proj( C_gate ⊙ depthwise_causal_conv( B_gate ⊙ x ) )   [in_proj→conv→out_proj]
//!   if attn:   h += out_proj( attn( qk_norm(q/k) + full-RoPE, v ) )             [GQA, Q8 KV]
//!   ffn_tmp = ffn_norm(h)
//!   if dense:  h += w2( silu(w1·ffn_tmp) ⊙ (w3·ffn_tmp) )                        [SwiGLU, Q8]
//!   if moe:    h += combine( experts( sigmoid+bias top-4 route(ffn_tmp) ) )      [FWHT MQ4 experts]
//! then logits = lm_head( embedding_norm(h) )   (lm_head tied to embed_tokens).
//!
//! Non-expert linears (attention q/k/v/out, conv in/out, dense w1/w2/w3, router)
//! are Q8 (plain input). Routed experts are FWHT-pre-rotated MQ4G256: the input
//! is rotated (`rotate_x_mq_for`) and the silu output rotated
//! (`fused_silu_mul_rotate_mq_batched_for`) before the indexed-MoE GEMVs —
//! exactly qwen35's / minimax's MoE path, but with k_top = num_experts_per_tok
//! = 4 (the batched GEMV variants take k_top as a runtime arg).

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Ffn, Lfm2MoeState, Lfm2MoeWeights, Mixer};
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_for, weight_gemv, weight_gemv_residual,
};
use rdna_compute::Gpu;

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state
/// (after the full layer, before the final norm) to `capture[layer]` — used by
/// the oracle dumper. Set `HIPFIRE_LFM2_CAPTURE_POSTMIXER` to capture the
/// post-mixer residual (pre-FFN) instead, for conv/attn-vs-FFN localization.
pub fn decode_step_capture(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, Some(capture))
}

fn decode_step_inner(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let capture_postmixer = std::env::var_os("HIPFIRE_LFM2_CAPTURE_POSTMIXER").is_some();

    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;

    // Embedding lookup → residual stream h (Q8 table).
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm2moe: embed lookup: {e:?}"))?;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Mixer block (pre-norm) ──────────────────────────────────────────
        gpu.rmsnorm_f32(&state.h, &layer.operator_norm, &state.tmp, eps)
            .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;

        match &layer.mixer {
            Mixer::Conv(c) => {
                // in_proj → [3*hidden] (B | C_gate | x), Q8 plain.
                weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
                    .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
                // double-gated depthwise causal short-conv (advances conv state).
                gpu.conv1d_gated_decode_f32(
                    &state.conv_bcx,
                    &state.conv_states[c.conv_state_idx],
                    &c.conv_weight,
                    &state.conv_y,
                    1,
                    hidden,
                    cfg.conv_kernel_size,
                )
                .map_err(|e| format!("lfm2moe L{l}: conv gated decode: {e:?}"))?;
                // out_proj + residual: h += W_out · y (Q8).
                weight_gemv_residual(gpu, &c.out_proj, &state.conv_y, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))?;
            }
            Mixer::Attention(a) => {
                weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
                    .map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
                weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
                    .map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
                weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
                    .map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;

                // Per-HEAD QK-norm: RMSNorm over each head's head_dim slice,
                // sharing the [head_dim] weight across heads (batch = n_heads).
                gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
                    .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
                gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
                    .map_err(|e| format!("lfm2moe L{l}: k_norm: {e:?}"))?;

                // Full-dim rotate_half RoPE (no partial rotary).
                gpu.rope_f32(
                    &state.fa_q,
                    &state.fa_k,
                    &state.pos_buf,
                    n_heads,
                    n_kv,
                    head_dim,
                    cfg.rope_theta,
                )
                .map_err(|e| format!("lfm2moe L{l}: rope: {e:?}"))?;

                // KV cache write (Q8) + GQA flash attention.
                let kv_idx = a.kv_idx;
                gpu.kv_cache_write_q8_0(
                    &state.kv.k_gpu[kv_idx],
                    &state.fa_k,
                    &state.pos_buf,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe L{l}: kv write k: {e:?}"))?;
                gpu.kv_cache_write_q8_0(
                    &state.kv.v_gpu[kv_idx],
                    &state.fa_v,
                    &state.pos_buf,
                    n_kv,
                    head_dim,
                )
                .map_err(|e| format!("lfm2moe L{l}: kv write v: {e:?}"))?;
                gpu.attention_q8_0_kv(
                    &state.fa_q,
                    &state.kv.k_gpu[kv_idx],
                    &state.kv.v_gpu[kv_idx],
                    &state.fa_attn_out,
                    &state.pos_buf,
                    seq_len,
                    n_heads,
                    n_kv,
                    head_dim,
                    state.kv.physical_cap,
                )
                .map_err(|e| format!("lfm2moe L{l}: attention: {e:?}"))?;

                // out_proj + residual: h += W_out · attn_out (Q8).
                weight_gemv_residual(gpu, &a.wo, &state.fa_attn_out, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: out_proj: {e}"))?;
            }
        }

        if capture_postmixer {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("lfm2moe L{l}: postmixer capture: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }

        // ── FFN block (pre-norm): dense SwiGLU OR top-4 MoE ─────────────────
        gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
            .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;

        match &layer.ffn {
            Ffn::Dense(d) => {
                weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
                    .map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
                weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
                    .map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))?;
                gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
                    .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
                weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
                    .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))?;
            }
            Ffn::Moe(m) => {
                // FWHT-rotate the FFN input for the MQ4 experts (router stays plain).
                rotate_x_mq_for(gpu, &m.experts[0].gate_up, &state.ffn_tmp, &state.ffn_x_rot, hidden)
                    .map_err(|e| format!("lfm2moe L{l}: ffn rotate: {e:?}"))?;

                // Router: sigmoid(logits) + bias-aware top-k (gather unbiased,
                // renormalize, scale). expert_bias steers SELECTION only.
                weight_gemv(gpu, &m.router, &state.ffn_tmp, &state.router_logits)
                    .map_err(|e| format!("lfm2moe L{l}: router: {e}"))?;
                gpu.sigmoid_f32(&state.router_logits)
                    .map_err(|e| format!("lfm2moe L{l}: sigmoid: {e:?}"))?;
                gpu.deepseek4_moe_topk_bias_aware_f32(
                    &state.router_logits,
                    &m.expert_bias,
                    &state.topk_indices,
                    &state.topk_weights,
                    n_exp as i32,
                    k_top as i32,
                    cfg.routed_scaling_factor,
                )
                .map_err(|e| format!("lfm2moe L{l}: topk: {e:?}"))?;

                // gate_up (rotated input, batched k_top) → silu·mul·rotate → down → combine.
                gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                    &m.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * moe_inter,
                    hidden,
                    k_top,
                    1,
                )
                .map_err(|e| format!("lfm2moe L{l}: gate_up: {e:?}"))?;

                fused_silu_mul_rotate_mq_batched_for(
                    gpu,
                    &m.experts[0].down,
                    &state.gate_batch,
                    &state.up_batch,
                    &state.rot_batch,
                    moe_inter,
                    k_top,
                )
                .map_err(|e| format!("lfm2moe L{l}: silu_mul_rotate: {e:?}"))?;

                gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    &m.expert_down_ptrs,
                    &state.topk_indices,
                    &state.rot_batch,
                    &state.down_expanded,
                    hidden,
                    moe_inter,
                    k_top,
                    1,
                )
                .map_err(|e| format!("lfm2moe L{l}: down: {e:?}"))?;

                gpu.moe_down_combine_k8_batched(
                    &state.down_expanded,
                    &state.topk_weights,
                    &state.h,
                    hidden,
                    k_top,
                    1,
                )
                .map_err(|e| format!("lfm2moe L{l}: combine: {e:?}"))?;
            }
        }

        // Capture post-layer residual (pre final-norm) for the oracle compare.
        if !capture_postmixer {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("lfm2moe L{l}: capture download: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (tied to embed_tokens, Q8).
    gpu.rmsnorm_f32(&state.h, &weights.embedding_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    Ok(())
}
