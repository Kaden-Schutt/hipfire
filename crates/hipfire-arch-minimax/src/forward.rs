// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (validated vs the HF `MiniMaxM2` modeling oracle to
//! cosine 0.9996):
//!   h += o_proj · attn( qk_norm(q/k/v_proj(rmsnorm(h))) + partial-RoPE )   [GQA, Q8 KV]
//!   h += combine( experts( sigmoid+bias top-8 route( rmsnorm(h) ) ) )       [MoE]
//! then logits = lm_head( rmsnorm(h) ).
//!
//! Attention weights are Q8 (plain input). The router is Q8 (plain). Routed
//! experts are FWHT-pre-rotated (MQ4G256 / MQ2G256Lloyd / MQ6G256): the input
//! is rotated (`rotate_x_mq_for`) and the silu output rotated
//! (`fused_silu_mul_rotate_mq_batched_for`) before the indexed-MoE GEMV kernels
//! — exactly qwen35's / deepseek4's MoE path. Routing uses `sigmoid_f32` +
//! `deepseek4_moe_topk_bias_aware_f32` with route_scale = 1.0 (MiniMax-M2
//! applies no routed-scaling factor).

use crate::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_for, weight_gemv, weight_gemv_residual,
};
use rdna_compute::{DType, Gpu};

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("minimax: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state
/// (pre final-norm) to `capture[layer]` — used by the oracle dumper. Set
/// `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` to capture the post-attention residual
/// (pre-MoE) instead, for attention-vs-MoE divergence localization.
pub fn decode_step_capture(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, Some(capture))
}

fn decode_step_inner(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let capture_postattn = std::env::var_os("HIPFIRE_MINIMAX_CAPTURE_POSTATTN").is_some();

    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("minimax: htod pos: {e:?}"))?;

    // Embedding lookup → residual stream h.
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("minimax: embed lookup: {e:?}"))?;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Attention block (Q8 projections → plain input) ──────────────────
        gpu.rmsnorm_f32(&state.h, &layer.attn_norm, &state.tmp, eps)
            .map_err(|e| format!("minimax L{l}: attn rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &layer.wq, &state.tmp, &state.fa_q)
            .map_err(|e| format!("minimax L{l}: q_proj: {e}"))?;
        weight_gemv(gpu, &layer.wk, &state.tmp, &state.fa_k)
            .map_err(|e| format!("minimax L{l}: k_proj: {e}"))?;
        weight_gemv(gpu, &layer.wv, &state.tmp, &state.fa_v)
            .map_err(|e| format!("minimax L{l}: v_proj: {e}"))?;

        // Per-LAYER QK-norm: RMSNorm over the whole flat q[q_dim]/k[kv_dim]
        // vector (batch=1), BEFORE head reshape.
        if cfg.use_qk_norm {
            gpu.rmsnorm_batched(&state.fa_q, &layer.q_norm, &state.fa_q, 1, q_dim, eps)
                .map_err(|e| format!("minimax L{l}: q_norm: {e:?}"))?;
            gpu.rmsnorm_batched(&state.fa_k, &layer.k_norm, &state.fa_k, 1, kv_dim, eps)
                .map_err(|e| format!("minimax L{l}: k_norm: {e:?}"))?;
        }

        // Partial rotate_half RoPE on the first `rotary_dim` of each head.
        gpu.rope_partial_interleaved_f32(
            &state.fa_q,
            &state.fa_k,
            &state.pos_buf,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
        )
        .map_err(|e| format!("minimax L{l}: rope: {e:?}"))?;

        // KV cache write (Q8) + GQA attention.
        gpu.kv_cache_write_q8_0(
            &state.kv.k_gpu[l],
            &state.fa_k,
            &state.pos_buf,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("minimax L{l}: kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0(
            &state.kv.v_gpu[l],
            &state.fa_v,
            &state.pos_buf,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("minimax L{l}: kv write v: {e:?}"))?;
        gpu.attention_q8_0_kv(
            &state.fa_q,
            &state.kv.k_gpu[l],
            &state.kv.v_gpu[l],
            &state.fa_attn_out,
            &state.pos_buf,
            seq_len,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            state.kv.physical_cap,
        )
        .map_err(|e| format!("minimax L{l}: attention: {e:?}"))?;

        // o_proj + residual: h += W_o · attn_out.
        weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
            .map_err(|e| format!("minimax L{l}: o_proj: {e}"))?;

        if capture_postattn {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("minimax L{l}: postattn capture: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }

        // ── MoE block (no shared expert) ────────────────────────────────────
        // ffn_tmp = rmsnorm(h) (plain, feeds the Q8 router); ffn_x_rot =
        // FWHT(ffn_tmp) (feeds the FWHT-pre-rotated experts).
        gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
            .map_err(|e| format!("minimax L{l}: ffn rmsnorm: {e:?}"))?;
        rotate_x_mq_for(
            gpu,
            &layer.experts[0].gate_up,
            &state.ffn_tmp,
            &state.ffn_x_rot,
            hidden,
        )
        .map_err(|e| format!("minimax L{l}: ffn rotate: {e:?}"))?;

        // Router: sigmoid(logits) + bias-aware top-k (gather unbiased + normalize;
        // route_scale = 1.0 — MiniMax-M2 applies no routed-scaling factor).
        weight_gemv(gpu, &layer.router, &state.ffn_tmp, &state.router_logits)
            .map_err(|e| format!("minimax L{l}: router: {e}"))?;
        gpu.sigmoid_f32(&state.router_logits)
            .map_err(|e| format!("minimax L{l}: sigmoid: {e:?}"))?;
        gpu.deepseek4_moe_topk_bias_aware_f32(
            &state.router_logits,
            &layer.routing_bias,
            &state.topk_indices,
            &state.topk_weights,
            n_exp as i32,
            k_top as i32,
            1.0,
        )
        .map_err(|e| format!("minimax L{l}: topk: {e:?}"))?;

        // Routed experts: gate_up (rotated input) → silu·mul·rotate → down → combine.
        // Dispatch the indexed-MoE GEMV by expert dtype. MQ4/MQ6/MQ2-Lloyd are
        // FWHT-pre-rotated (byte-compatible with the matching hfq/lloyd kernels
        // given rotated input). The hfq4/hfq6 family uses a separate down +
        // `moe_down_combine`; the MQ2-Lloyd down is residual-scaled (fuses the
        // weighted combine into the down GEMV, accumulating into h directly).
        let edt = layer.experts[0].gate_up.gpu_dtype;
        match edt {
            DType::MQ4G256 | DType::HFQ4G256 => gpu
                .gemv_hfq4g256_moe_gate_up_k8_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                )
                .map_err(|e| format!("minimax L{l}: gate_up hfq4: {e:?}"))?,
            DType::MQ6G256 | DType::HFQ6G256 => gpu
                .gemv_hfq6g256_moe_gate_up_k8_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                )
                .map_err(|e| format!("minimax L{l}: gate_up hfq6: {e:?}"))?,
            DType::MQ2G256Lloyd => gpu
                .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: gate_up mq2l: {e:?}"))?,
            DType::MQ3G256Lloyd => gpu
                .deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
                    &layer.expert_gate_up_ptrs,
                    &state.topk_indices,
                    &state.ffn_x_rot,
                    &state.gate_batch,
                    &state.up_batch,
                    2 * inter,
                    hidden,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: gate_up mq3l: {e:?}"))?,
            other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
        }

        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &layer.experts[0].down,
            &state.gate_batch,
            &state.up_batch,
            &state.rot_batch,
            inter,
            k_top,
        )
        .map_err(|e| format!("minimax L{l}: silu_mul_rotate: {e:?}"))?;

        // Down dispatches on the DOWN proj's own dtype (may differ from gate_up:
        // e.g. gate_up=mq2-lloyd + down=mq4, since down carries ~24x the energy).
        let ddt = layer.experts[0].down.gpu_dtype;
        match ddt {
            DType::MQ4G256 | DType::HFQ4G256 => {
                gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.rot_batch,
                    &state.down_expanded,
                    hidden,
                    inter,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: down hfq4: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(
                    &state.down_expanded,
                    &state.topk_weights,
                    &state.h,
                    hidden,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
            }
            DType::MQ6G256 | DType::HFQ6G256 => {
                gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.rot_batch,
                    &state.down_expanded,
                    hidden,
                    inter,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: down hfq6: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(
                    &state.down_expanded,
                    &state.topk_weights,
                    &state.h,
                    hidden,
                    k_top,
                    1,
                )
                .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
            }
            DType::MQ2G256Lloyd => {
                // Fused down + weighted residual accumulate (no separate combine).
                gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.topk_weights,
                    &state.rot_batch,
                    &state.h,
                    hidden,
                    inter,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: down mq2l: {e:?}"))?;
            }
            DType::MQ3G256Lloyd => {
                // Fused down + weighted residual accumulate (no separate combine).
                gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
                    &layer.expert_down_ptrs,
                    &state.topk_indices,
                    &state.topk_weights,
                    &state.rot_batch,
                    &state.h,
                    hidden,
                    inter,
                    k_top,
                )
                .map_err(|e| format!("minimax L{l}: down mq3l: {e:?}"))?;
            }
            other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
        }

        // Capture post-layer residual (pre final-norm) for the oracle compare.
        if !capture_postattn {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("minimax L{l}: capture download: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (Q8 → plain).
    gpu.rmsnorm_f32(&state.h, &weights.final_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("minimax: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("minimax: lm_head: {e}"))?;
    Ok(())
}
