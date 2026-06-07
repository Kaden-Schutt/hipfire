// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2Moe forward pass (free functions — hot-path static dispatch).
//!
//! **Parallel decoder block** (the Cohere distinctive — every existing hipfire
//! arch is sequential): a SINGLE `input_layernorm` feeds both attention and the
//! MLP, and BOTH are summed into the residual:
//!
//! ```text
//!   n  = rmsnorm(h, input_layernorm)          // computed ONCE per layer
//!   h += o_proj( attn( q/k/v_proj(n) + interleaved-RoPE ) )   [GQA, Q8 KV]
//!   h += dense_or_moe( n )                     // reads the SAME n, not a re-norm
//! ```
//!
//! Both sub-blocks read `n` (kept in `state.tmp`); attention accumulates its
//! o_proj into `h` first, then the FFN accumulates into `h` — order-independent
//! because neither reads `h` for its input (only `n`). Layer 0 (the
//! `first_k_dense_replace` prefix) is a dense SwiGLU MLP @ `dense_intermediate`;
//! layers 1.. are 128-expert top-8 MoE @ `moe_intermediate`.
//!
//! Conventions verified against `modeling_cohere2_moe.py` (2026-06-07):
//!   * RMSNorm (standard, `weight * x̂`, no +1) — `rms_norm_eps` set in config.
//!   * RoPE is INTERLEAVED over the FULL head_dim → `rope_partial_interleaved_f32`
//!     with `rotary_dim = head_dim` (the kernel rotates pairs (2i, 2i+1), which
//!     is exactly Cohere's `repeat_interleave(freqs, 2)` convention).
//!   * MoE routing = `sigmoid` selection, NO routing-bias term, and
//!     `norm_topk_prob = false` (NO top-k weight renormalization).
//!   * `logit_scale` = 1.0 (no-op for this checkpoint).
//!
//! ── ORACLE-LOOP TODOs (per docs/methodology/arch-port-validation.md) ──
//!  [T1] RESOLVED (2026-06-07): routing uses `moe_topk_renorm_k8` with
//!       `norm_topk = cfg.norm_topk_prob` (false), giving un-renormalized
//!       sigmoid weights and no bias term — the Cohere2Moe convention. The
//!       earlier renormalizing kernel cost ~0.012 layer-1 cosine.
//!  [T2] Sliding-window attention: all layers run FULL causal attention here.
//!       Correct for prompts < `sliding_window` (4096). Add windowed KV for
//!       long-context once forward-correctness on short prompts is proven.
//!  [T3] Interleaved-vs-half RoPE: if the oracle shows attention divergence,
//!       the Q/K projection layout may need a load-time permute to match the
//!       interleaved kernel (the llama.cpp permute trick). Resolve by reading
//!       the rope kernel + the reference, not by guessing.

use crate::cohere2moe::{CohereState, CohereWeights, Ffn};
use crate::config::Cohere2MoeConfig;
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_for, weight_gemv, weight_gemv_residual,
};
use rdna_compute::{DType, Gpu};

/// Decode one token (eager); returns the full logits vector.
pub fn decode_step(
    cfg: &Cohere2MoeConfig,
    weights: &CohereWeights,
    state: &mut CohereState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("cohere2moe: embed lookup: {e:?}"))?;
    decode_step_body(cfg, weights, state, gpu, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("cohere2moe: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state (after
/// the full parallel block, before the final norm) to `capture[layer]` — used by
/// the oracle dumper. Set `HIPFIRE_COHERE_CAPTURE_POSTATTN` to capture the
/// post-attention residual (after `h += o_proj(attn)`, before the FFN) instead,
/// for attention-vs-FFN divergence localization.
pub fn decode_step_capture(
    cfg: &Cohere2MoeConfig,
    weights: &CohereWeights,
    state: &mut CohereState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("cohere2moe: embed lookup: {e:?}"))?;
    decode_step_body(cfg, weights, state, gpu, position, Some(capture))
}

fn decode_step_body(
    cfg: &Cohere2MoeConfig,
    weights: &CohereWeights,
    state: &mut CohereState,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let capture_postattn = std::env::var_os("HIPFIRE_COHERE_CAPTURE_POSTATTN").is_some();

    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("cohere2moe: htod pos: {e:?}"))?;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Shared input norm (parallel block) ──────────────────────────────
        // n = rmsnorm(h); BOTH attention and the FFN consume `n` (state.tmp).
        gpu.rmsnorm_f32(&state.h, &layer.input_norm, &state.tmp, eps)
            .map_err(|e| format!("cohere2moe L{l}: input rmsnorm: {e:?}"))?;

        // ── Attention (reads n) → h += o_proj(attn) ─────────────────────────
        weight_gemv(gpu, &layer.wq, &state.tmp, &state.fa_q)
            .map_err(|e| format!("cohere2moe L{l}: q_proj: {e}"))?;
        weight_gemv(gpu, &layer.wk, &state.tmp, &state.fa_k)
            .map_err(|e| format!("cohere2moe L{l}: k_proj: {e}"))?;
        weight_gemv(gpu, &layer.wv, &state.tmp, &state.fa_v)
            .map_err(|e| format!("cohere2moe L{l}: v_proj: {e}"))?;

        // Interleaved RoPE over the FULL head_dim (rotary_dim = head_dim).
        gpu.rope_partial_interleaved_f32(
            &state.fa_q,
            &state.fa_k,
            &state.pos_buf,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.head_dim, // full-dim rotation
            cfg.rope_theta,
        )
        .map_err(|e| format!("cohere2moe L{l}: rope: {e:?}"))?;

        gpu.kv_cache_write_q8_0(
            &state.kv.k_gpu[l],
            &state.fa_k,
            &state.pos_buf,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("cohere2moe L{l}: kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0(
            &state.kv.v_gpu[l],
            &state.fa_v,
            &state.pos_buf,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("cohere2moe L{l}: kv write v: {e:?}"))?;
        // [T2] Full causal attention for every layer (SWA deferred). The kernel
        // reads the live KV length from pos_buf[0]+1.
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
        .map_err(|e| format!("cohere2moe L{l}: attention: {e:?}"))?;

        // h += W_o · attn_out.
        weight_gemv_residual(gpu, &layer.wo, &state.fa_attn_out, &state.h)
            .map_err(|e| format!("cohere2moe L{l}: o_proj: {e}"))?;

        if capture_postattn {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("cohere2moe L{l}: postattn capture: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }

        // ── FFN (reads the SAME n = state.tmp) → h += dense_or_moe(n) ────────
        match &layer.ffn {
            Ffn::Dense(d) => {
                // SwiGLU: w2( silu(w1·n) ⊙ (w3·n) ). Reads `n` (state.tmp).
                weight_gemv(gpu, &d.w1, &state.tmp, &state.dense_gate)
                    .map_err(|e| format!("cohere2moe L{l}: dense w1: {e}"))?;
                weight_gemv(gpu, &d.w3, &state.tmp, &state.dense_up)
                    .map_err(|e| format!("cohere2moe L{l}: dense w3: {e}"))?;
                gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
                    .map_err(|e| format!("cohere2moe L{l}: dense silu_mul: {e:?}"))?;
                weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
                    .map_err(|e| format!("cohere2moe L{l}: dense w2: {e}"))?;
            }
            Ffn::Moe(m) => {
                // FWHT-rotate the FFN input `n` for the MQ4 experts (router plain).
                rotate_x_mq_for(gpu, &m.experts[0].gate_up, &state.tmp, &state.ffn_x_rot, hidden)
                    .map_err(|e| format!("cohere2moe L{l}: ffn rotate: {e:?}"))?;

                // Router: Cohere2Moe = sigmoid activation, select top-k by score,
                // weight = sigmoid(selected logit). NO routing bias, and
                // norm_topk_prob = false → NO top-k renormalization. sigmoid is
                // monotonic, so top-k by sigmoid == top-k by logit (matching the
                // reference, which top-ks on logits then applies sigmoid).
                weight_gemv(gpu, &m.router, &state.tmp, &state.router_logits)
                    .map_err(|e| format!("cohere2moe L{l}: router: {e}"))?;
                gpu.sigmoid_f32(&state.router_logits)
                    .map_err(|e| format!("cohere2moe L{l}: sigmoid: {e:?}"))?;
                gpu.moe_topk_renorm_k8(
                    &state.router_logits,
                    &state.topk_indices,
                    &state.topk_weights,
                    n_exp,
                    cfg.norm_topk_prob, // false for Cohere2Moe
                )
                .map_err(|e| format!("cohere2moe L{l}: topk: {e:?}"))?;

                // gate_up (rotated input) → silu·mul·rotate → down → combine.
                let edt = m.experts[0].gate_up.gpu_dtype;
                match edt {
                    DType::MQ4G256 | DType::HFQ4G256 => gpu
                        .gemv_hfq4g256_moe_gate_up_k8_indexed(
                            &m.expert_gate_up_ptrs,
                            &state.topk_indices,
                            &state.ffn_x_rot,
                            &state.gate_batch,
                            &state.up_batch,
                            2 * moe_inter,
                            hidden,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: gate_up hfq4: {e:?}"))?,
                    DType::MQ6G256 | DType::HFQ6G256 => gpu
                        .gemv_hfq6g256_moe_gate_up_k8_indexed(
                            &m.expert_gate_up_ptrs,
                            &state.topk_indices,
                            &state.ffn_x_rot,
                            &state.gate_batch,
                            &state.up_batch,
                            2 * moe_inter,
                            hidden,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: gate_up hfq6: {e:?}"))?,
                    other => {
                        return Err(format!("cohere2moe L{l}: unsupported expert dtype {other:?}"))
                    }
                }

                fused_silu_mul_rotate_mq_batched_for(
                    gpu,
                    &m.experts[0].down,
                    &state.gate_batch,
                    &state.up_batch,
                    &state.rot_batch,
                    moe_inter,
                    k_top,
                )
                .map_err(|e| format!("cohere2moe L{l}: silu_mul_rotate: {e:?}"))?;

                let ddt = m.experts[0].down.gpu_dtype;
                match ddt {
                    DType::MQ4G256 | DType::HFQ4G256 => {
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
                        .map_err(|e| format!("cohere2moe L{l}: down hfq4: {e:?}"))?;
                        gpu.moe_down_combine_k8_batched(
                            &state.down_expanded,
                            &state.topk_weights,
                            &state.h,
                            hidden,
                            k_top,
                            1,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: combine: {e:?}"))?;
                    }
                    DType::MQ6G256 | DType::HFQ6G256 => {
                        gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                            &m.expert_down_ptrs,
                            &state.topk_indices,
                            &state.rot_batch,
                            &state.down_expanded,
                            hidden,
                            moe_inter,
                            k_top,
                            1,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: down hfq6: {e:?}"))?;
                        gpu.moe_down_combine_k8_batched(
                            &state.down_expanded,
                            &state.topk_weights,
                            &state.h,
                            hidden,
                            k_top,
                            1,
                        )
                        .map_err(|e| format!("cohere2moe L{l}: combine: {e:?}"))?;
                    }
                    other => {
                        return Err(format!("cohere2moe L{l}: unsupported down dtype {other:?}"))
                    }
                }
            }
        }

        // Capture post-layer residual (pre final-norm) for the oracle compare.
        if !capture_postattn {
            if let Some(cap) = capture.as_deref_mut() {
                let h = gpu
                    .download_f32(&state.h)
                    .map_err(|e| format!("cohere2moe L{l}: capture download: {e:?}"))?;
                cap[l].extend_from_slice(&h);
            }
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (tied to embed_tokens). logit_scale = 1.0 → no-op;
    // wire a scalar-mul here if a future Cohere2Moe checkpoint sets it != 1.
    gpu.rmsnorm_f32(&state.h, &weights.final_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("cohere2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("cohere2moe: lm_head: {e}"))?;
    Ok(())
}
