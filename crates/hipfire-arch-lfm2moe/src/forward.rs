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
use crate::lfm2moe::{
    AttnWeights, ConvWeights, DenseFfn, Ffn, Lfm2MoeLayerWeights, Lfm2MoeState, Lfm2MoeWeights,
    Mixer, MoeFfn,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_for, weight_gemv, weight_gemv_residual,
};
use rdna_compute::{DType, Gpu};

/// Decode one token; returns the full logits vector.
///
/// Routes to the hipGraph capture/replay path when `HIPFIRE_LFM2_GRAPH=1`
/// (default OFF → exact prior behavior). The graph path amortizes the ~377
/// per-token kernel launches by replaying a single captured graph; see
/// `decode_step_with_graph`.
pub fn decode_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if graph_enabled() {
        return decode_step_with_graph(cfg, weights, state, gpu, token_id, position);
    }
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
}

/// `HIPFIRE_LFM2_GRAPH=1` opt-in switch. Default OFF (unset / "0") →
/// byte-identical to the legacy per-launch decode path. Parsed once.
fn graph_enabled() -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_LFM2_GRAPH").ok().as_deref(),
            Some("1")
        )
    })
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
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;

    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;

    // Embedding lookup → residual stream h (Q8 table).
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm2moe: embed lookup: {e:?}"))?;

    decode_step_layers_and_head(cfg, weights, state, gpu, position, capture)
}

/// Per-layer mixer/FFN stack + final norm + lm_head. Reads the residual
/// stream `state.h` (already seeded by the embedding lookup) and the device
/// position scalar `state.pos_buf` (already staged); writes `state.logits`.
///
/// This is the hipGraph-captureable region: it issues only kernel launches
/// that read STABLE device buffers and (on the MoE path) compute their
/// topk/positions on-device, so a single capture replays correctly at every
/// later position once `state.pos_buf` is refreshed. The per-token-varying
/// embedding lookup (token_id is a kernarg) and the `pos_buf` htod are the
/// caller's responsibility OUTSIDE the captured region.
///
/// `capture` (oracle dumper) is incompatible with hipGraph capture — it issues
/// a sync `download_f32` per layer. The graph path always passes `None`.
fn decode_step_layers_and_head(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
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

    // #397 Ship 6 — forward-as-pipeline. HIPFIRE_FORWARD_LOWERED=1 routes the
    // per-layer decode through the super-op executor (run_layer_program). Skipped
    // when capturing (the oracle dumper needs the per-layer hand path) — that path
    // stays byte-identical. Default off (opt-in) until fleet byte-parity validated.
    if lfm2_forward_lowered_enabled() && capture.is_none() {
        return decode_step_layers_and_head_lowered(cfg, weights, state, gpu, position);
    }

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
                rotate_x_mq_for(
                    gpu,
                    &m.experts[0].gate_up,
                    &state.ffn_tmp,
                    &state.ffn_x_rot,
                    hidden,
                )
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
                // Experts are uniform per layer (gate_up/down share dtype). MQ6G256
                // experts use the HFQ6 (200 B/group, 6-bit) indexed kernels; MQ4G256
                // (default) uses the HFQ4 (136 B/group, 4-bit) siblings. Both consume
                // the same FWHT-rotated `ffn_x_rot` — only the weight dequant differs.
                let experts_mq6 = m.experts[0].gate_up.gpu_dtype == DType::MQ6G256;
                if experts_mq6 {
                    gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
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
                    .map_err(|e| format!("lfm2moe L{l}: gate_up(mq6): {e:?}"))?;
                } else {
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
                .map_err(|e| format!("lfm2moe L{l}: silu_mul_rotate: {e:?}"))?;

                if experts_mq6 {
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
                    .map_err(|e| format!("lfm2moe L{l}: down(mq6): {e:?}"))?;
                } else {
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
                }

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
    gpu.rmsnorm_f32(
        &state.h,
        &weights.embedding_norm,
        &state.final_norm_buf,
        eps,
    )
    .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: LFM2.5 lowered decode (the run_conv slot).
//
// LFM2 is the substrate's Conv super-op proving ground. Each layer lowers to a
// short LayerProgram of coarse super-ops; the per-token executor (run_layer_
// program) calls these arch handlers. ADDITIVE + opt-in (HIPFIRE_FORWARD_LOWERED,
// default off) — the hand loop in decode_step_layers_and_head is untouched, so
// the default path stays byte-identical; the lowered path is validated byte-
// identical via the FORWARD_LOWERED=0-vs-=1 committed-token md5 A/B before flip.
//
// Super-op map (pre-norm folded into each handler):
//   Conv         = operator_norm + in_proj + conv1d_gated + out_proj(+resid)
//   Attend       = operator_norm + q/k/v + qk_norm + rope + kv + attn + o(+resid)
//   Proj(GU)     = ffn_norm + w1 + w3            ResidualGemv(DOWN) = silu·mul + w2(+resid)
//   Moe          = ffn_norm + rotate + router + top-k + experts + combine
// ─────────────────────────────────────────────────────────────────────────

/// Conv mixer block (operator-norm folded in). Mirrors the hand-loop Conv arm.
fn conv_mixer_block(
    gpu: &mut Gpu, cfg: &Lfm2MoeConfig, op_norm: &rdna_compute::GpuTensor,
    c: &ConvWeights, state: &Lfm2MoeState, l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
        .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
    gpu.conv1d_gated_decode_f32(
        &state.conv_bcx, &state.conv_states[c.conv_state_idx], &c.conv_weight,
        &state.conv_y, 1, hidden, cfg.conv_kernel_size,
    )
    .map_err(|e| format!("lfm2moe L{l}: conv gated decode: {e:?}"))?;
    weight_gemv_residual(gpu, &c.out_proj, &state.conv_y, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))
}

/// Attention mixer block (operator-norm folded in). Mirrors the hand-loop Attn arm.
fn attn_mixer_block(
    gpu: &mut Gpu, cfg: &Lfm2MoeConfig, op_norm: &rdna_compute::GpuTensor,
    a: &AttnWeights, state: &Lfm2MoeState, l: usize, seq_len: usize,
) -> Result<(), String> {
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q).map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k).map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v).map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;
    gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: k_norm: {e:?}"))?;
    gpu.rope_f32(&state.fa_q, &state.fa_k, &state.pos_buf, n_heads, n_kv, head_dim, cfg.rope_theta)
        .map_err(|e| format!("lfm2moe L{l}: rope: {e:?}"))?;
    let kv_idx = a.kv_idx;
    gpu.kv_cache_write_q8_0(&state.kv.k_gpu[kv_idx], &state.fa_k, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("lfm2moe L{l}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&state.kv.v_gpu[kv_idx], &state.fa_v, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("lfm2moe L{l}: kv write v: {e:?}"))?;
    gpu.attention_q8_0_kv(
        &state.fa_q, &state.kv.k_gpu[kv_idx], &state.kv.v_gpu[kv_idx], &state.fa_attn_out,
        &state.pos_buf, seq_len, n_heads, n_kv, head_dim, state.kv.physical_cap,
    )
    .map_err(|e| format!("lfm2moe L{l}: attention: {e:?}"))?;
    weight_gemv_residual(gpu, &a.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: out_proj: {e}"))
}

/// Dense FFN gate/up half (ffn-norm folded in). Mirrors the hand-loop Dense head.
fn dense_gate_up_block(
    gpu: &mut Gpu, cfg: &Lfm2MoeConfig, ffn_norm: &rdna_compute::GpuTensor,
    d: &DenseFfn, state: &Lfm2MoeState, l: usize,
) -> Result<(), String> {
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate).map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
    weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up).map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))
}

/// Dense FFN down half (silu·mul + w2 residual). Mirrors the hand-loop Dense tail.
fn dense_down_block(gpu: &mut Gpu, d: &DenseFfn, state: &Lfm2MoeState, l: usize) -> Result<(), String> {
    gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
        .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
    weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))
}

/// MoE FFN block (ffn-norm folded in). Mirrors the hand-loop Moe arm.
fn moe_ffn_block(
    gpu: &mut Gpu, cfg: &Lfm2MoeConfig, ffn_norm: &rdna_compute::GpuTensor,
    m: &MoeFfn, state: &Lfm2MoeState, l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    rotate_x_mq_for(gpu, &m.experts[0].gate_up, &state.ffn_tmp, &state.ffn_x_rot, hidden)
        .map_err(|e| format!("lfm2moe L{l}: ffn rotate: {e:?}"))?;
    weight_gemv(gpu, &m.router, &state.ffn_tmp, &state.router_logits)
        .map_err(|e| format!("lfm2moe L{l}: router: {e}"))?;
    gpu.sigmoid_f32(&state.router_logits).map_err(|e| format!("lfm2moe L{l}: sigmoid: {e:?}"))?;
    gpu.deepseek4_moe_topk_bias_aware_f32(
        &state.router_logits, &m.expert_bias, &state.topk_indices, &state.topk_weights,
        n_exp as i32, k_top as i32, cfg.routed_scaling_factor,
    )
    .map_err(|e| format!("lfm2moe L{l}: topk: {e:?}"))?;
    let experts_mq6 = m.experts[0].gate_up.gpu_dtype == DType::MQ6G256;
    if experts_mq6 {
        gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
            &m.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
            &state.gate_batch, &state.up_batch, 2 * moe_inter, hidden, k_top, 1,
        )
        .map_err(|e| format!("lfm2moe L{l}: gate_up(mq6): {e:?}"))?;
    } else {
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            &m.expert_gate_up_ptrs, &state.topk_indices, &state.ffn_x_rot,
            &state.gate_batch, &state.up_batch, 2 * moe_inter, hidden, k_top, 1,
        )
        .map_err(|e| format!("lfm2moe L{l}: gate_up: {e:?}"))?;
    }
    fused_silu_mul_rotate_mq_batched_for(
        gpu, &m.experts[0].down, &state.gate_batch, &state.up_batch, &state.rot_batch, moe_inter, k_top,
    )
    .map_err(|e| format!("lfm2moe L{l}: silu_mul_rotate: {e:?}"))?;
    if experts_mq6 {
        gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
            &m.expert_down_ptrs, &state.topk_indices, &state.rot_batch, &state.down_expanded,
            hidden, moe_inter, k_top, 1,
        )
        .map_err(|e| format!("lfm2moe L{l}: down(mq6): {e:?}"))?;
    } else {
        gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
            &m.expert_down_ptrs, &state.topk_indices, &state.rot_batch, &state.down_expanded,
            hidden, moe_inter, k_top, 1,
        )
        .map_err(|e| format!("lfm2moe L{l}: down: {e:?}"))?;
    }
    gpu.moe_down_combine_k8_batched(&state.down_expanded, &state.topk_weights, &state.h, hidden, k_top, 1)
        .map_err(|e| format!("lfm2moe L{l}: combine: {e:?}"))
}

/// lfm2-local super-op opcodes (encoded in OpBinding.weights[0]).
mod lfm2_op {
    pub const DENSE_GATE_UP: u32 = 0;
    pub const DENSE_DOWN: u32 = 1;
}

/// The four lfm2 decoder-layer shapes (mixer × FFN). Pure → unit-testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lfm2Variant {
    ConvDense,
    ConvMoe,
    AttnDense,
    AttnMoe,
}

fn lfm2_variant_of(layer: &Lfm2MoeLayerWeights) -> Lfm2Variant {
    match (&layer.mixer, &layer.ffn) {
        (Mixer::Conv(_), Ffn::Dense(_)) => Lfm2Variant::ConvDense,
        (Mixer::Conv(_), Ffn::Moe(_)) => Lfm2Variant::ConvMoe,
        (Mixer::Attention(_), Ffn::Dense(_)) => Lfm2Variant::AttnDense,
        (Mixer::Attention(_), Ffn::Moe(_)) => Lfm2Variant::AttnMoe,
    }
}

#[inline]
fn lfm2_superop(kind: SuperOpKind, code: u32) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: vec![WeightSlot(code)],
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// Lower one lfm2 decoder layer to a coarse super-op LayerProgram (mirrors the
/// hand-loop order: mixer block, then FFN). Pure (no GpuTensor) → unit-testable.
fn lfm2_lower_variant(v: Lfm2Variant) -> superop::LayerProgram {
    use lfm2_op::{DENSE_DOWN, DENSE_GATE_UP};
    use SuperOpKind::{Attend, Conv, Moe, Proj, ResidualGemv};
    match v {
        Lfm2Variant::ConvDense => vec![
            lfm2_superop(Conv, 0),
            lfm2_superop(Proj, DENSE_GATE_UP),
            lfm2_superop(ResidualGemv, DENSE_DOWN),
        ],
        Lfm2Variant::AttnDense => vec![
            lfm2_superop(Attend, 0),
            lfm2_superop(Proj, DENSE_GATE_UP),
            lfm2_superop(ResidualGemv, DENSE_DOWN),
        ],
        Lfm2Variant::ConvMoe => vec![lfm2_superop(Conv, 0), lfm2_superop(Moe, 0)],
        Lfm2Variant::AttnMoe => vec![lfm2_superop(Attend, 0), lfm2_superop(Moe, 0)],
    }
}

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
struct Lfm2MoeBindings<'a> {
    cfg: &'a Lfm2MoeConfig,
    layer: &'a Lfm2MoeLayerWeights,
    state: &'a Lfm2MoeState,
    l: usize,
    seq_len: usize,
}

impl<'a> ForwardBindings for Lfm2MoeBindings<'a> {
    fn run_conv(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Conv(c) => conv_mixer_block(gpu, self.cfg, &self.layer.operator_norm, c, self.state, self.l),
            _ => Err("run_conv on non-Conv layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_attend(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Attention(a) => attn_mixer_block(gpu, self.cfg, &self.layer.operator_norm, a, self.state, self.l, self.seq_len),
            _ => Err("run_attend on non-Attention layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_proj(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_GATE_UP, Ffn::Dense(d)) => {
                dense_gate_up_block(gpu, self.cfg, &self.layer.ffn_norm, d, self.state, self.l)
            }
            _ => Err(format!("run_proj bad opcode {code} / non-Dense ffn")),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_residual_gemv(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, op: &OpBinding) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_DOWN, Ffn::Dense(d)) => dense_down_block(gpu, d, self.state, self.l),
            _ => Err(format!("run_residual_gemv bad opcode {code} / non-Dense ffn")),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_moe(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        match &self.layer.ffn {
            Ffn::Moe(m) => moe_ffn_block(gpu, self.cfg, &self.layer.ffn_norm, m, self.state, self.l),
            _ => Err("run_moe on non-Moe ffn".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_norm(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("lfm2 has no standalone Norm super-op".into()))
    }
    fn run_recurrent(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("lfm2 has no Recurrent super-op".into()))
    }
    fn run_escape(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding, kind: superop::EscapeKind) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!("lfm2 has no Escape super-op ({kind:?})")))
    }
}

/// Cached HIPFIRE_FORWARD_LOWERED toggle for lfm2. #397 Ship 6: the lfm2 lowered
/// decode is **DEFAULT ON** as of 2026-06-07 — fleet byte-parity validated
/// (k9lin gfx1100 / hiptrx gfx1201 / hipx gfx1151, lowered == hand token-text md5
/// 754a38b5…). Escape hatch: `HIPFIRE_FORWARD_LOWERED=0` forces the legacy hand
/// loop (still present in decode_step_layers_and_head); any other value / unset → lowered.
fn lfm2_forward_lowered_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0"))
}

/// Lowered (#397 Ship 6) per-layer decode loop + final norm/head. Behaviorally
/// equivalent to decode_step_layers_and_head's hand loop (validated via the
/// FORWARD_LOWERED=0-vs-=1 committed-token md5 A/B). No oracle-capture support.
fn decode_step_layers_and_head_lowered(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let ctx = DispatchCtx::new(gpu);
    for (l, layer) in weights.layers.iter().enumerate() {
        let program = lfm2_lower_variant(lfm2_variant_of(layer));
        {
            let mut bind = Lfm2MoeBindings { cfg, layer, state, l, seq_len };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| format!("lfm2moe L{l}: lowered run_layer_program: {e}"))?;
        }
    }
    state.n_tokens = seq_len;
    gpu.rmsnorm_f32(&state.h, &weights.embedding_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    Ok(())
}

/// hipGraph-amortized decode_step. Opt-in via `HIPFIRE_LFM2_GRAPH=1`
/// (default OFF → exact `decode_step_inner` behavior). Mirrors the working
/// DeepSeek-V4 integration (`decode_step_with_graph`).
///
/// Three-state machine driven by `state.graph_warmed_up` and `gpu.graph_exec`:
///   1. !warmed_up                 → direct dispatch once (so kernel JIT and
///                                    any lazy hipMalloc happen OUTSIDE the
///                                    captured region), set the flag.
///   2. warmed_up && no graph      → embedding+pos direct, then capture the
///                                    layer loop + head, instantiate, launch
///                                    once for this position's output.
///   3. graph instantiated         → embedding+pos direct, then `graph_launch`
///                                    re-runs the captured ops which re-read
///                                    `state.pos_buf` (refreshed below) and the
///                                    KV / conv-state / topk device buffers.
///
/// Per-token-varying values handled OUTSIDE the captured region:
///   * `token_id` — baked into `embedding_lookup_q8`'s kernarg, so the
///     embedding lookup runs DIRECT each token (writes `state.h`); the
///     captured region begins at layer 0's rmsnorm reading `state.h`.
///   * `position` — staged into the STABLE device buffer `state.pos_buf` via a
///     direct `memcpy_htod` before each `graph_launch`; every captured kernel
///     (rope/kv-write/attention) reads `pos_buf` from the device, so replay at
///     a new position is correct without re-capture. The attention kernel's
///     launch-baked `block_size`/`shared_mem` are sized to `max_seq` under
///     capture (see `attention_q8_0_kv` in dispatch.rs), so one capture
///     replays correctly at every later position.
///
/// `state.n_tokens` is advanced here to match `decode_step_inner` semantics.
pub fn decode_step_with_graph(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let hidden = cfg.hidden_size;

    // ── Warmup phase: direct dispatch, no capture ──────────────────────────
    // Run the legacy path once so inline JIT / lazy scratch alloc happen
    // before any stream capture (capturing a hipMalloc errors).
    if !state.graph_warmed_up {
        state.graph_warmed_up = true;
        decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
        return gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("lfm2moe: download logits (graph warmup): {e:?}"));
    }

    // Capture/replay needs an explicit (non-null) stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("lfm2moe: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Per-token-varying ops, DIRECT (outside the captured region).
    // pos_buf: refreshed each token; the captured kernels re-read it on replay.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos (graph): {e:?}"))?;
    // embedding lookup: token_id is a kernarg → must run per-token, not captured.
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm2moe: embed lookup (graph): {e:?}"))?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────────
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: begin_graph_capture: {e:?}"))?;
        decode_step_layers_and_head(cfg, weights, state, gpu, position, None)?;
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: end_graph_capture: {e:?}"))?;
        // Recorded, not executed — launch once so this position's logits are real.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: graph_launch (capture-end): {e:?}"))?;
        eprintln!(
            "[LFM2.5-MoE hipGraph] captured forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
        // decode_step_layers_and_head set n_tokens; capture-end launch ran it.
    } else {
        // ── Replay phase ────────────────────────────────────────────────────
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: graph_launch (replay): {e:?}"))?;
        // Mirror decode_step_layers_and_head's `state.n_tokens = position + 1`,
        // which the replayed graph does NOT execute (it is host-side state).
        state.n_tokens = position as usize + 1;
    }

    // Logits download outside the captured region (sync D2H on the null stream;
    // completes after the captured kernels finish on the captured stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits (graph): {e:?}"))
}

#[cfg(test)]
mod ship6_lower_tests {
    use super::*;
    use superop::SuperOpKind::{Attend, Conv, Moe, Proj, ResidualGemv};

    // #397 Ship 6 — lfm2 lowered LayerProgram shapes must mirror the hand-loop
    // order (mixer block, then FFN). CPU-pure (no GPU).
    #[test]
    fn lfm2_variant_shapes() {
        let kinds = |v| lfm2_lower_variant(v).iter().map(|o| o.kind).collect::<Vec<_>>();
        assert_eq!(kinds(Lfm2Variant::ConvDense), vec![Conv, Proj, ResidualGemv]);
        assert_eq!(kinds(Lfm2Variant::AttnDense), vec![Attend, Proj, ResidualGemv]);
        assert_eq!(kinds(Lfm2Variant::ConvMoe), vec![Conv, Moe]);
        assert_eq!(kinds(Lfm2Variant::AttnMoe), vec![Attend, Moe]);
        let p = lfm2_lower_variant(Lfm2Variant::ConvDense);
        assert_eq!(p[1].binding.weights[0].0, lfm2_op::DENSE_GATE_UP);
        assert_eq!(p[2].binding.weights[0].0, lfm2_op::DENSE_DOWN);
    }
}
