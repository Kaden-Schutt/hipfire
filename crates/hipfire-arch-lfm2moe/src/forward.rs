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
            hipfire_config::developer_var("HIPFIRE_LFM2_GRAPH")
                .ok()
                .as_deref(),
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
    let capture_postmixer =
        hipfire_config::developer_var_os("HIPFIRE_LFM2_CAPTURE_POSTMIXER").is_some();

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

/// Conv mixer block (operator-norm folded in). Mirrors the hand-loop Conv arm.
fn conv_mixer_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    op_norm: &rdna_compute::GpuTensor,
    c: &ConvWeights,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
        .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
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
    weight_gemv_residual(gpu, &c.out_proj, &state.conv_y, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))
}

/// Attention mixer block (operator-norm folded in). Mirrors the hand-loop Attn arm.
fn attn_mixer_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    op_norm: &rdna_compute::GpuTensor,
    a: &AttnWeights,
    state: &Lfm2MoeState,
    l: usize,
    seq_len: usize,
) -> Result<(), String> {
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, eps)
        .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
        .map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
        .map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
        .map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;
    gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
        .map_err(|e| format!("lfm2moe L{l}: k_norm: {e:?}"))?;
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
    let kv_idx = a.kv_idx;
    // KV write (Q8) + attention via the shared KV-usage abstraction. lfm2moe is
    // Q8 non-flash unconditional → derive's q8_attend_key returns AttnQ8_0Kv at
    // pos+1<=15000 (byte-identical; needs no partials, hence flash_partials:
    // None). It flips to AttnFlashQ8_0 at pos+1>15000 (the documented
    // Q8-fidelity edge — rare for this decode model). capture_mode is NOT
    // threaded: the non-flash kernel is capture-safe and lfm2moe captures it.
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(
        hipfire_dispatch::families::kv_tier::KvTierInputs {
            pos: seq_len - 1,
            ..state.kv.tier_inputs()
        },
    )
    .map_err(|e| format!("lfm2moe L{l}: kv tier: {e}"))?;
    let io = hipfire_dispatch::families::attention::AttnParams {
        q: &state.fa_q,
        k: &state.fa_k,
        v: &state.fa_v,
        k_cache: &state.kv.k_gpu[kv_idx],
        v_cache: &state.kv.v_gpu[kv_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &state.pos_buf,
        pos: seq_len - 1,
        positions: None,
        n_heads,
        n_kv_heads: n_kv,
        head_dim,
        physical_cap: state.kv.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: None,
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: None,
        output: &state.fa_attn_out,
    };
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        &ctx,
        &[hipfire_dispatch::pipeline::Step::Attend { plan, io }],
    )
    .map_err(|e| format!("lfm2moe L{l}: attention: {e:?}"))?;
    weight_gemv_residual(gpu, &a.wo, &state.fa_attn_out, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: out_proj: {e}"))
}

/// Dense FFN gate/up half (ffn-norm folded in). Mirrors the hand-loop Dense head.
fn dense_gate_up_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    ffn_norm: &rdna_compute::GpuTensor,
    d: &DenseFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
        .map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
    weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
        .map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))
}

/// Dense FFN down half (silu·mul + w2 residual). Mirrors the hand-loop Dense tail.
fn dense_down_block(
    gpu: &mut Gpu,
    d: &DenseFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
        .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
    weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
        .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))
}

/// MoE FFN block (ffn-norm folded in). Mirrors the hand-loop Moe arm.
fn moe_ffn_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    ffn_norm: &rdna_compute::GpuTensor,
    m: &MoeFfn,
    state: &Lfm2MoeState,
    l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
    rotate_x_mq_for(
        gpu,
        &m.experts[0].gate_up,
        &state.ffn_tmp,
        &state.ffn_x_rot,
        hidden,
    )
    .map_err(|e| format!("lfm2moe L{l}: ffn rotate: {e:?}"))?;
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
    .map_err(|e| format!("lfm2moe L{l}: combine: {e:?}"))
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
