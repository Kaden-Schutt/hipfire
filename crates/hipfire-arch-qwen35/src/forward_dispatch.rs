// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Alternative forward_scratch_layers using the hipfire-dispatch family API.
//!
//! Gated behind `#[cfg(feature = "new-dispatch")]`. Replaces inline
//! match-dtype trees with calls to [`RotationFamily`], [`GemvFamily`],
//! and transitional fused-kernel dispatch helpers.

use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, WeightRef};
use hipfire_dispatch::families::rotation::{RotationFamily, RotationParams};
use hipfire_dispatch::types::RotationVariant;
use hipfire_dispatch::types::dtype_needs_fwht;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::qwen35::{
    self, DeltaNetState, LayerType, LayerWeights, MoeFfnWeights, Qwen35Config, Qwen35Scratch,
    Qwen35Weights, StateQuant,
};
use crate::speculative::HiddenStateRingBuffer;

// ── Forward scratch layers (dispatch family version) ────────────────────

pub fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
) -> HipResult<()> {
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let ctx = DispatchCtx::new(gpu);
    let rotation = RotationFamily::new();
    let gemv = GemvFamily::new();

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // ── RMSNorm + FWHT rotation ──
                let (x_rot, x_rot_paro) = rmsnorm_rotate_dispatch(
                    gpu, &ctx, &rotation, &s.x, &layer.attn_norm,
                    &layer.wqkv, &s.tmp, &s.x_rot, config.norm_eps,
                )?;

                // ── Fused QKVZA (4-way) or fallback ──
                fused_qkvza_dispatch(
                    gpu, &gemv, &ctx,
                    &layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha,
                    &s.tmp, x_rot, x_rot_paro,
                    &s.dn_qkv, &s.dn_z, &s.dn_beta, &s.dn_alpha,
                )?;

                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta, &s.dn_alpha, &layer.dt_bias, &layer.a_log, n_v_heads,
                )?;

                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw, &s.dn_k_raw, &s.dn_v,
                    &s.dn_qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim, v_dim,
                )?;

                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw, &s.dn_k_raw,
                    config.linear_num_key_heads, hd,
                    1.0 / (hd as f32).sqrt(), config.norm_eps,
                )?;

                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw, &s.dn_k_raw, &s.dn_q, &s.dn_k,
                        config.linear_num_key_heads, ratio, hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                }

                gpu.gated_norm_f32(&s.dn_attn_out, &s.dn_z, &layer.norm_weight,
                    &s.dn_normed, n_v_heads, config.linear_value_head_dim, config.norm_eps)?;
                hipfire_runtime::llama::weight_gemv_residual(gpu, &layer.wo, &s.dn_normed, &s.x)?;

                // ── FFN ──
                let (x_rot, x_rot_paro) = rmsnorm_rotate_dispatch(
                    gpu, &ctx, &rotation, &s.x, &layer.ffn_norm,
                    &layer.w_gate, &s.tmp, &s.x_rot, config.norm_eps,
                )?;

                fused_gate_up_dispatch(
                    gpu, &gemv, &ctx,
                    &layer.w_gate, &layer.w_up,
                    &s.tmp, x_rot, x_rot_paro,
                    &s.gate_ffn, &s.up,
                )?;

                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu, &layer.w_down, &s.gate_ffn, &s.up, &s.ffn_hidden, &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                qwen35::trace_finite_if_enabled(gpu,
                    &format!("layer {layer_idx} LinearAttention residual"), &s.x)?;
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                let (x_rot, x_rot_paro) = rmsnorm_rotate_dispatch(
                    gpu, &ctx, &rotation, &s.x, &layer.attn_norm,
                    &layer.wq, &s.tmp, &s.x_rot, config.norm_eps,
                )?;

                fused_qkv_dispatch(
                    gpu, &gemv, &ctx,
                    &layer.wq, &layer.wk, &layer.wv,
                    &s.tmp, x_rot, x_rot_paro,
                    &s.fa_q_full, &s.fa_k, &s.fa_v,
                )?;

                gpu.deinterleave_f32(&s.fa_q_full, &s.fa_q, &s.fa_gate,
                    config.n_heads, config.head_dim)?;
                gpu.rmsnorm_batched(&s.fa_q, &layer.q_norm, &s.fa_q,
                    config.n_heads, config.head_dim, config.norm_eps)?;
                gpu.rmsnorm_batched(&s.fa_k, &layer.k_norm, &s.fa_k,
                    config.n_kv_heads, config.head_dim, config.norm_eps)?;

                if hipfire_runtime::triattn::tap_enabled() {
                    triattn_tap(gpu, layer_idx, &s, config)?;
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                kv_cache_attention_dispatch(gpu, kv_cache, s, config, layer_idx, pos)?;

                gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                hipfire_runtime::llama::weight_gemv_residual(gpu, &layer.wo, &s.fa_attn_out, &s.x)?;

                // ── FFN ──
                let (x_rot, x_rot_paro) = rmsnorm_rotate_dispatch(
                    gpu, &ctx, &rotation, &s.x, &layer.ffn_norm,
                    &layer.w_gate, &s.tmp, &s.x_rot, config.norm_eps,
                )?;

                fused_gate_up_dispatch(
                    gpu, &gemv, &ctx,
                    &layer.w_gate, &layer.w_up,
                    &s.tmp, x_rot, x_rot_paro,
                    &s.gate_ffn, &s.up,
                )?;

                hipfire_runtime::llama::weight_gemv_swiglu_residual(
                    gpu, &layer.w_down, &s.gate_ffn, &s.up, &s.ffn_hidden, &s.x,
                )?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                qwen35::trace_finite_if_enabled(gpu,
                    &format!("layer {layer_idx} FullAttention residual"), &s.x)?;
                kv_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                let (x_rot, x_rot_paro) = rmsnorm_rotate_dispatch(
                    gpu, &ctx, &rotation, &s.x, &layer.attn_norm,
                    &layer.wqkv, &s.tmp, &s.x_rot, config.norm_eps,
                )?;

                fused_qkvza_dispatch(
                    gpu, &gemv, &ctx,
                    &layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha,
                    &s.tmp, x_rot, x_rot_paro,
                    &s.dn_qkv, &s.dn_z, &s.dn_beta, &s.dn_alpha,
                )?;

                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta, &s.dn_alpha, &layer.dt_bias, &layer.a_log, n_v_heads,
                )?;
                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw, &s.dn_k_raw, &s.dn_v,
                    &s.dn_qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim, v_dim,
                )?;
                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw, &s.dn_k_raw,
                    config.linear_num_key_heads, hd,
                    1.0 / (hd as f32).sqrt(), config.norm_eps,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw, &s.dn_k_raw, &s.dn_q, &s.dn_k,
                        config.linear_num_key_heads, ratio, hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q, &s.dn_k, &s.dn_v, &s.dn_alpha, &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &s.dn_attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                }

                gpu.gated_norm_f32(&s.dn_attn_out, &s.dn_z, &layer.norm_weight,
                    &s.dn_normed, n_v_heads, config.linear_value_head_dim, config.norm_eps)?;
                hipfire_runtime::llama::weight_gemv_residual(gpu, &layer.wo, &s.dn_normed, &s.x)?;

                // ── MoE FFN ──
                moe_ffn_dispatch(gpu, &layer.ffn, &s.x, &layer.ffn_norm, config, s)?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                let (x_rot, x_rot_paro) = rmsnorm_rotate_dispatch(
                    gpu, &ctx, &rotation, &s.x, &layer.attn_norm,
                    &layer.wq, &s.tmp, &s.x_rot, config.norm_eps,
                )?;

                fused_qkv_dispatch(
                    gpu, &gemv, &ctx,
                    &layer.wq, &layer.wk, &layer.wv,
                    &s.tmp, x_rot, x_rot_paro,
                    &s.fa_q_full, &s.fa_k, &s.fa_v,
                )?;

                gpu.deinterleave_f32(&s.fa_q_full, &s.fa_q, &s.fa_gate,
                    config.n_heads, config.head_dim)?;
                gpu.rmsnorm_batched(&s.fa_q, &layer.q_norm, &s.fa_q,
                    config.n_heads, config.head_dim, config.norm_eps)?;
                gpu.rmsnorm_batched(&s.fa_k, &layer.k_norm, &s.fa_k,
                    config.n_kv_heads, config.head_dim, config.norm_eps)?;

                if hipfire_runtime::triattn::tap_enabled() {
                    triattn_tap(gpu, layer_idx, s, config)?;
                }

                if kv_cache.compact_offset > 0 {
                    let abs = (pos + kv_cache.compact_offset) as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32(&s.fa_q, &s.fa_k, &s.pos_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                kv_cache_attention_dispatch(gpu, kv_cache, s, config, layer_idx, pos)?;

                gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                hipfire_runtime::llama::weight_gemv_residual(gpu, &layer.wo, &s.fa_attn_out, &s.x)?;

                // ── MoE FFN ──
                moe_ffn_dispatch(gpu, &layer.ffn, &s.x, &layer.ffn_norm, config, s)?;

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                kv_layer_idx += 1;
            }

            // Mismatched layer weight / type combinations are unreachable
            // (the loader guarantees alignment).
            _ => unreachable!(),
        }
    }
    Ok(())
}

// ── Dispatch helpers ─────────────────────────────────────────────────────

/// RMSNorm + FWHT rotation dispatch.
///
/// Returns `(Some(x_rot), None)` when the weight dtype requires FWHT rotation
/// (MQ family). Returns `(None, None)` when the plain rmsnorm output is in
/// `tmp`. `x_rot_paro` is only set on the experimental PARO per-group path.
fn rmsnorm_rotate_dispatch<'a>(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    rotation: &RotationFamily,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    sample_weight: &hipfire_runtime::llama::WeightTensor,
    tmp: &'a GpuTensor,
    x_rot_scratch: &'a GpuTensor,
    eps: f32,
) -> HipResult<(Option<&'a GpuTensor>, Option<&'a GpuTensor>)> {
    let is_mq = matches!(sample_weight.gpu_dtype,
        DType::MQ4G256 | DType::MQ3G256 | DType::MQ2G256
        | DType::MQ6G256 | DType::MQ8G256
        | DType::MQ2G256Lloyd | DType::MQ3G256Lloyd | DType::MQ4G256Lloyd
        | DType::MFP4G32);

    if is_mq {
        let awq_scale = sample_weight.awq_scale.as_ref();
        rotation.run(ctx, gpu, RotationParams {
            x,
            x_up: None,
            w_norm: Some(norm_weight),
            x_plain: tmp,
            x_rot: x_rot_scratch,
            awq_scale,
            k: sample_weight.k,
            eps,
            batch_size: 1,
            variant: RotationVariant::WithRmsnorm,
        })?;
        Ok((Some(x_rot_scratch), None))
    } else {
        gpu.rmsnorm_f32(x, norm_weight, tmp, eps)?;
        // Lever 1: experimental PARO per-group rotation.
        let paro_opt_in = sample_weight.gpu_dtype == DType::PARO4G128T
            && sample_weight.k % 128 == 0
            && sample_weight.m % 8 == 0
            && std::env::var("HIPFIRE_PARO_FUSE_RMSNORM")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        if paro_opt_in {
            gpu.fused_rmsnorm_paro4g128t_rotate(
                &sample_weight.buf, x, norm_weight,
                x_rot_scratch, Some(tmp),
                sample_weight.m, sample_weight.k, eps,
            )?;
            Ok((None, Some(x_rot_scratch)))
        } else {
            Ok((None, None))
        }
    }
}

/// Fused QKVZA (4-way) dispatch.
///
/// Routes to the fused kernel for known dtype combinations, or falls back to
/// individual `gemv_prerotated_or_plain` calls.
#[allow(clippy::too_many_arguments)]
fn fused_qkvza_dispatch(
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    ctx: &DispatchCtx,
    wqkv: &hipfire_runtime::llama::WeightTensor,
    wz: &hipfire_runtime::llama::WeightTensor,
    w_beta: &hipfire_runtime::llama::WeightTensor,
    w_alpha: &hipfire_runtime::llama::WeightTensor,
    tmp: &GpuTensor,
    eff_rot: Option<&GpuTensor>,
    _x_rot_paro: Option<&GpuTensor>,
    dn_qkv: &GpuTensor,
    dn_z: &GpuTensor,
    dn_beta: &GpuTensor,
    dn_alpha: &GpuTensor,
) -> HipResult<()> {
    let dt = wqkv.gpu_dtype;
    let same = wz.gpu_dtype == dt && w_beta.gpu_dtype == dt && w_alpha.gpu_dtype == dt;

    let use_fused = same && (dt == DType::MQ4G256 || dt == DType::HFQ4G256
        || dt == DType::MQ3G256Lloyd || dt == DType::MQ4G256Lloyd
        || ((dt == DType::MQ6G256 || dt == DType::HFQ6G256) && ctx.arch.has_dot2_f32_f16()));

    if use_fused {
        let x = eff_rot.unwrap_or(tmp);
        if dt == DType::MQ4G256 || dt == DType::HFQ4G256 {
            gpu.fused_qkvza_hfq4g256(
                &wqkv.buf, &wz.buf, &w_beta.buf, &w_alpha.buf, x,
                dn_qkv, dn_z, dn_beta, dn_alpha,
                wqkv.m, wz.m, w_beta.m, w_alpha.m, wqkv.k,
            )
        } else if dt == DType::MQ3G256Lloyd {
            gpu.fused_qkvza_mq3g256_lloyd(
                &wqkv.buf, &wz.buf, &w_beta.buf, &w_alpha.buf, x,
                dn_qkv, dn_z, dn_beta, dn_alpha,
                wqkv.m, wz.m, w_beta.m, w_alpha.m, wqkv.k,
            )
        } else if dt == DType::MQ4G256Lloyd {
            gpu.fused_qkvza_mq4g256_lloyd(
                &wqkv.buf, &wz.buf, &w_beta.buf, &w_alpha.buf, x,
                dn_qkv, dn_z, dn_beta, dn_alpha,
                wqkv.m, wz.m, w_beta.m, w_alpha.m, wqkv.k,
            )
        } else if dt == DType::MQ6G256 || dt == DType::HFQ6G256 {
            gpu.fused_qkvza_hfq6g256_dp4a(
                &wqkv.buf, &wz.buf, &w_beta.buf, &w_alpha.buf, x,
                dn_qkv, dn_z, dn_beta, dn_alpha,
                wqkv.m, wz.m, w_beta.m, w_alpha.m, wqkv.k,
            )
        } else {
            unreachable!()
        }
    } else {
        let mut run = |w: &hipfire_runtime::llama::WeightTensor, y: &GpuTensor| -> HipResult<()> {
            let x = if dtype_needs_fwht(w.gpu_dtype) {
                eff_rot.ok_or_else(|| {
                    HipError::new(0, "MQ-weight GEMV requires prerotated input")
                })?
            } else {
                tmp
            };
            gemv.run_auto(ctx, gpu,
                &WeightRef { buf: &w.buf, dtype: w.gpu_dtype, m: w.m, k: w.k },
                x, y,
            ).map_err(|e| HipError::new(0, &e.to_string()))
        };
        run(wqkv, dn_qkv)?;
        run(wz, dn_z)?;
        run(w_beta, dn_beta)?;
        run(w_alpha, dn_alpha)
    }
}

/// Fused QKV (3-way) dispatch for full attention projections.
#[allow(clippy::too_many_arguments)]
fn fused_qkv_dispatch(
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    ctx: &DispatchCtx,
    wq: &hipfire_runtime::llama::WeightTensor,
    wk: &hipfire_runtime::llama::WeightTensor,
    wv: &hipfire_runtime::llama::WeightTensor,
    tmp: &GpuTensor,
    eff_rot: Option<&GpuTensor>,
    _x_rot_paro: Option<&GpuTensor>,
    fa_q: &GpuTensor,
    fa_k: &GpuTensor,
    fa_v: &GpuTensor,
) -> HipResult<()> {
    let dt = wq.gpu_dtype;
    let same = wk.gpu_dtype == dt && wv.gpu_dtype == dt;

    let use_fused = same && (dt == DType::MQ4G256 || dt == DType::HFQ4G256
        || dt == DType::MQ3G256Lloyd || dt == DType::MQ4G256Lloyd
        || ((dt == DType::MQ6G256 || dt == DType::HFQ6G256) && ctx.arch.has_dot2_f32_f16()));

    if use_fused {
        let x = eff_rot.unwrap_or(tmp);
        if dt == DType::MQ4G256 || dt == DType::HFQ4G256 {
            gpu.fused_qkv_hfq4g256(
                &wq.buf, &wk.buf, &wv.buf, x,
                fa_q, fa_k, fa_v,
                wq.m, wk.m, wv.m, wq.k,
            )
        } else if dt == DType::MQ3G256Lloyd {
            gpu.fused_qkv_mq3g256_lloyd(
                &wq.buf, &wk.buf, &wv.buf, x,
                fa_q, fa_k, fa_v,
                wq.m, wk.m, wv.m, wq.k,
            )
        } else if dt == DType::MQ4G256Lloyd {
            gpu.fused_qkv_mq4g256_lloyd(
                &wq.buf, &wk.buf, &wv.buf, x,
                fa_q, fa_k, fa_v,
                wq.m, wk.m, wv.m, wq.k,
            )
        } else if dt == DType::MQ6G256 || dt == DType::HFQ6G256 {
            gpu.fused_qkv_hfq6g256_dp4a(
                &wq.buf, &wk.buf, &wv.buf, x,
                fa_q, fa_k, fa_v,
                wq.m, wk.m, wv.m, wq.k,
            )
        } else {
            unreachable!()
        }
    } else {
        let mut run = |w: &hipfire_runtime::llama::WeightTensor, y: &GpuTensor| -> HipResult<()> {
            let x = if dtype_needs_fwht(w.gpu_dtype) {
                eff_rot.ok_or_else(|| {
                    HipError::new(0, "MQ-weight GEMV requires prerotated input")
                })?
            } else {
                tmp
            };
            gemv.run_auto(ctx, gpu,
                &WeightRef { buf: &w.buf, dtype: w.gpu_dtype, m: w.m, k: w.k },
                x, y,
            ).map_err(|e| HipError::new(0, &e.to_string()))
        };
        run(wq, fa_q)?;
        run(wk, fa_k)?;
        run(wv, fa_v)
    }
}

/// Fused Gate+Up (2-way) dispatch for the FFN path.
#[allow(clippy::too_many_arguments)]
fn fused_gate_up_dispatch(
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    ctx: &DispatchCtx,
    w_gate: &hipfire_runtime::llama::WeightTensor,
    w_up: &hipfire_runtime::llama::WeightTensor,
    tmp: &GpuTensor,
    eff_rot: Option<&GpuTensor>,
    _x_rot_paro: Option<&GpuTensor>,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
) -> HipResult<()> {
    let dt = w_gate.gpu_dtype;
    let same = w_up.gpu_dtype == dt;

    let use_fused = same && (dt == DType::MQ4G256 || dt == DType::HFQ4G256
        || dt == DType::MQ3G256Lloyd || dt == DType::MQ4G256Lloyd
        || ((dt == DType::MQ6G256 || dt == DType::HFQ6G256) && ctx.arch.has_dot2_f32_f16()));

    if use_fused {
        let x = eff_rot.unwrap_or(tmp);
        if dt == DType::MQ4G256 || dt == DType::HFQ4G256 {
            gpu.fused_gate_up_hfq4g256(
                &w_gate.buf, &w_up.buf, x,
                gate_out, up_out,
                w_gate.m, w_up.m, w_gate.k,
            )
        } else if dt == DType::MQ3G256Lloyd {
            gpu.fused_gate_up_mq3g256_lloyd(
                &w_gate.buf, &w_up.buf, x,
                gate_out, up_out,
                w_gate.m, w_up.m, w_gate.k,
            )
        } else if dt == DType::MQ4G256Lloyd {
            gpu.fused_gate_up_mq4g256_lloyd(
                &w_gate.buf, &w_up.buf, x,
                gate_out, up_out,
                w_gate.m, w_up.m, w_gate.k,
            )
        } else if dt == DType::MQ6G256 || dt == DType::HFQ6G256 {
            gpu.fused_gate_up_hfq6g256_dp4a(
                &w_gate.buf, &w_up.buf, x,
                gate_out, up_out,
                w_gate.m, w_up.m, w_gate.k,
            )
        } else {
            unreachable!()
        }
    } else {
        let mut run = |w: &hipfire_runtime::llama::WeightTensor, y: &GpuTensor| -> HipResult<()> {
            let x = if dtype_needs_fwht(w.gpu_dtype) {
                eff_rot.ok_or_else(|| {
                    HipError::new(0, "MQ-weight GEMV requires prerotated input")
                })?
            } else {
                tmp
            };
            gemv.run_auto(ctx, gpu,
                &WeightRef { buf: &w.buf, dtype: w.gpu_dtype, m: w.m, k: w.k },
                x, y,
            ).map_err(|e| HipError::new(0, &e.to_string()))
        };
        run(w_gate, gate_out)?;
        run(w_up, up_out)
    }
}

/// MoE FFN dispatch — mirrors the two-path logic from the original.
fn moe_ffn_dispatch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    if qwen35::ffn_all_mq4_for_moe(ffn) {
        gpu.fused_rmsnorm_rotate_mq(
            x, ffn_norm,
            s.moe_x_rot.as_ref().expect("MoE scratch"),
            config.dim, config.norm_eps,
        )?;
        qwen35::moe_ffn_decode_with_scratch_prerotated(gpu, ffn, x, x, config, s)
    } else {
        gpu.rmsnorm_f32(x, ffn_norm, &s.tmp, config.norm_eps)?;
        qwen35::moe_ffn_decode_with_scratch(gpu, ffn, &s.tmp, x, config, s)
    }
}

/// TriAttention tap helper (inline from original forward).
fn triattn_tap(
    gpu: &mut Gpu,
    layer_idx: usize,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
) -> HipResult<()> {
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu, layer_idx, &s.fa_q.buf, 1, config.n_heads, config.head_dim,
    )?;
    if !gpu_handled {
        let n_q = config.n_heads * config.head_dim;
        let q_cpu = gpu.download_f32(&s.fa_q)?;
        if hipfire_runtime::triattn::tap_needs_k() {
            let n_k = config.n_kv_heads * config.head_dim;
            let k_cpu = gpu.download_f32(&s.fa_k)?;
            hipfire_runtime::triattn::record_prerope_qk(
                layer_idx, &q_cpu[..n_q], Some(&k_cpu[..n_k]));
        } else {
            hipfire_runtime::triattn::record_prerope_q(
                layer_idx, &q_cpu[..n_q]);
        }
    }
    Ok(())
}

/// KV cache write + attention dispatch. Inline from original.
fn kv_cache_attention_dispatch(
    gpu: &mut Gpu,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    layer_idx: usize,
    pos: usize,
) -> HipResult<()> {
    if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht4_fused(
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_k, &s.fa_v, &s.pos_buf, ct, st,
                config.n_kv_heads, config.head_dim)?;
            gpu.attention_flash_fwht4(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap, &s.flash_partials,
            )?;
        } else {
            gpu.kv_cache_write_asym4_fused(
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_k, &s.fa_v, &s.pos_buf, ct, st,
                config.n_kv_heads, config.head_dim)?;
            gpu.attention_flash_asym4(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap, &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht3_fused(
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_k, &s.fa_v, &s.pos_buf, ct, st,
                config.n_kv_heads, config.head_dim)?;
            gpu.attention_flash_fwht3(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap, &s.flash_partials,
            )?;
        } else {
            gpu.kv_cache_write_asym3_fused(
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_k, &s.fa_v, &s.pos_buf, ct, st,
                config.n_kv_heads, config.head_dim)?;
            gpu.attention_flash_asym3(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap, &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht2_fused(
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_k, &s.fa_v, &s.pos_buf, ct, st,
                config.n_kv_heads, config.head_dim)?;
            gpu.attention_flash_fwht2(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap, &s.flash_partials,
            )?;
        } else {
            gpu.kv_cache_write_asym2_fused(
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_k, &s.fa_v, &s.pos_buf, ct, st,
                config.n_kv_heads, config.head_dim)?;
            gpu.attention_flash_asym2(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, ct, st, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap, &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(
            &kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf,
            config.n_kv_heads, config.head_dim)?;
        gpu.kv_cache_write_q8_0(
            &kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf,
            config.n_kv_heads, config.head_dim)?;
        let use_flash = gpu.capture_mode
            || s.flash_mode == 2
            || (s.flash_mode == 1 && pos + 1 >= 2048)
            || pos + 1 > 15000;
        if use_flash {
            gpu.attention_flash_q8_0(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap, &s.flash_partials,
            )?;
        } else {
            gpu.attention_q8_0_kv(
                &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out, &s.pos_buf, pos + 1,
                config.n_heads, config.n_kv_heads, config.head_dim,
                kv_cache.physical_cap,
            )?;
        }
    } else {
        let kv_dim = config.n_kv_heads * config.head_dim;
        gpu.kv_cache_write(
            &kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
        gpu.kv_cache_write(
            &kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
        gpu.attention_f32(
            &s.fa_q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out, &s.pos_buf, pos + 1,
            config.n_heads, config.n_kv_heads, config.head_dim,
            kv_cache.physical_cap,
        )?;
    }
    Ok(())
}
