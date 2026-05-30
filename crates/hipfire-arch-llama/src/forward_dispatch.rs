// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dispatch-based forward pass — proof of concept for [`RotationFamily`].
//!
//! When `feature = "new-dispatch"` is active, this module provides
//! `forward_scratch_layers` which replaces inline rotation calls with
//! [`RotationFamily::run`] dispatch via `RotationVariant::WithRmsnorm`.
//! Everything else (attention, KV cache, sampling) delegates to the same
//! code paths as the runtime's `llama::forward_scratch_layers`.
//!
//! Build: `cargo check -p hipfire-arch-llama --features new-dispatch`

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu};

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::rotation::{RotationFamily, RotationParams};
use hipfire_dispatch::types::RotationVariant;
use hipfire_runtime::llama::{self, ForwardScratch, KvCache, LlamaConfig, LlamaWeights};

/// Returns `true` when `dtype` requires FWHT rotation before GEMV.
fn needs_mq_rotation(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256
            | DType::MQ6G256
            | DType::MQ3G256
            | DType::MQ2G256
            | DType::MQ2G256Lloyd
            | DType::MQ3G256Lloyd
            | DType::MQ4G256Lloyd
            | DType::MFP4G32
    )
}

/// Dispatch-aware layer loop that replaces inline rotation calls with
/// [`RotationFamily::run`].
///
/// Signature matches [`hipfire_runtime::llama::forward_scratch_layers`].
///
/// # Rotation dispatch
///
/// When the weight's [`DType`] is an MQ-family format requiring FWHT
/// rotation (see [`needs_mq_rotation`]), this function uses
/// [`RotationFamily::run`] with [`RotationVariant::WithRmsnorm`] to fuse
/// RMSNorm + FWHT into a single kernel launch, reading from `scratch.x`
/// and writing the rotated activation to `scratch.x_rot`.
///
/// For non-MQ dtypes, the original split path is used: `rmsnorm_f32`
/// into `scratch.tmp`, then `weight_gemv` without pre-rotation.
pub fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    temperature: f32,
    top_p: f32,
    rng_state: u32,
    repeat_window: usize,
    repeat_penalty: f32,
) -> HipResult<(u32, u32)> {
    let ctx = DispatchCtx::new(gpu);
    let rotation = RotationFamily::new();

    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // ── Attention QKV path ──────────────────────────────
        if layer.wq.gpu_dtype == DType::Q4K && layer.wk.gpu_dtype == DType::Q4K {
            gpu.fused_qkv_q4k(
                &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                &scratch.tmp, &scratch.q, &scratch.k, &scratch.v,
                layer.wq.m, layer.wk.m, layer.wv.m, layer.wq.k,
            )?;
        } else if needs_mq_rotation(layer.wq.gpu_dtype) {
            // Dispatch: fused RMSNorm + FWHT rotation via RotationFamily.
            // Reads from scratch.x, writes rotated result to scratch.x_rot.
            rotation.run(&ctx, gpu, RotationParams {
                x: &scratch.x,
                x_up: None,
                w_norm: Some(&layer.attn_norm),
                x_plain: &scratch.tmp,
                x_rot: &scratch.x_rot,
                awq_scale: layer.wq.awq_scale.as_ref(),
                k: layer.wq.k,
                eps: config.norm_eps,
                batch_size: 1,
                variant: RotationVariant::WithRmsnorm,
            })?;
            llama::weight_gemv_prerotated(
                gpu, &layer.wq, &scratch.tmp, Some(&scratch.x_rot), &scratch.q,
            )?;
            llama::weight_gemv_prerotated(
                gpu, &layer.wk, &scratch.tmp, Some(&scratch.x_rot), &scratch.k,
            )?;
            llama::weight_gemv_prerotated(
                gpu, &layer.wv, &scratch.tmp, Some(&scratch.x_rot), &scratch.v,
            )?;
        } else {
            gpu.rmsnorm_f32(&scratch.x, &layer.attn_norm, &scratch.tmp, config.norm_eps)?;
            llama::weight_gemv(gpu, &layer.wq, &scratch.tmp, &scratch.q)?;
            llama::weight_gemv(gpu, &layer.wk, &scratch.tmp, &scratch.k)?;
            llama::weight_gemv(gpu, &layer.wv, &scratch.tmp, &scratch.v)?;
        }

        // ── QK norm (optional per config) ───────────────────
        if config.has_qk_norm {
            if let Some(ref qn) = layer.q_norm {
                gpu.rmsnorm_batched(
                    &scratch.q, qn, &scratch.q, n_heads, head_dim, config.norm_eps,
                )?;
            }
            if let Some(ref kn) = layer.k_norm {
                gpu.rmsnorm_batched(
                    &scratch.k, kn, &scratch.k, n_kv_heads, head_dim, config.norm_eps,
                )?;
            }
        }

        // ── RoPE ────────────────────────────────────────────
        gpu.rope_f32(
            &scratch.q, &scratch.k, &scratch.pos_buf,
            n_heads, n_kv_heads, head_dim, config.rope_freq_base,
        )?;

        // ── KV cache write + attention ──────────────────────
        if kv_cache.quant_hfq4 {
            gpu.kv_cache_write_hfq4(
                &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.kv_cache_write_hfq4(
                &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.attention_hfq4_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1,
                n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized
            && !kv_cache.k_scales.is_empty()
            && !kv_cache.quant_int8
            && !kv_cache.quant_q8
        {
            gpu.kv_cache_write_hfq8(
                &kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx],
                &scratch.k, &scratch.pos_buf, n_kv_heads, head_dim,
            )?;
            gpu.kv_cache_write_hfq8(
                &kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx],
                &scratch.v, &scratch.pos_buf, n_kv_heads, head_dim,
            )?;
            gpu.attention_hfq8_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.k_scales[layer_idx],
                &kv_cache.v_gpu[layer_idx], &kv_cache.v_scales[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1,
                n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quant_int8 {
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.kv_cache_write_int8c_f16(
                &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.attention_int8c_f16_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1,
                n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized && kv_cache.quant_q8 {
            gpu.kv_cache_write_q8_0(
                &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.kv_cache_write_q8_0(
                &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.attention_q8_0_kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1,
                n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else if kv_cache.quantized {
            gpu.kv_cache_write_q4(
                &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.kv_cache_write_q4(
                &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf,
                n_kv_heads, head_dim,
            )?;
            gpu.attention_q4kv(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1,
                n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        } else {
            gpu.kv_cache_write(
                &kv_cache.k_gpu[layer_idx], &scratch.k, &scratch.pos_buf, kv_dim,
            )?;
            gpu.kv_cache_write(
                &kv_cache.v_gpu[layer_idx], &scratch.v, &scratch.pos_buf, kv_dim,
            )?;
            gpu.attention_f32(
                &scratch.q,
                &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                &scratch.attn_out, &scratch.pos_buf, pos + 1,
                n_heads, n_kv_heads, head_dim, kv_cache.physical_cap,
            )?;
        }

        // ── Attention output projection + residual ─────────
        llama::weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        // ── FFN path ────────────────────────────────────────
        if layer.w_gate.gpu_dtype == DType::Q4K && layer.w_up.gpu_dtype == DType::Q4K {
            gpu.fused_gate_up_q4k(
                &layer.w_gate.buf, &layer.w_up.buf,
                &scratch.tmp, &scratch.gate, &scratch.up,
                layer.w_gate.m, layer.w_up.m, layer.w_gate.k,
            )?;
        } else if needs_mq_rotation(layer.w_gate.gpu_dtype) {
            // Dispatch: fused RMSNorm + FWHT rotation via RotationFamily.
            rotation.run(&ctx, gpu, RotationParams {
                x: &scratch.x,
                x_up: None,
                w_norm: Some(&layer.ffn_norm),
                x_plain: &scratch.tmp,
                x_rot: &scratch.x_rot,
                awq_scale: layer.w_gate.awq_scale.as_ref(),
                k: layer.w_gate.k,
                eps: config.norm_eps,
                batch_size: 1,
                variant: RotationVariant::WithRmsnorm,
            })?;
            llama::weight_gemv_prerotated(
                gpu, &layer.w_gate, &scratch.tmp, Some(&scratch.x_rot), &scratch.gate,
            )?;
            llama::weight_gemv_prerotated(
                gpu, &layer.w_up, &scratch.tmp, Some(&scratch.x_rot), &scratch.up,
            )?;
        } else {
            gpu.rmsnorm_f32(&scratch.x, &layer.ffn_norm, &scratch.tmp, config.norm_eps)?;
            llama::weight_gemv(gpu, &layer.w_gate, &scratch.tmp, &scratch.gate)?;
            llama::weight_gemv(gpu, &layer.w_up, &scratch.tmp, &scratch.up)?;
        }

        // ── SwiGLU + down projection + residual ─────────────
        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        llama::weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;
    }

    // ── Final norm + logits + sampling ──────────────────────
    gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, config.norm_eps)?;
    llama::weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;

    gpu.sample_top_p(
        &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
        config.vocab_size, temperature, top_p, rng_state,
        repeat_window, repeat_penalty,
    )
}
