// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 dense forward pass (free functions — hot-path static dispatch).
//!
//! Ported from the old branch's `forward_scratch` / `sliding_layer_decode` /
//! `full_layer_decode`, dropping MoE / E-series / vision. Per-token pipeline:
//!
//!   x = embed(token) * sqrt(dim)
//!   for each layer (sandwich RMSNorm around BOTH attn and FFN):
//!     residual = x
//!     n1 = input_layernorm(x)
//!     q = q_proj(n1); k = k_proj(n1)
//!       full + attention_k_eq_v: V = copy of k BEFORE k_norm, then weight-less
//!         RMSNorm on V (ones buffer); sliding: V = v_proj(n1)
//!     per-head q_norm / k_norm over head_dim; q *= sqrt(head_dim) (Gemma
//!       scale = 1.0 vs the kernel's 1/sqrt)
//!     RoPE: sliding → rope_f32(theta 10000, full rotate-half);
//!            full   → rope_partial_halved_f32(theta 1e6, n_rot = head_dim*0.25/2)
//!     KV write (Q8); attention_q8_0_kv_swa(window 1024 sliding / 0 full)
//!     attn = o_proj(attn_out); attn = post_attention_layernorm(attn)
//!     x = residual + attn
//!     residual = x
//!     n2 = pre_feedforward_layernorm(x)
//!     ffn = gelu_tanh(gate_proj(n2)) * up_proj(n2); ffn = down_proj(ffn)
//!     ffn = post_feedforward_layernorm(ffn)
//!     x = residual + ffn
//!     x *= layer_scalar
//!   x = norm(x); logits = lm_head(x); logits = logit_softcap(logits, 30)
//!
//! All RMSNorm here is plain `x * w` (baked at load — see `load_norm`).

use crate::config::{Gemma4Config, LayerType, RopeType};
use crate::gemma4::{FullLayerWeights, Gemma4State, Gemma4Weights, LayerWeights, SlidingLayerWeights};
use hipfire_runtime::llama::{weight_gemv, KvCache};
use rdna_compute::Gpu;

/// Decode one token (eager); returns the full logits vector.
pub fn decode_step(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    decode_step_body(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("gemma4: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state (pre
/// final-norm) to `capture[layer]` — used by the oracle dumper. Eager only.
pub fn decode_step_capture(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    decode_step_body(cfg, weights, state, gpu, token_id, position, Some(capture))
}

fn decode_step_body(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    let dim = cfg.dim;
    let eps = cfg.norm_eps;

    // 1) Embedding lookup → x, then scale by sqrt(dim).
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed hfq4g256: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed hfq4g128: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed q8: {e:?}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed f32: {e:?}"))?,
        EmbeddingFormat::Q4K => {
            return Err("gemma4: Q4K embedding format unsupported".to_string())
        }
    }
    gpu.scale_f32(&state.x, cfg.embed_scale)
        .map_err(|e| format!("gemma4: embed scale: {e:?}"))?;

    // 2) Update device pos_buf.
    let pos_i32 = position as i32;
    gpu.hip
        .memcpy_htod(&state.pos_buf, &pos_i32.to_ne_bytes())
        .map_err(|e| format!("gemma4: htod pos: {e:?}"))?;

    // 3) Per-layer forward.
    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        match (cfg.layer_types[layer_idx], &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                sliding_layer_decode(gpu, cfg, lw, position, slot, state)?;
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                full_layer_decode(gpu, cfg, lw, position, slot, state)?;
            }
            _ => {
                return Err(format!(
                    "gemma4 layer {layer_idx} type/weights mismatch"
                ))
            }
        }
        if let Some(cap) = capture.as_deref_mut() {
            let h = gpu
                .download_f32(&state.x)
                .map_err(|e| format!("gemma4 L{layer_idx}: capture download: {e:?}"))?;
            cap[layer_idx].extend_from_slice(&h);
        }
    }
    state.n_tokens = position as usize + 1;

    // 4) Final RMSNorm → tmp.
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, eps)
        .map_err(|e| format!("gemma4: final rmsnorm: {e:?}"))?;

    // 5) LM head → logits (tied embed bytes via lm_head.buf alias).
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("gemma4: lm_head: {e}"))?;

    // 6) Final logit softcap: logits = tanh(logits / cap) * cap.
    if cfg.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping)
            .map_err(|e| format!("gemma4: logit softcap: {e:?}"))?;
    }
    Ok(())
}

/// One sliding-window attention layer (head_dim 256, own v_proj, full RoPE).
fn sliding_layer_decode(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    lw: &SlidingLayerWeights,
    _pos: u32,
    kv_slot: usize,
    state: &mut Gemma4State,
) -> Result<(), String> {
    let dim = cfg.dim;
    let head_dim = cfg.sliding_head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.sliding_n_kv_heads;
    let eps = cfg.norm_eps;
    let dim_bytes = dim * 4;

    // residual = x
    gpu.hip
        .memcpy_dtod(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("gemma4 sliding: save residual: {e:?}"))?;

    // n1 = input_layernorm(x) → tmp.
    gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4 sliding: input rmsnorm: {e:?}"))?;

    // q/k/v projections.
    weight_gemv(gpu, &lw.q_proj, &state.tmp, &state.q)
        .map_err(|e| format!("gemma4 sliding: q_proj: {e}"))?;
    weight_gemv(gpu, &lw.k_proj, &state.tmp, &state.k)
        .map_err(|e| format!("gemma4 sliding: k_proj: {e}"))?;
    weight_gemv(gpu, &lw.v_proj, &state.tmp, &state.v)
        .map_err(|e| format!("gemma4 sliding: v_proj: {e}"))?;

    // Per-head q_norm / k_norm over head_dim, and weight-less V RMSNorm (ones).
    // (V uses the no-scale RMS pattern — matches full layers and the HF
    // sliding-layer Vcur = rms_norm(Vcur) on the v_norm path.)
    gpu.rmsnorm_batched(&state.q, &lw.q_norm, &state.q, n_heads, head_dim, eps)
        .map_err(|e| format!("gemma4 sliding: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.k, &lw.k_norm, &state.k, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 sliding: k_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.v, &state.v_norm_ones, &state.v, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 sliding: v_norm: {e:?}"))?;

    // Pre-scale Q by sqrt(head_dim) so the kernel's 1/sqrt(head_dim) cancels →
    // effective Gemma 4 scale of 1.0.
    gpu.scale_f32(&state.q, (head_dim as f32).sqrt())
        .map_err(|e| format!("gemma4 sliding: q scale: {e:?}"))?;

    // RoPE: full rotate-half over the whole head_dim, theta = 10000.
    gpu.rope_f32(
        &state.q,
        &state.k,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        cfg.sliding_rope_theta,
    )
    .map_err(|e| format!("gemma4 sliding: rope: {e:?}"))?;

    // KV write (Q8) + windowed attention (window = sliding_window).
    attn_q8_swa(
        gpu,
        &mut state.kv_sliding,
        kv_slot,
        &state.k,
        &state.v,
        &state.q,
        &state.attn_out,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        state.max_seq,
        cfg.sliding_window,
    )?;

    // o_proj → tmp, post_attention_layernorm(tmp), x = residual + tmp.
    finish_attn_and_ffn(gpu, cfg, state, &lw_common_sliding(lw))?;
    Ok(())
}

/// One full (global) attention layer (head_dim 512, K=V sharing, partial RoPE).
fn full_layer_decode(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    lw: &FullLayerWeights,
    _pos: u32,
    kv_slot: usize,
    state: &mut Gemma4State,
) -> Result<(), String> {
    let dim = cfg.dim;
    let head_dim = cfg.full_head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.full_n_kv_heads;
    let eps = cfg.norm_eps;
    let dim_bytes = dim * 4;
    let kv_bytes = n_kv * head_dim * 4;

    // residual = x
    gpu.hip
        .memcpy_dtod(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("gemma4 full: save residual: {e:?}"))?;

    // n1 = input_layernorm(x) → tmp.
    gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4 full: input rmsnorm: {e:?}"))?;

    // q/k projections.
    weight_gemv(gpu, &lw.q_proj, &state.tmp, &state.q)
        .map_err(|e| format!("gemma4 full: q_proj: {e}"))?;
    weight_gemv(gpu, &lw.k_proj, &state.tmp, &state.k)
        .map_err(|e| format!("gemma4 full: k_proj: {e}"))?;

    // V handling:
    //   attention_k_eq_v (12B): V = K's PRE-k_norm output (memcpy k → v BEFORE
    //     applying k_norm). Then weight-less RMSNorm on V.
    //   else: V = v_proj(n1).
    match &lw.v_proj {
        Some(vw) => {
            weight_gemv(gpu, vw, &state.tmp, &state.v)
                .map_err(|e| format!("gemma4 full: v_proj: {e}"))?;
        }
        None => {
            // CRITICAL ordering: capture V from the PRE-k_norm K output.
            gpu.hip
                .memcpy_dtod(&state.v.buf, &state.k.buf, kv_bytes)
                .map_err(|e| format!("gemma4 full: k→v copy: {e:?}"))?;
        }
    }

    // q_norm / k_norm over head_dim, weight-less V RMSNorm.
    gpu.rmsnorm_batched(&state.q, &lw.q_norm, &state.q, n_heads, head_dim, eps)
        .map_err(|e| format!("gemma4 full: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.k, &lw.k_norm, &state.k, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 full: k_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.v, &state.v_norm_ones, &state.v, n_kv, head_dim, eps)
        .map_err(|e| format!("gemma4 full: v_norm: {e:?}"))?;

    // Pre-scale Q by sqrt(head_dim=512).
    gpu.scale_f32(&state.q, (head_dim as f32).sqrt())
        .map_err(|e| format!("gemma4 full: q scale: {e:?}"))?;

    // Proportional / partial RoPE: rotate the first `partial_rotary_factor ×
    // head_dim` dims; theta = full_rope_theta. n_rot_pairs = factor*head_dim/2.
    let n_rot_pairs = match cfg.full_rope_type {
        RopeType::Proportional => {
            ((head_dim as f32) * cfg.full_partial_rotary_factor * 0.5) as usize
        }
        // Default → all pairs rotate (head_dim/2). rope_partial_halved with
        // n_rot_pairs = head_dim/2 == full rotate-half.
        RopeType::Default => head_dim / 2,
    };
    gpu.rope_partial_halved_f32(
        &state.q,
        &state.k,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        n_rot_pairs,
        cfg.full_rope_theta,
    )
    .map_err(|e| format!("gemma4 full: rope: {e:?}"))?;

    // KV write (Q8) + full causal attention (window = 0).
    attn_q8_swa(
        gpu,
        &mut state.kv_full,
        kv_slot,
        &state.k,
        &state.v,
        &state.q,
        &state.attn_out,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        state.max_seq,
        0,
    )?;

    finish_attn_and_ffn(gpu, cfg, state, &lw_common_full(lw))?;
    Ok(())
}

/// KV write (Q8) + windowed/full attention via `attention_q8_0_kv_swa`.
/// `window = 0` ⇒ full causal; `window > 0` ⇒ sliding window.
#[allow(clippy::too_many_arguments)]
fn attn_q8_swa(
    gpu: &mut Gpu,
    kv: &mut KvCache,
    kv_slot: usize,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    q: &rdna_compute::GpuTensor,
    attn_out: &rdna_compute::GpuTensor,
    pos_buf: &hip_bridge::DeviceBuffer,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    max_seq: usize,
    window: usize,
) -> Result<(), String> {
    gpu.kv_cache_write_q8_0(&kv.k_gpu[kv_slot], k, pos_buf, n_kv, head_dim)
        .map_err(|e| format!("gemma4: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&kv.v_gpu[kv_slot], v, pos_buf, n_kv, head_dim)
        .map_err(|e| format!("gemma4: kv write v: {e:?}"))?;
    gpu.attention_q8_0_kv_swa(
        q,
        &kv.k_gpu[kv_slot],
        &kv.v_gpu[kv_slot],
        attn_out,
        pos_buf,
        max_seq,
        n_heads,
        n_kv,
        head_dim,
        kv.physical_cap,
        window,
    )
    .map_err(|e| format!("gemma4: attention swa: {e:?}"))
}

/// Common per-layer tail shared by sliding + full layers: o_proj, post-attn
/// norm, attn residual add, pre-FFN norm, SwiGLU(gelu_tanh), post-FFN norm,
/// FFN residual add, learned layer_scalar. Operates on `state` scratch.
struct LayerTail<'a> {
    o_proj: &'a hipfire_runtime::llama::WeightTensor,
    post_attention_layernorm: &'a rdna_compute::GpuTensor,
    pre_feedforward_layernorm: &'a rdna_compute::GpuTensor,
    post_feedforward_layernorm: &'a rdna_compute::GpuTensor,
    gate_proj: &'a hipfire_runtime::llama::WeightTensor,
    up_proj: &'a hipfire_runtime::llama::WeightTensor,
    down_proj: &'a hipfire_runtime::llama::WeightTensor,
    layer_scalar_host: f32,
}

fn lw_common_sliding<'a>(lw: &'a SlidingLayerWeights) -> LayerTail<'a> {
    LayerTail {
        o_proj: &lw.o_proj,
        post_attention_layernorm: &lw.post_attention_layernorm,
        pre_feedforward_layernorm: &lw.pre_feedforward_layernorm,
        post_feedforward_layernorm: &lw.post_feedforward_layernorm,
        gate_proj: &lw.gate_proj,
        up_proj: &lw.up_proj,
        down_proj: &lw.down_proj,
        layer_scalar_host: lw.layer_scalar_host,
    }
}

fn lw_common_full<'a>(lw: &'a FullLayerWeights) -> LayerTail<'a> {
    LayerTail {
        o_proj: &lw.o_proj,
        post_attention_layernorm: &lw.post_attention_layernorm,
        pre_feedforward_layernorm: &lw.pre_feedforward_layernorm,
        post_feedforward_layernorm: &lw.post_feedforward_layernorm,
        gate_proj: &lw.gate_proj,
        up_proj: &lw.up_proj,
        down_proj: &lw.down_proj,
        layer_scalar_host: lw.layer_scalar_host,
    }
}

fn finish_attn_and_ffn(
    gpu: &mut Gpu,
    cfg: &Gemma4Config,
    state: &mut Gemma4State,
    tail: &LayerTail,
) -> Result<(), String> {
    let dim = cfg.dim;
    let eps = cfg.norm_eps;
    let dim_bytes = dim * 4;
    let ffn_hd = cfg.hidden_dim;

    // o_proj(attn_out) → tmp.
    weight_gemv(gpu, tail.o_proj, &state.attn_out, &state.tmp)
        .map_err(|e| format!("gemma4: o_proj: {e}"))?;

    // Sandwich post-attn norm (in-place on tmp).
    gpu.rmsnorm_f32(&state.tmp, tail.post_attention_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4: post_attn rmsnorm: {e:?}"))?;

    // x = residual + tmp.
    gpu.hip
        .memcpy_dtod(&state.x.buf, &state.residual.buf, dim_bytes)
        .map_err(|e| format!("gemma4: reset x: {e:?}"))?;
    gpu.add_inplace_f32(&state.x, &state.tmp)
        .map_err(|e| format!("gemma4: attn residual add: {e:?}"))?;

    // residual = x (FFN residual stream).
    gpu.hip
        .memcpy_dtod(&state.residual.buf, &state.x.buf, dim_bytes)
        .map_err(|e| format!("gemma4: save ffn residual: {e:?}"))?;

    // Pre-FFN norm → tmp.
    gpu.rmsnorm_f32(&state.x, tail.pre_feedforward_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4: pre_ffn rmsnorm: {e:?}"))?;

    // SwiGLU with gelu_pytorch_tanh: gate = gate_proj(tmp); up = up_proj(tmp);
    // hidden = gelu_tanh(gate) * up; ffn_out = down_proj(hidden).
    weight_gemv(gpu, tail.gate_proj, &state.tmp, &state.gate_ffn)
        .map_err(|e| format!("gemma4: gate_proj: {e}"))?;
    weight_gemv(gpu, tail.up_proj, &state.tmp, &state.up_ffn)
        .map_err(|e| format!("gemma4: up_proj: {e}"))?;
    gpu.gelu_tanh_f32(&state.gate_ffn, &state.ffn_hidden, ffn_hd)
        .map_err(|e| format!("gemma4: gelu_tanh: {e:?}"))?;
    gpu.mul_f32(&state.ffn_hidden, &state.up_ffn, &state.ffn_hidden)
        .map_err(|e| format!("gemma4: silu mul: {e:?}"))?;
    weight_gemv(gpu, tail.down_proj, &state.ffn_hidden, &state.ffn_out)
        .map_err(|e| format!("gemma4: down_proj: {e}"))?;

    // Sandwich post-FFN norm (ffn_out → tmp).
    gpu.rmsnorm_f32(&state.ffn_out, tail.post_feedforward_layernorm, &state.tmp, eps)
        .map_err(|e| format!("gemma4: post_ffn rmsnorm: {e:?}"))?;

    // x = residual + tmp.
    gpu.hip
        .memcpy_dtod(&state.x.buf, &state.residual.buf, dim_bytes)
        .map_err(|e| format!("gemma4: reset x (ffn): {e:?}"))?;
    gpu.add_inplace_f32(&state.x, &state.tmp)
        .map_err(|e| format!("gemma4: ffn residual add: {e:?}"))?;

    // Learned per-layer scalar multiplier (no-op = 1.0 when tensor absent).
    if tail.layer_scalar_host != 1.0 {
        gpu.scale_f32(&state.x, tail.layer_scalar_host)
            .map_err(|e| format!("gemma4: layer_scalar: {e:?}"))?;
    }
    Ok(())
}
