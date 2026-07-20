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
//! Non-expert linears are cohort-dependent: 8B-A1B keeps them Q8, while dense
//! `.mq4` projections are MQ4G256 (qt13/group_bytes136) and require exactly-once
//! FWHT before HFQ4-layout GEMMs. Routed experts (8B) are FWHT-pre-rotated
//! MQ4G256 via `rotate_x_mq_for` / `fused_silu_mul_rotate_mq_batched_for`.
//! Batched prefill (gfx1201 + HIPFIRE_LFM2_PREFILL_BATCH=1) currently admits
//! only the frozen 350M dense MQ4 fixture.

use crate::config::{Lfm2MoeConfig, MixerKind};
use crate::kernels;
use crate::lfm2moe::{
    lfm2_prefill_batch_capacity, AttnWeights, ConvWeights, DenseFfn, Ffn, Lfm2MoeLayerWeights,
    Lfm2MoeState, Lfm2MoeWeights, Mixer, MoeFfn,
};
use crate::redline_plan::{DecodeExecutionMode, RetainedFixtureEvidence};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama::{
    fused_rmsnorm_rotate_for_mq, fused_rmsnorm_rotate_mq_batched_for,
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemv,
    weight_gemv_prerotated, weight_gemv_prerotated_residual, weight_gemv_residual,
    weight_gemv_swiglu_residual, WeightTensor,
};
use rdna_compute::{replay::ReplayBackendRequest, DType, Gpu};

const MQ4_GROUP_BYTES: usize = 136;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RetainedDecodeRoute {
    Hip,
    Aql,
    Pm4,
}

fn retained_decode_route(
    eligible: bool,
    should_route_aql: bool,
    should_route_pm4: bool,
) -> RetainedDecodeRoute {
    if !eligible {
        RetainedDecodeRoute::Hip
    } else if should_route_aql {
        RetainedDecodeRoute::Aql
    } else if should_route_pm4 {
        RetainedDecodeRoute::Pm4
    } else {
        RetainedDecodeRoute::Hip
    }
}

fn complete_retained_replay(n_tokens: &mut usize, position: u32) {
    *n_tokens = position as usize + 1;
}

fn prefill_chunk_error(
    stage: &str,
    layer: Option<usize>,
    chunk_base: usize,
    chunk_len: usize,
    error: impl std::fmt::Debug,
) -> String {
    match layer {
        Some(layer) => format!(
            "lfm2moe prefill L{layer} chunk base={chunk_base} len={chunk_len} {stage}: {error:?}"
        ),
        None => {
            format!("lfm2moe prefill chunk base={chunk_base} len={chunk_len} {stage}: {error:?}")
        }
    }
}

fn require_q8_weight(weight: &WeightTensor, m: usize, k: usize, name: &str) -> Result<(), String> {
    if weight.gpu_dtype != DType::Q8_0 || weight.m != m || weight.k != k {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} Q8_0 [{m},{k}], got {:?} [{},{}]",
            weight.gpu_dtype, weight.m, weight.k
        ));
    }
    Ok(())
}

fn require_mq4_proj(weight: &WeightTensor, m: usize, k: usize, name: &str) -> Result<(), String> {
    if weight.gpu_dtype != DType::MQ4G256 || weight.m != m || weight.k != k {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} MQ4G256 [{m},{k}], got {:?} [{},{}]",
            weight.gpu_dtype, weight.m, weight.k
        ));
    }
    if weight.awq_scale.is_some() {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} without an AWQ sidecar"
        ));
    }
    if k == 0 || k % 256 != 0 {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} K multiple of 256, got {k}"
        ));
    }
    let expected = m
        .checked_mul(k / 256)
        .and_then(|g| g.checked_mul(MQ4_GROUP_BYTES))
        .ok_or_else(|| format!("lfm2moe: {name} group-byte overflow"))?;
    let got = weight.buf.buf.size();
    if got != expected {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} qt13/group_bytes{MQ4_GROUP_BYTES} payload {expected} bytes, got {got}"
        ));
    }
    Ok(())
}

fn require_f32_tensor(
    tensor: &rdna_compute::GpuTensor,
    numel: usize,
    name: &str,
) -> Result<(), String> {
    if tensor.dtype != DType::F32 || tensor.numel() != numel {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} F32 [{numel}], got {:?} [{}]",
            tensor.dtype,
            tensor.numel()
        ));
    }
    Ok(())
}

/// One-time structural validation for the exact LFM2.5-350M dense-MQ4 fixture
/// (md5 cb5284b8 provenance): arch11, hidden1024, heads16/8, hd64, q_dim1024,
/// kv_dim512, inter4608, vocab65536, theta1e6, and 16 dense layers.
pub fn validate_lfm_retained_fixture(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &Lfm2MoeState,
    arch_id: u32,
) -> Result<(), String> {
    if arch_id != crate::ARCH_ID {
        return Err(format!(
            "lfm2moe: retained fixture requires arch_id {}, got {arch_id}",
            crate::ARCH_ID
        ));
    }
    validate_350m_mq4_model(cfg, weights, state)?;
    validate_350m_retained_state(state)
}

fn validate_350m_mq4_model(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &Lfm2MoeState,
) -> Result<(), String> {
    const MIXERS: [MixerKind; 16] = [
        MixerKind::Conv,
        MixerKind::Conv,
        MixerKind::Attention,
        MixerKind::Conv,
        MixerKind::Conv,
        MixerKind::Attention,
        MixerKind::Conv,
        MixerKind::Conv,
        MixerKind::Attention,
        MixerKind::Conv,
        MixerKind::Attention,
        MixerKind::Conv,
        MixerKind::Attention,
        MixerKind::Conv,
        MixerKind::Attention,
        MixerKind::Conv,
    ];
    let shape_ok = cfg.hidden_size == 1024
        && cfg.vocab_size == 65_536
        && cfg.num_attention_heads == 16
        && cfg.num_key_value_heads == 8
        && cfg.head_dim == 64
        && cfg.q_dim() == 1024
        && cfg.kv_dim() == 512
        && cfg.rope_theta == 1_000_000.0
        && cfg.rms_norm_eps == 1e-5
        && cfg.intermediate_size == 4608
        && cfg.num_hidden_layers == 16
        && cfg.num_experts == 0
        && cfg.num_dense_layers == 16
        && cfg.conv_kernel_size == 3
        && cfg.tie_word_embeddings
        && cfg.layer_types.as_slice() == MIXERS;
    if !shape_ok {
        return Err(
            "lfm2moe: 350m.mq4 admission requires the frozen dense fixture shape".to_string(),
        );
    }
    if weights.layers.len() != cfg.num_hidden_layers {
        return Err(format!(
            "lfm2moe: layer weight count {} != config {}",
            weights.layers.len(),
            cfg.num_hidden_layers
        ));
    }
    if state.conv_states.len() != cfg.num_conv_layers()
        || state.kv.k_gpu.len() != cfg.num_attention_layers()
        || state.kv.v_gpu.len() != cfg.num_attention_layers()
    {
        return Err(format!(
            "lfm2moe: state topology mismatch: conv={} K={} V={}, expected conv={} attention={}",
            state.conv_states.len(),
            state.kv.k_gpu.len(),
            state.kv.v_gpu.len(),
            cfg.num_conv_layers(),
            cfg.num_attention_layers()
        ));
    }
    if !state.kv.quant_q8
        || state.kv.n_kv_heads != cfg.num_key_value_heads
        || state.kv.head_dim != cfg.head_dim
        || state.kv.max_seq != state.max_seq
        || state.kv.physical_cap < state.max_seq
    {
        return Err("lfm2moe: 350m.mq4 admission requires matching Q8 KV capacity".to_string());
    }

    require_q8_weight(&weights.lm_head, cfg.vocab_size, cfg.hidden_size, "lm_head")?;
    let embed_bytes = cfg
        .vocab_size
        .checked_mul(cfg.hidden_size / 32)
        .and_then(|blocks| blocks.checked_mul(34))
        .ok_or_else(|| "lfm2moe: embedding byte-size overflow".to_string())?;
    if weights.embed.buf.size() != embed_bytes {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires Q8 embedding bytes {embed_bytes}, got {}",
            weights.embed.buf.size()
        ));
    }
    require_f32_tensor(&weights.embedding_norm, cfg.hidden_size, "embedding_norm")?;

    let mut conv_slots = vec![false; cfg.num_conv_layers()];
    let mut kv_slots = vec![false; cfg.num_attention_layers()];
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        require_f32_tensor(
            &layer.operator_norm,
            cfg.hidden_size,
            &format!("L{layer_idx}.operator_norm"),
        )?;
        require_f32_tensor(
            &layer.ffn_norm,
            cfg.hidden_size,
            &format!("L{layer_idx}.ffn_norm"),
        )?;
        match (&layer.mixer, MIXERS[layer_idx]) {
            (Mixer::Conv(conv), MixerKind::Conv) => {
                require_mq4_proj(
                    &conv.in_proj,
                    3 * cfg.hidden_size,
                    cfg.hidden_size,
                    &format!("L{layer_idx}.conv.in_proj"),
                )?;
                require_mq4_proj(
                    &conv.out_proj,
                    cfg.hidden_size,
                    cfg.hidden_size,
                    &format!("L{layer_idx}.conv.out_proj"),
                )?;
                require_f32_tensor(
                    &conv.conv_weight,
                    cfg.hidden_size * cfg.conv_kernel_size,
                    &format!("L{layer_idx}.conv.weight"),
                )?;
                let occupied = conv_slots.get_mut(conv.conv_state_idx).ok_or_else(|| {
                    format!(
                        "lfm2moe: L{layer_idx} conv_state_idx {} out of range",
                        conv.conv_state_idx
                    )
                })?;
                if *occupied {
                    return Err(format!(
                        "lfm2moe: duplicate conv_state_idx {}",
                        conv.conv_state_idx
                    ));
                }
                *occupied = true;
            }
            (Mixer::Attention(attn), MixerKind::Attention) => {
                for (weight, m, name) in [
                    (&attn.wq, cfg.q_dim(), "wq"),
                    (&attn.wk, cfg.kv_dim(), "wk"),
                    (&attn.wv, cfg.kv_dim(), "wv"),
                    (&attn.wo, cfg.hidden_size, "wo"),
                ] {
                    let k = if name == "wo" {
                        cfg.q_dim()
                    } else {
                        cfg.hidden_size
                    };
                    require_mq4_proj(weight, m, k, &format!("L{layer_idx}.attention.{name}"))?;
                }
                require_f32_tensor(
                    &attn.q_norm,
                    cfg.head_dim,
                    &format!("L{layer_idx}.attention.q_norm"),
                )?;
                require_f32_tensor(
                    &attn.k_norm,
                    cfg.head_dim,
                    &format!("L{layer_idx}.attention.k_norm"),
                )?;
                let occupied = kv_slots.get_mut(attn.kv_idx).ok_or_else(|| {
                    format!("lfm2moe: L{layer_idx} kv_idx {} out of range", attn.kv_idx)
                })?;
                if *occupied {
                    return Err(format!("lfm2moe: duplicate kv_idx {}", attn.kv_idx));
                }
                *occupied = true;
            }
            _ => {
                return Err(format!(
                    "lfm2moe: L{layer_idx} mixer does not match frozen 350m topology"
                ));
            }
        }
        let Ffn::Dense(dense) = &layer.ffn else {
            return Err(format!(
                "lfm2moe: L{layer_idx} must use dense FFN for 350m.mq4"
            ));
        };
        require_mq4_proj(
            &dense.w1,
            cfg.intermediate_size,
            cfg.hidden_size,
            &format!("L{layer_idx}.ffn.w1"),
        )?;
        require_mq4_proj(
            &dense.w3,
            cfg.intermediate_size,
            cfg.hidden_size,
            &format!("L{layer_idx}.ffn.w3"),
        )?;
        require_mq4_proj(
            &dense.w2,
            cfg.hidden_size,
            cfg.intermediate_size,
            &format!("L{layer_idx}.ffn.w2"),
        )?;
    }
    if conv_slots.iter().any(|occupied| !occupied) || kv_slots.iter().any(|occupied| !occupied) {
        return Err("lfm2moe: incomplete conv/KV slot coverage".to_string());
    }
    for (index, conv_state) in state.conv_states.iter().enumerate() {
        require_f32_tensor(
            conv_state,
            cfg.hidden_size * (cfg.conv_kernel_size - 1),
            &format!("conv_state[{index}]"),
        )?;
    }
    Ok(())
}

fn validate_350m_retained_state(state: &Lfm2MoeState) -> Result<(), String> {
    // Exact retained flash surface: max_seq/physical_cap 2048, always-flash
    // mode, and decode partials F32 length 16*ceil(2048/128)*(2+64)=16896.
    // HIPFIRE_ATTN_FLASH=never stays ordinary-HIP A/B but is retained-ineligible.
    if state.max_seq != 2048 {
        return Err("lfm2moe: 350m.mq4 admission requires max_seq 2048".to_string());
    }
    if state.kv.physical_cap != 2048 {
        return Err("lfm2moe: 350m.mq4 admission requires physical_cap 2048".to_string());
    }
    if state.flash_mode != 2 {
        return Err("lfm2moe: 350m.mq4 admission requires frozen flash_mode 2".to_string());
    }
    require_f32_tensor(&state.flash_partials, 16_896, "flash_partials")
}

fn checked_prefill_i32_product(label: &str, factors: &[usize]) -> Result<usize, String> {
    let value = factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| format!("lfm2moe: {label} overflow"))
    })?;
    if value > i32::MAX as usize {
        return Err(format!("lfm2moe: {label}={value} exceeds i32::MAX"));
    }
    Ok(value)
}

fn validate_prefill_launch_dimensions(cfg: &Lfm2MoeConfig, chunk_len: usize) -> Result<(), String> {
    for (label, factors) in [
        ("N*n_heads", vec![chunk_len, cfg.num_attention_heads]),
        ("N*n_kv_heads", vec![chunk_len, cfg.num_key_value_heads]),
        ("N*k_top", vec![chunk_len, cfg.num_experts_per_tok.max(1)]),
        ("N*hidden", vec![chunk_len, cfg.hidden_size]),
        ("N*3*hidden", vec![chunk_len, 3, cfg.hidden_size]),
        ("N*q_dim", vec![chunk_len, cfg.q_dim()]),
        ("N*kv_dim", vec![chunk_len, cfg.kv_dim()]),
        (
            "N*intermediate_size",
            vec![chunk_len, cfg.intermediate_size],
        ),
    ] {
        checked_prefill_i32_product(label, &factors)?;
    }
    for (label, value) in [
        ("hidden", cfg.hidden_size),
        ("q_dim", cfg.q_dim()),
        ("kv_dim", cfg.kv_dim()),
        ("intermediate_size", cfg.intermediate_size),
        ("num_attention_heads", cfg.num_attention_heads),
        ("num_key_value_heads", cfg.num_key_value_heads),
        ("head_dim", cfg.head_dim),
    ] {
        checked_prefill_i32_product(label, &[value])?;
    }
    Ok(())
}

/// Batched prompt prefill for LFM2.5 on gfx1201.
///
/// Ingests non-empty `token_ids` beginning at absolute `start_pos`, advances
/// KV, convolution state, and `state.n_tokens` as sequential decode steps
/// would, and returns only the final token's host logits.
pub fn forward_prefill_batch(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_ids: &[u32],
    start_pos: u32,
) -> Result<Vec<f32>, String> {
    gpu.replay.set_forward_eligible(false);
    forward_prefill_batch_impl(cfg, weights, state, gpu, token_ids, start_pos, None)
}

#[doc(hidden)]
pub fn forward_prefill_batch_capture(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_ids: &[u32],
    start_pos: u32,
    capture: &mut [Vec<f32>],
) -> Result<Vec<f32>, String> {
    gpu.replay.set_forward_eligible(false);
    forward_prefill_batch_impl(
        cfg,
        weights,
        state,
        gpu,
        token_ids,
        start_pos,
        Some(capture),
    )
}

fn forward_prefill_batch_impl(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_ids: &[u32],
    start_pos: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("lfm2moe: batched prefill requires non-empty token_ids".to_string());
    }
    if !gpu.arch_caps.is_gfx1201() {
        return Err(format!(
            "lfm2moe: batched prefill requires gfx1201, got {}",
            gpu.arch
        ));
    }
    if std::env::var("HIPFIRE_LFM2_PREFILL_BATCH").ok().as_deref() != Some("1") {
        return Err("lfm2moe: batched prefill requires HIPFIRE_LFM2_PREFILL_BATCH=1".to_string());
    }
    if start_pos as usize != state.n_tokens {
        return Err(format!(
            "lfm2moe: batched prefill start_pos {start_pos} != state.n_tokens {}",
            state.n_tokens
        ));
    }
    for (index, &token_id) in token_ids.iter().enumerate() {
        if token_id > i32::MAX as u32 || token_id as usize >= cfg.vocab_size {
            return Err(format!(
                "lfm2moe: invalid token_ids[{index}]={token_id} for vocab {}",
                cfg.vocab_size
            ));
        }
    }
    let end = (start_pos as usize)
        .checked_add(token_ids.len())
        .ok_or_else(|| "lfm2moe: batched prefill end position overflow".to_string())?;
    if end > i32::MAX as usize {
        return Err(format!(
            "lfm2moe: batched prefill exclusive end {end} exceeds i32::MAX"
        ));
    }
    if end > state.max_seq || end > state.kv.max_seq || end > state.kv.physical_cap {
        return Err(format!(
            "lfm2moe: batched prefill end {end} exceeds state/KV capacity {}/{}/{}",
            state.max_seq, state.kv.max_seq, state.kv.physical_cap
        ));
    }
    validate_350m_mq4_model(cfg, weights, state)?;
    let prospective_max_batch = match state.prefill_batch.as_ref() {
        Some(scratch) => scratch.max_batch,
        None => lfm2_prefill_batch_capacity(cfg, state.max_seq)?,
    };
    validate_prefill_launch_dimensions(cfg, token_ids.len().min(prospective_max_batch))?;
    if let Some(capture) = capture.as_ref() {
        if capture.len() != weights.layers.len() {
            return Err(format!(
                "lfm2moe: prefill capture layers {} != weights {}",
                capture.len(),
                weights.layers.len()
            ));
        }
    }

    state.ensure_prefill_batch(gpu, cfg)?;
    let max_batch = state
        .prefill_batch
        .as_ref()
        .ok_or_else(|| "lfm2moe: prefill scratch missing".to_string())?
        .max_batch;
    let mut offset = 0usize;
    while offset < token_ids.len() {
        let chunk_len = max_batch.min(token_ids.len() - offset);
        let chunk_base = (start_pos as usize)
            .checked_add(offset)
            .ok_or_else(|| "lfm2moe: chunk base overflow".to_string())?;
        let emit_head = offset + chunk_len == token_ids.len();
        let chunk = &token_ids[offset..offset + chunk_len];
        let chunk_base_u32 = u32::try_from(chunk_base)
            .map_err(|_| format!("lfm2moe: chunk base {chunk_base} exceeds u32"))?;
        if let Some(capture) = capture.as_deref_mut() {
            forward_prefill_chunk_impl(
                cfg,
                weights,
                state,
                gpu,
                chunk,
                chunk_base_u32,
                emit_head,
                Some(capture),
            )?;
        } else {
            forward_prefill_chunk(
                cfg,
                weights,
                state,
                gpu,
                chunk,
                chunk_base_u32,
                emit_head,
            )?;
        }
        offset += chunk_len;
    }
    let logits = gpu.download_f32(&state.logits).map_err(|e| {
        prefill_chunk_error(
            "download logits",
            None,
            start_pos as usize,
            token_ids.len(),
            e,
        )
    })?;
    if logits.len() != cfg.vocab_size {
        return Err(format!(
            "lfm2moe: downloaded logits len {} != vocab {}",
            logits.len(),
            cfg.vocab_size
        ));
    }
    Ok(logits)
}

fn forward_prefill_chunk(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_ids: &[u32],
    start_pos: u32,
    emit_head: bool,
) -> Result<(), String> {
    forward_prefill_chunk_impl(
        cfg, weights, state, gpu, token_ids, start_pos, emit_head, None,
    )
}

fn forward_prefill_chunk_impl(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_ids: &[u32],
    start_pos: u32,
    emit_head: bool,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let n = token_ids.len();
    if n == 0 {
        return Err("lfm2moe: forward_prefill_chunk received an empty chunk".to_string());
    }
    let p = start_pos as usize;
    let end = p
        .checked_add(n)
        .ok_or_else(|| "lfm2moe: chunk end overflow".to_string())?;
    let scratch = state
        .prefill_batch
        .as_ref()
        .ok_or_else(|| "lfm2moe: prefill scratch missing".to_string())?;
    let capture_postmixer =
        capture.is_some() && std::env::var_os("HIPFIRE_LFM2_CAPTURE_POSTMIXER").is_some();
    if scratch.flash_partials_batch == 0 {
        return Err("lfm2moe: flash partials sub-batch capacity is zero".to_string());
    }
    if n > scratch.max_batch {
        return Err(format!(
            "lfm2moe: chunk len {n} exceeds scratch max_batch {}",
            scratch.max_batch
        ));
    }

    let token_ids_i32: Vec<i32> = token_ids
        .iter()
        .map(|&id| i32::try_from(id).unwrap())
        .collect();
    let positions_i32: Vec<i32> = (0..n).map(|i| i32::try_from(p + i).unwrap()).collect();
    let token_bytes = unsafe {
        std::slice::from_raw_parts(
            token_ids_i32.as_ptr().cast::<u8>(),
            token_ids_i32.len() * std::mem::size_of::<i32>(),
        )
    };
    let position_bytes = unsafe {
        std::slice::from_raw_parts(
            positions_i32.as_ptr().cast::<u8>(),
            positions_i32.len() * std::mem::size_of::<i32>(),
        )
    };
    gpu.hip
        .memcpy_htod(&scratch.token_ids_batch.buf, token_bytes)
        .map_err(|e| prefill_chunk_error("upload token ids", None, p, n, e))?;
    gpu.hip
        .memcpy_htod(&scratch.positions_batch.buf, position_bytes)
        .map_err(|e| prefill_chunk_error("upload positions", None, p, n, e))?;
    gpu.embedding_lookup_q8_batched(
        &weights.embed,
        &scratch.h_batch,
        &scratch.token_ids_batch,
        n,
        cfg.hidden_size,
    )
    .map_err(|e| prefill_chunk_error("embedding", None, p, n, e))?;

    let dense_gate_n = scratch
        .dense_gate_batch
        .sub_offset(0, n * cfg.intermediate_size);
    let dense_up_n = scratch
        .dense_up_batch
        .sub_offset(0, n * cfg.intermediate_size);
    let dense_act_rot_n = scratch
        .dense_act_rot_batch
        .sub_offset(0, n * cfg.intermediate_size);

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
            // Operator-norm + exactly-once FWHT into operator_x_rot_batch for MQ4 projections.
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &scratch.h_batch,
                &layer.operator_norm,
                match &layer.mixer {
                    Mixer::Conv(c) => &c.in_proj,
                    Mixer::Attention(a) => &a.wq,
                },
                &scratch.operator_x_rot_batch,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                n,
            )
            .map_err(|e| {
                prefill_chunk_error("operator rmsnorm+rotate", Some(layer_idx), p, n, e)
            })?;

        match &layer.mixer {
            Mixer::Conv(conv) => {
                // Zero-then-residual so conv in_proj hits HFQ4 residual WMMA (gfx12).
                let conv_bcx_n = scratch
                    .conv_bcx_batch
                    .sub_offset(0, n * 3 * cfg.hidden_size);
                gpu.fill_f32(&conv_bcx_n, 0.0).map_err(|e| {
                    prefill_chunk_error("conv in_proj zero", Some(layer_idx), p, n, e)
                })?;
                kernels::lfm2_350m_residual_wmma_gfx1201(
                    gpu,
                    &conv.in_proj.buf,
                    &scratch.operator_x_rot_batch,
                    &scratch.conv_bcx_batch,
                    3 * cfg.hidden_size,
                    cfg.hidden_size,
                    n,
                )
                .map_err(|e| prefill_chunk_error("conv in_proj", Some(layer_idx), p, n, e))?;
                kernels::conv1d_gated_scan_n(
                    gpu,
                    &scratch.conv_bcx_batch,
                    &state.conv_states[conv.conv_state_idx],
                    &conv.conv_weight,
                    &scratch.conv_y_batch,
                    n,
                    cfg.hidden_size,
                )
                .map_err(|e| prefill_chunk_error("conv scan", Some(layer_idx), p, n, e))?;
                rotate_x_mq_batched_for(
                    gpu,
                    &conv.out_proj,
                    &scratch.conv_y_batch,
                    &scratch.conv_y_rot_batch,
                    cfg.hidden_size,
                    n,
                )
                .map_err(|e| prefill_chunk_error("conv out rotate", Some(layer_idx), p, n, e))?;
                kernels::lfm2_350m_residual_wmma_gfx1201(
                    gpu,
                    &conv.out_proj.buf,
                    &scratch.conv_y_rot_batch,
                    &scratch.h_batch,
                    cfg.hidden_size,
                    cfg.hidden_size,
                    n,
                )
                .map_err(|e| prefill_chunk_error("conv out_proj", Some(layer_idx), p, n, e))?;
            }
            Mixer::Attention(attn) => {
                gpu.gemm_qkv_hfq4g256(
                    &attn.wq.buf,
                    &attn.wk.buf,
                    &attn.wv.buf,
                    &scratch.operator_x_rot_batch,
                    &scratch.fa_q_batch,
                    &scratch.fa_k_batch,
                    &scratch.fa_v_batch,
                    cfg.q_dim(),
                    cfg.kv_dim(),
                    cfg.kv_dim(),
                    cfg.hidden_size,
                    n,
                )
                .map_err(|e| prefill_chunk_error("attention qkv", Some(layer_idx), p, n, e))?;
                gpu.rmsnorm_batched(
                    &scratch.fa_q_batch,
                    &attn.q_norm,
                    &scratch.fa_q_batch,
                    n * cfg.num_attention_heads,
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                )
                .map_err(|e| prefill_chunk_error("attention q_norm", Some(layer_idx), p, n, e))?;
                gpu.rmsnorm_batched(
                    &scratch.fa_k_batch,
                    &attn.k_norm,
                    &scratch.fa_k_batch,
                    n * cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                )
                .map_err(|e| prefill_chunk_error("attention k_norm", Some(layer_idx), p, n, e))?;
                gpu.rope_batched_f32(
                    &scratch.fa_q_batch,
                    &scratch.fa_k_batch,
                    &scratch.positions_batch,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.rope_theta,
                    n,
                )
                .map_err(|e| prefill_chunk_error("attention rope", Some(layer_idx), p, n, e))?;
                let kv_idx = attn.kv_idx;
                gpu.kv_cache_write_q8_0_batched(
                    &state.kv.k_gpu[kv_idx],
                    &scratch.fa_k_batch,
                    &scratch.positions_batch,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    n,
                )
                .map_err(|e| {
                    prefill_chunk_error("attention K cache write", Some(layer_idx), p, n, e)
                })?;
                gpu.kv_cache_write_q8_0_batched(
                    &state.kv.v_gpu[kv_idx],
                    &scratch.fa_v_batch,
                    &scratch.positions_batch,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    n,
                )
                .map_err(|e| {
                    prefill_chunk_error("attention V cache write", Some(layer_idx), p, n, e)
                })?;
                gpu.attention_flash_q8_0_batched_masked(
                    &scratch.fa_q_batch,
                    &state.kv.k_gpu[kv_idx],
                    &state.kv.v_gpu[kv_idx],
                    &scratch.fa_attn_out_batch,
                    &scratch.positions_batch,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    end,
                    end,
                    n,
                    &scratch.fa_partials_batch,
                    None,
                    0,
                    0,
                )
                .map_err(|e| prefill_chunk_error("attention flash", Some(layer_idx), p, n, e))?;
                rotate_x_mq_batched_for(
                    gpu,
                    &attn.wo,
                    &scratch.fa_attn_out_batch,
                    &scratch.fa_attn_out_rot_batch,
                    cfg.q_dim(),
                    n,
                )
                .map_err(|e| {
                    prefill_chunk_error("attention out rotate", Some(layer_idx), p, n, e)
                })?;
                kernels::lfm2_350m_residual_wmma_gfx1201(
                    gpu,
                    &attn.wo.buf,
                    &scratch.fa_attn_out_rot_batch,
                    &scratch.h_batch,
                    cfg.hidden_size,
                    cfg.q_dim(),
                    n,
                )
                .map_err(|e| prefill_chunk_error("attention out_proj", Some(layer_idx), p, n, e))?;
            }
        }
        if capture_postmixer {
            if let Some(capture) = capture.as_deref_mut() {
                let hidden = gpu.download_f32(&scratch.h_batch).map_err(|e| {
                    prefill_chunk_error("capture postmixer hidden", Some(layer_idx), p, n, e)
                })?;
                capture[layer_idx].extend_from_slice(&hidden[..n * cfg.hidden_size]);
            }
        }

        // FFN-norm + exactly-once FWHT into ffn_x_rot_batch.
        let Ffn::Dense(dense) = &layer.ffn else {
            return Err(format!(
                "lfm2moe prefill L{layer_idx} chunk base={p} len={n}: non-dense FFN escaped admission"
            ));
        };
        fused_rmsnorm_rotate_mq_batched_for(
            gpu,
            &scratch.h_batch,
            &layer.ffn_norm,
            &dense.w1,
            &scratch.ffn_x_rot_batch,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            n,
        )
        .map_err(|e| prefill_chunk_error("ffn rmsnorm+rotate", Some(layer_idx), p, n, e))?;
        kernels::lfm2_350m_gate_up_wmma_gfx1201(
            gpu,
            &dense.w1.buf,
            &dense.w3.buf,
            &scratch.ffn_x_rot_batch,
            &scratch.dense_gate_batch,
            &scratch.dense_up_batch,
            cfg.intermediate_size,
            cfg.intermediate_size,
            cfg.hidden_size,
            n,
        )
        .map_err(|e| prefill_chunk_error("ffn gate_up", Some(layer_idx), p, n, e))?;
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &dense.w2,
            &dense_gate_n,
            &dense_up_n,
            &dense_act_rot_n,
            cfg.intermediate_size,
            n,
        )
        .map_err(|e| prefill_chunk_error("ffn silu_mul+rotate", Some(layer_idx), p, n, e))?;
        kernels::lfm2_350m_residual_wmma_gfx1201(
            gpu,
            &dense.w2.buf,
            &dense_act_rot_n,
            &scratch.h_batch,
            cfg.hidden_size,
            cfg.intermediate_size,
            n,
        )
        .map_err(|e| prefill_chunk_error("ffn down", Some(layer_idx), p, n, e))?;
        if !capture_postmixer {
            if let Some(capture) = capture.as_deref_mut() {
                let hidden = gpu
                    .download_f32(&scratch.h_batch)
                    .map_err(|e| prefill_chunk_error("capture hidden", Some(layer_idx), p, n, e))?;
                capture[layer_idx].extend_from_slice(&hidden[..n * cfg.hidden_size]);
            }
        }
    }

    state.n_tokens = end;
    if emit_head {
        let row_bytes = cfg
            .hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "lfm2moe: final row byte-size overflow".to_string())?;
        let row_offset = (n - 1)
            .checked_mul(row_bytes)
            .ok_or_else(|| "lfm2moe: final row offset overflow".to_string())?;
        gpu.memcpy_dtod_at_auto(&state.h.buf, 0, &scratch.h_batch.buf, row_offset, row_bytes)
            .map_err(|e| prefill_chunk_error("final row copy", None, p, n, e))?;
        gpu.rmsnorm_f32(
            &state.h,
            &weights.embedding_norm,
            &state.final_norm_buf,
            cfg.rms_norm_eps,
        )
        .map_err(|e| prefill_chunk_error("final rmsnorm", None, p, n, e))?;
        weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
            .map_err(|e| prefill_chunk_error("lm_head", None, p, n, e))?;
    }
    Ok(())
}

/// Decode one token and return the full host logits vector.
///
/// Product paths that consume logits on-device should call
/// [`decode_step_device`] and avoid the 256 KiB full-vocabulary readback.
pub fn decode_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    retained_fixture_evidence: RetainedFixtureEvidence,
    mode: DecodeExecutionMode,
) -> Result<Vec<f32>, String> {
    if graph_enabled() {
        return decode_step_with_graph(cfg, weights, state, gpu, token_id, position);
    }
    decode_step_device(
        cfg,
        weights,
        state,
        gpu,
        token_id,
        position,
        retained_fixture_evidence,
        mode,
    )?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
}

/// Decode one token while leaving the full logits vector device-resident.
///
/// The retained tape starts with the pointer-driven n=1 Q8 embedding, so the
/// only per-token work outside AQL/PM4 is two synchronous 4-byte scalar uploads.
/// `HIPFIRE_LFM2_GRAPH=1` remains an experimental compatibility path and still
/// uses its legacy logits readback internally.
pub fn decode_step_device(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    retained_fixture_evidence: RetainedFixtureEvidence,
    mode: DecodeExecutionMode,
) -> Result<(), String> {
    let graph = graph_enabled();
    if graph {
        return decode_step_with_graph(cfg, weights, state, gpu, token_id, position).map(drop);
    }
    let lowered = lfm2_forward_lowered_enabled();
    let fusion_enabled = lfm2_decode_fusion_enabled(retained_fixture_evidence, &gpu.arch);
    let retained_eligible = state.flash_mode == 2
        && crate::redline_plan::retained_route_eligible(
            retained_fixture_evidence,
            &gpu.arch,
            gpu.replay.request() != ReplayBackendRequest::Hip,
            mode,
            position,
            state.n_tokens,
            graph,
            lowered,
            fusion_enabled,
        );

    stage_decode_inputs(state, gpu, token_id, position)?;

    gpu.replay.set_forward_eligible(retained_eligible);
    if retained_eligible {
        gpu.replay
            .begin_auto_capture_if_armed()
            .map_err(|reason| format!("lfm2moe: begin replay capture: {reason}"))?;
    }
    let route = retained_decode_route(
        retained_eligible,
        gpu.replay.should_route_aql(),
        gpu.replay.should_route_pm4(),
    );

    match route {
        RetainedDecodeRoute::Aql => {
            if let Err(reason) = unsafe { gpu.replay.replay_linear_aql(position as usize) } {
                gpu.replay
                    .poison(format!("prepared AQL replay failed: {reason}"));
                return Err(format!("lfm2moe: prepared AQL replay failed: {reason}"));
            }
            complete_retained_replay(&mut state.n_tokens, position);
        }
        RetainedDecodeRoute::Pm4 => {
            if let Err(reason) = unsafe { gpu.replay.replay_pm4(position as usize) } {
                gpu.replay
                    .poison(format!("prepared PM4 replay failed: {reason}"));
                return Err(format!("lfm2moe: prepared PM4 replay failed: {reason}"));
            }
            complete_retained_replay(&mut state.n_tokens, position);
        }
        RetainedDecodeRoute::Hip => {
            run_decode_embedding(cfg, weights, state, gpu)?;
            decode_step_layers_and_head(
                cfg,
                weights,
                state,
                gpu,
                position,
                true,
                fusion_enabled,
                None,
            )?;
        }
    }

    if retained_eligible && gpu.replay.should_auto_finalize_capture() {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("lfm2moe: synchronize replay capture: {e:?}"))?;
        gpu.replay
            .finish_capture()
            .map_err(|reason| format!("lfm2moe: finish replay capture: {reason}"))?;
        let prepare = if gpu.replay.uses_pm4_transport() {
            let launches = gpu.replay.recorded_launches().len();
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        } else {
            gpu.replay
                .prepare_linear_aql(gpu.device_id as usize)
                .map(|_| ())
        };
        if let Err(reason) = prepare {
            gpu.replay
                .poison(format!("Redline prepare after warmup failed: {reason}"));
            eprintln!("[redline] falling back to HIP: {reason}");
        }
    }
    Ok(())
}

/// Prefill a single non-final prompt token: full mixer/FFN layer stack + state
/// advance (KV, conv, n_tokens), WITHOUT the final RMSNorm + lm_head + logits
/// D2H. The intermediate token's logits are never consumed by the prefill loop,
/// so skipping the 128000-vocab lm_head GEMV + full-vocab download is a pure,
/// output-identical speedup. When the graph path is explicitly selected
/// (HIPFIRE_LFM2_GRAPH=1) it is preserved (no head-elision). Used by the
/// gfx1201-gated LFM prefill loop (Phase 0 head-elision).
pub fn decode_step_prefill(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    gpu.replay.set_forward_eligible(false);
    if graph_enabled() {
        // Explicit graph mode: do not bypass the user-selected path. Run the
        // full decode (graph) and discard the logits; head-elision is a
        // non-graph optimization.
        return decode_step(
            cfg,
            weights,
            state,
            gpu,
            token_id,
            position,
            RetainedFixtureEvidence::ABSENT,
            DecodeExecutionMode::Prefill,
        )
        .map(|_| ());
    }
    decode_step_inner(cfg, weights, state, gpu, token_id, position, false, None)
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
fn resolve_lfm2_decode_fusion_request(value: Option<&std::ffi::OsStr>) -> bool {
    match value {
        None => true,
        Some(value) => matches!(value.to_str(), Some("1")),
    }
}

fn lfm2_decode_fusion_requested() -> bool {
    use std::sync::LazyLock;
    static REQUESTED: LazyLock<bool> = LazyLock::new(|| {
        resolve_lfm2_decode_fusion_request(
            std::env::var_os("HIPFIRE_LFM2_GFX1201_DECODE_FUSION").as_deref(),
        )
    });
    *REQUESTED
}

pub(crate) fn lfm2_decode_fusion_enabled(
    fixture_evidence: RetainedFixtureEvidence,
    gpu_arch: &str,
) -> bool {
    let graph = graph_enabled();
    let lowered = lfm2_forward_lowered_enabled();
    let enabled = lfm2_decode_fusion_requested()
        && crate::redline_plan::decode_fusion_eligible(fixture_evidence, gpu_arch, graph, lowered);
    if enabled {
        use std::sync::Once;
        static ANNOUNCE: Once = Once::new();
        ANNOUNCE.call_once(|| {
            eprintln!("[lfm2moe] exact gfx1201 350m decode fusion active: shared RMSNorm+FWHT");
        });
    }
    enabled
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
    gpu.replay.set_forward_eligible(false);
    decode_step_inner(
        cfg,
        weights,
        state,
        gpu,
        token_id,
        position,
        true,
        Some(capture),
    )
}

fn decode_step_inner(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    emit_head: bool,
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    prepare_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    decode_step_layers_and_head(
        cfg, weights, state, gpu, position, emit_head, false, capture,
    )
}

/// Upload the per-token position and token-id scalars outside retained capture.
///
/// `hipMemcpy` is synchronous, so a prepared AQL/PM4 replay can consume these
/// stable buffers without a separate device-wide synchronization.
pub fn stage_decode_inputs(
    state: &Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    gpu.replay.set_forward_eligible(false);
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&state.token_buf.buf, &token_id.to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod token: {e:?}"))
}

fn run_decode_embedding(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &Lfm2MoeState,
    gpu: &mut Gpu,
) -> Result<(), String> {
    gpu.embedding_lookup_q8_batched(
        &weights.embed,
        &state.h,
        &state.token_buf,
        1,
        cfg.hidden_size,
    )
    .map_err(|e| format!("lfm2moe: batched embed lookup: {e:?}"))
}

/// Stage scalars and run the embedding directly. Used by non-retained prefill
/// and the experimental hipGraph path, where embedding remains outside capture.
pub fn prepare_decode_inputs(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    stage_decode_inputs(state, gpu, token_id, position)?;
    run_decode_embedding(cfg, weights, state, gpu)
}

fn validate_lfm_speculative_graph(graph: Option<&std::ffi::OsStr>) -> Result<(), String> {
    if graph == Some(std::ffi::OsStr::new("1")) {
        return Err(
            "lfm2moe: speculative direct-HIP execution rejects HIPFIRE_LFM2_GRAPH=1"
                .to_string(),
        );
    }
    Ok(())
}

/// Run one full-model speculative target token through direct HIP. This entry
/// rejects hipGraph and never enters retained replay.
pub(crate) fn decode_step_speculative_device(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    retained_fixture_evidence: RetainedFixtureEvidence,
) -> Result<(), String> {
    validate_lfm_speculative_graph(std::env::var_os("HIPFIRE_LFM2_GRAPH").as_deref())?;
    let fusion_enabled = lfm2_decode_fusion_enabled(retained_fixture_evidence, &gpu.arch);
    gpu.replay.set_forward_eligible(false);
    prepare_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    decode_step_layers_and_head(
        cfg,
        weights,
        state,
        gpu,
        position,
        true,
        fusion_enabled,
        None,
    )
}

/// Execute the scalar-staged embedding + layer/head region directly through HIP.
///
/// This bypasses retained routing and lifecycle transitions so shadow
/// validation can compare direct HIP with the full prepared replay.
pub fn run_prepared_decode_layers_and_head(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
    emit_head: bool,
    fusion_enabled: bool,
) -> Result<(), String> {
    gpu.replay.set_forward_eligible(false);
    run_decode_embedding(cfg, weights, state, gpu)?;
    decode_step_layers_and_head(
        cfg,
        weights,
        state,
        gpu,
        position,
        emit_head,
        fusion_enabled,
        None,
    )
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
    emit_head: bool,
    fusion_enabled: bool,
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

    // #397 Ship 6 — the default-on lowered route runs the per-layer decode
    // through run_layer_program. Oracle capture keeps the byte-identical hand
    // path because it downloads each layer; HIPFIRE_FORWARD_LOWERED=0 remains
    // the explicit legacy escape hatch.
    if lfm2_forward_lowered_enabled() && capture.is_none() {
        return decode_step_layers_and_head_lowered(
            cfg,
            weights,
            state,
            gpu,
            position,
            emit_head,
            fusion_enabled,
        );
    }

    for (l, layer) in weights.layers.iter().enumerate() {
            // ── Mixer block (pre-norm) ──────────────────────────────────────
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
                    gpu.rmsnorm_batched(
                        &state.fa_q,
                        &a.q_norm,
                        &state.fa_q,
                        n_heads,
                        head_dim,
                        eps,
                    )
                    .map_err(|e| format!("lfm2moe L{l}: q_norm: {e:?}"))?;
                    gpu.rmsnorm_batched(
                        &state.fa_k,
                        &a.k_norm,
                        &state.fa_k,
                        n_kv,
                        head_dim,
                        eps,
                    )
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
                    hfq4_moe_down_expanded(gpu, m, state, hidden, moe_inter, k_top, l)?;
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

    // Final RMSNorm + lm_head (tied to embed_tokens, Q8). Non-final prefill
    // tokens (Phase 0 head-elision) skip this: their logits are never read.
    if emit_head {
        gpu.rmsnorm_f32(
            &state.h,
            &weights.embedding_norm,
            &state.final_norm_buf,
            eps,
        )
        .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
            .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    }
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

fn use_lfm2_350m_conv_wide_hfq4(
    is_gfx1201: bool,
    fusion_enabled: bool,
    dtype: DType,
    m: usize,
    k: usize,
) -> bool {
    is_gfx1201 && fusion_enabled && dtype == DType::MQ4G256 && m == 3 * 1024 && k == 1024
}

/// Conv mixer block (operator-norm folded in). Mirrors the hand-loop Conv arm.
fn conv_mixer_block(
    gpu: &mut Gpu,
    cfg: &Lfm2MoeConfig,
    op_norm: &rdna_compute::GpuTensor,
    c: &ConvWeights,
    state: &Lfm2MoeState,
    l: usize,
    fusion_enabled: bool,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    if fusion_enabled {
        let x_rot = fused_rmsnorm_rotate_for_mq(
            gpu,
            &c.in_proj,
            &state.h,
            op_norm,
            &state.tmp,
            &state.ffn_x_rot,
            cfg.rms_norm_eps,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused operator norm+rotate: {e:?}"))?;
        if use_lfm2_350m_conv_wide_hfq4(
            gpu.arch_caps.is_gfx1201(),
            fusion_enabled,
            c.in_proj.gpu_dtype,
            c.in_proj.m,
            c.in_proj.k,
        ) {
            let x_rot = x_rot.ok_or_else(|| {
                format!("lfm2moe L{l}: exact fusion expected MQ4 conv projection")
            })?;
            gpu.gemv_hfq4g256_wide(
                &c.in_proj.buf,
                x_rot,
                &state.conv_bcx,
                c.in_proj.m,
                c.in_proj.k,
            )
            .map_err(|e| format!("lfm2moe L{l}: conv in_proj wide: {e:?}"))?;
        } else {
            weight_gemv_prerotated(gpu, &c.in_proj, &state.tmp, x_rot, &state.conv_bcx)
                .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
        }
    } else {
        gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, cfg.rms_norm_eps)
            .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
            .map_err(|e| format!("lfm2moe L{l}: conv in_proj: {e}"))?;
    }
    if fusion_enabled {
        gpu.conv1d_gated_decode_mq_rotate_f32(
            &state.conv_bcx,
            &state.conv_states[c.conv_state_idx],
            &c.conv_weight,
            &state.ffn_x_rot,
            1,
            hidden,
            cfg.conv_kernel_size,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused conv decode+rotate: {e:?}"))?;
        weight_gemv_prerotated_residual(
            gpu,
            &c.out_proj,
            &state.conv_y,
            Some(&state.ffn_x_rot),
            &state.h,
        )
        .map_err(|e| format!("lfm2moe L{l}: conv out_proj: {e}"))
    } else {
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
    fusion_enabled: bool,
) -> Result<(), String> {
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    if fusion_enabled {
        let x_rot = fused_rmsnorm_rotate_for_mq(
            gpu,
            &a.wq,
            &state.h,
            op_norm,
            &state.tmp,
            &state.ffn_x_rot,
            eps,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused operator norm+rotate: {e:?}"))?;
        let x_rot = x_rot.ok_or_else(|| {
            format!("lfm2moe L{l}: exact fusion expected MQ4 attention projections")
        })?;
        gpu.fused_qkv_hfq4g256(
            &a.wq.buf,
            &a.wk.buf,
            &a.wv.buf,
            x_rot,
            &state.fa_q,
            &state.fa_k,
            &state.fa_v,
            a.wq.m,
            a.wk.m,
            a.wv.m,
            a.wq.k,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused qkv: {e:?}"))?;
    } else {
        gpu.rmsnorm_f32(&state.h, op_norm, &state.tmp, eps)
            .map_err(|e| format!("lfm2moe L{l}: operator rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
            .map_err(|e| format!("lfm2moe L{l}: q_proj: {e}"))?;
        weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
            .map_err(|e| format!("lfm2moe L{l}: k_proj: {e}"))?;
        weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
            .map_err(|e| format!("lfm2moe L{l}: v_proj: {e}"))?;
    }
    if fusion_enabled {
        gpu.lfm_qk_norm_rope_cached_f32(
            &state.fa_q,
            &state.fa_k,
            &a.q_norm,
            &a.k_norm,
            &state.rope_cos_cache,
            &state.rope_sin_cache,
            &state.pos_buf,
            n_heads,
            n_kv,
            head_dim,
            eps,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused q/k norm+cached rope: {e:?}"))?;
    } else {
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
    }
    let kv_idx = a.kv_idx;
    // KV write (Q8) + attention via the shared Qwen-style KvTierPlan flash
    // policy. flash_mode is frozen at state construction (HIPFIRE_ATTN_FLASH);
    // capture_mode forces flash while graph/replay recording so the variable-
    // LDS non-flash Q8 kernel is never retained. Partials are always threaded;
    // the non-flash arm ignores them when HIPFIRE_ATTN_FLASH=never.
    let capture_mode = gpu.graphs.capture_mode || gpu.replay.is_recording();
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    let tier_inputs = hipfire_dispatch::families::kv_tier::KvTierInputs {
        pos: seq_len - 1,
        flash_mode: state.flash_mode as usize,
        capture_mode,
        ..state.kv.tier_inputs()
    };
    let plan = hipfire_dispatch::families::kv_tier::KvTierPlan::derive(tier_inputs)
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
        flash_partials: Some(&state.flash_partials),
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
    fusion_enabled: bool,
) -> Result<(), String> {
    if fusion_enabled {
        let x_rot = fused_rmsnorm_rotate_for_mq(
            gpu,
            &d.w1,
            &state.h,
            ffn_norm,
            &state.ffn_tmp,
            &state.ffn_x_rot,
            cfg.rms_norm_eps,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused ffn norm+rotate: {e:?}"))?;
        let x_rot =
            x_rot.ok_or_else(|| format!("lfm2moe L{l}: exact fusion expected MQ4 dense FFN"))?;
        gpu.fused_gate_up_hfq4g256(
            &d.w1.buf,
            &d.w3.buf,
            x_rot,
            &state.dense_gate,
            &state.dense_up,
            d.w1.m,
            d.w3.m,
            d.w1.k,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused dense gate/up: {e:?}"))?;
    } else {
        gpu.rmsnorm_f32(&state.h, ffn_norm, &state.ffn_tmp, cfg.rms_norm_eps)
            .map_err(|e| format!("lfm2moe L{l}: ffn rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
            .map_err(|e| format!("lfm2moe L{l}: dense w1: {e}"))?;
        weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
            .map_err(|e| format!("lfm2moe L{l}: dense w3: {e}"))?;
    }
    Ok(())
}

/// Dense FFN down half (silu·mul + w2 residual). Mirrors the hand-loop Dense tail.
fn dense_down_block(
    gpu: &mut Gpu,
    d: &DenseFfn,
    state: &Lfm2MoeState,
    l: usize,
    fusion_enabled: bool,
) -> Result<(), String> {
    if fusion_enabled {
        weight_gemv_swiglu_residual(
            gpu,
            &d.w2,
            &state.dense_gate,
            &state.dense_up,
            &state.dense_act,
            &state.h,
        )
        .map_err(|e| format!("lfm2moe L{l}: fused dense w2: {e}"))
    } else {
        gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
            .map_err(|e| format!("lfm2moe L{l}: dense silu_mul: {e:?}"))?;
        weight_gemv_residual(gpu, &d.w2, &state.dense_act, &state.h)
            .map_err(|e| format!("lfm2moe L{l}: dense w2: {e}"))
    }
}

fn hfq4_moe_down_expanded(
    gpu: &mut Gpu,
    m: &MoeFfn,
    state: &Lfm2MoeState,
    hidden: usize,
    moe_inter: usize,
    k_top: usize,
    l: usize,
) -> Result<(), String> {
    if (hidden, moe_inter, k_top)
        == (
            crate::kernels::LFM2_A1B_HIDDEN,
            crate::kernels::LFM2_A1B_MOE_INTERMEDIATE,
            crate::kernels::LFM2_A1B_TOP_K,
        )
    {
        return crate::kernels::lfm2_a1b_moe_down(
            gpu,
            &m.expert_down_ptrs,
            &state.topk_indices,
            &state.rot_batch,
            &state.down_expanded,
            m.experts[0].down.gpu_dtype,
            hidden,
            moe_inter,
            k_top,
        )
        .map_err(|e| format!("lfm2moe L{l}: down: {e}"));
    }

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
    .map_err(|e| format!("lfm2moe L{l}: down: {e:?}"))
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
        hfq4_moe_down_expanded(gpu, m, state, hidden, moe_inter, k_top, l)?;
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

    let mixer = match v {
        Lfm2Variant::ConvDense | Lfm2Variant::ConvMoe => Conv,
        Lfm2Variant::AttnDense | Lfm2Variant::AttnMoe => Attend,
    };
    let mut program = Vec::with_capacity(3);
    program.push(lfm2_superop(mixer, 0));
    match v {
        Lfm2Variant::ConvDense | Lfm2Variant::AttnDense => {
            program.push(lfm2_superop(Proj, DENSE_GATE_UP));
            program.push(lfm2_superop(ResidualGemv, DENSE_DOWN));
        }
        Lfm2Variant::ConvMoe | Lfm2Variant::AttnMoe => {
            program.push(lfm2_superop(Moe, 0));
        }
    }
    program
}

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
struct Lfm2MoeBindings<'a> {
    cfg: &'a Lfm2MoeConfig,
    layer: &'a Lfm2MoeLayerWeights,
    state: &'a Lfm2MoeState,
    l: usize,
    seq_len: usize,
    fusion_enabled: bool,
}

impl<'a> ForwardBindings for Lfm2MoeBindings<'a> {
    fn run_conv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Conv(c) => conv_mixer_block(
                gpu,
                self.cfg,
                &self.layer.operator_norm,
                c,
                self.state,
                self.l,
                self.fusion_enabled,
            ),
            _ => Err("run_conv on non-Conv layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.mixer {
            Mixer::Attention(a) => attn_mixer_block(
                gpu,
                self.cfg,
                &self.layer.operator_norm,
                a,
                self.state,
                self.l,
                self.seq_len,
                self.fusion_enabled,
            ),
            _ => Err("run_attend on non-Attention layer".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_GATE_UP, Ffn::Dense(d)) => dense_gate_up_block(
                gpu,
                self.cfg,
                &self.layer.ffn_norm,
                d,
                self.state,
                self.l,
                self.fusion_enabled,
            ),
            _ => Err(format!("run_proj bad opcode {code} / non-Dense ffn")),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let code = op.weights.first().map(|w| w.0).unwrap_or(u32::MAX);
        match (code, &self.layer.ffn) {
            (lfm2_op::DENSE_DOWN, Ffn::Dense(d)) => {
                dense_down_block(gpu, d, self.state, self.l, self.fusion_enabled)
            }
            _ => Err(format!(
                "run_residual_gemv bad opcode {code} / non-Dense ffn"
            )),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match &self.layer.ffn {
            Ffn::Moe(m) => {
                moe_ffn_block(gpu, self.cfg, &self.layer.ffn_norm, m, self.state, self.l)
            }
            _ => Err("run_moe on non-Moe ffn".to_string()),
        }
        .map_err(DispatchError::Hip)
    }

    fn run_norm(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "lfm2 has no standalone Norm super-op".into(),
        ))
    }
    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("lfm2 has no Recurrent super-op".into()))
    }
    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "lfm2 has no Escape super-op ({kind:?})"
        )))
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
    emit_head: bool,
    fusion_enabled: bool,
) -> Result<(), String> {
    let eps = cfg.rms_norm_eps;
    let seq_len = position as usize + 1;
    let ctx = DispatchCtx::new(gpu);
    for (l, layer) in weights.layers.iter().enumerate() {
        let program = lfm2_lower_variant(lfm2_variant_of(layer));
        {
            let mut bind = Lfm2MoeBindings {
                cfg,
                layer,
                state,
                l,
                seq_len,
                fusion_enabled,
            };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| format!("lfm2moe L{l}: lowered run_layer_program: {e}"))?;
        }
    }
    state.n_tokens = seq_len;
    if emit_head {
        gpu.rmsnorm_f32(
            &state.h,
            &weights.embedding_norm,
            &state.final_norm_buf,
            eps,
        )
        .map_err(|e| format!("lfm2moe: final rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
            .map_err(|e| format!("lfm2moe: lm_head: {e}"))?;
    }
    Ok(())
}

/// Experimental hipGraph-amortized decode. Opt-in via `HIPFIRE_LFM2_GRAPH=1`;
/// default OFF. Do not enable for production until Q8 attention capture uses
/// max-sequence or tiled launch geometry.
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
///   * `position` — staged into the stable `state.pos_buf` before replay.
///     Kernel data arguments therefore see the new position, but
///     `attention_q8_0_kv` currently bakes `block_size` and shared-memory size
///     from the capture-time `seq_len`. Replaying at longer lengths can exceed
///     that geometry; this is why the whole graph path remains experimental
///     and default-off.
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
    gpu.replay.set_forward_eligible(false);

    // ── Warmup phase: direct dispatch, no capture ──────────────────────────
    // Run the legacy path once so inline JIT / lazy scratch alloc happen
    // before any stream capture (capturing a hipMalloc errors).
    if !state.graph_warmed_up {
        state.graph_warmed_up = true;
        decode_step_inner(cfg, weights, state, gpu, token_id, position, true, None)?;
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

    prepare_decode_inputs(cfg, weights, state, gpu, token_id, position)?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────────
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: begin_graph_capture: {e:?}"))?;
        decode_step_layers_and_head(cfg, weights, state, gpu, position, true, false, None)?;
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

    #[test]
    fn fusion_request_defaults_on_and_fails_closed_for_malformed_values() {
        use std::ffi::OsStr;

        assert!(resolve_lfm2_decode_fusion_request(None));
        assert!(resolve_lfm2_decode_fusion_request(Some(OsStr::new("1"))));
        assert!(!resolve_lfm2_decode_fusion_request(Some(OsStr::new("0"))));
        assert!(!resolve_lfm2_decode_fusion_request(Some(OsStr::new("true"))));
        assert!(!resolve_lfm2_decode_fusion_request(Some(OsStr::new(""))));

        #[cfg(unix)]
        {
            use std::ffi::OsString;
            use std::os::unix::ffi::OsStringExt;

            let non_unicode = OsString::from_vec(vec![0xff]);
            assert!(!resolve_lfm2_decode_fusion_request(Some(
                non_unicode.as_os_str(),
            )));
        }
    }

    // #397 Ship 6 — lfm2 lowered LayerProgram shapes must mirror the hand-loop
    // order (mixer block, then FFN). CPU-pure (no GPU).
    #[test]
    fn lfm2_variant_shapes() {
        let kinds = |v| {
            lfm2_lower_variant(v)
                .iter()
                .map(|o| o.kind)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kinds(Lfm2Variant::ConvDense),
            vec![Conv, Proj, ResidualGemv]
        );
        assert_eq!(
            kinds(Lfm2Variant::AttnDense),
            vec![Attend, Proj, ResidualGemv]
        );
        assert_eq!(kinds(Lfm2Variant::ConvMoe), vec![Conv, Moe]);
        assert_eq!(kinds(Lfm2Variant::AttnMoe), vec![Attend, Moe]);
        let p = lfm2_lower_variant(Lfm2Variant::ConvDense);
        assert_eq!(p[1].binding.weights[0].0, lfm2_op::DENSE_GATE_UP);
        assert_eq!(p[2].binding.weights[0].0, lfm2_op::DENSE_DOWN);
    }

    #[test]
    fn retained_route_requires_current_eligibility_and_prefers_aql() {
        assert_eq!(
            retained_decode_route(false, true, true),
            RetainedDecodeRoute::Hip
        );
        assert_eq!(
            retained_decode_route(true, true, true),
            RetainedDecodeRoute::Aql
        );
        assert_eq!(
            retained_decode_route(true, false, true),
            RetainedDecodeRoute::Pm4
        );
        assert_eq!(
            retained_decode_route(true, false, false),
            RetainedDecodeRoute::Hip
        );
    }

    #[test]
    fn gfx1201_350m_conv_uses_wide_hfq4_decode_gemv_only() {
        assert!(use_lfm2_350m_conv_wide_hfq4(
            true,
            true,
            DType::MQ4G256,
            3072,
            1024
        ));
        assert!(!use_lfm2_350m_conv_wide_hfq4(
            false,
            true,
            DType::MQ4G256,
            3072,
            1024
        ));
        assert!(!use_lfm2_350m_conv_wide_hfq4(
            true,
            false,
            DType::MQ4G256,
            3072,
            1024
        ));
        assert!(!use_lfm2_350m_conv_wide_hfq4(
            true,
            true,
            DType::Q8_0,
            3072,
            1024
        ));
        assert!(!use_lfm2_350m_conv_wide_hfq4(
            true,
            true,
            DType::MQ4G256,
            1024,
            1024
        ));
        assert!(!use_lfm2_350m_conv_wide_hfq4(
            true,
            true,
            DType::MQ4G256,
            3072,
            4608
        ));
    }

    #[test]
    fn successful_replay_advances_exactly_to_the_next_position() {
        let mut n_tokens = 7;
        complete_retained_replay(&mut n_tokens, 7);
        assert_eq!(n_tokens, 8);
        complete_retained_replay(&mut n_tokens, 7);
        assert_eq!(n_tokens, 8);
    }


    #[test]
    fn lfm_speculative_route_fails_closed_when_graph_is_requested() {
        use std::ffi::OsStr;

        assert!(validate_lfm_speculative_graph(None).is_ok());
        assert!(validate_lfm_speculative_graph(Some(OsStr::new("0"))).is_ok());
        let error = validate_lfm_speculative_graph(Some(OsStr::new("1"))).unwrap_err();
        assert!(error.contains("HIPFIRE_LFM2_GRAPH"));
        assert!(error.contains("speculative"));
    }
}
