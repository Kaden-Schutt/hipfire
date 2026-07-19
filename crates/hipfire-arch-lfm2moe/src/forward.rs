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
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama::{
    fused_rmsnorm_rotate_mq_batched_for, fused_silu_mul_rotate_mq_batched_for,
    rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemv, weight_gemv_residual, WeightTensor,
};
use rdna_compute::{DType, Gpu};

const MQ4_GROUP_BYTES: usize = 136;

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
        None => format!(
            "lfm2moe prefill chunk base={chunk_base} len={chunk_len} {stage}: {error:?}"
        ),
    }
}

fn require_q8_weight(
    weight: &WeightTensor,
    m: usize,
    k: usize,
    name: &str,
) -> Result<(), String> {
    if weight.gpu_dtype != DType::Q8_0 || weight.m != m || weight.k != k {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} Q8_0 [{m},{k}], got {:?} [{},{}]",
            weight.gpu_dtype, weight.m, weight.k
        ));
    }
    Ok(())
}

fn require_mq4_proj(
    weight: &WeightTensor,
    m: usize,
    k: usize,
    name: &str,
) -> Result<(), String> {
    if weight.gpu_dtype != DType::MQ4G256 || weight.m != m || weight.k != k {
        return Err(format!(
            "lfm2moe: 350m.mq4 admission requires {name} MQ4G256 [{m},{k}], got {:?} [{},{}]",
            weight.gpu_dtype, weight.m, weight.k
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

/// Exact 350M dense MQ4 fixture (md5 cb5284b8 provenance): hidden1024, heads16/8,
/// hd64, q_dim1024, kv_dim512, inter4608, vocab65536, theta1e6, 16 dense layers.
fn validate_350m_mq4_admission(
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
            "lfm2moe: batched prefill admits only the frozen 350m.mq4 dense fixture shape"
                .to_string(),
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
    require_q8_weight(
        &weights.lm_head,
        cfg.vocab_size,
        cfg.hidden_size,
        "lm_head",
    )?;
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
    require_f32_tensor(
        &weights.embedding_norm,
        cfg.hidden_size,
        "embedding_norm",
    )?;

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
                    require_mq4_proj(
                        weight,
                        m,
                        k,
                        &format!("L{layer_idx}.attention.{name}"),
                    )?;
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
                    format!(
                        "lfm2moe: L{layer_idx} kv_idx {} out of range",
                        attn.kv_idx
                    )
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
    if conv_slots.iter().any(|occupied| !occupied)
        || kv_slots.iter().any(|occupied| !occupied)
    {
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

fn validate_prefill_launch_dimensions(
    cfg: &Lfm2MoeConfig,
    chunk_len: usize,
) -> Result<(), String> {
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
        return Err(
            "lfm2moe: batched prefill requires HIPFIRE_LFM2_PREFILL_BATCH=1".to_string(),
        );
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
    validate_350m_mq4_admission(cfg, weights, state)?;
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
    let positions_i32: Vec<i32> = (0..n)
        .map(|i| i32::try_from(p + i).unwrap())
        .collect();
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
        .map_err(|e| prefill_chunk_error("operator rmsnorm+rotate", Some(layer_idx), p, n, e))?;

        match &layer.mixer {
            Mixer::Conv(conv) => {
                // Zero-then-residual so conv in_proj hits HFQ4 residual WMMA (gfx12).
                let conv_bcx_n = scratch
                    .conv_bcx_batch
                    .sub_offset(0, n * 3 * cfg.hidden_size);
                gpu.fill_f32(&conv_bcx_n, 0.0).map_err(|e| {
                    prefill_chunk_error("conv in_proj zero", Some(layer_idx), p, n, e)
                })?;
                gpu.gemm_hfq4g256_residual(
                    &conv.in_proj.buf,
                    &scratch.operator_x_rot_batch,
                    &scratch.conv_bcx_batch,
                    3 * cfg.hidden_size,
                    cfg.hidden_size,
                    n,
                )
                .map_err(|e| {
                    prefill_chunk_error("conv in_proj", Some(layer_idx), p, n, e)
                })?;
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
                .map_err(|e| {
                    prefill_chunk_error("conv out rotate", Some(layer_idx), p, n, e)
                })?;
                gpu.gemm_hfq4g256_residual(
                    &conv.out_proj.buf,
                    &scratch.conv_y_rot_batch,
                    &scratch.h_batch,
                    cfg.hidden_size,
                    cfg.hidden_size,
                    n,
                )
                .map_err(|e| {
                    prefill_chunk_error("conv out_proj", Some(layer_idx), p, n, e)
                })?;
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
                .map_err(|e| {
                    prefill_chunk_error("attention flash", Some(layer_idx), p, n, e)
                })?;
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
                gpu.gemm_hfq4g256_residual(
                    &attn.wo.buf,
                    &scratch.fa_attn_out_rot_batch,
                    &scratch.h_batch,
                    cfg.hidden_size,
                    cfg.q_dim(),
                    n,
                )
                .map_err(|e| {
                    prefill_chunk_error("attention out_proj", Some(layer_idx), p, n, e)
                })?;
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
        gpu.gemm_gate_up_hfq4g256(
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
        gpu.gemm_hfq4g256_residual(
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
                let hidden = gpu.download_f32(&scratch.h_batch).map_err(|e| {
                    prefill_chunk_error("capture hidden", Some(layer_idx), p, n, e)
                })?;
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
        gpu.memcpy_dtod_at_auto(
            &state.h.buf,
            0,
            &scratch.h_batch.buf,
            row_offset,
            row_bytes,
        )
        .map_err(|e| prefill_chunk_error("final row copy", None, p, n, e))?;
        gpu.rmsnorm_f32(
            &state.h,
            &weights.embedding_norm,
            &state.final_norm_buf,
            cfg.rms_norm_eps,
        )
        .map_err(|e| prefill_chunk_error("final rmsnorm", None, p, n, e))?;
        weight_gemv(
            gpu,
            &weights.lm_head,
            &state.final_norm_buf,
            &state.logits,
        )
        .map_err(|e| prefill_chunk_error("lm_head", None, p, n, e))?;
    }
    Ok(())
}

/// Decode one token; returns the full logits vector.
///
/// `HIPFIRE_LFM2_GRAPH=1` selects the experimental graph path. It is default
/// OFF and not safe for production: Q8 attention launch geometry is captured
/// at the initial sequence length and does not grow during replay.
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
    decode_step_inner(cfg, weights, state, gpu, token_id, position, true, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
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
    if graph_enabled() {
        // Explicit graph mode: do not bypass the user-selected path. Run the
        // full decode (graph) and discard the logits; head-elision is a
        // non-graph optimization.
        return decode_step(cfg, weights, state, gpu, token_id, position).map(|_| ());
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
    decode_step_inner(cfg, weights, state, gpu, token_id, position, true, Some(capture))
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
    let hidden = cfg.hidden_size;

    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;

    // Embedding lookup → residual stream h (Q8 table).
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm2moe: embed lookup: {e:?}"))?;

    decode_step_layers_and_head(cfg, weights, state, gpu, position, emit_head, capture)
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
        return decode_step_layers_and_head_lowered(cfg, weights, state, gpu, position, emit_head);
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
            (lfm2_op::DENSE_GATE_UP, Ffn::Dense(d)) => {
                dense_gate_up_block(gpu, self.cfg, &self.layer.ffn_norm, d, self.state, self.l)
            }
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
            (lfm2_op::DENSE_DOWN, Ffn::Dense(d)) => dense_down_block(gpu, d, self.state, self.l),
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
    let hidden = cfg.hidden_size;

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
        decode_step_layers_and_head(cfg, weights, state, gpu, position, true, None)?;
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
}
