// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline capture/replay fixtures, moved out of the daemon.
//!
//! These 28 functions and their 7 snapshot types drive PM4 capture, replay and
//! decode benches for the Redline path. They are a research/diagnostic surface
//! that happened to live in `main.rs`, and they accounted for 25 of the
//! daemon's remaining architecture-type references — the largest cluster after
//! batch staging.
//!
//! Moved verbatim. The kernels, dispatch and PM4 lowering they exercise are
//! untouched and out of scope (leanup map, §6); this only changes where the
//! fixture code lives.

use crate::common::*;
use hipfire_loader::{LoadedModel, ModelState};
use std::time::Instant;
use hipfire_engine::redline::{
    redline_append_buffer, redline_append_tensor, redline_append_tensor_region,
    redline_capture_json, redline_hash, RedlineRegionHash,
};
use hipfire_arch_qwen35::carrier::Qwen35Bundle;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_qwen35::qwen35;
#[derive(PartialEq)]
pub struct RedlineQwenSnapshot {
    pub logits: Vec<u8>,
    pub kv: Vec<u8>,
    pub recurrent: Vec<u8>,
}

impl RedlineQwenSnapshot {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "logits_bytes": self.logits.len(),
            "logits_hash": format!("{:016x}", redline_hash(&self.logits)),
            "kv_bytes": self.kv.len(),
            "kv_hash": format!("{:016x}", redline_hash(&self.kv)),
            "recurrent_bytes": self.recurrent.len(),
            "recurrent_hash": format!("{:016x}", redline_hash(&self.recurrent)),
        })
    }
}

#[derive(PartialEq)]
pub struct RedlineDeepseek4Snapshot {
    pub logits: Vec<u8>,
    pub kv: Vec<u8>,
    pub kv_regions: Vec<RedlineRegionHash>,
    pub recurrent: Vec<u8>,
}

impl RedlineDeepseek4Snapshot {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "logits_bytes": self.logits.len(),
            "logits_hash": format!("{:016x}", redline_hash(&self.logits)),
            "kv_bytes": self.kv.len(),
            "kv_hash": format!("{:016x}", redline_hash(&self.kv)),
            "kv_regions": self.kv_regions.iter().map(|region| serde_json::json!({
                "name": region.name,
                "bytes": region.bytes,
                "hash": format!("{:016x}", region.hash),
            })).collect::<Vec<_>>(),
            "recurrent_bytes": self.recurrent.len(),
            "recurrent_hash": format!("{:016x}", redline_hash(&self.recurrent)),
        })
    }
}

#[derive(PartialEq)]
pub struct RedlineDsparkVerifySnapshot {
    pub target: RedlineDeepseek4Snapshot,
    pub captures: Vec<u8>,
    pub streams: Vec<u8>,
    pub picks: Vec<u32>,
}

impl RedlineDsparkVerifySnapshot {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "target": self.target.json(),
            "captures_bytes": self.captures.len(),
            "captures_hash": format!("{:016x}", redline_hash(&self.captures)),
            "streams_bytes": self.streams.len(),
            "streams_hash": format!("{:016x}", redline_hash(&self.streams)),
            "picks": self.picks,
        })
    }
}

#[derive(PartialEq)]
pub enum RedlineSnapshot {
    Qwen(RedlineQwenSnapshot),
    Deepseek4(RedlineDeepseek4Snapshot),
    Lfm2Moe(RedlineLfm2MoeSnapshot),
}

impl RedlineSnapshot {
    pub fn logits(&self) -> &[u8] {
        match self {
            Self::Qwen(snapshot) => &snapshot.logits,
            Self::Deepseek4(snapshot) => &snapshot.logits,
            Self::Lfm2Moe(snapshot) => &snapshot.logits,
        }
    }

    pub fn kv(&self) -> &[u8] {
        match self {
            Self::Qwen(snapshot) => &snapshot.kv,
            Self::Deepseek4(snapshot) => &snapshot.kv,
            Self::Lfm2Moe(snapshot) => &snapshot.kv,
        }
    }

    pub fn recurrent(&self) -> &[u8] {
        match self {
            Self::Qwen(snapshot) => &snapshot.recurrent,
            Self::Deepseek4(snapshot) => &snapshot.recurrent,
            Self::Lfm2Moe(snapshot) => &snapshot.recurrent,
        }
    }

    pub fn json(&self) -> serde_json::Value {
        match self {
            Self::Qwen(snapshot) => snapshot.json(),
            Self::Deepseek4(snapshot) => snapshot.json(),
            Self::Lfm2Moe(snapshot) => snapshot.json(),
        }
    }
}

pub fn redline_qwen_snapshot(
    gpu: &rdna_compute::Gpu,
    bundle: &Qwen35Bundle,
) -> Result<RedlineQwenSnapshot, String> {
    let mut logits = Vec::new();
    redline_append_buffer(gpu, &mut logits, &bundle.scratch.logits.buf)?;
    let mut kv = Vec::new();
    for tensor in bundle
        .kv_cache
        .k_gpu
        .iter()
        .chain(bundle.kv_cache.v_gpu.iter())
        .chain(bundle.kv_cache.k_scales.iter())
        .chain(bundle.kv_cache.v_scales.iter())
    {
        redline_append_buffer(gpu, &mut kv, &tensor.buf)?;
    }
    let mut recurrent = Vec::new();
    for tensor in bundle
        .dn_state
        .s_matrices
        .iter()
        .chain(bundle.dn_state.s_scales.iter())
        .chain(bundle.dn_state.conv_states.iter())
        .chain(bundle.dn_state.s_ef_residual.iter())
    {
        redline_append_buffer(gpu, &mut recurrent, &tensor.buf)?;
    }
    Ok(RedlineQwenSnapshot {
        logits,
        kv,
        recurrent,
    })
}

pub fn redline_deepseek4_snapshot(
    gpu: &rdna_compute::Gpu,
    bundle: &deepseek4::Deepseek4Bundle,
) -> Result<RedlineDeepseek4Snapshot, String> {
    let mut logits = Vec::new();
    redline_append_tensor(gpu, &mut logits, &bundle.state.logits)?;

    let mut kv = Vec::new();
    let mut kv_regions = Vec::new();
    for (layer_idx, layer) in bundle.state._indexer.iter().enumerate() {
        redline_append_tensor_region(
            gpu,
            &mut kv,
            &mut kv_regions,
            format!("indexer.{layer_idx}.main_kv_cache"),
            &layer.main_kv_cache,
        )?;
        redline_append_tensor_region(
            gpu,
            &mut kv,
            &mut kv_regions,
            format!("indexer.{layer_idx}.indexer_kv_cache"),
            &layer.indexer_kv_cache,
        )?;
    }
    for (layer_idx, layer) in bundle.state._attention.iter().enumerate() {
        for (field, tensor) in [
            ("swa_k", &layer.swa_k),
            ("swa_v", &layer.swa_v),
            ("full_k_cache", &layer.full_k_cache),
            ("full_v_cache", &layer.full_v_cache),
        ] {
            redline_append_tensor_region(
                gpu,
                &mut kv,
                &mut kv_regions,
                format!("attention.{layer_idx}.{field}"),
                tensor,
            )?;
        }
    }

    let mut recurrent = Vec::new();
    for layer in &bundle.state._indexer {
        redline_append_tensor(gpu, &mut recurrent, &layer.main_kv_state)?;
        redline_append_tensor(gpu, &mut recurrent, &layer.main_score_state)?;
        redline_append_tensor(gpu, &mut recurrent, &layer.indexer_kv_state)?;
        redline_append_tensor(gpu, &mut recurrent, &layer.indexer_score_state)?;
    }
    redline_append_tensor(gpu, &mut recurrent, &bundle.state.residual_streams)?;
    redline_append_tensor(gpu, &mut recurrent, &bundle.state.residual_streams_next)?;
    redline_append_tensor(gpu, &mut recurrent, &bundle.state.attn_state_buf)?;

    Ok(RedlineDeepseek4Snapshot {
        logits,
        kv,
        kv_regions,
        recurrent,
    })
}
#[derive(PartialEq)]
pub struct RedlineLfm2MoeSnapshot {
    pub logits: Vec<u8>,
    pub kv: Vec<u8>,
    pub recurrent: Vec<u8>,
}

impl RedlineLfm2MoeSnapshot {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "logits_bytes": self.logits.len(),
            "logits_hash": format!("{:016x}", redline_hash(&self.logits)),
            "kv_bytes": self.kv.len(),
            "kv_hash": format!("{:016x}", redline_hash(&self.kv)),
            "recurrent_bytes": self.recurrent.len(),
            "recurrent_hash": format!("{:016x}", redline_hash(&self.recurrent)),
        })
    }
}

pub fn redline_lfm2moe_snapshot(
    gpu: &rdna_compute::Gpu,
    bundle: &lfm2moe::Lfm2MoeBundle,
) -> Result<RedlineLfm2MoeSnapshot, String> {
    let mut logits = Vec::new();
    redline_append_buffer(gpu, &mut logits, &bundle.state.logits.buf)?;
    let mut kv = Vec::new();
    for tensor in bundle
        .state
        .kv
        .k_gpu
        .iter()
        .chain(bundle.state.kv.v_gpu.iter())
        .chain(bundle.state.kv.k_scales.iter())
        .chain(bundle.state.kv.v_scales.iter())
    {
        redline_append_buffer(gpu, &mut kv, &tensor.buf)?;
    }
    let mut recurrent = Vec::new();
    for tensor in bundle.state.conv_states.iter() {
        redline_append_buffer(gpu, &mut recurrent, &tensor.buf)?;
    }
    Ok(RedlineLfm2MoeSnapshot {
        logits,
        kv,
        recurrent,
    })
}

pub fn redline_reset_lfm2moe(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut lfm2moe::Lfm2MoeBundle,
) -> Result<(), String> {
    bundle.state.reset(gpu)?;
    gpu.invalidate_graph_state();
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

pub fn redline_is_dense_lfm(loaded: &LoadedModel) -> bool {
    if loaded.pp != 1 || loaded.ep.is_some() {
        return false;
    }
    match loaded.state.as_ref() {
        Some(ModelState::Lfm2Moe(bundle)) => bundle.config.is_dense(),
        _ => false,
    }
}

pub fn redline_append_tensor_slice(
    gpu: &rdna_compute::Gpu,
    output: &mut Vec<u8>,
    tensor: &rdna_compute::GpuTensor,
    offset: usize,
    len: usize,
) -> Result<(), String> {
    if offset.saturating_add(len) > tensor.numel() {
        return Err(format!(
            "redline tensor slice {}+{} exceeds {}",
            offset,
            len,
            tensor.numel()
        ));
    }
    let view = tensor.sub_offset(offset, len);
    redline_append_buffer(gpu, output, &view.buf)
}

pub fn redline_dspark_verify_snapshot(
    gpu: &rdna_compute::Gpu,
    bundle: &deepseek4::Deepseek4Bundle,
    batch: usize,
    picks: Vec<u32>,
) -> Result<RedlineDsparkVerifySnapshot, String> {
    let target = redline_deepseek4_snapshot(gpu, bundle)?;
    let hidden = bundle.config.hidden_size;
    let n_targets = bundle.state.dspark_target_layers.len();
    let mut captures = Vec::new();
    if let Some(tensor) = bundle.state.dspark_caps.as_ref() {
        redline_append_tensor_slice(gpu, &mut captures, tensor, 0, batch * n_targets * hidden)?;
    }
    let pbs = bundle
        .state
        .dspark_verify_pbs
        .as_ref()
        .ok_or_else(|| "DSpark verify snapshot: PBS missing".to_string())?;
    let mut streams = Vec::new();
    redline_append_tensor_slice(
        gpu,
        &mut streams,
        &pbs.streams_batch,
        0,
        batch * bundle.config.hc_mult * hidden,
    )?;
    Ok(RedlineDsparkVerifySnapshot {
        target,
        captures,
        streams,
        picks,
    })
}

/// Hash one inactive row from the highest-risk batch-shaped verify buffers.
/// The row immediately after the active batch is a same-allocation red zone:
/// every fixed-node B-shaped kernel must leave it byte-identical.
pub fn redline_dspark_verify_guard(
    gpu: &rdna_compute::Gpu,
    bundle: &deepseek4::Deepseek4Bundle,
    batch: usize,
) -> Result<Vec<u8>, String> {
    let pbs = bundle
        .state
        .dspark_verify_pbs
        .as_ref()
        .ok_or_else(|| "DSpark verify guard: PBS missing".to_string())?;
    if batch >= pbs.max_batch {
        return Err(format!(
            "DSpark verify guard needs inactive row after B={batch}, max_batch={}",
            pbs.max_batch
        ));
    }
    let cfg = &bundle.config;
    let hidden = cfg.hidden_size;
    let mut guard = Vec::new();
    for (tensor, row) in [
        (&pbs.embed_batch, hidden),
        (&pbs.streams_batch, cfg.hc_mult * hidden),
        (&pbs.q_batch, cfg.num_attention_heads * cfg.head_dim),
        (&pbs.kv_batch, cfg.num_key_value_heads * cfg.head_dim),
        (&pbs.attn_out_batch, hidden),
        (&pbs.ffn_out_batch, hidden),
        (&pbs.moe_scores_batch, cfg.n_routed_experts),
        (&pbs.moe_topk_indices_batch, cfg.num_experts_per_tok),
        (&pbs.moe_topk_weights_batch, cfg.num_experts_per_tok),
        (&pbs.idx_q_batch, cfg.index_n_heads * cfg.index_head_dim),
        (&pbs.idx_topk_indices_batch, cfg.index_topk),
    ] {
        redline_append_tensor_slice(gpu, &mut guard, tensor, batch * row, row)?;
    }
    let n_targets = bundle.state.dspark_target_layers.len();
    if n_targets > 0 {
        if let Some(caps) = bundle.state.dspark_caps.as_ref() {
            let row = n_targets * hidden;
            redline_append_tensor_slice(gpu, &mut guard, caps, batch * row, row)?;
        }
    }
    Ok(guard)
}

pub fn redline_snapshot(
    gpu: &rdna_compute::Gpu,
    loaded: &LoadedModel,
) -> Result<RedlineSnapshot, String> {
    match loaded.state.as_ref() {
        Some(ModelState::Qwen35(bundle)) => {
            redline_qwen_snapshot(gpu, bundle).map(RedlineSnapshot::Qwen)
        }
        Some(ModelState::Deepseek4(bundle)) => {
            redline_deepseek4_snapshot(gpu, bundle).map(RedlineSnapshot::Deepseek4)
        }
        Some(ModelState::Lfm2Moe(bundle)) => {
            if !bundle.config.is_dense() {
                return Err("retained snapshot requires dense LFM".to_string());
            }
            redline_lfm2moe_snapshot(gpu, bundle).map(RedlineSnapshot::Lfm2Moe)
        }
        _ => Err("retained snapshot requires Qwen3.5, DeepSeek4 or dense LFM".to_string()),
    }
}

pub fn redline_qwen_debug_hashes(
    gpu: &rdna_compute::Gpu,
    bundle: &Qwen35Bundle,
) -> Result<std::collections::BTreeMap<String, String>, String> {
    let tensors: &[(&str, &hip_bridge::DeviceBuffer)] = &[
        ("x", &bundle.scratch.x.buf),
        ("tmp", &bundle.scratch.tmp.buf),
        ("pos", &bundle.scratch.pos_buf),
        ("dn_qkv", &bundle.scratch.dn_qkv.buf),
        ("dn_z", &bundle.scratch.dn_z.buf),
        ("dn_alpha", &bundle.scratch.dn_alpha.buf),
        ("dn_beta", &bundle.scratch.dn_beta.buf),
        ("dn_conv_out", &bundle.scratch.dn_conv_out.buf),
        ("dn_q", &bundle.scratch.dn_q.buf),
        ("dn_k", &bundle.scratch.dn_k.buf),
        ("dn_v", &bundle.scratch.dn_v.buf),
        ("dn_q_raw", &bundle.scratch.dn_q_raw.buf),
        ("dn_k_raw", &bundle.scratch.dn_k_raw.buf),
        ("dn_attn_out", &bundle.scratch.dn_attn_out.buf),
        ("dn_normed", &bundle.scratch.dn_normed.buf),
        ("fa_q_full", &bundle.scratch.fa_q_full.buf),
        ("fa_q", &bundle.scratch.fa_q.buf),
        ("fa_gate", &bundle.scratch.fa_gate.buf),
        ("fa_k", &bundle.scratch.fa_k.buf),
        ("fa_v", &bundle.scratch.fa_v.buf),
        ("fa_attn_out", &bundle.scratch.fa_attn_out.buf),
        ("o", &bundle.scratch.o.buf),
        ("gate_ffn", &bundle.scratch.gate_ffn.buf),
        ("up", &bundle.scratch.up.buf),
        ("ffn_hidden", &bundle.scratch.ffn_hidden.buf),
        ("ffn_out", &bundle.scratch.ffn_out.buf),
        ("logits", &bundle.scratch.logits.buf),
        ("x_rot", &bundle.scratch.x_rot.buf),
        ("flash_partials", &bundle.scratch.flash_partials.buf),
    ];
    let mut hashes = std::collections::BTreeMap::new();
    for (name, buffer) in tensors {
        let mut bytes = Vec::new();
        redline_append_buffer(gpu, &mut bytes, buffer)?;
        hashes.insert((*name).to_owned(), format!("{:016x}", redline_hash(&bytes)));
    }
    let state = redline_qwen_snapshot(gpu, bundle)?;
    hashes.insert(
        "all_kv".to_owned(),
        format!("{:016x}", redline_hash(&state.kv)),
    );
    hashes.insert(
        "all_recurrent".to_owned(),
        format!("{:016x}", redline_hash(&state.recurrent)),
    );
    Ok(hashes)
}

pub fn redline_reset_qwen(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut Qwen35Bundle,
) -> Result<(), String> {
    bundle
        .kv_cache
        .clear_gpu(gpu)
        .map_err(|error| error.to_string())?;
    bundle.kv_cache.compact_offset = 0;
    for tensor in bundle
        .dn_state
        .s_matrices
        .iter()
        .chain(bundle.dn_state.s_scales.iter())
        .chain(bundle.dn_state.conv_states.iter())
        .chain(bundle.dn_state.s_ef_residual.iter())
    {
        gpu.hip
            .memset(&tensor.buf, 0, tensor.buf.size())
            .map_err(|error| error.to_string())?;
    }
    let scratch = &bundle.scratch;
    let buffers: &[&hip_bridge::DeviceBuffer] = &[
        &scratch.x.buf,
        &scratch.tmp.buf,
        &scratch.pos_buf,
        &scratch.dn_qkv.buf,
        &scratch.dn_z.buf,
        &scratch.dn_alpha.buf,
        &scratch.dn_beta.buf,
        &scratch.dn_conv_out.buf,
        &scratch.dn_q.buf,
        &scratch.dn_k.buf,
        &scratch.dn_v.buf,
        &scratch.dn_q_raw.buf,
        &scratch.dn_k_raw.buf,
        &scratch.dn_attn_out.buf,
        &scratch.dn_normed.buf,
        &scratch.fa_q_full.buf,
        &scratch.fa_q.buf,
        &scratch.fa_gate.buf,
        &scratch.fa_k.buf,
        &scratch.fa_v.buf,
        &scratch.fa_attn_out.buf,
        &scratch.o.buf,
        &scratch.gate_ffn.buf,
        &scratch.up.buf,
        &scratch.ffn_hidden.buf,
        &scratch.ffn_out.buf,
        &scratch.logits.buf,
        &scratch.sample_buf.buf,
        &scratch.repeat_buf.buf,
        &scratch.x_rot.buf,
        &scratch.flash_partials.buf,
    ];
    for buffer in buffers {
        gpu.hip
            .memset(buffer, 0, buffer.size())
            .map_err(|error| error.to_string())?;
    }
    for buffer in [
        gpu.scratch.mq_x_rot.as_ref().map(|tensor| &tensor.buf),
        gpu.scratch.mq_x_rot_fp8.as_ref(),
        gpu.scratch.mq_x_q8.as_ref(),
        gpu.scratch.mq_x_scales.as_ref(),
        gpu.scratch.fp16_x_scratch.as_ref(),
        gpu.scratch.fp8_x_scratch.as_ref(),
        gpu.scratch.q8_1_mmq_x_scratch.as_ref(),
        gpu.scratch.ksplit_det_partials.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        gpu.hip
            .memset(buffer, 0, buffer.size())
            .map_err(|error| error.to_string())?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

pub fn redline_prime_qwen(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut Qwen35Bundle,
    context: usize,
) -> Result<(), String> {
    let synthetic: Vec<u32> = (0..context as u32).map(|i| 10 + (i % 1000)).collect();
    qwen35::forward_prefill_batch(
        gpu,
        &bundle.weights,
        &bundle.config,
        &synthetic,
        0,
        &mut bundle.kv_cache,
        &mut bundle.dn_state,
        &bundle.scratch,
        None,
        None,
        None,
        None,
    )
    .map_err(|error| error.to_string())?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

pub fn redline_reset_deepseek4(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut deepseek4::Deepseek4Bundle,
) -> Result<(), String> {
    bundle.state.reset();
    bundle.state.zero_decode_caches(gpu);
    gpu.invalidate_graph_state();
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

pub fn redline_prime_deepseek4(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut deepseek4::Deepseek4Bundle,
    pbs: &mut deepseek4::forward::PrefillBatchScratch,
    context: usize,
) -> Result<(), String> {
    let synthetic = (0..context as u32)
        .map(|index| 10 + (index % 1000))
        .collect::<Vec<_>>();
    deepseek4::forward::forward_prefill_batch_chunked(
        &bundle.config,
        &bundle.weights,
        &mut bundle.state,
        gpu,
        &synthetic,
        0,
        pbs,
    )?;
    bundle.state.n_tokens = context as u64;
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

pub fn redline_run_deepseek4_decode(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut deepseek4::Deepseek4Bundle,
    context: usize,
    iterations: usize,
) -> Result<(), String> {
    for index in 0..iterations {
        let token = 101 + (index as u32 % 1000);
        deepseek4::forward::decode_step_with_graph(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            token,
            (context + index) as u32,
        )?;
    }
    Ok(())
}

pub fn redline_prime_retained_fixture(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    context: usize,
) -> Result<(), String> {
    match loaded.state.as_mut() {
        Some(ModelState::Qwen35(bundle)) => {
            redline_reset_qwen(gpu, bundle)?;
            redline_prime_qwen(gpu, bundle, context)
        }
        Some(ModelState::Deepseek4(bundle)) => {
            let pbs = loaded
                .deepseek4_pbs
                .as_mut()
                .ok_or_else(|| "DeepSeek4 prefill scratch missing".to_string())?;
            redline_reset_deepseek4(gpu, bundle)?;
            redline_prime_deepseek4(gpu, bundle, pbs, context)
        }
        Some(ModelState::Lfm2Moe(bundle)) => {
            if !bundle.config.is_dense() {
                return Err("retained fixture requires dense LFM".to_string());
            }
            redline_reset_lfm2moe(gpu, bundle)?;
            for pos in 0..context {
                let token = 10 + (pos as u32 % 1000);
                lfm2moe::forward::prepare_retained_decode_inputs(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    token,
                    pos as u32,
                )?;
                lfm2moe::forward::run_retained_decode_body(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    pos as u32,
                )?;
            }
            loaded.seq_pos = context;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            Ok(())
        }
        _ => Err("retained fixture requires Qwen3.5, DeepSeek4 or dense LFM".to_string()),
    }
}

pub fn redline_prepare_retained_fixture(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    token_id: u32,
    context: usize,
) -> Result<(), String> {
    match loaded.state.as_mut() {
        Some(ModelState::Qwen35(bundle)) => qwen35::prepare_scratch_inputs(
            gpu,
            &bundle.weights,
            &bundle.config,
            token_id,
            context,
            &bundle.scratch,
        )
        .map_err(|error| error.to_string()),
        Some(ModelState::Deepseek4(bundle)) => {
            bundle.state.n_tokens = context as u64;
            deepseek4::forward::prepare_retained_decode_inputs(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                token_id,
                context as u32,
            )
        }
        Some(ModelState::Lfm2Moe(bundle)) => {
            if !bundle.config.is_dense() {
                return Err("retained fixture requires dense LFM".to_string());
            }
            lfm2moe::forward::prepare_retained_decode_inputs(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                token_id,
                context as u32,
            )
        }
        _ => Err("retained fixture requires Qwen3.5, DeepSeek4 or dense LFM".to_string()),
    }
}

pub fn redline_run_direct_fixture(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    context: usize,
    iterations: usize,
) -> Result<(), String> {
    match loaded.state.as_mut() {
        Some(ModelState::Qwen35(bundle)) => {
            for index in 0..iterations {
                qwen35::forward_scratch(
                    gpu,
                    &bundle.weights,
                    &bundle.config,
                    101 + index as u32,
                    context + index,
                    &mut bundle.kv_cache,
                    &mut bundle.dn_state,
                    &bundle.scratch,
                )
                .map_err(|error| error.to_string())?;
            }
            Ok(())
        }
        Some(ModelState::Deepseek4(bundle)) => {
            for index in 0..iterations {
                bundle.state.n_tokens = (context + index) as u64;
                deepseek4::forward::decode_step(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    101 + index as u32,
                    (context + index) as u32,
                )?;
            }
            Ok(())
        }
        Some(ModelState::Lfm2Moe(bundle)) => {
            if !bundle.config.is_dense() {
                return Err("retained fixture requires dense LFM".to_string());
            }
            for index in 0..iterations {
                let token = 101 + index as u32;
                let pos = (context + index) as u32;
                lfm2moe::forward::prepare_retained_decode_inputs(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    token,
                    pos,
                )?;
                lfm2moe::forward::run_retained_decode_body(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    pos,
                )?;
            }
            loaded.seq_pos = context + iterations;
            Ok(())
        }
        _ => Err("retained fixture requires Qwen3.5, DeepSeek4 or dense LFM".to_string()),
    }
}

pub fn redline_bench_decode_deepseek4(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    msg: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    if loaded.pp > 1
        || loaded.ep.is_some()
        || !matches!(loaded.state.as_ref(), Some(ModelState::Deepseek4(_)))
    {
        return Err("bench_decode requires a loaded single-GPU DeepSeek4 model".to_string());
    }
    let context = msg
        .get("context_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(128) as usize;
    let iterations = msg
        .get("iterations")
        .and_then(|value| value.as_u64())
        .unwrap_or(1) as usize;
    let capture = msg
        .get("redline_capture")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let product_route = msg
        .get("redline_product_route")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let capture_detail = msg
        .get("redline_detail")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if capture && product_route {
        return Err("redline_capture and redline_product_route are mutually exclusive".to_string());
    }
    if context == 0 || iterations == 0 {
        return Err("bench_decode context_tokens and iterations must be non-zero".to_string());
    }
    if context.saturating_add(iterations).saturating_add(32) > loaded.physical_cap {
        return Err(format!(
            "bench_decode context+iterations exceeds loaded physical_cap={}",
            loaded.physical_cap
        ));
    }

    loaded.seq_pos = 0;
    loaded.conversation_tokens.clear();
    let pbs = loaded
        .deepseek4_pbs
        .as_mut()
        .ok_or_else(|| "DeepSeek4 prefill scratch missing".to_string())?;
    let ModelState::Deepseek4(bundle) = loaded.state.as_mut().unwrap() else {
        unreachable!()
    };
    redline_reset_deepseek4(gpu, bundle)?;
    redline_prime_deepseek4(gpu, bundle, pbs, context)
        .map_err(|error| format!("bench_decode prefill prime failed: {error}"))?;
    loaded.seq_pos = context;

    if capture || (product_route && gpu.replay.prepared_route_identity().is_some()) {
        // Manual capture and prepared product routes are already warm paths.
        // The first product warmup must still materialize lazy allocations and
        // record the route; later requests replay from their first timed token.
        bundle.state.ar_forward_warmed_up = true;
    }
    if capture {
        gpu.replay
            .begin_capture()
            .map_err(|reason| format!("redline decode capture refused: {reason}"))?;
    }

    if product_route {
        gpu.replay.begin_replay_observation_window();
    }
    let replay_before = gpu.replay.replay_observation();
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    redline_run_deepseek4_decode(gpu, bundle, context, iterations)
        .map_err(|error| format!("bench_decode forward failed: {error}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())?;
    let elapsed = started.elapsed().as_secs_f64();
    let replay_after = gpu.replay.replay_observation();
    let capture_summary = if capture {
        Some(
            gpu.replay
                .finish_capture()
                .map_err(|reason| format!("redline decode capture failed: {reason}"))?,
        )
    } else {
        None
    };

    loaded.seq_pos = 0;
    loaded.conversation_tokens.clear();
    redline_reset_deepseek4(gpu, bundle)?;

    let mut response = serde_json::json!({
        "type": "decode_result",
        "context_tokens": context,
        "iterations": iterations,
        "ms": elapsed * 1000.0,
        "us_per_token": elapsed * 1_000_000.0 / iterations as f64,
        "tok_s": iterations as f64 / elapsed.max(f64::MIN_POSITIVE),
    });
    if let Some(summary) = capture_summary {
        response["redline_capture"] = redline_capture_json(gpu, summary, capture_detail);
    }
    if product_route {
        let prepared = gpu.replay.prepared_route_identity().map(|identity| {
            serde_json::json!({
                "dispatches": identity.dispatch_count,
                "packets": identity.packet_count,
                "queue_id": identity.queue_id,
                "command_dwords": identity.command_dwords,
                "queues": identity.queue_count,
                "phases": identity.phase_count,
            })
        });
        let sequence = gpu.replay.capture_summary();
        let replay_delta = replay_after.count.saturating_sub(replay_before.count);
        response["redline_route"] = serde_json::json!({
            "requested_backend": format!("{:?}", gpu.replay.request()).to_ascii_lowercase(),
            "transport": gpu.replay.transport_name(),
            "state": format!("{:?}", gpu.replay.state()).to_ascii_lowercase(),
            "fallback_reason": gpu.replay.fallback_reason(),
            "execution_mode": "plain_ar",
            "prepared": prepared,
            "sequence": {
                "launches": sequence.launch_count,
                "unique_kernels": sequence.unique_kernel_count,
                "hash": format!("{:016x}", sequence.sequence_hash),
            },
            "observed": {
                "count_before": replay_before.count,
                "count_after": replay_after.count,
                "count_delta": replay_delta,
                "first_position": replay_after.first_position,
                "last_position": replay_after.last_position,
            },
            "retained_replay_observed": replay_delta > 0,
        });
    }
    Ok(response)
}
pub fn redline_bench_decode_lfm2moe(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    msg: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    if !redline_is_dense_lfm(loaded) {
        return Err("bench_decode requires a loaded single-GPU dense LFM model".to_string());
    }
    let context = msg
        .get("context_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(128) as usize;
    let iterations = msg
        .get("iterations")
        .and_then(|value| value.as_u64())
        .unwrap_or(1) as usize;
    let capture = msg
        .get("redline_capture")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let product_route = msg
        .get("redline_product_route")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let capture_detail = msg
        .get("redline_detail")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if capture && product_route {
        return Err("redline_capture and redline_product_route are mutually exclusive".to_string());
    }
    if context == 0 || iterations == 0 {
        return Err("bench_decode context_tokens and iterations must be non-zero".to_string());
    }
    if capture && iterations != 1 {
        return Err("redline_capture requires iterations==1".to_string());
    }
    if context.saturating_add(iterations).saturating_add(32) > loaded.physical_cap {
        return Err(format!(
            "bench_decode context+iterations exceeds loaded physical_cap={}",
            loaded.physical_cap
        ));
    }
    // Capture/forward cleanup: guarantee reset on every path.
    let mut capture_started = false;
    let inner = (|| -> Result<serde_json::Value, String> {
        loaded.seq_pos = 0;
        loaded.conversation_tokens.clear();
        redline_prime_retained_fixture(gpu, loaded, context)
            .map_err(|error| format!("bench_decode prefill prime failed: {error}"))?;
        loaded.seq_pos = context;
        // Manual capture and prepared product routes are already warm paths.
        // The first product warmup must still materialize lazy allocations and
        // record the route; later requests replay from their first timed token.
        if capture || (product_route && gpu.replay.prepared_route_identity().is_some()) {
            if let Some(ModelState::Lfm2Moe(bundle)) = loaded.state.as_mut() {
                bundle.state.retained_warmed_up = true;
            }
        }
        if capture {
            redline_prepare_retained_fixture(gpu, loaded, 101, context)
                .map_err(|error| format!("bench_decode stage failed: {error}"))?;
            gpu.replay
                .begin_capture()
                .map_err(|reason| format!("redline decode capture refused: {reason}"))?;
            capture_started = true;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let started = Instant::now();
            {
                let bundle = match loaded.state.as_mut() {
                    Some(ModelState::Lfm2Moe(bundle)) => bundle,
                    _ => unreachable!(),
                };
                lfm2moe::forward::run_retained_decode_body(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    context as u32,
                )
                .map_err(|error| format!("bench_decode forward failed: {error}"))?;
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let elapsed = started.elapsed().as_secs_f64();
            let summary = gpu
                .replay
                .finish_capture()
                .map_err(|reason| format!("redline decode capture failed: {reason}"))?;
            capture_started = false;
            loaded.seq_pos = 0;
            loaded.conversation_tokens.clear();
            let bundle = match loaded.state.as_mut() {
                Some(ModelState::Lfm2Moe(bundle)) => bundle,
                _ => unreachable!(),
            };
            redline_reset_lfm2moe(gpu, bundle)?;
            let mut response = serde_json::json!({
                "type": "decode_result",
                "context_tokens": context,
                "iterations": iterations,
                "ms": elapsed * 1000.0,
                "us_per_token": elapsed * 1_000_000.0 / iterations as f64,
                "tok_s": iterations as f64 / elapsed.max(f64::MIN_POSITIVE),
            });
            response["redline_capture"] = redline_capture_json(gpu, summary, capture_detail);
            Ok(response)
        } else if product_route {
            // Production timed arm: call production decode_step so retained
            // replay selection and host n_tokens commit happen in-runtime.
            // Route proof is observation-only — never the requested backend.
            gpu.replay.begin_replay_observation_window();
            let replay_before = gpu.replay.replay_observation();
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let started = Instant::now();
            for i in 0..iterations {
                let token = 101 + (i as u32 % 1000);
                let pos = (context + i) as u32;
                {
                    let bundle = match loaded.state.as_mut() {
                        Some(ModelState::Lfm2Moe(bundle)) => bundle,
                        _ => unreachable!(),
                    };
                    lfm2moe::forward::decode_step(
                        &bundle.config,
                        &bundle.weights,
                        &mut bundle.state,
                        gpu,
                        token,
                        pos,
                    )
                    .map_err(|error| format!("bench_decode forward failed: {error}"))?;
                }
                loaded.seq_pos = context + i + 1;
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let elapsed = started.elapsed().as_secs_f64();
            let replay_after = gpu.replay.replay_observation();
            loaded.seq_pos = 0;
            loaded.conversation_tokens.clear();
            let bundle = match loaded.state.as_mut() {
                Some(ModelState::Lfm2Moe(bundle)) => bundle,
                _ => unreachable!(),
            };
            redline_reset_lfm2moe(gpu, bundle)?;
            let mut response = serde_json::json!({
                "type": "decode_result",
                "context_tokens": context,
                "iterations": iterations,
                "ms": elapsed * 1000.0,
                "us_per_token": elapsed * 1_000_000.0 / iterations as f64,
                "tok_s": iterations as f64 / elapsed.max(f64::MIN_POSITIVE),
            });
            let prepared = gpu.replay.prepared_route_identity().map(|identity| {
                serde_json::json!({
                    "dispatches": identity.dispatch_count,
                    "packets": identity.packet_count,
                    "queue_id": identity.queue_id,
                    "command_dwords": identity.command_dwords,
                    "queues": identity.queue_count,
                    "phases": identity.phase_count,
                })
            });
            let sequence = gpu.replay.capture_summary();
            let replay_delta = replay_after.count.saturating_sub(replay_before.count);
            response["redline_route"] = serde_json::json!({
                "requested_backend": format!("{:?}", gpu.replay.request()).to_ascii_lowercase(),
                "transport": gpu.replay.transport_name(),
                "state": format!("{:?}", gpu.replay.state()).to_ascii_lowercase(),
                "fallback_reason": gpu.replay.fallback_reason(),
                "execution_mode": "plain_ar",
                "prepared": prepared,
                "sequence": {
                    "launches": sequence.launch_count,
                    "unique_kernels": sequence.unique_kernel_count,
                    "hash": format!("{:016x}", sequence.sequence_hash),
                },
                "observed": {
                    "count_before": replay_before.count,
                    "count_after": replay_after.count,
                    "count_delta": replay_delta,
                    "first_position": replay_after.first_position,
                    "last_position": replay_after.last_position,
                },
                "retained_replay_observed": replay_delta > 0,
            });
            Ok(response)
        } else {
            // Manual oracle timing path: stage outside, body only (no product decode_step).
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let started = Instant::now();
            for i in 0..iterations {
                let token = 101 + (i as u32 % 1000);
                let pos = context + i;
                redline_prepare_retained_fixture(gpu, loaded, token, pos)?;
                {
                    let bundle = match loaded.state.as_mut() {
                        Some(ModelState::Lfm2Moe(bundle)) => bundle,
                        _ => unreachable!(),
                    };
                    lfm2moe::forward::run_retained_decode_body(
                        &bundle.config,
                        &bundle.weights,
                        &mut bundle.state,
                        gpu,
                        pos as u32,
                    )
                    .map_err(|error| format!("bench_decode forward failed: {error}"))?;
                }
                loaded.seq_pos = context + i + 1;
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let elapsed = started.elapsed().as_secs_f64();
            loaded.seq_pos = 0;
            loaded.conversation_tokens.clear();
            let bundle = match loaded.state.as_mut() {
                Some(ModelState::Lfm2Moe(bundle)) => bundle,
                _ => unreachable!(),
            };
            redline_reset_lfm2moe(gpu, bundle)?;
            Ok(serde_json::json!({
                "type": "decode_result",
                "context_tokens": context,
                "iterations": iterations,
                "ms": elapsed * 1000.0,
                "us_per_token": elapsed * 1_000_000.0 / iterations as f64,
                "tok_s": iterations as f64 / elapsed.max(f64::MIN_POSITIVE),
            }))
        }
    })();
    match inner {
        Ok(value) => Ok(value),
        Err(error) => {
            if capture_started {
                gpu.replay.poison("bench_decode aborted during capture");
            }
            // Ensure host state is cleaned even on failure.
            loaded.seq_pos = 0;
            loaded.conversation_tokens.clear();
            if let Some(ModelState::Lfm2Moe(bundle)) = loaded.state.as_mut() {
                let _ = redline_reset_lfm2moe(gpu, bundle);
            } else {
                let _ = gpu.hip.device_synchronize();
            }
            Err(error)
        }
    }
}

pub fn redline_shadow_deepseek4(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    pm4: bool,
    context: usize,
    iterations: usize,
) -> Result<serde_json::Value, String> {
    let is_ds4 = loaded.pp == 1
        && loaded.ep.is_none()
        && matches!(loaded.state.as_ref(), Some(ModelState::Deepseek4(_)));
    let is_lfm = redline_is_dense_lfm(loaded);
    if !is_ds4 && !is_lfm {
        return Err(
            "redline shadow requires a loaded single-GPU DeepSeek4 or dense LFM model".to_string(),
        );
    }
    if is_ds4 {
        // DS4 byte-identical path — preserve existing behavior exactly.
        let prepared = if pm4 {
            let launch_count = gpu.replay.recorded_launches().len();
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launch_count)
                .map(|(dispatches, dwords, queue)| (dispatches, 1, queue, Some(dwords)))
        } else {
            gpu.replay
                .prepare_linear_aql(gpu.device_id as usize)
                .map(|(dispatches, packets, queue)| (dispatches, packets, queue, None))
        }
        .map_err(|reason| format!("redline AQL prepare failed: {reason}"))?;
        let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        // Inner to allow cleanup on error without altering success bytes.
        let inner = (|| -> Result<serde_json::Value, String> {
            redline_prime_retained_fixture(gpu, loaded, context)?;
            let started = Instant::now();
            let mut gpu_us = 0.0;
            for index in 0..iterations {
                redline_prepare_retained_fixture(gpu, loaded, 101 + index as u32, context + index)?;
                if pm4 {
                    let timing = unsafe { gpu.replay.replay_pm4(context + index) }?;
                    gpu_us += timing.span_microseconds();
                } else {
                    let timing = unsafe { gpu.replay.replay_linear_aql(context + index) }?;
                    gpu_us += timing.span_microseconds();
                }
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let aql_host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            let aql_snapshot = redline_snapshot(gpu, loaded)?;
            redline_prime_retained_fixture(gpu, loaded, context)?;
            for index in 0..iterations {
                redline_prepare_retained_fixture(gpu, loaded, 101 + index as u32, context + index)?;
                gpu.replay_recorded_hip_prefix(prepared.0)
                    .map_err(|error| error.to_string())?;
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let blob_snapshot = redline_snapshot(gpu, loaded)?;
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
            redline_prime_retained_fixture(gpu, loaded, context)?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let started = Instant::now();
            redline_run_direct_fixture(gpu, loaded, context, iterations)?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let hip_host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            let hip_snapshot = redline_snapshot(gpu, loaded)?;
            let logits_equal = aql_snapshot.logits() == hip_snapshot.logits();
            let kv_equal = aql_snapshot.kv() == hip_snapshot.kv();
            let recurrent_equal = aql_snapshot.recurrent() == hip_snapshot.recurrent();
            let blob_bit_exact = aql_snapshot.logits() == blob_snapshot.logits()
                && aql_snapshot.kv() == blob_snapshot.kv()
                && aql_snapshot.recurrent() == blob_snapshot.recurrent();
            Ok(serde_json::json!({
                "type": "redline_shadow_result",
                "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
                "context_tokens": context,
                "iterations": iterations,
                "dispatches": prepared.0,
                "packets": prepared.1,
                "queue_id": prepared.2,
                "command_dwords": prepared.3,
                "bit_exact": logits_equal && kv_equal && recurrent_equal,
                "blob_bit_exact": blob_bit_exact,
                "logits_equal": logits_equal,
                "kv_equal": kv_equal,
                "recurrent_equal": recurrent_equal,
                "aql_host_us": aql_host_us,
                "aql_gpu_us": gpu_us,
                "hip_host_us": hip_host_us,
                "aql": aql_snapshot.json(),
                "hip": hip_snapshot.json(),
                "blob": blob_snapshot.json(),
            }))
        })();
        match inner {
            Ok(value) => Ok(value),
            Err(error) => {
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                if let Some(ModelState::Deepseek4(bundle)) = loaded.state.as_mut() {
                    let _ = redline_reset_deepseek4(gpu, bundle);
                    let _ = gpu.hip.device_synchronize();
                }
                Err(error)
            }
        }
    } else {
        // Dense LFM retained shadow: each oracle arm starts from identical prime;
        // PM4/blob stage inputs before each replay and commit host n_tokens after success.
        let prepared = if pm4 {
            let launch_count = gpu.replay.recorded_launches().len();
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launch_count)
                .map(|(dispatches, dwords, queue)| (dispatches, 1, queue, Some(dwords)))
        } else {
            gpu.replay
                .prepare_linear_aql(gpu.device_id as usize)
                .map(|(dispatches, packets, queue)| (dispatches, packets, queue, None))
        }
        .map_err(|reason| format!("redline AQL prepare failed: {reason}"))?;
        let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        let inner = (|| -> Result<serde_json::Value, String> {
            redline_prime_retained_fixture(gpu, loaded, context)?;
            let started = Instant::now();
            let mut gpu_us = 0.0;
            for index in 0..iterations {
                redline_prepare_retained_fixture(gpu, loaded, 101 + index as u32, context + index)?;
                // Input staging runs on the HIP stream, while retained AQL/PM4
                // executes on a ROCr queue. Complete the producer handoff
                // before the replay queue reads h and pos_buf.
                gpu.hip
                    .device_synchronize()
                    .map_err(|error| error.to_string())?;
                if pm4 {
                    let timing = unsafe { gpu.replay.replay_pm4(context + index) }?;
                    if let Some(ModelState::Lfm2Moe(bundle)) = loaded.state.as_mut() {
                        bundle.state.n_tokens = context + index + 1;
                        loaded.seq_pos = context + index + 1;
                    }
                    gpu_us += timing.span_microseconds();
                } else {
                    let timing = unsafe { gpu.replay.replay_linear_aql(context + index) }?;
                    if let Some(ModelState::Lfm2Moe(bundle)) = loaded.state.as_mut() {
                        bundle.state.n_tokens = context + index + 1;
                        loaded.seq_pos = context + index + 1;
                    }
                    gpu_us += timing.span_microseconds();
                }
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let aql_host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            let aql_snapshot = redline_snapshot(gpu, loaded)?;
            redline_prime_retained_fixture(gpu, loaded, context)?;
            for index in 0..iterations {
                redline_prepare_retained_fixture(gpu, loaded, 101 + index as u32, context + index)?;
                gpu.replay_recorded_hip_prefix(prepared.0)
                    .map_err(|error| error.to_string())?;
                if let Some(ModelState::Lfm2Moe(bundle)) = loaded.state.as_mut() {
                    bundle.state.n_tokens = context + index + 1;
                    loaded.seq_pos = context + index + 1;
                }
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let blob_snapshot = redline_snapshot(gpu, loaded)?;
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
            redline_prime_retained_fixture(gpu, loaded, context)?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let started = Instant::now();
            redline_run_direct_fixture(gpu, loaded, context, iterations)?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let hip_host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            let hip_snapshot = redline_snapshot(gpu, loaded)?;
            let logits_equal = aql_snapshot.logits() == hip_snapshot.logits();
            let kv_equal = aql_snapshot.kv() == hip_snapshot.kv();
            let recurrent_equal = aql_snapshot.recurrent() == hip_snapshot.recurrent();
            let blob_bit_exact = aql_snapshot.logits() == blob_snapshot.logits()
                && aql_snapshot.kv() == blob_snapshot.kv()
                && aql_snapshot.recurrent() == blob_snapshot.recurrent();
            Ok(serde_json::json!({
                "type": "redline_shadow_result",
                "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
                "context_tokens": context,
                "iterations": iterations,
                "dispatches": prepared.0,
                "packets": prepared.1,
                "queue_id": prepared.2,
                "command_dwords": prepared.3,
                "bit_exact": logits_equal && kv_equal && recurrent_equal,
                "blob_bit_exact": blob_bit_exact,
                "logits_equal": logits_equal,
                "kv_equal": kv_equal,
                "recurrent_equal": recurrent_equal,
                "aql_host_us": aql_host_us,
                "aql_gpu_us": gpu_us,
                "hip_host_us": hip_host_us,
                "aql": aql_snapshot.json(),
                "hip": hip_snapshot.json(),
                "blob": blob_snapshot.json(),
            }))
        })();
        match inner {
            Ok(value) => {
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                if let Some(ModelState::Lfm2Moe(bundle)) = loaded.state.as_mut() {
                    let _ = redline_reset_lfm2moe(gpu, bundle);
                    loaded.seq_pos = 0;
                    loaded.conversation_tokens.clear();
                    let _ = gpu.hip.device_synchronize();
                }
                Ok(value)
            }
            Err(error) => {
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                if let Some(ModelState::Lfm2Moe(bundle)) = loaded.state.as_mut() {
                    let _ = redline_reset_lfm2moe(gpu, bundle);
                    loaded.seq_pos = 0;
                    loaded.conversation_tokens.clear();
                    let _ = gpu.hip.device_synchronize();
                }
                Err(error)
            }
        }
    }
}

pub struct RedlineDsparkArm {
    pub snapshot: RedlineDsparkVerifySnapshot,
    pub guard_before: Vec<u8>,
    pub guard_after: Vec<u8>,
    pub host_us: f64,
}

impl RedlineDsparkArm {
    pub fn guard_unchanged(&self) -> bool {
        self.guard_before == self.guard_after
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "host_us": self.host_us,
            "guard_unchanged": self.guard_unchanged(),
            "guard_bytes": self.guard_after.len(),
            "guard_before_hash": format!("{:016x}", redline_hash(&self.guard_before)),
            "guard_after_hash": format!("{:016x}", redline_hash(&self.guard_after)),
            "snapshot": self.snapshot.json(),
        })
    }
}

pub fn redline_prime_dspark_shadow_arm(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    context: usize,
) -> Result<(), String> {
    let pbs = loaded
        .deepseek4_pbs
        .as_mut()
        .ok_or_else(|| "DSpark shadow: prefill scratch missing".to_string())?;
    let bundle = match loaded.state.as_mut() {
        Some(ModelState::Deepseek4(bundle)) => bundle,
        _ => return Err("DSpark shadow requires DeepSeek4".to_string()),
    };
    redline_reset_deepseek4(gpu, bundle)?;
    redline_prime_deepseek4(gpu, bundle, pbs, context)?;
    bundle.state.n_tokens = context as u64;
    Ok(())
}

pub fn redline_dspark_shadow_block(step: usize, batch: usize) -> Vec<u32> {
    (0..batch)
        .map(|slot| 101 + ((step * batch + slot) % 1000) as u32)
        .collect()
}

pub fn redline_run_dspark_direct_arm(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    context: usize,
    batch: usize,
    iterations: usize,
    capture_safe: bool,
) -> Result<RedlineDsparkArm, String> {
    redline_prime_dspark_shadow_arm(gpu, loaded, context)?;
    let bundle = match loaded.state.as_mut() {
        Some(ModelState::Deepseek4(bundle)) => bundle,
        _ => unreachable!(),
    };
    let guard_before = redline_dspark_verify_guard(gpu, bundle, batch)?;
    let started = Instant::now();
    let mut picks = Vec::with_capacity(batch * iterations);
    for step in 0..iterations {
        let block = redline_dspark_shadow_block(step, batch);
        let position = context + step * batch;
        picks.extend(bundle.redline_dspark_verify_direct(gpu, &block, position, capture_safe)?);
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())?;
    let host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    let guard_after = redline_dspark_verify_guard(gpu, bundle, batch)?;
    let snapshot = redline_dspark_verify_snapshot(gpu, bundle, batch, picks)?;
    Ok(RedlineDsparkArm {
        snapshot,
        guard_before,
        guard_after,
        host_us,
    })
}

pub fn redline_run_dspark_capture_arm(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    controller: &mut rdna_compute::replay::ReplayController,
    context: usize,
    batch: usize,
    iterations: usize,
) -> Result<
    (
        RedlineDsparkArm,
        deepseek4::spec_impl::DsparkVerifyCaptureInfo,
    ),
    String,
> {
    redline_prime_dspark_shadow_arm(gpu, loaded, context)?;
    let bundle = match loaded.state.as_mut() {
        Some(ModelState::Deepseek4(bundle)) => bundle,
        _ => unreachable!(),
    };
    let guard_before = redline_dspark_verify_guard(gpu, bundle, batch)?;
    let started = Instant::now();
    let first_block = redline_dspark_shadow_block(0, batch);
    let (first_picks, capture) =
        bundle.redline_dspark_verify_capture_pm4(gpu, controller, &first_block, context)?;
    let mut picks = Vec::with_capacity(batch * iterations);
    picks.extend(first_picks);
    for step in 1..iterations {
        let block = redline_dspark_shadow_block(step, batch);
        picks.extend(bundle.redline_dspark_verify_direct(
            gpu,
            &block,
            context + step * batch,
            true,
        )?);
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())?;
    let host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    let guard_after = redline_dspark_verify_guard(gpu, bundle, batch)?;
    let snapshot = redline_dspark_verify_snapshot(gpu, bundle, batch, picks)?;
    Ok((
        RedlineDsparkArm {
            snapshot,
            guard_before,
            guard_after,
            host_us,
        },
        capture,
    ))
}

#[derive(Clone, Copy)]
pub enum RedlineDsparkReplayArm {
    CapturedHip,
    Pm4,
}

pub fn redline_run_dspark_replay_arm(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    controller: &mut rdna_compute::replay::ReplayController,
    context: usize,
    batch: usize,
    iterations: usize,
    route: RedlineDsparkReplayArm,
) -> Result<RedlineDsparkArm, String> {
    redline_prime_dspark_shadow_arm(gpu, loaded, context)?;
    let bundle = match loaded.state.as_mut() {
        Some(ModelState::Deepseek4(bundle)) => bundle,
        _ => unreachable!(),
    };
    let guard_before = redline_dspark_verify_guard(gpu, bundle, batch)?;
    let started = Instant::now();
    let mut picks = Vec::with_capacity(batch * iterations);
    for step in 0..iterations {
        let block = redline_dspark_shadow_block(step, batch);
        let position = context + step * batch;
        let window = match route {
            RedlineDsparkReplayArm::CapturedHip => {
                bundle.redline_dspark_verify_captured_hip(gpu, controller, &block, position)?
            }
            RedlineDsparkReplayArm::Pm4 => {
                bundle.redline_dspark_verify_pm4(gpu, controller, &block, position)?
            }
        };
        picks.extend(window);
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())?;
    let host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    let guard_after = redline_dspark_verify_guard(gpu, bundle, batch)?;
    let snapshot = redline_dspark_verify_snapshot(gpu, bundle, batch, picks)?;
    Ok(RedlineDsparkArm {
        snapshot,
        guard_before,
        guard_after,
        host_us,
    })
}

/// DSpark-specific retained-verify parity oracle.
///
/// Four arms start from an identical synthetic prefill state:
/// shipping ordinary HIP, capture-safe HIP, exact captured HIP blobs, and one
/// conservative single-queue PM4 IB. Dynamic tokens/positions/counts change at
/// every window.  Promotion requires equality of outputs, logits, KV,
/// compressor/recurrent state, hidden captures, active streams, and inactive
/// same-allocation guard rows.
pub fn redline_shadow_dspark_verify_pm4(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    context: usize,
    batch: usize,
    iterations: usize,
) -> Result<serde_json::Value, String> {
    if loaded.pp > 1
        || loaded.ep.is_some()
        || !matches!(loaded.state.as_ref(), Some(ModelState::Deepseek4(_)))
    {
        return Err("DSpark shadow requires a loaded single-GPU DeepSeek4 model".to_string());
    }
    if batch == 0 || iterations == 0 {
        return Err("DSpark shadow batch and iterations must be non-zero".to_string());
    }
    if context
        .saturating_add(batch.saturating_mul(iterations))
        .saturating_add(32)
        > loaded.physical_cap
    {
        return Err(format!(
            "DSpark shadow context+windows exceeds physical_cap={}",
            loaded.physical_cap
        ));
    }
    {
        let bundle = match loaded.state.as_mut() {
            Some(ModelState::Deepseek4(bundle)) => bundle,
            _ => unreachable!(),
        };
        if bundle.weights.dspark.is_none() {
            return Err("DSpark shadow requires a loaded DSpark sidecar".to_string());
        }
        bundle.redline_ensure_dspark_verify_pbs(gpu, batch + 1)?;
    }

    // Materialize lazy verify allocations and code objects before any arm takes
    // a guard snapshot or starts recording.
    redline_prime_dspark_shadow_arm(gpu, loaded, context)?;
    {
        let bundle = match loaded.state.as_mut() {
            Some(ModelState::Deepseek4(bundle)) => bundle,
            _ => unreachable!(),
        };
        let warm_block = redline_dspark_shadow_block(0, batch);
        let _ = bundle.redline_dspark_verify_direct(gpu, &warm_block, context, false)?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())?;

    let direct = redline_run_dspark_direct_arm(gpu, loaded, context, batch, iterations, false)?;
    let mut controller = rdna_compute::replay::ReplayController::new_manual_pm4();
    let (capture_safe, capture_info) =
        redline_run_dspark_capture_arm(gpu, loaded, &mut controller, context, batch, iterations)?;
    let captured_hip = redline_run_dspark_replay_arm(
        gpu,
        loaded,
        &mut controller,
        context,
        batch,
        iterations,
        RedlineDsparkReplayArm::CapturedHip,
    )?;
    controller.begin_replay_observation_window();
    let pm4 = redline_run_dspark_replay_arm(
        gpu,
        loaded,
        &mut controller,
        context,
        batch,
        iterations,
        RedlineDsparkReplayArm::Pm4,
    )?;

    let direct_capture_exact = direct.snapshot == capture_safe.snapshot;
    let blob_bit_exact = capture_safe.snapshot == captured_hip.snapshot;
    let pm4_bit_exact = capture_safe.snapshot == pm4.snapshot;
    let guard_exact = direct.guard_unchanged()
        && capture_safe.guard_unchanged()
        && captured_hip.guard_unchanged()
        && pm4.guard_unchanged();
    let observation = controller.replay_observation();
    let identity = controller
        .prepared_route_identity()
        .ok_or_else(|| "DSpark shadow PM4 identity missing after prepare".to_string())?;
    let response = serde_json::json!({
        "type": "redline_dspark_shadow_result",
        "backend": "pm4_ib",
        "execution_mode": "dspark_verify",
        "context_tokens": context,
        "verify_batch": batch,
        "iterations": iterations,
        "bit_exact": direct_capture_exact && blob_bit_exact && pm4_bit_exact && guard_exact,
        "direct_capture_exact": direct_capture_exact,
        "blob_bit_exact": blob_bit_exact,
        "pm4_bit_exact": pm4_bit_exact,
        "guard_exact": guard_exact,
        "pm4_components": {
            "picks_equal": capture_safe.snapshot.picks == pm4.snapshot.picks,
            "logits_equal": capture_safe.snapshot.target.logits == pm4.snapshot.target.logits,
            "kv_equal": capture_safe.snapshot.target.kv == pm4.snapshot.target.kv,
            "recurrent_equal": capture_safe.snapshot.target.recurrent == pm4.snapshot.target.recurrent,
            "captures_equal": capture_safe.snapshot.captures == pm4.snapshot.captures,
            "streams_equal": capture_safe.snapshot.streams == pm4.snapshot.streams,
        },
        "capture": {
            "launches": capture_info.capture.launch_count,
            "unique_kernels": capture_info.capture.unique_kernel_count,
            "sequence_hash": format!("{:016x}", capture_info.capture.sequence_hash),
            "aql_contracts": capture_info.aql_contracts,
        },
        "prepared": {
            "dispatches": identity.dispatch_count,
            "packets": identity.packet_count,
            "queue_id": identity.queue_id,
            "command_dwords": identity.command_dwords,
            "queue_count": identity.queue_count,
            "phase_count": identity.phase_count,
        },
        "observed": {
            "replays": observation.count,
            "first_position": observation.first_position,
            "last_position": observation.last_position,
            "failed": observation.failed,
        },
        "direct": direct.json(),
        "capture_safe": capture_safe.json(),
        "captured_hip": captured_hip.json(),
        "pm4": pm4.json(),
    });
    if let Some(ModelState::Deepseek4(bundle)) = loaded.state.as_mut() {
        redline_reset_deepseek4(gpu, bundle)?;
    }
    Ok(response)
}

pub fn redline_pm4_prefix_profile_deepseek4(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    context: usize,
    start: usize,
    step: usize,
    repeats: usize,
    steady_state: bool,
) -> Result<serde_json::Value, String> {
    if loaded.pp > 1
        || loaded.ep.is_some()
        || !matches!(loaded.state.as_ref(), Some(ModelState::Deepseek4(_)))
    {
        return Err("prefix profile requires a loaded single-GPU DeepSeek4 model".to_string());
    }
    let launch_count = gpu.replay.recorded_launches().len();
    if launch_count == 0 || step == 0 || repeats == 0 || start == 0 || start > launch_count {
        return Err(
            "prefix profile requires captured launches and valid start/step/repeats".into(),
        );
    }
    let mut prefixes = (start..launch_count).step_by(step).collect::<Vec<_>>();
    if prefixes.last().copied() != Some(launch_count) {
        prefixes.push(launch_count);
    }
    let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
    if steady_state {
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
        redline_prime_retained_fixture(gpu, loaded, context)?;
    }
    let mut rows = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let launch = gpu.replay.recorded_launches()[prefix - 1].clone();
        let (_, dwords, _) = gpu
            .replay
            .prepare_pm4_prefix(gpu.device_id as usize, prefix)?;
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            if !steady_state {
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                redline_prime_retained_fixture(gpu, loaded, context)?;
            }
            redline_prepare_retained_fixture(gpu, loaded, 101, context)?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let timing = unsafe { gpu.replay.replay_pm4(context) }?;
            samples.push(timing.span_microseconds());
        }
        let mut ordered = samples.clone();
        ordered.sort_by(f64::total_cmp);
        rows.push(serde_json::json!({
            "prefix": prefix,
            "last_kernel": launch.kernel,
            "last_grid": launch.grid,
            "last_block": launch.block,
            "command_dwords": dwords,
            "samples_gpu_us": samples,
            "median_gpu_us": ordered[ordered.len() / 2],
        }));
    }
    Ok(serde_json::json!({
        "type": "redline_pm4_prefix_profile",
        "context_tokens": context,
        "launches": launch_count,
        "start": start,
        "step": step,
        "repeats": repeats,
        "steady_state": steady_state,
        "rows": rows,
    }))
}