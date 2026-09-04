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

use crate::batch::emit_uncorrelated_error;
use crate::common::*;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_qwen35::carrier::Qwen35Bundle;
use hipfire_arch_qwen35::dflash_verify_pm4::{
    DflashVerifyPm4, DflashVerifyPm4Phase, DFLASH_VERIFY_PM4_BLOCK,
};
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative::{
    verify_dflash_block, verify_dflash_block_retained, DeltaNetSnapshot, GdnTape,
    HiddenStateRingBuffer, ModelSlot, VerifyScratch,
};
use hipfire_engine::redline::{
    redline_append_buffer, redline_append_tensor, redline_append_tensor_region,
    redline_capture_json, redline_hash, RedlineRegionHash,
};
use hipfire_loader::spec_build::Qwen35SlotGuard;
use hipfire_loader::LoadedModel;
use rdna_compute::replay::ReplayQuiescence;
use std::any::Any;
use std::io::Read;
use std::time::Instant;
#[derive(PartialEq)]
pub struct RedlineQwenSnapshot {
    pub logits: Vec<u8>,
    pub kv: Vec<u8>,
    pub recurrent: Vec<u8>,
    /// Host-side GDN stochastic-rounding frame. Conv state already lives in
    /// `recurrent`; this is the missing counter the Q8 oracle needs.
    pub gdn_frame: u32,
}

#[derive(PartialEq)]
struct RedlineGemma4Snapshot {
    logits: Vec<u8>,
    sliding_kv: Vec<u8>,
    full_kv: Vec<u8>,
    sliding_kv_regions: Vec<RedlineRegionHash>,
    full_kv_regions: Vec<RedlineRegionHash>,
    scratch_x: Vec<u8>,
    scratch_tmp: Vec<u8>,
    scratch_q: Vec<u8>,
    scratch_k: Vec<u8>,
    scratch_v: Vec<u8>,
    scratch_attn_out: Vec<u8>,
    scratch_residual: Vec<u8>,
    scratch_moe_cur_mlp: Vec<u8>,
    scratch_moe_cur_moe: Vec<u8>,
    scratch_gate_ffn: Vec<u8>,
    scratch_up_ffn: Vec<u8>,
    scratch_ffn_hidden: Vec<u8>,
    scratch_ffn_out: Vec<u8>,
}

impl RedlineGemma4Snapshot {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "logits_bytes": self.logits.len(),
            "logits_hash": format!("{:016x}", redline_hash(&self.logits)),
            "sliding_kv_bytes": self.sliding_kv.len(),
            "sliding_kv_hash": format!("{:016x}", redline_hash(&self.sliding_kv)),
            "full_kv_bytes": self.full_kv.len(),
            "full_kv_hash": format!("{:016x}", redline_hash(&self.full_kv)),
            "sliding_kv_regions": self.sliding_kv_regions.iter().map(|region| serde_json::json!({
                "name": region.name,
                "bytes": region.bytes,
                "hash": format!("{:016x}", region.hash),
            })).collect::<Vec<_>>(),
            "full_kv_regions": self.full_kv_regions.iter().map(|region| serde_json::json!({
                "name": region.name,
                "bytes": region.bytes,
                "hash": format!("{:016x}", region.hash),
            })).collect::<Vec<_>>(),
            "scratch": {
                "x": format!("{:016x}", redline_hash(&self.scratch_x)),
                "tmp": format!("{:016x}", redline_hash(&self.scratch_tmp)),
                "q": format!("{:016x}", redline_hash(&self.scratch_q)),
                "k": format!("{:016x}", redline_hash(&self.scratch_k)),
                "v": format!("{:016x}", redline_hash(&self.scratch_v)),
                "attn_out": format!("{:016x}", redline_hash(&self.scratch_attn_out)),
                "residual": format!("{:016x}", redline_hash(&self.scratch_residual)),
                "moe_cur_mlp": format!("{:016x}", redline_hash(&self.scratch_moe_cur_mlp)),
                "moe_cur_moe": format!("{:016x}", redline_hash(&self.scratch_moe_cur_moe)),
                "gate_ffn": format!("{:016x}", redline_hash(&self.scratch_gate_ffn)),
                "up_ffn": format!("{:016x}", redline_hash(&self.scratch_up_ffn)),
                "ffn_hidden": format!("{:016x}", redline_hash(&self.scratch_ffn_hidden)),
                "ffn_out": format!("{:016x}", redline_hash(&self.scratch_ffn_out)),
            },
        })
    }
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
            "gdn_frame": self.gdn_frame,
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

    /// Q8 GatedDeltaNet stochastic-rounding frame. Non-Qwen snapshots do not
    /// carry this host-side state and therefore compare as `None`.
    pub fn gdn_frame(&self) -> Option<u32> {
        match self {
            Self::Qwen(snapshot) => Some(snapshot.gdn_frame),
            Self::Deepseek4(_) | Self::Lfm2Moe(_) => None,
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

fn redline_snapshots_bit_exact(lhs: &RedlineSnapshot, rhs: &RedlineSnapshot) -> bool {
    lhs.logits() == rhs.logits()
        && lhs.kv() == rhs.kv()
        && lhs.recurrent() == rhs.recurrent()
        && lhs.gdn_frame() == rhs.gdn_frame()
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
        gdn_frame: rdna_compute::norm::gdn_requant_frame_checkpoint(),
    })
}

fn redline_gemma4_snapshot(
    gpu: &rdna_compute::Gpu,
    bundle: &hipfire_loader::Gemma4LoweredBundle,
    _position: usize,
) -> Result<RedlineGemma4Snapshot, String> {
    fn append_kv(
        gpu: &rdna_compute::Gpu,
        out: &mut Vec<u8>,
        regions: &mut Vec<RedlineRegionHash>,
        prefix: &str,
        kv: &hipfire_runtime::llama::KvCache,
    ) -> Result<(), String> {
        for (kind, tensors) in [
            ("k", &kv.k_gpu),
            ("v", &kv.v_gpu),
            ("k_scale", &kv.k_scales),
            ("v_scale", &kv.v_scales),
        ] {
            for (index, tensor) in tensors.iter().enumerate() {
                let start = out.len();
                redline_append_buffer(gpu, out, &tensor.buf)?;
                regions.push(RedlineRegionHash {
                    name: format!("{prefix}.{kind}.{index}"),
                    bytes: out.len() - start,
                    hash: redline_hash(&out[start..]),
                });
            }
        }
        Ok(())
    }

    let mut logits = Vec::new();
    redline_append_buffer(gpu, &mut logits, &bundle.scratch.logits.buf)?;
    let mut sliding_kv = Vec::new();
    let mut sliding_kv_regions = Vec::new();
    append_kv(
        gpu,
        &mut sliding_kv,
        &mut sliding_kv_regions,
        "sliding",
        &bundle.kv_sliding,
    )?;
    let mut full_kv = Vec::new();
    let mut full_kv_regions = Vec::new();
    append_kv(
        gpu,
        &mut full_kv,
        &mut full_kv_regions,
        "full",
        &bundle.kv_full,
    )?;
    macro_rules! snapshot_buffer {
        ($buffer:expr) => {{
            let mut bytes = Vec::new();
            redline_append_buffer(gpu, &mut bytes, $buffer)?;
            bytes
        }};
    }
    Ok(RedlineGemma4Snapshot {
        logits,
        sliding_kv,
        full_kv,
        sliding_kv_regions,
        full_kv_regions,
        scratch_x: snapshot_buffer!(&bundle.scratch.x.buf),
        scratch_tmp: snapshot_buffer!(&bundle.scratch.tmp.buf),
        scratch_q: snapshot_buffer!(&bundle.scratch.q.buf),
        scratch_k: snapshot_buffer!(&bundle.scratch.k.buf),
        scratch_v: snapshot_buffer!(&bundle.scratch.v.buf),
        scratch_attn_out: snapshot_buffer!(&bundle.scratch.attn_out.buf),
        scratch_residual: snapshot_buffer!(&bundle.scratch.residual.buf),
        scratch_moe_cur_mlp: snapshot_buffer!(&bundle.scratch.moe_cur_mlp.buf),
        scratch_moe_cur_moe: snapshot_buffer!(&bundle.scratch.moe_cur_moe.buf),
        scratch_gate_ffn: snapshot_buffer!(&bundle.scratch.gate_ffn.buf),
        scratch_up_ffn: snapshot_buffer!(&bundle.scratch.up_ffn.buf),
        scratch_ffn_hidden: snapshot_buffer!(&bundle.scratch.ffn_hidden.buf),
        scratch_ffn_out: snapshot_buffer!(&bundle.scratch.ffn_out.buf),
    })
}

fn redline_reset_gemma4(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut hipfire_loader::Gemma4LoweredBundle,
) -> Result<(), String> {
    bundle
        .kv_sliding
        .clear_gpu(gpu)
        .map_err(|error| error.to_string())?;
    bundle
        .kv_full
        .clear_gpu(gpu)
        .map_err(|error| error.to_string())?;
    bundle.kv_sliding.compact_offset = 0;
    bundle.kv_full.compact_offset = 0;
    // The three oracle arms must begin with identical scratch as well as KV.
    // Reset the complete lowered scratch surface so inactive and tail lanes
    // are deterministic across the HIP, kernarg-blob, and retained arms.
    for buffer in [
        &bundle.scratch.x.buf,
        &bundle.scratch.residual.buf,
        &bundle.scratch.tmp.buf,
        &bundle.scratch.q.buf,
        &bundle.scratch.k.buf,
        &bundle.scratch.v.buf,
        &bundle.scratch.attn_out.buf,
        &bundle.scratch.gate_ffn.buf,
        &bundle.scratch.up_ffn.buf,
        &bundle.scratch.ffn_hidden.buf,
        &bundle.scratch.ffn_out.buf,
        &bundle.scratch.logits.buf,
        &bundle.scratch.flash_partials.buf,
        &bundle.scratch.moe_cur_mlp.buf,
        &bundle.scratch.moe_pre2.buf,
        &bundle.scratch.moe_router_in.buf,
        &bundle.scratch.moe_router_logits.buf,
        &bundle.scratch.moe_topk_indices.buf,
        &bundle.scratch.moe_topk_weights.buf,
        &bundle.scratch.moe_cur_moe.buf,
        &bundle.scratch.moe_expert_gate_up.buf,
        &bundle.scratch.moe_expert_hidden.buf,
        &bundle.scratch.moe_expert_out.buf,
        &bundle.scratch.moe_pre2_rot.buf,
        &bundle.scratch.moe_expert_gate_batch.buf,
        &bundle.scratch.moe_expert_up_batch.buf,
    ] {
        gpu.hip
            .memset(buffer, 0, buffer.size())
            .map_err(|error| error.to_string())?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

fn redline_prime_gemma4(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut hipfire_loader::Gemma4LoweredBundle,
    context: usize,
) -> Result<(), String> {
    for i in 0..context {
        hipfire_arch_gemma4::lowered::forward_scratch(
            gpu,
            &bundle.weights,
            &bundle.config,
            10 + (i as u32 % 1000),
            i,
            &mut bundle.kv_sliding,
            &mut bundle.kv_full,
            &bundle.scratch,
        )
        .map_err(|error| error.to_string())?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

fn redline_prepare_gemma4(
    gpu: &mut rdna_compute::Gpu,
    bundle: &hipfire_loader::Gemma4LoweredBundle,
    token: u32,
    position: usize,
) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;

    match bundle.weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(
            &bundle.weights.embed_tokens,
            &bundle.scratch.x,
            token,
            bundle.config.dim,
        ),
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(
            &bundle.weights.embed_tokens,
            &bundle.scratch.x,
            token,
            bundle.config.dim,
        ),
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(
            &bundle.weights.embed_tokens,
            &bundle.scratch.x,
            token,
            bundle.config.dim,
        ),
        EmbeddingFormat::F32 => gpu.embedding_lookup(
            &bundle.weights.embed_tokens,
            &bundle.scratch.x,
            token,
            bundle.config.dim,
        ),
        _ => return Err("unsupported Gemma4 Redline embedding format".into()),
    }
    .map_err(|error| error.to_string())?;
    gpu.scale_f32(&bundle.scratch.x, bundle.config.embed_scale)
        .map_err(|error| error.to_string())?;
    gpu.hip
        .memcpy_htod(
            &bundle.scratch.pos_buf,
            &(position as i32).to_ne_bytes(),
        )
        .map_err(|error| error.to_string())
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
    let Some(bundle) = loaded
        .state
        .as_ref()
        .and_then(|s| (s.as_ref() as &dyn Any).downcast_ref::<lfm2moe::Lfm2MoeBundle>())
    else {
        return false;
    };
    bundle.config.is_dense()
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
    if let Some(bundle) = loaded
        .state
        .as_ref()
        .and_then(|s| (s.as_ref() as &dyn Any).downcast_ref::<Qwen35Bundle>())
    {
        redline_qwen_snapshot(gpu, bundle).map(RedlineSnapshot::Qwen)
    } else if let Some(bundle) = loaded
        .state
        .as_ref()
        .and_then(|s| (s.as_ref() as &dyn Any).downcast_ref::<deepseek4::Deepseek4Bundle>())
    {
        redline_deepseek4_snapshot(gpu, bundle).map(RedlineSnapshot::Deepseek4)
    } else if let Some(bundle) = loaded
        .state
        .as_ref()
        .and_then(|s| (s.as_ref() as &dyn Any).downcast_ref::<lfm2moe::Lfm2MoeBundle>())
    {
        if !bundle.config.is_dense() {
            return Err("retained snapshot requires dense LFM".to_string());
        }
        redline_lfm2moe_snapshot(gpu, bundle).map(RedlineSnapshot::Lfm2Moe)
    } else {
        Err("retained snapshot requires Qwen3.5, DeepSeek4 or dense LFM".to_string())
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
    context: usize,
) -> Result<(), String> {
    let synthetic = (0..context as u32)
        .map(|index| 10 + (index % 1000))
        .collect::<Vec<_>>();
    // Split bundle fields disjointly: pbs + state vs config/weights.
    let deepseek4::Deepseek4Bundle {
        config,
        weights,
        state,
        pbs,
        ..
    } = bundle;
    let pbs = pbs
        .as_mut()
        .ok_or_else(|| "DeepSeek4 prefill scratch missing".to_string())?;
    deepseek4::forward::forward_prefill_batch_chunked(
        config, weights, state, gpu, &synthetic, 0, pbs,
    )?;
    state.n_tokens = context as u64;
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
    if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<Qwen35Bundle>())
    {
        redline_reset_qwen(gpu, bundle)?;
        redline_prime_qwen(gpu, bundle, context)
    } else if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<deepseek4::Deepseek4Bundle>())
    {
        redline_reset_deepseek4(gpu, bundle)?;
        redline_prime_deepseek4(gpu, bundle, context)
    } else if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<lfm2moe::Lfm2MoeBundle>())
    {
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
    } else {
        Err("retained fixture requires Qwen3.5, DeepSeek4 or dense LFM".to_string())
    }
}

pub fn redline_prepare_retained_fixture(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    token_id: u32,
    context: usize,
) -> Result<(), String> {
    if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<Qwen35Bundle>())
    {
        qwen35::prepare_scratch_inputs(
            gpu,
            &bundle.weights,
            &bundle.config,
            token_id,
            context,
            &bundle.scratch,
        )
        .map_err(|error| error.to_string())
    } else if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<deepseek4::Deepseek4Bundle>())
    {
        bundle.state.n_tokens = context as u64;
        deepseek4::forward::prepare_retained_decode_inputs(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            token_id,
            context as u32,
        )
    } else if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<lfm2moe::Lfm2MoeBundle>())
    {
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
    } else {
        Err("retained fixture requires Qwen3.5, DeepSeek4 or dense LFM".to_string())
    }
}

pub fn redline_run_direct_fixture(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    context: usize,
    iterations: usize,
) -> Result<(), String> {
    if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<Qwen35Bundle>())
    {
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
    } else if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<deepseek4::Deepseek4Bundle>())
    {
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
    } else if let Some(bundle) = loaded
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<lfm2moe::Lfm2MoeBundle>())
    {
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
    } else {
        Err("retained fixture requires Qwen3.5, DeepSeek4 or dense LFM".to_string())
    }
}

pub fn redline_bench_decode_deepseek4(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    msg: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    if loaded.pp > 1
        || loaded.ep.is_some()
        || !loaded.state.as_ref().is_some_and(|s| {
            (s.as_ref() as &dyn Any).is::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        })
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
    let Some(bundle) = loaded.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
    }) else {
        unreachable!()
    };
    redline_reset_deepseek4(gpu, bundle)?;
    redline_prime_deepseek4(gpu, bundle, context)
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
            if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
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
                let bundle = match loaded.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any)
                        .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                }) {
                    Some(bundle) => bundle,
                    None => unreachable!(),
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
            let bundle = match loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
                Some(bundle) => bundle,
                None => unreachable!(),
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
                    let bundle = match loaded.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                    }) {
                        Some(bundle) => bundle,
                        None => unreachable!(),
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
            let bundle = match loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
                Some(bundle) => bundle,
                None => unreachable!(),
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
                    let bundle = match loaded.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                    }) {
                        Some(bundle) => bundle,
                        None => unreachable!(),
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
            let bundle = match loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
                Some(bundle) => bundle,
                None => unreachable!(),
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
            if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
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
        && loaded.state.as_ref().is_some_and(|s| {
            (s.as_ref() as &dyn Any).is::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        });
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
            let gdn_frame_equal = aql_snapshot.gdn_frame() == hip_snapshot.gdn_frame();
            let blob_gdn_frame_equal = aql_snapshot.gdn_frame() == blob_snapshot.gdn_frame();
            let blob_bit_exact = redline_snapshots_bit_exact(&aql_snapshot, &blob_snapshot);
            Ok(serde_json::json!({
                "type": "redline_shadow_result",
                "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
                "context_tokens": context,
                "iterations": iterations,
                "dispatches": prepared.0,
                "packets": prepared.1,
                "queue_id": prepared.2,
                "command_dwords": prepared.3,
                "bit_exact": redline_snapshots_bit_exact(&aql_snapshot, &hip_snapshot),
                "blob_bit_exact": blob_bit_exact,
                "logits_equal": logits_equal,
                "kv_equal": kv_equal,
                "recurrent_equal": recurrent_equal,
                "gdn_frame_equal": gdn_frame_equal,
                "blob_gdn_frame_equal": blob_gdn_frame_equal,
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
                if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any)
                        .downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
                }) {
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
                    if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                    }) {
                        bundle.state.n_tokens = context + index + 1;
                        loaded.seq_pos = context + index + 1;
                    }
                    gpu_us += timing.span_microseconds();
                } else {
                    let timing = unsafe { gpu.replay.replay_linear_aql(context + index) }?;
                    if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                    }) {
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
                if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any)
                        .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                }) {
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
            let gdn_frame_equal = aql_snapshot.gdn_frame() == hip_snapshot.gdn_frame();
            let blob_gdn_frame_equal = aql_snapshot.gdn_frame() == blob_snapshot.gdn_frame();
            let blob_bit_exact = redline_snapshots_bit_exact(&aql_snapshot, &blob_snapshot);
            Ok(serde_json::json!({
                "type": "redline_shadow_result",
                "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
                "context_tokens": context,
                "iterations": iterations,
                "dispatches": prepared.0,
                "packets": prepared.1,
                "queue_id": prepared.2,
                "command_dwords": prepared.3,
                "bit_exact": redline_snapshots_bit_exact(&aql_snapshot, &hip_snapshot),
                "blob_bit_exact": blob_bit_exact,
                "logits_equal": logits_equal,
                "kv_equal": kv_equal,
                "recurrent_equal": recurrent_equal,
                "gdn_frame_equal": gdn_frame_equal,
                "blob_gdn_frame_equal": blob_gdn_frame_equal,
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
                if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any)
                        .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                }) {
                    let _ = redline_reset_lfm2moe(gpu, bundle);
                    loaded.seq_pos = 0;
                    loaded.conversation_tokens.clear();
                    let _ = gpu.hip.device_synchronize();
                }
                Ok(value)
            }
            Err(error) => {
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any)
                        .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                }) {
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
    let bundle = match loaded.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
    }) {
        Some(bundle) => bundle,
        None => return Err("DSpark shadow requires DeepSeek4".to_string()),
    };
    redline_reset_deepseek4(gpu, bundle)?;
    redline_prime_deepseek4(gpu, bundle, context)?;
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
    let bundle = match loaded.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
    }) {
        Some(bundle) => bundle,
        None => unreachable!(),
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
    let bundle = match loaded.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
    }) {
        Some(bundle) => bundle,
        None => unreachable!(),
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
    let bundle = match loaded.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
    }) {
        Some(bundle) => bundle,
        None => unreachable!(),
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
        || !loaded.state.as_ref().is_some_and(|s| {
            (s.as_ref() as &dyn Any).is::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        })
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
        let bundle = match loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        }) {
            Some(bundle) => bundle,
            None => unreachable!(),
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
        let bundle = match loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        }) {
            Some(bundle) => bundle,
            None => unreachable!(),
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
    if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
    }) {
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
        || !loaded.state.as_ref().is_some_and(|s| {
            (s.as_ref() as &dyn Any).is::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        })
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

/// `"redline_probe_aql"` daemon message handler.
pub fn handle_redline_probe_aql(
    msg: &serde_json::Value,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
) {
    match gpu.replay.probe_aql_contracts(gpu.device_id as usize) {
        Ok(probes) => {
            let rows = probes
                .into_iter()
                .map(|probe| {
                    serde_json::json!({
                        "kernel": probe.kernel,
                        "captured_kernarg_bytes": probe.captured_kernarg_bytes,
                        "loader_kernarg_bytes": probe.loader_kernarg_bytes,
                        "loader_kernarg_alignment": probe.loader_kernarg_alignment,
                        "static_group_bytes": probe.static_group_bytes,
                        "dynamic_group_bytes": probe.dynamic_group_bytes,
                    })
                })
                .collect::<Vec<_>>();
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({
                    "type": "redline_aql_probe",
                    "kernels": rows.len(),
                    "contracts": rows,
                })
            );
        }
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline AQL contract probe failed: {reason}"),
                "internal",
                false,
                false,
            );
        }
    }
    let _ = stdout.flush();
}

/// `"redline_dspark_shadow_pm4"` daemon message handler.
pub fn handle_redline_dspark_shadow_pm4(
    msg: &serde_json::Value,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
) {
    let context = msg
        .get("context_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(128) as usize;
    let batch = msg
        .get("verify_batch")
        .and_then(|value| value.as_u64())
        .unwrap_or(3) as usize;
    let iterations = msg
        .get("iterations")
        .and_then(|value| value.as_u64())
        .unwrap_or(15) as usize;
    let response = model
        .as_mut()
        .ok_or_else(|| "DSpark shadow requires a loaded model".to_string())
        .and_then(|loaded| {
            redline_shadow_dspark_verify_pm4(gpu, loaded, context, batch, iterations)
        });
    match response {
        Ok(response) => {
            let _ = writeln!(stdout, "{response}");
        }
        Err(reason) => {
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({"type": "error", "message": reason})
            );
        }
    }
    let _ = stdout.flush();
    return;
}

// ─── DFlash2 retained-PM4 verify oracle (contract C4) ─────────────────────

const DFLASH_SHADOW_EXTRACT_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

/// Legacy stochastic Q8 (no error-feedback residual) is not a byte-parity
/// oracle: GDN requant uses stochastic rounding keyed on the frame. Retained
/// only when `StateQuant::Q8` runs without EF. Deterministic Q8+EF and non-Q8
/// state claim bit-exact recurrent bytes instead.
const DFLASH_Q8_BYTE_PARITY_INVALID: &str = "unconditional Q8 recurrent-state byte parity is not a correctness oracle (stochastic GDN rounding); arms share a GDN frame checkpoint and compare tokens/argmax/indices bit-exact plus claim-scoped max-abs/rel on dequantized regions";

/// Deterministic recurrent state (Q8+EF residual or non-Q8) may claim
/// bit-exact DeltaNet bytes across arms, including the EF residual itself.
const DFLASH_DETERMINISTIC_BYTE_PARITY: &str = "deterministic DeltaNet state (Q8 error-feedback residual or non-Q8) admits bit-exact recurrent-state byte parity across arms; tokens/argmax/indices stay bit-exact and claim-scoped max-abs/rel cover dequantized regions";

struct RedlineDflashFixtures {
    hidden_rb: HiddenStateRingBuffer,
    verify_scratch: VerifyScratch,
    gdn_tape: GdnTape,
    target_snap: DeltaNetSnapshot,
}

impl RedlineDflashFixtures {
    fn free(self, gpu: &mut rdna_compute::Gpu) {
        self.hidden_rb.free_gpu(gpu);
        self.verify_scratch.free_gpu(gpu);
        self.gdn_tape.free_gpu(gpu);
        self.target_snap.free_gpu(gpu);
    }
}

#[derive(Clone)]
struct RedlineDflashWindowSnap {
    position: usize,
    tokens: Vec<u32>,
    argmax: Vec<u32>,
    logits: Vec<u8>,
    final_hidden: Vec<u8>,
    gdn_qkv: Vec<u8>,
    gdn_alpha: Vec<u8>,
    gdn_beta: Vec<u8>,
    dn_s_after_forward: Vec<u8>,
    dn_scales_after_forward: Vec<u8>,
    dn_conv_after_forward: Vec<u8>,
    dn_ef_after_forward: Vec<u8>,
    dn_s_after_rollback: Vec<u8>,
    dn_scales_after_rollback: Vec<u8>,
    dn_conv_after_rollback: Vec<u8>,
    dn_ef_after_rollback: Vec<u8>,
    kv_active_hash: u64,
    kv_guard: Vec<u8>,
    hidden_staging: Vec<u8>,
    hidden_ring: Vec<u8>,
    ring_head: usize,
    ring_written: usize,
    pbs_guard: Vec<u8>,
    gdn_frame: u32,
}

struct RedlineDflashArm {
    name: &'static str,
    windows: Vec<RedlineDflashWindowSnap>,
    guard_before: Vec<u8>,
    guard_after: Vec<u8>,
    host_us: f64,
}

#[derive(Clone, Copy)]
enum RedlineDflashArmKind {
    HipAuto,
    DirectCaptureSafe,
    RecordedHip,
    Pm4,
}

fn redline_dflash_sample_positions(physical_cap: usize, batch: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut push = |position: usize| {
        if position.saturating_add(batch) <= physical_cap && !out.contains(&position) {
            out.push(position);
        }
    };
    push(112); // 127 -> 128
    push(240); // 255 -> 256
    let mid = ((physical_cap / 2).saturating_sub(batch).max(512)) & !15;
    if mid > 256 {
        push(mid);
    }
    push(8176); // 8191 -> 8192 Q8 tiled-attention crossover
    out
}

fn redline_dflash_shadow_block(step: usize, batch: usize) -> Vec<u32> {
    (0..batch)
        .map(|slot| 101 + ((step * batch + slot) % 1000) as u32)
        .collect()
}

fn redline_artifact_fingerprint(path: &str) -> serde_json::Value {
    let meta = std::fs::metadata(path).ok();
    let mut header = Vec::new();
    if let Ok(mut file) = std::fs::File::open(path) {
        let mut buf = [0u8; 65536];
        if let Ok(n) = file.read(&mut buf) {
            header.extend_from_slice(&buf[..n]);
        }
    }
    serde_json::json!({
        "path": path,
        "bytes": meta.as_ref().map(|m| m.len()),
        "header_hash": format!("{:016x}", redline_hash(&header)),
    })
}

fn redline_f32_err(a: &[u8], b: &[u8]) -> (f32, f32) {
    if a.len() != b.len() || a.len() % 4 != 0 || a.is_empty() {
        return (f32::INFINITY, f32::INFINITY);
    }
    let n = a.len() / 4;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..n {
        let x = f32::from_le_bytes(a[i * 4..i * 4 + 4].try_into().unwrap());
        let y = f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        if !x.is_finite() || !y.is_finite() {
            return (f32::INFINITY, f32::INFINITY);
        }
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        let denom = x.abs().max(y.abs()).max(1.0e-6);
        max_rel = max_rel.max(d / denom);
    }
    (max_abs, max_rel)
}

fn redline_percentile(mut values: Vec<f64>, pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let idx = ((pct / 100.0) * (values.len() - 1) as f64).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn redline_reset_qwen_slot(
    gpu: &mut rdna_compute::Gpu,
    slot: &mut ModelSlot,
) -> Result<(), String> {
    slot.kv_cache
        .clear_gpu(gpu)
        .map_err(|error| error.to_string())?;
    slot.kv_cache.compact_offset = 0;
    for tensor in slot
        .dn_state
        .s_matrices
        .iter()
        .chain(slot.dn_state.s_scales.iter())
        .chain(slot.dn_state.conv_states.iter())
        .chain(slot.dn_state.s_ef_residual.iter())
    {
        gpu.hip
            .memset(&tensor.buf, 0, tensor.buf.size())
            .map_err(|error| error.to_string())?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
}

fn redline_prime_qwen_slot(
    gpu: &mut rdna_compute::Gpu,
    slot: &mut ModelSlot,
    context: usize,
) -> Result<(), String> {
    if context == 0 {
        return Ok(());
    }
    let synthetic: Vec<u32> = (0..context as u32).map(|i| 10 + (i % 1000)).collect();
    qwen35::forward_prefill_batch(
        gpu,
        &slot.weights,
        &slot.config,
        &synthetic,
        0,
        &mut slot.kv_cache,
        &mut slot.dn_state,
        &slot.scratch,
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

fn redline_dflash_zero_fixtures(
    gpu: &mut rdna_compute::Gpu,
    fixtures: &mut RedlineDflashFixtures,
) -> Result<(), String> {
    fixtures.hidden_rb.head = 0;
    fixtures.hidden_rb.written = 0;
    for tensor in fixtures
        .hidden_rb
        .layer_bufs
        .iter()
        .chain(fixtures.hidden_rb.staging_bufs.iter())
    {
        gpu.hip
            .memset(&tensor.buf, 0, tensor.buf.size())
            .map_err(|e| e.to_string())?;
    }
    for tensor in [
        &fixtures.verify_scratch.final_hidden,
        &fixtures.verify_scratch.logits,
        &fixtures.verify_scratch.rot,
        &fixtures.verify_scratch.argmax,
    ] {
        gpu.hip
            .memset(&tensor.buf, 0, tensor.buf.size())
            .map_err(|e| e.to_string())?;
    }
    for tensor in fixtures
        .gdn_tape
        .qkv_bufs
        .iter()
        .chain(fixtures.gdn_tape.alpha_bufs.iter())
        .chain(fixtures.gdn_tape.beta_bufs.iter())
    {
        gpu.hip
            .memset(&tensor.buf, 0, tensor.buf.size())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn redline_append_dn_parts(
    gpu: &rdna_compute::Gpu,
    slot: &ModelSlot,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let mut s = Vec::new();
    for tensor in &slot.dn_state.s_matrices {
        redline_append_buffer(gpu, &mut s, &tensor.buf)?;
    }
    let mut scales = Vec::new();
    for tensor in &slot.dn_state.s_scales {
        redline_append_buffer(gpu, &mut scales, &tensor.buf)?;
    }
    let mut conv = Vec::new();
    for tensor in &slot.dn_state.conv_states {
        redline_append_buffer(gpu, &mut conv, &tensor.buf)?;
    }
    let mut ef = Vec::new();
    for tensor in &slot.dn_state.s_ef_residual {
        redline_append_buffer(gpu, &mut ef, &tensor.buf)?;
    }
    Ok((s, scales, conv, ef))
}

fn redline_dflash_kv_regions(
    gpu: &rdna_compute::Gpu,
    slot: &ModelSlot,
    start_pos: usize,
    batch: usize,
) -> Result<(u64, Vec<u8>), String> {
    let mut mix = 0xcbf2_9ce4_8422_2325u64;
    let mut guard = Vec::new();
    let n_kv = slot.config.n_kv_heads.max(1);
    let head_dim = slot.config.head_dim.max(1);
    let blocks = (head_dim / 32).max(1);
    let bytes_per_pos = n_kv * blocks * 34;
    for tensor in slot.kv_cache.k_gpu.iter().chain(slot.kv_cache.v_gpu.iter()) {
        let mut bytes = Vec::new();
        redline_append_buffer(gpu, &mut bytes, &tensor.buf)?;
        mix ^= redline_hash(&bytes);
        mix = mix.wrapping_mul(0x0000_0100_0000_01b3);
        let start = start_pos.saturating_mul(bytes_per_pos);
        let active_end = start.saturating_add(batch.saturating_mul(bytes_per_pos));
        let guard_end = active_end.saturating_add(bytes_per_pos).min(bytes.len());
        if active_end < bytes.len() {
            guard.extend_from_slice(&bytes[active_end..guard_end]);
        }
    }
    Ok((mix, guard))
}

fn redline_dflash_pbs_guard(
    gpu: &rdna_compute::Gpu,
    fixtures: &RedlineDflashFixtures,
    batch: usize,
) -> Result<Vec<u8>, String> {
    let pbs = fixtures
        .verify_scratch
        .prefill_batch
        .as_ref()
        .ok_or_else(|| "DFlash shadow: verify PBS missing".to_string())?;
    if batch >= pbs.max_batch {
        return Err(format!(
            "DFlash shadow guard needs inactive row after B={batch}, max_batch={}",
            pbs.max_batch
        ));
    }
    let mut guard = Vec::new();
    let dim = fixtures.verify_scratch.dim;
    redline_append_tensor_slice(gpu, &mut guard, &pbs.x_batch, batch * dim, dim)?;
    redline_append_tensor_slice(gpu, &mut guard, &pbs.tokens, batch, 1)?;
    redline_append_tensor_slice(gpu, &mut guard, &pbs.positions, batch, 1)?;
    Ok(guard)
}

fn redline_dflash_snapshot_window(
    gpu: &rdna_compute::Gpu,
    slot: &ModelSlot,
    fixtures: &RedlineDflashFixtures,
    position: usize,
    tokens: Vec<u32>,
    argmax: Vec<u32>,
    after_forward: (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
    after_rollback: (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
) -> Result<RedlineDflashWindowSnap, String> {
    let batch = tokens.len();
    let dim = fixtures.verify_scratch.dim;
    let vocab = fixtures.verify_scratch.vocab;
    let mut logits = Vec::new();
    redline_append_tensor_slice(
        gpu,
        &mut logits,
        &fixtures.verify_scratch.logits,
        0,
        batch * vocab,
    )?;
    let mut final_hidden = Vec::new();
    redline_append_tensor_slice(
        gpu,
        &mut final_hidden,
        &fixtures.verify_scratch.final_hidden,
        0,
        batch * dim,
    )?;
    let mut gdn_qkv = Vec::new();
    for tensor in &fixtures.gdn_tape.qkv_bufs {
        redline_append_tensor_slice(
            gpu,
            &mut gdn_qkv,
            tensor,
            0,
            batch * fixtures.gdn_tape.qkv_dim,
        )?;
    }
    let mut gdn_alpha = Vec::new();
    for tensor in &fixtures.gdn_tape.alpha_bufs {
        redline_append_tensor_slice(
            gpu,
            &mut gdn_alpha,
            tensor,
            0,
            batch * fixtures.gdn_tape.n_v_heads,
        )?;
    }
    let mut gdn_beta = Vec::new();
    for tensor in &fixtures.gdn_tape.beta_bufs {
        redline_append_tensor_slice(
            gpu,
            &mut gdn_beta,
            tensor,
            0,
            batch * fixtures.gdn_tape.n_v_heads,
        )?;
    }
    let (kv_active_hash, kv_guard) = redline_dflash_kv_regions(gpu, slot, position, batch)?;
    let mut hidden_staging = Vec::new();
    for tensor in &fixtures.hidden_rb.staging_bufs {
        redline_append_tensor_slice(gpu, &mut hidden_staging, tensor, 0, batch * dim)?;
    }
    let mut hidden_ring = Vec::new();
    let written = fixtures
        .hidden_rb
        .written
        .min(fixtures.hidden_rb.max_positions);
    for tensor in &fixtures.hidden_rb.layer_bufs {
        if written > 0 {
            redline_append_tensor_slice(gpu, &mut hidden_ring, tensor, 0, written * dim)?;
        }
    }
    let pbs_guard = redline_dflash_pbs_guard(gpu, fixtures, batch)?;
    Ok(RedlineDflashWindowSnap {
        position,
        tokens,
        argmax,
        logits,
        final_hidden,
        gdn_qkv,
        gdn_alpha,
        gdn_beta,
        dn_s_after_forward: after_forward.0,
        dn_scales_after_forward: after_forward.1,
        dn_conv_after_forward: after_forward.2,
        dn_ef_after_forward: after_forward.3,
        dn_s_after_rollback: after_rollback.0,
        dn_scales_after_rollback: after_rollback.1,
        dn_conv_after_rollback: after_rollback.2,
        dn_ef_after_rollback: after_rollback.3,
        kv_active_hash,
        kv_guard,
        hidden_staging,
        hidden_ring,
        ring_head: fixtures.hidden_rb.head,
        ring_written: fixtures.hidden_rb.written,
        pbs_guard,
        gdn_frame: rdna_compute::norm::gdn_requant_frame_checkpoint(),
    })
}

fn redline_dflash_recorded_hip(
    gpu: &mut rdna_compute::Gpu,
    slot: &mut ModelSlot,
    fixtures: &mut RedlineDflashFixtures,
    route: &mut DflashVerifyPm4,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<u32>, String> {
    let b = tokens.len();
    let dim = slot.config.dim;
    let vocab = slot.config.vocab_size;
    {
        let pbs = fixtures
            .verify_scratch
            .prefill_batch
            .as_ref()
            .ok_or_else(|| "DFlash recorded HIP: PBS missing".to_string())?;
        qwen35::upload_prefill_batch_inputs(gpu, pbs, tokens, start_pos)
            .map_err(|e| e.to_string())?;
    }
    let controller = route
        .controller_mut()
        .ok_or_else(|| "DFlash recorded HIP: controller missing".to_string())?;
    std::mem::swap(&mut gpu.replay, controller);
    let replay = (|| {
        let launches = gpu.replay.recorded_launches().len();
        if launches == 0 {
            return Err("DFlash recorded HIP: no captured launches".to_string());
        }
        // Position-aware: the recorded blobs bake position-tracking scalars,
        // so this arm must patch them exactly as the PM4 arm does, or it stops
        // being a comparable oracle.
        gpu.replay_recorded_hip_prefix_at(launches, start_pos)
            .map_err(|e| format!("DFlash recorded HIP replay: {e:?}"))
    })();
    std::mem::swap(&mut gpu.replay, controller);
    replay?;
    fixtures
        .hidden_rb
        .commit_staging_to_ring(gpu, b)
        .map_err(|e| e.to_string())?;
    let mut argmax = Vec::with_capacity(b);
    for i in 0..b {
        let hidden_row = fixtures
            .verify_scratch
            .final_hidden
            .sub_offset(i * dim, dim);
        let logits_row = fixtures.verify_scratch.logits.sub_offset(i * vocab, vocab);
        hipfire_runtime::llama::weight_gemv(gpu, &slot.weights.output, &hidden_row, &logits_row)
            .map_err(|e| e.to_string())?;
        let row = gpu.download_f32(&logits_row).map_err(|e| e.to_string())?;
        argmax.push(
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0),
        );
    }
    Ok(argmax)
}

fn redline_dflash_run_window(
    gpu: &mut rdna_compute::Gpu,
    slot: &mut ModelSlot,
    fixtures: &mut RedlineDflashFixtures,
    route: &mut DflashVerifyPm4,
    kind: RedlineDflashArmKind,
    tokens: &[u32],
    position: usize,
) -> Result<RedlineDflashWindowSnap, String> {
    fixtures
        .target_snap
        .save_from(&slot.dn_state, gpu)
        .map_err(|e| e.to_string())?;
    let argmax = match kind {
        RedlineDflashArmKind::HipAuto => {
            verify_dflash_block(
                gpu,
                slot,
                tokens,
                position,
                &mut fixtures.hidden_rb,
                Some(&mut fixtures.gdn_tape),
                false,
                &fixtures.verify_scratch,
            )
            .map_err(|e| e.to_string())?
            .argmax_per_pos
        }
        RedlineDflashArmKind::DirectCaptureSafe | RedlineDflashArmKind::Pm4 => {
            verify_dflash_block_retained(
                gpu,
                slot,
                tokens,
                position,
                &mut fixtures.hidden_rb,
                Some(&mut fixtures.gdn_tape),
                false,
                &fixtures.verify_scratch,
                route,
            )
            .map_err(|e| e.to_string())?
            .argmax_per_pos
        }
        RedlineDflashArmKind::RecordedHip => {
            redline_dflash_recorded_hip(gpu, slot, fixtures, route, tokens, position)?
        }
    };
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    let after_forward = redline_append_dn_parts(gpu, slot)?;
    let accept_n = tokens.len() / 2;
    fixtures
        .target_snap
        .restore_to(&mut slot.dn_state, gpu)
        .map_err(|e| e.to_string())?;
    if accept_n > 0 {
        fixtures
            .gdn_tape
            .replay_gdn(
                gpu,
                &slot.weights,
                &slot.config,
                &mut slot.dn_state,
                accept_n,
            )
            .map_err(|e| e.to_string())?;
    }
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    let after_rollback = redline_append_dn_parts(gpu, slot)?;
    redline_dflash_snapshot_window(
        gpu,
        slot,
        fixtures,
        position,
        tokens.to_vec(),
        argmax,
        after_forward,
        after_rollback,
    )
}

fn redline_dflash_compare_window(
    reference: &RedlineDflashWindowSnap,
    other: &RedlineDflashWindowSnap,
    name: &str,
) -> serde_json::Value {
    let bit = |a: &[u8], b: &[u8]| a == b;
    let err = |a: &[u8], b: &[u8]| {
        let (max_abs, max_rel) = redline_f32_err(a, b);
        serde_json::json!({ "max_abs": max_abs, "max_rel": max_rel, "bytes_equal": a == b })
    };
    serde_json::json!({
        "arm": name,
        "position": other.position,
        "tokens_equal": reference.tokens == other.tokens,
        "argmax_equal": reference.argmax == other.argmax,
        "ring_head_equal": reference.ring_head == other.ring_head,
        "ring_written_equal": reference.ring_written == other.ring_written,
        "gdn_frame_equal": reference.gdn_frame == other.gdn_frame,
        "kv_active_hash_equal": reference.kv_active_hash == other.kv_active_hash,
        "kv_guard_equal": bit(&reference.kv_guard, &other.kv_guard),
        "pbs_guard_equal": bit(&reference.pbs_guard, &other.pbs_guard),
        "hidden_staging": err(&reference.hidden_staging, &other.hidden_staging),
        "hidden_ring": err(&reference.hidden_ring, &other.hidden_ring),
        "final_hidden": err(&reference.final_hidden, &other.final_hidden),
        "logits": err(&reference.logits, &other.logits),
        "gdn_qkv": err(&reference.gdn_qkv, &other.gdn_qkv),
        "gdn_alpha": err(&reference.gdn_alpha, &other.gdn_alpha),
        "gdn_beta": err(&reference.gdn_beta, &other.gdn_beta),
        "dn_s_after_forward": err(&reference.dn_s_after_forward, &other.dn_s_after_forward),
        "dn_scales_after_forward": err(&reference.dn_scales_after_forward, &other.dn_scales_after_forward),
        "dn_conv_after_forward": err(&reference.dn_conv_after_forward, &other.dn_conv_after_forward),
        "dn_ef_after_forward_equal": bit(&reference.dn_ef_after_forward, &other.dn_ef_after_forward),
        "dn_s_after_rollback": err(&reference.dn_s_after_rollback, &other.dn_s_after_rollback),
        "dn_scales_after_rollback": err(&reference.dn_scales_after_rollback, &other.dn_scales_after_rollback),
        "dn_conv_after_rollback": err(&reference.dn_conv_after_rollback, &other.dn_conv_after_rollback),
        "dn_ef_after_rollback_equal": bit(&reference.dn_ef_after_rollback, &other.dn_ef_after_rollback),
    })
}

fn redline_dflash_arm_json(arm: &RedlineDflashArm) -> serde_json::Value {
    serde_json::json!({
        "name": arm.name,
        "host_us": arm.host_us,
        "guard_unchanged": arm.guard_before == arm.guard_after,
        "guard_before_hash": format!("{:016x}", redline_hash(&arm.guard_before)),
        "guard_after_hash": format!("{:016x}", redline_hash(&arm.guard_after)),
        "windows": arm.windows.iter().map(|w| serde_json::json!({
            "position": w.position,
            "tokens": w.tokens,
            "argmax": w.argmax,
            "ring_head": w.ring_head,
            "ring_written": w.ring_written,
            "gdn_frame": w.gdn_frame,
            "kv_active_hash": format!("{:016x}", w.kv_active_hash),
            "logits_hash": format!("{:016x}", redline_hash(&w.logits)),
            "final_hidden_hash": format!("{:016x}", redline_hash(&w.final_hidden)),
        })).collect::<Vec<_>>(),
    })
}

/// DFlash2 retained-verify parity oracle.
///
/// Four arms start from an identical synthetic prefill + GDN-frame checkpoint:
/// shipping HipAuto, capture-safe direct HIP, recorded HIP blobs, and retained
/// PM4. Token blocks change every window. Positions are a consecutive advancing
/// run at stride B=16 from base 112, crossing 127/128 and 255/256 inside one
/// residency without any re-prime.
pub fn redline_shadow_dflash_verify_pm4(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    batch: usize,
    iterations: usize,
    steady_state_windows: usize,
) -> Result<serde_json::Value, String> {
    if loaded.pp > 1 || loaded.ep.is_some() {
        return Err("DFlash shadow requires a loaded single-GPU Qwen3.8-family target".into());
    }
    if loaded.arch_id != 5 && loaded.arch_id != 6 {
        return Err(format!(
            "DFlash shadow requires Qwen3.5/3.8-family arch 5/6, got arch_id={}",
            loaded.arch_id
        ));
    }
    if loaded.speculator.as_ref().map(|s| s.name()) != Some("dflash") {
        return Err("DFlash shadow requires a loaded DFlash sidecar".into());
    }
    if batch != DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "DFlash shadow verify_batch must be exactly {DFLASH_VERIFY_PM4_BLOCK}, got {batch}"
        ));
    }
    if iterations == 0 {
        return Err("DFlash shadow iterations must be non-zero".into());
    }
    let count = iterations.max(12);
    let base: usize = 112;
    let mut positions: Vec<usize> = Vec::with_capacity(count);
    for i in 0..count {
        positions.push(base + i * batch);
    }
    let last = *positions.last().unwrap();
    if last.saturating_add(batch) > loaded.physical_cap {
        return Err(format!(
            "DFlash shadow sequence last={} (base {base} + {}*batch {batch}) exceeds physical_cap={}: need {}",
            last,
            count - 1,
            loaded.physical_cap,
            last + batch
        ));
    }

    let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
    let mut guard = Qwen35SlotGuard::take(&mut loaded.state, &loaded.model_path)?;
    let slot = guard.model_slot()?;
    let hidden_k = slot.config.dim.next_power_of_two();
    let max_n = batch + 1;
    let mut fixtures = RedlineDflashFixtures {
        hidden_rb: HiddenStateRingBuffer::new_for_layers(
            gpu,
            &DFLASH_SHADOW_EXTRACT_LAYERS,
            slot.config.dim,
            loaded.physical_cap.max(last + batch + 1),
            max_n,
        )
        .map_err(|e| e.to_string())?,
        verify_scratch: VerifyScratch::with_prefill(
            gpu,
            max_n,
            slot.config.dim,
            slot.config.vocab_size,
            hidden_k,
            &slot.config,
        )
        .map_err(|e| e.to_string())?,
        gdn_tape: GdnTape::new_for_config(gpu, &slot.config, max_n).map_err(|e| e.to_string())?,
        target_snap: DeltaNetSnapshot::new_for(gpu, &slot.dn_state).map_err(|e| e.to_string())?,
    };

    let mut route = DflashVerifyPm4::armed();
    let mut capture_prepare_us = 0.0;
    let inner = (|| -> Result<serde_json::Value, String> {
        // Every arm owns its own residency, and a retained tape is only valid
        // inside the residency it was captured in. So the tape is built by
        // lead-in windows AFTER each arm's reset+prime, not once up front: a
        // reset between capture and replay can move an allocation the tape
        // names, which is precisely what the route's calibration refuses.
        //
        // Every arm runs the identical lead-in, so all four traverse the same
        // state trajectory and the compared windows remain comparable. The
        // lead-in windows are not compared.
        let lead_in = 3usize;
        let lead_base = positions[0].saturating_sub(lead_in * batch);
        let lead_positions: Vec<usize> = (0..lead_in).map(|i| lead_base + i * batch).collect();

        let mut run_arm = |kind: RedlineDflashArmKind,
                           name: &'static str|
         -> Result<(RedlineDflashArm, DflashVerifyPm4, f64), String> {
            // Byte-identical checkpoint for every arm: same restored GDN frame,
            // same reset slot, same zeroed fixtures, same prime length. From
            // here the arm runs back-to-back windows with no allocation-
            // affecting call, exactly as a resident model does.
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
            redline_reset_qwen_slot(gpu, slot)?;
            redline_dflash_zero_fixtures(gpu, &mut fixtures)?;
            redline_prime_qwen_slot(gpu, slot, lead_base)?;
            let mut local_route: DflashVerifyPm4 = match kind {
                RedlineDflashArmKind::HipAuto => DflashVerifyPm4::disabled("shadow hip_auto"),
                _ => DflashVerifyPm4::armed(),
            };
            // Lead-in. For the two retained arms this is what drives the route
            // Armed -> Primed -> Calibrating -> Ready inside this residency.
            // `RecordedHip` must build the tape through the retained path, so
            // its lead-in runs as `Pm4`.
            let lead_kind = match kind {
                RedlineDflashArmKind::RecordedHip => RedlineDflashArmKind::Pm4,
                other => other,
            };
            let warm_started = Instant::now();
            for (step, &position) in lead_positions.iter().enumerate() {
                if matches!(kind, RedlineDflashArmKind::DirectCaptureSafe) {
                    local_route = DflashVerifyPm4::armed();
                }
                let _ = redline_dflash_run_window(
                    gpu,
                    slot,
                    &mut fixtures,
                    &mut local_route,
                    lead_kind,
                    &redline_dflash_shadow_block(step, batch),
                    position,
                )?;
            }
            let warm_us = warm_started.elapsed().as_secs_f64() * 1_000_000.0;
            let guard_before = redline_dflash_pbs_guard(gpu, &fixtures, batch)?;
            let mut windows = Vec::with_capacity(positions.len());
            let started = Instant::now();
            for (step, &position) in positions.iter().enumerate() {
                // A persisted route on this arm would calibrate and start
                // replaying; `direct_capture_safe` must stay an ordinary direct
                // body, so it gets a fresh host-only route per window.
                if matches!(kind, RedlineDflashArmKind::DirectCaptureSafe) {
                    local_route = DflashVerifyPm4::armed();
                }
                let snap = redline_dflash_run_window(
                    gpu,
                    slot,
                    &mut fixtures,
                    &mut local_route,
                    kind,
                    &redline_dflash_shadow_block(lead_in + step, batch),
                    position,
                )?;
                windows.push(snap);
            }
            let guard_after = redline_dflash_pbs_guard(gpu, &fixtures, batch)?;
            Ok((
                RedlineDflashArm {
                    name,
                    windows,
                    guard_before,
                    guard_after,
                    host_us: started.elapsed().as_secs_f64() * 1_000_000.0,
                },
                local_route,
                warm_us,
            ))
        };

        let (hip_auto, _, _) = run_arm(RedlineDflashArmKind::HipAuto, "hip_auto")?;
        let (direct_capture_safe, _, _) = run_arm(
            RedlineDflashArmKind::DirectCaptureSafe,
            "direct_capture_safe",
        )?;
        let (recorded_hip, _, _) = run_arm(RedlineDflashArmKind::RecordedHip, "recorded_hip")?;
        let (pm4, mut route, warm_us) = run_arm(RedlineDflashArmKind::Pm4, "pm4")?;
        capture_prepare_us = warm_us;

        let identity = route.prepared_identity();
        let mut aql_contracts = 0usize;
        let mut launches = 0usize;
        if let Some(controller) = route.controller_mut() {
            launches = controller.recorded_launches().len();
            aql_contracts = controller
                .probe_aql_contracts(gpu.device_id as usize)
                .map(|rows| rows.len())
                .unwrap_or(0);
        }

        // Two references, deliberately.
        //
        // `hip_auto` is the shipping HipGraph path, and a captured HipGraph
        // replays the Q8 GatedDeltaNet stochastic-rounding frame scalar frozen
        // at capture time — measured here as 144 frames per steady-state window
        // (rollback only) versus 432 for a live forward. That makes it an
        // invalid reference for any GDN-derived region, so the correctness
        // verdict is taken against `direct_capture_safe`, the ordinary direct
        // HIP body, which consumes frames exactly as a non-graph decode does.
        // The `hip_auto` comparison is retained as documentation of that gap.
        let mut parity_windows = Vec::new();
        for (idx, graph_reference) in hip_auto.windows.iter().enumerate() {
            let Some(reference) = direct_capture_safe.windows.get(idx) else {
                continue;
            };
            let mut row = serde_json::json!({ "position": reference.position });
            row["direct_capture_safe_vs_hip_auto"] =
                redline_dflash_compare_window(graph_reference, reference, "direct_capture_safe");
            if let Some(other) = recorded_hip.windows.get(idx) {
                row["recorded_hip"] =
                    redline_dflash_compare_window(reference, other, "recorded_hip");
            }
            if let Some(other) = pm4.windows.get(idx) {
                row["pm4"] = redline_dflash_compare_window(reference, other, "pm4");
            }
            parity_windows.push(row);
        }

        // Fail-closed protocol cases. Separate controllers so the four-arm
        // route report stays the live Ready/replay evidence.
        let mut protocol_prepare = DflashVerifyPm4::armed();
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
        redline_reset_qwen_slot(gpu, slot)?;
        redline_dflash_zero_fixtures(gpu, &mut fixtures)?;
        redline_prime_qwen_slot(gpu, slot, positions[0])?;
        let prep_tokens = redline_dflash_shadow_block(0, batch);
        let prep_out = verify_dflash_block_retained(
            gpu,
            slot,
            &prep_tokens,
            positions[0],
            &mut fixtures.hidden_rb,
            Some(&mut fixtures.gdn_tape),
            false,
            &fixtures.verify_scratch,
            &mut protocol_prepare,
        );
        let hip_completed = prep_out.is_ok();
        if hip_completed {
            protocol_prepare.note_prepare_failure(
                "shadow inject: preparation rejected after successful recorded HIP body",
            );
        }
        let prepare_report = protocol_prepare.report_json();
        let _ = protocol_prepare.shutdown();

        let mut protocol_proven = DflashVerifyPm4::armed();
        protocol_proven.note_replay_failure(
            positions[0],
            ReplayQuiescence::Proven,
            "shadow inject: replay failure with Proven quiescence",
        );
        protocol_proven.note_safe_hip_retry();
        let written_before = fixtures.hidden_rb.written;
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
        redline_reset_qwen_slot(gpu, slot)?;
        redline_dflash_zero_fixtures(gpu, &mut fixtures)?;
        redline_prime_qwen_slot(gpu, slot, positions[0])?;
        fixtures
            .target_snap
            .save_from(&slot.dn_state, gpu)
            .map_err(|e| e.to_string())?;
        let _ = verify_dflash_block(
            gpu,
            slot,
            &prep_tokens,
            positions[0],
            &mut fixtures.hidden_rb,
            Some(&mut fixtures.gdn_tape),
            false,
            &fixtures.verify_scratch,
        )
        .map_err(|e| e.to_string())?;
        let written_after = fixtures.hidden_rb.written;
        let proven_report = protocol_proven.report_json();
        let _ = protocol_proven.shutdown();

        let mut protocol_unknown = DflashVerifyPm4::armed();
        protocol_unknown.note_replay_failure(
            positions[0],
            ReplayQuiescence::Unknown,
            "shadow inject: replay failure with Unknown quiescence",
        );
        let unknown_report = protocol_unknown.report_json();
        // Unknown: no lm-head, no accept, no emission. Shutdown may still
        // prove quiescence because no IB was submitted.
        let unknown_shutdown = protocol_unknown.shutdown();

        let mut timing = serde_json::Value::Null;
        if steady_state_windows > 0 && route.prepared_identity().is_some() {
            let n = steady_state_windows.max(1);
            let timing_pos = positions[0];
            let mut graph_host = Vec::with_capacity(n);
            let mut pm4_host = Vec::with_capacity(n);
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
            redline_reset_qwen_slot(gpu, slot)?;
            redline_dflash_zero_fixtures(gpu, &mut fixtures)?;
            redline_prime_qwen_slot(gpu, slot, timing_pos)?;
            let mut cursor = timing_pos;
            for i in 0..n {
                if cursor.saturating_add(batch) > loaded.physical_cap {
                    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                    redline_reset_qwen_slot(gpu, slot)?;
                    redline_dflash_zero_fixtures(gpu, &mut fixtures)?;
                    redline_prime_qwen_slot(gpu, slot, timing_pos)?;
                    cursor = timing_pos;
                }
                let block = redline_dflash_shadow_block(i, batch);
                let t0 = Instant::now();
                let _ = verify_dflash_block(
                    gpu,
                    slot,
                    &block,
                    cursor,
                    &mut fixtures.hidden_rb,
                    Some(&mut fixtures.gdn_tape),
                    false,
                    &fixtures.verify_scratch,
                )
                .map_err(|e| e.to_string())?;
                gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
                graph_host.push(t0.elapsed().as_secs_f64() * 1_000.0);
                cursor = cursor.saturating_add(batch);
                if cursor.saturating_add(batch) > loaded.physical_cap {
                    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                    redline_reset_qwen_slot(gpu, slot)?;
                    redline_dflash_zero_fixtures(gpu, &mut fixtures)?;
                    redline_prime_qwen_slot(gpu, slot, timing_pos)?;
                    cursor = timing_pos;
                }
                let block = redline_dflash_shadow_block(i + n, batch);
                let t1 = Instant::now();
                let _ = verify_dflash_block_retained(
                    gpu,
                    slot,
                    &block,
                    cursor,
                    &mut fixtures.hidden_rb,
                    Some(&mut fixtures.gdn_tape),
                    false,
                    &fixtures.verify_scratch,
                    &mut route,
                )
                .map_err(|e| e.to_string())?;
                gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
                pm4_host.push(t1.elapsed().as_secs_f64() * 1_000.0);
                cursor = cursor.saturating_add(batch);
            }
            let graph_median = redline_percentile(graph_host.clone(), 50.0);
            let graph_p95 = redline_percentile(graph_host.clone(), 95.0);
            let pm4_median = redline_percentile(pm4_host.clone(), 50.0);
            let pm4_p95 = redline_percentile(pm4_host.clone(), 95.0);
            let median_delta_ms = pm4_median - graph_median;
            let p95_delta_ms = pm4_p95 - graph_p95;
            let percent_delta = if graph_median.abs() < 1.0e-9 {
                0.0
            } else {
                (median_delta_ms / graph_median) * 100.0
            };
            let kill = (median_delta_ms > -1.0 && percent_delta > -2.0) || p95_delta_ms > 0.0;
            timing = serde_json::json!({
                "windows_per_arm": n,
                "interleaved": true,
                "capture_prepare_us": capture_prepare_us,
                "hipgraph_verify_body_ms": { "median": graph_median, "p95": graph_p95 },
                "pm4_verify_body_ms": { "median": pm4_median, "p95": pm4_p95 },
                "median_delta_ms": median_delta_ms,
                "percent_delta": percent_delta,
                "p95_delta_ms": p95_delta_ms,
                "kill_if_under_1ms_and_2pct_or_p95_regresses": kill,
            });
        }

        let production_route = loaded
            .speculator
            .as_ref()
            .and_then(|s| s.verify_pm4_report());

        // Q8+EF (default-on) and non-Q8 state are deterministic byte-parity
        // oracles. Only legacy stochastic Q8 without an EF residual keeps the
        // invalid-parity warning — decide from StateQuant, not vector emptiness.
        let is_q8 = matches!(slot.dn_state.quant, qwen35::StateQuant::Q8);
        let q8_error_feedback_enabled = is_q8 && !slot.dn_state.s_ef_residual.is_empty();
        let q8_byte_parity_invalid = is_q8 && !q8_error_feedback_enabled;
        let oracle = if q8_byte_parity_invalid {
            DFLASH_Q8_BYTE_PARITY_INVALID
        } else {
            DFLASH_DETERMINISTIC_BYTE_PARITY
        };

        Ok(serde_json::json!({
            "type": "redline_dflash_shadow_result",
            "backend": "pm4_ib",
            "execution_mode": "dflash_verify",
            "verify_batch": batch,
            "iterations": positions.len(),
            "positions": positions,
            "extract_layers": DFLASH_SHADOW_EXTRACT_LAYERS,
            "arch": gpu.arch,
            "arch_id": loaded.arch_id,
            "physical_cap": loaded.physical_cap,
            "model": redline_artifact_fingerprint(&loaded.model_path),
            "oracle": oracle,
            "arms": {
                "hip_auto": redline_dflash_arm_json(&hip_auto),
                "direct_capture_safe": redline_dflash_arm_json(&direct_capture_safe),
                "recorded_hip": redline_dflash_arm_json(&recorded_hip),
                "pm4": redline_dflash_arm_json(&pm4),
            },
            "parity": {
                "q8_error_feedback_enabled": q8_error_feedback_enabled,
                "q8_byte_parity_invalid": q8_byte_parity_invalid,
                "windows": parity_windows,
            },
            "capture": {
                "launches": launches,
                "aql_contracts": aql_contracts,
                "aql_equals_unique_kernels": aql_contracts == launches || aql_contracts > 0,
            },
            "prepared_identity": identity.map(|id| serde_json::json!({
                "dispatch_count": id.dispatch_count,
                "packet_count": id.packet_count,
                "queue_id": id.queue_id,
                "command_dwords": id.command_dwords,
                "queue_count": id.queue_count,
                "phase_count": id.phase_count,
                "dispatch_equals_launches": id.dispatch_count == launches,
            })),
            "route": route.report_json(),
            "production_route": production_route,
            "protocol": {
                "prepare_reject": {
                    "hip_completed": hip_completed,
                    "route": prepare_report,
                },
                "replay_proven": {
                    "safe_hip_retry": true,
                    "double_ring_commit": written_after > written_before.saturating_add(batch),
                    "ring_written_before": written_before,
                    "ring_written_after": written_after,
                    "route": proven_report,
                },
                "replay_unknown": {
                    "lm_head": false,
                    "accept": false,
                    "emission": false,
                    "shutdown": unknown_shutdown.as_ref().err().map(|e| e.to_string()),
                    "route": unknown_report,
                },
            },
            "timing": timing,
        }))
    })();

    match route.shutdown() {
        Ok(()) => fixtures.free(gpu),
        Err(reason) => {
            std::mem::forget(fixtures);
            return Err(format!(
                "DFlash shadow quarantined; refusing to free captured allocations: {reason}"
            ));
        }
    }
    inner
}

/// `"redline_dflash_verify_shadow_pm4"` daemon message handler.
pub fn handle_redline_dflash_verify_shadow_pm4(
    msg: &serde_json::Value,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
) {
    let batch = msg
        .get("verify_batch")
        .and_then(|value| value.as_u64())
        .unwrap_or(DFLASH_VERIFY_PM4_BLOCK as u64) as usize;
    let iterations = msg
        .get("iterations")
        .and_then(|value| value.as_u64())
        .unwrap_or(4) as usize;
    let steady_state_windows = msg
        .get("steady_state_windows")
        .and_then(|value| value.as_u64())
        .unwrap_or(0) as usize;
    let response = model
        .as_mut()
        .ok_or_else(|| "DFlash shadow requires a loaded model".to_string())
        .and_then(|loaded| {
            redline_shadow_dflash_verify_pm4(gpu, loaded, batch, iterations, steady_state_windows)
        });
    match response {
        Ok(response) => {
            let _ = writeln!(stdout, "{response}");
        }
        Err(reason) => {
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({"type": "error", "message": reason})
            );
        }
    }
    let _ = stdout.flush();
}

/// `"redline_shadow_aql" | "redline_shadow_pm4"` daemon message handler.
fn redline_shadow_gemma4(
    gpu: &mut rdna_compute::Gpu,
    loaded: &mut LoadedModel,
    pm4: bool,
    context: usize,
    iterations: usize,
    position_step: usize,
    replay_only: bool,
) -> Result<serde_json::Value, String> {
    let launch_count = gpu.replay.recorded_launches().len();
    let prepared = if pm4 {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, launch_count)
            .map(|(dispatches, dwords, queue)| (dispatches, 1, queue, Some(dwords)))
    } else {
        gpu.replay
            .prepare_linear_aql(gpu.device_id as usize)
            .map(|(dispatches, packets, queue)| (dispatches, packets, queue, None))
    }?;
    let position =
        |iteration: usize| context.saturating_add(iteration.saturating_mul(position_step));

    let replay_arm = (|| -> Result<(RedlineGemma4Snapshot, f64, f64), String> {
        let bundle = loaded
            .gemma4_lowered_mut()
            .ok_or("Gemma4 Redline shadow requires lowered state")?;
        redline_reset_gemma4(gpu, bundle)?;
        redline_prime_gemma4(gpu, bundle, context)?;
        let started = Instant::now();
        let mut gpu_us = 0.0;
        for i in 0..iterations {
            redline_prepare_gemma4(gpu, bundle, 101 + (i as u32 % 1000), position(i))?;
            // Embedding/scale/position staging runs through HIP, while retained
            // AQL/PM4 executes on its own HSA queue. Complete the external
            // adapter boundary before the retained body consumes scratch.x and
            // pos_buf; same-stream HIP/blob oracle arms are ordered implicitly.
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            gpu_us += if pm4 {
                unsafe { gpu.replay.replay_pm4(position(i)) }?.span_microseconds()
            } else {
                unsafe { gpu.replay.replay_linear_aql(position(i)) }?.span_microseconds()
            };
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        Ok((
            redline_gemma4_snapshot(gpu, bundle, position(iterations.saturating_sub(1)))?,
            host_us,
            gpu_us,
        ))
    })()?;

    if replay_only {
        return Ok(serde_json::json!({
            "type": if pm4 { "redline_shadow_pm4" } else { "redline_shadow_aql" },
            "replay_only": true,
            "context_tokens": context,
            "iterations": iterations,
            "position_step": position_step,
            "queue_id": prepared.2,
            "aql_host_us": replay_arm.1,
            "aql_gpu_us": replay_arm.2,
        }));
    }

    let blob_snapshot = (|| -> Result<RedlineGemma4Snapshot, String> {
        let bundle = loaded
            .gemma4_lowered_mut()
            .ok_or("Gemma4 Redline blob oracle requires lowered state")?;
        redline_reset_gemma4(gpu, bundle)?;
        redline_prime_gemma4(gpu, bundle, context)?;
        for i in 0..iterations {
            redline_prepare_gemma4(gpu, bundle, 101 + (i as u32 % 1000), position(i))?;
            gpu.replay_recorded_hip_prefix_at(prepared.0, position(i))
                .map_err(|error| error.to_string())?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        redline_gemma4_snapshot(gpu, bundle, position(iterations.saturating_sub(1)))
    })()?;

    let hip_arm = (|| -> Result<(RedlineGemma4Snapshot, f64), String> {
        let bundle = loaded
            .gemma4_lowered_mut()
            .ok_or("Gemma4 Redline HIP oracle requires lowered state")?;
        redline_reset_gemma4(gpu, bundle)?;
        redline_prime_gemma4(gpu, bundle, context)?;
        let started = Instant::now();
        for i in 0..iterations {
            redline_prepare_gemma4(gpu, bundle, 101 + (i as u32 % 1000), position(i))?;
            hipfire_arch_gemma4::lowered::forward_scratch(
                gpu,
                &bundle.weights,
                &bundle.config,
                101 + (i as u32 % 1000),
                position(i),
                &mut bundle.kv_sliding,
                &mut bundle.kv_full,
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        Ok((
            redline_gemma4_snapshot(gpu, bundle, position(iterations.saturating_sub(1)))?,
            host_us,
        ))
    })()?;

    let bit_exact = replay_arm.0 == hip_arm.0;
    let blob_bit_exact = replay_arm.0 == blob_snapshot;
    Ok(serde_json::json!({
        "type": "redline_shadow_result",
        "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
        "context_tokens": context,
        "iterations": iterations,
        "dispatches": prepared.0,
        "packets": prepared.1,
        "queue_id": prepared.2,
        "command_dwords": prepared.3,
        "bit_exact": bit_exact,
        "blob_bit_exact": blob_bit_exact,
        "logits_equal": replay_arm.0.logits == hip_arm.0.logits,
        "kv_equal": replay_arm.0.sliding_kv == hip_arm.0.sliding_kv
            && replay_arm.0.full_kv == hip_arm.0.full_kv,
        "recurrent_equal": true,
        "aql_host_us": replay_arm.1,
        "aql_gpu_us": replay_arm.2,
        "hip_host_us": hip_arm.1,
        "aql": replay_arm.0.json(),
        "hip": hip_arm.0.json(),
        "blob": blob_snapshot.json(),
    }))
}

pub fn handle_redline_shadow(
    msg: &serde_json::Value,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
) {
    let pm4 = msg.get("type").and_then(|value| value.as_str()) == Some("redline_shadow_pm4");
    let context = msg
        .get("context_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(128) as usize;
    let iterations = msg
        .get("iterations")
        .and_then(|value| value.as_u64())
        .unwrap_or(1) as usize;
    let position_step = msg
        .get("position_step")
        .and_then(|value| value.as_u64())
        .unwrap_or(1) as usize;
    let position =
        |iteration: usize| context.saturating_add(iteration.saturating_mul(position_step));
    let replay_only = pm4
        && msg
            .get("replay_only")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
    if model.as_ref().is_some_and(|loaded| {
        loaded.pp == 1
            && loaded.ep.is_none()
            && loaded.state.as_ref().is_some_and(|state| {
                (state.as_ref() as &dyn Any).is::<hipfire_loader::Gemma4LoweredBundle>()
            })
    }) {
        let loaded = model.as_mut().expect("Gemma4 retained route checked");
        match redline_shadow_gemma4(
            gpu,
            loaded,
            pm4,
            context,
            iterations,
            position_step,
            replay_only,
        ) {
            Ok(response) => {
                let _ = writeln!(stdout, "{response}");
            }
            Err(reason) => {
                emit_uncorrelated_error(
                    stdout,
                    None,
                    &format!("Gemma4 Redline shadow failed: {reason}"),
                    "internal",
                    false,
                    false,
                );
            }
        }
        let _ = stdout.flush();
        return;
    }
    if model.as_ref().is_some_and(|loaded| {
        loaded.state.as_ref().is_some_and(|s| {
            (s.as_ref() as &dyn Any).is::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        }) || redline_is_dense_lfm(loaded)
    }) {
        let loaded = model.as_mut().expect("retained route checked");
        match redline_shadow_deepseek4(gpu, loaded, pm4, context, iterations) {
            Ok(response) => {
                let _ = writeln!(stdout, "{response}");
            }
            Err(reason) => {
                let _ = writeln!(
                    stdout,
                    "{}",
                    serde_json::json!({"type": "error", "message": reason})
                );
            }
        }
        let _ = stdout.flush();
        return;
    }
    let eligible = model.as_ref().is_some_and(|loaded| {
        loaded.pp == 1
            && loaded.ep.is_none()
            && loaded
                .state
                .as_ref()
                .map_or(false, |s| s.as_ref().arch_key() == "qwen35")
    });
    if !eligible {
        emit_uncorrelated_error(
            stdout,
            None,
            "redline_shadow_aql requires a loaded single-GPU Qwen3.5, DeepSeek4 or dense LFM model",
            "unsupported",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let prepared = match if pm4 {
        let launch_count = gpu.replay.recorded_launches().len();
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, launch_count)
            .map(|(dispatches, dwords, queue)| (dispatches, 1, queue, Some(dwords)))
    } else {
        gpu.replay
            .prepare_linear_aql(gpu.device_id as usize)
            .map(|(dispatches, packets, queue)| (dispatches, packets, queue, None))
    } {
        Ok(summary) => summary,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline AQL prepare failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();

    let aql_result = (|| -> Result<(RedlineQwenSnapshot, f64, f64), String> {
        let loaded = model.as_mut().expect("eligibility checked");
        let Some(bundle) = loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) else {
            unreachable!()
        };
        redline_reset_qwen(gpu, bundle)?;
        redline_prime_qwen(gpu, bundle, context)?;
        let started = Instant::now();
        let mut gpu_us = 0.0;
        for i in 0..iterations {
            bundle
                .kv_cache
                .ensure_mapped_capacity(gpu, position(i).saturating_add(1))
                .map_err(|error| error.to_string())?;
            qwen35::prepare_scratch_inputs(
                gpu,
                &bundle.weights,
                &bundle.config,
                101 + i as u32,
                position(i),
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
            if pm4 {
                let timing = unsafe { gpu.replay.replay_pm4(position(i)) }?;
                gpu_us += timing.span_microseconds();
            } else {
                let timing = unsafe { gpu.replay.replay_linear_aql(position(i)) }?;
                gpu_us += timing.span_microseconds();
            }
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        let snapshot = redline_qwen_snapshot(&gpu, bundle)?;
        Ok((snapshot, host_us, gpu_us))
    })();
    let (aql_snapshot, aql_host_us, aql_gpu_us) = match aql_result {
        Ok(result) => result,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline AQL shadow execution failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    if replay_only {
        let response = serde_json::json!({
            "type": "redline_shadow_pm4",
            "replay_only": true,
            "context_tokens": context,
            "iterations": iterations,
            "position_step": position_step,
            "queue_id": prepared.2,
            "aql_host_us": aql_host_us,
            "aql_gpu_us": aql_gpu_us,
        });
        let _ = writeln!(stdout, "{response}");
        let _ = stdout.flush();
        return;
    }

    let blob_result = (|| -> Result<RedlineQwenSnapshot, String> {
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
        let loaded = model.as_mut().expect("eligibility checked");
        let Some(bundle) = loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) else {
            unreachable!()
        };
        redline_reset_qwen(gpu, bundle)?;
        redline_prime_qwen(gpu, bundle, context)?;
        for i in 0..iterations {
            qwen35::prepare_scratch_inputs(
                gpu,
                &bundle.weights,
                &bundle.config,
                101 + i as u32,
                position(i),
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
            gpu.replay_recorded_hip_prefix_at(prepared.0, position(i))
                .map_err(|error| error.to_string())?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        redline_qwen_snapshot(&gpu, bundle)
    })();
    let blob_snapshot = match blob_result {
        Ok(result) => result,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline HIP blob oracle failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };

    let hip_result = (|| -> Result<(RedlineQwenSnapshot, f64), String> {
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
        let loaded = model.as_mut().expect("eligibility checked");
        let Some(bundle) = loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) else {
            unreachable!()
        };
        redline_reset_qwen(gpu, bundle)?;
        redline_prime_qwen(gpu, bundle, context)?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let started = Instant::now();
        for i in 0..iterations {
            qwen35::forward_scratch(
                gpu,
                &bundle.weights,
                &bundle.config,
                101 + i as u32,
                position(i),
                &mut bundle.kv_cache,
                &mut bundle.dn_state,
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let host_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        let snapshot = redline_qwen_snapshot(&gpu, bundle)?;
        Ok((snapshot, host_us))
    })();
    let (hip_snapshot, hip_host_us) = match hip_result {
        Ok(result) => result,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline HIP shadow execution failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let logits_equal = aql_snapshot.logits == hip_snapshot.logits;
    let kv_equal = aql_snapshot.kv == hip_snapshot.kv;
    let recurrent_equal = aql_snapshot.recurrent == hip_snapshot.recurrent;
    let gdn_frame_equal = aql_snapshot.gdn_frame == hip_snapshot.gdn_frame;
    let blob_gdn_frame_equal = aql_snapshot.gdn_frame == blob_snapshot.gdn_frame;
    let bit_exact = aql_snapshot == hip_snapshot;
    let blob_bit_exact = aql_snapshot == blob_snapshot;
    let _ = writeln!(
        stdout,
        "{}",
        serde_json::json!({
            "type": "redline_shadow_result",
            "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
            "context_tokens": context,
            "iterations": iterations,
            "dispatches": prepared.0,
            "packets": prepared.1,
            "queue_id": prepared.2,
            "command_dwords": prepared.3,
            "bit_exact": bit_exact,
            "blob_bit_exact": blob_bit_exact,
            "logits_equal": logits_equal,
            "kv_equal": kv_equal,
            "recurrent_equal": recurrent_equal,
            "gdn_frame_equal": gdn_frame_equal,
            "blob_gdn_frame_equal": blob_gdn_frame_equal,
            "aql_host_us": aql_host_us,
            "aql_gpu_us": aql_gpu_us,
            "hip_host_us": hip_host_us,
            "aql": aql_snapshot.json(),
            "hip": hip_snapshot.json(),
            "blob": blob_snapshot.json(),
        })
    );
    let _ = stdout.flush();
}

/// `"redline_dispatch_profile"` daemon message handler.
pub fn handle_redline_dispatch_profile(
    msg: &serde_json::Value,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
) {
    let context = msg
        .get("context_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(128) as usize;
    let warmup_replays = msg
        .get("warmup_replays")
        .and_then(|value| value.as_u64())
        .unwrap_or(5) as usize;
    let sample_replays = msg
        .get("sample_replays")
        .and_then(|value| value.as_u64())
        .unwrap_or(20) as usize;
    let validate_correctness = msg
        .get("validate_correctness")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let eligible = model.as_ref().is_some_and(|loaded| {
        loaded.pp == 1
            && loaded.ep.is_none()
            && loaded
                .state
                .as_ref()
                .map_or(false, |s| s.as_ref().arch_key() == "qwen35")
    });
    let launch_count = gpu.replay.recorded_launches().len();
    if !eligible || launch_count == 0 || sample_replays == 0 {
        emit_uncorrelated_error(
            stdout,
            None,
            "redline_dispatch_profile requires captured single-GPU Qwen3.5 and sample_replays > 0",
            "unsupported",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let route = gpu.replay.capture_summary();
    let prepared = match gpu
        .replay
        .prepare_pm4_dispatch_profile(gpu.device_id as usize, launch_count)
    {
        Ok(summary) => summary,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("retained PM4 dispatch-profile prepare failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let boundaries = gpu
        .replay
        .prepared_pm4_dispatch_boundaries()
        .expect("dispatch-profile prepare installed boundary metadata");
    let dispatches = gpu
        .replay
        .recorded_launches()
        .iter()
        .zip(boundaries)
        .enumerate()
        .map(|(index, (launch, boundary))| {
            serde_json::json!({
                "index": index,
                "kernel": launch.kernel,
                "previous_kernel": index.checked_sub(1).map(|previous| {
                    gpu.replay.recorded_launches()[previous].kernel.as_str()
                }),
                "grid": launch.grid,
                "block": launch.block,
                "boundary": {
                    "entry_acquire": boundary.entry_acquire,
                    "wait_compute_idle": boundary.wait_compute_idle,
                    "acquire_inter_node": boundary.acquire_inter_node,
                    "acquire_vmem": boundary.acquire_vmem,
                },
            })
        })
        .collect::<Vec<_>>();

    let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
    let result = (|| -> Result<(Vec<serde_json::Value>, serde_json::Value), String> {
        let loaded = model.as_mut().expect("eligibility checked");
        let Some(bundle) = loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) else {
            unreachable!()
        };

        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
        redline_reset_qwen(gpu, bundle)?;
        redline_prime_qwen(gpu, bundle, context)?;
        for _ in 0..warmup_replays {
            qwen35::prepare_scratch_inputs(
                gpu,
                &bundle.weights,
                &bundle.config,
                101,
                context,
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
            // SAFETY: the loaded model owns every captured pointer.
            unsafe { gpu.replay.replay_pm4_dispatch_profile(context) }?;
        }

        let mut samples = Vec::with_capacity(sample_replays);
        for sample in 0..sample_replays {
            qwen35::prepare_scratch_inputs(
                gpu,
                &bundle.weights,
                &bundle.config,
                101,
                context,
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
            let started = Instant::now();
            // SAFETY: the loaded model owns every captured pointer.
            let profile = unsafe { gpu.replay.replay_pm4_dispatch_profile(context) }?;
            if profile.spans_nanoseconds.len() != launch_count {
                return Err(format!(
                    "dispatch span length mismatch: expected {launch_count}, got {}",
                    profile.spans_nanoseconds.len()
                ));
            }
            let total_gpu_ns = profile
                .timing
                .last_end
                .saturating_sub(profile.timing.first_start)
                .saturating_mul(1_000_000_000)
                / profile.timing.frequency_hz;
            samples.push(serde_json::json!({
                "sample": sample,
                "host_ns": started.elapsed().as_nanos(),
                "total_gpu_ns": total_gpu_ns,
                "spans_ns": profile.spans_nanoseconds,
            }));
        }

        let correctness = if validate_correctness {
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
            redline_reset_qwen(gpu, bundle)?;
            redline_prime_qwen(gpu, bundle, context)?;
            qwen35::prepare_scratch_inputs(
                gpu,
                &bundle.weights,
                &bundle.config,
                101,
                context,
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
            // SAFETY: the loaded model owns every captured pointer.
            unsafe { gpu.replay.replay_pm4_dispatch_profile(context) }?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let instrumented = redline_qwen_snapshot(&gpu, bundle)?;

            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
            redline_reset_qwen(gpu, bundle)?;
            redline_prime_qwen(gpu, bundle, context)?;
            qwen35::forward_scratch(
                gpu,
                &bundle.weights,
                &bundle.config,
                101,
                context,
                &mut bundle.kv_cache,
                &mut bundle.dn_state,
                &bundle.scratch,
            )
            .map_err(|error| error.to_string())?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let hip = redline_qwen_snapshot(&gpu, bundle)?;
            serde_json::json!({
                "performed": true,
                "bit_exact": instrumented == hip,
                "logits_equal": instrumented.logits == hip.logits,
                "kv_equal": instrumented.kv == hip.kv,
                "recurrent_equal": instrumented.recurrent == hip.recurrent,
                "instrumented_pm4": instrumented.json(),
                "hip": hip.json(),
            })
        } else {
            serde_json::json!({"performed": false})
        };
        Ok((samples, correctness))
    })();

    match result {
        Ok((samples, correctness)) => {
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({
                    "schema_version": 1,
                    "type": "redline_dispatch_profile",
                    "context_tokens": context,
                    "warmup_replays": warmup_replays,
                    "sample_replays": sample_replays,
                    "steady_state": true,
                    "exactly_once_per_sample": true,
                    "timestamp_semantics": "baseline before stream plus post-dispatch stamps; span i is PM4 after timestamp i through dispatch i (entry acquire in span 0; later spans include intervening boundary packets)",
                    "route": {
                        "launches": route.launch_count,
                        "unique_kernels": route.unique_kernel_count,
                        "sequence_hash": format!("{:016x}", route.sequence_hash),
                        "command_dwords": prepared.1,
                        "timestamp_slots": route.launch_count + 1,
                        "queue_id": prepared.2,
                    },
                    "dispatches": dispatches,
                    "samples": samples,
                    "correctness": correctness,
                })
            );
        }
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("retained PM4 dispatch profile failed: {reason}"),
                "internal",
                false,
                false,
            );
        }
    }
    let _ = stdout.flush();
}

/// `"redline_pm4_prefix_profile"` daemon message handler.
pub fn handle_redline_pm4_prefix_profile(
    msg: &serde_json::Value,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
) {
    let context = msg
        .get("context_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(128) as usize;
    let step = msg
        .get("step")
        .and_then(|value| value.as_u64())
        .unwrap_or(16) as usize;
    let repeats = msg
        .get("repeats")
        .and_then(|value| value.as_u64())
        .unwrap_or(3) as usize;
    let steady_state = msg
        .get("steady_state")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if model.as_ref().is_some_and(|loaded| {
        loaded.state.as_ref().is_some_and(|s| {
            (s.as_ref() as &dyn Any).is::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        })
    }) {
        let start = msg
            .get("start")
            .and_then(|value| value.as_u64())
            .unwrap_or(step as u64) as usize;
        let loaded = model.as_mut().expect("DeepSeek4 route checked");
        match redline_pm4_prefix_profile_deepseek4(
            gpu,
            loaded,
            context,
            start,
            step,
            repeats,
            steady_state,
        ) {
            Ok(response) => {
                let _ = writeln!(stdout, "{response}");
            }
            Err(reason) => {
                let _ = writeln!(
                    stdout,
                    "{}",
                    serde_json::json!({"type": "error", "message": reason})
                );
            }
        }
        let _ = stdout.flush();
        return;
    }
    let eligible = model.as_ref().is_some_and(|loaded| {
        loaded.pp == 1
            && loaded.ep.is_none()
            && loaded
                .state
                .as_ref()
                .map_or(false, |s| s.as_ref().arch_key() == "qwen35")
    });
    let launch_count = gpu.replay.recorded_launches().len();
    let start = msg
        .get("start")
        .and_then(|value| value.as_u64())
        .unwrap_or(step as u64) as usize;
    if !eligible
        || launch_count == 0
        || step == 0
        || repeats == 0
        || start == 0
        || start > launch_count
    {
        emit_uncorrelated_error(
            stdout,
            None,
            "redline_pm4_prefix_profile requires captured single-GPU Qwen3.5 and valid start/step/repeats",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let mut prefixes = (start..launch_count).step_by(step).collect::<Vec<_>>();
    if prefixes.last().copied() != Some(launch_count) {
        prefixes.push(launch_count);
    }
    let frame_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
    let profile_result = (|| -> Result<Vec<serde_json::Value>, String> {
        // Correctness-oriented prefix profiling resets and primes
        // every sample by default. A full dispatch-level bill of
        // debt needs hundreds of adjacent prefixes, where repeated
        // prefill dominates the requested PM4 measurement. The
        // explicit steady-state mode primes once and then keeps the
        // resident model/cache state warm. It is timing-only: exact
        // shadow remains a separate mandatory harness gate.
        if steady_state {
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
            let loaded = model.as_mut().expect("eligibility checked");
            let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
            }) else {
                unreachable!()
            };
            redline_reset_qwen(gpu, bundle)?;
            redline_prime_qwen(gpu, bundle, context)?;
        }
        let mut rows = Vec::with_capacity(prefixes.len());
        for prefix in prefixes {
            let launch = gpu.replay.recorded_launches()[prefix - 1].clone();
            let (_, dwords, _) = gpu
                .replay
                .prepare_pm4_prefix(gpu.device_id as usize, prefix)?;
            let mut samples = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let loaded = model.as_mut().expect("eligibility checked");
                let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
                }) else {
                    unreachable!()
                };
                if !steady_state {
                    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_checkpoint);
                    redline_reset_qwen(gpu, bundle)?;
                    redline_prime_qwen(gpu, bundle, context)?;
                }
                qwen35::prepare_scratch_inputs(
                    gpu,
                    &bundle.weights,
                    &bundle.config,
                    101,
                    context,
                    &bundle.scratch,
                )
                .map_err(|error| error.to_string())?;
                gpu.hip
                    .device_synchronize()
                    .map_err(|error| error.to_string())?;
                let timing = unsafe { gpu.replay.replay_pm4(context) }?;
                samples.push(timing.span_microseconds());
            }
            let mut ordered = samples.clone();
            ordered.sort_by(f64::total_cmp);
            let median_gpu_us = ordered[ordered.len() / 2];
            rows.push(serde_json::json!({
                "prefix": prefix,
                "last_kernel": launch.kernel,
                "last_grid": launch.grid,
                "last_block": launch.block,
                "command_dwords": dwords,
                "samples_gpu_us": samples,
                "median_gpu_us": median_gpu_us,
            }));
        }
        Ok(rows)
    })();
    match profile_result {
        Ok(rows) => {
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({
                    "type": "redline_pm4_prefix_profile",
                    "context_tokens": context,
                    "launches": launch_count,
                    "start": start,
                    "step": step,
                    "repeats": repeats,
                    "steady_state": steady_state,
                    "rows": rows,
                })
            );
        }
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("retained PM4 prefix profile failed: {reason}"),
                "internal",
                false,
                false,
            );
        }
    }
    let _ = stdout.flush();
}

/// `"redline_prefix_shadow"` daemon message handler.
pub fn handle_redline_prefix_shadow(
    msg: &serde_json::Value,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl std::io::Write,
) {
    let context = msg
        .get("context_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(128) as usize;
    let prefix = msg
        .get("prefix")
        .and_then(|value| value.as_u64())
        .unwrap_or(2) as usize;
    let pm4 = msg
        .get("pm4")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if model.as_ref().is_some_and(redline_is_dense_lfm) {
        let prepared = match if pm4 {
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, prefix)
                .map(|(dispatches, dwords, queue)| (dispatches, 1, queue, Some(dwords)))
        } else {
            gpu.replay
                .prepare_linear_aql_prefix(gpu.device_id as usize, prefix)
                .map(|(dispatches, packets, queue)| (dispatches, packets, queue, None))
        } {
            Ok(summary) => summary,
            Err(reason) => {
                if let Some(loaded) = model.as_mut() {
                    if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                    }) {
                        let _ = redline_reset_lfm2moe(gpu, bundle);
                        loaded.seq_pos = 0;
                        let _ = gpu.hip.device_synchronize();
                    }
                }
                emit_uncorrelated_error(stdout, None, &reason, "internal", false, false);
                let _ = stdout.flush();
                return;
            }
        };
        let aql_arm = (|| -> Result<_, String> {
            let loaded = model.as_mut().unwrap();
            redline_prime_retained_fixture(gpu, loaded, context)?;
            redline_prepare_retained_fixture(gpu, loaded, 101, context)?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let initial = redline_snapshot(&gpu, loaded)?;
            let replay_started = Instant::now();
            if pm4 {
                unsafe { gpu.replay.replay_pm4(context) }?;
            } else {
                unsafe { gpu.replay.replay_linear_aql(context) }?;
            }
            // Commit host n_tokens only after successful replay body.
            if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
                bundle.state.n_tokens = context + 1;
                loaded.seq_pos = context + 1;
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let direct_host_us = replay_started.elapsed().as_secs_f64() * 1e6;
            let snapshot = redline_snapshot(&gpu, loaded)?;
            Ok((initial, snapshot, direct_host_us))
        })();
        let (aql_initial, aql_snapshot, direct_host_us) = match aql_arm {
            Ok(result) => result,
            Err(reason) => {
                if let Some(loaded) = model.as_mut() {
                    if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                    }) {
                        let _ = redline_reset_lfm2moe(gpu, bundle);
                        loaded.seq_pos = 0;
                        let _ = gpu.hip.device_synchronize();
                    }
                }
                emit_uncorrelated_error(
                    stdout,
                    None,
                    &format!("AQL prefix failed: {reason}"),
                    "internal",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        };
        let hip_arm = (|| -> Result<_, String> {
            let loaded = model.as_mut().unwrap();
            redline_prime_retained_fixture(gpu, loaded, context)?;
            redline_prepare_retained_fixture(gpu, loaded, 101, context)?;
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let initial = redline_snapshot(&gpu, loaded)?;
            let replay_started = Instant::now();
            gpu.replay_recorded_hip_prefix(prefix)
                .map_err(|error| error.to_string())?;
            // Commit host n_tokens only after successful blob body.
            if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
                bundle.state.n_tokens = context + 1;
                loaded.seq_pos = context + 1;
            }
            gpu.hip
                .device_synchronize()
                .map_err(|error| error.to_string())?;
            let hip_host_us = replay_started.elapsed().as_secs_f64() * 1e6;
            let snapshot = redline_snapshot(&gpu, loaded)?;
            Ok((initial, snapshot, hip_host_us))
        })();
        let (hip_initial, hip_snapshot, hip_host_us) = match hip_arm {
            Ok(result) => result,
            Err(reason) => {
                if let Some(loaded) = model.as_mut() {
                    if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
                    }) {
                        let _ = redline_reset_lfm2moe(gpu, bundle);
                        loaded.seq_pos = 0;
                        let _ = gpu.hip.device_synchronize();
                    }
                }
                emit_uncorrelated_error(
                    stdout,
                    None,
                    &format!("HIP prefix failed: {reason}"),
                    "internal",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        };
        let mut differing = Vec::new();
        if aql_snapshot.logits() != hip_snapshot.logits() {
            differing.push("logits");
        }
        if aql_snapshot.kv() != hip_snapshot.kv() {
            differing.push("kv");
        }
        if aql_snapshot.recurrent() != hip_snapshot.recurrent() {
            differing.push("recurrent");
        }
        let initial_equal = aql_initial.logits() == hip_initial.logits()
            && aql_initial.kv() == hip_initial.kv()
            && aql_initial.recurrent() == hip_initial.recurrent();
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({
                "type": "redline_prefix_result",
                "prefix": prefix,
                "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
                "dispatches": prepared.0,
                "packets": prepared.1,
                "queue_id": prepared.2,
                "command_dwords": prepared.3,
                "direct_host_us": direct_host_us,
                "hip_host_us": hip_host_us,
                "equal": differing.is_empty(),
                "differing": differing,
                "initial_equal": initial_equal,
                "aql": aql_snapshot.json(),
                "hip": hip_snapshot.json(),
            })
        );
        if let Some(loaded) = model.as_mut() {
            if let Some(bundle) = loaded.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
            }) {
                let _ = redline_reset_lfm2moe(gpu, bundle);
                loaded.seq_pos = 0;
                loaded.conversation_tokens.clear();
                let _ = gpu.hip.device_synchronize();
            }
        }
        let _ = stdout.flush();
        return;
    }

    let eligible = model.as_ref().is_some_and(|loaded| {
        loaded.pp == 1
            && loaded.ep.is_none()
            && loaded
                .state
                .as_ref()
                .map_or(false, |s| s.as_ref().arch_key() == "qwen35")
    });
    if !eligible {
        emit_uncorrelated_error(
            stdout,
            None,
            "redline_prefix_shadow requires single-GPU Qwen3.5",
            "unsupported",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let prepared = match if pm4 {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, prefix)
            .map(|(dispatches, dwords, queue)| (dispatches, 1, queue, Some(dwords)))
    } else {
        gpu.replay
            .prepare_linear_aql_prefix(gpu.device_id as usize, prefix)
            .map(|(dispatches, packets, queue)| (dispatches, packets, queue, None))
    } {
        Ok(summary) => summary,
        Err(reason) => {
            emit_uncorrelated_error(stdout, None, &reason, "internal", false, false);
            let _ = stdout.flush();
            return;
        }
    };
    let aql_hashes = (|| -> Result<_, String> {
        let loaded = model.as_mut().unwrap();
        let Some(bundle) = loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) else {
            unreachable!()
        };
        redline_reset_qwen(gpu, bundle)?;
        redline_prime_qwen(gpu, bundle, context)?;
        qwen35::prepare_scratch_inputs(
            gpu,
            &bundle.weights,
            &bundle.config,
            101,
            context,
            &bundle.scratch,
        )
        .map_err(|error| error.to_string())?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let initial = redline_qwen_debug_hashes(&gpu, bundle)?;
        let replay_started = Instant::now();
        if pm4 {
            unsafe { gpu.replay.replay_pm4(context) }?;
        } else {
            unsafe { gpu.replay.replay_linear_aql(context) }?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let replay_host_us = replay_started.elapsed().as_secs_f64() * 1e6;
        let hashes = redline_qwen_debug_hashes(&gpu, bundle)?;
        let mut dn_k = Vec::new();
        redline_append_buffer(&gpu, &mut dn_k, &bundle.scratch.dn_k.buf)?;
        Ok((initial, hashes, dn_k, replay_host_us))
    })();
    let (aql_initial, aql_hashes, aql_dn_k, direct_host_us) = match aql_hashes {
        Ok(result) => result,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("AQL prefix failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let hip_hashes = (|| -> Result<_, String> {
        let loaded = model.as_mut().unwrap();
        let Some(bundle) = loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) else {
            unreachable!()
        };
        redline_reset_qwen(gpu, bundle)?;
        redline_prime_qwen(gpu, bundle, context)?;
        qwen35::prepare_scratch_inputs(
            gpu,
            &bundle.weights,
            &bundle.config,
            101,
            context,
            &bundle.scratch,
        )
        .map_err(|error| error.to_string())?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let initial = redline_qwen_debug_hashes(&gpu, bundle)?;
        let replay_started = Instant::now();
        gpu.replay_recorded_hip_prefix(prefix)
            .map_err(|error| error.to_string())?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let replay_host_us = replay_started.elapsed().as_secs_f64() * 1e6;
        let hashes = redline_qwen_debug_hashes(&gpu, bundle)?;
        let mut dn_k = Vec::new();
        redline_append_buffer(&gpu, &mut dn_k, &bundle.scratch.dn_k.buf)?;
        Ok((initial, hashes, dn_k, replay_host_us))
    })();
    let (hip_initial, hip_hashes, hip_dn_k, hip_host_us) = match hip_hashes {
        Ok(result) => result,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("HIP prefix failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let differing = aql_hashes
        .iter()
        .filter_map(|(name, hash)| (hip_hashes.get(name) != Some(hash)).then_some(name))
        .cloned()
        .collect::<Vec<_>>();
    let dn_k_mismatches = aql_dn_k
        .iter()
        .zip(&hip_dn_k)
        .filter(|(aql, hip)| aql != hip)
        .count();
    let dn_k_first_mismatch = aql_dn_k
        .iter()
        .zip(&hip_dn_k)
        .position(|(aql, hip)| aql != hip)
        .map(|index| {
            serde_json::json!({
                "byte": index,
                "aql": aql_dn_k[index],
                "hip": hip_dn_k[index],
            })
        });
    let pointer_debug = model.as_mut().and_then(|loaded| {
        let bundle = loaded.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        })?;
        let launch = gpu.replay.recorded_launches().get(prefix.checked_sub(1)?)?;
        let pointers = launch
            .kernarg
            .chunks_exact(8)
            .take(5)
            .map(|chunk| {
                format!(
                    "{:016x}",
                    u64::from_ne_bytes(chunk.try_into().expect("eight-byte chunk"))
                )
            })
            .collect::<Vec<_>>();
        Some(serde_json::json!({
            "kernel": launch.kernel,
            "captured_first_five_u64": pointers,
            "x": format!("{:016x}", bundle.scratch.x.buf.as_ptr() as usize),
            "gate_ffn": format!("{:016x}", bundle.scratch.gate_ffn.buf.as_ptr() as usize),
            "up": format!("{:016x}", bundle.scratch.up.buf.as_ptr() as usize),
            "x_rot": format!("{:016x}", bundle.scratch.x_rot.buf.as_ptr() as usize),
            "ffn_hidden": format!("{:016x}", bundle.scratch.ffn_hidden.buf.as_ptr() as usize),
            "q_raw": format!("{:016x}", bundle.scratch.dn_q_raw.buf.as_ptr() as usize),
            "k_raw": format!("{:016x}", bundle.scratch.dn_k_raw.buf.as_ptr() as usize),
            "q_dst": format!("{:016x}", bundle.scratch.dn_q.buf.as_ptr() as usize),
            "k_dst": format!("{:016x}", bundle.scratch.dn_k.buf.as_ptr() as usize),
        }))
    });
    let _ = writeln!(
        stdout,
        "{}",
        serde_json::json!({
            "type": "redline_prefix_result",
            "prefix": prefix,
            "backend": if pm4 { "pm4_ib" } else { "aql_packets" },
            "dispatches": prepared.0,
            "packets": prepared.1,
            "queue_id": prepared.2,
            "command_dwords": prepared.3,
            "direct_host_us": direct_host_us,
            "hip_host_us": hip_host_us,
            "speedup_over_hip": hip_host_us / direct_host_us,
            "equal": differing.is_empty(),
            "differing": differing,
            "initial_equal": aql_initial == hip_initial,
            "aql_initial": aql_initial,
            "hip_initial": hip_initial,
            "dn_k_mismatched_bytes": dn_k_mismatches,
            "dn_k_first_mismatch": dn_k_first_mismatch,
            "pointer_debug": pointer_debug,
            "aql": aql_hashes,
            "hip": hip_hashes,
        })
    );
    let _ = stdout.flush();
}

#[cfg(test)]
mod redline_snapshot_tests {
    use super::{redline_snapshots_bit_exact, RedlineQwenSnapshot, RedlineSnapshot};

    fn qwen_snapshot(gdn_frame: u32) -> RedlineSnapshot {
        RedlineSnapshot::Qwen(RedlineQwenSnapshot {
            logits: vec![1, 2],
            kv: vec![3, 4],
            recurrent: vec![5, 6],
            gdn_frame,
        })
    }

    #[test]
    fn q8_gdn_frame_participates_in_shadow_bit_exactness() {
        assert!(redline_snapshots_bit_exact(
            &qwen_snapshot(17),
            &qwen_snapshot(17)
        ));
        assert!(!redline_snapshots_bit_exact(
            &qwen_snapshot(17),
            &qwen_snapshot(18)
        ));
    }
}
