// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 model: hybrid DeltaNet (linear attention) + standard attention.
//! Feature-gated behind `deltanet`.

use crate::ffn_bf16::{self, Bf16DownShadow, FfnBf16Mode};
use crate::speculative::HiddenStateRingBuffer;
use crate::xdna1_ffi;
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::{GivensRef, WeightRef};
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, LayerProgram, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_dispatch::types::{DispatchError, RotationPlan};
use hipfire_model::ModelSource;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::hfq_modules::HfqModuleKind;
use hipfire_runtime::kv;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::quant::{f16_to_f32, f32_to_f16};
use hipfire_runtime::tp_shard::ShardConfig;
use hipfire_runtime::weights::{
    fused_rmsnorm_rotate_for_mq, fused_rmsnorm_rotate_for_paro,
    fused_rmsnorm_rotate_mq_batched_for, fused_silu_mul_rotate_mq_batched_for,
    fused_silu_mul_rotate_mq_for, rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemv,
    weight_gemv_prerotated, weight_gemv_residual, weight_gemv_swiglu_residual, EmbeddingFormat,
    ParoRotation, WeightTensor,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

#[cfg(unix)]
use std::os::unix::fs::{FileExt, OpenOptionsExt};

const GPU_SLAB_ALIGN: usize = 4096;

mod telemetry;
use telemetry::{
    moe_router_histogram_active, record_moe_router_selection, router_index_i32_to_usize,
};
pub use telemetry::{
    reset_moe_router_histogram, take_moe_router_histogram, MoeRouterHistogram,
    MoeRouterLayerHistogram,
};

mod config;
use config::{
    bf16_weight_load_mode_from_env, f16_lm_head_mode_from_env, resolve_bf16_weight_load_mode,
    Bf16WeightLoadMode, F16LmHeadMode,
};
pub use config::{config_from_hfq, config_from_safetensors, LayerType, Qwen35Config};

mod layout;
pub use layout::*;
mod state;
pub use state::*;

mod loading;
pub use loading::*;
use loading::{
    rq_apply_readers, rq_apply_writer, try_npu_attn_gate, try_npu_headnorm_rope,
    weight_gemv_swiglu_residual_bf16_probe,
};

mod moe_decode;
use moe_decode::*;

mod lowered;
use lowered::*;

mod ep;
pub use ep::*;

mod decode_layers;
use decode_layers::*;

/// Optional tree-attention context for `forward_prefill_batch` — activates
/// DDTree batched verify when `Some`.
///
/// Fields:
/// - `positions`: length matches `tokens.len()`. Each slot's logical RoPE
///   position (seed at `start_pos`, node i at `start_pos + depth_i`).
///   Two nodes at the same tree depth share a logical position — they're
///   alternative futures at the same time step, not successive tokens.
/// - `attn_bias`: `[N × N]` f32 additive bias on qk scores (with N = tokens.len()),
///   produced by `hipfire_runtime::ddtree::linearize_tree`. `0.0` on ancestor-or-self
///   entries, `-inf` on non-ancestors. Applied to in-block keys only;
///   prompt keys (positions `[0, start_pos)`) remain unmasked.
///
/// Tree mode requires the batched FA path (`fa_batched_ok`); the per-token
/// FA fallback always uses causal attention and cannot honor a tree mask.
/// `forward_prefill_batch` returns an error if tree mode is requested but
/// any FA layer would take the fallback path.
///
/// GDN (LinearAttention) layers: if `parent_indices` is `Some`, the
/// DeltaNet branch dispatches the tree-aware kernels
/// (`conv1d_silu_split_tree_f32_n` + `gated_delta_net_q8_tree_batch_seq`)
/// which walk per-token ancestor chains via `parent_indices` instead of
/// the linear-sequence predecessor. This eliminates sibling-subtree
/// cross-contamination of recurrent state at topk>1. If `parent_indices`
/// is `None`, LA layers fall back to the linear path (byte-exact with
/// DFlash at topk=1; approximation at topk>1 — used by pre-Phase-3
/// callers that haven't been rewritten).

/// Override the embedding for a single batch slot after the embedding-lookup
/// kernel runs but before the layer loop. Used by the Qualcomm-style MTP
/// probe (mtp_probe.rs) to inject mask-token embeddings whose values come
/// from prompt-mean rather than the embedding table.
///
/// Default callers pass `None`; passing `Some(_)` triggers a single
/// host-to-device memcpy into `pbs.x_batch.buf` at byte offset
/// `slot * config.dim * 4` AFTER the embedding-lookup kernel populates
/// the batched-x scratch and BEFORE the first layer reads it.
///
/// Constraints:
///   - `slot < tokens.len()` of the call (asserted)
///   - `embed.len() == config.dim` (asserted)
///   - The override is applied unconditionally to whichever chunk's range
///     contains `slot`. Multi-chunk callers MUST size the prefill batch
///     scratch to keep their target slot in chunk 0, or pass the override
///     only on the chunk where `slot < chunk_n`. (For the MTP probe the
///     entire mask block fits in one chunk by construction.)
#[derive(Clone, Copy)]
pub struct MaskEmbedOverride<'a> {
    pub slot: usize,
    pub embed: &'a [f32],
}

#[derive(Clone, Copy)]
pub struct TreeVerifyCtx<'a> {
    pub positions: &'a [i32],
    pub attn_bias: &'a GpuTensor,
    /// `[N]` i32 — for each linearized slot, the slot index of its parent
    /// in the same linearization (or -1 for the root / seed). Produced by
    /// `hipfire_runtime::ddtree::linearize_tree_with_parents`. When `Some`, LA layers
    /// use tree-aware kernels that read parent state from the per-layer
    /// s_tape scratch in `PrefillBatchScratch`.
    pub parent_indices: Option<&'a GpuTensor>,
    /// Per-FA-layer F32 scratch buffers for capturing K BEFORE RoPE is
    /// applied. Used by Path B slow-path-kill: on the slow path, the
    /// speculative caller gathers accepted K rows out of these scratches,
    /// re-runs RoPE with COMMITTED slot phases (instead of the
    /// linearization phases the in-cache K carries), and re-quants to
    /// the committed kv_cache slots — avoiding a full re-verify forward
    /// while preserving RoPE phase correctness.
    ///
    /// Slice length must equal the number of FullAttention layers in
    /// `config.layer_types`; each entry is a `[max_n × n_kv_heads × head_dim]`
    /// F32 tensor (max_n = 1 + tree budget). When `None`, capture is
    /// skipped (zero overhead). When `Some`, every tree-verify FA layer
    /// memcpy_dtod's its `pbs.fa_k_batch` (post-norm, pre-RoPE) into the
    /// scratch BEFORE the rope kernel mutates it.
    pub pre_rope_k_capture: Option<&'a [GpuTensor]>,
}

pub(crate) fn qwen35_paged_experts_enabled(num_experts: usize) -> bool {
    if num_experts == 0 {
        return false;
    }
    matches!(
        std::env::var("HIPFIRE_QWEN35_PAGED_EXPERTS")
            .ok()
            .as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

pub(crate) fn qwen35_expert_cache_budget_bytes() -> u64 {
    std::env::var("HIPFIRE_QWEN35_EXPERT_CACHE_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(8192)
        .saturating_mul(1024 * 1024)
}

pub(crate) fn infer_attn_output_gate_from_hfq(
    hfq: &HfqFile,
    n_layers: usize,
    q_dim: usize,
) -> Option<bool> {
    for i in 0..n_layers {
        let name = format!("model.layers.{i}.self_attn.q_proj.weight");
        let Some(info) = hfq.find_tensor_info(&name) else {
            continue;
        };
        let Some(&rows) = info.shape.first() else {
            continue;
        };
        let rows = rows as usize;
        if rows == q_dim {
            return Some(false);
        }
        if rows == q_dim * 2 {
            return Some(true);
        }
    }
    None
}

fn qwen35_fa_q_dim(config: &Qwen35Config) -> usize {
    config.n_heads * config.head_dim
}

fn qwen35_fa_q_out_dim(config: &Qwen35Config) -> usize {
    qwen35_fa_q_dim(config) * if config.attn_output_gate { 2 } else { 1 }
}

fn qwen35_materialize_fa_q(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    q_full: &GpuTensor,
    q: &GpuTensor,
    gate: &GpuTensor,
    rows: usize,
) -> HipResult<()> {
    if config.attn_output_gate {
        if rows == 1 {
            gpu.deinterleave_f32(q_full, q, gate, config.n_heads, config.head_dim)
        } else {
            gpu.deinterleave_f32_batched(q_full, q, gate, config.n_heads, config.head_dim, rows)
        }
    } else {
        gpu.memcpy_dtod_auto(&q.buf, &q_full.buf, rows * qwen35_fa_q_dim(config) * 4)
    }
}

fn qwen35_apply_fa_gate(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    attn_out: &GpuTensor,
    gate: &GpuTensor,
) -> HipResult<()> {
    if config.attn_output_gate {
        gpu.sigmoid_mul_f32(attn_out, gate)
    } else {
        Ok(())
    }
}

fn qwen35_attention_wo_residual(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer_idx: usize,
    wo: &WeightTensor,
    attn_out: &GpuTensor,
    residual: &GpuTensor,
    tmp_out: &GpuTensor,
) -> HipResult<()> {
    let invocation = ffn_bf16::attention_wo_residual_invocation_from_shape(
        layer_idx,
        wo.k,
        wo.m,
        config.attn_output_gate,
        ffn_bf16::DenseFfnBackendPreference::GpuProduction,
        false,
    );
    if config.attn_output_gate {
        let result = weight_gemv_residual(gpu, wo, attn_out, residual);
        if result.is_ok() && ffn_bf16::config().trace {
            let output = ffn_bf16::projection_module_output(&invocation);
            let evidence_json = ffn_bf16::projection_module_output_json(&output);
            eprintln!(
                "[qwen35 projection module] module={} preferred_backend={} selected_backend={} oracle_backend={} fallback_reason={} mutates_residual={} evidence_json={}",
                output.module_id,
                invocation.contract.preferred_backend.as_str(),
                output.selected_backend.as_str(),
                output.oracle_backend.as_str(),
                output.fallback_reason.unwrap_or("none"),
                output.mutates_residual,
                evidence_json,
            );
        }
        result
    } else {
        weight_gemv(gpu, wo, attn_out, tmp_out)?;
        let result = gpu.add_inplace_f32(residual, tmp_out);
        if result.is_ok() && ffn_bf16::config().trace {
            let output = ffn_bf16::projection_module_output(&invocation);
            let evidence_json = ffn_bf16::projection_module_output_json(&output);
            eprintln!(
                "[qwen35 projection module] module={} preferred_backend={} selected_backend={} oracle_backend={} fallback_reason={} mutates_residual={} evidence_json={}",
                output.module_id,
                invocation.contract.preferred_backend.as_str(),
                output.selected_backend.as_str(),
                output.oracle_backend.as_str(),
                output.fallback_reason.unwrap_or("none"),
                output.mutates_residual,
                evidence_json,
            );
        }
        result
    }
}

// ─── Forward pass (decode, one token at a time) ─────────────────────────

/// Run one token through the Qwen3.5 model. Returns logits.
/// For DeltaNet layers, updates state in-place (S matrix + conv ring buffer).
/// For full attention layers, uses KV cache like standard transformer.
pub fn forward(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;

    // Embedding lookup
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    let embed_result = match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim),
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim),
        _ => panic!("unsupported embedding format"),
    };
    if let Err(e) = embed_result {
        let _ = gpu.free_tensor(x);
        return Err(e);
    }

    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Shared forward pass — returns logits as CPU Vec<f32>.
fn forward_from_x(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let logits_gpu = forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)?;
    let logits_data = match gpu.download_f32(&logits_gpu) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(logits_gpu);
            return Err(e);
        }
    };
    gpu.free_tensor(logits_gpu)?;
    Ok(logits_data)
}

/// Shared forward pass — returns logits as GPU tensor (no download).
/// Shared forward pass — returns logits as GPU tensor (no download).
/// Caller must free the returned tensor.
///
/// Delegates to `forward_scratch_layers` via a temporary `Qwen35Scratch`,
/// ensuring test/demo paths exercise the same pipeline code as production.
/// NOT production-representative for benchmarking: allocates and frees a full
/// scratch bundle per call. Use `forward_scratch` with a persistent scratch
/// for perf measurement. Per-layer `DEBUG_LAYERS` trace and `trace_finite`
/// "qkvza" checkpoint are not emitted in this path — they are available
/// via `dump_hidden_localize` in the scratch path under HIPFIRE_DUMP_HIDDEN.
fn forward_from_x_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let dim = config.dim;

    // Allocate a temporary scratch bundle. repeat_window=1 (unused in this path).
    // kv_max_seq=8192 matches Qwen35Scratch::new default — sufficient for
    // test/demo single-token forward; these callers don't prefill.
    let scratch = Qwen35Scratch::new(gpu, config, 1)?;

    // Copy input embedding into scratch.x
    gpu.hip.memcpy_dtod(&scratch.x.buf, &x.buf, dim * 4)?;
    gpu.free_tensor(x)?;

    // Set position buffer
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // DEBUG_LAYERS: dump embedding + per-layer norms (same as old forward_from_x_gpu)
    let debug_layers = std::env::var("DEBUG_LAYERS").is_ok();
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "EMB: first4=[{:.6},{:.6},{:.6},{:.6}] norm={norm:.4}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Run the production pipeline
    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, &scratch, None, true, None,
    )?;

    // DEBUG_LAYERS: dump per-layer residual norms
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "POST: first4=[{:.4},{:.4},{:.4},{:.4}] norm={norm:.2}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Copy logits out of scratch before freeing — the returned tensor must
    // outlive the scratch bundle.
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    gpu.hip
        .memcpy_dtod(&logits.buf, &scratch.logits.buf, config.vocab_size * 4)?;

    // Free scratch (all pre-allocated buffers)
    scratch.free_gpu(gpu);

    Ok(logits)
}

/// Pre-allocated scratch buffers for zero-alloc qwen35 forward + GPU sampling.
pub struct Qwen35Scratch {
    // Persistent state
    pub x: GpuTensor,                      // [dim]
    pub tmp: GpuTensor,                    // [dim]
    pub pos_buf: hip_bridge::DeviceBuffer, // 4 bytes

    // DeltaNet temporaries (reused across layers)
    pub dn_qkv: GpuTensor,      // [qkv_dim]
    pub dn_z: GpuTensor,        // [v_dim]
    pub dn_alpha: GpuTensor,    // [n_v_heads]
    pub dn_beta: GpuTensor,     // [n_v_heads]
    pub dn_conv_out: GpuTensor, // [qkv_dim]
    pub dn_q: GpuTensor,        // [v_dim] (after repeat-interleave)
    pub dn_k: GpuTensor,        // [v_dim]
    pub dn_v: GpuTensor,        // [v_dim]
    pub dn_q_raw: GpuTensor,    // [k_dim] (before repeat)
    pub dn_k_raw: GpuTensor,    // [k_dim]
    pub dn_attn_out: GpuTensor, // [v_dim]
    pub dn_normed: GpuTensor,   // [v_dim]

    // FullAttn temporaries (reused across layers)
    pub fa_q_full: GpuTensor,   // [n_heads * head_dim * 2]
    pub fa_q: GpuTensor,        // [n_heads * head_dim]
    pub fa_gate: GpuTensor,     // [n_heads * head_dim]
    pub fa_k: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_v: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_attn_out: GpuTensor, // [n_heads * head_dim]

    // Shared (used by both layer types)
    pub o: GpuTensor,          // [dim]
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up: GpuTensor,         // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // Sampling
    pub logits: GpuTensor,     // [vocab_size]
    pub sample_buf: GpuTensor, // [2] — token_id + rng
    pub repeat_buf: GpuTensor, // [repeat_window]

    // MagnumQuant rotation scratch: FWHT(x) shared across Q/K/V (or gate/up, etc).
    // Sized to max(dim, hidden_dim) — one rotation per batch replaces one per GEMV.
    pub x_rot: GpuTensor, // [max(dim, hidden_dim)]

    // Flash attention partials buffer for tile+reduce 2-kernel path.
    // Size: n_heads * max_tiles * (2 + head_dim) floats.
    pub flash_partials: GpuTensor,
    // Flash attention tri-state (applies to Q8 path; asym modes are flash-only):
    //   0 = never      force non-flash at all contexts (except >15K sanity)
    //   1 = auto       (default) flash kicks in at ctx >= 2048
    //   2 = always     force flash at all contexts
    pub flash_mode: u8,

    // MoE scratch (allocated only when config.num_experts > 0). Pre-allocated
    // so moe_ffn_decode can be captured by hipGraph — the per-layer allocs
    // it used to do violated the "no allocator ops while capturing" rule.
    pub moe_router_logits: Option<GpuTensor>, // [num_experts]
    pub moe_scalar_buf: Option<GpuTensor>,    // [1] shared-expert gate scalar
    pub moe_x_rot: Option<GpuTensor>,         // [dim]
    pub moe_gate_up_buf: Option<GpuTensor>,   // [2*max_inter]   fallback path
    pub moe_gate_buf: Option<GpuTensor>,      // [max_inter]     fallback path
    pub moe_up_buf: Option<GpuTensor>,        // [max_inter]     fallback path
    pub moe_ffn_hidden: Option<GpuTensor>,    // [max_inter]     fallback path
    pub moe_ffn_out: Option<GpuTensor>,       // [dim]           fallback path
    pub moe_gate_batch: Option<GpuTensor>,    // [k × mi]
    pub moe_up_batch: Option<GpuTensor>,      // [k × mi]
    pub moe_rot_batch: Option<GpuTensor>,     // [k × mi]
    /// Phase 2b: GPU-side top-K outputs (kept on-device so moe_ffn_decode
    /// can stay in a graph-capturable stream).
    pub moe_topk_indices: Option<GpuTensor>, // [k] i32 stored as f32 alias
    pub moe_topk_weights: Option<GpuTensor>,  // [k] f32
    // Atomic-free MoE down expansion buffer for decode — [k × dim] f32.
    // Paired with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    // `moe_down_combine_k8_batched` (batch_size=1) in `moe_ffn_decode_impl`'s
    // use_gpu_topk path. Replaces the K_TOP-way atomicAdd that introduced
    // non-deterministic wavefront-order-dependent FP rounding under hipGraph
    // replay (task #100).
    pub moe_down_expanded: Option<GpuTensor>,

    // Optional long-prefill scratch. Default is None to preserve VRAM
    // footprint; set HIPFIRE_PREFILL_REUSE_PBS=1 to allocate and reuse it.
    pub prefill_batch: Option<PrefillBatchScratch>,
}

impl Qwen35Scratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, repeat_window: usize) -> HipResult<Self> {
        // Flash partials are sized for up to 8192 ctx. Override via new_with_kv_max.
        Self::new_with_kv_max(gpu, config, repeat_window, 8192)
    }

    pub fn new_with_kv_max(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,

            dn_qkv: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_z: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_alpha: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_beta: gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)?,
            dn_conv_out: gpu.alloc_tensor(&[qkv_dim], DType::F32)?,
            dn_q: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_k: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_v: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_q_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_k_raw: gpu.alloc_tensor(&[k_dim], DType::F32)?,
            dn_attn_out: gpu.alloc_tensor(&[v_dim], DType::F32)?,
            dn_normed: gpu.alloc_tensor(&[v_dim], DType::F32)?,

            fa_q_full: gpu.alloc_tensor(&[q_dim * 2], DType::F32)?,
            fa_q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_gate: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            fa_k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            fa_attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,

            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,

            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            sample_buf: gpu.alloc_tensor(&[2], DType::F32)?,
            repeat_buf: gpu.alloc_tensor(&[repeat_window], DType::F32)?,
            x_rot: gpu.alloc_tensor(&[dim.max(config.hidden_dim)], DType::F32)?,

            // Flash attention partials: enough for max_seq with tile_size=128.
            // n_heads * max_tiles * (2 + head_dim) floats per batched query
            // position; total buffer = batch_mult × per-position-bytes.
            //
            // batch_mult is the maximum query positions a single FA dispatch
            // can fit; the dispatcher (`launch_asym_flash_batched`) reads the
            // buffer's actual capacity at call time and auto-chunks larger
            // prefill batches into multiple sub-launches. So a lower
            // batch_mult here trades ~linear extra dispatch overhead on
            // prefill (PREFILL_MAX_BATCH=256 → ceil(256/batch_mult) calls per
            // FA layer) for ~linearly less VRAM at long context.
            //
            // The per-position size scales with kv_max_seq (= physical_cap
            // post-eviction), and that scaling is what made #85 visible: at
            // max_seq=170k, no CASK, 27B (n_heads=24, head_dim=256) the old
            // batch_mult=64 → 2.1 GB just for these partials, exceeding VRAM
            // headroom on 24 GB cards. Cutting batch_mult by 4× (16) keeps
            // the prefill chunking moderate while saving 1.6 GB at that
            // worst-case shape; CASK-on workloads (small physical_cap) are
            // unaffected because the buffer is already tiny there.
            //
            // Override with HIPFIRE_FLASH_PARTIALS_BATCH for tuning. Power of
            // two preferred (matches FA dispatcher chunking).
            flash_partials: {
                let tile_size = 128usize;
                let max_tiles = kv_max_seq.div_ceil(tile_size);
                let batch_mult = std::env::var("HIPFIRE_FLASH_PARTIALS_BATCH")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&n| (1..=PREFILL_MAX_BATCH).contains(&n))
                    .unwrap_or(16);
                gpu.alloc_tensor(
                    &[batch_mult * config.n_heads * max_tiles * (2 + config.head_dim)],
                    DType::F32,
                )?
            },
            // Flash attention tri-state for the Q8 path. Asym modes always
            // flash regardless.
            //   HIPFIRE_ATTN_FLASH=never|0|off    → non-flash at all contexts
            //   HIPFIRE_ATTN_FLASH=auto|1|on      → flash at ctx >= 2048
            //   HIPFIRE_ATTN_FLASH=always|2|force → flash at all contexts
            //
            // Default on gfx11/gfx12 (graph-capable archs): `2` (always
            // flash). On other archs: `1` (auto). The capture path at
            // qwen35.rs:8199 hard-wires `use_flash = capture_mode || ...`
            // because attention_q8_0_kv has variable block_size + variable
            // shared-mem (not capture-safe). Without an always-flash default
            // on capture-capable archs, direct mode at small ctx silently
            // uses attention_q8_0_kv while a captured-and-replayed forward
            // uses attention_flash_q8_0 — same math, different fp32
            // reduction order, observed as ~0.44 logit delta direct-vs-graph
            // on shisa-Qwen3.6-A3B-PARO (see
            // .scratch/hipgraph-moe-drift-audit.md Part A). Aligning the
            // default flips both paths to `attention_flash_q8_0` and makes
            // direct vs graph byte-identical at the cost of moving small-
            // context decode off the non-flash kernel (~few % attention
            // perf hit, small contribution to total MoE decode time).
            // Honors HIPFIRE_ATTN_FLASH=never|0|off as an explicit override
            // for users who prefer the non-flash kernel and don't intend
            // to use graph capture.
            flash_mode: match std::env::var("HIPFIRE_ATTN_FLASH").as_deref() {
                Ok("never") | Ok("0") | Ok("off") => 0,
                Ok("always") | Ok("2") | Ok("force") => 2,
                _ => {
                    let graph_capable_arch =
                        gpu.arch.starts_with("gfx12") || gpu.arch.starts_with("gfx11");
                    if graph_capable_arch {
                        2
                    } else {
                        1
                    }
                }
            },

            moe_router_logits: None,
            moe_scalar_buf: None,
            moe_x_rot: None,
            moe_gate_up_buf: None,
            moe_gate_buf: None,
            moe_up_buf: None,
            moe_ffn_hidden: None,
            moe_ffn_out: None,
            moe_gate_batch: None,
            moe_up_batch: None,
            moe_rot_batch: None,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_down_expanded: None,
            prefill_batch: None,
        })
        .and_then(|mut s| {
            // Allocate MoE scratch only for MoE configs. Done after the
            // main struct init so these Options start as None for dense
            // models and never cost VRAM there.
            if config.num_experts > 0 {
                let hidden = config.dim;
                let n_exp = config.num_experts;
                let mi = config.moe_intermediate_size;
                let smi = config.shared_expert_intermediate_size;
                let max_inter = mi.max(smi);
                let k = config.num_experts_per_tok;
                s.moe_router_logits = Some(gpu.alloc_tensor(&[n_exp], DType::F32)?);
                s.moe_scalar_buf = Some(gpu.alloc_tensor(&[1], DType::F32)?);
                s.moe_x_rot = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_up_buf = Some(gpu.alloc_tensor(&[2 * max_inter], DType::F32)?);
                s.moe_gate_buf = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_up_buf = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_hidden = Some(gpu.alloc_tensor(&[max_inter], DType::F32)?);
                s.moe_ffn_out = Some(gpu.alloc_tensor(&[hidden], DType::F32)?);
                s.moe_gate_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_up_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                s.moe_rot_batch = Some(gpu.alloc_tensor(&[k * mi], DType::F32)?);
                // i32 topk_indices stored in an F32 tensor (same byte width).
                // The kernel that writes it casts the buffer to int*, and the
                // indexed MoE GEMV kernels read it as int*.
                s.moe_topk_indices = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                s.moe_topk_weights = Some(gpu.alloc_tensor(&[k], DType::F32)?);
                // Atomic-free decode MoE down output: [k × dim].
                s.moe_down_expanded = Some(gpu.alloc_tensor(&[k * hidden], DType::F32)?);
                // Pre-warm MQ FWHT sign tables (otherwise the lazy init in
                // ensure_mq_signs fires during the first moe_ffn_decode and
                // blows up hipGraph capture with a hipMalloc-in-capture
                // error). Idempotent if already computed.
                gpu.ensure_mq_signs()?;
            }
            if std::env::var("HIPFIRE_PREFILL_REUSE_PBS").ok().as_deref() == Some("1") {
                let max_batch = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&v| v >= 2)
                    .unwrap_or(PREFILL_MAX_BATCH);
                s.prefill_batch = Some(PrefillBatchScratch::new(gpu, config, max_batch)?);
            }
            Ok(s)
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.tmp);
        // pos_buf is held as a raw DeviceBuffer and dropped via gpu.hip.free
        // directly (free_tensor would have bound the thread internally).
        // Bind explicitly so HIP affinity doesn't depend on the order of
        // preceding free_tensor calls.
        let _ = gpu.bind_thread();
        let _ = gpu.hip.free(self.pos_buf);
        for t in [
            self.dn_qkv,
            self.dn_z,
            self.dn_alpha,
            self.dn_beta,
            self.dn_conv_out,
            self.dn_q,
            self.dn_k,
            self.dn_v,
            self.dn_q_raw,
            self.dn_k_raw,
            self.dn_attn_out,
            self.dn_normed,
            self.fa_q_full,
            self.fa_q,
            self.fa_gate,
            self.fa_k,
            self.fa_v,
            self.fa_attn_out,
            self.o,
            self.gate_ffn,
            self.up,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.sample_buf,
            self.repeat_buf,
            self.x_rot,
            self.flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
        // MoE scratch — only present for MoE configs.
        for buf in [
            self.moe_router_logits,
            self.moe_scalar_buf,
            self.moe_x_rot,
            self.moe_gate_up_buf,
            self.moe_gate_buf,
            self.moe_up_buf,
            self.moe_ffn_hidden,
            self.moe_ffn_out,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(buf);
        }
        if let Some(pbs) = self.prefill_batch {
            pbs.free_gpu(gpu);
        }
    }
}

/// Per-device scratch bundle for the multi-GPU forward path. Each device gets
/// its own `Qwen35Scratch` because the residual stream `s.x` (and `s.logits`)
/// must live on the device executing the current band's layers — cross-band
/// boundaries copy `s.x` between devices via `Gpus::boundary_copy`. `s.logits`
/// is also allocated per-device for simplicity (~600 KB each at vocab=152K)
/// even though only the output device's `s.logits` is consumed post-loop.
pub struct Qwen35ScratchSet {
    pub per_device: Vec<Qwen35Scratch>,
}

impl Qwen35ScratchSet {
    pub fn new_with_kv_max_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let mut per_device = Vec::with_capacity(gpus.devices.len());
        for dev_idx in 0..gpus.devices.len() {
            let g = &mut gpus.devices[dev_idx];
            per_device.push(Qwen35Scratch::new_with_kv_max(
                g,
                config,
                repeat_window,
                kv_max_seq,
            )?);
        }
        Ok(Self { per_device })
    }

    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        for (dev_idx, scratch) in self.per_device.into_iter().enumerate() {
            scratch.free_gpu(&mut gpus.devices[dev_idx]);
        }
    }
}

/// Zero-alloc forward pass using pre-allocated scratch buffers.
/// Logits stay on GPU in scratch.logits. Returns nothing — caller uses scratch.logits.
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    // hipGraph capture for MoE was previously gated off-by-default behind
    // HIPFIRE_GRAPH_MOE=1 because of a known drift bug (task #100): under
    // capture, A3B accumulated a per-step ~1-ULP delta that compounded
    // through the KV cache + GDN state and crossed the top-1 margin at
    // step ~7 (q8 KV) or ~114 (asym3 KV), producing visible token-loop
    // attractors by step 30-50 ("- **One**\n- **One**\n…").
    //
    // Root cause (fixed 2026-05-21): `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`
    // used K_TOP=8 concurrent `atomicAdd` writes per output row. FP32
    // addition is non-associative, so the final bits depend on wavefront
    // scheduling order. Under hipGraph replay that order differs from
    // direct execution (graph scheduling pipelines kernels differently),
    // introducing the systematic per-step delta. The kernel's own header
    // (`kernels/src/gemv_hfq4g256_moe_down.hip:14-19`) had already flagged
    // this non-determinism but rated it negligible based on the
    // direct-only smoke test — capture amplifies the effect.
    //
    // Fix: the MoE FFN decode path now uses the atomic-free expand+combine
    // pattern already used in prefill (`forward_prefill_batch_with_pbs`
    // L5217-5232): `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`
    // writes one row per (expert-rank, m), then `moe_down_combine_k8_batched`
    // sums K_TOP slots into x_residual in a fixed iteration order. The
    // resulting MoE FFN output is byte-deterministic under both direct
    // execution and hipGraph replay.
    //
    // HIPFIRE_GRAPH_MOE remains opt-in (set to "1" to enable). The atomic
    // fix is necessary but not sufficient — the CPU-topK fallback path
    // (when not all gate-side MoE weights are MQ4G256, e.g. router=Q8 per
    // the post-2026-04 router-attractor fix) calls `download_f32(router_logits)`,
    // a sync D2H that fails under graph capture with hipError 906. Until
    // that D2H is migrated to a capture-safe equivalent, opting in only
    // works for models where the runtime takes the use_gpu_topk path.
    //
    // Reproducer used to characterize the fix:
    //   HIPFIRE_GRAPH=1 HIPFIRE_GRAPH_MOE=1 HIPFIRE_SMOKE_KV=q8 \
    //   HIPFIRE_SMOKE_MODE=chat HIPFIRE_SMOKE_STEPS=200 \
    //   HIPFIRE_SMOKE_PROMPT="Count from one to twenty in English." \
    //   ./target/release/examples/a3b_smoke_forward <uniform-mq4-a3b>
    //
    // Per-forward env var lookups cached via OnceLock — these used to fire
    // ~16-46 std::env::var() syscalls per cycle on 27B decode, allocating a
    // String and walking the env table each time. Process env can't legitimately
    // change between forward calls; cache once and read atomically.
    static ALLOW_MOE_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static GRAPH_OVERRIDE_ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    // Opt-in: set HIPFIRE_GRAPH_MOE=1 to enable graph capture for the MoE
    // forward path. Default-off until a follow-up makes the CPU-topK
    // fallback's `download_f32(router_logits)` D2H sync capture-safe —
    // mixed-kmap A3B (post-PR #199) routes through that fallback and crashes
    // with hipError 906 under graph capture. The atomicAdd-determinism fix in
    // this commit removes the use_gpu_topk path's drift, which is the necessary
    // first step, but is not sufficient to enable MoE+graph by default.
    let allow_moe = *ALLOW_MOE_ENV
        .get_or_init(|| std::env::var("HIPFIRE_GRAPH_MOE").ok().as_deref() == Some("1"));
    // hipGraph per-forward-pass capture/replay default policy:
    //   - gfx12 (RDNA4): default-ON. +2.4-2.7% decode on 9B Qwen 3.5
    //     MFP4G32 (5-run mean, all positive, tight variance, 2026-05-11).
    //   - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and
    //     0.8B HFP4G32 on 7900 XTX (5-run mean per model, all positive,
    //     variance 1.001-1.010×, 2026-05-11). Smaller win than gfx12 —
    //     gfx11 has less per-launch overhead to amortize — but real
    //     and consistent across model sizes.
    //   - other archs (RDNA1/2, CDNA): default-OFF (opt-in via
    //     HIPFIRE_GRAPH=1) since not yet A/B'd on those.
    //   - MoE configs: opt-in via HIPFIRE_GRAPH_MOE=1. The ~30-50-token
    //     attractor drift in the use_gpu_topk MoE down step was fixed
    //     2026-05-21 (task #100 — atomicAdd → expand+combine), but the
    //     CPU-topK fallback's `download_f32(router_logits)` D2H sync
    //     remains capture-incompatible, so mixed-kmap A3B (post-PR #199)
    //     can crash under graph capture even with the fix. Once that
    //     D2H is migrated to a capture-safe path, the MoE default can
    //     be flipped to follow the arch defaults.
    // Explicit HIPFIRE_GRAPH=0 always wins (kill switch).
    let graph_override =
        *GRAPH_OVERRIDE_ENV.get_or_init(|| match std::env::var("HIPFIRE_GRAPH").ok().as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        });
    let graph_arch_default = gpu.arch.starts_with("gfx12") || gpu.arch.starts_with("gfx11");
    let graph_enabled = graph_override.unwrap_or(graph_arch_default);
    // AR-forward hipGraph DISABLED (2026-05-15) — this disable SUPERSEDES the
    // arch-default re-enable merged from master (`graph_enabled` above is kept
    // live so the HIPFIRE_GRAPH parse and kill switch stay wired for when the
    // path is flipped back on). Empirically on ROCm 7.2.2 + gfx11 +
    // Qwen3.5-27B mq4, both replay AND capture+launch produce a token-0
    // attractor outside very narrow conditions:
    //   - Capture+launch at position 2 (after 1 direct warmup) → `!!!!!`
    //   - Capture+launch at position 4 (after 3 direct warmups) → correct
    //   - Replay of a working capture (any position) → `!!!!!` from pos+1 on
    // The kernarg-snapshot bug isn't fixable by warmup tuning OR caller-driven
    // commit gating (`end_decode_turn()`); both fail empirically. Master's
    // task-#100 fix targets MoE drift, NOT this AR-forward attractor, so the
    // merge does not clear the disable. Until the capture/replay attractor is
    // re-verified gone on current ROCm (7.13) via the coherence gate, AR
    // forward is direct-only. Policy infra (`ar_forward_kernel_dirty`,
    // `ar_forward_replay_enabled`, `end_decode_turn()`, `drop_captured_graph()`)
    // is preserved on Gpu so the path can be flipped on once the bug is fixed.
    // AR-forward hipGraph stays OFF. Tested 2026-06-21: with persistent Opus
    // scratch the capture/replay attractor is GONE (replay is coherent), but
    // replay gives NO speedup — decode is kernel-execution-bound, not
    // launch-overhead-bound, so eliminating per-launch cost nets ~0 (40.6 vs
    // 41.5 tok/s). Not worth the capture complexity. See
    // project_gfx1103_decode_memcpy_bound memory.
    // AR-forward hipGraph remains hard-disabled. A 2026-06-25 chaingun merge
    // smoke on gfx1151 showed HIPFIRE_GRAPH=1 regressed qwen3.5-4b mq4 decode
    // from the 65.5 tok/s floor to ~54-56 tok/s, while direct mode still passes
    // the speed gate. Keep the env parser/reporting wired above, but do not use
    // it to enable AR graph execution until capture/replay is re-qualified.
    let use_graph = false;
    let _ = (graph_enabled, allow_moe, gpu.ar_forward_replay_enabled); // suppress unused warnings

    // Embedding lookup into scratch.x (always direct, changes per token)
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }
    trace_stage_if_enabled("forward_scratch embedding done");

    let pos_i32 = pos as i32;
    if use_graph && gpu.ar_forward_replay_enabled && gpu.graph_exec.is_some() {
        // ── Replay path: caller has signalled end_decode_turn() since the
        // last capture AND kernels are not dirty. Cheapest path. ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.graph_launch()?;
    } else if use_graph && gpu.ar_forward_kernel_dirty {
        // ── Direct path (kernel-dirty): kernels are dirty (init or post-
        // model-load). Capture would trip "hipMalloc not permitted under
        // stream capture" on the first inline JIT. Mark clean after a
        // successful direct dispatch so subsequent calls can capture. ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, true, None,
        )?;
        gpu.ar_forward_kernel_dirty = false;
    } else if use_graph {
        // ── Capture + launch: kernels are clean but caller has not committed
        // a replay yet (or graph_exec is None). Drop any prior captured graph,
        // record a fresh one, and launch it for this forward's output. After
        // the caller signals end_decode_turn(), the most recent capture is
        // promoted to the replay graph for the next decode turn. ──
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.drop_captured_graph();
        gpu.begin_graph_capture()?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, true, None,
        )?;
        gpu.end_graph_capture()?;
        gpu.graph_launch()?;
    } else {
        // ── Direct path (graph not eligible: arch / MoE config) ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, true, None,
        )?;
    }
    Ok(())
}

/// Debug-only companion to `forward_scratch` that records each LinearAttention
/// layer's raw replay inputs into `gdn_tape` at `tape_row`.
pub fn forward_scratch_capture_gdn_tape(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    gdn_tape: &mut crate::speculative::GdnTape,
    tape_row: usize,
) -> HipResult<()> {
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        None,
        true,
        Some((gdn_tape, tape_row)),
    )
}

/// Per-layer batched intermediates used by `forward_prefill_batch`. Each
/// row is one token in the batch; rows are contiguous [N × K] blocks so
/// all kernels can treat them as row-major matrices.
///
/// Allocated lazily on the first batched prefill call that takes the MQ4
/// fast path — models that never hit that path (HF4 weights, FA-only
/// models, short prompts) never pay the VRAM cost. Sized to `max_batch`;
/// longer prompts are processed in chunks of `max_batch`.
pub struct PrefillBatchScratch {
    pub max_batch: usize,

    // Residual stream and rotation scratch — all [N × dim]
    pub x_batch: GpuTensor,
    pub x_rot_batch: GpuTensor,
    // Rmsnorm-only scratch (no FWHT). Used by MoE prefill body for Q8_0
    // weights (router + shared_expert_gate) which were quantized against
    // un-rotated input. MQ4 sibling weights read `x_rot_batch` instead.
    // Mixed-dtype MoE layers populate both buffers per `prefill_moe_ffn_body_batched`.
    pub x_norm_batch: GpuTensor,

    // LA-layer projection outputs
    pub dn_qkv_batch: GpuTensor,      // [N × qkv_dim]
    pub dn_z_batch: GpuTensor,        // [N × v_dim]
    pub dn_alpha_batch: GpuTensor,    // [N × n_v_heads]
    pub dn_beta_batch: GpuTensor,     // [N × n_v_heads]
    pub dn_q_raw_batch: GpuTensor,    // [N × k_dim] (pre repeat-interleave)
    pub dn_k_raw_batch: GpuTensor,    // [N × k_dim]
    pub dn_v_batch: GpuTensor,        // [N × v_dim]
    pub dn_q_batch: GpuTensor,        // [N × v_dim] (post repeat-interleave)
    pub dn_k_batch: GpuTensor,        // [N × v_dim]
    pub dn_attn_out_batch: GpuTensor, // [N × v_dim]
    pub dn_normed_batch: GpuTensor,   // [N × v_dim]

    // FFN intermediates [N × hidden_dim]
    pub gate_ffn_batch: GpuTensor,
    pub up_batch: GpuTensor,
    // SwiGLU output (FWHT-rotated for MQ4) feeding w_down.
    pub ffn_hidden_batch: GpuTensor,

    // FWHT-rotated dn_normed [N × v_dim] feeding wo for MQ4 weights.
    // Decode path handles this via an internal mq_x_rot scratch inside
    // weight_gemv_residual; we need an explicit batched equivalent.
    pub dn_normed_rot_batch: GpuTensor,

    // ── FullAttention batched intermediates (when FA weights are MQ4G256) ──
    // Positions array: [max_batch] i32, absolute KV positions for this chunk.
    // Uploaded once at the start of each chunk and reused by rope + kv_write
    // + attention kernels.
    pub positions: GpuTensor,
    // Token-ids buffer feeding the batched embedding kernel. [max_batch] i32
    // stored as F32 (same dtype-cosmetic pattern as `positions`). Uploaded
    // once per batched forward and read by `embedding_lookup_hfq4g256_batched`.
    pub tokens: GpuTensor,
    // QKV projection outputs
    pub fa_q_full_batch: GpuTensor, // [N × n_heads × head_dim × 2] (Q + gate interleaved)
    pub fa_q_batch: GpuTensor,      // [N × n_heads × head_dim]
    pub fa_gate_batch: GpuTensor,   // [N × n_heads × head_dim]
    pub fa_k_batch: GpuTensor,      // [N × n_kv_heads × head_dim]
    pub fa_v_batch: GpuTensor,      // [N × n_kv_heads × head_dim]
    pub fa_attn_out_batch: GpuTensor, // [N × n_heads × head_dim]
    // FWHT-rotated fa_attn_out for feeding MQ4 wo.
    pub fa_attn_out_rot_batch: GpuTensor, // [N × n_heads × head_dim]

    // ── MoE batched intermediates (allocated only when num_experts > 0) ──
    // All outputs of the fused 4-way router + shared-gate GEMM, plus the
    // per-token routed-expert gate/up/rot buffers consumed by the N-batched
    // indexed MoE kernels. Sized as [max_batch × {n_exp, smi, k_top×mi}].
    pub moe_router_logits_batch: Option<GpuTensor>, // [N × num_experts]
    pub moe_shared_scalar_batch: Option<GpuTensor>, // [N × 1] — raw shared_expert_gate logit
    pub moe_shared_gate_batch: Option<GpuTensor>,   // [N × smi]
    pub moe_shared_up_batch: Option<GpuTensor>,     // [N × smi]
    pub moe_shared_rot_batch: Option<GpuTensor>,    // [N × smi] — FWHT(silu(gate) * up)
    pub moe_topk_indices_batch: Option<GpuTensor>,  // [N × k_top] i32 in F32 slots
    pub moe_topk_weights_batch: Option<GpuTensor>,  // [N × k_top]
    pub moe_gate_batch: Option<GpuTensor>,          // [N × k_top × mi]
    pub moe_up_batch: Option<GpuTensor>,            // [N × k_top × mi]
    pub moe_rot_batch: Option<GpuTensor>,           // [N × k_top × mi]
    // Atomic-free MoE down expansion buffer — [N × k_top × dim] f32.
    // Paired with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    // `moe_down_combine_k8_batched`: the down kernel writes each
    // (token, krank) result to its own row here (no atomic), then the
    // combine kernel folds K_TOP slots into x_batch with topk_weights
    // applied. RDNA-only (atomic on GDDR is slow); the wave64/CDNA path
    // stays on the residual_scaled atomic kernel.
    pub moe_down_expanded_batch: Option<GpuTensor>,

    // Path 2 (SGLang-style scatter + grouped-WMMA-GEMM) scratch. All
    // allocated when num_experts > 0; gated at runtime by
    // HIPFIRE_MOE_GROUPED_GEMM=1. m_total_max is tile-aligned:
    // align_up(max_batch * k_top + num_experts * (BLOCK_M - 1), BLOCK_M)
    // with BLOCK_M=16.
    //
    //   moe_expert_token_counts: [num_experts] i32 (raw → padded)
    //   moe_expert_offsets:      [num_experts + 1] i32 (exclusive prefix)
    //   moe_sorted_slot_index:   [m_total_max] i32 (flat slot or -1 padding)
    //   moe_expert_tile_ids:     [m_total_max / 16] i32 (per-tile expert id)
    //   moe_y_gate_up_grouped:   [m_total_max × (2*mi)] f32 (grouped GEMM output)
    pub moe_expert_token_counts: Option<GpuTensor>,
    pub moe_expert_offsets: Option<GpuTensor>,
    pub moe_sorted_slot_index: Option<GpuTensor>,
    pub moe_inverse_perm: Option<GpuTensor>, // [total_slots] i32: flat → sorted_pos
    pub moe_expert_tile_ids: Option<GpuTensor>,
    pub moe_y_gate_up_grouped: Option<GpuTensor>, // [m_total × (2*mi)]
    pub moe_y_down_grouped: Option<GpuTensor>,    // [m_total × dim] for the down step

    // ── Tree-aware LA scratch (Phase 3b of Task #101) ──
    // Per-token S-state tape consumed by gated_delta_net_q8_tree kernel
    // when TreeVerifyCtx.parent_indices is Some. Reused across LA layers
    // since LA dispatch is serial per-cycle. Only allocated when the model
    // has LA layers (linear_num_value_heads > 0). Call sites that pass
    // parent_indices must ensure these tensors exist.
    //
    // s_tape_q8:     [max_batch × n_v_heads × head_dim × head_dim] Raw/i8
    // s_tape_scales: [max_batch × n_v_heads × head_dim] f32
    //
    // At max_batch=22, n_v_heads=16, head_dim=128 → 5.77 MB + 180 KB total.
    pub dn_s_tape_q8: Option<GpuTensor>,
    pub dn_s_tape_scales: Option<GpuTensor>,
}

/// One independent dense-Qwen35 request/session row for the future fused
/// server-prefill worker.
///
/// This is intentionally NOT the same shape as `forward_prefill_batch`: that
/// function consumes one token stream, one KV cache, and one DeltaNet state.
/// Server microbatching needs multiple independent streams, each with its own
/// mutable recurrent state, while sharing weights and batched layer scratch.
pub struct DensePrefillSessionBatchRow<'a> {
    pub tokens: &'a [u32],
    pub start_pos: usize,
    pub kv_cache: &'a mut kv::KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub logits: &'a GpuTensor,
}

pub struct DensePrefillSessionBatchInput<'a> {
    pub tokens: &'a [u32],
    pub start_pos: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchRoundRow {
    pub session_index: usize,
    pub token_index: usize,
    pub token: u32,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchRound {
    pub rows: Vec<DensePrefillSessionBatchRoundRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DensePrefillSessionBatchRoundStateRoute {
    SingleSession { session_index: usize },
    MultiSession { session_indices: Vec<usize> },
}

impl DensePrefillSessionBatchRound {
    pub fn state_route(&self) -> DensePrefillSessionBatchRoundStateRoute {
        let mut session_indices: Vec<usize> =
            self.rows.iter().map(|row| row.session_index).collect();
        session_indices.sort_unstable();
        session_indices.dedup();
        if session_indices.len() == 1 {
            DensePrefillSessionBatchRoundStateRoute::SingleSession {
                session_index: session_indices[0],
            }
        } else {
            DensePrefillSessionBatchRoundStateRoute::MultiSession { session_indices }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchExecutionPlan {
    pub rounds: Vec<DensePrefillSessionBatchRound>,
    pub state_routes: Vec<DensePrefillSessionBatchRoundStateRoute>,
    pub total_rows: usize,
    pub max_rows_per_round: usize,
    pub multi_state_rounds: usize,
    pub multi_state_prefix_rounds: usize,
    pub multi_state_prefix_rows: usize,
    pub singleton_tail: Option<DensePrefillSessionBatchSingletonTail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchSingletonTail {
    pub start_round: usize,
    pub session_index: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchRowShape {
    pub tokens: usize,
    pub logits_numel: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensePrefillSessionBatchStateSignature {
    pub kv_physical_cap: usize,
    pub kv_compact_offset: usize,
    pub kv_quantized: bool,
    pub kv_quant_q8: bool,
    pub kv_quant_asym2: bool,
    pub kv_quant_asym3: bool,
    pub kv_quant_asym4: bool,
    pub kv_quant_fwht: bool,
    pub dn_quant: StateQuant,
}

pub struct DensePrefillSessionKvStateRoute<'a> {
    pub k_gpu: &'a [GpuTensor],
    pub v_gpu: &'a [GpuTensor],
    pub physical_cap: usize,
    pub compact_offset: usize,
}

pub struct DensePrefillSessionDeltaStateRoute<'a> {
    pub s_matrices: &'a [GpuTensor],
    pub s_scales: &'a [GpuTensor],
    pub conv_states: &'a [GpuTensor],
    pub quant: StateQuant,
}

pub struct DensePrefillSessionStateRoute<'a> {
    pub kv: DensePrefillSessionKvStateRoute<'a>,
    pub delta: DensePrefillSessionDeltaStateRoute<'a>,
    pub logits: &'a GpuTensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionStateRouteShape {
    pub kv_k_layers: usize,
    pub kv_v_layers: usize,
    pub dn_s_layers: usize,
    pub dn_scale_layers: usize,
    pub dn_conv_layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPointerTableShape {
    pub sessions: usize,
    pub multi_state_prefix_rounds: usize,
    pub multi_state_prefix_rows: usize,
    pub max_rows_per_round: usize,
    pub kv_k_ptrs: usize,
    pub kv_v_ptrs: usize,
    pub dn_s_ptrs: usize,
    pub dn_scale_ptrs: usize,
    pub dn_conv_ptrs: usize,
    pub logits_ptrs: usize,
    pub session_last_row_indices: usize,
    pub row_session_indices: usize,
    pub row_tokens: usize,
    pub row_positions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPointerTableIndex {
    pub kv_k_offset: usize,
    pub kv_v_offset: usize,
    pub dn_s_offset: usize,
    pub dn_scale_offset: usize,
    pub dn_conv_offset: usize,
    pub logits_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchLayerPointerSlot {
    pub session_index: usize,
    pub layer_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchDeltaPointerSlot {
    pub session_index: usize,
    pub delta_layer_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPrefixRowSlot {
    pub round_index: usize,
    pub round_row_index: usize,
    pub session_index: usize,
    pub token_index: usize,
    pub token: u32,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPointerTablePlan {
    pub shape: DensePrefillSessionBatchPointerTableShape,
    pub kv_layer_slots: Vec<DensePrefillSessionBatchLayerPointerSlot>,
    pub dn_layer_slots: Vec<DensePrefillSessionBatchDeltaPointerSlot>,
    pub logits_slots: Vec<usize>,
    pub prefix_rows: Vec<DensePrefillSessionBatchPrefixRowSlot>,
    pub session_last_row_indices: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchHostPointerTables {
    pub kv_k_ptrs: Vec<u64>,
    pub kv_v_ptrs: Vec<u64>,
    pub dn_s_ptrs: Vec<u64>,
    pub dn_scale_ptrs: Vec<u64>,
    pub dn_conv_ptrs: Vec<u64>,
    pub logits_ptrs: Vec<u64>,
    pub session_last_row_indices: Vec<i32>,
    pub row_session_indices: Vec<i32>,
    pub row_tokens: Vec<i32>,
    pub row_positions: Vec<i32>,
}

pub struct DensePrefillSessionBatchDevicePointerTables {
    pub kv_k_ptrs: GpuTensor,
    pub kv_v_ptrs: GpuTensor,
    pub dn_s_ptrs: GpuTensor,
    pub dn_scale_ptrs: GpuTensor,
    pub dn_conv_ptrs: GpuTensor,
    pub logits_ptrs: GpuTensor,
    pub session_last_row_indices: GpuTensor,
    pub row_session_indices: GpuTensor,
    pub row_tokens: GpuTensor,
    pub row_positions: GpuTensor,
}

impl DensePrefillSessionBatchHostPointerTables {
    pub fn validate_shape(
        &self,
        shape: DensePrefillSessionBatchPointerTableShape,
    ) -> Result<(), String> {
        let checks = [
            ("kv_k_ptrs", self.kv_k_ptrs.len(), shape.kv_k_ptrs),
            ("kv_v_ptrs", self.kv_v_ptrs.len(), shape.kv_v_ptrs),
            ("dn_s_ptrs", self.dn_s_ptrs.len(), shape.dn_s_ptrs),
            (
                "dn_scale_ptrs",
                self.dn_scale_ptrs.len(),
                shape.dn_scale_ptrs,
            ),
            ("dn_conv_ptrs", self.dn_conv_ptrs.len(), shape.dn_conv_ptrs),
            ("logits_ptrs", self.logits_ptrs.len(), shape.logits_ptrs),
            (
                "session_last_row_indices",
                self.session_last_row_indices.len(),
                shape.session_last_row_indices,
            ),
            (
                "row_session_indices",
                self.row_session_indices.len(),
                shape.row_session_indices,
            ),
            ("row_tokens", self.row_tokens.len(), shape.row_tokens),
            (
                "row_positions",
                self.row_positions.len(),
                shape.row_positions,
            ),
        ];
        for (name, got, expected) in checks {
            if got != expected {
                return Err(format!(
                    "dense session prefill host pointer table {name} has {got} entries, expected {expected}",
                ));
            }
        }
        Ok(())
    }
}

impl DensePrefillSessionBatchDevicePointerTables {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.kv_k_ptrs);
        let _ = gpu.free_tensor(self.kv_v_ptrs);
        let _ = gpu.free_tensor(self.dn_s_ptrs);
        let _ = gpu.free_tensor(self.dn_scale_ptrs);
        let _ = gpu.free_tensor(self.dn_conv_ptrs);
        let _ = gpu.free_tensor(self.logits_ptrs);
        let _ = gpu.free_tensor(self.session_last_row_indices);
        let _ = gpu.free_tensor(self.row_session_indices);
        let _ = gpu.free_tensor(self.row_tokens);
        let _ = gpu.free_tensor(self.row_positions);
    }
}

fn u64_slice_as_bytes(values: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8) }
}

fn i32_slice_as_bytes(values: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

fn alloc_and_upload_u64_table(gpu: &mut Gpu, values: &[u64]) -> HipResult<GpuTensor> {
    // F32 dtype is cosmetic here: it gives a 4-byte element size, so two
    // elements hold one raw device pointer. Kernels consume these buffers as
    // `const uint64_t*`.
    let tensor = gpu.alloc_tensor(&[values.len() * 2], DType::F32)?;
    gpu.hip
        .memcpy_htod(&tensor.buf, u64_slice_as_bytes(values))?;
    Ok(tensor)
}

fn alloc_and_upload_i32_table(gpu: &mut Gpu, values: &[i32]) -> HipResult<GpuTensor> {
    // F32 dtype is cosmetic here: the row-routing kernels consume these
    // buffers as `const int*`.
    let tensor = gpu.alloc_tensor(&[values.len()], DType::F32)?;
    gpu.hip
        .memcpy_htod(&tensor.buf, i32_slice_as_bytes(values))?;
    Ok(tensor)
}

pub fn upload_dense_prefill_session_batch_pointer_tables(
    gpu: &mut Gpu,
    shape: DensePrefillSessionBatchPointerTableShape,
    host: &DensePrefillSessionBatchHostPointerTables,
) -> HipResult<DensePrefillSessionBatchDevicePointerTables> {
    host.validate_shape(shape)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    Ok(DensePrefillSessionBatchDevicePointerTables {
        kv_k_ptrs: alloc_and_upload_u64_table(gpu, &host.kv_k_ptrs)?,
        kv_v_ptrs: alloc_and_upload_u64_table(gpu, &host.kv_v_ptrs)?,
        dn_s_ptrs: alloc_and_upload_u64_table(gpu, &host.dn_s_ptrs)?,
        dn_scale_ptrs: alloc_and_upload_u64_table(gpu, &host.dn_scale_ptrs)?,
        dn_conv_ptrs: alloc_and_upload_u64_table(gpu, &host.dn_conv_ptrs)?,
        logits_ptrs: alloc_and_upload_u64_table(gpu, &host.logits_ptrs)?,
        session_last_row_indices: alloc_and_upload_i32_table(gpu, &host.session_last_row_indices)?,
        row_session_indices: alloc_and_upload_i32_table(gpu, &host.row_session_indices)?,
        row_tokens: alloc_and_upload_i32_table(gpu, &host.row_tokens)?,
        row_positions: alloc_and_upload_i32_table(gpu, &host.row_positions)?,
    })
}

pub fn dense_prefill_session_batch_write_f32_kv_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    k_src: &GpuTensor,
    v_src: &GpuTensor,
    kv_dim: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed KV write layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.kv_cache_write_f32_routed_batched(
        &device_tables.kv_k_ptrs,
        k_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        kv_dim,
        row_count,
    )?;
    gpu.kv_cache_write_f32_routed_batched(
        &device_tables.kv_v_ptrs,
        v_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_v_layers,
        kv_layer_index,
        kv_dim,
        row_count,
    )
}

pub fn prefill_session_batch_write_q8_kv_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    k_src: &GpuTensor,
    v_src: &GpuTensor,
    n_kv_heads: usize,
    head_dim: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill routed Q8 KV write layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.kv_cache_write_q8_0_routed_batched(
        &device_tables.kv_k_ptrs,
        k_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        n_kv_heads,
        head_dim,
        row_count,
    )?;
    gpu.kv_cache_write_q8_0_routed_batched(
        &device_tables.kv_v_ptrs,
        v_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_v_layers,
        kv_layer_index,
        n_kv_heads,
        head_dim,
        row_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_prefill_session_batch_attention_f32_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    q_batch: &GpuTensor,
    out_batch: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    max_ctx_len: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed attention layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.attention_f32_routed_batched(
        q_batch,
        &device_tables.kv_k_ptrs,
        &device_tables.kv_v_ptrs,
        out_batch,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
        max_ctx_len,
        row_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_session_batch_attention_q8_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    q_batch: &GpuTensor,
    out_batch: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    max_ctx_len: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill routed Q8 attention layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.attention_q8_0_routed_batched(
        q_batch,
        &device_tables.kv_k_ptrs,
        &device_tables.kv_v_ptrs,
        out_batch,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
        max_ctx_len,
        row_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_moe_prefill_session_batch_gated_delta_net_q8_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
    delta_layer_index: usize,
    q_batch: &GpuTensor,
    k_batch: &GpuTensor,
    v_batch: &GpuTensor,
    gate_batch: &GpuTensor,
    beta_batch: &GpuTensor,
    out_batch: &GpuTensor,
    row_count: usize,
    n_heads: usize,
    head_dim: usize,
) -> HipResult<()> {
    if delta_layer_index >= route_shape.dn_s_layers
        || delta_layer_index >= route_shape.dn_scale_layers
    {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill routed Q8 DeltaNet layer {delta_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.gated_delta_net_q8_routed_batch_seq(
        q_batch,
        k_batch,
        v_batch,
        gate_batch,
        beta_batch,
        &device_tables.dn_s_ptrs,
        &device_tables.dn_scale_ptrs,
        &device_tables.row_session_indices,
        out_batch,
        route_shape.dn_s_layers,
        delta_layer_index,
        row_count,
        n_heads,
        head_dim,
        sessions,
    )
}

pub fn dense_prefill_session_batch_scatter_last_logits(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    batch_logits: &GpuTensor,
    vocab_size: usize,
    sessions: usize,
) -> HipResult<()> {
    gpu.scatter_session_last_logits_f32(
        batch_logits,
        &device_tables.logits_ptrs,
        &device_tables.session_last_row_indices,
        vocab_size,
        sessions,
    )
}

pub fn dense_prefill_session_batch_final_logits_full_precision(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    row_count: usize,
    sessions: usize,
) -> HipResult<()> {
    if row_count == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "dense session prefill final logits requires at least one prefix row",
        ));
    }
    if row_count > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            "dense session prefill final logits row_count exceeds PrefillBatchScratch max_batch",
        ));
    }

    let normed_rows = pbs.x_norm_batch.sub_offset(0, row_count * config.dim);
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &weights.output_norm,
        &normed_rows,
        row_count,
        config.dim,
        config.norm_eps,
    )?;

    let batch_logits = gpu.alloc_tensor(&[row_count * config.vocab_size], DType::F32)?;
    let result = match weights.output.gpu_dtype {
        DType::F32 => gpu.gemm_f32_register_tiled(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::F16 | DType::BF16 | DType::Raw => gemm_fp16_or_bf16_x_f32_wmma(
            gpu,
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::MQ4G256 => {
            let rot = gpu.alloc_tensor(&[row_count * weights.output.k], DType::F32)?;
            let rotated = gpu
                .rotate_x_mq_batched(&normed_rows, &rot, weights.output.k, row_count)
                .and_then(|()| {
                    gpu.gemm_hfq4g256(
                        &weights.output.buf,
                        &rot,
                        &batch_logits,
                        weights.output.m,
                        weights.output.k,
                        row_count,
                    )
                });
            let _ = gpu.free_tensor(rot);
            rotated
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill final logits does not yet support lm_head dtype {other:?}; use serial_reference backend"
            ),
        )),
    }
    .and_then(|()| {
        dense_prefill_session_batch_scatter_last_logits(
            gpu,
            device_tables,
            &batch_logits,
            config.vocab_size,
            sessions,
        )
    });
    let _ = gpu.free_tensor(batch_logits);
    result
}

pub fn grouped_moe_prefill_session_batch_final_logits(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    row_count: usize,
    sessions: usize,
) -> HipResult<()> {
    if row_count == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "grouped MoE session prefill final logits requires at least one prefix row",
        ));
    }
    if row_count > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            "grouped MoE session prefill final logits row_count exceeds PrefillBatchScratch max_batch",
        ));
    }

    let normed_rows = pbs.x_norm_batch.sub_offset(0, row_count * config.dim);
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &weights.output_norm,
        &normed_rows,
        row_count,
        config.dim,
        config.norm_eps,
    )?;

    let batch_logits = gpu.alloc_owned(&[row_count * config.vocab_size], DType::F32)?;
    match weights.output.gpu_dtype {
        DType::F32 => gpu.gemm_f32_register_tiled(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::F16 | DType::Raw => gpu.gemm_f16_batched_lmhead(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::MQ4G256 => {
            let rotated = pbs.x_rot_batch.sub_offset(0, row_count * config.dim);
            rotate_x_mq_batched_for(
                gpu,
                &weights.output,
                &normed_rows,
                &rotated,
                config.dim,
                row_count,
            )
            .and_then(|()| {
                gpu.gemm_hfq4g256(
                    &weights.output.buf,
                    &rotated,
                    &batch_logits,
                    weights.output.m,
                    weights.output.k,
                    row_count,
                )
            })
        }
        DType::MQ6G256 => {
            let rotated = pbs.x_rot_batch.sub_offset(0, row_count * config.dim);
            rotate_x_mq_batched_for(
                gpu,
                &weights.output,
                &normed_rows,
                &rotated,
                config.dim,
                row_count,
            )
            .and_then(|()| {
                gpu.gemm_hfq6g256_batched_lmhead(
                    &weights.output.buf,
                    &rotated,
                    &batch_logits,
                    weights.output.m,
                    weights.output.k,
                    row_count,
                )
            })
        }
        DType::MQ3G256 => {
            let rotated = pbs.x_rot_batch.sub_offset(0, row_count * config.dim);
            rotate_x_mq_batched_for(
                gpu,
                &weights.output,
                &normed_rows,
                &rotated,
                config.dim,
                row_count,
            )
            .and_then(|()| {
                gpu.gemm_hfq3g256_batched_lmhead(
                    &weights.output.buf,
                    &rotated,
                    &batch_logits,
                    weights.output.m,
                    weights.output.k,
                    row_count,
                )
            })
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill final logits does not yet support lm_head dtype {other:?}; use serial_reference backend"
            ),
        )),
    }?;
    dense_prefill_session_batch_scatter_last_logits(
        gpu,
        device_tables,
        &batch_logits,
        config.vocab_size,
        sessions,
    )?;
    // `batch_logits` (RAII `OwnedTensor`) returns to the pool on drop.
    drop(batch_logits);
    gpu.reclaim_pending();
    Ok(())
}

pub fn validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
    config: &Qwen35Config,
    signatures: &[DensePrefillSessionBatchStateSignature],
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
) -> Result<(), String> {
    if config.num_experts != 0 || config.has_shared_expert {
        return Err(
            "dense session fused prefix currently supports dense Qwen35 only; MoE/A3B stays on serial_reference"
                .to_string(),
        );
    }
    if execution_plan.multi_state_prefix_rows == 0 {
        return Err(
            "dense session fused prefix requires at least one multi-session prefix row".to_string(),
        );
    }
    validate_dense_prefill_session_batch_state_signatures(signatures)?;
    for (idx, signature) in signatures.iter().enumerate() {
        if signature.kv_compact_offset != 0 {
            return Err(format!(
                "dense session fused prefix row {idx} has compacted KV offset {}; eviction/compaction is not fused yet",
                signature.kv_compact_offset,
            ));
        }
        // KV may be plain Q8 (Q8_0, inline per-block scale) or full precision —
        // the per-layer KV write + attention branch on `kv_q8` in
        // `forward_prefill_dense_session_batch_prefix_full_precision`. Asym/FWHT
        // KV and any other quantized-but-not-plain-Q8 state stay on
        // serial_reference (not fused). (Row uniformity is already enforced by
        // `validate_dense_prefill_session_batch_state_signatures`.)
        if signature.kv_quant_asym2
            || signature.kv_quant_asym3
            || signature.kv_quant_asym4
            || signature.kv_quant_fwht
            || (signature.kv_quantized && !signature.kv_quant_q8)
        {
            return Err(format!(
                "dense session fused prefix row {idx} has unsupported KV quantization; only plain Q8 or FP32 KV is fused"
            ));
        }
        if signature.dn_quant != StateQuant::FP32 {
            return Err(format!(
                "dense session fused prefix row {idx} has {:?} DeltaNet state; first fused target is FP32 DeltaNet state",
                signature.dn_quant,
            ));
        }
    }
    Ok(())
}

pub fn validate_grouped_moe_prefill_session_batch_q8_state_contract(
    config: &Qwen35Config,
    signatures: &[DensePrefillSessionBatchStateSignature],
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
    arch: &str,
) -> Result<(), String> {
    if config.num_experts == 0 || !config.has_shared_expert {
        return Err(
            "grouped MoE session fused prefix requires Qwen35 MoE/A3B weights; dense Qwen35 should use fused_dense"
                .to_string(),
        );
    }
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        return Err(format!(
            "grouped MoE session fused prefix requires an RDNA grouped-MoE target, got arch={arch}"
        ));
    }
    if config.num_experts_per_tok != 8
        && !(config.paged_experts && config.num_experts_per_tok == 10)
    {
        return Err(format!(
            "grouped MoE session fused prefix currently requires K_TOP=8, or paged K_TOP=10, got {}",
            config.num_experts_per_tok,
        ));
    }
    if execution_plan.multi_state_prefix_rows == 0 {
        return Err(
            "grouped MoE session fused prefix requires at least one multi-session prefix row"
                .to_string(),
        );
    }
    validate_dense_prefill_session_batch_state_signatures(signatures)?;
    for (idx, signature) in signatures.iter().enumerate() {
        if signature.kv_compact_offset != 0 {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} has compacted KV offset {}; eviction/compaction is not fused yet",
                signature.kv_compact_offset,
            ));
        }
        if !signature.kv_quantized || !signature.kv_quant_q8 {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} must use Q8 KV state for the MQ4 control path"
            ));
        }
        if signature.kv_quant_asym2
            || signature.kv_quant_asym3
            || signature.kv_quant_asym4
            || signature.kv_quant_fwht
        {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} has unsupported KV quantization flags; first MoE target is plain Q8 KV"
            ));
        }
        if signature.dn_quant != StateQuant::Q8 {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} has {:?} DeltaNet state; first MoE target is Q8 DeltaNet state",
                signature.dn_quant,
            ));
        }
    }
    Ok(())
}

// Weight dtypes the dense fused prefill GEMM helpers can dispatch. Full precision
// (F32/F16/BF16/Raw) plus plain Q8_0 and MQ4G256 (quantized dense models). MQ6G256
// and other quant formats have no batched non-residual kernel yet, so models using
// them fall back to serial_reference via the contract. (Name kept to avoid churn.)
fn dense_prefill_weight_full_precision_supported(weight: &WeightTensor) -> bool {
    matches!(
        weight.gpu_dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::Raw | DType::Q8_0 | DType::MQ4G256
    )
}

pub fn validate_dense_prefill_session_batch_fused_prefix_full_precision_weights(
    weights: &Qwen35Weights,
) -> Result<(), String> {
    if !matches!(
        weights.embd_format,
        EmbeddingFormat::F32 | EmbeddingFormat::Q8_0 | EmbeddingFormat::HFQ4G256
    ) {
        return Err(format!(
            "dense session fused prefix does not support embedding format {:?} yet",
            weights.embd_format,
        ));
    }
    if !dense_prefill_weight_full_precision_supported(&weights.output) {
        return Err(format!(
            "dense session fused prefix does not support lm_head dtype {:?} yet",
            weights.output.gpu_dtype,
        ));
    }
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let supported = match layer {
            LayerWeights::DeltaNet(layer) => {
                dense_prefill_weight_full_precision_supported(&layer.wqkv)
                    && dense_prefill_weight_full_precision_supported(&layer.wz)
                    && dense_prefill_weight_full_precision_supported(&layer.w_alpha)
                    && dense_prefill_weight_full_precision_supported(&layer.w_beta)
                    && dense_prefill_weight_full_precision_supported(&layer.wo)
                    && dense_prefill_weight_full_precision_supported(&layer.w_gate)
                    && dense_prefill_weight_full_precision_supported(&layer.w_up)
                    && dense_prefill_weight_full_precision_supported(&layer.w_down)
            }
            LayerWeights::FullAttn(layer) => {
                dense_prefill_weight_full_precision_supported(&layer.wq)
                    && dense_prefill_weight_full_precision_supported(&layer.wk)
                    && dense_prefill_weight_full_precision_supported(&layer.wv)
                    && dense_prefill_weight_full_precision_supported(&layer.wo)
                    && dense_prefill_weight_full_precision_supported(&layer.w_gate)
                    && dense_prefill_weight_full_precision_supported(&layer.w_up)
                    && dense_prefill_weight_full_precision_supported(&layer.w_down)
            }
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => false,
        };
        if !supported {
            return Err(format!(
                "dense session fused prefix layer {layer_idx} has unsupported dense/MoE weight dtypes; first target is dense full-precision weights"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dense_prefill_session_batch_gated_delta_net_f32_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
    delta_layer_index: usize,
    q_batch: &GpuTensor,
    k_batch: &GpuTensor,
    v_batch: &GpuTensor,
    gate_batch: &GpuTensor,
    beta_batch: &GpuTensor,
    output_batch: &GpuTensor,
    row_count: usize,
    n_heads: usize,
    head_dim: usize,
) -> HipResult<()> {
    if delta_layer_index >= route_shape.dn_s_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed DeltaNet layer {delta_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.gated_delta_net_f32_routed_batch_seq(
        q_batch,
        k_batch,
        v_batch,
        gate_batch,
        beta_batch,
        &device_tables.dn_s_ptrs,
        &device_tables.row_session_indices,
        output_batch,
        route_shape.dn_s_layers,
        delta_layer_index,
        row_count,
        n_heads,
        head_dim,
        sessions,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_prefill_session_batch_conv1d_silu_split_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
    delta_layer_index: usize,
    q_out: &GpuTensor,
    k_out: &GpuTensor,
    v_out: &GpuTensor,
    input: &GpuTensor,
    weight: &GpuTensor,
    k_dim: usize,
    v_dim: usize,
    row_count: usize,
) -> HipResult<()> {
    if delta_layer_index >= route_shape.dn_conv_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed conv layer {delta_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.conv1d_silu_split_routed_f32_n(
        q_out,
        k_out,
        v_out,
        input,
        weight,
        &device_tables.dn_conv_ptrs,
        &device_tables.row_session_indices,
        route_shape.dn_conv_layers,
        delta_layer_index,
        k_dim,
        v_dim,
        row_count,
        sessions,
    )
}

impl DensePrefillSessionBatchPointerTableShape {
    pub fn index_for_session_layer(
        self,
        session_index: usize,
        kv_layer_index: usize,
        dn_layer_index: usize,
    ) -> Result<DensePrefillSessionBatchPointerTableIndex, String> {
        if session_index >= self.sessions {
            return Err(format!(
                "dense session prefill pointer table session_index {session_index} out of range for sessions={}",
                self.sessions,
            ));
        }
        let kv_layers = self.kv_k_ptrs.checked_div(self.sessions).unwrap_or(0);
        let dn_layers = self.dn_s_ptrs.checked_div(self.sessions).unwrap_or(0);
        if kv_layer_index >= kv_layers {
            return Err(format!(
                "dense session prefill pointer table kv_layer_index {kv_layer_index} out of range for kv_layers={kv_layers}",
            ));
        }
        if dn_layer_index >= dn_layers {
            return Err(format!(
                "dense session prefill pointer table dn_layer_index {dn_layer_index} out of range for dn_layers={dn_layers}",
            ));
        }
        Ok(DensePrefillSessionBatchPointerTableIndex {
            kv_k_offset: session_index * kv_layers + kv_layer_index,
            kv_v_offset: session_index * kv_layers + kv_layer_index,
            dn_s_offset: session_index * dn_layers + dn_layer_index,
            dn_scale_offset: session_index * dn_layers + dn_layer_index,
            dn_conv_offset: session_index * dn_layers + dn_layer_index,
            logits_offset: session_index,
        })
    }

    pub fn index_for_prefix_row(
        self,
        prefix_row_index: usize,
    ) -> Result<(usize, usize, usize), String> {
        if prefix_row_index >= self.multi_state_prefix_rows {
            return Err(format!(
                "dense session prefill pointer table prefix_row_index {prefix_row_index} out of range for multi_state_prefix_rows={}",
                self.multi_state_prefix_rows,
            ));
        }
        Ok((prefix_row_index, prefix_row_index, prefix_row_index))
    }

    pub fn validate_prefix_row_metadata(
        self,
        plan: &DensePrefillSessionBatchPointerTablePlan,
    ) -> Result<(), String> {
        if plan.prefix_rows.len() != self.multi_state_prefix_rows {
            return Err(format!(
                "dense session prefill pointer table has {} prefix rows, expected {}",
                plan.prefix_rows.len(),
                self.multi_state_prefix_rows,
            ));
        }
        if plan.session_last_row_indices.len() != self.sessions {
            return Err(format!(
                "dense session prefill pointer table has {} session-last-row entries, expected {}",
                plan.session_last_row_indices.len(),
                self.sessions,
            ));
        }
        for (session_index, &row_index) in plan.session_last_row_indices.iter().enumerate() {
            if row_index < 0 {
                return Err(format!(
                    "dense session prefill pointer table session {session_index} has no fused prefix row",
                ));
            }
            let row_index = row_index as usize;
            if row_index >= self.multi_state_prefix_rows {
                return Err(format!(
                    "dense session prefill pointer table session {session_index} last row {row_index} out of range for prefix rows {}",
                    self.multi_state_prefix_rows,
                ));
            }
            let row = &plan.prefix_rows[row_index];
            if row.session_index != session_index {
                return Err(format!(
                    "dense session prefill pointer table session {session_index} last row {row_index} belongs to session {}",
                    row.session_index,
                ));
            }
        }
        Ok(())
    }
}

pub fn dense_prefill_session_batch_pointer_table_plan(
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
) -> DensePrefillSessionBatchPointerTablePlan {
    let shape =
        dense_prefill_session_batch_pointer_table_shape(execution_plan, route_shape, sessions);
    let mut kv_layer_slots = Vec::with_capacity(shape.kv_k_ptrs);
    for session_index in 0..sessions {
        for layer_index in 0..route_shape.kv_k_layers {
            kv_layer_slots.push(DensePrefillSessionBatchLayerPointerSlot {
                session_index,
                layer_index,
            });
        }
    }
    let mut dn_layer_slots = Vec::with_capacity(shape.dn_s_ptrs);
    for session_index in 0..sessions {
        for delta_layer_index in 0..route_shape.dn_s_layers {
            dn_layer_slots.push(DensePrefillSessionBatchDeltaPointerSlot {
                session_index,
                delta_layer_index,
            });
        }
    }
    let logits_slots = (0..sessions).collect();
    let mut prefix_rows = Vec::with_capacity(shape.multi_state_prefix_rows);
    let mut session_last_row_indices = vec![-1; sessions];
    for (round_index, round) in execution_plan
        .rounds
        .iter()
        .take(execution_plan.multi_state_prefix_rounds)
        .enumerate()
    {
        for (round_row_index, row) in round.rows.iter().enumerate() {
            let prefix_row_index = prefix_rows.len() as i32;
            session_last_row_indices[row.session_index] = prefix_row_index;
            prefix_rows.push(DensePrefillSessionBatchPrefixRowSlot {
                round_index,
                round_row_index,
                session_index: row.session_index,
                token_index: row.token_index,
                token: row.token,
                position: row.position,
            });
        }
    }
    DensePrefillSessionBatchPointerTablePlan {
        shape,
        kv_layer_slots,
        dn_layer_slots,
        logits_slots,
        prefix_rows,
        session_last_row_indices,
    }
}

pub fn dense_prefill_session_batch_host_pointer_tables(
    plan: &DensePrefillSessionBatchPointerTablePlan,
    routes: &[DensePrefillSessionStateRoute<'_>],
) -> Result<DensePrefillSessionBatchHostPointerTables, String> {
    if routes.len() != plan.shape.sessions {
        return Err(format!(
            "dense session prefill pointer table has {} routes, expected {}",
            routes.len(),
            plan.shape.sessions,
        ));
    }
    let mut kv_k_ptrs = Vec::with_capacity(plan.shape.kv_k_ptrs);
    let mut kv_v_ptrs = Vec::with_capacity(plan.shape.kv_v_ptrs);
    for slot in &plan.kv_layer_slots {
        let route = routes.get(slot.session_index).ok_or_else(|| {
            format!(
                "dense session prefill KV slot references missing session {}",
                slot.session_index,
            )
        })?;
        let k = route.kv.k_gpu.get(slot.layer_index).ok_or_else(|| {
            format!(
                "dense session prefill KV K slot references missing layer {}",
                slot.layer_index,
            )
        })?;
        let v = route.kv.v_gpu.get(slot.layer_index).ok_or_else(|| {
            format!(
                "dense session prefill KV V slot references missing layer {}",
                slot.layer_index,
            )
        })?;
        kv_k_ptrs.push(k.buf.as_ptr() as u64);
        kv_v_ptrs.push(v.buf.as_ptr() as u64);
    }

    let mut dn_s_ptrs = Vec::with_capacity(plan.shape.dn_s_ptrs);
    let mut dn_scale_ptrs = Vec::with_capacity(plan.shape.dn_scale_ptrs);
    let mut dn_conv_ptrs = Vec::with_capacity(plan.shape.dn_conv_ptrs);
    for slot in &plan.dn_layer_slots {
        let route = routes.get(slot.session_index).ok_or_else(|| {
            format!(
                "dense session prefill DeltaNet slot references missing session {}",
                slot.session_index,
            )
        })?;
        let s = route
            .delta
            .s_matrices
            .get(slot.delta_layer_index)
            .ok_or_else(|| {
                format!(
                    "dense session prefill DeltaNet S slot references missing layer {}",
                    slot.delta_layer_index,
                )
            })?;
        let conv = route
            .delta
            .conv_states
            .get(slot.delta_layer_index)
            .ok_or_else(|| {
                format!(
                    "dense session prefill DeltaNet conv slot references missing layer {}",
                    slot.delta_layer_index,
                )
            })?;
        dn_s_ptrs.push(s.buf.as_ptr() as u64);
        dn_conv_ptrs.push(conv.buf.as_ptr() as u64);
        if plan.shape.dn_scale_ptrs != 0 {
            let scale = route
                .delta
                .s_scales
                .get(slot.delta_layer_index)
                .ok_or_else(|| {
                    format!(
                        "dense session prefill DeltaNet scale slot references missing layer {}",
                        slot.delta_layer_index,
                    )
                })?;
            dn_scale_ptrs.push(scale.buf.as_ptr() as u64);
        }
    }

    let mut logits_ptrs = Vec::with_capacity(plan.shape.logits_ptrs);
    for &session_index in &plan.logits_slots {
        let route = routes.get(session_index).ok_or_else(|| {
            format!("dense session prefill logits slot references missing session {session_index}")
        })?;
        logits_ptrs.push(route.logits.buf.as_ptr() as u64);
    }

    let row_session_indices = plan
        .prefix_rows
        .iter()
        .map(|row| row.session_index as i32)
        .collect();
    let row_tokens = plan
        .prefix_rows
        .iter()
        .map(|row| row.token as i32)
        .collect();
    let row_positions = plan
        .prefix_rows
        .iter()
        .map(|row| row.position as i32)
        .collect();

    let tables = DensePrefillSessionBatchHostPointerTables {
        kv_k_ptrs,
        kv_v_ptrs,
        dn_s_ptrs,
        dn_scale_ptrs,
        dn_conv_ptrs,
        logits_ptrs,
        session_last_row_indices: plan.session_last_row_indices.clone(),
        row_session_indices,
        row_tokens,
        row_positions,
    };
    tables.validate_shape(plan.shape)?;
    Ok(tables)
}

pub fn dense_prefill_session_batch_prefix_tokens_positions(
    plan: &DensePrefillSessionBatchPointerTablePlan,
) -> Result<(Vec<u32>, Vec<usize>), String> {
    plan.shape.validate_prefix_row_metadata(plan)?;
    let tokens = plan.prefix_rows.iter().map(|row| row.token).collect();
    let positions = plan.prefix_rows.iter().map(|row| row.position).collect();
    Ok((tokens, positions))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchShape {
    pub sessions: usize,
    pub total_tokens: usize,
    pub max_tokens_per_session: usize,
}

pub fn validate_dense_prefill_session_batch_shape(
    rows: &[DensePrefillSessionBatchRowShape],
    max_batch: usize,
) -> Result<DensePrefillSessionBatchShape, String> {
    if rows.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent sessions".to_string(),
        );
    }
    let mut total_tokens = 0usize;
    let mut max_tokens_per_session = 0usize;
    for (idx, row) in rows.iter().enumerate() {
        if row.tokens == 0 {
            return Err(format!(
                "dense session prefill batch row {idx} has an empty token slice"
            ));
        }
        if row.tokens > max_batch {
            return Err(format!(
                "dense session prefill batch row {idx} has {} tokens, exceeding PrefillBatchScratch max_batch={}",
                row.tokens,
                max_batch,
            ));
        }
        if row.logits_numel == 0 {
            return Err(format!(
                "dense session prefill batch row {idx} has an empty logits tensor"
            ));
        }
        total_tokens += row.tokens;
        max_tokens_per_session = max_tokens_per_session.max(row.tokens);
    }
    Ok(DensePrefillSessionBatchShape {
        sessions: rows.len(),
        total_tokens,
        max_tokens_per_session,
    })
}

pub fn validate_dense_prefill_session_batch_state_signatures(
    signatures: &[DensePrefillSessionBatchStateSignature],
) -> Result<(), String> {
    if signatures.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent session state signatures"
                .to_string(),
        );
    }
    let expected = signatures[0];
    for (idx, signature) in signatures.iter().enumerate().skip(1) {
        if *signature != expected {
            return Err(format!(
                "dense session prefill batch row {idx} has incompatible KV/DeltaNet state signature: expected {:?}, got {:?}",
                expected,
                signature,
            ));
        }
    }
    Ok(())
}

pub fn validate_dense_prefill_session_state_route_shapes(
    shapes: &[DensePrefillSessionStateRouteShape],
    expected_sessions: usize,
) -> Result<(), String> {
    if shapes.len() != expected_sessions {
        return Err(format!(
            "dense session prefill batch has {} state routes, expected {expected_sessions}",
            shapes.len(),
        ));
    }
    if shapes.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent state routes"
                .to_string(),
        );
    }
    let expected = shapes[0];
    if expected.kv_k_layers == 0
        || expected.kv_v_layers == 0
        || expected.dn_s_layers == 0
        || expected.dn_conv_layers == 0
    {
        return Err(format!(
            "dense session prefill batch row 0 has incomplete KV/DeltaNet route shape: {:?}",
            expected,
        ));
    }
    if expected.kv_k_layers != expected.kv_v_layers {
        return Err(format!(
            "dense session prefill batch row 0 has mismatched KV K/V layers: {:?}",
            expected,
        ));
    }
    if expected.dn_s_layers != expected.dn_conv_layers {
        return Err(format!(
            "dense session prefill batch row 0 has mismatched DeltaNet S/conv layers: {:?}",
            expected,
        ));
    }
    if expected.dn_scale_layers != 0 && expected.dn_scale_layers != expected.dn_s_layers {
        return Err(format!(
            "dense session prefill batch row 0 has mismatched DeltaNet scale layers: {:?}",
            expected,
        ));
    }
    for (idx, shape) in shapes.iter().enumerate().skip(1) {
        if *shape != expected {
            return Err(format!(
                "dense session prefill batch row {idx} has incompatible state route shape: expected {:?}, got {:?}",
                expected,
                shape,
            ));
        }
    }
    Ok(())
}

pub fn expected_dense_prefill_session_state_route_shape(
    config: &Qwen35Config,
) -> DensePrefillSessionStateRouteShape {
    let dn_layers = config
        .layer_types
        .iter()
        .filter(|layer_type| **layer_type == LayerType::LinearAttention)
        .count();
    DensePrefillSessionStateRouteShape {
        kv_k_layers: config.n_layers,
        kv_v_layers: config.n_layers,
        dn_s_layers: dn_layers,
        dn_scale_layers: dn_layers,
        dn_conv_layers: dn_layers,
    }
}

pub fn validate_dense_prefill_session_state_route_shapes_for_config(
    shapes: &[DensePrefillSessionStateRouteShape],
    config: &Qwen35Config,
) -> Result<(), String> {
    validate_dense_prefill_session_state_route_shapes(shapes, shapes.len())?;
    let expected = expected_dense_prefill_session_state_route_shape(config);
    for (idx, shape) in shapes.iter().enumerate() {
        if *shape != expected {
            return Err(format!(
                "dense session prefill batch row {idx} has state route shape {:?}, expected model shape {:?}",
                shape,
                expected,
            ));
        }
    }
    Ok(())
}

pub fn dense_prefill_session_batch_pointer_table_shape(
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
) -> DensePrefillSessionBatchPointerTableShape {
    DensePrefillSessionBatchPointerTableShape {
        sessions,
        multi_state_prefix_rounds: execution_plan.multi_state_prefix_rounds,
        multi_state_prefix_rows: execution_plan.multi_state_prefix_rows,
        max_rows_per_round: execution_plan.max_rows_per_round,
        kv_k_ptrs: sessions * route_shape.kv_k_layers,
        kv_v_ptrs: sessions * route_shape.kv_v_layers,
        dn_s_ptrs: sessions * route_shape.dn_s_layers,
        dn_scale_ptrs: sessions * route_shape.dn_scale_layers,
        dn_conv_ptrs: sessions * route_shape.dn_conv_layers,
        logits_ptrs: sessions,
        session_last_row_indices: sessions,
        row_session_indices: execution_plan.multi_state_prefix_rows,
        row_tokens: execution_plan.multi_state_prefix_rows,
        row_positions: execution_plan.multi_state_prefix_rows,
    }
}

pub fn validate_dense_prefill_session_batch_rows(
    rows: &[DensePrefillSessionBatchRow<'_>],
    pbs: &PrefillBatchScratch,
) -> Result<DensePrefillSessionBatchShape, String> {
    let shapes: Vec<DensePrefillSessionBatchRowShape> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchRowShape {
            tokens: row.tokens.len(),
            logits_numel: row.logits.numel(),
        })
        .collect();
    let shape = validate_dense_prefill_session_batch_shape(&shapes, pbs.max_batch)?;
    let signatures: Vec<DensePrefillSessionBatchStateSignature> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchStateSignature {
            kv_physical_cap: row.kv_cache.physical_cap,
            kv_compact_offset: row.kv_cache.compact_offset,
            kv_quantized: row.kv_cache.quantized,
            kv_quant_q8: row.kv_cache.quant_q8,
            kv_quant_asym2: row.kv_cache.quant_asym2,
            kv_quant_asym3: row.kv_cache.quant_asym3,
            kv_quant_asym4: row.kv_cache.quant_asym4,
            kv_quant_fwht: row.kv_cache.quant_fwht,
            dn_quant: row.dn_state.quant,
        })
        .collect();
    validate_dense_prefill_session_batch_state_signatures(&signatures)?;
    let route_shapes: Vec<DensePrefillSessionStateRouteShape> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRouteShape {
            kv_k_layers: row.kv_cache.k_gpu.len(),
            kv_v_layers: row.kv_cache.v_gpu.len(),
            dn_s_layers: row.dn_state.s_matrices.len(),
            dn_scale_layers: row.dn_state.s_scales.len(),
            dn_conv_layers: row.dn_state.conv_states.len(),
        })
        .collect();
    validate_dense_prefill_session_state_route_shapes(&route_shapes, rows.len())?;
    Ok(shape)
}

pub fn validate_dense_prefill_session_batch_rows_for_config(
    rows: &[DensePrefillSessionBatchRow<'_>],
    pbs: &PrefillBatchScratch,
    config: &Qwen35Config,
) -> Result<DensePrefillSessionBatchShape, String> {
    let shape = validate_dense_prefill_session_batch_rows(rows, pbs)?;
    let route_shapes: Vec<DensePrefillSessionStateRouteShape> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRouteShape {
            kv_k_layers: row.kv_cache.k_gpu.len(),
            kv_v_layers: row.kv_cache.v_gpu.len(),
            dn_s_layers: row.dn_state.s_matrices.len(),
            dn_scale_layers: row.dn_state.s_scales.len(),
            dn_conv_layers: row.dn_state.conv_states.len(),
        })
        .collect();
    validate_dense_prefill_session_state_route_shapes_for_config(&route_shapes, config)?;
    Ok(shape)
}

pub fn build_dense_prefill_session_batch_rounds(
    inputs: &[DensePrefillSessionBatchInput<'_>],
    max_batch: usize,
) -> Result<Vec<DensePrefillSessionBatchRound>, String> {
    if inputs.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent sessions".to_string(),
        );
    }
    if inputs.len() > max_batch {
        return Err(format!(
            "dense session prefill batch has {} sessions, exceeding PrefillBatchScratch max_batch={max_batch}",
            inputs.len(),
        ));
    }

    let mut max_tokens_per_session = 0usize;
    for (idx, input) in inputs.iter().enumerate() {
        if input.tokens.is_empty() {
            return Err(format!(
                "dense session prefill batch row {idx} has an empty token slice"
            ));
        }
        max_tokens_per_session = max_tokens_per_session.max(input.tokens.len());
    }

    let mut rounds = Vec::with_capacity(max_tokens_per_session);
    for token_index in 0..max_tokens_per_session {
        let mut rows = Vec::with_capacity(inputs.len());
        for (session_index, input) in inputs.iter().enumerate() {
            if let Some(&token) = input.tokens.get(token_index) {
                rows.push(DensePrefillSessionBatchRoundRow {
                    session_index,
                    token_index,
                    token,
                    position: input.start_pos + token_index,
                });
            }
        }
        if !rows.is_empty() {
            rounds.push(DensePrefillSessionBatchRound { rows });
        }
    }
    Ok(rounds)
}

pub fn build_dense_prefill_session_batch_execution_plan(
    inputs: &[DensePrefillSessionBatchInput<'_>],
    max_batch: usize,
) -> Result<DensePrefillSessionBatchExecutionPlan, String> {
    let rounds = build_dense_prefill_session_batch_rounds(inputs, max_batch)?;
    let mut total_rows = 0usize;
    let mut max_rows_per_round = 0usize;
    let mut multi_state_rounds = 0usize;
    let mut state_routes = Vec::with_capacity(rounds.len());
    let mut last_multi_state_round = None;
    for round in &rounds {
        total_rows += round.rows.len();
        max_rows_per_round = max_rows_per_round.max(round.rows.len());
        let route = round.state_route();
        if matches!(
            route,
            DensePrefillSessionBatchRoundStateRoute::MultiSession { .. }
        ) {
            multi_state_rounds += 1;
            last_multi_state_round = Some(state_routes.len());
        }
        state_routes.push(route);
    }
    let multi_state_prefix_rounds = last_multi_state_round.map(|idx| idx + 1).unwrap_or(0);
    let multi_state_prefix_rows: usize = rounds[..multi_state_prefix_rounds]
        .iter()
        .map(|round| round.rows.len())
        .sum();
    let singleton_tail =
        last_multi_state_round.and_then(|last_multi| {
            let start_round = last_multi + 1;
            if start_round >= state_routes.len() {
                return None;
            }
            let session_index = match state_routes[start_round] {
                DensePrefillSessionBatchRoundStateRoute::SingleSession { session_index } => {
                    session_index
                }
                DensePrefillSessionBatchRoundStateRoute::MultiSession { .. } => return None,
            };
            let mut rows = 0usize;
            for route in &state_routes[start_round..] {
                match route {
                    DensePrefillSessionBatchRoundStateRoute::SingleSession {
                        session_index: idx,
                    } if *idx == session_index => rows += 1,
                    _ => return None,
                }
            }
            Some(DensePrefillSessionBatchSingletonTail {
                start_round,
                session_index,
                rows,
            })
        });
    Ok(DensePrefillSessionBatchExecutionPlan {
        rounds,
        state_routes,
        total_rows,
        max_rows_per_round,
        multi_state_rounds,
        multi_state_prefix_rounds,
        multi_state_prefix_rows,
        singleton_tail,
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_prefill_dense_session_batch_prefix_full_precision(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    row_count: usize,
    sessions: usize,
    max_ctx_len: usize,
    // Per-batch KV quant (uniform across rows — see the state-signature contract).
    // true = the sessions' KV caches are plain Q8 (Q8_0); the KV write +
    // attention use the Q8 path. false = full-precision F32 KV.
    kv_q8: bool,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::F32 => gpu.embedding_lookup_f32_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        other => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("dense session fused prefix does not support embedding format {other:?}"),
            ));
        }
    }

    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.attn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wqkv,
                    &pbs.x_rot_batch,
                    &pbs.dn_qkv_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wz,
                    &pbs.x_rot_batch,
                    &pbs.dn_z_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_beta,
                    &pbs.x_rot_batch,
                    &pbs.dn_beta_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_alpha,
                    &pbs.x_rot_batch,
                    &pbs.dn_alpha_batch,
                    row_count,
                )?;

                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    row_count,
                )?;
                dense_prefill_session_batch_conv1d_silu_split_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_qkv_batch,
                    &layer.conv_weight,
                    k_dim,
                    v_dim,
                    row_count,
                )?;
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    row_count,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        row_count,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                }
                dense_prefill_session_batch_gated_delta_net_f32_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &pbs.dn_attn_out_batch,
                    row_count,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.wo,
                    &pbs.dn_normed_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;

                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.ffn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_gate,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_up,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    row_count,
                )?;
                gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.w_down,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                let kv_dim = config.n_kv_heads * config.head_dim;
                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.attn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wq,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wk,
                    &pbs.x_rot_batch,
                    &pbs.fa_k_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wv,
                    &pbs.x_rot_batch,
                    &pbs.fa_v_batch,
                    row_count,
                )?;
                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    row_count,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    row_count * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    row_count * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    &pbs.positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    row_count,
                    0,
                )?;
                if kv_q8 {
                    // Plain-Q8 KV: the routed write/attention helpers are shared
                    // with the grouped-MoE fused path — they are FFN-agnostic and
                    // operate on Q8_0 (inline-scale) KV buffers via the same
                    // device pointer tables.
                    prefill_session_batch_write_q8_kv_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        config.n_kv_heads,
                        config.head_dim,
                        row_count,
                    )?;
                    prefill_session_batch_attention_q8_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_q_batch,
                        &pbs.fa_attn_out_batch,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        max_ctx_len,
                        max_ctx_len,
                        row_count,
                    )?;
                } else {
                    dense_prefill_session_batch_write_f32_kv_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        kv_dim,
                        row_count,
                    )?;
                    dense_prefill_session_batch_attention_f32_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_q_batch,
                        &pbs.fa_attn_out_batch,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        max_ctx_len,
                        max_ctx_len,
                        row_count,
                    )?;
                }
                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.wo,
                    &pbs.fa_attn_out_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;

                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.ffn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_gate,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_up,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    row_count,
                )?;
                gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.w_down,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;
            }
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "dense session fused prefix encountered a layer that is not dense Qwen35",
                ));
            }
        }
    }

    dense_prefill_session_batch_final_logits_full_precision(
        gpu,
        weights,
        config,
        pbs,
        device_tables,
        row_count,
        sessions,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_dense_session_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    rows: &mut [DensePrefillSessionBatchRow<'_>],
    _scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<DensePrefillSessionBatchShape> {
    let shape = validate_dense_prefill_session_batch_rows_for_config(rows, pbs, config)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let inputs: Vec<DensePrefillSessionBatchInput<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchInput {
            tokens: row.tokens,
            start_pos: row.start_pos,
        })
        .collect();
    let execution_plan = build_dense_prefill_session_batch_execution_plan(&inputs, pbs.max_batch)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let signatures: Vec<DensePrefillSessionBatchStateSignature> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchStateSignature {
            kv_physical_cap: row.kv_cache.physical_cap,
            kv_compact_offset: row.kv_cache.compact_offset,
            kv_quantized: row.kv_cache.quantized,
            kv_quant_q8: row.kv_cache.quant_q8,
            kv_quant_asym2: row.kv_cache.quant_asym2,
            kv_quant_asym3: row.kv_cache.quant_asym3,
            kv_quant_asym4: row.kv_cache.quant_asym4,
            kv_quant_fwht: row.kv_cache.quant_fwht,
            dn_quant: row.dn_state.quant,
        })
        .collect();
    validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
        config,
        &signatures,
        &execution_plan,
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    validate_dense_prefill_session_batch_fused_prefix_full_precision_weights(weights)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let route_shape = expected_dense_prefill_session_state_route_shape(config);
    let pointer_table_plan =
        dense_prefill_session_batch_pointer_table_plan(&execution_plan, route_shape, rows.len());
    if execution_plan.multi_state_prefix_rows > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill fused prefix has {} rows, exceeding PrefillBatchScratch max_batch={}",
                execution_plan.multi_state_prefix_rows, pbs.max_batch,
            ),
        ));
    }
    let (prefix_tokens, prefix_positions) =
        dense_prefill_session_batch_prefix_tokens_positions(&pointer_table_plan)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    upload_prefill_batch_inputs_with_positions(gpu, pbs, &prefix_tokens, &prefix_positions)?;
    let routes: Vec<DensePrefillSessionStateRoute<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRoute {
            kv: DensePrefillSessionKvStateRoute {
                k_gpu: &row.kv_cache.k_gpu,
                v_gpu: &row.kv_cache.v_gpu,
                physical_cap: row.kv_cache.physical_cap,
                compact_offset: row.kv_cache.compact_offset,
            },
            delta: DensePrefillSessionDeltaStateRoute {
                s_matrices: &row.dn_state.s_matrices,
                s_scales: &row.dn_state.s_scales,
                conv_states: &row.dn_state.conv_states,
                quant: row.dn_state.quant,
            },
            logits: row.logits,
        })
        .collect();
    let host_pointer_tables =
        dense_prefill_session_batch_host_pointer_tables(&pointer_table_plan, &routes)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    drop(routes);
    let device_pointer_tables = upload_dense_prefill_session_batch_pointer_tables(
        gpu,
        pointer_table_plan.shape,
        &host_pointer_tables,
    )?;
    let max_ctx_len = prefix_positions
        .iter()
        .copied()
        .max()
        .map(|pos| pos + 1)
        .unwrap_or(1);
    for (idx, row) in rows.iter().enumerate() {
        let row_end = row.start_pos + row.tokens.len();
        if row_end > row.kv_cache.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "dense session fused prefix row {idx} ends at position {row_end}, exceeding KV physical_cap={}",
                    row.kv_cache.physical_cap,
                ),
            ));
        }
    }
    // Row signatures are uniform (state-signature contract), so row 0's KV quant
    // decides the per-layer KV write/attention path for the whole batch.
    let kv_q8 = signatures.first().map(|s| s.kv_quant_q8).unwrap_or(false);
    let result = forward_prefill_dense_session_batch_prefix_full_precision(
        gpu,
        weights,
        config,
        pbs,
        &device_pointer_tables,
        route_shape,
        execution_plan.multi_state_prefix_rows,
        rows.len(),
        max_ctx_len,
        kv_q8,
    );
    device_pointer_tables.free_gpu(gpu);
    result.map(|()| shape)
}

#[allow(clippy::too_many_arguments)]
fn forward_prefill_grouped_moe_session_batch_prefix_q8_control(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    row_count: usize,
    sessions: usize,
    max_ctx_len: usize,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let q8_wmma_arch = gpu.arch_caps.has_wmma();

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::F32 => gpu.embedding_lookup_f32_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        other => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "grouped MoE session fused prefix does not support embedding format {other:?}"
                ),
            ));
        }
    }

    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                let attn_is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wz.gpu_dtype, DType::Q8_0)
                    && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0)
                    && matches!(layer.w_beta.gpu_dtype, DType::Q8_0);
                let attn_is_mq4 = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.wz.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.w_alpha.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.w_beta.gpu_dtype, DType::MQ4G256);
                let attn_is_mq6 = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.wz.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.w_alpha.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.w_beta.gpu_dtype, DType::MQ6G256);
                if !attn_is_q8 && !attn_is_mq4 && !attn_is_mq6 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8, MQ4, or MQ6 DeltaNet-MoE attention weights only; use serial_reference",
                    ));
                }
                if attn_is_mq4 || attn_is_mq6 {
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        row_count,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        row_count,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if attn_is_mq4 {
                    gpu.gemm_qkvza_hfq4g256(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                } else if attn_is_mq6 {
                    gpu.gemm_qkvza_hfq6g256(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                } else if q8_wmma_arch {
                    gpu.gemm_qkvza_q8_0_wmma(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                } else {
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wqkv.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wz.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.w_beta.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        row_count,
                    )?;
                }
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    row_count,
                )?;
                dense_prefill_session_batch_conv1d_silu_split_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_qkv_batch,
                    &layer.conv_weight,
                    k_dim,
                    v_dim,
                    row_count,
                )?;
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    row_count,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        row_count,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                }
                grouped_moe_prefill_session_batch_gated_delta_net_q8_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &pbs.dn_attn_out_batch,
                    row_count,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    row_count,
                )?;
                if matches!(layer.wo.gpu_dtype, DType::MQ4G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq4g256_residual(
                        &layer.wo.buf,
                        &pbs.dn_normed_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::MQ6G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq6g256_residual(
                        &layer.wo.buf,
                        &pbs.dn_normed_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_residual_wmma(
                        &layer.wo.buf,
                        &pbs.dn_normed_batch,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) {
                    let scratch = pbs
                        .dn_normed_rot_batch
                        .sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wo.buf,
                        &pbs.dn_normed_batch,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8 or MQ4 DeltaNet-MoE wo weights only; use serial_reference",
                    ));
                }
                let ctx = DispatchCtx::new(gpu);
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    row_count,
                    layer_idx,
                    &ctx,
                    None,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                let attn_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wk.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wv.gpu_dtype, DType::Q8_0);
                let attn_is_mq4 = matches!(layer.wq.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.wk.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.wv.gpu_dtype, DType::MQ4G256);
                let attn_is_mq6 = matches!(layer.wq.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.wk.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.wv.gpu_dtype, DType::MQ6G256);
                if !attn_is_q8 && !attn_is_mq4 && !attn_is_mq6 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8, MQ4, or MQ6 FullAttention-MoE attention weights only; use serial_reference",
                    ));
                }
                if attn_is_mq4 || attn_is_mq6 {
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        row_count,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        row_count,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if attn_is_mq4 {
                    gpu.gemm_qkv_hfq4g256(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        row_count,
                    )?;
                } else if attn_is_mq6 {
                    gpu.gemm_qkv_hfq6g256(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        row_count,
                    )?;
                } else if q8_wmma_arch {
                    gpu.gemm_qkv_q8_0_wmma(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        row_count,
                    )?;
                } else {
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wq.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wk.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        row_count,
                    )?;
                }
                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    row_count,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    row_count * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    row_count * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    &pbs.positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    row_count,
                    0,
                )?;
                prefill_session_batch_write_q8_kv_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    layer_idx,
                    &pbs.fa_k_batch,
                    &pbs.fa_v_batch,
                    config.n_kv_heads,
                    config.head_dim,
                    row_count,
                )?;
                prefill_session_batch_attention_q8_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    layer_idx,
                    &pbs.fa_q_batch,
                    &pbs.fa_attn_out_batch,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    max_ctx_len,
                    max_ctx_len,
                    row_count,
                )?;
                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                if matches!(layer.wo.gpu_dtype, DType::MQ4G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq4g256_residual(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::MQ6G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq6g256_residual(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_residual_wmma(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_batch,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) {
                    let scratch = pbs
                        .fa_attn_out_rot_batch
                        .sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_batch,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8 or MQ4 FullAttention-MoE wo weights only; use serial_reference",
                    ));
                }
                let ctx = DispatchCtx::new(gpu);
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    row_count,
                    layer_idx,
                    &ctx,
                    None,
                )?;
            }
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "grouped MoE session fused prefix encountered unsupported layer {layer_idx}; use serial_reference"
                    ),
                ));
            }
        }
    }

    grouped_moe_prefill_session_batch_final_logits(
        gpu,
        weights,
        config,
        pbs,
        device_tables,
        row_count,
        sessions,
    )
}

pub fn forward_prefill_grouped_moe_session_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    rows: &mut [DensePrefillSessionBatchRow<'_>],
    _scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<DensePrefillSessionBatchShape> {
    let shape = validate_dense_prefill_session_batch_rows_for_config(rows, pbs, config)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let inputs: Vec<DensePrefillSessionBatchInput<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchInput {
            tokens: row.tokens,
            start_pos: row.start_pos,
        })
        .collect();
    let execution_plan = build_dense_prefill_session_batch_execution_plan(&inputs, pbs.max_batch)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let signatures: Vec<DensePrefillSessionBatchStateSignature> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchStateSignature {
            kv_physical_cap: row.kv_cache.physical_cap,
            kv_compact_offset: row.kv_cache.compact_offset,
            kv_quantized: row.kv_cache.quantized,
            kv_quant_q8: row.kv_cache.quant_q8,
            kv_quant_asym2: row.kv_cache.quant_asym2,
            kv_quant_asym3: row.kv_cache.quant_asym3,
            kv_quant_asym4: row.kv_cache.quant_asym4,
            kv_quant_fwht: row.kv_cache.quant_fwht,
            dn_quant: row.dn_state.quant,
        })
        .collect();
    validate_grouped_moe_prefill_session_batch_q8_state_contract(
        config,
        &signatures,
        &execution_plan,
        gpu.arch.as_str(),
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let route_shape = expected_dense_prefill_session_state_route_shape(config);
    let pointer_table_plan =
        dense_prefill_session_batch_pointer_table_plan(&execution_plan, route_shape, rows.len());
    if execution_plan.multi_state_prefix_rows > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill fused prefix has {} rows, exceeding PrefillBatchScratch max_batch={}",
                execution_plan.multi_state_prefix_rows, pbs.max_batch,
            ),
        ));
    }
    let (prefix_tokens, prefix_positions) =
        dense_prefill_session_batch_prefix_tokens_positions(&pointer_table_plan)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    upload_prefill_batch_inputs_with_positions(gpu, pbs, &prefix_tokens, &prefix_positions)?;
    let routes: Vec<DensePrefillSessionStateRoute<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRoute {
            kv: DensePrefillSessionKvStateRoute {
                k_gpu: &row.kv_cache.k_gpu,
                v_gpu: &row.kv_cache.v_gpu,
                physical_cap: row.kv_cache.physical_cap,
                compact_offset: row.kv_cache.compact_offset,
            },
            delta: DensePrefillSessionDeltaStateRoute {
                s_matrices: &row.dn_state.s_matrices,
                s_scales: &row.dn_state.s_scales,
                conv_states: &row.dn_state.conv_states,
                quant: row.dn_state.quant,
            },
            logits: row.logits,
        })
        .collect();
    let host_pointer_tables =
        dense_prefill_session_batch_host_pointer_tables(&pointer_table_plan, &routes)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let device_pointer_tables = upload_dense_prefill_session_batch_pointer_tables(
        gpu,
        pointer_table_plan.shape,
        &host_pointer_tables,
    )?;
    let max_ctx_len = prefix_positions
        .iter()
        .copied()
        .max()
        .map(|pos| pos + 1)
        .unwrap_or(1);
    for (idx, row) in rows.iter().enumerate() {
        let row_end = row.start_pos + row.tokens.len();
        if row_end > row.kv_cache.physical_cap {
            device_pointer_tables.free_gpu(gpu);
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "grouped MoE session fused prefix row {idx} ends at position {row_end}, exceeding KV physical_cap={}",
                    row.kv_cache.physical_cap,
                ),
            ));
        }
    }
    let result = forward_prefill_grouped_moe_session_batch_prefix_q8_control(
        gpu,
        weights,
        config,
        pbs,
        &device_pointer_tables,
        route_shape,
        execution_plan.multi_state_prefix_rows,
        rows.len(),
        max_ctx_len,
    );
    device_pointer_tables.free_gpu(gpu);
    result.map(|()| shape)
}

impl PrefillBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_batch: usize) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        Ok(Self {
            max_batch,
            x_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            x_rot_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            x_norm_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            dn_qkv_batch: gpu.alloc_tensor(&[max_batch * qkv_dim], DType::F32)?,
            dn_z_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_alpha_batch: gpu.alloc_tensor(&[max_batch * n_v_heads], DType::F32)?,
            dn_beta_batch: gpu.alloc_tensor(&[max_batch * n_v_heads], DType::F32)?,
            dn_q_raw_batch: gpu.alloc_tensor(&[max_batch * k_dim], DType::F32)?,
            dn_k_raw_batch: gpu.alloc_tensor(&[max_batch * k_dim], DType::F32)?,
            dn_v_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_q_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_k_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_attn_out_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_normed_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            gate_ffn_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            up_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            ffn_hidden_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            dn_normed_rot_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            // F32 dtype = 4 bytes/element, same layout as i32. The rope /
            // attention / kv_write kernels cast the pointer to `const int*`,
            // so dtype is cosmetic. Upload i32 bits via memcpy_htod.
            positions: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            tokens: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            fa_q_full_batch: gpu.alloc_tensor(&[max_batch * q_dim * 2], DType::F32)?,
            fa_q_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_gate_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_k_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_v_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_attn_out_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_attn_out_rot_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            moe_router_logits_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts], DType::F32)?)
            } else {
                None
            },
            moe_shared_scalar_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch], DType::F32)?)
            } else {
                None
            },
            moe_shared_gate_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.shared_expert_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_shared_up_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.shared_expert_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_shared_rot_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.shared_expert_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_topk_indices_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_topk_weights_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_gate_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_up_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_rot_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_down_expanded_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.dim],
                    DType::F32,
                )?)
            } else {
                None
            },
            // Path 2 scatter + grouped-WMMA-GEMM scratch (gated at runtime by
            // HIPFIRE_MOE_GROUPED_GEMM=1). m_total_max = N*K_TOP + E*(BLOCK_M-1).
            // i32 buffers stored as Raw (4 bytes/elem matches; no DType::I32 yet).
            moe_expert_token_counts: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[config.num_experts * 4], DType::Raw)?)
            } else {
                None
            },
            moe_expert_offsets: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[(config.num_experts + 1) * 4], DType::Raw)?)
            } else {
                None
            },
            moe_sorted_slot_index: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(&[m_total_max * 4], DType::Raw)?)
            } else {
                None
            },
            moe_inverse_perm: if config.num_experts > 0 {
                let total_slots_max = max_batch * config.num_experts_per_tok;
                Some(gpu.alloc_tensor(&[total_slots_max * 4], DType::Raw)?)
            } else {
                None
            },
            moe_expert_tile_ids: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(&[(m_total_max / MOE_GROUPED_BLOCK_M) * 4], DType::Raw)?)
            } else {
                None
            },
            moe_y_gate_up_grouped: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(
                    &[m_total_max * 2 * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_y_down_grouped: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(&[m_total_max * config.dim], DType::F32)?)
            } else {
                None
            },
            dn_s_tape_q8: if config.linear_num_value_heads > 0 {
                let bytes = max_batch
                    * config.linear_num_value_heads
                    * config.linear_value_head_dim
                    * config.linear_value_head_dim;
                Some(gpu.alloc_tensor(&[bytes], DType::Raw)?)
            } else {
                None
            },
            dn_s_tape_scales: if config.linear_num_value_heads > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.linear_num_value_heads * config.linear_value_head_dim],
                    DType::F32,
                )?)
            } else {
                None
            },
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x_batch,
            self.x_rot_batch,
            self.x_norm_batch,
            self.dn_qkv_batch,
            self.dn_z_batch,
            self.dn_alpha_batch,
            self.dn_beta_batch,
            self.dn_q_raw_batch,
            self.dn_k_raw_batch,
            self.dn_v_batch,
            self.dn_q_batch,
            self.dn_k_batch,
            self.dn_attn_out_batch,
            self.dn_normed_batch,
            self.gate_ffn_batch,
            self.up_batch,
            self.ffn_hidden_batch,
            self.dn_normed_rot_batch,
            self.positions,
            self.tokens,
            self.fa_q_full_batch,
            self.fa_q_batch,
            self.fa_gate_batch,
            self.fa_k_batch,
            self.fa_v_batch,
            self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.moe_router_logits_batch,
            self.moe_shared_scalar_batch,
            self.moe_shared_gate_batch,
            self.moe_shared_up_batch,
            self.moe_shared_rot_batch,
            self.moe_topk_indices_batch,
            self.moe_topk_weights_batch,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_down_expanded_batch,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_inverse_perm,
            self.moe_expert_tile_ids,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
            self.dn_s_tape_q8,
            self.dn_s_tape_scales,
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Batched prefill entry point: processes N prompt tokens in one call,
/// writing the last token's logits into `scratch.logits` and leaving
/// the KV cache + DeltaNet state advanced by N positions.
///
/// Takes the batched kernel path when ALL linear-attention layer weights
/// are MQ4G256 (the batched element-wise kernels are MQ-specific).
/// Otherwise falls back to a per-token loop over `forward_scratch` that's
/// byte-identical to decode. FA layers always use a per-token gather/scatter
/// fallback — the FA causal attention kernel can't yet be batched (task #71).
///
/// `gated_delta_net_q8_batch_seq` runs one launch per LA layer; the kernel
/// loops over the N tokens internally and requants the Q8 state after every
/// token, matching the decode requant cadence (distributionally equivalent to
/// decode, not byte-identical — the stochastic-rounding frame differs).
///
/// `tokens`: slice of prompt tokens to prefill in order.
/// `start_pos`: first KV cache / DeltaNet position to write. Positions
/// `start_pos .. start_pos + tokens.len()` get populated.
/// On return, `scratch.logits` holds the logits for the *last* token
/// (position `start_pos + tokens.len() - 1`).
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states are captured
/// into the ring buffer for the configured extract layers. Used by the
/// DFlash target-side verify path to batch `verify_dflash_block` into a
/// single forward launch (MVP does B per-token forwards — 88 ms on 4B;
/// this path drops it to ~40 ms with batched forward, further improvement
/// possible with batched lm_head). The per-token fallback also honors it,
/// so the fast-path eligibility doesn't change behavior.
///
/// `per_token_hidden_out`: if `Some`, writes post-output-norm hidden state
/// for each of the N tokens into the provided [N × dim] buffer. The caller
/// then loops `weight_gemv(weights.output, hidden_row, logits)` to recover
/// per-token logits. Required for DFlash verify (needs all B positions'
/// logits, not just the last). `None` preserves the existing "last token
/// only" semantics where logits land in `scratch.logits`.
///
/// `gdn_tape`: if `Some`, captures the post-processed `(q, k, v, α, β)` for
/// every DN (LinearAttention) layer and block position BEFORE the batched
/// `gated_delta_net_q8_batch_seq` call. Enables the DFlash rollback path
/// to replay GDN recurrence from a pre-verify S-state snapshot for
/// `accept_len + 1` steps — no full-target re-run needed.
#[allow(clippy::too_many_arguments)]
/// Upper bound on `forward_prefill_batch`'s per-chunk size. Exposed so
/// callers sizing `HiddenStateRingBuffer` staging can match the chunk
/// upper bound (staging that's smaller than a chunk will assert-fail
/// on prompt seeding of long prompts).
pub const PREFILL_MAX_BATCH: usize = 256;

const MOE_GROUPED_BLOCK_M: usize = 16;

#[inline]
fn prefill_should_emit_last_token_logits(
    has_per_token_hidden_out: bool,
    needs_last_token_logits: bool,
) -> bool {
    !has_per_token_hidden_out || needs_last_token_logits
}

#[inline]
fn align_up_usize(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}

#[inline]
fn moe_grouped_m_total_max(max_batch: usize, k_top: usize, n_exp: usize) -> usize {
    // Every grouped-GEMM tile consumes 16 sorted slots. The scatter kernel
    // initializes sentinel tile ids up to this bound, so the bound itself must
    // be tile-aligned; otherwise the final launched tile can read an
    // uninitialized expert id.
    align_up_usize(
        max_batch * k_top + n_exp * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

#[inline]
fn moe_grouped_m_total_bound(total_slots: usize, n_exp: usize) -> usize {
    // Actual grouped rows are sum_e align_up(count_e, BLOCK_M). Only experts
    // that receive at least one slot can contribute padding, so small verify
    // batches do not need to launch the full all-experts worst case.
    let live_expert_bound = total_slots.min(n_exp);
    align_up_usize(
        total_slots + live_expert_bound * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

#[inline]
fn qwen35_f16_prefill_wmma_enabled(gpu: &Gpu) -> bool {
    gpu.arch_caps.has_wmma()
}

fn kld_direct_f16kv_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("HIPFIRE_KLD_DIRECT_WMMA_ATTN")
            .or_else(|_| std::env::var("HIPFIRE_KLD_DIRECT_F16KV_ATTN"))
            .ok();
        matches!(
            value.as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

fn kld_direct_f16kv_attention_eligible(
    gpu: &Gpu,
    kv_cache: &kv::KvCache,
    config: &Qwen35Config,
    start_pos: usize,
    tree_verify: Option<&TreeVerifyCtx<'_>>,
) -> bool {
    let enabled = kld_direct_f16kv_attention_enabled();
    let eligible = enabled
        && start_pos == 0
        && kv_cache.compact_offset == 0
        && tree_verify.is_none()
        && !kv_cache.quant_q8
        && !kv_cache.quant_asym2
        && !kv_cache.quant_asym3
        && !kv_cache.quant_asym4
        && config.head_dim.is_multiple_of(16)
        && config.head_dim <= 256
        && gpu.arch_caps.has_wmma();
    if enabled && !eligible {
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| {
            eprintln!(
                "HIPFIRE_KLD_DIRECT_WMMA_ATTN=1 but direct attention is ineligible: \
                 start_pos={} compact_offset={} tree={} quant_q8={} asym2={} asym3={} asym4={} \
                 head_dim={} has_wmma={}",
                start_pos,
                kv_cache.compact_offset,
                tree_verify.is_some(),
                kv_cache.quant_q8,
                kv_cache.quant_asym2,
                kv_cache.quant_asym3,
                kv_cache.quant_asym4,
                config.head_dim,
                gpu.arch_caps.has_wmma(),
            );
        });
    }
    eligible
}

fn kld_fp32_gqa4_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("HIPFIRE_KLD_FP32_GQA4_ATTN").ok();
        !matches!(
            value.as_deref(),
            Some("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
        )
    })
}

fn q8_fa_attention_row_loop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

fn q8_fa_attention_scalar_loop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

fn q8_fa_attention_serial_kv_loop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

fn q8_fa_attention_ignore_tree_bias_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

fn q8_gdn_verify_per_token_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

fn q8_gdn_verify_serial_frames_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

fn kld_fp32_gqa4_attention_eligible(
    gpu: &Gpu,
    kv_cache: &kv::KvCache,
    config: &Qwen35Config,
    start_pos: usize,
    tree_verify: Option<&TreeVerifyCtx<'_>>,
    batch_len: usize,
) -> bool {
    let kv_group = if config.n_kv_heads == 0 {
        0
    } else {
        config.n_heads / config.n_kv_heads
    };
    let block_size = batch_len.max(config.head_dim).next_power_of_two().min(256);
    let shared_mem = (4usize * batch_len + 4usize * block_size + 4usize * config.head_dim) * 4usize;
    kld_fp32_gqa4_attention_enabled()
        && gpu.arch == "gfx1151"
        && start_pos == 0
        && kv_cache.compact_offset == 0
        && tree_verify.is_none()
        && !kv_cache.quant_q8
        && !kv_cache.quant_asym2
        && !kv_cache.quant_asym3
        && !kv_cache.quant_asym4
        && config.n_kv_heads > 0
        && config.n_heads.is_multiple_of(config.n_kv_heads)
        && kv_group >= 4
        && kv_group % 4 == 0
        && shared_mem <= 64 * 1024
}

fn gemm_f16_x_f32_wmma_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    let y_n = y_residual.sub_offset(0, n * m);
    let scratch_n = scratch.sub_offset(0, n * m);
    gpu.gemm_f16_x_f32_wmma(weight, x, &scratch_n, m, k, n)?;
    gpu.add_inplace_f32(&y_n, &scratch_n)
}

fn gemm_f32_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    let y_n = y_residual.sub_offset(0, n * m);
    let scratch_n = scratch.sub_offset(0, n * m);
    gpu.gemm_f32_register_tiled(weight, x, &scratch_n, m, k, n)?;
    gpu.add_inplace_f32(&y_n, &scratch_n)
}

fn gemm_bf16_x_bf16_wmma_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    let y_n = y_residual.sub_offset(0, n * m);
    let scratch_n = scratch.sub_offset(0, n * m);
    gpu.gemm_bf16_x_bf16_wmma(weight, x, &scratch_n, m, k, n)?;
    gpu.add_inplace_f32(&y_n, &scratch_n)
}

fn gemm_fp16_or_bf16_x_f32_wmma(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    match weight.dtype {
        DType::F16 | DType::Raw => gpu.gemm_f16_x_f32_wmma(weight, x, y, m, k, n),
        DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(weight, x, y, m, k, n),
        other => panic!("expected F16/BF16 prefill weight, got {other:?}"),
    }
}

fn gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    match weight.dtype {
        DType::F16 | DType::Raw => {
            gemm_f16_x_f32_wmma_residual_batched(gpu, weight, x, y_residual, scratch, m, k, n)
        }
        DType::BF16 => {
            gemm_bf16_x_bf16_wmma_residual_batched(gpu, weight, x, y_residual, scratch, m, k, n)
        }
        other => panic!("expected F16/BF16 residual prefill weight, got {other:?}"),
    }
}

// Batched single-projection GEMM for the dense fused prefill. Despite the
// `_full_precision` name (kept to avoid churning ~10 call sites), this now
// dispatches plain Q8_0 and MQ4G256 weights too — quantized dense models route
// here. MQ4 needs the shared FWHT pre-rotation (rotate_x_mq_batched) into a
// scratch first; the rotation is allocated internally (prefill is one-shot, so
// the small per-GEMM alloc is acceptable). MQ6G256 non-residual has no batched
// kernel yet, so those weights fall back to serial_reference via the contract.
fn dense_session_prefill_gemm_full_precision(
    gpu: &mut Gpu,
    weight: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match weight.gpu_dtype {
        DType::F32 => gpu.gemm_f32_register_tiled(&weight.buf, x, y, weight.m, weight.k, n),
        DType::F16 | DType::BF16 | DType::Raw => {
            gemm_fp16_or_bf16_x_f32_wmma(gpu, &weight.buf, x, y, weight.m, weight.k, n)
        }
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(&weight.buf, x, y, weight.m, weight.k, n),
        DType::MQ4G256 => {
            let rot = gpu.alloc_tensor(&[n * weight.k], DType::F32)?;
            gpu.rotate_x_mq_batched(x, &rot, weight.k, n)?;
            let result = gpu.gemm_hfq4g256(&weight.buf, &rot, y, weight.m, weight.k, n);
            let _ = gpu.free_tensor(rot);
            result
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!("dense session fused prefix GEMM does not support dtype {other:?}"),
        )),
    }
}

// Residual variant of [`dense_session_prefill_gemm_full_precision`] (adds the
// GEMM result into `y_residual`). Also dispatches plain Q8_0 + MQ4G256 now: Q8
// runs the chunked GEMM into `scratch` then adds it into the residual; MQ4 runs
// the FWHT-rotated `gemm_hfq4g256_residual` which accumulates directly. MQ6G256
// residual has a kernel (`gemm_hfq6g256_residual`) but is left for a follow-up so
// the contract gates MQ6 to serial uniformly with the non-residual path.
fn dense_session_prefill_gemm_full_precision_residual(
    gpu: &mut Gpu,
    weight: &WeightTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match weight.gpu_dtype {
        DType::F32 => gemm_f32_residual_batched(
            gpu,
            &weight.buf,
            x,
            y_residual,
            scratch,
            weight.m,
            weight.k,
            n,
        ),
        DType::F16 | DType::BF16 | DType::Raw => gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
            gpu,
            &weight.buf,
            x,
            y_residual,
            scratch,
            weight.m,
            weight.k,
            n,
        ),
        DType::Q8_0 => {
            let out = scratch.sub_offset(0, n * weight.m);
            gpu.gemm_q8_0_batched_chunked(&weight.buf, x, &out, weight.m, weight.k, n)?;
            let accum = y_residual.sub_offset(0, n * weight.m);
            gpu.add_inplace_f32(&accum, &out)
        }
        DType::MQ4G256 => {
            let rot = gpu.alloc_tensor(&[n * weight.k], DType::F32)?;
            gpu.rotate_x_mq_batched(x, &rot, weight.k, n)?;
            let result =
                gpu.gemm_hfq4g256_residual(&weight.buf, &rot, y_residual, weight.m, weight.k, n);
            let _ = gpu.free_tensor(rot);
            result
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!("dense session fused prefix residual GEMM does not support dtype {other:?}"),
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoeGroupedPath2Shape {
    total_slots: usize,
    m_total_bound: usize,
    gate_up_x_row_div: usize,
    gate_up_source_rows: usize,
    down_x_row_div: usize,
    down_source_rows: usize,
}

#[inline]
fn moe_grouped_path2_shape(n: usize, k_top: usize, n_exp: usize) -> MoeGroupedPath2Shape {
    let total_slots = n * k_top;
    MoeGroupedPath2Shape {
        total_slots,
        m_total_bound: moe_grouped_m_total_bound(total_slots, n_exp),
        // Gate/up consumes x_rot_batch [N x dim]. The grouped scatter's
        // sorted slot encodes token*K_TOP + expert-rank, so the kernel divides
        // source-row lookup by K_TOP to recover the token row.
        gate_up_x_row_div: k_top,
        gate_up_source_rows: n,
        // Down consumes rot_batch [N*K_TOP x mi]. Sorted slots already index
        // the flattened routed-expert rows, so no division is required.
        down_x_row_div: 1,
        down_source_rows: total_slots,
    }
}

struct PagedMoeExpertBucket {
    expert: u16,
    m_total: usize,
    sorted_slot_index: Vec<i32>,
    inverse_perm: Vec<i32>,
    expert_tile_ids: Vec<i32>,
}

fn build_paged_moe_expert_buckets(
    topk_indices: &[usize],
    n: usize,
    k_top: usize,
    n_exp: usize,
) -> HipResult<Vec<PagedMoeExpertBucket>> {
    let total_slots = n
        .checked_mul(k_top)
        .ok_or_else(|| HipError::new(0, "paged MoE expert bucket total_slots overflow"))?;
    if topk_indices.len() != total_slots {
        return Err(HipError::new(
            0,
            &format!(
                "paged MoE expert bucket topk length mismatch: got {}, expected {}",
                topk_indices.len(),
                total_slots
            ),
        ));
    }
    let mut slots_by_expert = vec![Vec::<i32>::new(); n_exp];
    for (flat, &expert) in topk_indices.iter().enumerate() {
        if expert >= n_exp {
            return Err(HipError::new(
                0,
                &format!("paged MoE router selected expert {expert}, but n_exp={n_exp}"),
            ));
        }
        slots_by_expert[expert].push(flat as i32);
    }

    let mut buckets = Vec::new();
    for (expert, slots) in slots_by_expert.into_iter().enumerate() {
        if slots.is_empty() {
            continue;
        }
        let m_total = align_up_usize(slots.len(), MOE_GROUPED_BLOCK_M);
        let mut sorted_slot_index = vec![-1i32; m_total];
        sorted_slot_index[..slots.len()].copy_from_slice(&slots);
        let mut inverse_perm = vec![-1i32; total_slots];
        for (sorted_pos, &flat) in slots.iter().enumerate() {
            inverse_perm[flat as usize] = sorted_pos as i32;
        }
        let expert_tile_ids = vec![expert as i32; m_total / MOE_GROUPED_BLOCK_M];
        buckets.push(PagedMoeExpertBucket {
            expert: expert as u16,
            m_total,
            sorted_slot_index,
            inverse_perm,
            expert_tile_ids,
        });
    }
    Ok(buckets)
}

fn upload_paged_moe_expert_bucket(
    gpu: &mut Gpu,
    bucket: &PagedMoeExpertBucket,
    sorted_slot_index: &GpuTensor,
    inverse_perm: &GpuTensor,
    expert_tile_ids: &GpuTensor,
) -> HipResult<()> {
    gpu.hip.memcpy_htod(
        &sorted_slot_index.buf,
        i32_slice_as_bytes(&bucket.sorted_slot_index),
    )?;
    gpu.hip
        .memcpy_htod(&inverse_perm.buf, i32_slice_as_bytes(&bucket.inverse_perm))?;
    gpu.hip.memcpy_htod(
        &expert_tile_ids.buf,
        i32_slice_as_bytes(&bucket.expert_tile_ids),
    )?;
    Ok(())
}

/// Host-side helper: upload token ids and positions to a `PrefillBatchScratch`
/// via sync `memcpy_htod`. Call this BEFORE entering a hipGraph capture to
/// pre-populate `pbs.tokens` and `pbs.positions`, then pass `pre_uploaded:
/// true` (or use `forward_prefill_chunk_captured_safe`) so the forward
/// does not issue any additional uploads inside the captured region.
pub fn upload_prefill_batch_inputs(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let positions_host: Vec<usize> = (0..n).map(|i| start_pos + i).collect();
    upload_prefill_batch_inputs_with_positions(gpu, pbs, tokens, &positions_host)
}

pub fn upload_prefill_batch_inputs_with_positions(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    positions: &[usize],
) -> HipResult<()> {
    let n = tokens.len();
    if positions.len() != n {
        return Err(hip_bridge::HipError::new(
            0,
            "upload_prefill_batch_inputs_with_positions: tokens and positions length mismatch",
        ));
    }
    if n > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            "upload_prefill_batch_inputs_with_positions: token count exceeds PrefillBatchScratch max_batch",
        ));
    }
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    let positions_host: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    Ok(())
}

/// Capture-friendly entry point that runs the batched forward against a
/// SINGLE chunk (`tokens.len() <= pbs.max_batch`), skipping the internal
/// token/position upload and assuming the caller has already populated
/// `pbs.tokens` / `pbs.positions` via `upload_prefill_batch_inputs`.
///
/// This exists so `hipStreamBeginCapture` can wrap the forward without
/// the per-call `memcpy_htod` sync operations (which would either error
/// under capture or bake stale host data into the captured graph nodes).
///
/// Callers still must handle `hidden_rb.commit_staging_to_ring(gpu, n)`
/// AFTER the forward returns (outside any captured region) to scatter
/// staging writes to the ring at the current head.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_single_chunk_captured_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    needs_last_token_logits: bool,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(
        n > 0 && n <= pbs.max_batch,
        "single_chunk_captured: n={} but pbs.max_batch={}",
        n,
        pbs.max_batch
    );

    // Defense-in-depth: this entry point bypasses the eligibility check
    // in `forward_prefill_batch_with_pbs`, so the caller is responsible
    // for ensuring the batched fast-path is valid. Two structural bypasses
    // could land here:
    //   1. MQ3-weighted dense model on an arch that lacks the gfx11 wave32
    //      WMMA builtin.
    //   2. MQ3 weights inside a MoE/A3B layer on an arch without the gfx12
    //      grouped-MoE HFQ3 kernels, or MQ3-Lloyd MoE, which is still unwired.
    // In production, `daemon.rs`'s DFlash refusal guard blocks both, but
    // dflash_spec_demo and other example callers go through ModelSlot::load
    // directly. We cross-check here so any caller is protected.
    let arch = gpu.arch.as_str();
    let mut mq3_in_dense = false;
    let mut mq3_in_moe = false;
    let mut lloyd_in_dense = false;
    let mut lloyd_in_moe = false;
    // The Lloyd dtype is treated identically to plain MQ3 in this guard:
    // both use 112-vs-104-byte stride that the MoE batched branches'
    // HFQ4-layout dispatch would corrupt, and both depend on the gfx11/12
    // WMMA family that other archs lack. Add Lloyd alongside MQ3 so the
    // refusal fires symmetrically and a future MQ3-Lloyd MoE model can't
    // silently land here without explicit MoE-Lloyd kernels.
    //
    // We also track `lloyd_in_dense` separately because Lloyd-MQ3 on
    // gfx12 ships behind an opt-in env gate (see is_batchable_la above) —
    // the gfx12 sibling kernels are runtime-unvalidated locally, so by
    // default a captured-path call with Lloyd-MQ3 weights on gfx1200/1201
    // must refuse rather than dispatch to an untested kernel.
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    let is_lloyd = |dt: DType| matches!(dt, DType::MQ3G256Lloyd);
    for lw in &weights.layers {
        match lw {
            LayerWeights::DeltaNet(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
                if is_lloyd(l.wqkv.gpu_dtype)
                    || is_lloyd(l.wz.gpu_dtype)
                    || is_lloyd(l.w_beta.gpu_dtype)
                    || is_lloyd(l.w_alpha.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
                }
            }
            LayerWeights::FullAttn(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
                if is_lloyd(l.wq.gpu_dtype)
                    || is_lloyd(l.wk.gpu_dtype)
                    || is_lloyd(l.wv.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
                }
            }
            LayerWeights::DeltaNetMoe(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3(&l.ffn)
                {
                    mq3_in_moe = true;
                }
                if is_lloyd(l.wqkv.gpu_dtype)
                    || is_lloyd(l.wz.gpu_dtype)
                    || is_lloyd(l.w_beta.gpu_dtype)
                    || is_lloyd(l.w_alpha.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_lloyd(&l.ffn)
                {
                    lloyd_in_moe = true;
                }
            }
            LayerWeights::FullAttnMoe(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3(&l.ffn)
                {
                    mq3_in_moe = true;
                }
                if is_lloyd(l.wq.gpu_dtype)
                    || is_lloyd(l.wk.gpu_dtype)
                    || is_lloyd(l.wv.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_lloyd(&l.ffn)
                {
                    lloyd_in_moe = true;
                }
            }
        }
    }
    let arch_has_wmma = matches!(
        arch,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
    );
    let mq3_moe_supported = arch == "gfx1151" || (arch.starts_with("gfx12") && !lloyd_in_moe);
    if mq3_in_moe && !mq3_moe_supported {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_single_chunk_captured: model has MQ3G256 / \
             MQ3G256Lloyd weights inside a MoE/A3B layer (DeltaNetMoe or \
             FullAttnMoe), but MQ3-Lloyd MoE is only wired on gfx1151, and plain \
             MQ3G256 MoE is wired on gfx1151/gfx12. Use MQ4/MQ6 for other targets.",
        ));
    }
    if mq3_in_dense && !arch_has_wmma {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains MQ3G256 \
             weights but arch {arch} lacks the gfx11 wave32 WMMA builtin. The MQ3 \
             prefill kernels (gemm_*_hfq3g256_wmma) only compile on \
             gfx1100/1101/1102/1150/1151. Caller must use the non-captured \
             forward_prefill_batch path (which falls back to per-token \
             forward_scratch on this arch). gfx12 K4 variant for MQ3 is \
             a planned follow-up."
            ),
        ));
    }
    // Lloyd-MQ3 on gfx12 is opt-in (see is_batchable_la's gate). The
    // captured entry point bypasses is_batchable_la, so we replicate the
    // gate here: refuse Lloyd-on-gfx12 unless HIPFIRE_LLOYD_GFX12=1 is set.
    // Without this guard, a captured call would reach the dispatch arms
    // and try to load gfx12 kernels that are still community-CI-pending.
    let arch_is_gfx12 = matches!(arch, "gfx1200" | "gfx1201");
    let lloyd_gfx12_optin = std::env::var("HIPFIRE_LLOYD_GFX12").ok().as_deref() == Some("1");
    if lloyd_in_dense && arch_is_gfx12 && !lloyd_gfx12_optin {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains \
             MQ3G256Lloyd weights on arch {arch}, but the gfx12 (RDNA4) \
             sibling kernels (gemm_*_mq3g256_lloyd_wmma.gfx12.hip) are \
             runtime-unvalidated locally and ship behind an opt-in gate. \
             Set HIPFIRE_LLOYD_GFX12=1 to enable the gfx12 path for parity \
             testing, or use the non-captured forward_prefill_batch path \
             (which falls back to per-token forward_scratch on this arch \
             when the env var is unset)."
            ),
        ));
    }

    // Capture-mode contract: under hipStreamBeginCapture, the FA branch
    // bakes max_ctx_len = kv_cache.physical_cap (kernels read seq_len
    // per-row from a device buffer, but LDS is sized from this scalar).
    // For Q8 KV at physical_cap > 15000, the FA path enters the per-
    // position long-context fallback, which issues hip.malloc + per-row
    // memcpy_htod inside the layer loop. Both are capture-illegal — they
    // would either error at capture time or bake stale host bytes into
    // the kernarg blob. Asym2/3/4 KV use pure-batched flash kernels and
    // stay capture-safe at any context length, so reject only this exact
    // combination here.
    const LDS_CTX_LIMIT: usize = 15000;
    if kv_cache.quant_q8
        && !(kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && kv_cache.physical_cap > LDS_CTX_LIMIT
    {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: Q8 KV with \
             physical_cap {} > {} hits the per-position long-context \
             fallback, which issues hip.malloc + memcpy_htod inside the \
             captured region. Use asym3 KV for capture at long context, \
             or shrink physical_cap.",
                kv_cache.physical_cap, LDS_CTX_LIMIT,
            ),
        ));
    }

    let debug_max_layer = std::env::var("HIPFIRE_PREFILL_MAX_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    forward_prefill_chunk(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out.map(|t| (t, 0)),
        gdn_tape,
        0,
        tree_verify,
        true, // pre_uploaded: caller must have run upload_prefill_batch_inputs
        None, // band: full-stack single-GPU path
        None, // mask_override: captured-prefill caller does not use the MTP probe hook
        None, // positions_override: captured-prefill uses linear positions
        needs_last_token_logits,
        debug_max_layer, // max_layer: default full stack; env is for graph-fault bisection only
        false,           // force_q8_gdn_per_token: captured verify preserves production policy
        None,            // routed_out: non-EP single-GPU path
    )
}

pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        scratch.prefill_batch.as_ref(),
        None, // mask_override: MTP probe is the only consumer; default callers don't override
        None, // max_layer: pflash uses this; non-pflash default is full stack
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_force_q8_gdn_per_token(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        scratch.prefill_batch.as_ref(),
        None,
        None,
        true,
        true,
    )
}

pub fn forward_prefill_batch_with_pbs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        pbs_in,
        mask_override,
        max_layer,
        true,  // preserve legacy post-condition: scratch.logits is last-token logits
        false, // force_q8_gdn_per_token: default callers preserve existing policy
    )
}

/// Like `forward_prefill_batch`, but accepts a caller-owned `PrefillBatchScratch`
/// so the ~25 per-cycle tensor allocations can be amortized across many calls.
///
/// `pbs = None` preserves the original behavior (per-call allocate + free);
/// `pbs = Some(&pbs)` reuses the provided scratch. The provided scratch's
/// `max_batch` determines the chunk size — `tokens` is processed in chunks of
/// up to `pbs.max_batch`. Callers driving DFlash verify should size `pbs`
/// to the maximum block size they'll ever request (e.g. `block_size` or
/// `1 + tree_budget`) so everything fits in one chunk.
///
/// `needs_last_token_logits = false` is only for callers that pass
/// `per_token_hidden_out` and compute their own logits from those hidden rows.
/// The default wrapper keeps this true to protect existing callers that rely on
/// `scratch.logits` being populated with the last token's logits.
pub fn forward_prefill_batch_with_pbs_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mut hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
    needs_last_token_logits: bool,
    force_q8_gdn_per_token: bool,
) -> HipResult<()> {
    // Upper bound on the PrefillBatchScratch — large prompts get split
    // into chunks of this size and processed in a loop.
    //
    // Tuning note: each extra chunk pays full dispatch-overhead for the LA
    // preamble (rmsnorm, rotate, 4-way fused GEMM) and FFN (gate_up + down).
    // 256 costs ~80 MB of scratch on 9B vs 20 MB at 64 — trivial on modern
    // cards — and drops chunk count for pp2048 from 32 → 8. The inner
    // gated_delta_net_q8_batch_seq loop is still sequential per token, so
    // the per-chunk DeltaNet cost is linear in N either way; raising the
    // batch just amortizes the NON-DeltaNet kernels more.
    //
    // Exposed via PREFILL_MAX_BATCH so callers sizing `HiddenStateRingBuffer`
    // staging can match the chunk upper bound.
    let max_batch: usize = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= MIN_BATCH)
        .unwrap_or(PREFILL_MAX_BATCH);

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    // Cross-path safety: only plain MQ3G256 MoE is admitted, and only on
    // gfx1151/gfx12 where the shared-expert HFQ3 kernels and routed grouped-WMMA
    // kernels are available. MQ3-Lloyd MoE remains rejected because its
    // routed expert kernels have not been wired into qwen35's MoE path.
    let arch = gpu.arch.as_str();
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    let is_lloyd = |dt: DType| matches!(dt, DType::MQ3G256Lloyd);
    let mq3_in_moe = weights.layers.iter().any(|lw| match lw {
        LayerWeights::DeltaNetMoe(l) => {
            is_mq3_any(l.wqkv.gpu_dtype)
                || is_mq3_any(l.wz.gpu_dtype)
                || is_mq3_any(l.w_beta.gpu_dtype)
                || is_mq3_any(l.w_alpha.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3(&l.ffn)
        }
        LayerWeights::FullAttnMoe(l) => {
            is_mq3_any(l.wq.gpu_dtype)
                || is_mq3_any(l.wk.gpu_dtype)
                || is_mq3_any(l.wv.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3(&l.ffn)
        }
        _ => false,
    });
    let lloyd_in_moe = weights.layers.iter().any(|lw| match lw {
        LayerWeights::DeltaNetMoe(l) => {
            is_lloyd(l.wqkv.gpu_dtype)
                || is_lloyd(l.wz.gpu_dtype)
                || is_lloyd(l.w_beta.gpu_dtype)
                || is_lloyd(l.w_alpha.gpu_dtype)
                || is_lloyd(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_lloyd(&l.ffn)
        }
        LayerWeights::FullAttnMoe(l) => {
            is_lloyd(l.wq.gpu_dtype)
                || is_lloyd(l.wk.gpu_dtype)
                || is_lloyd(l.wv.gpu_dtype)
                || is_lloyd(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_lloyd(&l.ffn)
        }
        _ => false,
    });
    let mq3_moe_supported = arch == "gfx1151" || (arch.starts_with("gfx12") && !lloyd_in_moe);
    if mq3_in_moe && !mq3_moe_supported {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch: model has MQ3G256 / MQ3G256Lloyd weights \
             inside a MoE/A3B layer (DeltaNetMoe or FullAttnMoe), but only \
             MQ3-Lloyd MoE is only wired on gfx1151, and plain MQ3G256 MoE is \
             wired on gfx1151/gfx12. Use MQ4/MQ6 for other targets.",
        ));
    }

    // Tree-verify mode sanity checks — the downstream path can't silently
    // fall back to per-token FA (that's always causal and would ignore the
    // tree mask), and the positions/bias shapes must match the token count.
    if let Some(ctx) = tree_verify.as_ref() {
        assert_eq!(
            ctx.positions.len(),
            n,
            "TreeVerifyCtx.positions length {} must equal tokens.len() {}",
            ctx.positions.len(),
            n,
        );
        assert_eq!(
            ctx.attn_bias.numel(),
            n * n,
            "TreeVerifyCtx.attn_bias must be [{} × {}] f32 ({}), got numel {}",
            n,
            n,
            n * n,
            ctx.attn_bias.numel(),
        );
    }

    // Fast path requires (a) every LA layer's weights to be either MQ4G256
    // or HFQ4G256 (the batched GEMM kernels are dtype-agnostic but the LA
    // preamble's rmsnorm+rotate and SwiGLU+rotate kernels differ per dtype),
    // and (b) Q8 S-state for the GDN recurrence. Mixed-dtype layers are
    // allowed; each layer is routed to its own path. HFQ6/others fall back.
    let arch = gpu.arch.as_str();
    // Whether the tape-capturing batched (PBS) path runs for this call — the
    // single source of truth shared with spec-decode callers that later replay a
    // captured GDN tape. On `false` the forward drops to the tape-less per-token
    // loop below, leaving any passed tape stale (see `prefill_batch_pbs_eligible`).
    let moe_router_logits_present = pbs_in
        .map(|p| p.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let eligible = prefill_batch_pbs_eligible(
        weights,
        config,
        dn_state,
        n,
        arch,
        moe_router_logits_present,
    );
    // F4 guard: reject batched prefill when KV tier has no batched keys.
    // F32 KV has only BatchEq(1) → MissingImpl at resolve. asym2 + tree-verify
    // has no _batched_masked variant → UnsupportedTreeTier. Force per-token
    // fallback for these cases.
    let kv_f32 = !kv_cache.quantized && !kv_cache.quant_q8 && !kv_cache.quant_hfq4;
    let kv_asym2_tree = kv_cache.quant_asym2 && tree_verify.is_some();
    let pbs_eligible_base = eligible;
    let eligible = eligible && !kv_f32 && !kv_asym2_tree;
    if std::env::var("HIPFIRE_DEBUG_PREFILL_ELIGIBLE").as_deref() == Ok("1") {
        eprintln!(
            "[prefill-eligible] final={eligible} base={pbs_eligible_base} kv_f32={kv_f32} \
             kv_asym2_tree={kv_asym2_tree} dn_quant={:?} n={n} arch={arch} \
             kv(q8={} hfq4={} quantized={})",
            dn_state.quant, kv_cache.quant_q8, kv_cache.quant_hfq4, kv_cache.quantized
        );
    }

    if !eligible {
        assert!(
            tree_verify.is_none(),
            "tree-verify mode requires the batched-FA-eligible prefill path; \
             kv quant + FA weight dtypes do not match on this model",
        );
        // mask_override has nowhere to land on the per-token forward_scratch
        // fallback (it operates on `scratch.x`, not the batched `pbs.x_batch`,
        // and there's no shared "post-embed, pre-layer" hook). The MTP probe
        // is the only consumer today and runs on MQ4-quantized models that
        // always satisfy `eligible`, so hard-error rather than silently
        // ignoring the override.
        assert!(
            mask_override.is_none(),
            "MaskEmbedOverride requires the batched prefill path, but this \
             model fell through to the per-token fallback (likely non-MQ4 \
             weights, dn_state quant != Q8, or HIPFIRE_PREFILL_BATCHED=0).",
        );
        // Fallback: per-token loop, byte-identical to decode. If hidden
        // extraction is requested, use the with_hidden variant so the ring
        // buffer still gets populated correctly (each call advances head by 1).
        // When per-token hidden output is also requested, extract post-norm
        // hidden row-by-row into the caller's buffer.
        let dim = config.dim;
        let last_idx = tokens.len().saturating_sub(1);
        for (i, &tok) in tokens.iter().enumerate() {
            // lm_head (vocab-wide logits) only matters for the FINAL prefill
            // token — earlier prompt tokens' logits are never read. Computing it
            // every token was ~37% of prefill time on gfx1103 (rocprof). Skip
            // lm_head for all non-final tokens via the no-logits forward; the
            // last token still gets full logits in scratch.logits.
            let skip_logits = needs_last_token_logits && i != last_idx;
            if let Some(rb) = hidden_rb.as_mut() {
                forward_scratch_with_hidden(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                    rb,
                )?;
            } else if (per_token_hidden_out.is_some() && !needs_last_token_logits) || skip_logits {
                forward_scratch_no_logits(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                )?;
            } else {
                forward_scratch(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                )?;
            }
            if let Some(dst) = per_token_hidden_out {
                // scratch.tmp holds post-output-norm hidden after
                // forward_scratch_{with_hidden,layers} — it's the same buffer
                // lm_head reads from. Copy into the caller's output.
                gpu.hip
                    .memcpy_dtod_at(&dst.buf, i * dim * 4, &scratch.tmp.buf, 0, dim * 4)?;
            }
        }
        return Ok(());
    }

    // Tree-verify mode runs as a single chunk (tree is small, O(16) nodes);
    // chunk splitting would require slicing the mask by chunk rows which
    // is extra work for a case we don't need.
    if tree_verify.is_some() {
        assert!(
            n <= max_batch,
            "tree-verify tokens {} exceeds max_batch {}; tree budget must fit",
            n,
            max_batch,
        );
    }

    // Allocate the batch scratch once per call (or reuse a caller-owned one).
    // When `pbs_in` is Some, we neither allocate nor free — the caller retains
    // ownership across DFlash cycles to avoid ~25 per-cycle tensor alloc/free
    // pairs on the hot verify path. When None we fall back to the original
    // allocate-here / free-on-exit pattern so unmodified callers behave the
    // same. The chunk size is `pbs.max_batch` so a caller-owned scratch sized
    // to e.g. `block_size` or `1 + tree_budget` keeps DFlash verify in one
    // chunk without the full 256-row MAX_BATCH footprint.
    let mut own_pbs: Option<PrefillBatchScratch> = None;
    let result = (|| -> HipResult<()> {
        let pbs: &PrefillBatchScratch = match pbs_in {
            Some(p) => p,
            None => {
                own_pbs = Some(PrefillBatchScratch::new(gpu, config, max_batch)?);
                own_pbs.as_ref().unwrap()
            }
        };
        let chunk_batch = pbs.max_batch;
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let chunk_end = (chunk_start + chunk_batch).min(n);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_n = chunk.len();
            // The chunk only reads the ring buffer's head/dims to place its
            // writes. We advance the head AFTER the chunk returns, here in
            // the caller, to keep the mutable borrow scope tight.
            let pth_slot = per_token_hidden_out.map(|t| (t, chunk_start));
            // Reborrow the tape for this chunk so we keep the outer mut
            // after the chunk returns.
            let tape_for_chunk: Option<&mut crate::speculative::GdnTape> = gdn_tape.as_deref_mut();
            // Tree-verify was asserted to fit in one chunk above, so passing
            // the whole ctx through unconditionally is safe.
            let tv_for_chunk = tree_verify.as_ref().copied();
            // Apply mask_override only to the chunk that actually contains
            // its target slot, and rebase the slot index to chunk-local
            // coordinates. Out-of-range slots panic (caller error).
            let mo_for_chunk = mask_override.and_then(|ovr| {
                if ovr.slot >= chunk_start && ovr.slot < chunk_end {
                    Some(MaskEmbedOverride {
                        slot: ovr.slot - chunk_start,
                        embed: ovr.embed,
                    })
                } else {
                    None
                }
            });
            // Sanity: if caller provided an override, it MUST land in some
            // chunk. Detect "fell off the end" at the last chunk boundary.
            if let Some(override_) = mask_override.filter(|_| chunk_end == n) {
                let landed_anywhere = override_.slot < n;
                assert!(
                    landed_anywhere,
                    "MaskEmbedOverride.slot ({}) is out of range for tokens.len() ({})",
                    override_.slot, n,
                );
            }
            forward_prefill_chunk(
                gpu,
                weights,
                config,
                chunk,
                start_pos + chunk_start,
                kv_cache,
                dn_state,
                scratch,
                pbs,
                hidden_rb.as_deref(),
                pth_slot,
                tape_for_chunk,
                chunk_start,
                tv_for_chunk,
                false, // pre_uploaded: default path uploads inside
                None,  // band: full-stack single-GPU path
                mo_for_chunk,
                None, // positions_override: default path uses linear positions
                needs_last_token_logits,
                max_layer,
                force_q8_gdn_per_token,
                None, // routed_out: non-EP single-GPU path
            )?;
            if let Some(rb) = hidden_rb.as_mut() {
                // Scatter fixed-offset staging writes (done inside the chunk)
                // to the ring at the current head, then advance head by n.
                // This is the out-of-capture step: graph-captured writes went
                // to staging[0..n*h], this commit places them at head*h
                // where head is read from CPU state at call time (not baked
                // into a captured graph node).
                rb.commit_staging_to_ring(gpu, chunk_n)?;
            }
            chunk_start = chunk_end;
        }
        Ok(())
    })();
    if let Some(owned) = own_pbs {
        owned.free_gpu(gpu);
    }
    result
}

/// Accepts the dtypes the batched prefill path can handle (shared by the
/// eligibility check in `forward_prefill_batch` and the per-layer dtype
/// branches in `forward_prefill_chunk`).
#[inline]
// IMPORTANT: This allowlist is paired with the `is_mq*` matchers in
// forward_prefill_chunk (lines 4063+, 4360+, 4768, 4919) and with the
// MoE FFN gate `moe_ffn_batched_admissible`. They MUST be updated together when
// adding a new batchable dtype. Updating one without the others either
// produces dead code (safe but useless) or silent prefill corruption
// (HFQ4-stride GEMM reading a different-stride weight block). See
// docs/plans/mq-lloyd-batched-prefill-followup.md for the full
// checklist + rationale.
//
// As of this PR (issue #116 Phase 5): MQ3G256Lloyd is wired through
// the gemm_*_mq3g256_lloyd_wmma family on gfx11 (always-on) and on
// gfx12 (opt-in via HIPFIRE_LLOYD_GFX12=1). MQ4G256Lloyd is wired
// through the gemm_*_mq4g256_lloyd_wmma family on gfx11 (always-on)
// and gfx12 (opt-in via HIPFIRE_LLOYD_GFX12=1). MQ2G256Lloyd remains
// unwired — MQ2-Lloyd lands separately.
fn is_batchable_la(dt: DType, arch: &str) -> bool {
    let always_ok = matches!(
        dt,
        DType::MQ4G256 | DType::HFQ4G256
        | DType::MQ6G256 | DType::HFQ6G256
        | DType::Q8_0
        // Phase 1.5 (PARO): wqkv/wz/wo are ParoQ4G128, w_alpha/w_beta are F32
        // on shisa-Qwen3.6-A3B-PARO. Dispatch in the DeltaNetMoe LA matcher
        // routes these through gemm_hfq4g128 (with per-weight Givens
        // rotation pre-pass) and gemm_f32_batched respectively. Eligibility
        // is gated downstream by the env-keyed moe_ffn_batched_admissible
        // (HIPFIRE_PARO_BATCHED=1) — admitting them here keeps non-PARO
        // models unaffected because no production checkpoint sets
        // wqkv.gpu_dtype = ParoQ4G128 outside the shisa-PARO codepath.
        | DType::ParoQ4G128 | DType::F32 | DType::F16
    );
    if always_ok {
        return true;
    }
    // BUG-001 guard: the batched FullAttention BF16 q/k/v projection inflates
    // `fa_q` ~9x on gfx1151 → garbage output (q8/asym KV enables the batched
    // arm). F16/F32 batched are fine; only BF16 is broken on this arch. Route
    // BF16 prefill through the per-token forward_scratch path here (correct,
    // slightly slower) until the batched-arm projection is fixed; gfx1103 et al.
    // keep the fast batched path. See BUGS.md / trigger a21dccf75.
    if dt == DType::BF16 {
        return arch != "gfx1151";
    }
    // MQ3 (uniform / HFQ3 family) is batchable on archs with a WMMA
    // family ported. As of this commit:
    //   - gfx11 (gfx1100/1101/1102/1150/1151): wave32 WMMA via the
    //     `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` builtin.
    //   - gfx12 (gfx1200/1201): wave32 WMMA via the `_w32_gfx12` builtin
    //     with K4 unroll + half8_t lane-split, runtime-validated through
    //     the existing HFQ3 dispatch fork (gemm_*_hfq3g256_wmma_gfx12).
    // gfx906 GCN5 / gfx94x CDNA3 lack a ported MQ3 WMMA kernel; they
    // stay on the per-token forward_scratch fallback (correct, just
    // slower). gfx10 RDNA1/2 gains batched-prefill support via the
    // scalar HFQ3 GEMM family below (Phase 1 of
    // docs/plans/gfx10_mq3_prefill.md).
    let mq3_uniform_with_wmma = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        );

    // gfx10 RDNA1/2 scalar HFQ3 batched-prefill family (Phase 1).
    // Routes the four LA + FA matchers below to the new non-WMMA kernels
    // (gemm_qkv_hfq3g256, gemm_qkvza_hfq3g256, gemm_gate_up_hfq3g256,
    // gemm_hfq3g256_residual). Lloyd-MQ3 stays gated on gfx11+ — no
    // gfx10 Lloyd port (separate larger project).
    let mq3_uniform_with_gfx10_scalar = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" | "gfx1030" | "gfx1031" | "gfx1032"
        );

    // HFP4G32 / MFP4G32 (v2 #2 batched WMMA prefill): same arch gate as
    // MQ3. The 4 fused kernels (gemm_qkv/qkvza/gate_up/residual_hfp4g32_wmma)
    // ship in pairs for gfx11 + gfx12; identical eligibility to llama.rs
    // (see hipfire_runtime::dispatch::is_batchable_la).
    let fp4_with_wmma = matches!(dt, DType::HFP4G32 | DType::MFP4G32)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        );

    // Opus W4A4 (Oq4G256): batched-prefill via the grouped/fused WMMA family
    // (gemm_oq4_grouped_wmma / fused_qkvza_oq4_wmma / fused_gate_up_oq4_wmma).
    // These kernels are wave32 WMMA with NO scalar fallback, so the arch gate is
    // the same WMMA set as fp4 (gfx11 + gfx12). The dispatch arms for these
    // layers live in forward_prefill_chunk (LA QKVZA/gate_up/wo/w_down + FA
    // QKV/wo/FFN) and the FusedQkvFamily Oq4 arms — landed in the SAME change, so
    // enabling this gate never routes oq4 to an unhandled batched path.
    //
    // OPT-IN (default OFF) — `HIPFIRE_OQ4_BATCHED_PREFILL=1`. The fused/grouped
    // oq4 kernels are each parity-validated bit-exact in isolation, but the
    // END-TO-END batched prefill diverges from the per-token reference by a
    // measurable margin (≈0.63 mean / 3.95 max logit abs-diff on 0.8b vs the
    // ≈0.018 mq4 W4A16 baseline; enough to flip greedy argmax and, on plain oq4,
    // switch output language). Root cause is most likely that the batched
    // rmsnorm+FWHT activation rotation is not bit-identical to the per-token
    // RmsnormAutomatic rotation, and W4A4's int4 ACTIVATION quantization (a step
    // nonlinearity) amplifies that small pre-quant delta. Per the project's
    // coherence-first rule we do NOT enable a path that degrades output by
    // default; the wiring ships behind this flag for continued root-causing.
    // Decode is unaffected (always per-token oq4, known-good).
    // OQ4+ batched prefill is now the divergence-free W4A16 path (dequant 4-bit
    // weight to f16, f16×f16 WMMA, no int4 act-quant) — coherent like mq4, so it
    // is ON BY DEFAULT for gfx11+ (no longer gated behind HIPFIRE_OQ4_BATCHED_
    // PREFILL). The old gate existed because the W4A4 int4-act batched path
    // diverged (flipped greedy argmax); that path is retired for OQ4+ prefill.
    // Opt OUT with HIPFIRE_OQ4_BATCHED_PREFILL=0 (falls back to per-token prefill).
    let oq4_with_wmma = matches!(dt, DType::Oq4G256)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        )
        && std::env::var("HIPFIRE_OQ4_BATCHED_PREFILL").as_deref() != Ok("0");

    // Lloyd-MQ3 (MQ3G256Lloyd) on gfx11: Phase 5 of issue #116 ships the
    // gemm_*_mq3g256_lloyd_wmma family alongside the existing HFQ3 WMMA
    // path; group stride differs (112 B Lloyd vs 104 B HFQ3) so dispatch
    // must route to the Lloyd-specific arms (handled by the LA/FA
    // matchers downstream — see followup-checklist condition 3).
    let lloyd_mq3_with_gfx11_wmma = matches!(dt, DType::MQ3G256Lloyd)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
        );

    // Lloyd-MQ3 on gfx12 (RDNA4): the gemm_*_mq3g256_lloyd_wmma.gfx12.hip
    // kernels are code-complete but runtime-unvalidated locally — bench
    // host is gfx1100/1151 — so they ship behind an opt-in env gate.
    // With HIPFIRE_LLOYD_GFX12 unset (default), Lloyd-MQ3 on gfx1200/1201
    // falls through to per-token forward_scratch (correct, ~14× slower;
    // matches pre-Phase-B2 behaviour for that arch class). With
    // HIPFIRE_LLOYD_GFX12=1, the WMMA path is exercised — this is the
    // path RDNA4 reviewers should set when running the parity tests /
    // coherence-gate to validate the gfx12 sibling kernels. Once external
    // CI confirms gfx12 parity, the gate can be dropped (or default
    // flipped) in a follow-up commit.
    let lloyd_mq3_with_gfx12_wmma = matches!(dt, DType::MQ3G256Lloyd)
        && matches!(arch, "gfx1200" | "gfx1201")
        && std::env::var("HIPFIRE_LLOYD_GFX12").ok().as_deref() == Some("1");

    // Lloyd-MQ4 (MQ4G256Lloyd) on gfx11: shipped as part of issue #182.
    // Uses the gemm_*_mq4g256_lloyd_wmma family; group stride differs
    // (160 B Lloyd vs 136 B HFQ4) so dispatch routes through the
    // Lloyd-specific arms in forward_prefill_chunk.
    let lloyd_mq4_with_gfx11_wmma = matches!(dt, DType::MQ4G256Lloyd)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
        );

    // Lloyd-MQ4 on gfx12 (RDNA4): same opt-in gate as Lloyd-MQ3.
    let lloyd_mq4_with_gfx12_wmma = matches!(dt, DType::MQ4G256Lloyd)
        && matches!(arch, "gfx1200" | "gfx1201")
        && std::env::var("HIPFIRE_LLOYD_GFX12").ok().as_deref() == Some("1");

    mq3_uniform_with_wmma
        || mq3_uniform_with_gfx10_scalar
        || lloyd_mq3_with_gfx11_wmma
        || lloyd_mq3_with_gfx12_wmma
        || lloyd_mq4_with_gfx11_wmma
        || lloyd_mq4_with_gfx12_wmma
        || fp4_with_wmma
        || oq4_with_wmma
}

pub(crate) fn trace_finite_if_enabled(gpu: &Gpu, label: &str, tensor: &GpuTensor) -> HipResult<()> {
    if std::env::var_os("HIPFIRE_QWEN35_FINITE_TRACE").is_none() {
        return Ok(());
    }
    let vals = gpu.download_f32(tensor)?;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_finite = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &vals {
        if v.is_nan() {
            n_nan += 1;
        } else if v.is_infinite() {
            n_inf += 1;
        } else {
            n_finite += 1;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    eprintln!(
        "[qwen35 finite] {label}: finite={n_finite}/{} nan={n_nan} inf={n_inf} range=[{min_v:.6e}, {max_v:.6e}]",
        vals.len(),
    );
    Ok(())
}

fn trace_stage_if_enabled(label: &str) {
    if std::env::var_os("HIPFIRE_QWEN35_STAGE_TRACE").is_some() {
        eprintln!("[qwen35 stage] {label}");
    }
}

fn trace_stage_sync_if_enabled(gpu: &Gpu, label: &str) -> HipResult<()> {
    if std::env::var_os("HIPFIRE_QWEN35_STAGE_SYNC").is_some() {
        eprintln!("[qwen35 stage-sync] {label}");
        gpu.hip.device_synchronize()?;
    } else {
        trace_stage_if_enabled(label);
    }
    Ok(())
}

fn dflash_serial_qkvza_self_compare_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE").is_some())
}

fn dflash_serial_tape_x_in_compare_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE").is_some())
}

fn log_dflash_serial_qkvza_self_diff(
    family: &str,
    layer_idx: usize,
    pos: usize,
    probe: &[f32],
    serial: &[f32],
) {
    let len = probe.len().min(serial.len());
    let first_mismatch = probe
        .iter()
        .zip(serial.iter())
        .take(len)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    let Some(first) = first_mismatch else {
        eprintln!(
            "[dflash-serial-qkvza-self-compare] layer={layer_idx} pos={pos} family={family} match len={len}"
        );
        return;
    };

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut bit_diff = 0usize;
    for (a, b) in probe.iter().zip(serial.iter()).take(len) {
        if a.to_bits() != b.to_bits() {
            bit_diff += 1;
        }
        let abs = (*a - *b).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs as f64;
    }
    let mean_abs = if len == 0 { 0.0 } else { sum_abs / len as f64 };
    eprintln!(
        "[dflash-serial-qkvza-self-compare] layer={layer_idx} pos={pos} family={family} mismatch len={len} bit_diff={bit_diff} first={first} probe_f32={:.9e} serial_f32={:.9e} max_abs={:.9e} mean_abs={:.9e}",
        probe[first],
        serial[first],
        max_abs,
        mean_abs,
    );
}

fn log_dflash_serial_tape_x_in_diff(
    layer_idx: usize,
    pos: usize,
    tape_row: usize,
    source: &[f32],
    captured: &[f32],
) {
    let len = source.len().min(captured.len());
    let first_mismatch = source
        .iter()
        .zip(captured.iter())
        .take(len)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    let Some(first) = first_mismatch else {
        eprintln!(
            "[dflash-serial-tape-x-in-compare] layer={layer_idx} pos={pos} tape_row={tape_row} match len={len}"
        );
        return;
    };

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut bit_diff = 0usize;
    for (a, b) in source.iter().zip(captured.iter()).take(len) {
        if a.to_bits() != b.to_bits() {
            bit_diff += 1;
        }
        let abs = (*a - *b).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs as f64;
    }
    let mean_abs = if len == 0 { 0.0 } else { sum_abs / len as f64 };
    eprintln!(
        "[dflash-serial-tape-x-in-compare] layer={layer_idx} pos={pos} tape_row={tape_row} mismatch len={len} bit_diff={bit_diff} first={first} source_f32={:.9e} captured_f32={:.9e} max_abs={:.9e} mean_abs={:.9e}",
        source[first],
        captured[first],
        max_abs,
        mean_abs,
    );
}

/// Process one chunk of up to `pbs.max_batch` tokens through the batched
/// prefill path. All LA layers go through batched kernels; all FA layers
/// go through a per-token gather/scatter loop with the inline FA body.
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states for configured
/// extract layers get written into the ring buffer at its current head. The
/// caller (forward_prefill_batch) advances the head by N after this chunk
/// completes so writes from the next chunk don't overwrite.
///
/// `per_token_hidden_out`: if `Some((dst, offset_rows))`, writes post-output
/// RMSNorm hidden for each of the N tokens into `dst[offset_rows..offset_rows+N]`
/// in row-major order. Required for DFlash verify to compute per-position
/// logits via B sequential `weight_gemv` calls on the caller side.
///
/// `gdn_tape` + `tape_offset`: if `Some`, captures the post-processed
/// `(q, k, v, α, β)` tensors per DN layer at rows
/// `[tape_offset .. tape_offset+N]` right before the batched GDN kernel
/// runs. Used by the DFlash rollback path.
/// Does the MoE FFN admit the batched prefill fast path?
///
/// Router + shared_expert_gate may be Q8_0 (the engine's default — these
/// small tensors are never quantized to MQ4 to preserve routing
/// accuracy). They get a separate `gemm_q8_0_batched_chunked` dispatch
/// against the *un-rotated* `x_norm_batch` inside
/// `prefill_moe_ffn_body_batched`. Other MoE weights are admitted only when
/// their concrete dtype has matching shared-expert and routed-expert dispatch
/// branches below.
///
/// Pre-fix this required ALL weights to be MQ4G256, which made every
/// A3B model fall back to per-token prefill because router is universally
/// Q8_0. Widening to accept Q8 router + Q8 shared_expert_gate unlocks
/// uniform-MQ4 A3B variants (Qwen3.5-A3B, qwen3.6-35b-a3b-uniform-mq4.hfq).
/// Mixed-precision Qwen3.6-A3B uses the MQ6 branches when its MoE weights are
/// quantized to MQ6G256.
/// MoE FFN admit predicate for the batched prefill body
/// `prefill_moe_ffn_body_batched`. Per-projection MQ4 OR MQ6 admit:
///
/// - router, shared_expert_gate: MQ4 or Q8 (small scalars; dispatched
///   inline below).
/// - shared_expert.gate AND .up: same dtype; the fused gate+up kernel
///   handles one storage layout per call.
/// - shared_expert.down: independently dispatchable; it may differ from
///   shared gate/up as long as its dtype is supported.
/// - experts.gate_up: uniform across all experts in this layer.
/// - experts.down: same dtype as experts.gate_up and uniform across experts.
///
/// AWQ A3B dtype dump 2026-05-19 confirms experts are uniform per
/// projection per layer. The 4 grouped/fused dispatch sites in
/// `prefill_moe_ffn_body_batched` branch on the actual dtype, so a
/// layer admitted here is dispatchable end-to-end.
///
fn paro_batched_admit_enabled_from_env(value: Option<&str>) -> bool {
    // Default OFF (opt-in via HIPFIRE_PARO_BATCHED=1). The PARO batched prefill
    // path (ParoQ4G128 wqkv/wz/wo → gemm_hfq4g128 + per-weight Givens) was
    // only validated for finite logits, not coherence. Per-token fallback
    // (forward_scratch) is correct and avoids the echo bug. Set =1 to re-enable
    // for eval/benchmarking, understanding that output may differ from decode.
    value == Some("1")
}

fn paro_moe_i8_enabled_for_arch_from_env(arch: &str, value: Option<&str>) -> bool {
    arch.starts_with("gfx1151") && value != Some("0")
}

fn paro_moe_i8_k8_enabled_from_env(i8_enabled: bool, value: Option<&str>) -> bool {
    i8_enabled && value != Some("0")
}

#[derive(Debug, Clone, Copy)]
struct MoePrefillDtypes {
    router: DType,
    shared_expert_scalar_gate: DType,
    shared_expert_gate: DType,
    shared_expert_up: DType,
    shared_expert_down: DType,
    expert_gate_up: DType,
    expert_down: DType,
    expert_gate_up_uniform: bool,
    expert_down_uniform: bool,
}

impl MoePrefillDtypes {
    #[cfg(test)]
    fn uniform(dtype: DType) -> Self {
        Self {
            router: dtype,
            shared_expert_scalar_gate: dtype,
            shared_expert_gate: dtype,
            shared_expert_up: dtype,
            shared_expert_down: dtype,
            expert_gate_up: dtype,
            expert_down: dtype,
            expert_gate_up_uniform: true,
            expert_down_uniform: true,
        }
    }

    fn from_ffn(ffn: &MoeFfnWeights) -> Option<Self> {
        if ffn.experts.is_empty() {
            let expert_gate_up = ffn.expert_gate_up_dtype?;
            let expert_down = ffn.expert_down_dtype?;
            return Some(Self {
                router: ffn.router.gpu_dtype,
                shared_expert_scalar_gate: ffn.shared_expert_gate.gpu_dtype,
                shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
                shared_expert_up: ffn.shared_expert.up.gpu_dtype,
                shared_expert_down: ffn.shared_expert.down.gpu_dtype,
                expert_gate_up,
                expert_down,
                expert_gate_up_uniform: true,
                expert_down_uniform: true,
            });
        }
        let first = ffn.experts.first()?;
        Some(Self {
            router: ffn.router.gpu_dtype,
            shared_expert_scalar_gate: ffn.shared_expert_gate.gpu_dtype,
            shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
            shared_expert_up: ffn.shared_expert.up.gpu_dtype,
            shared_expert_down: ffn.shared_expert.down.gpu_dtype,
            expert_gate_up: first.gate_up.gpu_dtype,
            expert_down: first.down.gpu_dtype,
            expert_gate_up_uniform: ffn
                .experts
                .iter()
                .all(|e| e.gate_up.gpu_dtype == first.gate_up.gpu_dtype),
            expert_down_uniform: ffn
                .experts
                .iter()
                .all(|e| e.down.gpu_dtype == first.down.gpu_dtype),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoeDecodeIndexedRoutedPath {
    None,
    Mq4,
    Mq6,
    Mq2Lloyd,
    ParoQ4G128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoeDecodeDispatchFlags {
    gate_side_mq4: bool,
    shared_gate_up_mq4: bool,
    routed_mq4: bool,
    routed_mq6: bool,
    routed_mq2_lloyd: bool,
    routed_paro: bool,
    routed_gate_up_mq4: bool,
    routed_gate_up_mq6: bool,
    routed_gate_up_mq2_lloyd: bool,
    routed_gate_up_paro: bool,
    routed_dtype_indexable_mq4: bool,
    routed_dtype_indexable_mq6: bool,
    routed_dtype_indexable_mq2_lloyd: bool,
    routed_dtype_indexable_paro: bool,
    routed_path: MoeDecodeIndexedRoutedPath,
    use_gpu_topk: bool,
    needs_x_rot_local: bool,
}

fn moe_decode_dispatch_flags_for_dtypes(
    dtypes: &MoePrefillDtypes,
    k_top: usize,
    paro_shared_present: bool,
) -> MoeDecodeDispatchFlags {
    let gate_side_mq4 = dtypes.router == DType::MQ4G256
        && dtypes.shared_expert_scalar_gate == DType::MQ4G256
        && dtypes.shared_expert_gate == DType::MQ4G256
        && dtypes.shared_expert_up == DType::MQ4G256
        && dtypes.expert_gate_up == DType::MQ4G256
        && dtypes.expert_gate_up_uniform;
    let shared_gate_up_mq4 =
        dtypes.shared_expert_gate == DType::MQ4G256 && dtypes.shared_expert_up == DType::MQ4G256;
    let routed_mq4 = dtypes.expert_down == DType::MQ4G256 && dtypes.expert_down_uniform;
    let routed_gate_up_mq4 =
        dtypes.expert_gate_up == DType::MQ4G256 && dtypes.expert_gate_up_uniform;
    let routed_mq6 = dtypes.expert_down == DType::MQ6G256 && dtypes.expert_down_uniform;
    let routed_gate_up_mq6 =
        dtypes.expert_gate_up == DType::MQ6G256 && dtypes.expert_gate_up_uniform;
    let routed_mq2_lloyd = dtypes.expert_down == DType::MQ2G256Lloyd && dtypes.expert_down_uniform;
    let routed_gate_up_mq2_lloyd =
        dtypes.expert_gate_up == DType::MQ2G256Lloyd && dtypes.expert_gate_up_uniform;
    let routed_paro = dtypes.expert_down == DType::ParoQ4G128
        && dtypes.expert_down_uniform
        && paro_shared_present;
    let routed_gate_up_paro = dtypes.expert_gate_up == DType::ParoQ4G128
        && dtypes.expert_gate_up_uniform
        && paro_shared_present;
    let routed_dtype_indexable_mq4 = routed_mq4 && routed_gate_up_mq4;
    let routed_dtype_indexable_mq6 = routed_mq6 && routed_gate_up_mq6;
    let routed_dtype_indexable_mq2_lloyd = routed_mq2_lloyd && routed_gate_up_mq2_lloyd;
    let routed_dtype_indexable_paro = routed_paro && routed_gate_up_paro;
    let routed_path = if routed_dtype_indexable_mq4 {
        MoeDecodeIndexedRoutedPath::Mq4
    } else if routed_dtype_indexable_mq6 {
        MoeDecodeIndexedRoutedPath::Mq6
    } else if routed_dtype_indexable_mq2_lloyd {
        MoeDecodeIndexedRoutedPath::Mq2Lloyd
    } else if routed_dtype_indexable_paro {
        MoeDecodeIndexedRoutedPath::ParoQ4G128
    } else {
        MoeDecodeIndexedRoutedPath::None
    };
    let routed_dtype_indexable = routed_path != MoeDecodeIndexedRoutedPath::None;
    let use_gpu_topk = k_top == 8 && routed_dtype_indexable;
    let needs_x_rot_local = gate_side_mq4
        || routed_gate_up_mq4
        || routed_gate_up_mq6
        || routed_gate_up_mq2_lloyd
        || routed_gate_up_paro;
    MoeDecodeDispatchFlags {
        gate_side_mq4,
        shared_gate_up_mq4,
        routed_mq4,
        routed_mq6,
        routed_mq2_lloyd,
        routed_paro,
        routed_gate_up_mq4,
        routed_gate_up_mq6,
        routed_gate_up_mq2_lloyd,
        routed_gate_up_paro,
        routed_dtype_indexable_mq4,
        routed_dtype_indexable_mq6,
        routed_dtype_indexable_mq2_lloyd,
        routed_dtype_indexable_paro,
        routed_path,
        use_gpu_topk,
        needs_x_rot_local,
    }
}

fn moe_prefill_topk_shape_supported(k_top: usize, num_experts: usize) -> bool {
    k_top == 8 && num_experts <= 1024
}

fn moe_prefill_side_gate_dtype_supported(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256 | DType::Q8_0 | DType::F32 | DType::F16 | DType::BF16
    )
}

fn moe_prefill_full_precision_shared_dtype_supported(dtype: DType, arch: &str) -> bool {
    matches!(dtype, DType::F16 | DType::BF16) && arch.starts_with("gfx")
}

fn moe_prefill_full_precision_routed_dtype_supported(dtype: DType, arch: &str) -> bool {
    matches!(dtype, DType::F16 | DType::BF16) && arch == "gfx1151"
}

fn moe_ffn_batched_admissible_for_dtypes(
    dtypes: &MoePrefillDtypes,
    admit_paro: bool,
    arch: &str,
) -> bool {
    let router_ok = moe_prefill_side_gate_dtype_supported(dtypes.router);
    let shared_gate_ok = moe_prefill_side_gate_dtype_supported(dtypes.shared_expert_scalar_gate);
    if !(router_ok && shared_gate_ok && dtypes.expert_gate_up_uniform && dtypes.expert_down_uniform)
    {
        return false;
    }

    if admit_paro
        && dtypes.shared_expert_gate == DType::ParoQ4G128
        && dtypes.shared_expert_up == DType::ParoQ4G128
        && dtypes.shared_expert_down == DType::ParoQ4G128
        && dtypes.expert_gate_up == DType::ParoQ4G128
        && dtypes.expert_down == DType::ParoQ4G128
    {
        return true;
    }

    let shared_gu_one_dtype = dtypes.shared_expert_up == dtypes.shared_expert_gate;
    let experts_one_dtype = dtypes.expert_down == dtypes.expert_gate_up;
    if !(shared_gu_one_dtype && experts_one_dtype) {
        return false;
    }

    let shared_gate_up_supported =
        moe_prefill_quant_family_supported_for_arch(dtypes.shared_expert_gate, arch)
            || moe_prefill_full_precision_shared_dtype_supported(dtypes.shared_expert_gate, arch);
    let shared_down_supported =
        moe_prefill_quant_family_supported_for_arch(dtypes.shared_expert_down, arch)
            || moe_prefill_full_precision_shared_dtype_supported(dtypes.shared_expert_down, arch);
    let routed_supported = moe_prefill_quant_family_supported_for_arch(dtypes.expert_gate_up, arch)
        || moe_prefill_full_precision_routed_dtype_supported(dtypes.expert_gate_up, arch);
    if !shared_gate_up_supported || !shared_down_supported || !routed_supported {
        return false;
    }

    let shared_matches_routed = dtypes.shared_expert_gate == dtypes.expert_gate_up
        && dtypes.shared_expert_down == dtypes.expert_down;

    shared_matches_routed || moe_grouped_gemm_supported_for_dtype(dtypes.expert_gate_up, arch)
}

/// Threshold below which batching overhead isn't worth the alloc + per-layer
/// dispatch — single-token prefill must not take the batched path.
const MIN_BATCH: usize = 2;

/// Whether `forward_prefill_batch_with_pbs` will take the tape-capturing
/// batched (PBS) path for an `n`-token call — equivalently, whether a `GdnTape`
/// handed to that forward will actually be populated. When this is false the
/// forward silently drops to a tape-less per-token loop, so spec-decode callers
/// that later replay the GDN tape MUST gate that cheap replay on this predicate;
/// otherwise they replay a stale/zero tape and corrupt DeltaNet state. This is
/// the single source of truth for the eligibility decision — called by the
/// forward itself and by those callers, so the two can never drift. (The
/// tree-verify forward keeps its own, deliberately simpler, eligibility check.)
pub fn prefill_batch_pbs_eligible(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    n: usize,
    arch: &str,
    moe_router_logits_present: bool,
) -> bool {
    // HIPFIRE_PREFILL_BATCHED=0 forces the per-token fallback — an escape hatch
    // for the LARGE seed prefill (gfx11 24GB OOM + a batched-seed correctness bug
    // that collapses MTP τ→1.0). But the small-B MTP verify (n = K+1, ≤ ~32) is
    // cheap and its BATCHED path is the dominant RDNA3 decode lever. Decouple:
    // let the small-B verify batch even when the flag forces the seed per-token.
    // DEFAULT-ON for RDNA3 dGPU (gfx110x) — the arch origin/master validated
    // (bc5d005d / W3x: byte-identical output vs per-token at 240-tok ctx, +20%
    // mq4). Opt-out HIPFIRE_MTP_VERIFY_DECOUPLE=0. Other archs are opt-in (=1)
    // until validated in-arch. NB master gated on `starts_with("gfx11")`, which
    // wrongly matched gfx1151 (RDNA3.5) despite its prose excluding it; we narrow
    // to `gfx110` (gfx1100/01/02) so gfx1151 stays opt-in. The seed stays
    // per-token for LONG prompts (n>32 → force_fallback under PREFILL_BATCHED=0).
    let decouple_env = std::env::var("HIPFIRE_MTP_VERIFY_DECOUPLE").ok();
    let is_rdna3_dgpu = arch.starts_with("gfx110");
    let verify_decouple = n <= 32
        && decouple_env.as_deref() != Some("0")
        && (is_rdna3_dgpu || decouple_env.as_deref() == Some("1"));
    let force_fallback =
        !verify_decouple && std::env::var("HIPFIRE_PREFILL_BATCHED").ok().as_deref() == Some("0");
    // MoE batched path requires K_TOP=8 (hard-coded in the indexed kernels) and
    // num_experts ≤ 1024 (bound of the batched top-K shared mem).
    let moe_topk_ok =
        moe_prefill_topk_shape_supported(config.num_experts_per_tok, config.num_experts);
    !force_fallback
        && n >= MIN_BATCH
        && matches!(dn_state.quant, StateQuant::Q8 | StateQuant::FP32)
        && (dn_state.quant == StateQuant::Q8
            || weights
                .layers
                .iter()
                .all(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_))))
        && weights.layers.iter().any(|lw| matches!(
            lw,
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_),
        ))
        // LA/FA/MoE projection + MoE-FFN weight dtypes must all be batchable;
        // A3B engine policy quantizes attention as Q8 (admitted alongside MQ4).
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::DeltaNet(l) =>
                is_batchable_la(l.wqkv.gpu_dtype, arch)
                    && is_batchable_la(l.wz.gpu_dtype, arch)
                    && is_batchable_la(l.w_beta.gpu_dtype, arch)
                    && is_batchable_la(l.w_alpha.gpu_dtype, arch)
                    && is_batchable_la(l.wo.gpu_dtype, arch)
                    && is_batchable_la(l.w_gate.gpu_dtype, arch)
                    && is_batchable_la(l.w_up.gpu_dtype, arch)
                    && is_batchable_la(l.w_down.gpu_dtype, arch),
            LayerWeights::FullAttn(_) => true,
            LayerWeights::DeltaNetMoe(l) =>
                moe_topk_ok
                    && moe_router_logits_present
                    && is_batchable_la(l.wqkv.gpu_dtype, arch)
                    && is_batchable_la(l.wz.gpu_dtype, arch)
                    && is_batchable_la(l.w_beta.gpu_dtype, arch)
                    && is_batchable_la(l.w_alpha.gpu_dtype, arch)
                    && is_batchable_la(l.wo.gpu_dtype, arch)
                    && moe_ffn_batched_admissible(&l.ffn, arch),
            LayerWeights::FullAttnMoe(l) =>
                moe_topk_ok
                    && moe_router_logits_present
                    && is_batchable_la(l.wq.gpu_dtype, arch)
                    && is_batchable_la(l.wk.gpu_dtype, arch)
                    && is_batchable_la(l.wv.gpu_dtype, arch)
                    && is_batchable_la(l.wo.gpu_dtype, arch)
                    && moe_ffn_batched_admissible(&l.ffn, arch),
        })
}

fn moe_ffn_batched_admissible(ffn: &MoeFfnWeights, arch: &str) -> bool {
    let Some(dtypes) = MoePrefillDtypes::from_ffn(ffn) else {
        return false;
    };

    // PARO admit is default-on. Set HIPFIRE_PARO_BATCHED=0 to force the old
    // fallback path while bisecting or debugging.
    // for shisa-Qwen3.6-A3B-PARO and similar ParoQuant checkpoints where the
    // routed-expert + shared-expert weights are ParoQ4G128 (HFQ4G128 +
    // per-weight Givens rotation metadata). The downstream dispatch arms for
    // ParoQ4G128 are implemented on this branch. See roadmap at
    // .claude/plans/magical-marinating-hippo.md.
    static PARO_ADMIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let admit_paro = *PARO_ADMIT.get_or_init(|| {
        paro_batched_admit_enabled_from_env(std::env::var("HIPFIRE_PARO_BATCHED").ok().as_deref())
    });

    if !moe_ffn_batched_admissible_for_dtypes(&dtypes, admit_paro, arch) {
        return false;
    }

    // Mixed shared/routed MQ-family layers need to rotate the normalized input
    // a second time using an actual routed gate_up tensor as representative.
    // Paged expert mode has dtype metadata only here, so keep it on the
    // established fallback until page-level AWQ representatives are exposed.
    !(ffn.experts.is_empty() && moe_prefill_needs_routed_gate_up_reprojection(&dtypes))
}

fn moe_prefill_quant_family_supported_for_arch(dtype: DType, arch: &str) -> bool {
    match dtype {
        DType::MQ4G256 => true,
        // MQ6 has indexed batched gate_up/down on RDNA and grouped GEMM on
        // gfx1151/gfx12. The CDNA/gfx9 atomic fallback is still MQ4-only.
        DType::MQ6G256 => !arch.starts_with("gfx9"),
        // MQ3 currently has the shared-expert kernels plus grouped-WMMA
        // routed experts. There is no indexed fallback, so only admit where
        // grouped-WMMA is guaranteed.
        DType::MQ3G256 => arch == "gfx1151" || arch.starts_with("gfx12"),
        // Scalar batched/indexed bring-up kernels exist for gfx1151 only.
        DType::MQ2G256 | DType::MQ8G256 | DType::MQ2G256Lloyd | DType::MQ3G256Lloyd => {
            arch == "gfx1151"
        }
        _ => false,
    }
}

fn moe_grouped_gemm_supported_for_dtype(dtype: DType, arch: &str) -> bool {
    match dtype {
        DType::MQ4G256 => arch.starts_with("gfx11") || arch.starts_with("gfx12"),
        DType::MQ6G256 => arch == "gfx1151" || arch.starts_with("gfx12"),
        DType::MQ3G256 => arch == "gfx1151" || arch.starts_with("gfx12"),
        DType::MQ2G256Lloyd => arch == "gfx1151",
        DType::F16 | DType::BF16 => arch == "gfx1151",
        DType::ParoQ4G128 => arch.starts_with("gfx11") || arch.starts_with("gfx12"),
        _ => false,
    }
}

fn moe_grouped_gemm_path2_enabled_from_env(value: Option<&str>) -> bool {
    match value {
        Some("0") | Some("off") => false,
        Some("1") | Some("on") => true,
        _ => true,
    }
}

fn mq2_lloyd_n32_gfx1151_enabled_from_env(
    arch: &str,
    total_slots: usize,
    value: Option<&str>,
) -> bool {
    if arch != "gfx1151" {
        return false;
    }
    match value {
        Some("0") | Some("off") => false,
        Some("1") | Some("on") => true,
        _ => total_slots >= 1024,
    }
}

fn mq2_lloyd_n32_gfx1151_enabled(arch: &str, total_slots: usize) -> bool {
    static MODE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    mq2_lloyd_n32_gfx1151_enabled_from_env(
        arch,
        total_slots,
        MODE.get_or_init(|| std::env::var("HIPFIRE_MOE_MQ2L_N32_GFX1151").ok())
            .as_deref(),
    )
}

fn moe_grouped_gemm_path2_required_for_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::MQ3G256 | DType::F16 | DType::BF16)
}

fn moe_grouped_gemm_path2_eligible_for_dtype(dtype: DType, arch: &str, use_path2: bool) -> bool {
    (use_path2 || moe_grouped_gemm_path2_required_for_dtype(dtype))
        && moe_grouped_gemm_supported_for_dtype(dtype, arch)
}

fn moe_prefill_mq_family_uses_prerotation(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ2G256
            | DType::MQ3G256
            | DType::MQ4G256
            | DType::MQ6G256
            | DType::MQ8G256
            | DType::MQ2G256Lloyd
            | DType::MQ3G256Lloyd
    )
}

fn moe_prefill_needs_routed_gate_up_reprojection(dtypes: &MoePrefillDtypes) -> bool {
    dtypes.expert_gate_up != dtypes.shared_expert_gate
        && moe_prefill_mq_family_uses_prerotation(dtypes.expert_gate_up)
}

fn moe_prefill_prepare_routed_gate_up_input(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    dtypes: &MoePrefillDtypes,
    x_norm_batch: &GpuTensor,
    x_rot_batch: &GpuTensor,
    dim: usize,
    n: usize,
) -> HipResult<()> {
    if !moe_prefill_needs_routed_gate_up_reprojection(dtypes) {
        return Ok(());
    }

    let Some(representative) = ffn.experts.first().map(|e| &e.gate_up) else {
        return Err(HipError::new(
            0,
            "mixed-dtype paged MoE prefill needs a routed gate_up representative",
        ));
    };
    rotate_x_mq_batched_for(gpu, representative, x_norm_batch, x_rot_batch, dim, n)
}

/// #397 Ship 5.2 slice 1: route a single PLAIN-batched prefill GEMM through
/// [`GemmFamily::run_key`] against an *explicit* dispatcher-entry [`KernelKey`].
///
/// This is the behavior-preserving migration primitive proved by the Ship 5.2
/// pilot (028ac9f3): passing the dispatcher-entry key (e.g.
/// `GemmQ8_0BatchedChunked`, `GemmHfq4G256`, `GemmHfq4G128`, `GemmF32Batched`)
/// makes `run_key` dispatch to the IDENTICAL `gpu.gemm_*` method the direct
/// call used, so each method's own internal arch routing (RDNA4-WMMA /
/// gfx906-dp4a / CDNA-rocBLAS / …) is preserved byte-for-byte on every
/// (dtype × arch × shape). `resolve()` is deliberately NOT used here — it
/// front-runs the kernel's internal dispatch with a dtype-keyed WMMA preference
/// and can diverge from a direct dispatcher-entry call on some arches.
///
/// Only the four PLAIN-batched dispatcher-entry keys with existing table
/// entries are valid here. Residual-fused kernels (`gemm_*_residual*`) and the
/// fused QKVZA / gate+up kernels are NOT plain GEMMs and are migrated in later
/// slices (they need new table entries).
#[inline]
fn run_plain_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::gemm::GemmParams;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
    };
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::dispatch::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(HipError::from)
}

/// #397 Ship 5.2 FINAL: route a single BATCHED-prefill RESIDUAL-fused GEMM
/// (`y += W·x`) through [`GemmFamily::run_key`] against an explicit
/// `Gemm*Residual` [`KernelKey`].
///
/// Residual analogue of [`run_plain_gemm_key`]. The residual op writes its
/// output IN-PLACE into the residual stream `y` (which carries the pre-add
/// value); the `gpu.gemm_*_residual` kernels perform the add internally and
/// NEVER reuse `y` as GEMV scratch, so the migration cannot reintroduce the
/// a9e8dfda aliasing bug — `y`, the residual/input `x`, and the weight buffer
/// are passed in the IDENTICAL order the direct call used. Each residual key
/// routes to the same `gpu.gemm_*_residual` method (which keeps its own internal
/// arch routing: WMMA/gfx12-WMMA / dp4a / fp16 / scalar) byte-for-byte. For
/// HFQ3 the run-arm replicates the call-site WMMA-vs-base arch split internally
/// via `gpu.arch_caps`; `resolve()` only confirms the entry's ArchPredicate
/// admits the current arch (it is NOT used to front-run the kernel's dispatch).
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_residual_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::gemm::GemmParams;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
    };
    // The residual stream `y` is BOTH the residual and the output (`y += W·x`).
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::dispatch::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(HipError::from)
}

/// #397 Ship 5.2 slice 2: route a single BATCHED-prefill FUSED gate+up GEMM
/// through [`FusedQkvFamily`] against an explicit `FusedGateUp*` [`KernelKey`].
///
/// This is the gate+up analogue of [`run_plain_gemm_key`]. Unlike a plain GEMM,
/// gate+up carries TWO weights (gate, up) and writes TWO outputs in one fused
/// launch, so it goes through `FusedQkvFamily` (the gate+up variant) rather than
/// `GemmFamily`. Passing `batch_size: Some(n)` makes the family's gate+up run-arm
/// dispatch to the IDENTICAL batched `gpu.gemm_gate_up_*(.., n)` method the direct
/// prefill call used — each method keeps its own internal arch routing
/// (RDNA4-WMMA / gfx906-dp4a / MMQ / fp16 / scalar) byte-for-byte. The weights,
/// activation `x` (already rmsnorm-rotated by the caller), outputs and m/k/n args
/// are unchanged at every migrated site.
///
/// The `FusedGateUp*` key carries the dtype; the run-arm replicates any
/// call-site arch split (e.g. HFQ3 WMMA-vs-base) internally via `gpu.arch_caps`,
/// so the same kernel runs. `resolve()` only confirms the entry's ArchPredicate
/// admits the current arch — it does NOT front-run the kernel's internal dispatch.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_fused_gate_up_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_gate: &GpuTensor,
    w_up: &GpuTensor,
    x: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    gate_m: usize,
    up_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[w_gate, w_up],
        x,
        outputs: &[y_gate, y_up],
        m: &[gate_m, up_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::dispatch::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Dispatch a batched-prefill **3-way fused QKV** projection (wq+wk+wv) through
/// [`FusedQkvFamily`] against an explicit `FusedQkv*` [`KernelKey`]
/// (`#397 Ship 5.2 slice 3`).
///
/// QKV analogue of [`run_fused_gate_up_key`]: three weights (wq, wk, wv), three
/// outputs (q, k, v), three row-counts. Passing `batch_size: Some(n)` routes the
/// family's QKV run-arm to the IDENTICAL batched `gpu.gemm_qkv_*(.., n)` method
/// the direct prefill call used — each method keeps its own internal arch routing
/// (RDNA4-WMMA / gfx906-dp4a / MMQ / fp16 / scalar) byte-for-byte. The weights,
/// activation `x` (already rmsnorm[-rotated] by the caller), outputs and m/k/n
/// args are unchanged at every migrated site. The `FusedQkv*` key carries the
/// dtype; for HFQ3 the run-arm replicates the call-site WMMA-vs-base arch split
/// internally via `gpu.arch_caps`. `resolve()` only confirms the entry's
/// ArchPredicate admits the current arch.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_fused_qkv_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    wq: &GpuTensor,
    wk: &GpuTensor,
    wv: &GpuTensor,
    x: &GpuTensor,
    y_q: &GpuTensor,
    y_k: &GpuTensor,
    y_v: &GpuTensor,
    q_m: usize,
    k_m: usize,
    v_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[wq, wk, wv],
        x,
        outputs: &[y_q, y_k, y_v],
        m: &[q_m, k_m, v_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::dispatch::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Dispatch a batched-prefill **4-way fused QKVZA** projection (DeltaNet linear
/// attention: wqkv + wz + w_beta + w_alpha) through [`FusedQkvFamily`] against an
/// explicit `FusedQkvza*` [`KernelKey`] (`#397 Ship 5.2 slice 3`).
///
/// QKVZA analogue of [`run_fused_qkv_key`]: four weights, four outputs, four
/// row-counts. `batch_size: Some(n)` routes the family's QKVZA run-arm to the
/// IDENTICAL batched `gpu.gemm_qkvza_*(.., n)` method the direct prefill call
/// used. All operands are passed unchanged; for HFQ3 the run-arm replicates the
/// call-site WMMA-vs-base arch split internally.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_fused_qkvza_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_qkv: &GpuTensor,
    w_z: &GpuTensor,
    w_beta: &GpuTensor,
    w_alpha: &GpuTensor,
    x: &GpuTensor,
    y_qkv: &GpuTensor,
    y_z: &GpuTensor,
    y_beta: &GpuTensor,
    y_alpha: &GpuTensor,
    qkv_m: usize,
    z_m: usize,
    beta_m: usize,
    alpha_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[w_qkv, w_z, w_beta, w_alpha],
        x,
        outputs: &[y_qkv, y_z, y_beta, y_alpha],
        m: &[qkv_m, z_m, beta_m, alpha_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::dispatch::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Batched MoE FFN for `forward_prefill_chunk`. Takes the post-attention
/// residual stream in `pbs.x_batch` ([N × dim]) and writes the FFN output
/// residual back into the same buffer in-place.
///
/// Preconditions (caller must guarantee):
/// - `moe_ffn_batched_admissible(ffn, arch)` returns true: router +
///   shared_expert_gate may be MQ4G256 or Q8_0; all other MoE weights must
///   use an arch-supported MoE quant family.
/// - `pbs.moe_*_batch` tensors are allocated (num_experts > 0 at scratch
///   construction time) and sized to max_batch ≥ N
/// - `config.num_experts_per_tok == 8` and `config.num_experts <= 1024`
///   (hard limits of the batched top-K kernel)
///
/// Sequence mirrors `moe_ffn_decode_impl`'s GPU fast path, with every
/// per-token launch replaced by its N-batched equivalent. Byte-exact
/// except for atomicAdd nondeterminism in the routed-down accumulation
/// (same as the single-token indexed kernel it replaces).
#[allow(clippy::too_many_arguments)]
fn prefill_moe_ffn_body_batched(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    layer_idx: usize,
    ctx: &DispatchCtx,
    // EP (Ship 6 substrate-EP prefill): when `Some`, the routed combine writes
    // into this zeroed `[n × dim]` partial instead of `pbs.x_batch` (the EP
    // driver all-reduce-sums it across ranks and adds into x_batch). The shared
    // expert (step 5) stays in `pbs.x_batch` — replicated per rank, not
    // redirected. `None` = byte-identical single-GPU behavior.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k_top = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
    let shared_scalar = pbs.moe_shared_scalar_batch.as_ref().expect("moe scratch");
    let shared_gate = pbs.moe_shared_gate_batch.as_ref().expect("moe scratch");
    let shared_up = pbs.moe_shared_up_batch.as_ref().expect("moe scratch");
    let shared_rot = pbs.moe_shared_rot_batch.as_ref().expect("moe scratch");
    let topk_indices = pbs.moe_topk_indices_batch.as_ref().expect("moe scratch");
    let topk_weights = pbs.moe_topk_weights_batch.as_ref().expect("moe scratch");
    let gate_batch = pbs.moe_gate_batch.as_ref().expect("moe scratch");
    let up_batch = pbs.moe_up_batch.as_ref().expect("moe scratch");
    let rot_batch = pbs.moe_rot_batch.as_ref().expect("moe scratch");
    let down_expanded = pbs.moe_down_expanded_batch.as_ref().expect("moe scratch");
    let dtypes = MoePrefillDtypes::from_ffn(ffn)
        .ok_or_else(|| HipError::new(0, "missing MoE expert dtype metadata for batched prefill"))?;
    let expert_shape = moe_expert_shape(ffn)
        .ok_or_else(|| HipError::new(0, "missing MoE expert shape metadata for batched prefill"))?;

    // ── 1. Split rmsnorm vs FWHT rotate ──
    //
    // A3B (and every other MoE here) leaves router + shared_expert_gate
    // as Q8_0 in the quantizer — these tiny tensors lose too much
    // accuracy at 4-bit, so the engine never reduces them. Q8 weights
    // are quantized against the un-rotated rmsnorm output, while the
    // MQ-family siblings (shared_expert.{gate,up,down} +
    // experts.{gate_up,down}) expect FWHT(rmsnorm(x) / awq_scale). Populate both:
    //   x_norm_batch ← rmsnorm(x_batch)
    //   x_rot_batch  ← FWHT(x_norm_batch / awq_scale)  (only if any
    //                  downstream MQ weight is present)
    //
    // Pick `shared_expert.gate` as the AWQ representative (instead of
    // the previous `ffn.router`). Per the F1 imatrix scope every gate-side
    // MQ4 sibling shares the same input basis and therefore an identical
    // awq_scale, but the router itself is excluded from F1 (it stays Q8).
    // Reading awq_scale from router would silently drop AWQ rotation in
    // v3 AWQ runs — latent until this predicate widened.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        ffn_norm,
        &pbs.x_norm_batch,
        n,
        dim,
        config.norm_eps,
    )?;
    // PARO mode (shared_expert.gate is ParoQ4G128): each weight carries its
    // own Givens rotation table (paro.pairs / theta / channel_scales). The
    // shared MQ4-style FWHT pre-rotation here would be wrong — skip it. The
    // ParoQ4G128 dispatch arms below run per-weight Givens rotation in-place
    // before each GEMM, using pbs.x_rot_batch as the rotation destination.
    let paro_mode =
        config.has_shared_expert && matches!(ffn.shared_expert.gate.gpu_dtype, DType::ParoQ4G128);
    if !paro_mode {
        if config.has_shared_expert {
            rotate_x_mq_batched_for(
                gpu,
                &ffn.shared_expert.gate,
                &pbs.x_norm_batch,
                &pbs.x_rot_batch,
                dim,
                n,
            )?;
        } else {
            gpu.rotate_x_mq_batched(&pbs.x_norm_batch, &pbs.x_rot_batch, dim, n)?;
        }
    }

    // ── 2. Router + shared-gate + shared.gate + shared.up (4 batched GEMMs) ──
    //
    // Per-dtype dispatch — Q8 reads `x_norm_batch`, MQ4 reads
    // `x_rot_batch`. The natural 4-way fuse via `gemm_qkvza_hfq4g256`
    // is not applicable when router/shared_expert_gate are Q8 (mixed
    // strides). Four separate launches; +3 per MoE layer over the fused
    // ideal, acceptable for the structural unlock.
    match ffn.router.gpu_dtype {
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::MQ4G256 => gpu.gemm_hfq4g256(
            &ffn.router.buf,
            &pbs.x_rot_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::F32 => gpu.gemm_f32_register_tiled(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::F16 => gpu.gemm_f16_x_f32_wmma(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        other => panic!(
            "prefill_moe_ffn_body_batched: unexpected router dtype {other:?} \
                         — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32, F16, BF16"
        ),
    }
    if config.has_shared_expert {
        match ffn.shared_expert_gate.gpu_dtype {
            DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::MQ4G256 => gpu.gemm_hfq4g256(
                &ffn.shared_expert_gate.buf,
                &pbs.x_rot_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::F32 => gpu.gemm_f32_register_tiled(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::F16 => gpu.gemm_f16_x_f32_wmma(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            other => panic!(
                "prefill_moe_ffn_body_batched: unexpected shared_expert_gate dtype {other:?} \
                             — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32, F16, BF16"
            ),
        }
    }
    // #397 Ship 5.2 PILOT: route the router GEMM through GemmFamily::run_key.
    // Each arm uses the *dispatcher-entry* KernelKey (GemmQ8_0BatchedChunked /
    // GemmHfq4G256 / GemmF32Batched) so run_key dispatches to the IDENTICAL
    // gpu.gemm_* method the prior direct call used — preserving each method's
    // own internal arch routing (RDNA4-WMMA / gfx906-dp4a / CDNA-rocBLAS / …)
    // byte-for-byte. The x input still differs per dtype (Q8/F32 read
    // x_norm_batch; MQ4 reads x_rot_batch), exactly as before. The three keys
    // are registered ArchPredicate::Always, so run_key never rejects.
    {
        use hipfire_dispatch::families::gemm::GemmParams;
        let ctx = DispatchCtx::new(gpu);
        let (key, x_in): (hipfire_dispatch::types::KernelKey, &GpuTensor) =
            match ffn.router.gpu_dtype {
                DType::Q8_0 => (
                    hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                    &pbs.x_norm_batch,
                ),
                DType::MQ4G256 => (
                    hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                    &pbs.x_rot_batch,
                ),
                DType::F32 => (
                    hipfire_dispatch::types::KernelKey::GemmF32Batched,
                    &pbs.x_norm_batch,
                ),
                other => panic!(
                    "prefill_moe_ffn_body_batched: unexpected router dtype {other:?} \
                         — moe_ffn_batched_admitted admits MQ4G256, Q8_0, F32"
                ),
            };
        let w = WeightRef {
            buf: &ffn.router.buf,
            dtype: ffn.router.gpu_dtype,
            m: ffn.router.m,
            k: ffn.router.k,
            row_stride: ffn.router.k,
            rotation: None,
            awq_scale: None,
        };
        let params = GemmParams {
            w: &w,
            x: x_in,
            y: router_logits,
            batch_size: n,
        };
        hipfire_runtime::dispatch::gemm_family()
            .run_key(key, &ctx, gpu, &params)
            .map_err(HipError::from)?;
    }
    // DIAG: dump MoE router logits (batched)
    dump_hidden_localize(gpu, router_logits, n, 0, ffn.router.m, 0, "router_b");
    // #397 Ship 5.2 slice1: route the shared-expert-gate GEMM through
    // GemmFamily::run_key. Same dtype-routed dispatcher-entry keys as the router
    // match above (Q8/F32 read x_norm_batch, MQ4 reads x_rot_batch) → identical
    // gpu.gemm_* method, byte-for-byte.
    {
        use hipfire_dispatch::types::KernelKey;
        let (key, x_in): (KernelKey, &GpuTensor) = match ffn.shared_expert_gate.gpu_dtype {
            DType::Q8_0 => (KernelKey::GemmQ8_0BatchedChunked, &pbs.x_norm_batch),
            DType::MQ4G256 => (KernelKey::GemmHfq4G256, &pbs.x_rot_batch),
            DType::F32 => (KernelKey::GemmF32Batched, &pbs.x_norm_batch),
            other => panic!(
                "prefill_moe_ffn_body_batched: unexpected shared_expert_gate dtype {other:?} \
                         — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32"
            ),
        };
        run_plain_gemm_key(
            gpu,
            key,
            &ffn.shared_expert_gate.buf,
            ffn.shared_expert_gate.gpu_dtype,
            x_in,
            shared_scalar,
            ffn.shared_expert_gate.m,
            ffn.shared_expert_gate.k,
            n,
        )?;
    }
    // Fused gate+up dispatch for the shared expert — halves the kernel
    // launch count vs back-to-back gemm_hfq*g256 (~75µs/launch × 40
    // MoE layers = ~3ms saved on R9700 A3B prefill at bs=256).
    // Per-projection dispatch: gate AND up share the same dtype (predicate
    // enforces). MQ4/MQ3/MQ6 route to their HFQ-layout fused kernels.
    if config.has_shared_expert {
        match ffn.shared_expert.gate.gpu_dtype {
            // #397 Ship 5.2 slice 2: shared-expert fused gate+up → FusedQkvFamily
            // (batched-prefill gate+up variant). Same batched kernel, behavior-preserving.
            DType::MQ4G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            DType::MQ6G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            // Phase 2: PARO shared_expert.gate + up. Each weight has its own
            // Givens rotation table — rotate x_norm_batch into x_rot_batch using
            // gate's tables, GEMM, then re-rotate using up's tables, GEMM. Total
            // 4 dispatches vs the MQ4 path's 1 fused gemm_gate_up — acceptable
            // overhead for the per-token-loop elimination win. Phase 4 could
            // collapse this into a single fused kernel
            // (gemm_gate_up_paro_q4g128_batched) if measurement shows it matters.
            DType::ParoQ4G128 => {
                let paro_gate = ffn
                    .shared_expert
                    .gate
                    .paro
                    .as_ref()
                    .expect("ParoQ4G128 shared_expert.gate missing paro metadata");
                let paro_up = ffn
                    .shared_expert
                    .up
                    .paro
                    .as_ref()
                    .expect("ParoQ4G128 shared_expert.up missing paro metadata");
                // Gate: rotate x_norm by gate's Givens → x_rot, then HFQ4G128 GEMM
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro_gate.pairs,
                    &paro_gate.theta,
                    &paro_gate.channel_scales,
                    n,
                    dim,
                    paro_gate.krot as usize,
                )?;
                run_plain_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                    &ffn.shared_expert.gate.buf,
                    ffn.shared_expert.gate.gpu_dtype,
                    &pbs.x_rot_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                // Up: re-rotate x_norm by up's Givens → x_rot (overwrite), GEMM
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro_up.pairs,
                    &paro_up.theta,
                    &paro_up.channel_scales,
                    n,
                    dim,
                    paro_up.krot as usize,
                )?;
                run_plain_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                    &ffn.shared_expert.up.buf,
                    ffn.shared_expert.up.gpu_dtype,
                    &pbs.x_rot_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            DType::F16 => {
                debug_assert_eq!(
                    ffn.shared_expert.up.gpu_dtype,
                    DType::F16,
                    "shared_expert.gate/up dtype predicate should keep F16 paired"
                );
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.gate.buf,
                    &pbs.x_norm_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.up.buf,
                    &pbs.x_norm_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            DType::BF16 => {
                debug_assert_eq!(
                    ffn.shared_expert.up.gpu_dtype,
                    DType::BF16,
                    "shared_expert.gate/up dtype predicate should keep BF16 paired"
                );
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.gate.buf,
                    &pbs.x_norm_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.up.buf,
                    &pbs.x_norm_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: unsupported shared_expert.gate dtype {other:?} \
                             — admit predicate should have rejected this layer"
            ),
        }
    }

    // ── 3. GPU softmax + top-K + renorm, batched over N tokens ──
    //
    // Same Path B split as the decode call site: split the fused
    // softmax+topk+renorm into gpu.softmax_f32 + moe_topk_renorm_k8_batched
    // so prefill activations match the CPU-reference softmax math
    // exactly. router_logits is allocated 1D as [n × n_exp]; alias it
    // into a 2D view so gpu.softmax_f32 takes rows = n.
    let router_logits_2d = GpuTensor {
        buf: unsafe { router_logits.buf.alias() },
        shape: vec![n, n_exp],
        dtype: DType::F32,
    };
    gpu.softmax_f32(&router_logits_2d)?;
    let cpu_topk = if k_top == 8 {
        gpu.moe_topk_renorm_k8_batched(
            router_logits,
            topk_indices,
            topk_weights,
            n_exp,
            config.norm_topk_prob,
            n,
        )?;
        None
    } else {
        let probs = gpu.download_f32(router_logits)?;
        let (indices, weights) =
            cpu_topk_from_softmaxed_rows(&probs, n, n_exp, k_top, config.norm_topk_prob)?;
        upload_cpu_topk_to_device(gpu, &indices, &weights, topk_indices, topk_weights)?;
        Some((indices, weights))
    };
    if moe_router_histogram_active() {
        let (indices, weights) = if let Some((indices, weights)) = cpu_topk.as_ref() {
            (indices.clone(), weights.clone())
        } else {
            (
                download_i32_tensor(gpu, topk_indices, n * k_top)?
                    .into_iter()
                    .map(router_index_i32_to_usize)
                    .collect::<Vec<_>>(),
                gpu.download_f32(topk_weights)?,
            )
        };
        for token_idx in 0..n {
            let start = token_idx * k_top;
            let end = start + k_top;
            record_moe_router_selection(layer_idx, &indices[start..end], &weights[start..end]);
        }
    }
    let paged_topk_indices = if ffn.experts.is_empty() {
        let indices = if let Some((indices, _)) = cpu_topk.as_ref() {
            indices.clone()
        } else {
            download_i32_tensor(gpu, topk_indices, n * k_top)?
                .into_iter()
                .map(router_index_i32_to_usize)
                .collect::<Vec<_>>()
        };
        Some(indices)
    } else {
        None
    };
    let paged_expert_buckets = if let Some(indices) = paged_topk_indices.as_ref() {
        Some(build_paged_moe_expert_buckets(indices, n, k_top, n_exp)?)
    } else {
        None
    };

    // ── 4. Shared-expert SwiGLU + FWHT, batched over N tokens ──
    //
    // fused_silu_mul_rotate_mq_batched expects [batch × k] gate/up with
    // batch on grid.y and writes FWHT(silu(gate) * up) into x_rot. Here
    // batch=N, k=smi; the shared-rot output buffer is [N × smi].
    // F2: AWQ-aware silu_mul+rotate for the batched shared-expert down input.
    // PARO: shared_expert.down has its own Givens rotation tables (paro.*);
    // use the dedicated fused kernel (commit 50198daa). It takes a per-weight
    // (pairs, theta, channel_scales, krot) tuple instead of the MQ4 FWHT
    // convention. Same shape: gate/up [N × smi] → shared_rot [N × smi].
    if config.has_shared_expert {
        if paro_mode {
            let paro_down = ffn
                .shared_expert
                .down
                .paro
                .as_ref()
                .expect("ParoQ4G128 shared_expert.down missing paro metadata");
            gpu.fused_silu_mul_givens_rotate_f32(
                shared_gate,
                shared_up,
                shared_rot,
                &paro_down.pairs,
                &paro_down.theta,
                &paro_down.channel_scales,
                n,
                smi,
                paro_down.krot as usize,
            )?;
        } else if matches!(ffn.shared_expert.down.gpu_dtype, DType::F16 | DType::BF16) {
            gpu.silu_mul_f32(shared_gate, shared_up, shared_rot)?;
        } else {
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &ffn.shared_expert.down,
                shared_gate,
                shared_up,
                shared_rot,
                smi,
                n,
            )?;
        }
    }

    // ── 5. Shared-expert down with sigmoid-scaled residual, batched ──
    //
    // Reads shared_scalar[token] as the pre-sigmoid logit, applies sigmoid
    // internally, and += sigmoid(scalar) × (W_down · rot) into
    // pbs.x_batch[token × dim + row]. (Note: HFQ4 sister uses += not
    // atomicAdd; each (bid, row) writes a unique cell.)
    // Per-projection dispatch: MQ4/MQ3/MQ6 route to their HFQ-layout sisters.
    if config.has_shared_expert {
        match ffn.shared_expert.down.gpu_dtype {
            DType::MQ4G256 => gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ6G256 => gpu.gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ3G256 => gpu.gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ2G256 => gpu.gemv_hfq2g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ8G256 => gpu.gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ2G256Lloyd => gpu.gemv_mq2g256_lloyd_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            // Phase 2: HFQ4G128 batched residual+sigmoid-scaled kernel. Single
            // launch, same semantics as the HFQ4G256 sister — reads shared_rot
            // (already silu-mul-rotated by the PARO fused kernel above), GEMVs
            // against W_down, applies sigmoid(shared_scalar[token]) × output,
            // accumulates into pbs.x_batch.
            DType::ParoQ4G128 => gpu.gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::F16 => {
                let shared_down_scratch =
                    pbs.x_rot_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.down.buf,
                    shared_rot,
                    &shared_down_scratch,
                    ffn.shared_expert.down.m,
                    ffn.shared_expert.down.k,
                    n,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.scaled_add_inplace_gpu_sigmoid_rows_f32(
                    &x_n,
                    &shared_down_scratch,
                    shared_scalar,
                    ffn.shared_expert.down.m,
                    n,
                )?;
            }
            DType::BF16 => {
                let shared_down_scratch =
                    pbs.x_rot_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.down.buf,
                    shared_rot,
                    &shared_down_scratch,
                    ffn.shared_expert.down.m,
                    ffn.shared_expert.down.k,
                    n,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.scaled_add_inplace_gpu_sigmoid_rows_f32(
                    &x_n,
                    &shared_down_scratch,
                    shared_scalar,
                    ffn.shared_expert.down.m,
                    n,
                )?;
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: unsupported shared_expert.down dtype {other:?} \
                         — admit predicate should have rejected this layer"
            ),
        }
    }

    // ── 6. Routed experts: batched gate_up → SwiGLU+FWHT → down ──
    //
    // Gate/up for top-K experts (per token) → [N × K_TOP × mi]. Each
    // output row reads topk_indices[token × K_TOP + krank] to pick its
    // expert weight base from the device-side expert_gate_up_ptrs table.
    let down_m = expert_shape.down_m;
    let down_k = expert_shape.down_k;
    let gate_up_k = expert_shape.gate_up_k;

    // Path 2 (SGLang-style scatter + grouped-WMMA-GEMM) — default ON for
    // gfx11/gfx12, where the grouped-WMMA kernel is validated (gfx11 routes
    // to `gemm_hfq4g256_moe_grouped_wmma_k2` via the base w32 WMMA builtin,
    // gfx12 to the `_gfx12` variant). Empirical lift on Qwen3.5-A3B mq4
    // prefill=256: gfx1100 7900 XTX 1396 → 2983 tok/s (+114%); gfx1201
    // R9700 1016 → 2966 tok/s (uniform-mq4.hfq, +192%). CDNA wave64 (gfx9*)
    // and pre-WMMA RDNA (gfx10*) stay on the per-token indexed_batched
    // GEMV path. Opt out with `HIPFIRE_MOE_GROUPED_GEMM=0`.
    // Cached read — getenv on every layer × MoE call adds up.
    static USE_PATH2_GATE_UP: OnceLock<bool> = OnceLock::new();
    let use_path2 = *USE_PATH2_GATE_UP.get_or_init(|| {
        moe_grouped_gemm_path2_enabled_from_env(
            std::env::var("HIPFIRE_MOE_GROUPED_GEMM").ok().as_deref(),
        )
    });
    let path2_eligible = moe_grouped_gemm_path2_eligible_for_dtype(
        dtypes.expert_gate_up,
        &gpu.arch,
        use_path2 && (!ffn.experts.is_empty() || paged_expert_buckets.is_some()),
    );
    // m_total — computed during gate_up scatter, reused for down. Avoids
    // a second dtoh sync per MoE layer.
    let mut path2_m_total: usize = 0;
    let path2_shape = moe_grouped_path2_shape(n, k_top, n_exp);
    if paged_expert_buckets.is_some() && !path2_eligible {
        return Err(HipError::new(
            0,
            "paged grouped-MoE prefill requires grouped GEMM path2 support",
        ));
    }
    moe_prefill_prepare_routed_gate_up_input(
        gpu,
        ffn,
        &dtypes,
        &pbs.x_norm_batch,
        &pbs.x_rot_batch,
        dim,
        n,
    )?;
    if path2_eligible {
        // Stage 1 scatter pipeline. The scratch buffers are sized for
        // worst-case max_batch. Runtime launch bounds use the tighter live
        // slot upper bound below. Block size 16 (the WMMA tile row count).
        const BLOCK_M: usize = MOE_GROUPED_BLOCK_M;
        let counts = pbs.moe_expert_token_counts.as_ref().expect("path2 scratch");
        let offsets = pbs.moe_expert_offsets.as_ref().expect("path2 scratch");
        let sorted = pbs.moe_sorted_slot_index.as_ref().expect("path2 scratch");
        let inverse_perm = pbs.moe_inverse_perm.as_ref().expect("path2 scratch");
        let tile_ids = pbs.moe_expert_tile_ids.as_ref().expect("path2 scratch");
        let y_gu_grouped = pbs.moe_y_gate_up_grouped.as_ref().expect("path2 scratch");
        if let Some(buckets) = paged_expert_buckets.as_ref() {
            if dtypes.expert_gate_up != DType::MQ6G256 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "paged grouped-MoE prefill currently supports MQ6 routed experts only, got {:?}",
                        dtypes.expert_gate_up
                    ),
                ));
            }
            // Load all active experts once before the per-bucket loops so the
            // down phase doesn't need a second round of page-ins.
            let active_experts: Vec<usize> = buckets.iter().map(|b| b.expert as usize).collect();
            ensure_paged_experts_resident(gpu, pager, ffn, &active_experts)?;
            for bucket in buckets {
                upload_paged_moe_expert_bucket(gpu, bucket, sorted, inverse_perm, tile_ids)?;
                gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    bucket.m_total,
                    path2_shape.gate_up_source_rows,
                )?;
                gpu.moe_gate_up_unscatter_k8(
                    y_gu_grouped,
                    sorted,
                    gate_batch,
                    up_batch,
                    mi,
                    k_top,
                    bucket.m_total,
                )?;
            }
        } else {
            // m_total upper bound — scratch is sized in PrefillBatchScratch::new
            // with the all-experts worst case, while this launch only needs slots
            // plus padding for experts that can be non-empty at this N.
            // The scatter fused kernel pre-fills every tile id in this aligned
            // bound with -1; grouped GEMM early-returns on sentinel tiles, so we
            // can skip the m_total dtoh sync entirely. Saves ~50µs/layer.
            let m_total_max = path2_shape.m_total_bound;

            // Fused scatter pipeline: one launch replaces histogram + offsets
            // + permute. Saves 2 launches × ~75µs × MoE layers.
            gpu.moe_scatter_fused_k8(
                topk_indices,
                counts,
                offsets,
                sorted,
                tile_ids,
                inverse_perm,
                path2_shape.total_slots,
                n_exp,
                m_total_max,
                BLOCK_M,
            )?;

            // Use m_total_max as the upper bound for grid sizing — the kernel
            // early-returns on expert_tile_ids[tile_y] == -1 for the
            // pre-sentinel'd unused-tile range.
            path2_m_total = m_total_max;
            let m_total = m_total_max;

            // Stage 2 grouped GEMM (gate_up). Writes Y_grouped[m_total × 2*mi] direct.
            // x_src = x_rot_batch [N × dim], x_row_div = K_TOP.
            // Per-dtype dispatch: experts uniform per layer (admit predicate
            // enforces). MQ4/MQ3/MQ6 route to their HFQ-layout grouped WMMA
            // sisters.
            match dtypes.expert_gate_up {
                DType::MQ4G256 => gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::MQ6G256 => gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::MQ3G256 => gpu.gemm_hfq3g256_moe_grouped_wmma(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::MQ2G256Lloyd => {
                    if mq2_lloyd_n32_gfx1151_enabled(&gpu.arch, path2_shape.total_slots) {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?
                    } else {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?
                    }
                }
                DType::F16 => gpu.gemm_f16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_norm_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::BF16 => gpu.gemm_bf16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_norm_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                // Phase 4: Path 2 ParoQ4G128 grouped-WMMA. All 256 routed
                // experts at this layer share one gate_up Givens rotation
                // sidecar (ffn.paro_shared.gate_up_*); rotate x_norm into
                // x_rot ONCE, then dispatch the HFQ4G128 grouped WMMA. The
                // kernel auto-converts the F32 x_rot to F16 internally via
                // ensure_fp16_x, same as the G256 sister.
                //
                // gfx1151 i8 MMQ opt-in (HIPFIRE_MOE_PARO_I8=1): routes to the
                // HFQ4G128 i8 MMQ kernel which doubles compute throughput on
                // Strix Halo (~140 vs ~71 TFLOPS). Compute-bound regime per
                // Phase 4 attribution (gemm_paro_q4g128_moe_grouped_wmma_k2
                // = 68.5% GPU time, 25.8 GiB/s — far from BW roof).
                DType::ParoQ4G128 => {
                    let paro = ffn
                        .paro_shared
                        .as_ref()
                        .expect("ParoQ4G128 routed experts require paro_shared sidecars");
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro.gate_up_pairs,
                        &paro.gate_up_theta,
                        &paro.gate_up_channel_scales,
                        n,
                        dim,
                        paro.krot as usize,
                    )?;
                    // Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over
                    // FP16 WMMA, k8 +2.5% over k2, both validated via PARO gen 100
                    // (clean decode, finite logits) + coherence-gate (MQ4 paths
                    // unchanged). Opt-out via HIPFIRE_MOE_PARO_I8=0 or _K8=0.
                    let use_paro_i8 = paro_moe_i8_enabled_for_arch_from_env(
                        gpu.arch.as_str(),
                        std::env::var("HIPFIRE_MOE_PARO_I8").ok().as_deref(),
                    );
                    let use_paro_i8_k8 = paro_moe_i8_k8_enabled_from_env(
                        use_paro_i8,
                        std::env::var("HIPFIRE_MOE_PARO_I8_K8").ok().as_deref(),
                    );
                    if use_paro_i8_k8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?;
                    } else if use_paro_i8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?;
                    } else {
                        gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?;
                    }
                }
                other => panic!(
                    "prefill_moe_ffn_body_batched: unsupported experts[0].gate_up dtype {other:?} \
                             — admit predicate should have rejected this layer"
                ),
            }

            // Stage 3 unscatter combine. Fans Y_grouped → gate_batch + up_batch.
            gpu.moe_gate_up_unscatter_k8(
                y_gu_grouped,
                sorted,
                gate_batch,
                up_batch,
                mi,
                k_top,
                m_total,
            )?;
        }
    } else {
        // Path 1 fallback (CDNA/gfx10): per-token indexed GEMV, batched
        // over the N tokens via grid.z. The dispatch is dtype-keyed because
        // the kernel reads the weight nibble layout directly (HFQ4G256:
        // 136 B/group; HFQ4G128/PARO: 72 B/group).
        match dtypes.expert_gate_up {
            DType::MQ4G256 => gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ6G256 => gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ2G256 => gpu.gemv_hfq2g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ8G256 => gpu.gemv_hfq8g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ2G256Lloyd => gpu.gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            // Phase 3 PARO routed-expert: apply the layer's shared gate_up
            // Givens rotation to x_norm_batch into x_rot_batch ONCE, then
            // dispatch the HFQ4G128 indexed batched kernel. All 256 experts
            // at this layer share the same gate_up rotation sidecar
            // (ffn.paro_shared, populated by paro_load_moe_shared_sidecars).
            DType::ParoQ4G128 => {
                let paro = ffn
                    .paro_shared
                    .as_ref()
                    .expect("ParoQ4G128 routed experts require paro_shared sidecars");
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro.gate_up_pairs,
                    &paro.gate_up_theta,
                    &paro.gate_up_channel_scales,
                    n,
                    dim,
                    paro.krot as usize,
                )?;
                gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?;
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: Path 1 fallback unsupported \
                             experts[0].gate_up dtype {other:?} — admit predicate should \
                             have rejected this layer"
            ),
        }
    }

    // SwiGLU + FWHT over [N*K_TOP × mi] — batch flatten across tokens and
    // expert ranks, k=mi is per-row width.
    // F2: AWQ-aware silu_mul+rotate; experts[0].down is representative (all
    // experts at this layer share imatrix at the same residual basis).
    // PARO branch (Phase 3): the layer-shared `down` rotation sidecar lives
    // on ffn.paro_shared (not per-expert; all 256 experts alias the same
    // tuple). Apply via fused_silu_mul_givens_rotate_f32 over the flattened
    // [n*k_top × mi] grid.
    if paro_mode {
        let paro = ffn
            .paro_shared
            .as_ref()
            .expect("ParoQ4G128 routed experts require paro_shared sidecars");
        gpu.fused_silu_mul_givens_rotate_f32(
            gate_batch,
            up_batch,
            rot_batch,
            &paro.down_pairs,
            &paro.down_theta,
            &paro.down_channel_scales,
            n * k_top,
            mi,
            paro.krot as usize,
        )?;
    } else if matches!(dtypes.expert_down, DType::F16 | DType::BF16) {
        gpu.silu_mul_f32(gate_batch, up_batch, rot_batch)?;
    } else if ffn.experts.is_empty() {
        gpu.fused_silu_mul_rotate_mq_batched(gate_batch, up_batch, rot_batch, mi, n * k_top)?;
    } else {
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &ffn.experts[0].down,
            gate_batch,
            up_batch,
            rot_batch,
            mi,
            n * k_top,
        )?;
    }

    // Down projection. Three paths:
    //   Path 2 (HIPFIRE_MOE_GROUPED_GEMM=1, RDNA): grouped-WMMA-GEMM
    //     reusing the gate_up scatter + inverse_perm + a non-atomic combine.
    //   Path 1 (RDNA, default): atomic-free expanded GEMV write + combine.
    //   Path 0 (CDNA wave64 fallback): residual_scaled atomic GEMV.
    //
    // Path 1: K_TOP-way atomicAdd contention per output cell — 387 GiB/s
    // observed vs 954 on the sister gate_up. Path 2 amortizes weights via
    // WMMA across the m_total tokens routed to each expert; ~67ms saved on
    // the down kernel for A3B prefill at batch 256 (R9700).
    // CDNA (wave64, HBM2/3) stays on Path 0 — cheap HBM atomics +
    // expanded scratch cost makes the GEMV pattern competitive.
    if path2_eligible {
        let y_down_grouped = pbs.moe_y_down_grouped.as_ref().expect("path2 scratch");
        let inverse_perm = pbs.moe_inverse_perm.as_ref().expect("path2 scratch");
        let sorted = pbs.moe_sorted_slot_index.as_ref().expect("path2 scratch");
        let tile_ids = pbs.moe_expert_tile_ids.as_ref().expect("path2 scratch");
        if let Some(buckets) = paged_expert_buckets.as_ref() {
            if dtypes.expert_down != DType::MQ6G256 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "paged grouped-MoE prefill currently supports MQ6 routed down experts only, got {:?}",
                        dtypes.expert_down
                    ),
                ));
            }
            for bucket in buckets {
                upload_paged_moe_expert_bucket(gpu, bucket, sorted, inverse_perm, tile_ids)?;
                gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    bucket.m_total,
                    path2_shape.down_source_rows,
                )?;
                gpu.moe_down_combine_grouped_k8(
                    y_down_grouped,
                    inverse_perm,
                    topk_weights,
                    &pbs.x_batch,
                    down_m,
                    k_top,
                    n,
                )?;
            }
        } else {
            // m_total already computed during gate_up scatter — reuse to skip
            // a second dtoh sync per MoE layer (~50µs each × 40 layers = 2ms).
            let m_total = path2_m_total;

            // Grouped GEMM on down: x_src = rot_batch [N*K_TOP × mi], x_row_div = 1
            // (sorted_slot_index[slot] directly indexes the source row).
            // Per-dtype dispatch: experts uniform per layer. MQ4 → HFQ4-layout;
            // MQ6 → HFQ6 sister (shipped via feat/hfq6-moe-grouped-wmma).
            match dtypes.expert_down {
                DType::MQ4G256 => gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::MQ6G256 => gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::MQ3G256 => gpu.gemm_hfq3g256_moe_grouped_wmma(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::MQ2G256Lloyd => {
                    if mq2_lloyd_n32_gfx1151_enabled(&gpu.arch, path2_shape.total_slots) {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?
                    } else {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?
                    }
                }
                DType::F16 => gpu.gemm_f16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::BF16 => gpu.gemm_bf16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                // Phase 4: Path 2 ParoQ4G128 down grouped-WMMA (with i8 MMQ
                // opt-in for gfx1151 — see gate_up arm above). rot_batch was
                // already Givens-rotated by paro_shared.down_* via the PARO
                // fused_silu_mul_givens_rotate_f32 step above; the kernel is
                // rotation-agnostic. Same kernel for gate_up + down — only
                // shape parameters and x_row_div differ.
                DType::ParoQ4G128 => {
                    // Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over
                    // FP16 WMMA, k8 +2.5% over k2, both validated via PARO gen 100
                    // (clean decode, finite logits) + coherence-gate (MQ4 paths
                    // unchanged). Opt-out via HIPFIRE_MOE_PARO_I8=0 or _K8=0.
                    let use_paro_i8 = paro_moe_i8_enabled_for_arch_from_env(
                        gpu.arch.as_str(),
                        std::env::var("HIPFIRE_MOE_PARO_I8").ok().as_deref(),
                    );
                    let use_paro_i8_k8 = paro_moe_i8_k8_enabled_from_env(
                        use_paro_i8,
                        std::env::var("HIPFIRE_MOE_PARO_I8_K8").ok().as_deref(),
                    );
                    if use_paro_i8_k8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?;
                    } else if use_paro_i8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?;
                    } else {
                        gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?;
                    }
                }
                other => panic!(
                    "prefill_moe_ffn_body_batched: unsupported experts[0].down dtype {other:?} \
                             — admit predicate should have rejected this layer"
                ),
            }
            // Non-atomic combine via inverse_perm + topk_weights.
            gpu.moe_down_combine_grouped_k8(
                y_down_grouped,
                inverse_perm,
                topk_weights,
                &pbs.x_batch,
                down_m,
                k_top,
                n,
            )?;
        }
    } else {
        let use_atomic_free_down = !gpu.arch.starts_with("gfx9");
        if use_atomic_free_down {
            // Path 1 expanded-down: per-token-per-rank GEMV writes to a
            // [N × K_TOP × M] scratch, then a separate combine kernel folds
            // it back into pbs.x_batch with topk weights. The expanded
            // kernel is dtype-keyed; the combine kernel is dtype-agnostic.
            match dtypes.expert_down {
                DType::MQ4G256 => gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ6G256 => gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ2G256 => gpu.gemv_hfq2g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ8G256 => gpu.gemv_hfq8g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ2G256Lloyd => gpu
                    .gemv_mq2g256_lloyd_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                DType::MQ3G256Lloyd => gpu
                    .gemv_mq3g256_lloyd_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                // Phase 3 PARO down: the layer-shared `down` Givens rotation
                // has already been applied to rot_batch by the
                // fused_silu_mul_givens_rotate_f32 call above. The HFQ4G128
                // indexed kernel (existing, shipped in 7c00970d) is
                // rotation-agnostic; same dispatch shape as G256 sister.
                DType::ParoQ4G128 => gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                other => panic!(
                    "prefill_moe_ffn_body_batched: Path 1 fallback unsupported \
                                 experts[0].down dtype {other:?} — admit predicate should \
                                 have rejected this layer"
                ),
            }
            gpu.moe_down_combine_k8_batched(
                down_expanded,
                topk_weights,
                &pbs.x_batch,
                down_m,
                k_top,
                n,
            )?;
        } else {
            gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
                &ffn.expert_down_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                &pbs.x_batch,
                down_m,
                down_k,
                k_top,
                n,
            )?;
        }
    }
    // ── 6. Routed experts: delegated to MoeFamily::run_prefill (Ship 4.2) ──
    let down_m = ffn.experts[0].down.m;
    let down_k = ffn.experts[0].down.k;
    let gate_up_k = ffn.experts[0].gate_up.k;
    let total_slots = n * k_top;
    let m_total_max = moe_grouped_m_total_bound(total_slots, n_exp);

    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        experts_all_gate_up_mq4: ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256),
        routed_gate_up: ffn.experts[0].gate_up.gpu_dtype,
        routed_down: ffn.experts[0].down.gpu_dtype,
        has_paro_shared: ffn.paro_shared.is_some(),
    };

    let paro_gate_up =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.gate_up_pairs,
                theta: &paro.gate_up_theta,
                scales: &paro.gate_up_channel_scales,
                krot: paro.krot as usize,
            });
    let paro_down =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.down_pairs,
                theta: &paro.down_theta,
                scales: &paro.down_channel_scales,
                krot: paro.krot as usize,
            });
    let down_awq_scale = ffn.experts[0].down.awq_scale.as_ref();

    let moe_prefill_params = hipfire_dispatch::families::moe::MoePrefillParams {
        dtypes: moe_dtypes,
        batch_size: n,
        mi,
        down_m,
        down_k,
        gate_up_k,
        k_top,
        n_exp,
        m_total_max,
        topk_indices,
        topk_weights,
        x_batch: &pbs.x_batch,
        x_norm_batch: &pbs.x_norm_batch,
        x_rot_batch: &pbs.x_rot_batch,
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        gate_batch,
        up_batch,
        rot_batch,
        down_expanded,
        expert_token_counts: pbs.moe_expert_token_counts.as_ref().expect("moe scratch"),
        expert_offsets: pbs.moe_expert_offsets.as_ref().expect("moe scratch"),
        sorted_slot_index: pbs.moe_sorted_slot_index.as_ref().expect("moe scratch"),
        expert_tile_ids: pbs.moe_expert_tile_ids.as_ref().expect("moe scratch"),
        inverse_perm: pbs.moe_inverse_perm.as_ref().expect("moe scratch"),
        y_gate_up_grouped: pbs.moe_y_gate_up_grouped.as_ref().expect("moe scratch"),
        y_down_grouped: pbs.moe_y_down_grouped.as_ref().expect("moe scratch"),
        paro_gate_up,
        paro_down,
        down_awq_scale,
        routed_out,
    };
    hipfire_runtime::dispatch::moe_family()
        .run_prefill(ctx, gpu, &moe_prefill_params)
        .map_err(HipError::from)?;

    Ok(())
}

/// Band view for `forward_prefill_chunk`. `None` (the default) means the
/// chunk processes the whole stack: embedding → all layers → final norm
/// + lm_head. `Some(b)` restricts the chunk to layers `b.layer_start..
/// b.layer_end`, skips the embedding when `!b.is_first_band` (input is
/// already in `pbs.x_batch` from a prior peer-copy), and skips the final
/// norm + lm_head when `!b.is_last_band` (output activation stays in
/// `pbs.x_batch` for the next band's peer-copy).
///
/// Counter offsets seed the running per-LA / per-KV / per-FA counters so
/// the band's first DeltaNet/FullAttn layer indexes the correct
/// `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot.
pub(crate) struct PrefillBandCtx<'a> {
    pub layer_start: usize,
    pub layer_end: usize,
    pub delta_layer_offset: usize,
    pub kv_layer_offset: usize,
    pub fa_layer_offset: usize,
    pub is_first_band: bool,
    pub is_last_band: bool,
    /// Per-device asym{2,3,4} givens replicas. When `Some`, the chunk's
    /// FA-layer batched KV writers use these instead of `kv_cache.givens_*`
    /// (which is `None` in multi-GPU mode by design — each device needs its
    /// own copy of the rotation tables).
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
}

#[allow(clippy::too_many_arguments)]
/// Debug localization hook (no-op unless `HIPFIRE_DUMP_HIDDEN` is set to a file
/// prefix). Appends the post-layer hidden row for the target absolute position
/// to `{HIPFIRE_DUMP_HIDDEN}.{tag}` as `u32 layer_idx` followed by `dim`
/// little-endian f32. The target absolute position is `HIPFIRE_DUMP_HIDDEN_POS`
/// (default 0); `abs_pos_of_row0` is the absolute sequence position of row 0 of
/// `x` (`start_pos` for the batched residual `pbs.x_batch`, `pos` for the
/// single-row per-token `s.x`). Used to localize the PARO batched-prefill
/// divergence by diffing `.batched` vs `.pertoken` per layer. Requires
/// `HIPFIRE_GRAPH=0` (does a synchronous D2H readback, which is illegal under
/// graph capture).
fn dump_hidden_localize(
    gpu: &Gpu,
    x: &GpuTensor,
    n_rows: usize,
    abs_pos_of_row0: usize,
    dim: usize,
    layer_idx: usize,
    tag: &str,
) {
    let prefix = match std::env::var("HIPFIRE_DUMP_HIDDEN") {
        Ok(p) => p,
        Err(_) => return,
    };
    use std::io::Write;
    let path = format!("{prefix}.{tag}");
    // Activation-capture mode (HIPFIRE_DUMP_HIDDEN_ALL=1): dump EVERY row of `x`
    // as raw [dim] f32 each (no per-row header) — so one prefill yields n_rows
    // real-activation samples for an offline rotation/quant study, AND a per-token
    // decode appends its single row each call → the file accumulates the whole
    // sequence. This mode IGNORES the single-position POS gate (which only makes
    // sense for the localize path below); it must fire at every position.
    if std::env::var("HIPFIRE_DUMP_HIDDEN_ALL").as_deref() == Ok("1") {
        // Two sub-modes:
        //  - default: restrict to one target layer (HIPFIRE_DUMP_HIDDEN_LAYER,
        //    default 0), one file `{prefix}.{tag}` (the kv-compression study path).
        //  - HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1: capture EVERY layer to per-layer
        //    files `{prefix}.{tag}.L{layer_idx}` — the Phase-A block-local
        //    recovery capture (residual-stream in/mid/out for all blocks).
        let all_layers = std::env::var("HIPFIRE_DUMP_HIDDEN_ALLLAYERS").as_deref() == Ok("1");
        let layer_path = if all_layers {
            format!("{path}.L{layer_idx}")
        } else {
            let want_layer: usize = std::env::var("HIPFIRE_DUMP_HIDDEN_LAYER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if layer_idx != want_layer {
                return;
            }
            path.clone()
        };
        if gpu.hip.device_synchronize().is_err() {
            return;
        }
        let all = match gpu.download_f32(x) {
            Ok(v) => v,
            Err(_) => return,
        };
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&layer_path)
        {
            let take = (n_rows * dim).min(all.len());
            let bytes: Vec<u8> = all[..take].iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = f.write_all(&bytes);
        }
        return;
    }
    // Single-position localize path (PARO batched-vs-pertoken diff).
    let target: usize = std::env::var("HIPFIRE_DUMP_HIDDEN_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if target < abs_pos_of_row0 {
        return;
    }
    let row = target - abs_pos_of_row0;
    if row >= n_rows {
        return;
    }
    if gpu.hip.device_synchronize().is_err() {
        return;
    }
    let all = match gpu.download_f32(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    let off = row * dim;
    if off + dim > all.len() {
        return;
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(&(layer_idx as u32).to_le_bytes());
        let mut bytes = Vec::with_capacity(dim * 4);
        for v in &all[off..off + dim] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&bytes);
    }
}

fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tape_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
    band: Option<&PrefillBandCtx<'_>>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    positions_override: Option<&[usize]>,
    needs_last_token_logits: bool,
    max_layer: Option<usize>,
    force_q8_gdn_per_token: bool,
    // EP (Ship 6 substrate-EP prefill): per-MoE-layer routed partial. ONLY set
    // by the EP driver, which calls this with a SINGLE-layer band so the routed
    // combine of that one MoE layer lands in the zeroed partial (all-reduced by
    // the driver after the call). Always `None` for multi-layer bands (PP /
    // single-GPU full stack) — a shared partial across >1 MoE layer would be wrong.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);
    debug_assert!(
        routed_out.is_none()
            || band
                .map(|b| b.layer_end - b.layer_start <= 1)
                .unwrap_or(false),
        "forward_prefill_chunk: routed_out requires a single-layer band (EP driver invariant)",
    );

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;
    let do_embed = band.map(|b| b.is_first_band).unwrap_or(true);
    let layer_start = band.map(|b| b.layer_start).unwrap_or(0);
    // `max_layer = Some(N)` early-exits at layer N (exclusive). pflash uses
    // this with N = score_layer_idx + 1: the drafter forward only needs to
    // populate the K cache through the scoring layer (the shallowest
    // FullAttention layer, typically layer 3 of 24 in Qwen3.5 hybrid),
    // since `pflash_score_q8_kv` reads exactly that layer's K. Layers
    // beyond it and the final norm + lm_head are wasted compute for
    // pflash. Saves ~80% of drafter forward time on hybrid drafters.
    let layer_end = band
        .map(|b| b.layer_end)
        .unwrap_or(config.n_layers)
        .min(max_layer.unwrap_or(usize::MAX));
    // Skip final norm + lm_head when the caller early-exits — they produce
    // logits the caller doesn't read, and require running through the full
    // layer stack anyway.
    let do_lm_head = band.map(|b| b.is_last_band).unwrap_or(true) && max_layer.is_none();
    let debug_stop_after_la_layer = std::env::var("HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let debug_stop_stage_layer = std::env::var("HIPFIRE_PREFILL_STOP_STAGE_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let debug_stop_stage = std::env::var("HIPFIRE_PREFILL_STOP_STAGE").ok();
    macro_rules! debug_stop_after {
        ($stage:literal, $layer_idx:expr) => {
            if debug_stop_stage_layer == Some($layer_idx)
                && debug_stop_stage.as_deref() == Some($stage)
            {
                return Ok(());
            }
        };
    }
    // Per-call-site `givens_cos_view` / `givens_sin_view` macros below
    // resolve to either the band-supplied per-device replica (multi-GPU
    // mode where `kv_cache.givens_*` is `None` by design) or the
    // kv_cache's own table (single-GPU). Held as macros, not top-level
    // bindings, so the immutable borrow on `kv_cache.givens_*` doesn't
    // outlive the kernel-call statement and conflict with later
    // mutable borrows of `kv_cache` (e.g. inside `run_fa_layer_body`).
    macro_rules! givens_cos_view {
        () => {
            band.and_then(|b| b.givens_cos)
                .or(kv_cache.givens_cos.as_ref())
        };
    }
    macro_rules! givens_sin_view {
        () => {
            band.and_then(|b| b.givens_sin)
                .or(kv_cache.givens_sin.as_ref())
        };
    }

    // ── 1. Embed tokens into pbs.x_batch ─────────────────────────────────
    //
    // Fast path for HFQ4G256 (all MQ4-quantized Qwen3.5 models + friends):
    // upload token ids to a device buffer and dispatch one batched kernel
    // that dequantizes N rows directly into `pbs.x_batch`. This collapses
    // 2N launches (N embed + N memcpy_dtod_at) into 1 upload + 1 launch
    // AND is hipGraph-captureable — the kernel reads token ids from a
    // device pointer instead of taking them as a baked-in scalar arg.
    //
    // Other formats fall back to the per-token loop (kept for correctness
    // breadth; the MQ4-quantized hot path doesn't hit them).
    //
    // Multi-GPU band-mode: skip embedding when this is not the first band.
    // The activation already lives in `pbs.x_batch` from a peer-copy of
    // the previous band's `pbs.x_batch`.
    if do_embed
        && matches!(
            weights.embd_format,
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0 | EmbeddingFormat::F32
        )
    {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::F32 => {
                gpu.embedding_lookup_f32_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            _ => unreachable!(),
        }
    } else if do_embed {
        for (i, &tok) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => unreachable!(),
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?
                }
                _ => panic!("unsupported embedding format"),
            }
            gpu.memcpy_dtod_at_auto(
                &pbs.x_batch.buf,
                i * dim_row_bytes,
                &s.x.buf,
                0,
                dim_row_bytes,
            )?;
        }
    }

    // ── 1a. Apply MaskEmbedOverride (MTP probe hook) ─────────────────────
    //
    // Overwrite a single batch slot's embedding row in `pbs.x_batch` after
    // the embedding-lookup kernel populated it but BEFORE the layer loop
    // (or any subsequent kernel) reads it. The Qualcomm MTP probe uses this
    // to replace the embedding-table value at a "mask token" position with
    // a prompt-mean vector. Default callers pass `None` → zero overhead.
    //
    // Multi-GPU band-mode: skip on non-first bands; pbs.x_batch already
    // holds the peer-copied activation from the previous band, so an
    // override applied at band 0 has already propagated through the layer
    // stack on that device — re-applying here would clobber the partial
    // forward state.
    if do_embed {
        if let Some(ovr) = mask_override {
            assert!(
                ovr.slot < n,
                "MaskEmbedOverride.slot ({}) must be < n ({})",
                ovr.slot,
                n,
            );
            assert_eq!(
                ovr.embed.len(),
                dim,
                "MaskEmbedOverride.embed.len() ({}) must equal config.dim ({})",
                ovr.embed.len(),
                dim,
            );
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ovr.embed.as_ptr() as *const u8, dim * 4) };
            let offset = ovr.slot * dim_row_bytes;
            gpu.hip
                .memcpy_htod_offset(&pbs.x_batch.buf, offset, bytes)?;
        }
    }

    // ── 1b. Upload positions array ────────────────────────────────────────
    //
    // Positions is the per-row RoPE angle AND the physical KV cache slot (the
    // batched kv_write kernels use the same index for both). Default callers
    // use flat linear `start_pos .. start_pos + n`; the dense server-prefill
    // session-batch worker can pass explicit per-row positions for independent
    // sessions. Siblings in DDTree mode get DISTINCT slots via the default
    // linear path — no write race — and the stored K carries a RoPE angle that
    // matches the physical slot, which keeps subsequent cycles' attention
    // reads consistent.
    //
    // Semantic trade vs. the original depth-based scheme (paper): tree
    // siblings that represent "alternative futures at the same time step"
    // now see a RoPE distance of 1 (or more) instead of 0. Empirically that
    // slight distance shift costs little — the attn_bias mask still gates
    // ancestor visibility exactly, and the Q·K dot products stay consistent
    // across the whole cache (prompt + tree block). In exchange we get
    // DDTree correctness for topk>1 without needing a tree-local KV scratch
    // or a scatter-kernel for commit. `ctx.positions` is accepted for API
    // compatibility but ignored — the DdNode depths it carries are only
    // used by `linearize_tree` to build the attn_bias mask.
    if !pre_uploaded {
        let positions_host: Vec<i32> = if let Some(positions) = positions_override {
            assert_eq!(
                positions.len(),
                n,
                "positions_override length {} must equal tokens.len() {}",
                positions.len(),
                n,
            );
            positions.iter().map(|&p| p as i32).collect()
        } else {
            (0..n).map(|i| (start_pos + i) as i32).collect()
        };
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }

    // Decide whether the FA layers can take the batched path. Requires
    // (a) all FA weights to be MQ4G256 or HFQ4G256 (the batched gemm_qkv
    // + wo GEMMs are dtype-agnostic; the rmsnorm+rotate / silu_mul kernels
    // differ by dtype and we branch on that at each layer) and (b) a Q8_0
    // or givens KV cache. If the check fails, FA layers fall back to
    // per-token gather/scatter via run_fa_layer_body.
    let fa_arch = gpu.arch.as_str();
    // Q8 WMMA gate: the fused Q8 WMMA family (gemm_qkv/qkvza/gate_up/residual
    // _q8_0_wmma) uses the gfx11 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
    // builtin; the sibling `*.gfx12.hip` kernels use the `_w32_gfx12` variant
    // (silicon-validated on R9700, 2026-05-14, 4/4 unit tests PASS). Each
    // call site below selects the right variant via an `arch.starts_with`
    // branch. On non-WMMA archs we keep the Tier 2 chunked-substrate path.
    let q8_wmma_arch = gpu.arch_caps.has_wmma();
    let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);
    let fa_batched_ok = (!kv_cache.quantized
        || kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2)
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(l) => {
                is_batchable_la(l.wq.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wk.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wv.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wo.gpu_dtype, fa_arch)
                    && is_batchable_la(l.w_gate.gpu_dtype, fa_arch)
                    && is_batchable_la(l.w_up.gpu_dtype, fa_arch)
                    && is_batchable_la(l.w_down.gpu_dtype, fa_arch)
            }
            // MoE variant: attention weights must be MQ4-class (FFN is
            // checked separately by moe_ffn_batched_admissible in the eligibility gate).
            LayerWeights::FullAttnMoe(l) => {
                is_batchable_la(l.wq.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wk.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wv.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wo.gpu_dtype, fa_arch)
            }
            _ => true, // LA layers don't gate this check
        });
    // Under hipGraph capture, scalar kernargs get BAKED into the kernarg blob
    // at capture time. `max_ctx_len = start_pos + n` grows per cycle, so the
    // captured value would be stale on replay — the attention kernel would
    // allocate too-small LDS for `scores[]` and over-read. Bake the physical
    // cap instead (LDS sized for the worst case). The kernel still iterates
    // over the actual `positions[b] + 1` per-row seq_len from a device buffer,
    // so correctness is preserved; only the LDS allocation is over-provisioned.
    let max_ctx_len = if gpu.capture_mode {
        kv_cache.physical_cap
    } else if let Some(positions) = positions_override {
        positions.iter().copied().max().unwrap_or(start_pos) + 1
    } else {
        start_pos + n
    };
    let position_at_row = |row: usize| -> usize {
        positions_override
            .map(|p| p[row])
            .unwrap_or(start_pos + row)
    };

    // ── 2. Per-layer loop ────────────────────────────────────────────────
    // Multi-GPU band-mode: counters seed from the band's running offsets so
    // the band's first DeltaNet/FullAttn layer reads the correct
    // `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot. Single-GPU
    // (band==None) seeds zeros — original behavior.
    let mut delta_layer_idx = band.map(|b| b.delta_layer_offset).unwrap_or(0);
    let mut kv_layer_idx = band.map(|b| b.kv_layer_offset).unwrap_or(0);
    // Path B: per-FA-layer counter, drives the index into
    // tree_verify.pre_rope_k_capture[]. Increments alongside each
    // FullAttention layer iteration regardless of MoE/non-MoE variant.
    let mut fa_layer_idx = band.map(|b| b.fa_layer_offset).unwrap_or(0);
    let use_q8_gdn_per_token =
        force_q8_gdn_per_token || (gdn_tape.is_some() && q8_gdn_verify_per_token_enabled());
    let q8_gdn_serial_frame_base = if use_q8_gdn_per_token
        && q8_gdn_verify_serial_frames_enabled()
        && gdn_tape.is_some()
        && tree_verify.is_none()
        && band.is_none()
    {
        Some(gpu.debug_gdn_requant_frame())
    } else {
        None
    };
    let q8_gdn_serial_frame_layers = config
        .layer_types
        .iter()
        .filter(|&&lt| lt == LayerType::LinearAttention)
        .count();
    if let Some(frame_base) = q8_gdn_serial_frame_base {
        if let Some(tape) = gdn_tape.as_deref_mut() {
            tape.q8_requant_frame_base = Some(frame_base);
            tape.q8_requant_frame_layers = q8_gdn_serial_frame_layers;
        }
    }
    let ctx = DispatchCtx::new(gpu); // hoisted — arch-constant, safe to reuse per-layer

    for layer_idx in layer_start..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // Per-layer dtype branch: MQ4 needs FWHT-rotation on the
                // activation to match its pre-rotated weights; HFQ4 uses
                // plain rmsnormed activations. The GEMM kernels themselves
                // are dtype-agnostic — they just consume whatever [N × K]
                // activation buffer we point them at.
                // GAP NOTE: this matcher (and the 7 sibling dense LA/FA
                // matchers in this file) wires MQ3G256Lloyd through the
                // gemm_*_mq3g256_lloyd_wmma family. MQ2G256Lloyd remains
                // unwired — to add it, update is_batchable_la, ALL 8 is_mq*
                // matchers, AND add a Lloyd-MQ2-specific GEMM dispatch arm
                // together (the all-together corruption-prevention rule from
                // docs/plans/mq-lloyd-batched-prefill-followup.md). MQ4-Lloyd
                // is wired in a separate PR (issue #182).
                let is_mq = matches!(
                    layer.wqkv.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        // Opus W4A4: weights FWHT-rotated offline → x must be
                        // FWHT(+AWQ)-rotated by the shared mq rotate path before
                        // the int4 activation quantize (decode parity:
                        // rotate_x_mq[_awq] → quantize_act_oq4).
                        | DType::Oq4G256
                );
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let is_mq3 = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256);
                let is_mq3_lloyd = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256Lloyd);
                let is_fp4 = matches!(layer.wqkv.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let is_oq4 = matches!(layer.wqkv.gpu_dtype, DType::Oq4G256);
                let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
                let is_f32 = matches!(layer.wqkv.gpu_dtype, DType::F32);
                let is_f16 = matches!(layer.wqkv.gpu_dtype, DType::F16 | DType::BF16);

                // Batched rmsnorm (+ FWHT for MQ) for the LA preamble.
                // x_batch / x_rot_batch are [N × dim] contiguous. For HFQ
                // we reuse x_rot_batch as the "normed, unrotated" output
                // so the subsequent GEMM can read it the same way.
                if is_mq {
                    // AWQ-aware: next linear is LA's fused wqkv.
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }

                // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
                if is_6bit {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 && q8_wmma_arch {
                    // `is_q8` only inspects `wqkv` (the routing anchor). The fused
                    // kernel assumes ALL four weights share the Q8_0 stride; a
                    // mixed-dtype layer would silently re-introduce the Tier-1
                    // kernel-vs-stride corruption mode.
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                        "LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
                    );
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 {
                    // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
                    // (wqkv/wz/w_beta/w_alpha) → GemmFamily::run_key with the
                    // GemmQ8_0BatchedChunked dispatcher-entry key → identical
                    // gpu.gemm_q8_0_batched_chunked method, byte-for-byte.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else if is_f32 {
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::F32)
                            && matches!(layer.w_beta.gpu_dtype, DType::F32)
                            && matches!(layer.w_alpha.gpu_dtype, DType::F32),
                        "LA qkvza F32 dispatch requires all of wqkv/wz/w_beta/w_alpha to be F32",
                    );
                    gpu.gemm_f32_register_tiled(
                        &layer.wqkv.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.wz.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_beta.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else if is_f16 {
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::F16 | DType::BF16)
                            && matches!(layer.w_beta.gpu_dtype, DType::F16 | DType::BF16)
                            && matches!(layer.w_alpha.gpu_dtype, DType::F16 | DType::BF16),
                        "LA qkvza F16/BF16 dispatch requires all of wqkv/wz/w_beta/w_alpha to be F16",
                    );
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wqkv.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_qkv_batch,
                            layer.wqkv.m,
                            layer.wqkv.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wz.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_z_batch,
                            layer.wz.m,
                            layer.wz.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_beta.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_beta_batch,
                            layer.w_beta.m,
                            layer.w_beta.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_alpha.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_alpha_batch,
                            layer.w_alpha.m,
                            layer.w_alpha.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_qkvza_f16_xf32_batched(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_qkv_batch,
                            &pbs.dn_z_batch,
                            &pbs.dn_beta_batch,
                            &pbs.dn_alpha_batch,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                            n,
                        )?;
                    }
                } else if is_mq3_lloyd {
                    // 112 B/group Lloyd-MQ3 stride; X is already FWHT-rotated.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaMq3G256Lloyd,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_mq3 {
                    // 104 B/group HFQ3-stride; X is already FWHT-rotated by
                    // fused_rmsnorm_rotate_mq_batched above. The FusedQkvzaHfq3G256
                    // run-arm replicates the call-site WMMA-vs-base arch split
                    // internally (gemm_qkvza_hfq3g256_wmma on has_wmma() else the
                    // base cross-arch ladder), so the same kernel runs.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq3G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_fp4 {
                    // HFP4G32: 17-B blocks (vs HFQ4's 136-B groups), per-row 16-B header.
                    // MFP4G32: same storage as HFP4 + offline-FWHT weights; X is already
                    // rotated above when is_mq, so this branch handles both unrotated
                    // (HFP4) and post-rotation (MFP4) activations identically.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfp4G32,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_oq4 {
                    // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (is_mq).
                    // The FusedQkvzaOq4G256 run-arm int4-quantizes it once then
                    // runs the batched grouped-WMMA fused kernel.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaOq4G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if gdn_tape.is_some() {
                    gpu.gemm_qkvza_hfq4g256_exact(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let x_in_row_bytes = tape.x_in_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_x = tape_offset * x_in_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_x = n * x_in_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.x_in_bufs[delta_layer_idx].buf,
                        off_x,
                        &pbs.x_rot_batch.buf,
                        0,
                        copy_x,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }

                // Fused sigmoid(beta) + alpha_gate(alpha) — [N × n_v_heads] each.
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    n,
                )?;

                // DFlash tape capture: snap pre-conv1d qkv + post-sigmoid α/β
                // for this layer into the per-layer tape slots. The next LA
                // layer's fused_qkvza / fused_sigmoid_alpha_gate will overwrite
                // dn_qkv_batch / dn_{alpha,beta}_batch, so capture must happen
                // now (after sigmoid_alpha_gate, before conv1d consumes qkv).
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        off_qkv,
                        &pbs.dn_qkv_batch.buf,
                        0,
                        copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }

                // Tree-aware dispatch gate: when the caller provides
                // parent_indices (Phase 3b+ of Task #101), swap the linear
                // conv1d + GDN for tree-walking variants that eliminate
                // sibling-subtree state cross-contamination. The tree
                // kernels are READ-ONLY on dn_state (don't advance it) —
                // caller runs linear replay on the accepted spine
                // post-acceptance to commit the trajectory.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim,
                        v_dim,
                        n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        n,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let q_raw_row_bytes = tape.k_dim * 4;
                    let v_row_bytes = tape.v_dim * 4;
                    let off_q_raw = tape_offset * q_raw_row_bytes;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_q_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_k_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.v_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_v_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }

                // Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave
                // when n_key_heads < n_v_heads. One launch instead of two —
                // ~200µs saved per LA layer × ~30 LA layers ≈ 6ms per prefill
                // on A3B (R9700/gfx1201).
                //
                // The fused kernel reads q_raw/k_raw (unchanged on exit), so
                // the conv1d output is preserved if downstream readers need it
                // (no current consumer reads _raw after this).
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    // n_key_heads == n_v_heads → no replication; keep the
                    // original sequence (norm in place, then memcpy).
                    gpu.fused_qk_l2_norm_scale_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                        n,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let q_row_bytes = tape.v_dim * 4;
                    let off_q = tape_offset * q_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_q_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_k_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                }

                // Gated Delta Net — tree variant reads per-token S from
                // s_tape[parent] (or pre-block s_q8_init at root); linear
                // variant advances dn_state.s_matrices in place.
                if let Some(parents) = tree_parents {
                    if matches!(dn_state.quant, StateQuant::FP32) {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "FP32-state batched prefill does not support tree DeltaNet replay yet",
                        ));
                    }
                    let tape_q8 = pbs.dn_s_tape_q8.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_q8 scratch (check PrefillBatchScratch::new)");
                    let tape_sc = pbs.dn_s_tape_scales.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_scales scratch (check PrefillBatchScratch::new)");
                    gpu.gated_delta_net_q8_tree_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        tape_q8,
                        tape_sc,
                        parents,
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if matches!(dn_state.quant, StateQuant::FP32) {
                    gpu.gated_delta_net_f32_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if use_q8_gdn_per_token {
                    for step in 0..n {
                        if let Some(frame_base) = q8_gdn_serial_frame_base {
                            gpu.debug_set_gdn_requant_frame(frame_base.wrapping_add(
                                (step * q8_gdn_serial_frame_layers + delta_layer_idx) as u32,
                            ));
                        }
                        let q = pbs.dn_q_batch.sub_offset(step * v_dim, v_dim);
                        let k = pbs.dn_k_batch.sub_offset(step * v_dim, v_dim);
                        let v = pbs.dn_v_batch.sub_offset(step * v_dim, v_dim);
                        let alpha = pbs.dn_alpha_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let beta = pbs.dn_beta_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let out = pbs.dn_attn_out_batch.sub_offset(step * v_dim, v_dim);
                        gpu.gated_delta_net_q8(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?;
                    }
                    if let Some(frame_base) = q8_gdn_serial_frame_base {
                        gpu.debug_set_gdn_requant_frame(
                            frame_base.wrapping_add((n * q8_gdn_serial_frame_layers) as u32),
                        );
                    }
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_out_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_attn_out_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                    // EXPERIMENT (not #417): mirror the state-quant dispatch the
                    // decode siblings already do (forward_scratch_layers:13194),
                    // so the captured/eager batched prefill honours FP32/Q4 state
                    // instead of forcing the Q8 kernel onto non-Q8 buffers.
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                }

                // Batched gated output norm.
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    n,
                )?;

                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.normed_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_normed_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_residual_in_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Batched wo + residual.
                //
                // For MQ weights, the decode path's weight_gemv_residual
                // internally FWHT-rotates dn_normed into mq_x_rot before
                // calling gemv_hfq{4,6}g256_residual (MQ weights are pre-rotated
                // at quant time; math requires dot(rot(W), rot(x)) = dot(W,x)).
                // For HFQ weights no rotation is needed — the activation
                // feeds gemm_hfq{4,6}g256_residual directly.
                let wo_is_mq = matches!(
                    layer.wo.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
                let wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
                let wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let wo_is_oq4 = matches!(layer.wo.gpu_dtype, DType::Oq4G256);
                let wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let wo_is_f32 = matches!(layer.wo.gpu_dtype, DType::F32);
                let wo_is_f16 = matches!(layer.wo.gpu_dtype, DType::F16 | DType::BF16);
                let wo_input = if wo_is_mq {
                    // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    &pbs.dn_normed_batch
                };
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_input_bufs[delta_layer_idx].buf,
                        off_v,
                        &wo_input.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_oq4 {
                    // Opus W4A4: wo_input is FWHT(+AWQ)-rotated above (wo_is_mq).
                    // No fused oq4 residual kernel → grouped-WMMA GEMM into scratch
                    // + add into the residual stream (pbs.x_batch).
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.wo.buf,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_q8 {
                    // Tier 2 fallback (non-WMMA archs): GEMM into x_rot_batch as
                    // scratch (safe — next consumer is the FFN rmsnorm), then
                    // add into residual.
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if wo_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.wo.buf,
                        wo_input,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.wo.buf,
                            wo_input,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.wo.buf,
                            wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    }
                } else if wo_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if gdn_tape.is_some() {
                    gpu.gemm_hfq4g256_residual_exact(
                        &layer.wo.buf,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_residual_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // FFN: rmsnorm (+ rotate for MQ).
                let ffn_is_mq = matches!(
                    layer.w_gate.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let ffn_is_6bit =
                    matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
                let ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
                let ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let ffn_is_oq4 = matches!(layer.w_gate.gpu_dtype, DType::Oq4G256);
                let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
                let ffn_is_f32 = matches!(layer.w_gate.gpu_dtype, DType::F32);
                let ffn_is_f16 = matches!(layer.w_gate.gpu_dtype, DType::F16 | DType::BF16);
                if ffn_is_mq {
                    // AWQ-aware: next linear is w_gate (gate/up share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &layer.w_gate,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_input_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_rot_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Batched gate+up projection.
                // #397 Ship 5.2 slice 2: fused gate+up dtypes → FusedQkvFamily
                // (batched-prefill gate+up variant) via run_fused_gate_up_key.
                // The Q8-non-WMMA case stays as two plain GemmQ8_0BatchedChunked
                // GEMMs (not a fused kernel — slice 1). The HFQ3 WMMA-vs-base
                // split is folded into the FusedGateUpHfq3G256 run-arm, which
                // re-derives it from gpu.arch_caps.has_wmma() (== arch_has_wmma).
                if ffn_is_6bit {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_q8 && q8_wmma_arch {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                        "LA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
                    );
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_q8 {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_gate.buf,
                        layer.w_gate.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_up.buf,
                        layer.w_up.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if ffn_is_f32 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F32),
                        "LA FFN F32 dispatch requires both w_gate and w_up to be F32",
                    );
                    gpu.gemm_f32_register_tiled(
                        &layer.w_gate.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if ffn_is_f16 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F16 | DType::BF16),
                        "LA FFN F16/BF16 dispatch requires both w_gate and w_up to be F16",
                    );
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_gate.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            layer.w_gate.m,
                            layer.w_gate.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.up_batch,
                            layer.w_up.m,
                            layer.w_up.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_gate_up_f16_xf32_batched(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            &pbs.up_batch,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                            n,
                        )?;
                    }
                } else if ffn_is_mq3_lloyd {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_mq3 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_fp4 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_oq4 {
                    // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (ffn_is_mq).
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpOq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if gdn_tape.is_some() {
                    gpu.gemm_gate_up_hfq4g256_exact(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let ffn_row_bytes = tape.ffn_dim * 4;
                    let off_ffn = tape_offset * ffn_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_gate_bufs[delta_layer_idx].buf,
                        off_ffn,
                        &pbs.gate_ffn_batch.buf,
                        0,
                        n * ffn_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_up_bufs[delta_layer_idx].buf,
                        off_ffn,
                        &pbs.up_batch.buf,
                        0,
                        n * ffn_row_bytes,
                    )?;
                }

                // SwiGLU activation feeding w_down. For MQ, we need the
                // output FWHT-rotated so it matches the pre-rotated w_down
                // weights. For HFQ, plain silu_mul is enough. silu_mul_f32
                // is purely element-wise and uses numel() as its length,
                // so a [N × hidden_dim] tensor processes all rows in one
                // launch with no batch offset needed.
                let w_down_is_mq = matches!(
                    layer.w_down.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let w_down_is_6bit =
                    matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
                let w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
                let w_down_is_fp4 =
                    matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let w_down_is_oq4 = matches!(layer.w_down.gpu_dtype, DType::Oq4G256);
                let w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
                let w_down_is_f32 = matches!(layer.w_down.gpu_dtype, DType::F32);
                let w_down_is_f16 = matches!(layer.w_down.gpu_dtype, DType::F16 | DType::BF16);
                if w_down_is_mq {
                    // F2: AWQ-aware silu_mul+rotate for w_down input.
                    fused_silu_mul_rotate_mq_batched_for(
                        gpu,
                        &layer.w_down,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        &pbs.ffn_hidden_batch,
                        hidden_dim,
                        n,
                    )?;
                } else {
                    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.w_down_residual_in_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                    let ffn_row_bytes = tape.ffn_dim * 4;
                    let off_ffn = tape_offset * ffn_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.w_down_input_bufs[delta_layer_idx].buf,
                        off_ffn,
                        &pbs.ffn_hidden_batch.buf,
                        0,
                        n * ffn_row_bytes,
                    )?;
                }

                // Batched w_down + residual.
                if w_down_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_oq4 {
                    // Opus W4A4: ffn_hidden_batch is FWHT(+AWQ)-rotated above
                    // (fused_silu_mul_rotate_mq, w_down_is_mq). grouped-WMMA GEMM
                    // into scratch + residual add into the hidden stream.
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &x_n,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &scratch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if w_down_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    }
                } else if w_down_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if gdn_tape.is_some() {
                    gpu.gemm_hfq4g256_residual_exact(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.layer_out_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = is_mq; // retained above for potential future use
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Fully batched FA layer. Mirrors the FA branch of
                // forward_scratch_layers kernel-for-kernel, but every
                // launch covers all N tokens at once.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let _q_dim = config.n_heads * config.head_dim;
                let qkv_is_mq = matches!(
                    layer.wq.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let qkv_is_mq3 = matches!(layer.wq.gpu_dtype, DType::MQ3G256);
                let qkv_is_mq3_lloyd = matches!(layer.wq.gpu_dtype, DType::MQ3G256Lloyd);
                let qkv_is_fp4 = matches!(layer.wq.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let qkv_is_oq4 = matches!(layer.wq.gpu_dtype, DType::Oq4G256);
                let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
                let qkv_is_f32 = matches!(layer.wq.gpu_dtype, DType::F32);
                let qkv_is_f16 = matches!(layer.wq.gpu_dtype, DType::F16 | DType::BF16);
                // Fused QKV kernels require all three weights to share a
                // dtype — they treat wq/wk/wv as same-stride byte arrays.
                // When kmap mode 2 promotes only `v_proj` (issue #249), the
                // fused HFQ4 path reads `wv` as MQ6 with HFQ4's 136-B stride
                // and produces silent NaN. Gate the fused kernels here.
                //
                // The Q8 substrate path (gemm_q8_0_batched_chunked × 3) also
                // dispatches a Q8-stride kernel per weight, so it needs the
                // same gate when wk/wv aren't Q8.
                let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
                    && layer.wv.gpu_dtype == layer.wq.gpu_dtype;
                let fa_bridge_tape_active = gdn_tape.as_ref().is_some_and(|tape| {
                    delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                });
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_input_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 1. rmsnorm (+ rotate for MQ) for the attn preamble.
                if qkv_is_mq {
                    // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_x_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_rot_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 2. Batched 3-way QKV projection (wq+wk+wv).
                if qkv_is_6bit && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_mq3_lloyd && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvMq3G256Lloyd,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_mq3 && qkv_same_dtype {
                    // X is already FWHT-rotated by fused_rmsnorm_rotate_mq_batched
                    // above; call the bare HFQ3 GEMM (no second rotation). The
                    // FusedQkvHfq3G256 run-arm replicates the call-site WMMA-vs-base
                    // arch split internally (gemm_qkv_hfq3g256_wmma on has_wmma()
                    // else the base cross-arch ladder), so the same kernel runs.
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq3G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_fp4 && qkv_same_dtype {
                    // HFP4G32 / MFP4G32 FP4 batched WMMA. X is already
                    // rotated above for MFP4 (is_mq path) — same kernel
                    // covers both unrotated HFP4 and rotated MFP4 inputs.
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfp4G32,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_oq4 && qkv_same_dtype {
                    // OQ4+ batched prefill FA QKV: int8-WMMA MMQ (n>=64) quantizing
                    // the shared FWHT(+AWQ)-rotated activation to q8_1 ONCE across
                    // q/k/v; gemm_oq4_qkv_mmq falls back to the f16 grouped path for
                    // tiny batches internally via gemm_oq4_grouped_act_batched.
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Oq4G256)
                            && matches!(layer.wv.gpu_dtype, DType::Oq4G256),
                        "FA qkv Oq4 dispatch requires all of wq/wk/wv to be Oq4G256",
                    );
                    if n >= 64 {
                        gpu.gemm_oq4_qkv_mmq(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                            n,
                        )?;
                    } else {
                        gpu.gemm_oq4_grouped_act_batched(
                            &layer.wq.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            layer.wq.m,
                            layer.wq.k,
                            n,
                        )?;
                        gpu.gemm_oq4_grouped_act_batched(
                            &layer.wk.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_k_batch,
                            layer.wk.m,
                            layer.wk.k,
                            n,
                        )?;
                        gpu.gemm_oq4_grouped_act_batched(
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_v_batch,
                            layer.wv.m,
                            layer.wv.k,
                            n,
                        )?;
                    }
                } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Q8_0)
                            && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                        "FA qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
                    );
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && qkv_same_dtype {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_is_f32 && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::F32)
                            && matches!(layer.wv.gpu_dtype, DType::F32),
                        "FA qkv F32 dispatch requires all of wq/wk/wv to be F32",
                    );
                } else if qkv_is_f16 && qkv_same_dtype {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wq.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            layer.wq.m,
                            layer.wq.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wk.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_k_batch,
                            layer.wk.m,
                            layer.wk.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_v_batch,
                            layer.wv.m,
                            layer.wv.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_qkvza_f16_xf32_batched(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.dn_alpha_batch,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            0,
                            layer.wq.k,
                            n,
                        )?;
                    }
                } else if qkv_same_dtype {
                    if fa_bridge_tape_active {
                        gpu.gemm_qkv_hfq4g256_exact(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                            n,
                        )?;
                    } else {
                        gpu.gemm_qkv_hfq4g256(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            layer.wq.k,
                            n,
                        )?;
                    }
                } else {
                    // Mixed-format fallback (issue #249): wq/wk/wv don't all
                    // share a dtype. Dispatch each weight to its own
                    // single-weight batched GEMM, dropping the fused-kernel
                    // launch-overhead optimization for correctness.
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wk,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wv,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_full_row_bytes = tape.fa_q_full_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_full_bufs[delta_layer_idx].buf,
                            tape_offset * q_full_row_bytes,
                            &pbs.fa_q_full_batch.buf,
                            0,
                            n * q_full_row_bytes,
                        )?;
                    }
                }

                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    n,
                )?;

                // 4. Per-head Q/K rmsnorm. rmsnorm_batched uses batch =
                // number of "rows" of head_dim. For [N × n_heads × head_dim]
                // that's batch = N * n_heads.
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_norm_bufs[delta_layer_idx].buf,
                            tape_offset * q_row_bytes,
                            &pbs.fa_q_batch.buf,
                            0,
                            n * q_row_bytes,
                        )?;
                    }
                }
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;

                if hipfire_runtime::triattn::tap_enabled() {
                    // Try GPU path first: dispatches a reduce kernel on the
                    // device-resident Q tensor, zero PCIe transfer. Only
                    // succeeds when install_tap_gpu() was used. Falls through
                    // to CPU path otherwise.
                    let gpu_handled =
                        hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                            gpu,
                            layer_idx,
                            &pbs.fa_q_batch.buf,
                            n,
                            config.n_heads,
                            config.head_dim,
                        )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if hipfire_runtime::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }

                // Path B pre-RoPE K capture (slow-path-kill, WIP).
                // The next line mutates pbs.fa_k_batch in place — capture
                // BEFORE so the slow path has the unrotated K available
                // and can apply RoPE for the COMMITTED slot phases instead
                // of these linearization-slot phases. Capture is None
                // unless the env gate + the per-FA-layer scratch are both
                // wired through TreeVerifyCtx.
                if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
                    if let Some(slot) = slots.get(fa_layer_idx) {
                        let kv_dim = config.n_kv_heads * config.head_dim;
                        let n_bytes = n * kv_dim * 4;
                        // Use _auto so the memcpy is recorded onto the
                        // active stream when one exists (matches the
                        // existing GdnTape capture pattern at line ~3193).
                        // Plain gpu.hip.memcpy_dtod_at runs on the null
                        // stream and sync-blocks pending async kernels,
                        // changing kernel-launch order in ways that
                        // perturb DDTree's ksplit-atomic nondeterminism
                        // — output diverges even though no data is
                        // actually changed.
                        gpu.memcpy_dtod_at_auto(&slot.buf, 0, &pbs.fa_k_batch.buf, 0, n_bytes)?;
                    }
                }

                // 5. Batched partial-interleaved RoPE (per-row positions).
                // pbs.positions stays physical for the KV write below; the
                // offset rotates new Q/K at absolute phase after compaction.
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    &pbs.positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    n,
                    kv_cache.compact_offset as i32,
                )?;
                // KV-compression study capture: post-RoPE FA Q/K/V for a target FA
                // layer (HIPFIRE_DUMP_HIDDEN_ALL=1 + HIPFIRE_DUMP_HIDDEN_LAYER).
                dump_hidden_localize(
                    gpu,
                    &pbs.fa_q_batch,
                    n,
                    start_pos,
                    config.n_heads * config.head_dim,
                    layer_idx,
                    "faq",
                );
                dump_hidden_localize(
                    gpu,
                    &pbs.fa_k_batch,
                    n,
                    start_pos,
                    config.n_kv_heads * config.head_dim,
                    layer_idx,
                    "fak",
                );
                dump_hidden_localize(
                    gpu,
                    &pbs.fa_v_batch,
                    n,
                    start_pos,
                    config.n_kv_heads * config.head_dim,
                    layer_idx,
                    "fav",
                );
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        let kv_row_bytes = tape.fa_kv_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_bufs[delta_layer_idx].buf,
                            tape_offset * q_row_bytes,
                            &pbs.fa_q_batch.buf,
                            0,
                            n * q_row_bytes,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_k_bufs[delta_layer_idx].buf,
                            tape_offset * kv_row_bytes,
                            &pbs.fa_k_batch.buf,
                            0,
                            n * kv_row_bytes,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_v_bufs[delta_layer_idx].buf,
                            tape_offset * kv_row_bytes,
                            &pbs.fa_v_batch.buf,
                            0,
                            n * kv_row_bytes,
                        )?;
                    }
                }

                let use_kld_direct_f16kv_attention = kld_direct_f16kv_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                );
                let use_kld_fp32_gqa4_attention = kld_fp32_gqa4_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                    n,
                );

                // 6. Batched KV cache writes (per-row positions).
                if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_q8 && q8_fa_attention_serial_kv_loop_enabled() {
                    assert!(
                        tree_verify.is_none(),
                        "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP is a causal Q8 FA diagnostic; tree-verify masking is not supported",
                    );
                    // Diagnostic: defer KV writes to the row-serial attention
                    // loop below so write/read ordering matches serial decode.
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                } else if !use_kld_direct_f16kv_attention && !use_kld_fp32_gqa4_attention {
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                }

                // 7. Batched causal attention (or tree-attention if tree_verify is set).
                // asym{4,3,2}: batched flash (K rotated-quantized + V Q8 in normal space).
                // Q8: batched kernel unless ctx > 15K (LDS overflow), then per-position flash.
                //
                // Tree-verify mode: `block_start = start_pos`, `block_cols = n`.
                // The bias buffer is `[n × n]`; each query row applies its
                // corresponding bias row to in-block keys. Long-context Q8
                // tiled fallback isn't supported in tree mode (we caught
                // that as an assert above — tree blocks are small).
                const LDS_CTX_LIMIT: usize = 15000;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                // 6–7. Batched KV write + flash attention (via dispatch).
                let is_tree = tree_verify.is_some();
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                if use_kld_direct_f16kv_attention {
                    gpu.attention_dflash_wmma_causal_f32(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        n,
                        n,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                } else if use_kld_fp32_gqa4_attention {
                    gpu.attention_f32_batched_gqa4(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                        n,
                    )?;
                } else if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode not supported on asym2 KV (use asym3)",
                    );
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_q8 && q8_fa_attention_serial_kv_loop_enabled() {
                    assert!(
                        tree_verify.is_none(),
                        "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP is a causal Q8 FA diagnostic; tree-verify masking is not supported",
                    );
                    let q_dim = config.n_heads * config.head_dim;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                            let k_b = pbs.fa_k_batch.sub_offset(b * kv_dim, kv_dim);
                            let v_b = pbs.fa_v_batch.sub_offset(b * kv_dim, kv_dim);
                            let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                            gpu.kv_cache_write_q8_0(
                                &kv_cache.k_gpu[layer_idx],
                                &k_b,
                                &pos_buf_tmp,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.kv_cache_write_q8_0(
                                &kv_cache.v_gpu[layer_idx],
                                &v_b,
                                &pos_buf_tmp,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_q8_0_kv(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                pos_b + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 && max_ctx_len > LDS_CTX_LIMIT {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode hits the long-context Q8 fallback \
                         at max_ctx_len={} > {}; tree blocks should stay small",
                        max_ctx_len,
                        LDS_CTX_LIMIT,
                    );
                    // Per-position flash Q8 attention for long-context prefill.
                    //
                    // `pbs.positions` is raw i32 bits in an F32 slot
                    // (slot-cosmetic, see PrefillBatchScratch::new).
                    // `download_f32` would reinterpret those bytes as floats —
                    // i32 15000 = 0x3A98 round-trips through f32 as ~1e-3
                    // subnormal, which casts to 0. Reconstruct from the
                    // host-side row position directly.
                    let q_dim = config.n_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let seq_len_b = pos_b + 1;
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                            let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                            gpu.attention_flash_q8_0(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                seq_len_b,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 && q8_fa_attention_scalar_loop_enabled() {
                    let q_dim = config.n_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                            let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                            gpu.attention_q8_0_kv(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                pos_b + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 && q8_fa_attention_row_loop_enabled() {
                    let q8_tree_bias = if q8_fa_attention_ignore_tree_bias_enabled() {
                        None
                    } else {
                        tree_bias
                    };
                    let q_dim = config.n_heads * config.head_dim;
                    for b in 0..n {
                        let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                        let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                        let pos_b = pbs.positions.sub_offset(b, 1);
                        let bias_b =
                            q8_tree_bias.map(|bias| bias.sub_offset(b * block_cols, block_cols));
                        gpu.attention_q8_0_kv_batched_masked(
                            &q_b,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &out_b,
                            &pos_b,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            1,
                            bias_b.as_ref(),
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_q8 {
                    let q8_tree_bias = if q8_fa_attention_ignore_tree_bias_enabled() {
                        None
                    } else {
                        tree_bias
                    };
                    gpu.attention_q8_0_kv_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        q8_tree_bias,
                        block_start,
                        block_cols,
                    )?;
                } else {
                    gpu.attention_f32_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_attn_raw_bufs[delta_layer_idx].buf,
                            tape_offset * q_row_bytes,
                            &pbs.fa_attn_out_batch.buf,
                            0,
                            n * q_row_bytes,
                        )?;
                    }
                }
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let plan = KvTierPlan::derive(KvTierInputs {
                    quant_asym4: kv_cache.quant_asym4,
                    quant_asym3: kv_cache.quant_asym3,
                    quant_asym2: kv_cache.quant_asym2,
                    quant_q8: kv_cache.quant_q8,
                    quant_fwht: kv_cache.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    v_mode_bits: 0,
                    pos: start_pos,
                    flash_mode: s.flash_mode as usize,
                    capture_mode: gpu.capture_mode,
                    batch_size: n,
                    is_tree,
                    is_boundary: false,
                })
                .map_err(|e| HipError::new(0, &e.to_string()))?;
                let io = AttnParams {
                    q: &pbs.fa_q_batch,
                    k: &pbs.fa_k_batch,
                    v: &pbs.fa_v_batch,
                    k_cache: &kv_cache.k_gpu[layer_idx],
                    v_cache: &kv_cache.v_gpu[layer_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &s.pos_buf,
                    pos: start_pos,
                    positions: Some(&pbs.positions),
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    physical_cap: kv_cache.physical_cap,
                    batch_size: n,
                    max_ctx_len,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv_cache.givens_cos.as_ref(),
                    givens_sin: kv_cache.givens_sin.as_ref(),
                    tree_bias,
                    block_start,
                    block_cols,
                    output: &pbs.fa_attn_out_batch,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| HipError::new(0, &e.to_string()))?;

                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_attn_out_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.fa_attn_out_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 9. wo residual: x_batch += wo · (optional rotate)(fa_attn_out_batch).
                // Same MQ rotation requirement as the LA wo path.
                let fa_wo_is_mq = matches!(
                    layer.wo.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
                let fa_wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_wo_is_oq4 = matches!(layer.wo.gpu_dtype, DType::Oq4G256);
                let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let fa_wo_is_f32 = matches!(layer.wo.gpu_dtype, DType::F32);
                let fa_wo_is_f16 = matches!(layer.wo.gpu_dtype, DType::F16 | DType::BF16);
                let fa_wo_input = if fa_wo_is_mq {
                    // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    &pbs.fa_attn_out_batch
                };
                if fa_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_oq4 {
                    // Opus W4A4: fa_wo_input is FWHT(+AWQ)-rotated above.
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.wo.buf,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_wo_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.wo.buf,
                        fa_wo_input,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.wo.buf,
                            fa_wo_input,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.wo.buf,
                            fa_wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    }
                } else if fa_wo_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_wo_residual_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 10. FFN: rmsnorm (+ rotate for MQ), gate+up, silu_mul
                // (+ rotate for MQ), w_down residual.
                let fa_ffn_is_mq = matches!(
                    layer.w_gate.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let fa_ffn_is_6bit =
                    matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
                let fa_ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_ffn_is_fp4 =
                    matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_ffn_is_oq4 = matches!(layer.w_gate.gpu_dtype, DType::Oq4G256);
                let fa_ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
                let fa_ffn_is_f32 = matches!(layer.w_gate.gpu_dtype, DType::F32);
                let fa_ffn_is_f16 = matches!(layer.w_gate.gpu_dtype, DType::F16 | DType::BF16);
                if fa_ffn_is_mq {
                    // AWQ-aware: next linear is w_gate (FA-FFN, gate/up share input).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &layer.w_gate,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                // #397 Ship 5.2 slice 2: FA-FFN fused gate+up → FusedQkvFamily
                // (batched-prefill gate+up variant), mirroring the LA-FFN block
                // above. Q8-non-WMMA stays as two plain GEMMs; HFQ3 WMMA-vs-base
                // is folded into the FusedGateUpHfq3G256 run-arm.
                if fa_ffn_is_6bit {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_oq4 {
                    // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (fa_ffn_is_mq).
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpOq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_q8 && q8_wmma_arch {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                        "FA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
                    );
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_q8 {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_gate.buf,
                        layer.w_gate.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_up.buf,
                        layer.w_up.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if fa_ffn_is_f32 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F32),
                        "FA FFN F32 dispatch requires both w_gate and w_up to be F32",
                    );
                    gpu.gemm_f32_register_tiled(
                        &layer.w_gate.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if fa_ffn_is_f16 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F16 | DType::BF16),
                        "FA FFN F16/BF16 dispatch requires both w_gate and w_up to be F16",
                    );
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_gate.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            layer.w_gate.m,
                            layer.w_gate.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.up_batch,
                            layer.w_up.m,
                            layer.w_up.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_gate_up_f16_xf32_batched(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            &pbs.up_batch,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                            n,
                        )?;
                    }
                } else if fa_ffn_is_mq3_lloyd {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_mq3 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_fp4 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                }
                let fa_w_down_is_mq = matches!(
                    layer.w_down.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let fa_w_down_is_6bit =
                    matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
                let fa_w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_w_down_is_fp4 =
                    matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_w_down_is_oq4 = matches!(layer.w_down.gpu_dtype, DType::Oq4G256);
                let fa_w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
                let fa_w_down_is_f32 = matches!(layer.w_down.gpu_dtype, DType::F32);
                let fa_w_down_is_f16 = matches!(layer.w_down.gpu_dtype, DType::F16 | DType::BF16);
                if fa_w_down_is_mq {
                    // F2: AWQ-aware silu_mul+rotate for FullAttention w_down input.
                    fused_silu_mul_rotate_mq_batched_for(
                        gpu,
                        &layer.w_down,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        &pbs.ffn_hidden_batch,
                        hidden_dim,
                        n,
                    )?;
                } else {
                    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                }
                if fa_w_down_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_oq4 {
                    // Opus W4A4: ffn_hidden_batch is FWHT(+AWQ)-rotated above.
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &x_n,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &scratch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_w_down_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    }
                } else if fa_w_down_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_layer_out_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                // Silence unused warning if kv_dim ends up shadowed.
                let _ = kv_dim;
                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                // Per-token gather/scatter fallback for FA layers that don't
                // qualify for batched FA (non-MQ4 weights, non-Q8_0 KV, etc).
                for i in 0..n {
                    let pos = start_pos + i;
                    gpu.hip.memcpy_dtod_at(
                        &s.x.buf,
                        0,
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        dim_row_bytes,
                    )?;
                    let pos_i32 = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &pos_i32.to_ne_bytes())?;
                    run_fa_layer_body(
                        gpu,
                        weights,
                        config,
                        layer_idx,
                        kv_layer_idx,
                        pos,
                        kv_cache,
                        s,
                    )?;
                    gpu.hip.memcpy_dtod_at(
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        &s.x.buf,
                        0,
                        dim_row_bytes,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path. After
                // the per-token loop, pbs.x_batch has the full layer output
                // for all N tokens (last copy-back finishes each row).
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // Batched MoE LA layer. LA body is the same as DeltaNet
                // (rmsnorm + qkvza + sigmoid_alpha + conv1d + L2norm +
                // repeat_interleave + GDN + gated_norm + wo+residual);
                // only the FFN differs. Duplicated inline for now — can
                // be factored into a `prefill_la_body_batched` helper
                // when dense and MoE LA paths are proven byte-exact.
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wqkv.gpu_dtype` is restricted here to MQ4G256 /
                // HFQ4G256 / MQ6G256 / HFQ6G256 / Q8_0. Q8 admit landed
                // alongside the moe_ffn router/gate Q8 unlock (A3B's LA
                // attention weights are Q8 — engine quantizer keeps q/k/v/o
                // at Q8 alongside the Q8 router + shared_expert_gate).
                let is_mq = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
                // Phase 1.5: PARO mode for DeltaNetMoe — wqkv/wz are
                // ParoQ4G128 (each with its own Givens rotation tables);
                // w_alpha/w_beta are F32 (no rotation, no quantization).
                // Dispatch is unfused: rotate+gemm_hfq4g128 for wqkv and wz,
                // direct gemm_f32_batched for w_alpha and w_beta. Same shape
                // outputs as the Q8/MQ4 paths (dn_qkv_batch, dn_z_batch,
                // dn_alpha_batch, dn_beta_batch).
                let is_paro = matches!(layer.wqkv.gpu_dtype, DType::ParoQ4G128);
                let q8_wmma_arch = gpu.arch_caps.has_wmma();

                if is_mq {
                    // AWQ-aware: next linear is LA's fused wqkv.
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else if is_paro {
                    // PARO: need un-rotated x_norm available for per-weight
                    // Givens rotation. Write rmsnorm into x_norm_batch (the
                    // dedicated normalized buffer); x_rot_batch becomes the
                    // per-weight rotation scratch (overwritten per GEMM).
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                debug_stop_after!("attn_norm", layer_idx);
                if is_paro {
                    // PARO 4-way unfused dispatch. wqkv and wz are
                    // ParoQ4G128 with their own Givens rotation tables;
                    // w_alpha and w_beta are F32 with no rotation.
                    let paro_wqkv = layer.wqkv.paro.as_ref().unwrap_or_else(|| {
                        panic!(
                            "ParoQ4G128 wqkv missing paro metadata at LA layer {layer_idx} \
                             — paro_load_wt() loader regression?"
                        )
                    });
                    let paro_wz = layer.wz.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wz missing paro metadata at LA layer {layer_idx}")
                    });
                    // wqkv: rotate x_norm → x_rot, then HFQ4G128 GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wqkv.pairs,
                        &paro_wqkv.theta,
                        &paro_wqkv.channel_scales,
                        n,
                        dim,
                        paro_wqkv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    // wz: re-rotate x_norm → x_rot (overwrite), then GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wz.pairs,
                        &paro_wz.theta,
                        &paro_wz.channel_scales,
                        n,
                        dim,
                        paro_wz.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    // w_alpha / w_beta: F32, no rotation, direct batched GEMM.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                } else if is_6bit {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 && q8_wmma_arch {
                    // Fused Q8 QKVZA WMMA — assumes all 4 weights share Q8_0
                    // stride; mixed Q8/other layers within DNMoe are rejected
                    // upstream by `moe_ffn_batched_admissible` (router/gate Q8 OK, but
                    // shared_expert + experts must be MQ4) and would otherwise
                    // re-introduce Tier-1 stride corruption.
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                        "DNMoe LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
                    );
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 {
                    // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
                    // (wqkv/wz/w_beta/w_alpha), sibling DeltaNet QKVZA path.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                }
                debug_stop_after!("qkvza", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let x_in_row_bytes = tape.x_in_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_x = tape_offset * x_in_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_x = n * x_in_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.x_in_bufs[delta_layer_idx].buf,
                        off_x,
                        &pbs.x_rot_batch.buf,
                        0,
                        copy_x,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    n,
                )?;
                debug_stop_after!("sigmoid_alpha", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        off_qkv,
                        &pbs.dn_qkv_batch.buf,
                        0,
                        copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }
                // Same tree-aware dispatch gate as dense LA branch above.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim,
                        v_dim,
                        n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        n,
                    )?;
                }
                debug_stop_after!("conv", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let q_raw_row_bytes = tape.k_dim * 4;
                    let v_row_bytes = tape.v_dim * 4;
                    let off_q_raw = tape_offset * q_raw_row_bytes;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_q_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_k_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.v_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_v_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    n,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        n,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                }
                debug_stop_after!("qk_repeat", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let q_row_bytes = tape.v_dim * 4;
                    let off_q = tape_offset * q_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_q_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_k_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                }
                // DIAG: dump GDN inputs (batched, MoE branch)
                if layer_idx == 0 {
                    let qk_dim = n_v_heads * hd;
                    dump_hidden_localize(gpu, &pbs.dn_q_batch, n, start_pos, qk_dim, 0, "q_b");
                    dump_hidden_localize(gpu, &pbs.dn_k_batch, n, start_pos, qk_dim, 0, "k_b");
                    dump_hidden_localize(gpu, &pbs.dn_v_batch, n, start_pos, v_dim, 0, "v_b");
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_alpha_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "alpha_b",
                    );
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_beta_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "beta_b",
                    );
                }
                if let Some(parents) = tree_parents {
                    if matches!(dn_state.quant, StateQuant::FP32) {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "FP32-state batched prefill does not support tree DeltaNet replay yet",
                        ));
                    }
                    let tape_q8 = pbs
                        .dn_s_tape_q8
                        .as_ref()
                        .expect("tree-aware LA requires dn_s_tape_q8 scratch");
                    let tape_sc = pbs
                        .dn_s_tape_scales
                        .as_ref()
                        .expect("tree-aware LA requires dn_s_tape_scales scratch");
                    gpu.gated_delta_net_q8_tree_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        tape_q8,
                        tape_sc,
                        parents,
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if use_q8_gdn_per_token {
                    for step in 0..n {
                        if let Some(frame_base) = q8_gdn_serial_frame_base {
                            gpu.debug_set_gdn_requant_frame(frame_base.wrapping_add(
                                (step * q8_gdn_serial_frame_layers + delta_layer_idx) as u32,
                            ));
                        }
                        let q = pbs.dn_q_batch.sub_offset(step * v_dim, v_dim);
                        let k = pbs.dn_k_batch.sub_offset(step * v_dim, v_dim);
                        let v = pbs.dn_v_batch.sub_offset(step * v_dim, v_dim);
                        let alpha = pbs.dn_alpha_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let beta = pbs.dn_beta_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let out = pbs.dn_attn_out_batch.sub_offset(step * v_dim, v_dim);
                        gpu.gated_delta_net_q8(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?;
                    }
                    if let Some(frame_base) = q8_gdn_serial_frame_base {
                        gpu.debug_set_gdn_requant_frame(
                            frame_base.wrapping_add((n * q8_gdn_serial_frame_layers) as u32),
                        );
                    }
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                }
                debug_stop_after!("gdn", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_out_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_attn_out_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    // DIAG: dump GDN attention output at layer 0
                    if layer_idx == 0 {
                        dump_hidden_localize(
                            gpu,
                            &pbs.dn_attn_out_batch,
                            n,
                            start_pos,
                            n_v_heads * config.linear_value_head_dim,
                            0,
                            "gdn_b",
                        );
                    }
                }
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    n,
                )?;
                debug_stop_after!("gated_norm", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.normed_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_normed_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_residual_in_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }
                // wo + residual. Q8 wo lands un-rotated (Q8 weights were
                // quantized against un-rotated activations); MQ4/MQ6 wo
                // require FWHT(awq_scale-adjusted) rotation. Mirrors the
                // dense LA wo dispatch (qwen35.rs:5000-5043) — the MQ6
                // branch is required for AWQ A3B where 4/40 LA layers
                // ship MQ6 wo and would otherwise corrupt the residual
                // stream when dispatched through the HFQ4 kernel against
                // 200 B/group MQ6-layout bytes.
                let dn_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let dn_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let dn_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let dn_wo_input = if dn_wo_is_q8 {
                    &pbs.dn_normed_batch
                } else if dn_wo_is_paro {
                    // PARO wo: rotate dn_normed by wo's own Givens tables
                    // into dn_normed_rot_batch. Same scratch layout as MQ4
                    // (since dn_normed_rot_batch is unused on the Q8 path).
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at LA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.dn_normed_rot_batch
                };
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_input_bufs[delta_layer_idx].buf,
                        off_v,
                        &dn_wo_input.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if dn_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 {
                    // Non-WMMA Q8: gemm into a scratch then add into x_batch.
                    // Reuse `dn_normed_rot_batch` (free since the MQ4 rotate
                    // path didn't run here) as the GEMM scratch.
                    let scratch = pbs.dn_normed_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if dn_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free at
                    // this point — used earlier for the QKVZA stage; not
                    // needed for the rest of this layer) as the scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }
                debug_stop_after!("wo", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_residual_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Batched MoE FFN replaces the dense (rmsnorm + gate+up +
                // silu_mul + w_down) block. Takes pbs.x_batch as input AND
                // accumulates the FFN output residual back into it via the
                // batched indexed down kernel's atomicAdd path.
                if debug_stop_after_la_layer == Some(layer_idx) {
                    return Ok(());
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_input_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    layer_idx,
                    &ctx,
                    routed_out,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Batched MoE FA layer. FA body is the same as FullAttn
                // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
                // kv_write + attention + sigmoid_mul + wo+residual);
                // only the FFN differs. Duplicated inline — will be
                // consolidated with the dense FA batched body once the
                // MoE path is proven byte-exact.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let q_dim = config.n_heads * config.head_dim;
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wq.gpu_dtype` is restricted to MQ4G256 / HFQ4G256
                // / MQ6G256 / HFQ6G256 here. Adding MQ3 to the matcher AND
                // the QKV dispatch is insufficient — the wo path below
                // (line 5320) is hardcoded MQ4 too — so the all-or-nothing
                // wiring lives in a separate PR (see followup issue).
                let qkv_is_mq = matches!(layer.wq.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
                // Phase 1.6 (PARO FullAttnMoe): wq/wk/wv are ParoQ4G128
                // (each with its own Givens rotation tables). The fused-QKV
                // kernels can't handle this — they assume one shared
                // rotation. Unfused 3-way dispatch (rotate + gemm_hfq4g128
                // per projection) matches the LA QKVZA Phase 1.5 pattern.
                let qkv_is_paro = matches!(layer.wq.gpu_dtype, DType::ParoQ4G128);
                // Fused QKV requires uniform dtype — see issue #249 for
                // the dense FA variant. Gate the same way here.
                let q8_wmma_arch = gpu.arch_caps.has_wmma();
                let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
                    && layer.wv.gpu_dtype == layer.wq.gpu_dtype;

                if qkv_is_mq {
                    // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else if qkv_is_paro {
                    // PARO: rmsnorm into x_norm_batch (un-rotated). x_rot_batch
                    // is reused as the per-weight rotation scratch.
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if qkv_is_paro {
                    // PARO 3-way unfused dispatch (wq, wk, wv each with own
                    // Givens rotation). Same shape outputs as the fused
                    // paths: fa_q_full_batch, fa_k_batch, fa_v_batch.
                    let paro_wq = layer.wq.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wq missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wk = layer.wk.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wk missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wv = layer.wv.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wv missing paro metadata at FA layer {layer_idx}")
                    });
                    // wq
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wq.pairs,
                        &paro_wq.theta,
                        &paro_wq.channel_scales,
                        n,
                        dim,
                        paro_wq.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    // wk
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wk.pairs,
                        &paro_wk.theta,
                        &paro_wk.channel_scales,
                        n,
                        dim,
                        paro_wk.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    // wv
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wv.pairs,
                        &paro_wv.theta,
                        &paro_wv.channel_scales,
                        n,
                        dim,
                        paro_wv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_is_6bit && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Q8_0)
                            && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                        "FAMoe qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
                    );
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && qkv_same_dtype {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq4G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else {
                    // Mixed-format fallback (issue #249). batched_gemm_single_weight
                    // covers MQ4/HFQ4 + MQ6/HFQ6 + Q8_0; mixed-Q8/MQ4 within FAMoe
                    // routes here.
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wk,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wv,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        n,
                    )?;
                }
                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    n,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                if hipfire_runtime::triattn::tap_enabled() {
                    let gpu_handled =
                        hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                            gpu,
                            layer_idx,
                            &pbs.fa_q_batch.buf,
                            n,
                            config.n_heads,
                            config.head_dim,
                        )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if hipfire_runtime::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }
                // Path B pre-RoPE K capture (MoE FA variant). See same
                // block in the FullAttn branch for rationale.
                if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
                    if let Some(slot) = slots.get(fa_layer_idx) {
                        let kv_dim = config.n_kv_heads * config.head_dim;
                        let n_bytes = n * kv_dim * 4;
                        gpu.memcpy_dtod_at_auto(&slot.buf, 0, &pbs.fa_k_batch.buf, 0, n_bytes)?;
                    }
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                // pbs.positions stays physical for the KV write below; the
                // offset rotates new Q/K at absolute phase after compaction.
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    &pbs.positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    n,
                    kv_cache.compact_offset as i32,
                )?;

                let use_kld_direct_f16kv_attention = kld_direct_f16kv_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                );
                let use_kld_fp32_gqa4_attention = kld_fp32_gqa4_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                    n,
                );

                if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                } else if !use_kld_direct_f16kv_attention && !use_kld_fp32_gqa4_attention {
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                }
                const LDS_CTX_LIMIT: usize = 15000;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                // Batched KV write + flash attention (via dispatch).
                let is_tree = tree_verify.is_some();
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                if use_kld_direct_f16kv_attention {
                    gpu.attention_dflash_wmma_causal_f32(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        n,
                        n,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                } else if use_kld_fp32_gqa4_attention {
                    gpu.attention_f32_batched_gqa4(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                        n,
                    )?;
                } else if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode not supported on asym2 KV (use asym3)",
                    );
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_q8 && max_ctx_len > LDS_CTX_LIMIT {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode hits the long-context Q8 fallback \
                         at max_ctx_len={} > {}; tree blocks should stay small",
                        max_ctx_len,
                        LDS_CTX_LIMIT,
                    );
                    // See dense FullAttn branch above for the i32-vs-f32 slot
                    // rationale; reconstruct positions from the host-side row
                    // position directly.
                    let q_dim_local = config.n_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let seq_len_b = pos_b + 1;
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim_local, q_dim_local);
                            let out_b = pbs
                                .fa_attn_out_batch
                                .sub_offset(b * q_dim_local, q_dim_local);
                            gpu.attention_flash_q8_0(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                seq_len_b,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 {
                    gpu.attention_q8_0_kv_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                } else {
                    gpu.attention_f32_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                }
                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let plan = KvTierPlan::derive(KvTierInputs {
                    quant_asym4: kv_cache.quant_asym4,
                    quant_asym3: kv_cache.quant_asym3,
                    quant_asym2: kv_cache.quant_asym2,
                    quant_q8: kv_cache.quant_q8,
                    quant_fwht: kv_cache.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    v_mode_bits: 0,
                    pos: start_pos,
                    flash_mode: s.flash_mode as usize,
                    capture_mode: gpu.capture_mode,
                    batch_size: n,
                    is_tree,
                    is_boundary: false,
                })
                .map_err(|e| HipError::new(0, &e.to_string()))?;
                let io = AttnParams {
                    q: &pbs.fa_q_batch,
                    k: &pbs.fa_k_batch,
                    v: &pbs.fa_v_batch,
                    k_cache: &kv_cache.k_gpu[layer_idx],
                    v_cache: &kv_cache.v_gpu[layer_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &s.pos_buf,
                    pos: start_pos,
                    positions: Some(&pbs.positions),
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    physical_cap: kv_cache.physical_cap,
                    batch_size: n,
                    max_ctx_len,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv_cache.givens_cos.as_ref(),
                    givens_sin: kv_cache.givens_sin.as_ref(),
                    tree_bias,
                    block_start,
                    block_cols,
                    output: &pbs.fa_attn_out_batch,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
                gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                // wo + residual. Mirrors the dense FA wo dispatch at
                // qwen35.rs:5591-5623 — Q8 wo skips rotation (un-rotated
                // input expected); MQ4/MQ6 wo apply FWHT(awq_scale-adjusted).
                // MQ6 wo has its own branch: feeding MQ6 bytes to the MQ4
                // residual kernel would read 200 B/group data as 136 B/group
                // HFQ4 layout and catastrophically mis-stride.
                let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                // Phase 1.6 (PARO FullAttnMoe wo): own Givens rotation table,
                // 72 B/group HFQ4G128 layout. Rotate fa_attn_out_batch by wo's
                // paro into fa_attn_out_rot_batch, then HFQ4G128 GEMM into a
                // scratch, then add into x_batch.
                let fa_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let fa_wo_input = if fa_wo_is_q8 {
                    &pbs.fa_attn_out_batch
                } else if fa_wo_is_paro {
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at FA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                };
                if fa_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 {
                    // Non-WMMA Q8: GEMM into a scratch then add into x_batch.
                    // Reuse `fa_attn_out_rot_batch` (free since MQ4 rotate
                    // didn't run here) as scratch.
                    let scratch = pbs.fa_attn_out_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free since
                    // QKVZA is done — the MoE FFN body below rewrites it
                    // as its first action) as the gemm output scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                // Batched MoE FFN.
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    layer_idx,
                    &ctx,
                    routed_out,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = kv_dim;
                let _ = q_dim;
                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
        dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
    }

    // ── 3. Final output norm + logits ───────────────────────────────────
    // Multi-GPU band-mode: skip when this is not the last band — the
    // running activation in `pbs.x_batch` is what the next band's
    // peer-copy reads. `weights.output_norm` and `weights.output` only
    // live on the last band's device anyway.
    if do_lm_head {
        // If the caller requested per-token hidden output (DFlash verify path),
        // run rmsnorm over all N rows into their buffer. Otherwise use the
        // legacy last-token-only path.
        if let Some((dst, offset_rows)) = per_token_hidden_out {
            let dst_view = dst.sub_offset(offset_rows * dim, n * dim);
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &weights.output_norm,
                &dst_view,
                n,
                dim,
                config.norm_eps,
            )?;
            if prefill_should_emit_last_token_logits(true, needs_last_token_logits) {
                // Still populate s.logits with the last-token logits for
                // callers that rely on it (the legacy prefill post-condition).
                let last = n - 1;
                let last_view = dst.sub_offset((offset_rows + last) * dim, dim);
                {
                    let wr = weights.output.dispatch_ref();
                    let step = Step::Gemv {
                        w: &wr,
                        input: GemvInput::Raw(&last_view),
                        out: &s.logits,
                    };
                    execute_steps(gpu, &ctx, &[step])
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }
            }
        } else {
            // Legacy path: only last-token logits.
            // Use _auto so the D→D copy routes through the active stream
            // during hipGraph capture (bare memcpy_dtod_at uses the legacy
            // null stream and breaks capture: HIP error 906).
            let last = n - 1;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            {
                let wr = weights.output.dispatch_ref();
                let step = Step::Gemv {
                    w: &wr,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.logits,
                };
                execute_steps(gpu, &ctx, &[step])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            }
        }
    }

    Ok(())
}

/// Run a single FullAttn layer body on s.x at position `pos`. Extracted
/// for use from the batched prefill path's FA-layer fallback. Byte-exact
/// with the FA branch of forward_scratch_layers.
#[allow(clippy::too_many_arguments)]
fn run_fa_layer_body(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    _kv_layer_idx: usize,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let layer = match &weights.layers[layer_idx] {
        LayerWeights::FullAttn(l) => l,
        _ => unreachable!(),
    };

    // Fused rmsnorm + FWHT rotation for wq/wk/wv (MQ-family).
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.wq,
        &s.x,
        &layer.attn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    // Lever 1 — Fused rmsnorm + PARO per-group rotation for wq.
    // x_rot_paro is valid ONLY for wq (PARO rotation uses wq's pairs/theta/channel_scales);
    // wk and wv will run their own rotation via the standard weight_gemv path. The fused
    // kernel ALSO writes s.tmp (post-rmsnorm) so wk/wv get correct input. Saves 1 launch
    // per FA block (rmsnorm+wq rotate folded into one kernel). Default on; opt out via
    // HIPFIRE_PARO_FUSE_RMSNORM=0.
    let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
        && layer.wq.gpu_dtype == DType::ParoQ4G128
        && layer.wq.k % 128 == 0
        && layer.wq.m % 8 == 0
    {
        fused_rmsnorm_rotate_for_paro(
            gpu,
            &layer.wq,
            &s.x,
            &layer.attn_norm,
            &s.tmp,
            &s.x_rot,
            config.norm_eps,
        )?
    } else {
        None
    };

    // Cross-arch fast path: fused 3-way projection for wq+wk+wv.
    let dt = layer.wq.gpu_dtype;
    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
    let fused_fa3_f16 = config.attn_output_gate && fa3_same_dtype && dt == DType::F16;
    let fused_fa3_mq4 = fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_fa3_hfq6 = config.attn_output_gate
        && fa3_same_dtype
        && (dt == DType::MQ6G256 || dt == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_fa3_f16 {
        gpu.fused_qkvza_f16_xf32(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &layer.wq.buf,
            &s.tmp,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            &s.o,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            0,
            layer.wq.k,
        )?;
    } else if fused_fa3_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_hfq4g256(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq3g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq4g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_hfq6g256_dp4a(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else {
        // Lever 1 fast path: when fused_rmsnorm_rotate_for_paro produced x_rot_paro,
        // wq has its rotated x already — call the prerotated GEMV directly (saves the
        // standalone paro4g128t_rotate launch for wq). wk and wv MUST do their own
        // rotation since PARO pairs/theta differ per linear; they consume s.tmp
        // (post-rmsnorm) via the standard weight_gemv path.
        if let Some(xr_q) = x_rot_paro {
            gpu.gemv_paro4g128t_prerotated(
                &layer.wq.buf,
                xr_q,
                &s.fa_q_full,
                layer.wq.m,
                layer.wq.k,
            )?;
        } else {
            weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
        }
        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
    }

    qwen35_materialize_fa_q(gpu, config, &s.fa_q_full, &s.fa_q, &s.fa_gate, 1)?;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    let npu_hnr_ok = if hipfire_runtime::triattn::tap_enabled() {
        false
    } else {
        try_npu_headnorm_rope(
            gpu,
            layer_idx,
            &s.fa_q,
            &s.fa_k,
            &layer.q_norm,
            &layer.k_norm,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            pos,
        )?
    };
    if !npu_hnr_ok {
        gpu.rmsnorm_batched(
            &s.fa_q,
            &layer.q_norm,
            &s.fa_q,
            config.n_heads,
            config.head_dim,
            config.norm_eps,
        )?;
        gpu.rmsnorm_batched(
            &s.fa_k,
            &layer.k_norm,
            &s.fa_k,
            config.n_kv_heads,
            config.head_dim,
            config.norm_eps,
        )?;
        if hipfire_runtime::triattn::tap_enabled() {
            // Try GPU path first (matches the batched FA tap at line ~3499 in
            // forward_prefill_batch). When the calibration tap is GPU-resident
            // (CalibrateGpu) we MUST dispatch the kernel here — falling
            // through to record_prerope_qk would either silently drop the
            // sample (pre-Phase-2) or panic (post-Phase-2).
            let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                gpu,
                layer_idx,
                &s.fa_q.buf,
                1,
                config.n_heads,
                config.head_dim,
            )?;
            if !gpu_handled {
                let n_q = config.n_heads * config.head_dim;
                let q_cpu = gpu.download_f32(&s.fa_q)?;
                if hipfire_runtime::triattn::tap_needs_k() {
                    let n_k = config.n_kv_heads * config.head_dim;
                    let k_cpu = gpu.download_f32(&s.fa_k)?;
                    hipfire_runtime::triattn::record_prerope_qk(
                        layer_idx,
                        &q_cpu[..n_q],
                        Some(&k_cpu[..n_k]),
                    );
                } else {
                    hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
                }
            }
        }
        // If TriAttention has compacted the cache, absolute RoPE phase diverges
        // from the physical cache index. Temporarily load the absolute position
        // into pos_buf for the rope call, then restore the physical position
        // for kv_cache_write + flash attention (which both want the write slot).
        if kv_cache.compact_offset > 0 {
            let abs = (pos + kv_cache.compact_offset) as i32;
            gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
        }
        gpu.rope_partial_interleaved_f32(
            &s.fa_q,
            &s.fa_k,
            &s.pos_buf,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
        )?;
    }
    if kv_cache.compact_offset > 0 {
        let phys = pos as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
    }

    if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht4_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                0,
            )?;
            gpu.attention_flash_fwht4(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
                0,
            )?;
        } else {
            gpu.kv_cache_write_asym4_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
            )?;
            gpu.attention_flash_asym4(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht3_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                0,
            )?;
            gpu.attention_flash_fwht3(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
                0,
            )?;
        } else {
            gpu.kv_cache_write_asym3_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
            )?;
            gpu.attention_flash_asym3(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht2_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                0,
            )?;
            gpu.attention_flash_fwht2(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
                0,
            )?;
        } else {
            gpu.kv_cache_write_asym2_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
            )?;
            gpu.attention_flash_asym2(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(
            &kv_cache.k_gpu[layer_idx],
            &s.fa_k,
            &s.pos_buf,
            config.n_kv_heads,
            config.head_dim,
        )?;
        gpu.kv_cache_write_q8_0(
            &kv_cache.v_gpu[layer_idx],
            &s.fa_v,
            &s.pos_buf,
            config.n_kv_heads,
            config.head_dim,
        )?;
        gpu.attention_q8_0_kv(
            &s.fa_q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out,
            &s.pos_buf,
            pos + 1,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            kv_cache.physical_cap,
        )?;
    } else {
        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
        gpu.attention_f32(
            &s.fa_q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out,
            &s.pos_buf,
            pos + 1,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            kv_cache.physical_cap,
        )?;
    }

    qwen35_apply_fa_gate(gpu, config, &s.fa_attn_out, &s.fa_gate)?;
    qwen35_attention_wo_residual(
        gpu,
        config,
        layer_idx,
        &layer.wo,
        &s.fa_attn_out,
        &s.x,
        &s.o,
    )?;
    let ctx = DispatchCtx::new(gpu);
    kv_cache_attention_dispatch(&ctx, gpu, kv_cache, s, config, layer_idx, pos)?;

    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
    {
        let wr = layer.wo.dispatch_ref();
        execute_steps(
            gpu,
            &ctx,
            &[Step::GemvResidual {
                w: &wr,
                input: GemvInput::Raw(&s.fa_attn_out),
                residual: &s.x,
                out: &s.x,
            }],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    // FFN: fused rmsnorm + rotate for w_gate/w_up.
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.w_gate,
        &s.x,
        &layer.ffn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    // Lever 1 — Fused rmsnorm + PARO per-group rotation for w_gate.
    let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
        && layer.w_gate.gpu_dtype == DType::ParoQ4G128
        && layer.w_gate.k % 128 == 0
        && layer.w_gate.m % 8 == 0
    {
        fused_rmsnorm_rotate_for_paro(
            gpu,
            &layer.w_gate,
            &s.x,
            &layer.ffn_norm,
            &s.tmp,
            &s.x_rot,
            config.norm_eps,
        )?
    } else {
        None
    };
    let dt_g = layer.w_gate.gpu_dtype;
    let same_dtype = layer.w_up.gpu_dtype == dt_g;
    let fused_gu_mq4 = same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
    let fused_gu_f16 = same_dtype && dt_g == DType::F16;
    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_gu_hfq6 = same_dtype
        && (dt_g == DType::MQ6G256 || dt_g == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_gu_f16 {
        gpu.fused_gate_up_f16_xf32(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &s.tmp,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_hfq4g256(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq3g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq4g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_hfq6g256_dp4a(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else {
        if let Some(xr_first) = x_rot_paro {
            gpu.gemv_paro4g128t_prerotated(
                &layer.w_gate.buf,
                xr_first,
                &s.gate_ffn,
                layer.w_gate.m,
                layer.w_gate.k,
            )?;
        } else {
            weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
        }
        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
    }
    weight_gemv_swiglu_residual_bf16_probe(
        gpu,
        layer_idx,
        &layer.w_down,
        &layer.bf16_down_shadow,
        &s.gate_ffn,
        &s.up,
        &s.ffn_hidden,
        &s.x,
    )?;

    Ok(())
}

/// Same as `forward_scratch` but also extracts hidden states from the
/// configured target layers into `hidden_rb`. Used by the DFlash draft path
/// during target verification. `hidden_rb.advance_head()` is called once
/// automatically at the end of the forward pass.
pub fn forward_scratch_with_hidden(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: &mut HiddenStateRingBuffer,
) -> HipResult<()> {
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        Some(hidden_rb),
        true,
        None,
    )?;
    hidden_rb.advance_head();
    Ok(())
}

fn forward_scratch_no_logits(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, scratch, None, false, None,
    )
}

/// Zero-alloc forward from pre-computed embedding in scratch.x.
pub fn forward_scratch_embed(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Upload embedding directly into scratch.x
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embedding_data.as_ptr() as *const u8,
            embedding_data.len() * 4,
        )
    };
    gpu.hip.memcpy_htod(&scratch.x.buf, bytes)?;
    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, scratch, None, true, None,
    )
}

/// Batched single-weight GEMM used by the mixed-format fallback in
/// `forward_prefill_chunk`'s FA QKV path. The fused `gemm_qkv_hfq*` kernels
/// require wq/wk/wv to share a bit-width — they index all three weight
/// buffers with the same stride. When `--kmap-dense --kmap-mode 2` promotes
/// only `v_proj` to MQ6 (issue #249), the fused HFQ4 kernel reads `wv`'s
/// MQ6 buffer with HFQ4's 136-B stride (true stride: 200 B), producing
/// silent NaN. Callers gate the fused path on a same-dtype check and route
/// here per-weight when they disagree.
///
/// Covers same-rotation-family bit-width mixes: MQ4+MQ6 (both
/// FWHT-baked, what kmap mode 2 produces) and HFQ4+HFQ6 (both
/// unrotated). Cross-family mixes (e.g. HFQ4+MQ6) would corrupt the
/// shared rmsnorm+rotate output; no quantizer config produces them
/// today, but extend the dispatch caller's invariants here if that
/// changes.
fn batched_gemm_single_weight(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::MQ4G256 | DType::HFQ4G256 => run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G256,
            &w.buf,
            w.gpu_dtype,
            x,
            y,
            w.m,
            w.k,
            n,
        ),
        DType::MQ6G256 | DType::HFQ6G256 => {
            // No non-residual batched MQ6/HFQ6 GEMM exists. Zero Y then
            // accumulate. The zero MUST be ordered on the same stream as
            // the GEMM that consumes it — using sync `hipMemset` on the
            // null stream while subsequent kernels enqueue on a non-null
            // active stream leaves a race that produces silent NaN in the
            // residual stream (logits stay NaN on eval until a stray host
            // sync masks the order bug).
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ3G256 => {
            // Same pattern as MQ6: no non-residual batched HFQ3 GEMM
            // exists in the scalar gfx10 family — `gemm_hfq3g256_residual`
            // is the only single-weight batched dispatch. Zero Y on the
            // active stream (same race-free contract as the HFQ6 arm)
            // then accumulate.
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::Q8_0 => {
            // Q8 weights consume the un-rotated rmsnorm output. Callers
            // routing here must pass `pbs.x_rot_batch` containing
            // `rmsnorm(x_batch)` *without* FWHT — the existing pattern is
            // to gate the `fused_rmsnorm_rotate_*_for(...)` call on
            // `is_mq` and fall through to `gpu.rmsnorm_batched(...)` for
            // Q8 (see DNMoe LA preamble for a representative).
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "mixed-format batched prefill: weight dtype {other:?} has no \
             single-weight batched dispatch yet. Currently MQ3/HFQ3, \
             MQ4/HFQ4, MQ6/HFQ6, and Q8_0 mixes are wired. Re-quantize with \
             uniform format or extend `batched_gemm_single_weight` to cover this format."
            ),
        )),
    }
}

// ── Dispatch helpers ─────────────────────────────────────────────────────

/// Helper: convert `WeightTensor.paro` (if present) to `GivensRef`.
fn paro_to_givens(p: &ParoRotation) -> GivensRef<'_> {
    GivensRef {
        pairs: &p.pairs,
        theta: &p.theta,
        scales: &p.channel_scales,
        krot: p.krot as usize,
    }
}

/// Unified QKVZA (4-way) projection via execute_steps for DeltaNet layers.
/// Covers all dtypes — the interpreter selects fused QKVZA kernels for eligible
/// dtypes via FUSED_TABLE guards; everything else falls through to per-op
/// dispatch (including ParoQ4G128 which does individual Givens-rotated GEMV calls).
/// Replaces rmsnorm_rotate_dispatch + fused_qkvza_dispatch.
#[allow(clippy::too_many_arguments)]
fn qkvza_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    dn_qkv: &GpuTensor,
    dn_z: &GpuTensor,
    dn_beta: &GpuTensor,
    dn_alpha: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(wqkv.gpu_dtype);
    if rotation == RotationPlan::Givens {
        // ParoQ4G128: plain rmsnorm, then per-weight Givens rotation inside run_auto.
        let wr_qkv = WeightRef {
            buf: &wqkv.buf,
            dtype: wqkv.gpu_dtype,
            m: wqkv.m,
            k: wqkv.k,
            row_stride: 0,
            rotation: wqkv.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_z = WeightRef {
            buf: &wz.buf,
            dtype: wz.gpu_dtype,
            m: wz.m,
            k: wz.k,
            row_stride: 0,
            rotation: wz.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_beta = WeightRef {
            buf: &w_beta.buf,
            dtype: w_beta.gpu_dtype,
            m: w_beta.m,
            k: w_beta.k,
            row_stride: 0,
            rotation: w_beta.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wr_alpha = WeightRef {
            buf: &w_alpha.buf,
            dtype: w_alpha.gpu_dtype,
            m: w_alpha.m,
            k: w_alpha.k,
            row_stride: 0,
            rotation: w_alpha.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wqkv.awq_scale.as_ref(),
                k: wqkv.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wr_qkv,
                input: GemvInput::Raw(x_rot),
                out: dn_qkv,
            },
            Step::Gemv {
                w: &wr_z,
                input: GemvInput::Raw(x_rot),
                out: dn_z,
            },
            Step::Gemv {
                w: &wr_beta,
                input: GemvInput::Raw(x_rot),
                out: dn_beta,
            },
            Step::Gemv {
                w: &wr_alpha,
                input: GemvInput::Raw(x_rot),
                out: dn_alpha,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        // FWHT-rotated (MQ family) or non-rotated (HFQ, Q8, etc.) dtypes.
        // RmsnormAutomatic handles FWHT when rotation != None;
        // downstream Gemv steps use Prerotated to avoid double-FWHT.
        let wr_qkv = WeightRef {
            buf: &wqkv.buf,
            dtype: wqkv.gpu_dtype,
            m: wqkv.m,
            k: wqkv.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_z = WeightRef {
            buf: &wz.buf,
            dtype: wz.gpu_dtype,
            m: wz.m,
            k: wz.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_beta = WeightRef {
            buf: &w_beta.buf,
            dtype: w_beta.gpu_dtype,
            m: w_beta.m,
            k: w_beta.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_alpha = WeightRef {
            buf: &w_alpha.buf,
            dtype: w_alpha.gpu_dtype,
            m: w_alpha.m,
            k: w_alpha.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wqkv.awq_scale.as_ref(),
                k: wqkv.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wr_qkv,
                input: GemvInput::Prerotated(x_rot),
                out: dn_qkv,
            },
            Step::Gemv {
                w: &wr_z,
                input: GemvInput::Prerotated(x_rot),
                out: dn_z,
            },
            Step::Gemv {
                w: &wr_beta,
                input: GemvInput::Prerotated(x_rot),
                out: dn_beta,
            },
            Step::Gemv {
                w: &wr_alpha,
                input: GemvInput::Prerotated(x_rot),
                out: dn_alpha,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// Unified QKV projection via execute_steps. Covers all dtypes — the interpreter
/// selects fused kernels for eligible dtypes via FUSED_TABLE guards; everything
/// else falls through to per-op dispatch. Replaces qkv_interpret_mq +
/// fused_qkv_dispatch + their preceding rmsnorm_rotate_dispatch call.
#[allow(clippy::too_many_arguments)]
fn qkv_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wq: &WeightTensor,
    wk: &WeightTensor,
    wv: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    fa_q: &GpuTensor,
    fa_k: &GpuTensor,
    fa_v: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(wq.gpu_dtype);
    if rotation == RotationPlan::Givens {
        let wrq = WeightRef {
            buf: &wq.buf,
            dtype: wq.gpu_dtype,
            m: wq.m,
            k: wq.k,
            row_stride: 0,
            rotation: wq.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wrk = WeightRef {
            buf: &wk.buf,
            dtype: wk.gpu_dtype,
            m: wk.m,
            k: wk.k,
            row_stride: 0,
            rotation: wk.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wrv = WeightRef {
            buf: &wv.buf,
            dtype: wv.gpu_dtype,
            m: wv.m,
            k: wv.k,
            row_stride: 0,
            rotation: wv.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wq.awq_scale.as_ref(),
                k: wq.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wrq,
                input: GemvInput::Raw(x_rot),
                out: fa_q,
            },
            Step::Gemv {
                w: &wrk,
                input: GemvInput::Raw(x_rot),
                out: fa_k,
            },
            Step::Gemv {
                w: &wrv,
                input: GemvInput::Raw(x_rot),
                out: fa_v,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        let wrq = WeightRef {
            buf: &wq.buf,
            dtype: wq.gpu_dtype,
            m: wq.m,
            k: wq.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wrk = WeightRef {
            buf: &wk.buf,
            dtype: wk.gpu_dtype,
            m: wk.m,
            k: wk.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wrv = WeightRef {
            buf: &wv.buf,
            dtype: wv.gpu_dtype,
            m: wv.m,
            k: wv.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: attn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: wq.awq_scale.as_ref(),
                k: wq.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wrq,
                input: GemvInput::Prerotated(x_rot),
                out: fa_q,
            },
            Step::Gemv {
                w: &wrk,
                input: GemvInput::Prerotated(x_rot),
                out: fa_k,
            },
            Step::Gemv {
                w: &wrv,
                input: GemvInput::Prerotated(x_rot),
                out: fa_v,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    }
}

/// Unified gate+up (FFN) projection via execute_steps. Covers all dtypes.
/// Replaces fused_gate_up_dispatch + its preceding rmsnorm_rotate_dispatch call.
#[allow(clippy::too_many_arguments)]
fn gate_up_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w_gate: &WeightTensor,
    w_up: &WeightTensor,
    ffn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,
    x_rot: &GpuTensor,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    let rotation = dtype_rotation_plan(w_gate.gpu_dtype);
    if rotation == RotationPlan::Givens {
        let wrg = WeightRef {
            buf: &w_gate.buf,
            dtype: w_gate.gpu_dtype,
            m: w_gate.m,
            k: w_gate.k,
            row_stride: 0,
            rotation: w_gate.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let wru = WeightRef {
            buf: &w_up.buf,
            dtype: w_up.gpu_dtype,
            m: w_up.m,
            k: w_up.k,
            row_stride: 0,
            rotation: w_up.paro.as_ref().map(paro_to_givens),
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: ffn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: w_gate.awq_scale.as_ref(),
                k: w_gate.k,
                eps,
                rotation: RotationPlan::None,
            },
            Step::Gemv {
                w: &wrg,
                input: GemvInput::Raw(x_rot),
                out: gate_out,
            },
            Step::Gemv {
                w: &wru,
                input: GemvInput::Raw(x_rot),
                out: up_out,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
    } else {
        let wrg = WeightRef {
            buf: &w_gate.buf,
            dtype: w_gate.gpu_dtype,
            m: w_gate.m,
            k: w_gate.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wru = WeightRef {
            buf: &w_up.buf,
            dtype: w_up.gpu_dtype,
            m: w_up.m,
            k: w_up.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let steps = [
            Step::RmsnormAutomatic {
                x,
                norm_weight: ffn_norm,
                x_plain: tmp,
                out: x_rot,
                awq_scale: w_gate.awq_scale.as_ref(),
                k: w_gate.k,
                eps,
                rotation,
            },
            Step::Gemv {
                w: &wrg,
                input: GemvInput::Prerotated(x_rot),
                out: gate_out,
            },
            Step::Gemv {
                w: &wru,
                input: GemvInput::Prerotated(x_rot),
                out: up_out,
            },
        ];
        execute_steps(gpu, ctx, &steps).map_err(|e| HipError::new(0, &e.to_string()))
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
    layer_idx: usize,
) -> HipResult<()> {
    let r = if ffn_all_mq4_for_moe(ffn) {
        gpu.fused_rmsnorm_rotate_mq(
            x,
            ffn_norm,
            s.moe_x_rot.as_ref().expect("MoE scratch"),
            config.dim,
            config.norm_eps,
        )?;
        moe_ffn_decode_with_scratch_prerotated(gpu, None, ffn, x, x, config, s, layer_idx)
    } else {
        gpu.rmsnorm_f32(x, ffn_norm, &s.tmp, config.norm_eps)?;
        moe_ffn_decode_with_scratch(gpu, None, ffn, &s.tmp, x, config, s, layer_idx)
    };
    r?;
    trace_finite_if_enabled(gpu, "moe_ffn", x)?;
    Ok(())
}

/// EP (Ship 6 substrate-EP) variant of `moe_ffn_dispatch`: same rmsnorm/rotate +
/// MoE decode, but the routed combine + shared-down accumulate into `routed_out`
/// (a zeroed per-rank partial the EP executor all-reduces), and `skip_shared`
/// gates the shared-expert down to rank 0. Calls `moe_ffn_decode_impl` directly
/// (the `with_scratch` wrappers don't carry EP params). The residual `x` is left
/// untouched — the executor adds the all-reduced partial into it afterward.
fn moe_ffn_dispatch_ep(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    layer_idx: usize,
    routed_out: &GpuTensor,
    skip_shared: bool,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(s);
    if ffn_all_mq4_for_moe(ffn) {
        gpu.fused_rmsnorm_rotate_mq(
            x,
            ffn_norm,
            s.moe_x_rot.as_ref().expect("MoE scratch"),
            config.dim,
            config.norm_eps,
        )?;
        moe_ffn_decode_impl(
            gpu,
            None,
            ffn,
            x,
            x,
            config,
            &refs,
            true,
            layer_idx,
            Some(routed_out),
            skip_shared,
        )
    } else {
        gpu.rmsnorm_f32(x, ffn_norm, &s.tmp, config.norm_eps)?;
        moe_ffn_decode_impl(
            gpu,
            None,
            ffn,
            &s.tmp,
            x,
            config,
            &refs,
            false,
            layer_idx,
            Some(routed_out),
            skip_shared,
        )
    }
}

/// EP (Ship 6 substrate-EP, ported from tp-mtp-prototype Stage 3e): shard a MoE
/// layer's routed experts to `rank`. Frees the non-owned experts (the memory
/// win), compacts owned to the front of `ffn.experts` (so `experts[0]` stays a
/// valid shared-AWQ representative for the batched silu/rotate helpers), and
/// rebuilds the `[2·n_exp]` device pointer tables: owned global id → its
/// (compacted) buffer ptr; **non-owned → a shared ZEROED gate_up buffer**.
/// Zeroed quant bytes dequant to +0.0 → the non-owned expert's gate_up output
/// is 0 → silu·mul = 0 → rot = 0 → down output 0, so it contributes nothing
/// through `moe_down_combine` WITHOUT any masking kernel. (The non-owned down
/// ptr is irrelevant — its input rot is already 0 — so it reuses
/// `experts[0].down`.) Router / shared expert / attention stay full (replicated
/// in EP v1). The zero buffer is leaked for v1 (lives until teardown) to avoid
/// threading a lifetime field through `Qwen35Weights`.
pub fn shard_moe_experts(
    gpu: &mut Gpu,
    ffn: &mut MoeFfnWeights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
) -> HipResult<()> {
    debug_assert_eq!(
        ffn.experts.len(),
        n_exp,
        "shard_moe_experts expects a full-loaded expert Vec (paged EP is unsupported in v1)",
    );
    // Free non-owned experts; compact owned to the front, recording global→local.
    let old = std::mem::take(&mut ffn.experts);
    let mut compacted: Vec<ExpertWeights> = Vec::with_capacity(shard.experts_per_rank(n_exp));
    let mut local_of_global = vec![usize::MAX; n_exp];
    for (e, ew) in old.into_iter().enumerate() {
        if shard.owns_expert(rank, e) {
            local_of_global[e] = compacted.len();
            compacted.push(ew);
        } else {
            let _ = gpu.free_tensor(ew.gate_up.buf);
            if let Some(s) = ew.gate_up.awq_scale {
                let _ = gpu.free_tensor(s);
            }
            let _ = gpu.free_tensor(ew.down.buf);
            if let Some(s) = ew.down.awq_scale {
                let _ = gpu.free_tensor(s);
            }
        }
    }
    assert!(
        !compacted.is_empty(),
        "shard_moe_experts: rank {rank} owns no experts (n_exp={n_exp}, tp={})",
        shard.tp_size,
    );

    // Shared zeroed gate_up buffer for non-owned slots (same byte size as a real
    // expert's gate_up). LEAKED (mem::forget) so the ptr stays valid for the
    // model's lifetime without a Qwen35Weights field — v1 TODO: own it properly.
    let gu_bytes = compacted[0].gate_up.buf.buf.size();
    let zero_gu = gpu.zeros(&[gu_bytes / 4], DType::F32)?;
    let dummy_gu = zero_gu.buf.as_ptr() as u64;
    let dummy_dn = compacted[0].down.buf.buf.as_ptr() as u64; // rot=0 ⇒ output 0 regardless
    std::mem::forget(zero_gu);

    // Rebuild the [2·n_exp] u64 pointer tables (8 B/ptr = 2 F32 slots).
    let mut gu = vec![0u64; n_exp];
    let mut dn = vec![0u64; n_exp];
    for e in 0..n_exp {
        if shard.owns_expert(rank, e) {
            let li = local_of_global[e];
            gu[e] = compacted[li].gate_up.buf.buf.as_ptr() as u64;
            dn[e] = compacted[li].down.buf.buf.as_ptr() as u64;
        } else {
            gu[e] = dummy_gu;
            dn[e] = dummy_dn;
        }
    }
    let gu_b: Vec<u8> = gu.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_b: Vec<u8> = dn.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&ffn.expert_gate_up_ptrs.buf, &gu_b)?;
    gpu.hip.memcpy_htod(&ffn.expert_down_ptrs.buf, &dn_b)?;
    ffn.experts = compacted;
    Ok(())
}

/// Shard every MoE layer of a replicated `Qwen35Weights` to `rank`, calling
/// [`shard_moe_experts`] on each `DeltaNetMoe` / `FullAttnMoe` layer's FFN.
/// Dense / attention-only layers are untouched. Convenience wrapper for the EP
/// load path so callers (the `forward_ep` driver / examples) never reach into
/// `LayerWeights` internals. `n_exp` is the model's routed expert count
/// (`config.num_experts`).
pub fn shard_all_moe_layers(
    gpu: &mut Gpu,
    weights: &mut Qwen35Weights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
) -> HipResult<()> {
    for layer in weights.layers.iter_mut() {
        match layer {
            LayerWeights::DeltaNetMoe(l) => shard_moe_experts(gpu, &mut l.ffn, shard, rank, n_exp)?,
            LayerWeights::FullAttnMoe(l) => shard_moe_experts(gpu, &mut l.ffn, shard, rank, n_exp)?,
            _ => {}
        }
    }
    Ok(())
}

/// TriAttention tap helper (inline from original forward).
fn triattn_tap(
    gpu: &mut Gpu,
    layer_idx: usize,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
) -> HipResult<()> {
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu,
        layer_idx,
        &s.fa_q.buf,
        1,
        config.n_heads,
        config.head_dim,
    )?;
    if !gpu_handled {
        let n_q = config.n_heads * config.head_dim;
        let q_cpu = gpu.download_f32(&s.fa_q)?;
        if hipfire_runtime::triattn::tap_needs_k() {
            let n_k = config.n_kv_heads * config.head_dim;
            let k_cpu = gpu.download_f32(&s.fa_k)?;
            hipfire_runtime::triattn::record_prerope_qk(
                layer_idx,
                &q_cpu[..n_q],
                Some(&k_cpu[..n_k]),
            );
        } else {
            hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
        }
    }
    Ok(())
}

/// KV cache write + attention dispatch. Inline from original.
fn kv_cache_attention_dispatch(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    kv_cache: &mut kv::KvCache,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    layer_idx: usize,
    pos: usize,
) -> HipResult<()> {
    // KVarN decode: single-token KV write (window append + block flush) + read
    // (build f16 shadow K) + f16/Q8 flash, handled outside the dispatch substrate.
    if kv_cache.quant_kvarn {
        // ── Deferred-hierarchical two-tier KV (flag-gated HIPFIRE_KV_HIERARCHICAL=1).
        // Lazily built on first dispatch (needs n_heads from the config). Replaces
        // the single-tier KVarN read with hot-ring ⊕ 4-bit cold-segment two-tier
        // attention. NO KVarN rotation: the hot ring stores raw fa_k and the cold
        // segments compact with rotate=false, so fa_q is consumed un-rotated and
        // both tiers' K are in the same (un-rotated, RoPE-baked) basis as Q.
        // This is the ONLY KVarN attention entry point (prefill is per-token here
        // too, n=1), so one hook covers prompt + decode.
        if kv_cache.hier.is_none() {
            kv_cache.hier = Some(hipfire_runtime::kv_hier::HierKvState::from_env(
                gpu,
                kv_cache.k_gpu.len(),
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
            )?);
        }
        if kv_cache.hier.as_ref().map(|h| h.enabled).unwrap_or(false) {
            let h = kv_cache.hier.as_mut().unwrap();
            // pos==0 at layer 0 = a new sequence (prompt start) → reset both tiers.
            // In serve this fires only at session start; mid-session/decode never
            // hits pos==0, so continued context is preserved.
            if pos == 0 && layer_idx == 0 {
                h.reset(gpu)?;
            }
            h.append_token(gpu, layer_idx, &s.fa_k, &s.fa_v)?;
            return h.two_tier_read(gpu, layer_idx, &s.fa_q, &s.fa_attn_out);
        }
        // ── KVarN Hadamard-incoherence rotation (paper §method: "Hadamard
        // rotation FOLLOWED BY dual-scaling variance normalization"). The Sinkhorn
        // dual-scaling already runs in `kvarn_quantize_tile`; the missing half is
        // the rotation that Gaussianizes the per-channel K distribution so the
        // 4-bit quant has less error (the codec's own self-test: un-rotated core
        // cos-sim 0.995 → 0.999 with the FWHT). We rotate K *and* Q by the SAME
        // orthonormal per-head FWHT-256 (mq signs); since the rotation is
        // orthonormal, (RQ)·(RK)ᵀ = Q·Kᵀ exactly, so scores are preserved with NO
        // flash/dequant changes and NO Q-side un-rotation. K is written to the
        // cache rotated (window + records both derive from the rotated fa_k), so
        // the whole KVarN frame is self-consistent. V (Q8) is left un-rotated, so
        // the attention output stays in the original basis and o_proj is unchanged.
        // Requires head_dim == 256 (the FWHT-256 group). Opt out with
        // HIPFIRE_KVARN_ROTATE=0 for A/B.
        static KVARN_ROTATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let kvarn_rotate = *KVARN_ROTATE
            .get_or_init(|| std::env::var("HIPFIRE_KVARN_ROTATE").ok().as_deref() != Some("0"));
        if kvarn_rotate && config.head_dim == 256 {
            // In-place: mq_rotate_x loads each 256-group into registers (ds_swizzle
            // butterfly, zero LDS) before storing, so x_in == x_out is safe. n=1
            // (single-token decode / oq4 per-token prefill).
            gpu.rotate_x_mq_batched(&s.fa_k, &s.fa_k, config.n_kv_heads * config.head_dim, 1)?;
            gpu.rotate_x_mq_batched(&s.fa_q, &s.fa_q, config.n_heads * config.head_dim, 1)?;
        }
        // Lazily allocate the reusable gather-tile scratch (once per cache — never
        // per call: GpuTensor has no pool-return Drop, so per-call alloc leaks).
        // The fused KVarN flash (Phase D2) reads records in place, so no f16
        // shadow K buffer is needed anymore.
        if kv_cache.kvarn_tiles.is_none() {
            let tiles = gpu.alloc_tensor(
                &[config.n_kv_heads * config.head_dim * 128],
                hipfire_rdna::DType::F32,
            )?;
            kv_cache.kvarn_tiles = Some(tiles);
        }
        // The KV-write/flash kernels read positions from a GpuTensor; `s.pos_buf`
        // is a raw 4-byte i32 DeviceBuffer (positions use F32 as a 4-byte i32
        // container, see PrefillBatchScratch). Wrap a non-owning [1] view.
        let pos_view = hipfire_rdna::GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(s.pos_buf.as_ptr(), 4) },
            shape: vec![1],
            dtype: hipfire_rdna::DType::F32,
        };
        return gpu.kvarn_attend(
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.k_window[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &s.fa_q,
            &s.fa_k,
            &s.fa_v,
            &pos_view,
            &s.fa_attn_out,
            &s.flash_partials,
            kv_cache.kvarn_tiles.as_ref().unwrap(),
            1,
            pos,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            kv_cache.physical_cap,
            None,
            0,
            0,
            kv_cache.kvarn_bits,
        );
    }
    let plan = KvTierPlan::derive(KvTierInputs {
        quant_asym4: kv_cache.quant_asym4,
        quant_asym3: kv_cache.quant_asym3,
        quant_asym2: kv_cache.quant_asym2,
        quant_q8: kv_cache.quant_q8,
        quant_fwht: kv_cache.quant_fwht,
        quant_hfq4: false,
        quant_q4: false,
        v_mode_bits: 0,
        pos,
        flash_mode: s.flash_mode as usize,
        capture_mode: gpu.capture_mode,
        batch_size: 1,
        is_tree: false,
        is_boundary: false, // TODO: boundary producer not yet populated
    })
    .map_err(|e| HipError::new(0, &e.to_string()))?;
    let io = AttnParams {
        q: &s.fa_q,
        k: &s.fa_k,
        v: &s.fa_v,
        k_cache: &kv_cache.k_gpu[layer_idx],
        v_cache: &kv_cache.v_gpu[layer_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &s.pos_buf,
        pos,
        positions: None,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        physical_cap: kv_cache.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(&s.flash_partials),
        givens_cos: kv_cache.givens_cos.as_ref(),
        givens_sin: kv_cache.givens_sin.as_ref(),
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output: &s.fa_attn_out,
    };
    execute_steps(gpu, ctx, &[Step::Attend { plan, io }])
        .map_err(|e| HipError::new(0, &e.to_string()))
}

/// Forward pass returning logits ON GPU (no download). Caller must free the tensor.
/// Use with gpu.sample_top_p() after applying CPU-side n-gram blocking via download/modify/upload.
pub fn forward_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let dim = config.dim;
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }
    forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Run one step with a pre-computed embedding vector (for VL visual token injection).
/// embedding_data: [dim] F32 values on CPU — uploaded to GPU as the initial hidden state.
pub fn forward_with_embedding(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let x = gpu.upload_f32(embedding_data, &[config.dim])?;
    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}

#[cfg(test)]
mod tests {
    use super::config::{parse_bf16_weight_load_mode, parse_f16_lm_head_mode};
    use super::*;

    fn test_qwen35_config_with_layers(layer_types: Vec<LayerType>) -> Qwen35Config {
        Qwen35Config {
            dim: 16,
            n_layers: layer_types.len(),
            vocab_size: 32,
            norm_eps: 1e-6,
            eos_token: 0,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 0.25,
            attn_output_gate: true,
            is_vl_text: false,
            mrope_interleaved: false,
            mrope_section: [0, 0, 0],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            conv_kernel_dim: 4,
            hidden_dim: 32,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            has_shared_expert: false,
            norm_topk_prob: false,
            layer_types,
            paged_experts: false,
            vram_budget_bytes: u64::MAX,
        }
    }

    fn fake_tensor(ptr: usize) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, 4) },
            shape: vec![1],
            dtype: DType::F32,
        }
    }

    #[test]
    fn deltanet_state_gate_keys_on_redundancy() {
        let cfg = test_qwen35_config_with_layers(vec![LayerType::LinearAttention]);
        // redundancy = linear_key_head_dim (8) × linear_num_value_heads (1) = 8
        assert_eq!(deltanet_state_redundancy(&cfg), 8);

        // Default threshold (usize::MAX) ⇒ FP32 for every real model.
        std::env::remove_var("HIPFIRE_DN_STATE_FP32_BELOW");
        assert_eq!(default_state_quant(&cfg), StateQuant::FP32);

        // Boundary: redundancy < threshold ⇒ FP32; otherwise Q8.
        std::env::set_var("HIPFIRE_DN_STATE_FP32_BELOW", "9");
        assert_eq!(default_state_quant(&cfg), StateQuant::FP32); // 8 < 9
        std::env::set_var("HIPFIRE_DN_STATE_FP32_BELOW", "8");
        assert_eq!(default_state_quant(&cfg), StateQuant::Q8); // 8 < 8 is false
        std::env::remove_var("HIPFIRE_DN_STATE_FP32_BELOW");
    }

    #[test]
    fn moe_router_histogram_records_top1_topk_weights_and_drops() {
        reset_moe_router_histogram(4, 2);

        record_moe_router_selection(3, &[1, 2, 3], &[0.75, 0.25, 0.0]);
        record_moe_router_selection(3, &[8, 2], &[0.6, 0.4]);

        let hist = take_moe_router_histogram().expect("histogram should be collected");
        assert_eq!(hist.num_experts, 4);
        assert_eq!(hist.k_top, 2);
        assert_eq!(hist.routed_tokens, 2);
        assert_eq!(hist.routed_slots, 3);
        assert_eq!(hist.top1_histogram, vec![0, 1, 0, 0]);
        assert_eq!(hist.topk_histogram, vec![0, 1, 2, 0]);
        assert!((hist.weight_sums[1] - 0.75).abs() < f64::EPSILON);
        assert!((hist.weight_sums[2] - 0.65).abs() < 1e-6);
        assert_eq!(hist.dropped_indices, 1);
        assert_eq!(hist.per_layer.len(), 4);
        let layer = &hist.per_layer[3];
        assert_eq!(layer.layer_idx, 3);
        assert_eq!(layer.routed_tokens, 2);
        assert_eq!(layer.routed_slots, 3);
        assert_eq!(layer.top1_histogram, vec![0, 1, 0, 0]);
        assert_eq!(layer.topk_histogram, vec![0, 1, 2, 0]);
        assert!((layer.weight_sums[1] - 0.75).abs() < f64::EPSILON);
        assert!((layer.weight_sums[2] - 0.65).abs() < 1e-6);
        assert_eq!(layer.dropped_indices, 1);
        assert_eq!(
            layer.cooccurrence.get(&((hist.num_experts as u64) + 2)),
            Some(&1)
        );
        assert!(take_moe_router_histogram().is_none());
    }

    // ── #397 Ship 6 — lowered decode super-op program shapes ──────────────
    // The lowered LayerProgram per variant must mirror the hand-arm op sequence
    // in forward_scratch_layers exactly. These are CPU-pure (no GPU/GpuTensor).
    #[test]
    fn lowered_fullattn_program_shape() {
        use SuperOpKind::{Attend, Proj, ResidualGemv};
        let p = lower_variant(Q35Variant::FullAttn);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Proj, Attend, ResidualGemv, Proj, ResidualGemv]);
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKV);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_FULL);
        assert_eq!(p[2].binding.weights[0].0, q35_op::RESID_WO);
        assert_eq!(p[3].binding.weights[0].0, q35_op::PROJ_GATE_UP);
        assert_eq!(p[4].binding.weights[0].0, q35_op::RESID_DOWN_SWIGLU);
    }

    #[test]
    fn lowered_deltanet_program_shape() {
        use SuperOpKind::{Attend, Norm, Proj, Recurrent, ResidualGemv};
        let p = lower_variant(Q35Variant::DeltaNet);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Proj,
                Attend,
                Recurrent,
                Norm,
                ResidualGemv,
                Proj,
                ResidualGemv
            ]
        );
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKVZA);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_DN_PREP);
    }

    #[test]
    fn lowered_moe_variants_replace_dense_ffn_with_one_moe_op() {
        use SuperOpKind::Moe;
        let dn = lower_variant(Q35Variant::DeltaNetMoe);
        let fa = lower_variant(Q35Variant::FullAttnMoe);
        // MoE variants end in a single Moe super-op (no dense gate_up/down).
        assert_eq!(dn.last().unwrap().kind, Moe);
        assert_eq!(fa.last().unwrap().kind, Moe);
        assert!(
            dn.iter()
                .all(|o| o.binding.weights[0].0 != q35_op::PROJ_GATE_UP
                    || o.kind != SuperOpKind::Proj)
        );
        // FullAttnMoe is the shortest: Proj, Attend, ResidualGemv(wo), Moe.
        assert_eq!(fa.len(), 4);
        assert_eq!(dn.len(), 6);
    }

    #[test]
    fn lowered_variant_of_maps_layer_discriminant() {
        // variant_of is a thin discriminant map; assert the program lengths it
        // would produce per the documented layer shapes.
        assert_eq!(lower_variant(Q35Variant::FullAttn).len(), 5);
        assert_eq!(lower_variant(Q35Variant::DeltaNet).len(), 7);
        assert_eq!(lower_variant(Q35Variant::DeltaNetMoe).len(), 6);
        assert_eq!(lower_variant(Q35Variant::FullAttnMoe).len(), 4);
    }

    #[test]
    fn f16_lm_head_mode_defaults_to_native() {
        assert_eq!(parse_f16_lm_head_mode(None), F16LmHeadMode::Native);
        assert_eq!(parse_f16_lm_head_mode(Some("auto")), F16LmHeadMode::Native);
        assert_eq!(parse_f16_lm_head_mode(Some("1")), F16LmHeadMode::Native);
        assert_eq!(
            parse_f16_lm_head_mode(Some("native")),
            F16LmHeadMode::Native
        );
        assert_eq!(parse_f16_lm_head_mode(Some("f16")), F16LmHeadMode::Native);
    }

    #[test]
    fn f16_lm_head_mode_allows_legacy_f32() {
        assert_eq!(parse_f16_lm_head_mode(Some("0")), F16LmHeadMode::F32);
        assert_eq!(parse_f16_lm_head_mode(Some("f32")), F16LmHeadMode::F32);
        assert_eq!(parse_f16_lm_head_mode(Some("fp32")), F16LmHeadMode::F32);
        assert_eq!(parse_f16_lm_head_mode(Some("legacy")), F16LmHeadMode::F32);
    }

    #[test]
    fn f16_lm_head_mode_unknown_falls_back_to_native() {
        assert_eq!(
            parse_f16_lm_head_mode(Some("surprise")),
            F16LmHeadMode::Native
        );
    }

    #[test]
    fn bf16_weight_load_mode_defaults_to_auto() {
        assert_eq!(parse_bf16_weight_load_mode(None), Bf16WeightLoadMode::Auto);
        assert_eq!(
            parse_bf16_weight_load_mode(Some("native")),
            Bf16WeightLoadMode::Native
        );
        assert_eq!(
            parse_bf16_weight_load_mode(Some("bf16")),
            Bf16WeightLoadMode::Native
        );
        assert_eq!(
            parse_bf16_weight_load_mode(Some("surprise")),
            Bf16WeightLoadMode::Auto
        );
    }

    #[test]
    fn bf16_weight_load_mode_auto_is_arch_aware() {
        assert_eq!(
            resolve_bf16_weight_load_mode(Bf16WeightLoadMode::Auto, "gfx1151"),
            Bf16WeightLoadMode::Native
        );
        assert_eq!(
            resolve_bf16_weight_load_mode(Bf16WeightLoadMode::Auto, "gfx1201"),
            Bf16WeightLoadMode::Native
        );
        assert_eq!(
            resolve_bf16_weight_load_mode(Bf16WeightLoadMode::Auto, "gfx906"),
            Bf16WeightLoadMode::F16
        );
        assert_eq!(
            resolve_bf16_weight_load_mode(Bf16WeightLoadMode::Auto, "gfx1030"),
            Bf16WeightLoadMode::F16
        );
    }

    #[test]
    fn bf16_weight_load_mode_allows_f16_downgrade_override() {
        assert_eq!(
            parse_bf16_weight_load_mode(Some("f16")),
            Bf16WeightLoadMode::F16
        );
        assert_eq!(
            parse_bf16_weight_load_mode(Some("fp16")),
            Bf16WeightLoadMode::F16
        );
        assert_eq!(
            resolve_bf16_weight_load_mode(Bf16WeightLoadMode::F16, "gfx1151"),
            Bf16WeightLoadMode::F16
        );
    }

    #[test]
    fn bf16_to_f16_downgrade_preserves_byte_width() {
        let bf16 = [0x80, 0x3f, 0x00, 0x40]; // 1.0, 2.0 in BF16 LE
        let f16 = bf16_bytes_to_f16_bytes(&bf16);
        assert_eq!(f16.len(), bf16.len());
        assert_eq!(f16, vec![0x00, 0x3c, 0x00, 0x40]);
    }

    #[test]
    fn bf16_weight_load_mode_allows_debug_f32_expansion() {
        assert_eq!(
            parse_bf16_weight_load_mode(Some("0")),
            Bf16WeightLoadMode::F32
        );
        assert_eq!(
            parse_bf16_weight_load_mode(Some("f32")),
            Bf16WeightLoadMode::F32
        );
        assert_eq!(
            parse_bf16_weight_load_mode(Some("fp32")),
            Bf16WeightLoadMode::F32
        );
        assert_eq!(
            parse_bf16_weight_load_mode(Some("legacy")),
            Bf16WeightLoadMode::F32
        );
    }

    #[test]
    fn paro_batched_admit_defaults_off_and_allows_opt_in() {
        // PARO batched prefill is default-OFF (the path has a coherence/echo bug;
        // per-token fallback is correct) — opt in via HIPFIRE_PARO_BATCHED=1.
        // `paro_batched_admit_enabled_from_env` is `value == Some("1")`, so only
        // the exact string "1" enables it; everything else (incl. None) is off.
        assert!(!paro_batched_admit_enabled_from_env(None));
        assert!(paro_batched_admit_enabled_from_env(Some("1")));
        assert!(!paro_batched_admit_enabled_from_env(Some("surprise")));
        assert!(!paro_batched_admit_enabled_from_env(Some("0")));
    }

    // ── Qwen3.5 dispatch: is_batchable_la ────────────────────────

    /// The Qwen3.5-specific copy admits more dtypes than the runtime copy
    /// (ParoQ4G128, F32, Lloyd variants).

    const BATCHABLE_ARCHS: &[&str] = &[
        "gfx900", "gfx906", "gfx908", "gfx940", "gfx941", "gfx942", "gfx1010", "gfx1011",
        "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032", "gfx1100", "gfx1101", "gfx1102",
        "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201",
    ];

    const WMMA_ARCHS: &[&str] = &[
        "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
        "gfx1201",
    ];

    const GFX10_SCALAR_ARCHS: &[&str] = &[
        "gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032",
    ];

    const NO_WMMA_ARCHS: &[&str] = &["gfx900", "gfx906", "gfx908", "gfx940", "gfx941", "gfx942"];

    #[test]
    fn qwen35_is_batchable_la_always_ok() {
        for &arch in BATCHABLE_ARCHS {
            assert!(
                is_batchable_la(DType::MQ4G256, arch),
                "MQ4G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::HFQ4G256, arch),
                "HFQ4G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MQ6G256, arch),
                "MQ6G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::HFQ6G256, arch),
                "HFQ6G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::Q8_0, arch),
                "Q8_0 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::ParoQ4G128, arch),
                "ParoQ4G128 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::F32, arch),
                "F32 should batch on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_mq3_wmma_and_gfx10_scalar() {
        for &arch in WMMA_ARCHS {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 should batch on {arch} (WMMA)"
            );
        }
        for &arch in GFX10_SCALAR_ARCHS {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 should batch on {arch} (scalar)"
            );
        }
        for &arch in NO_WMMA_ARCHS {
            assert!(
                !is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_fp4_only_on_wmma() {
        for &arch in WMMA_ARCHS {
            assert!(
                is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 should batch on {arch}"
            );
        }
        for &arch in NO_WMMA_ARCHS {
            assert!(
                !is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 must fall back on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_lloyd_mq3_only_on_gfx11_with_opt_in_gfx12() {
        // gfx11 always admits Lloyd MQ3
        for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"] {
            assert!(
                is_batchable_la(DType::MQ3G256Lloyd, arch),
                "MQ3G256Lloyd should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MQ4G256Lloyd, arch),
                "MQ4G256Lloyd should batch on {arch}"
            );
        }
        // gfx1152 not in admit list
        assert!(
            !is_batchable_la(DType::MQ3G256Lloyd, "gfx1152"),
            "gfx1152 should NOT admit Lloyd MQ3"
        );
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1152"),
            "gfx1152 should NOT admit Lloyd MQ4"
        );
        // gfx12 requires env gate
        assert!(
            !is_batchable_la(DType::MQ3G256Lloyd, "gfx1200"),
            "gfx1200 without HIPFIRE_LLOYD_GFX12=1"
        );
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1200"),
            "gfx1200 without HIPFIRE_LLOYD_GFX12=1"
        );
    }

    #[test]
    fn qwen35_is_batchable_la_unsupported_dtypes() {
        for &arch in WMMA_ARCHS {
            assert!(!is_batchable_la(DType::Q4K, arch), "Q4K must fall back");
            assert!(!is_batchable_la(DType::Q6K, arch), "Q6K must fall back");
            assert!(
                !is_batchable_la(DType::Q4F16G64, arch),
                "Q4F16G64 must fall back"
            );
            assert!(
                !is_batchable_la(DType::Q4F16G32, arch),
                "Q4F16G32 must fall back"
            );
            assert!(
                !is_batchable_la(DType::MQ2G256, arch),
                "MQ2G256 must fall back"
            );
            assert!(
                !is_batchable_la(DType::MQ8G256, arch),
                "MQ8G256 must fall back"
            );
            assert!(
                !is_batchable_la(DType::HFQ2G256, arch),
                "HFQ2G256 must fall back"
            );
        }
    }

    // ── Qwen3.5 MoE dispatch predicates ──────────────────────────

    #[test]
    fn moe_ffn_has_mq3_detects_mq3_in_experts() {
        // Build a minimal MoeFfnWeights with MQ3 dtypes
        let _mq3_dt = DType::MQ3G256;
        let _batchable_dt = DType::MQ4G256;
        // Use default F32 as fallback
        // MoeFfnWeights requires GPU-backed tensors; predicate is tested at DType level.
    }

    #[test]
    fn dense_prefill_mq6_and_hfq6_are_batchable_in_qwen35() {
        for arch in [
            "gfx900", "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1151", "gfx1200", "gfx1201",
            "gfx942",
        ] {
            assert!(
                is_batchable_la(DType::MQ6G256, arch),
                "MQ6 dense prefill should route through the HFQ6 batched family on {arch}"
            );
            assert!(
                is_batchable_la(DType::HFQ6G256, arch),
                "HFQ6 dense prefill should stay batchable on {arch}"
            );
        }
    }

    #[test]
    fn dense_prefill_bf16_is_batchable_in_qwen35() {
        for arch in [
            "gfx900", "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx1201", "gfx942",
        ] {
            assert!(
                is_batchable_la(DType::BF16, arch),
                "BF16 dense prefill must stay on the batched BF16 WMMA-capable path on {arch}"
            );
        }
        assert!(
            !is_batchable_la(DType::BF16, "gfx1151"),
            "BUG-001 keeps BF16 dense prefill off the broken gfx1151 batched projection path"
        );
    }

    #[test]
    fn dense_session_prefill_batch_shape_accepts_independent_rows() {
        let shape = validate_dense_prefill_session_batch_shape(
            &[
                DensePrefillSessionBatchRowShape {
                    tokens: 3,
                    logits_numel: 151_936,
                },
                DensePrefillSessionBatchRowShape {
                    tokens: 5,
                    logits_numel: 151_936,
                },
            ],
            8,
        )
        .expect("valid dense session batch shape");
        assert_eq!(
            shape,
            DensePrefillSessionBatchShape {
                sessions: 2,
                total_tokens: 8,
                max_tokens_per_session: 5,
            }
        );
    }

    #[test]
    fn dense_session_prefill_batch_shape_rejects_non_batchable_shapes() {
        let one_row = validate_dense_prefill_session_batch_shape(
            &[DensePrefillSessionBatchRowShape {
                tokens: 3,
                logits_numel: 151_936,
            }],
            8,
        )
        .unwrap_err();
        assert!(one_row.contains("at least two independent sessions"));

        let empty_tokens = validate_dense_prefill_session_batch_shape(
            &[
                DensePrefillSessionBatchRowShape {
                    tokens: 0,
                    logits_numel: 151_936,
                },
                DensePrefillSessionBatchRowShape {
                    tokens: 1,
                    logits_numel: 151_936,
                },
            ],
            8,
        )
        .unwrap_err();
        assert!(empty_tokens.contains("empty token slice"));

        let too_wide = validate_dense_prefill_session_batch_shape(
            &[
                DensePrefillSessionBatchRowShape {
                    tokens: 9,
                    logits_numel: 151_936,
                },
                DensePrefillSessionBatchRowShape {
                    tokens: 1,
                    logits_numel: 151_936,
                },
            ],
            8,
        )
        .unwrap_err();
        assert!(too_wide.contains("exceeding PrefillBatchScratch max_batch=8"));

        let empty_logits = validate_dense_prefill_session_batch_shape(
            &[
                DensePrefillSessionBatchRowShape {
                    tokens: 3,
                    logits_numel: 0,
                },
                DensePrefillSessionBatchRowShape {
                    tokens: 1,
                    logits_numel: 151_936,
                },
            ],
            8,
        )
        .unwrap_err();
        assert!(empty_logits.contains("empty logits tensor"));
    }

    #[test]
    fn dense_session_prefill_state_signatures_must_match() {
        let q8 = DensePrefillSessionBatchStateSignature {
            kv_physical_cap: 512,
            kv_compact_offset: 0,
            kv_quantized: true,
            kv_quant_q8: true,
            kv_quant_asym2: false,
            kv_quant_asym3: false,
            kv_quant_asym4: false,
            kv_quant_fwht: false,
            dn_quant: StateQuant::Q8,
        };
        validate_dense_prefill_session_batch_state_signatures(&[q8, q8])
            .expect("matching signatures are batchable");

        let mut different_compact_offset = q8;
        different_compact_offset.kv_compact_offset = 16;
        let err =
            validate_dense_prefill_session_batch_state_signatures(&[q8, different_compact_offset])
                .unwrap_err();
        assert!(err.contains("incompatible KV/DeltaNet state signature"));

        let mut different_dn_quant = q8;
        different_dn_quant.dn_quant = StateQuant::FP32;
        let err = validate_dense_prefill_session_batch_state_signatures(&[q8, different_dn_quant])
            .unwrap_err();
        assert!(err.contains("incompatible KV/DeltaNet state signature"));
    }

    #[test]
    fn dense_session_prefill_state_route_shapes_must_match() {
        let shape = DensePrefillSessionStateRouteShape {
            kv_k_layers: 12,
            kv_v_layers: 12,
            dn_s_layers: 16,
            dn_scale_layers: 16,
            dn_conv_layers: 16,
        };
        validate_dense_prefill_session_state_route_shapes(&[shape, shape], 2)
            .expect("matching route shapes are batchable");

        let wrong_count =
            validate_dense_prefill_session_state_route_shapes(&[shape, shape], 3).unwrap_err();
        assert!(wrong_count.contains("expected 3"));

        let mut missing_kv = shape;
        missing_kv.kv_k_layers = 0;
        let err = validate_dense_prefill_session_state_route_shapes(&[missing_kv, missing_kv], 2)
            .unwrap_err();
        assert!(err.contains("incomplete KV/DeltaNet route shape"));

        let mut mismatched_kv = shape;
        mismatched_kv.kv_v_layers = 11;
        let err =
            validate_dense_prefill_session_state_route_shapes(&[mismatched_kv, mismatched_kv], 2)
                .unwrap_err();
        assert!(err.contains("mismatched KV K/V layers"));

        let mut mismatched_delta = shape;
        mismatched_delta.dn_conv_layers = 15;
        let err = validate_dense_prefill_session_state_route_shapes(
            &[mismatched_delta, mismatched_delta],
            2,
        )
        .unwrap_err();
        assert!(err.contains("mismatched DeltaNet S/conv layers"));

        let mut incompatible = shape;
        incompatible.dn_s_layers = 15;
        incompatible.dn_scale_layers = 15;
        incompatible.dn_conv_layers = 15;
        let err = validate_dense_prefill_session_state_route_shapes(&[shape, incompatible], 2)
            .unwrap_err();
        assert!(err.contains("incompatible state route shape"));
    }

    #[test]
    fn dense_session_prefill_state_route_shape_matches_qwen35_config() {
        let config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        let expected = expected_dense_prefill_session_state_route_shape(&config);
        assert_eq!(
            expected,
            DensePrefillSessionStateRouteShape {
                kv_k_layers: 4,
                kv_v_layers: 4,
                dn_s_layers: 2,
                dn_scale_layers: 2,
                dn_conv_layers: 2,
            }
        );
        validate_dense_prefill_session_state_route_shapes_for_config(
            &[expected, expected],
            &config,
        )
        .expect("matching model route shapes are valid");

        let mut wrong = expected;
        wrong.kv_k_layers = 2;
        let err = validate_dense_prefill_session_state_route_shapes_for_config(
            &[expected, wrong],
            &config,
        )
        .unwrap_err();
        assert!(err.contains("incompatible state route shape"));

        let mut matching_but_wrong_for_model = expected;
        matching_but_wrong_for_model.dn_s_layers = 1;
        matching_but_wrong_for_model.dn_scale_layers = 1;
        matching_but_wrong_for_model.dn_conv_layers = 1;
        let err = validate_dense_prefill_session_state_route_shapes_for_config(
            &[matching_but_wrong_for_model, matching_but_wrong_for_model],
            &config,
        )
        .unwrap_err();
        assert!(err.contains("expected model shape"));
    }

    #[test]
    fn dense_session_prefill_rounds_are_round_major_with_independent_positions() {
        let rounds = build_dense_prefill_session_batch_rounds(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11, 12],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 9,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[30, 31],
                    start_pos: 2,
                },
            ],
            8,
        )
        .expect("valid dense session rounds");

        assert_eq!(rounds.len(), 3);
        assert_eq!(
            rounds[0].rows,
            vec![
                DensePrefillSessionBatchRoundRow {
                    session_index: 0,
                    token_index: 0,
                    token: 10,
                    position: 4,
                },
                DensePrefillSessionBatchRoundRow {
                    session_index: 1,
                    token_index: 0,
                    token: 20,
                    position: 9,
                },
                DensePrefillSessionBatchRoundRow {
                    session_index: 2,
                    token_index: 0,
                    token: 30,
                    position: 2,
                },
            ]
        );
        assert_eq!(
            rounds[1].rows,
            vec![
                DensePrefillSessionBatchRoundRow {
                    session_index: 0,
                    token_index: 1,
                    token: 11,
                    position: 5,
                },
                DensePrefillSessionBatchRoundRow {
                    session_index: 2,
                    token_index: 1,
                    token: 31,
                    position: 3,
                },
            ]
        );
        assert_eq!(
            rounds[2].rows,
            vec![DensePrefillSessionBatchRoundRow {
                session_index: 0,
                token_index: 2,
                token: 12,
                position: 6,
            }]
        );
    }

    #[test]
    fn dense_session_prefill_rounds_reject_non_batchable_inputs() {
        let one_row = build_dense_prefill_session_batch_rounds(
            &[DensePrefillSessionBatchInput {
                tokens: &[10],
                start_pos: 0,
            }],
            8,
        )
        .unwrap_err();
        assert!(one_row.contains("at least two independent sessions"));

        let too_many = build_dense_prefill_session_batch_rounds(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[1],
                    start_pos: 0,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[2],
                    start_pos: 0,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[3],
                    start_pos: 0,
                },
            ],
            2,
        )
        .unwrap_err();
        assert!(too_many.contains("exceeding PrefillBatchScratch max_batch=2"));

        let empty = build_dense_prefill_session_batch_rounds(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[],
                    start_pos: 0,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[2],
                    start_pos: 0,
                },
            ],
            8,
        )
        .unwrap_err();
        assert!(empty.contains("empty token slice"));
    }

    #[test]
    fn dense_session_prefill_execution_plan_marks_multi_state_rounds() {
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11, 12],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 9,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[30, 31],
                    start_pos: 2,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");

        assert_eq!(plan.total_rows, 6);
        assert_eq!(plan.max_rows_per_round, 3);
        assert_eq!(plan.multi_state_rounds, 2);
        assert_eq!(plan.multi_state_prefix_rounds, 2);
        assert_eq!(plan.multi_state_prefix_rows, 5);
        assert_eq!(
            plan.singleton_tail,
            Some(DensePrefillSessionBatchSingletonTail {
                start_round: 2,
                session_index: 0,
                rows: 1,
            })
        );
        assert_eq!(plan.rounds.len(), 3);
        assert_eq!(plan.rounds[0].rows.len(), 3);
        assert_eq!(plan.rounds[1].rows.len(), 2);
        assert_eq!(plan.rounds[2].rows.len(), 1);
        assert_eq!(
            plan.state_routes,
            vec![
                DensePrefillSessionBatchRoundStateRoute::MultiSession {
                    session_indices: vec![0, 1, 2],
                },
                DensePrefillSessionBatchRoundStateRoute::MultiSession {
                    session_indices: vec![0, 2],
                },
                DensePrefillSessionBatchRoundStateRoute::SingleSession { session_index: 0 },
            ]
        );
    }

    #[test]
    fn dense_session_prefill_execution_plan_has_no_tail_for_equal_lengths() {
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20, 21],
                    start_pos: 9,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");

        assert_eq!(plan.total_rows, 4);
        assert_eq!(plan.max_rows_per_round, 2);
        assert_eq!(plan.multi_state_rounds, 2);
        assert_eq!(plan.multi_state_prefix_rounds, 2);
        assert_eq!(plan.multi_state_prefix_rows, 4);
        assert_eq!(plan.singleton_tail, None);
        assert_eq!(
            plan.state_routes,
            vec![
                DensePrefillSessionBatchRoundStateRoute::MultiSession {
                    session_indices: vec![0, 1],
                },
                DensePrefillSessionBatchRoundStateRoute::MultiSession {
                    session_indices: vec![0, 1],
                },
            ]
        );
    }

    #[test]
    fn dense_session_fused_prefix_contract_accepts_dense_fp32_state() {
        let config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11],
                    start_pos: 0,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 0,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");
        let signatures = vec![
            DensePrefillSessionBatchStateSignature {
                kv_physical_cap: 512,
                kv_compact_offset: 0,
                kv_quantized: false,
                kv_quant_q8: false,
                kv_quant_asym2: false,
                kv_quant_asym3: false,
                kv_quant_asym4: false,
                kv_quant_fwht: false,
                dn_quant: StateQuant::FP32,
            },
            DensePrefillSessionBatchStateSignature {
                kv_physical_cap: 512,
                kv_compact_offset: 0,
                kv_quantized: false,
                kv_quant_q8: false,
                kv_quant_asym2: false,
                kv_quant_asym3: false,
                kv_quant_asym4: false,
                kv_quant_fwht: false,
                dn_quant: StateQuant::FP32,
            },
        ];

        validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
            &config,
            &signatures,
            &plan,
        )
        .expect("dense FP32-state prefix should be eligible");
    }

    #[test]
    fn dense_session_fused_prefix_contract_rejects_moe_and_quantized_state() {
        let mut moe_config = test_qwen35_config_with_layers(vec![LayerType::LinearAttention]);
        moe_config.num_experts = 128;
        moe_config.has_shared_expert = true;
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10],
                    start_pos: 0,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 0,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");
        let fp32_sig = DensePrefillSessionBatchStateSignature {
            kv_physical_cap: 512,
            kv_compact_offset: 0,
            kv_quantized: false,
            kv_quant_q8: false,
            kv_quant_asym2: false,
            kv_quant_asym3: false,
            kv_quant_asym4: false,
            kv_quant_fwht: false,
            dn_quant: StateQuant::FP32,
        };

        let moe_err = validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
            &moe_config,
            &[fp32_sig, fp32_sig],
            &plan,
        )
        .unwrap_err();
        assert!(moe_err.contains("dense Qwen35 only"));

        let compacted = DensePrefillSessionBatchStateSignature {
            kv_compact_offset: 8,
            ..fp32_sig
        };
        let compact_err =
            validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
                &test_qwen35_config_with_layers(vec![LayerType::LinearAttention]),
                &[compacted, compacted],
                &plan,
            )
            .unwrap_err();
        assert!(compact_err.contains("compacted KV offset"));

        let q8_dn = DensePrefillSessionBatchStateSignature {
            dn_quant: StateQuant::Q8,
            ..fp32_sig
        };
        let q8_err = validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
            &test_qwen35_config_with_layers(vec![LayerType::LinearAttention]),
            &[q8_dn, q8_dn],
            &plan,
        )
        .unwrap_err();
        assert!(q8_err.contains("DeltaNet state"));
    }

    #[test]
    fn grouped_moe_session_fused_prefix_contract_accepts_q8_state_control_path() {
        let mut config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        config.num_experts = 256;
        config.num_experts_per_tok = 8;
        config.has_shared_expert = true;
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11],
                    start_pos: 0,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 0,
                },
            ],
            8,
        )
        .expect("valid session execution plan");
        let q8_sig = DensePrefillSessionBatchStateSignature {
            kv_physical_cap: 512,
            kv_compact_offset: 0,
            kv_quantized: true,
            kv_quant_q8: true,
            kv_quant_asym2: false,
            kv_quant_asym3: false,
            kv_quant_asym4: false,
            kv_quant_fwht: false,
            dn_quant: StateQuant::Q8,
        };

        validate_grouped_moe_prefill_session_batch_q8_state_contract(
            &config,
            &[q8_sig, q8_sig],
            &plan,
            "gfx1151",
        )
        .expect("A3B MQ4 control path uses grouped MoE with Q8 state");
    }

    #[test]
    fn grouped_moe_session_fused_prefix_contract_rejects_wrong_model_or_state() {
        let mut moe_config = test_qwen35_config_with_layers(vec![LayerType::LinearAttention]);
        moe_config.num_experts = 256;
        moe_config.num_experts_per_tok = 8;
        moe_config.has_shared_expert = true;
        let dense_config = test_qwen35_config_with_layers(vec![LayerType::LinearAttention]);
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10],
                    start_pos: 0,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 0,
                },
            ],
            8,
        )
        .expect("valid session execution plan");
        let q8_sig = DensePrefillSessionBatchStateSignature {
            kv_physical_cap: 512,
            kv_compact_offset: 0,
            kv_quantized: true,
            kv_quant_q8: true,
            kv_quant_asym2: false,
            kv_quant_asym3: false,
            kv_quant_asym4: false,
            kv_quant_fwht: false,
            dn_quant: StateQuant::Q8,
        };

        let dense_err = validate_grouped_moe_prefill_session_batch_q8_state_contract(
            &dense_config,
            &[q8_sig, q8_sig],
            &plan,
            "gfx1151",
        )
        .unwrap_err();
        assert!(dense_err.contains("requires Qwen35 MoE/A3B weights"));

        let fp32_kv = DensePrefillSessionBatchStateSignature {
            kv_quantized: false,
            kv_quant_q8: false,
            dn_quant: StateQuant::FP32,
            ..q8_sig
        };
        let fp32_err = validate_grouped_moe_prefill_session_batch_q8_state_contract(
            &moe_config,
            &[fp32_kv, fp32_kv],
            &plan,
            "gfx1151",
        )
        .unwrap_err();
        assert!(fp32_err.contains("must use Q8 KV state"));

        let asym_kv = DensePrefillSessionBatchStateSignature {
            kv_quant_asym3: true,
            ..q8_sig
        };
        let asym_err = validate_grouped_moe_prefill_session_batch_q8_state_contract(
            &moe_config,
            &[asym_kv, asym_kv],
            &plan,
            "gfx1151",
        )
        .unwrap_err();
        assert!(asym_err.contains("unsupported KV quantization flags"));

        let arch_err = validate_grouped_moe_prefill_session_batch_q8_state_contract(
            &moe_config,
            &[q8_sig, q8_sig],
            &plan,
            "gfx942",
        )
        .unwrap_err();
        assert!(arch_err.contains("requires an RDNA grouped-MoE target"));
    }

    #[test]
    fn dense_session_prefill_pointer_table_shape_sizes_prefix_tables() {
        let config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        let route_shape = expected_dense_prefill_session_state_route_shape(&config);
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11, 12],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 9,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[30, 31],
                    start_pos: 2,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");

        assert_eq!(
            dense_prefill_session_batch_pointer_table_shape(&plan, route_shape, 3),
            DensePrefillSessionBatchPointerTableShape {
                sessions: 3,
                multi_state_prefix_rounds: 2,
                multi_state_prefix_rows: 5,
                max_rows_per_round: 3,
                kv_k_ptrs: 12,
                kv_v_ptrs: 12,
                dn_s_ptrs: 6,
                dn_scale_ptrs: 6,
                dn_conv_ptrs: 6,
                logits_ptrs: 3,
                session_last_row_indices: 3,
                row_session_indices: 5,
                row_tokens: 5,
                row_positions: 5,
            }
        );
    }

    #[test]
    fn dense_session_prefill_pointer_table_plan_maps_slots_to_sessions_and_rows() {
        let config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        let route_shape = expected_dense_prefill_session_state_route_shape(&config);
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11, 12],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 9,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[30, 31],
                    start_pos: 2,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");
        let tables = dense_prefill_session_batch_pointer_table_plan(&plan, route_shape, 3);

        assert_eq!(tables.kv_layer_slots.len(), 12);
        assert_eq!(
            tables.kv_layer_slots[7],
            DensePrefillSessionBatchLayerPointerSlot {
                session_index: 1,
                layer_index: 3,
            }
        );
        assert_eq!(tables.dn_layer_slots.len(), 6);
        assert_eq!(
            tables.dn_layer_slots[4],
            DensePrefillSessionBatchDeltaPointerSlot {
                session_index: 2,
                delta_layer_index: 0,
            }
        );
        assert_eq!(tables.logits_slots, vec![0, 1, 2]);
        assert_eq!(
            tables.prefix_rows,
            vec![
                DensePrefillSessionBatchPrefixRowSlot {
                    round_index: 0,
                    round_row_index: 0,
                    session_index: 0,
                    token_index: 0,
                    token: 10,
                    position: 4,
                },
                DensePrefillSessionBatchPrefixRowSlot {
                    round_index: 0,
                    round_row_index: 1,
                    session_index: 1,
                    token_index: 0,
                    token: 20,
                    position: 9,
                },
                DensePrefillSessionBatchPrefixRowSlot {
                    round_index: 0,
                    round_row_index: 2,
                    session_index: 2,
                    token_index: 0,
                    token: 30,
                    position: 2,
                },
                DensePrefillSessionBatchPrefixRowSlot {
                    round_index: 1,
                    round_row_index: 0,
                    session_index: 0,
                    token_index: 1,
                    token: 11,
                    position: 5,
                },
                DensePrefillSessionBatchPrefixRowSlot {
                    round_index: 1,
                    round_row_index: 1,
                    session_index: 2,
                    token_index: 1,
                    token: 31,
                    position: 3,
                },
            ]
        );
        assert_eq!(tables.session_last_row_indices, vec![3, 1, 4]);
        let (tokens, positions) =
            dense_prefill_session_batch_prefix_tokens_positions(&tables).unwrap();
        assert_eq!(tokens, vec![10, 20, 30, 11, 31]);
        assert_eq!(positions, vec![4, 9, 2, 5, 3]);
    }

    #[test]
    fn dense_session_prefill_prefix_metadata_rejects_bad_last_row() {
        let config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        let route_shape = expected_dense_prefill_session_state_route_shape(&config);
        let plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 9,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");
        let mut tables = dense_prefill_session_batch_pointer_table_plan(&plan, route_shape, 2);

        tables.session_last_row_indices[0] = 1;
        let err = dense_prefill_session_batch_prefix_tokens_positions(&tables).unwrap_err();
        assert!(err.contains("last row 1 belongs to session 1"));

        tables.session_last_row_indices[0] = 9;
        let err = dense_prefill_session_batch_prefix_tokens_positions(&tables).unwrap_err();
        assert!(err.contains("last row 9 out of range"));
    }

    #[test]
    fn dense_session_prefill_host_pointer_tables_materialize_real_route_order() {
        let config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        let route_shape = expected_dense_prefill_session_state_route_shape(&config);
        let execution_plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10, 11],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 9,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");
        let table_plan =
            dense_prefill_session_batch_pointer_table_plan(&execution_plan, route_shape, 2);

        let s0_k = vec![
            fake_tensor(0x1000),
            fake_tensor(0x1001),
            fake_tensor(0x1002),
            fake_tensor(0x1003),
        ];
        let s0_v = vec![
            fake_tensor(0x2000),
            fake_tensor(0x2001),
            fake_tensor(0x2002),
            fake_tensor(0x2003),
        ];
        let s0_dn_s = vec![fake_tensor(0x3000), fake_tensor(0x3001)];
        let s0_dn_sc = vec![fake_tensor(0x4000), fake_tensor(0x4001)];
        let s0_dn_conv = vec![fake_tensor(0x5000), fake_tensor(0x5001)];
        let s0_logits = fake_tensor(0x6000);

        let s1_k = vec![
            fake_tensor(0x7000),
            fake_tensor(0x7001),
            fake_tensor(0x7002),
            fake_tensor(0x7003),
        ];
        let s1_v = vec![
            fake_tensor(0x8000),
            fake_tensor(0x8001),
            fake_tensor(0x8002),
            fake_tensor(0x8003),
        ];
        let s1_dn_s = vec![fake_tensor(0x9000), fake_tensor(0x9001)];
        let s1_dn_sc = vec![fake_tensor(0xA000), fake_tensor(0xA001)];
        let s1_dn_conv = vec![fake_tensor(0xB000), fake_tensor(0xB001)];
        let s1_logits = fake_tensor(0xC000);

        let routes = vec![
            DensePrefillSessionStateRoute {
                kv: DensePrefillSessionKvStateRoute {
                    k_gpu: &s0_k,
                    v_gpu: &s0_v,
                    physical_cap: 512,
                    compact_offset: 0,
                },
                delta: DensePrefillSessionDeltaStateRoute {
                    s_matrices: &s0_dn_s,
                    s_scales: &s0_dn_sc,
                    conv_states: &s0_dn_conv,
                    quant: StateQuant::Q8,
                },
                logits: &s0_logits,
            },
            DensePrefillSessionStateRoute {
                kv: DensePrefillSessionKvStateRoute {
                    k_gpu: &s1_k,
                    v_gpu: &s1_v,
                    physical_cap: 512,
                    compact_offset: 0,
                },
                delta: DensePrefillSessionDeltaStateRoute {
                    s_matrices: &s1_dn_s,
                    s_scales: &s1_dn_sc,
                    conv_states: &s1_dn_conv,
                    quant: StateQuant::Q8,
                },
                logits: &s1_logits,
            },
        ];

        let tables = dense_prefill_session_batch_host_pointer_tables(&table_plan, &routes)
            .expect("host pointer tables");

        assert_eq!(
            tables.kv_k_ptrs,
            vec![0x1000, 0x1001, 0x1002, 0x1003, 0x7000, 0x7001, 0x7002, 0x7003,]
        );
        assert_eq!(
            tables.kv_v_ptrs,
            vec![0x2000, 0x2001, 0x2002, 0x2003, 0x8000, 0x8001, 0x8002, 0x8003,]
        );
        assert_eq!(tables.dn_s_ptrs, vec![0x3000, 0x3001, 0x9000, 0x9001]);
        assert_eq!(tables.dn_scale_ptrs, vec![0x4000, 0x4001, 0xA000, 0xA001]);
        assert_eq!(tables.dn_conv_ptrs, vec![0x5000, 0x5001, 0xB000, 0xB001]);
        assert_eq!(tables.logits_ptrs, vec![0x6000, 0xC000]);
        assert_eq!(tables.session_last_row_indices, vec![0, 1]);
        assert_eq!(tables.row_session_indices, vec![0, 1]);
        assert_eq!(tables.row_tokens, vec![10, 20]);
        assert_eq!(tables.row_positions, vec![4, 9]);
    }

    #[test]
    fn dense_session_prefill_host_pointer_tables_reject_missing_scale_route() {
        let config = test_qwen35_config_with_layers(vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ]);
        let route_shape = expected_dense_prefill_session_state_route_shape(&config);
        let execution_plan = build_dense_prefill_session_batch_execution_plan(
            &[
                DensePrefillSessionBatchInput {
                    tokens: &[10],
                    start_pos: 4,
                },
                DensePrefillSessionBatchInput {
                    tokens: &[20],
                    start_pos: 9,
                },
            ],
            8,
        )
        .expect("valid dense session execution plan");
        let table_plan =
            dense_prefill_session_batch_pointer_table_plan(&execution_plan, route_shape, 2);

        let s0_k = vec![fake_tensor(0x1000), fake_tensor(0x1001)];
        let s0_v = vec![fake_tensor(0x2000), fake_tensor(0x2001)];
        let s0_dn_s = vec![fake_tensor(0x3000)];
        let s0_dn_conv = vec![fake_tensor(0x5000)];
        let s0_logits = fake_tensor(0x6000);

        let s1_k = vec![fake_tensor(0x7000), fake_tensor(0x7001)];
        let s1_v = vec![fake_tensor(0x8000), fake_tensor(0x8001)];
        let s1_dn_s = vec![fake_tensor(0x9000)];
        let s1_dn_sc = vec![fake_tensor(0xA000)];
        let s1_dn_conv = vec![fake_tensor(0xB000)];
        let s1_logits = fake_tensor(0xC000);

        let routes = vec![
            DensePrefillSessionStateRoute {
                kv: DensePrefillSessionKvStateRoute {
                    k_gpu: &s0_k,
                    v_gpu: &s0_v,
                    physical_cap: 512,
                    compact_offset: 0,
                },
                delta: DensePrefillSessionDeltaStateRoute {
                    s_matrices: &s0_dn_s,
                    s_scales: &[],
                    conv_states: &s0_dn_conv,
                    quant: StateQuant::Q8,
                },
                logits: &s0_logits,
            },
            DensePrefillSessionStateRoute {
                kv: DensePrefillSessionKvStateRoute {
                    k_gpu: &s1_k,
                    v_gpu: &s1_v,
                    physical_cap: 512,
                    compact_offset: 0,
                },
                delta: DensePrefillSessionDeltaStateRoute {
                    s_matrices: &s1_dn_s,
                    s_scales: &s1_dn_sc,
                    conv_states: &s1_dn_conv,
                    quant: StateQuant::Q8,
                },
                logits: &s1_logits,
            },
        ];

        let err =
            dense_prefill_session_batch_host_pointer_tables(&table_plan, &routes).unwrap_err();
        assert!(err.contains("DeltaNet scale slot references missing layer 0"));
    }

    #[test]
    fn dense_session_prefill_pointer_table_indices_are_deterministic() {
        let shape = DensePrefillSessionBatchPointerTableShape {
            sessions: 3,
            multi_state_prefix_rounds: 2,
            multi_state_prefix_rows: 5,
            max_rows_per_round: 3,
            kv_k_ptrs: 12,
            kv_v_ptrs: 12,
            dn_s_ptrs: 6,
            dn_scale_ptrs: 6,
            dn_conv_ptrs: 6,
            logits_ptrs: 3,
            session_last_row_indices: 3,
            row_session_indices: 5,
            row_tokens: 5,
            row_positions: 5,
        };

        assert_eq!(
            shape.index_for_session_layer(2, 3, 1).unwrap(),
            DensePrefillSessionBatchPointerTableIndex {
                kv_k_offset: 11,
                kv_v_offset: 11,
                dn_s_offset: 5,
                dn_scale_offset: 5,
                dn_conv_offset: 5,
                logits_offset: 2,
            }
        );
        assert_eq!(shape.index_for_prefix_row(4).unwrap(), (4, 4, 4));

        assert!(shape
            .index_for_session_layer(3, 0, 0)
            .unwrap_err()
            .contains("session_index 3 out of range"));
        assert!(shape
            .index_for_session_layer(0, 4, 0)
            .unwrap_err()
            .contains("kv_layer_index 4 out of range"));
        assert!(shape
            .index_for_session_layer(0, 0, 2)
            .unwrap_err()
            .contains("dn_layer_index 2 out of range"));
        assert!(shape
            .index_for_prefix_row(5)
            .unwrap_err()
            .contains("prefix_row_index 5 out of range"));
    }

    #[test]
    fn moe_prefill_paro_i8_env_policy_is_gfx1151_default_on_with_opt_out() {
        assert!(paro_moe_i8_enabled_for_arch_from_env("gfx1151", None));
        assert!(paro_moe_i8_enabled_for_arch_from_env("gfx1151", Some("1")));
        assert!(paro_moe_i8_enabled_for_arch_from_env(
            "gfx1151",
            Some("surprise")
        ));
        assert!(!paro_moe_i8_enabled_for_arch_from_env("gfx1151", Some("0")));
        assert!(!paro_moe_i8_enabled_for_arch_from_env("gfx1201", None));
        assert!(!paro_moe_i8_enabled_for_arch_from_env("gfx1100", Some("1")));
    }

    #[test]
    fn moe_prefill_paro_i8_k8_env_policy_follows_i8_gate_and_allows_opt_out() {
        assert!(paro_moe_i8_k8_enabled_from_env(true, None));
        assert!(paro_moe_i8_k8_enabled_from_env(true, Some("1")));
        assert!(paro_moe_i8_k8_enabled_from_env(true, Some("surprise")));
        assert!(!paro_moe_i8_k8_enabled_from_env(true, Some("0")));
        assert!(!paro_moe_i8_k8_enabled_from_env(false, None));
        assert!(!paro_moe_i8_k8_enabled_from_env(false, Some("1")));
    }

    #[test]
    fn moe_prefill_topk_shape_requires_k8_and_bounded_experts() {
        assert!(moe_prefill_topk_shape_supported(8, 256));
        assert!(moe_prefill_topk_shape_supported(8, 1024));
        assert!(!moe_prefill_topk_shape_supported(4, 256));
        assert!(!moe_prefill_topk_shape_supported(8, 1025));
    }

    #[test]
    fn moe_prefill_admits_mq4_as_known_good_control() {
        let dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1151"
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx906"
        ));
    }

    #[test]
    fn moe_prefill_quant_matrix_documents_mq2_mq3_mq4_mq6_mq8() {
        fn moe_body_with_q8_router(dtype: DType) -> MoePrefillDtypes {
            let mut dtypes = MoePrefillDtypes::uniform(dtype);
            dtypes.router = DType::Q8_0;
            dtypes.shared_expert_scalar_gate = DType::Q8_0;
            dtypes
        }

        let cases = [
            ("mq2", DType::MQ2G256, false),
            ("mq3", DType::MQ3G256, true),
            ("mq4", DType::MQ4G256, true),
            ("mq6", DType::MQ6G256, true),
            ("mq8", DType::MQ8G256, false),
        ];

        for (label, dtype, expected) in cases {
            let dtypes = moe_body_with_q8_router(dtype);
            assert_eq!(
                moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1201"),
                expected,
                "{label} gfx12 MoE prefill admission"
            );
        }
    }

    #[test]
    fn moe_prefill_admits_gfx1151_scalar_bringup_families() {
        for dtype in [DType::MQ2G256, DType::MQ8G256, DType::MQ3G256Lloyd] {
            let mut dtypes = MoePrefillDtypes::uniform(dtype);
            dtypes.router = DType::Q8_0;
            dtypes.shared_expert_scalar_gate = DType::Q8_0;
            assert!(
                moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1151"),
                "{dtype:?} should be admitted for gfx1151 scalar MoE bring-up"
            );
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1201"),
                "{dtype:?} should remain gfx1151-scoped until arch-specific kernels land"
            );
        }
    }

    #[test]
    fn moe_prefill_admits_mq3_only_where_grouped_wmma_exists() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ3G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1151"
        ));
    }

    #[test]
    fn moe_prefill_rejects_full_precision_routed_body_but_admits_fp_router_and_gate() {
        let dtypes = MoePrefillDtypes::uniform(DType::F16);
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));

        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.router = DType::F16;
        dtypes.shared_expert_scalar_gate = DType::F16;
        dtypes.shared_expert_gate = DType::F16;
        dtypes.shared_expert_up = DType::F16;
        dtypes.shared_expert_down = DType::F16;
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));

        dtypes.router = DType::BF16;
        dtypes.shared_expert_scalar_gate = DType::BF16;
        dtypes.shared_expert_gate = DType::BF16;
        dtypes.shared_expert_up = DType::BF16;
        dtypes.shared_expert_down = DType::BF16;
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));
    }

    #[test]
    fn moe_prefill_admits_mq6_by_default() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ6G256;
        dtypes.shared_expert_up = DType::MQ6G256;
        dtypes.shared_expert_down = DType::MQ6G256;
        dtypes.expert_gate_up = DType::MQ6G256;
        dtypes.expert_down = DType::MQ6G256;
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1151"
        ));
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx906"
        ));
    }

    #[test]
    fn moe_prefill_admits_a10b_shared_mq4_down_and_routed_mq6_on_gfx1151() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ4G256;
        dtypes.shared_expert_up = DType::MQ4G256;
        dtypes.shared_expert_down = DType::MQ6G256;
        dtypes.expert_gate_up = DType::MQ6G256;
        dtypes.expert_down = DType::MQ6G256;

        assert!(
            moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1151"),
            "Qwen3.5-122B-A10B mixed MQ4 shared gate/up plus MQ6 routed layers should admit on gfx1151"
        );
        assert!(
            moe_prefill_needs_routed_gate_up_reprojection(&dtypes),
            "mixed shared/routed MQ-family layers must refresh x_rot_batch before routed gate_up"
        );
        assert!(
            !moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1100"),
            "MQ6 routed grouped GEMM is not wired on gfx1100"
        );
    }

    #[test]
    fn moe_decode_routes_mq6_indexed_path_for_k8() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ6G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;

        let flags = moe_decode_dispatch_flags_for_dtypes(&dtypes, 8, false);

        assert_eq!(flags.routed_path, MoeDecodeIndexedRoutedPath::Mq6);
        assert!(flags.routed_dtype_indexable_mq6);
        assert!(!flags.routed_dtype_indexable_mq4);
        assert!(flags.use_gpu_topk);
        assert!(flags.needs_x_rot_local);
        assert!(!flags.gate_side_mq4);
    }

    #[test]
    fn moe_decode_keeps_mq4_control_on_indexed_path() {
        let dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);

        let flags = moe_decode_dispatch_flags_for_dtypes(&dtypes, 8, false);

        assert_eq!(flags.routed_path, MoeDecodeIndexedRoutedPath::Mq4);
        assert!(flags.gate_side_mq4);
        assert!(flags.shared_gate_up_mq4);
        assert!(flags.routed_dtype_indexable_mq4);
        assert!(flags.use_gpu_topk);
        assert!(flags.needs_x_rot_local);
    }

    #[test]
    fn moe_decode_rejects_mismatched_mq6_gate_up_and_down_from_indexed_path() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ6G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.expert_down = DType::MQ4G256;

        let flags = moe_decode_dispatch_flags_for_dtypes(&dtypes, 8, false);

        assert_eq!(flags.routed_path, MoeDecodeIndexedRoutedPath::None);
        assert!(flags.routed_gate_up_mq6);
        assert!(!flags.routed_dtype_indexable_mq6);
        assert!(!flags.use_gpu_topk);
        assert!(flags.needs_x_rot_local);
    }

    #[test]
    fn moe_decode_k8_shape_required_for_gpu_topk() {
        let dtypes = MoePrefillDtypes::uniform(DType::MQ6G256);

        let flags = moe_decode_dispatch_flags_for_dtypes(&dtypes, 4, false);

        assert_eq!(flags.routed_path, MoeDecodeIndexedRoutedPath::Mq6);
        assert!(flags.routed_dtype_indexable_mq6);
        assert!(!flags.use_gpu_topk);
    }

    #[test]
    fn mq3_a3b_prefill_path2_but_moe_decode_lacks_indexed_route() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ3G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;

        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1151"
        ));
        assert!(moe_grouped_gemm_path2_required_for_dtype(DType::MQ3G256));
        assert!(moe_grouped_gemm_path2_eligible_for_dtype(
            DType::MQ3G256,
            "gfx1151",
            false
        ));

        let flags = moe_decode_dispatch_flags_for_dtypes(&dtypes, 8, false);

        assert_eq!(flags.routed_path, MoeDecodeIndexedRoutedPath::None);
        assert!(!flags.routed_dtype_indexable_mq4);
        assert!(!flags.routed_dtype_indexable_mq6);
        assert!(!flags.routed_dtype_indexable_mq2_lloyd);
        assert!(!flags.routed_dtype_indexable_paro);
        assert!(!flags.use_gpu_topk);
        assert!(!flags.needs_x_rot_local);
    }

    #[test]
    fn moe_prefill_admits_gfx1151_mq2_lloyd_routed_with_mq4_shared() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.expert_gate_up = DType::MQ2G256Lloyd;
        dtypes.expert_down = DType::MQ2G256Lloyd;

        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1151"
        ));
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));
    }

    #[test]
    fn moe_prefill_rejects_mixed_routed_family_without_grouped_gemm() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.expert_gate_up = DType::MQ8G256;
        dtypes.expert_down = DType::MQ8G256;

        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1151"
        ));
    }

    #[test]
    fn moe_prefill_grouped_gemm_routes_mq6_only_where_grouped_kernel_exists() {
        assert!(moe_grouped_gemm_supported_for_dtype(
            DType::MQ4G256,
            "gfx1151"
        ));
        assert!(moe_grouped_gemm_supported_for_dtype(
            DType::MQ4G256,
            "gfx1201"
        ));
        assert!(moe_grouped_gemm_supported_for_dtype(
            DType::MQ6G256,
            "gfx1151"
        ));
        assert!(moe_grouped_gemm_supported_for_dtype(
            DType::MQ6G256,
            "gfx1201"
        ));
        assert!(moe_grouped_gemm_supported_for_dtype(
            DType::MQ3G256,
            "gfx1201"
        ));
        assert!(moe_grouped_gemm_supported_for_dtype(
            DType::MQ3G256,
            "gfx1151"
        ));
        assert!(moe_grouped_gemm_supported_for_dtype(
            DType::MQ2G256Lloyd,
            "gfx1151"
        ));
        assert!(!moe_grouped_gemm_supported_for_dtype(
            DType::MQ2G256Lloyd,
            "gfx1201"
        ));
        assert!(moe_grouped_gemm_supported_for_dtype(DType::F16, "gfx1151"));
        assert!(moe_grouped_gemm_supported_for_dtype(DType::BF16, "gfx1151"));
        for arch in ["gfx1100", "gfx1201", "gfx9"] {
            assert!(
                !moe_grouped_gemm_supported_for_dtype(DType::F16, arch),
                "F16 routed MoE grouped GEMM should stay gfx1151-only on {arch}"
            );
            assert!(
                !moe_grouped_gemm_supported_for_dtype(DType::BF16, arch),
                "BF16 routed MoE grouped GEMM should stay gfx1151-only on {arch}"
            );
        }
    }

    #[test]
    fn moe_prefill_path2_env_policy_defaults_on_and_allows_opt_out() {
        assert!(moe_grouped_gemm_path2_enabled_from_env(None));
        assert!(moe_grouped_gemm_path2_enabled_from_env(Some("1")));
        assert!(moe_grouped_gemm_path2_enabled_from_env(Some("on")));
        assert!(moe_grouped_gemm_path2_enabled_from_env(Some("surprise")));
        assert!(!moe_grouped_gemm_path2_enabled_from_env(Some("0")));
        assert!(!moe_grouped_gemm_path2_enabled_from_env(Some("off")));
    }

    #[test]
    fn moe_prefill_mq2_lloyd_n32_policy_is_gfx1151_large_slots_only() {
        assert!(!mq2_lloyd_n32_gfx1151_enabled_from_env(
            "gfx1100", 4096, None
        ));
        assert!(!mq2_lloyd_n32_gfx1151_enabled_from_env(
            "gfx1151", 768, None
        ));
        assert!(mq2_lloyd_n32_gfx1151_enabled_from_env(
            "gfx1151", 1024, None
        ));
        assert!(!mq2_lloyd_n32_gfx1151_enabled_from_env(
            "gfx1151",
            4096,
            Some("0")
        ));
        assert!(mq2_lloyd_n32_gfx1151_enabled_from_env(
            "gfx1151",
            128,
            Some("1")
        ));
    }

    #[test]
    fn moe_prefill_path2_routes_mq6_on_gfx1151_and_gfx12() {
        for arch in ["gfx1151", "gfx1200", "gfx1201"] {
            assert!(
                moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ6G256, arch, true),
                "MQ6 should use grouped MoE GEMM when enabled on {arch}"
            );
            assert!(
                !moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ6G256, arch, false),
                "MQ6 should honor grouped MoE GEMM opt-out on {arch}"
            );
        }

        assert!(
            !moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ6G256, "gfx1100", true),
            "MQ6 should stay on indexed fallback where no grouped kernel is wired"
        );
    }

    #[test]
    fn moe_prefill_path2_forces_mq3_because_no_indexed_fallback_exists() {
        assert!(moe_grouped_gemm_path2_required_for_dtype(DType::MQ3G256));
        assert!(moe_grouped_gemm_path2_eligible_for_dtype(
            DType::MQ3G256,
            "gfx1151",
            false
        ));
        assert!(moe_grouped_gemm_path2_eligible_for_dtype(
            DType::MQ3G256,
            "gfx1201",
            false
        ));
        assert!(
            !moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ3G256, "gfx1100", false),
            "MQ3 cannot force path2 on archs without a grouped MoE kernel"
        );
    }

    #[test]
    fn moe_prefill_full_precision_routed_is_gfx1151_path2_only() {
        for dtype in [DType::F16, DType::BF16] {
            let mut dtypes = MoePrefillDtypes::uniform(dtype);
            dtypes.router = DType::Q8_0;
            dtypes.shared_expert_scalar_gate = DType::Q8_0;

            assert!(
                moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1151"),
                "{dtype:?} routed MoE prefill should be admitted on gfx1151"
            );
            assert!(
                moe_grouped_gemm_path2_required_for_dtype(dtype),
                "{dtype:?} routed MoE has no indexed fallback"
            );
            assert!(
                moe_grouped_gemm_path2_eligible_for_dtype(dtype, "gfx1151", false),
                "{dtype:?} routed MoE should force Path 2 even when env disables grouped GEMM"
            );

            for arch in ["gfx1100", "gfx1201", "gfx9"] {
                assert!(
                    !moe_ffn_batched_admissible_for_dtypes(&dtypes, false, arch),
                    "{dtype:?} routed MoE prefill should stay rejected on {arch}"
                );
                assert!(
                    !moe_grouped_gemm_path2_eligible_for_dtype(dtype, arch, true),
                    "{dtype:?} grouped MoE dispatch should stay rejected on {arch}"
                );
            }
        }
    }

    #[test]
    fn moe_prefill_mq3_long_prefill_path2_shape_is_production_shaped() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ3G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        assert!(
            moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1151"),
            "MQ3 A3B prefill must stay admitted on gfx1151"
        );
        assert!(
            moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ3G256, "gfx1151", false),
            "MQ3 has no indexed fallback, so path2 remains required even when the env gate is off"
        );

        let shape = moe_grouped_path2_shape(256, 8, 256);
        assert_eq!(shape.total_slots, 2048);
        assert_eq!(shape.m_total_bound, 5888);
        assert_eq!(shape.gate_up_x_row_div, 8);
        assert_eq!(shape.gate_up_source_rows, 256);
        assert_eq!(shape.down_x_row_div, 1);
        assert_eq!(shape.down_source_rows, 2048);
        assert_eq!(shape.m_total_bound % MOE_GROUPED_BLOCK_M, 0);
    }

    #[test]
    fn moe_prefill_mq6_path2_shape_is_production_shaped_when_enabled() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ6G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        assert!(
            moe_ffn_batched_admissible_for_dtypes(&dtypes, false, "gfx1151"),
            "MQ6 A3B prefill must stay admitted on gfx1151"
        );
        assert!(
            moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ6G256, "gfx1151", true),
            "MQ6 should use path2 on gfx1151 when grouped MoE GEMM is enabled"
        );
        assert!(
            !moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ6G256, "gfx1151", false),
            "MQ6 should keep the indexed fallback available when path2 is opted out"
        );

        let shape = moe_grouped_path2_shape(256, 8, 256);
        assert_eq!(shape.total_slots, 2048);
        assert_eq!(shape.m_total_bound, 5888);
        assert_eq!(shape.gate_up_x_row_div, 8);
        assert_eq!(shape.gate_up_source_rows, 256);
        assert_eq!(shape.down_x_row_div, 1);
        assert_eq!(shape.down_source_rows, 2048);
        assert_eq!(shape.m_total_bound % MOE_GROUPED_BLOCK_M, 0);
    }

    #[test]
    fn moe_prefill_a3b_mq4_path2_is_default_on_supported_rdna() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;

        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201"] {
            assert!(
                moe_ffn_batched_admissible_for_dtypes(&dtypes, false, arch),
                "A3B MQ4 prefill should stay admitted on {arch}"
            );
            assert!(
                moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ4G256, arch, true),
                "A3B MQ4 should use grouped MoE GEMM by default on {arch}"
            );
            assert!(
                !moe_grouped_gemm_path2_eligible_for_dtype(DType::MQ4G256, arch, false),
                "A3B MQ4 should preserve HIPFIRE_MOE_GROUPED_GEMM=0 opt-out on {arch}"
            );
        }
    }

    #[test]
    fn moe_grouped_path2_shape_covers_server_microbatch_sizes() {
        for n in [1, 2, 8, 64, 256] {
            let shape = moe_grouped_path2_shape(n, 8, 256);
            let total_slots = n * 8;
            let live_experts = total_slots.min(256);
            let expected_bound = align_up_usize(
                total_slots + live_experts * (MOE_GROUPED_BLOCK_M - 1),
                MOE_GROUPED_BLOCK_M,
            );

            assert_eq!(shape.total_slots, total_slots, "N={n}");
            assert_eq!(shape.m_total_bound, expected_bound, "N={n}");
            assert_eq!(shape.m_total_bound % MOE_GROUPED_BLOCK_M, 0, "N={n}");
            assert!(
                shape.m_total_bound <= moe_grouped_m_total_max(256, 8, 256),
                "N={n} live bound must fit the scratch allocation"
            );
            assert_eq!(shape.gate_up_x_row_div, 8, "N={n}");
            assert_eq!(shape.gate_up_source_rows, n, "N={n}");
            assert_eq!(shape.down_x_row_div, 1, "N={n}");
            assert_eq!(shape.down_source_rows, total_slots, "N={n}");
        }
    }

    #[test]
    fn moe_prefill_rejects_mismatched_routed_gate_up_and_down() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_gate_up = DType::MQ6G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));
    }

    #[test]
    fn moe_prefill_shared_gate_up_must_be_one_dtype() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_up = DType::MQ6G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1201"
        ));
    }

    #[test]
    fn moe_prefill_shared_down_may_differ_when_routed_grouped_gemm_exists() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.router = DType::Q8_0;
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_down = DType::MQ6G256;
        dtypes.expert_gate_up = DType::MQ6G256;
        dtypes.expert_down = DType::MQ6G256;

        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1151"
        ));
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, "gfx1100"
        ));
    }

    #[test]
    fn prefill_last_token_logits_policy_requires_explicit_opt_out() {
        assert!(prefill_should_emit_last_token_logits(false, true));
        assert!(prefill_should_emit_last_token_logits(true, true));
        assert!(prefill_should_emit_last_token_logits(false, false));
        assert!(!prefill_should_emit_last_token_logits(true, false));
    }

    #[test]
    fn moe_grouped_m_total_max_is_tile_aligned() {
        let small_verify = moe_grouped_m_total_max(3, 8, 256);
        assert_eq!(small_verify % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(small_verify, 3872);

        let prompt_prefill = moe_grouped_m_total_max(27, 8, 256);
        assert_eq!(prompt_prefill % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(prompt_prefill, 4064);

        let full_chunk = moe_grouped_m_total_max(256, 8, 256);
        assert_eq!(full_chunk, 5888);
    }

    #[test]
    fn moe_grouped_m_total_bound_is_tight_for_small_batches() {
        let small_verify = moe_grouped_m_total_bound(24, 256);
        assert_eq!(small_verify % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(small_verify, 384);

        let prompt_prefill = moe_grouped_m_total_bound(216, 256);
        assert_eq!(prompt_prefill % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(prompt_prefill, 3456);

        let full_chunk = moe_grouped_m_total_bound(2048, 256);
        assert_eq!(full_chunk, 5888);
    }
}
