// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kate, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 model: hybrid sliding-window + full attention, dense FFN (SwiGLU + gelu_pytorch_tanh).
//!
//! Architectural features vs. Qwen3.5:
//!   • Sliding-window attention on 5 of every 6 layers (window=1024).
//!   • Full attention layers use head_dim=512 (global_head_dim) with
//!     attention_k_eq_v: V is the pre-k_norm output of k_proj (no v_proj).
//!   • Partial proportional RoPE on full layers (first 64 of 512 dims rotate,
//!     rope_theta=1e6; sliding uses default RoPE with theta=10000).
//!   • Sandwich RMSNorm: input + post-attn + pre-FFN + post-FFN per layer,
//!     plus a learned per-layer `layer_scalar [1]` at layer end.
//!   • Attention scale = 1.0 (not 1/√d); Q/K norms absorb scaling.
//!   • Final logit softcap: `tanh(logits/30) * 30` before sampling.
//!   • MLP: SwiGLU with `gelu_pytorch_tanh` activation.
//!   • Tied LM head (embed_tokens.weight aliased).
//!   • Embed scale: sqrt(hidden_size) multiplied onto every embedding row lookup.

use hipfire_runtime::hfq::{HfqFile, load_awq_scale};
use hipfire_runtime::llama::{self, f16_to_f32, weight_gemv, WeightTensor, EmbeddingFormat};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemm::GemmParams;
use hipfire_dispatch::families::gemv::WeightRef;

/// #397 Ship 5.2: route a single PLAIN-batched prefill GEMM through
/// [`GemmFamily::run_key`] against an explicit dispatcher-entry key.
///
/// Byte-identical to the old `weight_gemm()` call for every (dtype × arch).
/// Only HFQ4G256, HFQ4G128 have batched kernels; other dtypes fall back to
/// the repeated-GEMV loop (same as `weight_gemm`'s fallback).
#[inline]
/// Env gate for WMMA prefill. Default OFF because WMMA F16 input quantization
/// loses ~3 mantissa bits vs F32 scalar, so WMMA results are NOT byte-identical
/// to the scalar GEMV path. Set HIPFIRE_WMMA_PREFILL=1 to opt in.
pub fn wmma_prefill_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("HIPFIRE_WMMA_PREFILL").map_or(false, |v| v == "1"))
}

/// Env gate for batched prefill (v2). Independent from WMMA.
/// Set HIPFIRE_BATCHED_PREFILL=1 to use batched projections + per-token attention.
pub fn batched_prefill_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("HIPFIRE_BATCHED_PREFILL").map_or(false, |v| v == "1")
    })
}

/// Batched GEMM for prefill projections.
///
/// Scalar path: routes through `GemmFamily::run_key` with the appropriate
/// kernel key for the weight dtype.
///
/// WMMA path (opt-in via `HIPFIRE_WMMA_PREFILL=1`): routes through
/// `GemmFamily::run` which resolves WMMA-eligible dtypes to their WMMA
/// kernel variant. The WMMA path converts F32 input to F16 automatically
/// (Bug 1 fix: `gemm_hfq4g256_wmma` now calls `ensure_fp16_x`).
///
/// MQ4G256 weights (Bug 2 fix): `GemmFamily::resolve` now maps
/// `DType::MQ4G256` → `GemmHfq4G256Wmma` / `GemmHfq4G256`, sharing the
/// same kernel binary since the layout is identical (136 B/group).
///
/// For dtypes that have no GEMM kernel, falls back to per-token GEMV.
fn run_prefill_gemm(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    // MQ4G256 weights are FWHT-rotated at quantize time. The per-token GEMV
    // path (weight_gemv) FWHT-rotates x before running the prerotated kernel;
    // the batched GEMM kernels (scalar gemm_hfq4g256 AND its WMMA siblings)
    // decode the same 136 B/group layout but do NOT rotate internally, so the
    // input must be pre-rotated here -- exactly like the eager
    // forward.rs::proj_gemm_batched (EAGLE verify) does. Skipping this fed raw
    // activations to the GEMM and produced garbage logits from token 0 on
    // every arch (root-caused on gfx1201 2026-06-10; gfx11 batched-prefill
    // validation predated the *.union.mq4 files, which is why it never fired
    // there).
    let x_rot_holder = if matches!(w.gpu_dtype, DType::MQ4G256) {
        let xr = gpu.alloc_tensor(&[batch_size, w.k], DType::F32)?;
        llama::rotate_x_mq_batched_for(gpu, w, x, &xr, w.k, batch_size)?;
        Some(xr)
    } else {
        None
    };
    let x_gemm: &GpuTensor = x_rot_holder.as_ref().unwrap_or(x);

    let result = run_prefill_gemm_inner(gpu, w, x, x_gemm, y, batch_size);
    if let Some(t) = x_rot_holder {
        gpu.free_tensor(t)?;
    }
    result
}

/// Inner body of `run_prefill_gemm`: `x_gemm` is the (possibly FWHT-rotated)
/// GEMM input; `x_raw` is the original activation, used only by the per-token
/// GEMV fallback + verify hook (both rotate internally via ).
fn run_prefill_gemm_inner(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_raw: &GpuTensor,
    x_gemm: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    let w_ref = WeightRef {
        buf: &w.buf,
        dtype: w.gpu_dtype,
        m: w.m,
        k: w.k,
        row_stride: w.k,
        rotation: None,
        awq_scale: None,
    };
    let params = GemmParams { w: &w_ref, x: x_gemm, y, batch_size };
    let family = llama::gemm_family();

    if wmma_prefill_enabled() {
        // WMMA path: use GemmFamily::run which resolves the best kernel
        // for each dtype (WMMA where available, scalar fallback otherwise).
        // The F32->F16 conversion is handled inside the kernel methods.
        //
        // HARD ERROR on failure -- no silent fallthrough. The old code fell
        // through to the scalar key path when  errored (e.g. missing
        // WMMA kernel for this dtype x arch), which silently produced
        // special-token garbage from token 0 on gfx1201 (RDNA4) where the
        // gfx11  WMMA kernels do not exist. Refuse-dont-degrade: the
        // operator opted into WMMA explicitly, so a missing kernel is an
        // error, not a degrade. Unset HIPFIRE_WMMA_PREFILL (or use the
        // scalar HIPFIRE_BATCHED_PREFILL=1 path) on archs without coverage.
        return family.run(&ctx, gpu, &params).map_err(|e| {
            hip_bridge::HipError::new(0, &format!(
                "gemma4 WMMA prefill: GemmFamily::run failed for dtype {:?} on arch {}                  (HIPFIRE_WMMA_PREFILL=1 requires a WMMA kernel for every projection                  dtype -- unset it to use the scalar batched path): {e}",
                w.gpu_dtype, gpu.arch,
            ))
        });
    }

    // Scalar path (default) or fallback for unsupported WMMA dtypes.
    let key = match w.gpu_dtype {
        DType::HFQ4G256 => hipfire_dispatch::types::KernelKey::GemmHfq4G256,
        DType::HFQ4G128 => hipfire_dispatch::types::KernelKey::GemmHfq4G128,
        // Same kernel as HFQ4G256 (layout-identical); input pre-rotated above.
        DType::MQ4G256 => hipfire_dispatch::types::KernelKey::GemmHfq4G256,
        DType::Q8_0 => hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
        // No batched GEMM kernel for this dtype -- fall back to repeated GEMV
        // on the RAW input (weight_gemv applies any needed rotation itself).
        // This matches the old  fallback path.
        _ => {
            let x_tok = gpu.alloc_tensor(&[w.k], DType::F32)?;
            let y_tok = gpu.alloc_tensor(&[w.m], DType::F32)?;
            for b in 0..batch_size {
                gpu.hip.memcpy_dtod_at(&x_tok.buf, 0, &x_raw.buf, b * w.k * 4, w.k * 4)?;
                weight_gemv(gpu, w, &x_tok, &y_tok)?;
                gpu.hip.memcpy_dtod_at(&y.buf, b * w.m * 4, &y_tok.buf, 0, w.m * 4)?;
            }
            gpu.free_tensor(x_tok)?;
            gpu.free_tensor(y_tok)?;
            return Ok(());
        }
    };
    family
        .run_key(key, &ctx, gpu, &params)
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    // Debug parity hook: re-run token 0 through the per-token GEMV path
    // (raw input; weight_gemv rotates internally) and diff against the GEMM.
    if std::env::var("HIPFIRE_GEMMA4_GEMM_VERIFY").ok().as_deref() == Some("1") {
        let x_tok = gpu.alloc_tensor(&[w.k], DType::F32)?;
        let y_tok = gpu.alloc_tensor(&[w.m], DType::F32)?;
        gpu.hip.memcpy_dtod_at(&x_tok.buf, 0, &x_raw.buf, 0, w.k * 4)?;
        weight_gemv(gpu, w, &x_tok, &y_tok)?;
        let yv = gpu.download_f32(&y_tok)?;
        let yg = gpu.download_f32(y)?;
        let mut worst = 0f32; let mut wi = 0usize;
        for i in 0..w.m { let d = (yv[i] - yg[i]).abs(); if d > worst { worst = d; wi = i; } }
        eprintln!("[gemm-verify] dtype={:?} m={} k={} b={} key={:?} worst={:.5} at {} gemv={:.4} gemm={:.4} head_gemv={:?} head_gemm={:?}",
            w.gpu_dtype, w.m, w.k, batch_size, key, worst, wi, yv[wi], yg[wi], &yv[..2], &yg[..2]);
        gpu.free_tensor(x_tok)?; gpu.free_tensor(y_tok)?;
    }
    Ok(())
}

/// Env-gated dump helper for v1-vs-v2 root-cause work.
/// Set HIPFIRE_GEMMA4_DUMP=1 to enable. Prints first 4 floats + sum + nan/inf count.
#[allow(dead_code)]
fn dbg_dump(gpu: &mut Gpu, label: &str, t: &GpuTensor, take: usize) {
    if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() != Some("1") { return; }
    let data = match gpu.download_f32(t) { Ok(d) => d, Err(_) => return };
    let take = take.min(data.len());
    let head: Vec<f32> = data[..take.min(4)].iter().copied().collect();
    let sum: f64 = data[..take].iter().map(|&v| v as f64).sum();
    let nans = data[..take].iter().filter(|&&v| v.is_nan()).count();
    let infs = data[..take].iter().filter(|&&v| v.is_infinite()).count();
    eprintln!("[dump] {label:42} sum={sum:>+14.4e} n={take:>6} head={head:?} nan={nans} inf={infs}");
}

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Sliding-window causal attention (window=1024 on 31B).
    Sliding,
    /// Full causal attention (global).
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeType {
    /// Standard RoPE: all head_dim positions rotate.
    Default,
    /// Proportional RoPE (Gemma 4 full layers): only the first
    /// `partial_rotary_factor × head_dim` positions rotate; rest are NoPE.
    Proportional,
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    // Common
    pub dim: usize,                        // hidden_size, e.g. 5376 on 31B
    pub n_layers: usize,                   // 60 on 31B
    pub vocab_size: usize,                 // 262144 on Gemma 4
    pub norm_eps: f32,                     // 1e-6
    pub bos_token: u32,                    // 2
    pub eos_token: u32,                    // 1
    pub pad_token: u32,                    // 0

    // Attention heads (same count for sliding + full)
    pub n_heads: usize,                    // 32 on 31B

    // Sliding-window attention
    pub sliding_head_dim: usize,           // 256 on 31B
    pub sliding_n_kv_heads: usize,         // 16 on 31B
    pub sliding_rope_theta: f32,           // 10000.0
    pub sliding_window: usize,             // 1024

    // Full attention (global)
    pub full_head_dim: usize,              // 512 on 31B (= global_head_dim)
    pub full_n_kv_heads: usize,            // 4 on 31B
    pub full_rope_theta: f32,              // 1_000_000.0
    pub full_rope_type: RopeType,          // Proportional on 31B
    pub full_partial_rotary_factor: f32,   // 0.25
    pub attention_k_eq_v: bool,            // true on 31B — V = pre-k_norm output

    // FFN (SwiGLU, gelu_pytorch_tanh)
    pub hidden_dim: usize,                 // intermediate_size = 21504 on 31B

    // MoE (26B-A4B). enable_moe_block=true → every layer carries a parallel
    // MoE branch whose output sums with the standard SwiGLU output before
    // the post_feedforward_layernorm. Zero on dense models (31B).
    pub enable_moe_block: bool,            // true on 26B-A4B
    pub moe_intermediate_size: usize,      // 704 on 26B-A4B (per-expert FFN hidden)
    pub num_experts: usize,                // 128 on 26B-A4B
    pub top_k_experts: usize,              // 8 on 26B-A4B (kernel hardcoded to 8)

    // Output
    pub final_logit_softcapping: f32,      // 30.0 — tanh(x/30)*30
    pub tie_word_embeddings: bool,         // true — lm_head aliases embed_tokens
    pub embed_scale: f32,                  // sqrt(dim), applied at embed lookup

    // Per-layer dispatch (len == n_layers)
    pub layer_types: Vec<LayerType>,

    // Vision integration (present even on text-only 31B since config ships it)
    pub has_vision: bool,
    pub image_token_id: u32,               // 258880
    pub boi_token_id: u32,                 // 255999
    pub eoi_token_id: u32,                 // 258882
    pub audio_token_id: u32,               // 258881 (reserved, unused on dense 31B)
    pub video_token_id: u32,               // 258884 (reserved)
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Gemma4Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let bos_token = tc.get("bos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
    let eos_token = tc.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let pad_token = tc.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;

    // Sliding attention params
    let sliding_head_dim = tc.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or(dim / n_heads);
    let sliding_n_kv_heads = tc.get("num_key_value_heads").and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let sliding_window = tc.get("sliding_window").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;

    // Full attention params (may differ from sliding)
    let full_head_dim = tc.get("global_head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or(sliding_head_dim);
    let full_n_kv_heads = tc.get("num_global_key_value_heads").and_then(|v| v.as_u64())
        .unwrap_or(sliding_n_kv_heads as u64) as usize;
    let attention_k_eq_v = tc.get("attention_k_eq_v").and_then(|v| v.as_bool()).unwrap_or(false);

    // rope_parameters is a dict with "sliding_attention" and "full_attention" sub-dicts
    // per the Gemma 4 config schema. Parse both independently.
    let rope_params = tc.get("rope_parameters");
    let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
    let full_rope = rope_params.and_then(|r| r.get("full_attention"));

    let sliding_rope_theta = sliding_rope.and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64()).unwrap_or(10_000.0) as f32;
    let full_rope_theta = full_rope.and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64()).unwrap_or(1_000_000.0) as f32;
    let full_rope_type = match full_rope.and_then(|r| r.get("rope_type")).and_then(|v| v.as_str()) {
        Some("proportional") => RopeType::Proportional,
        _ => RopeType::Default,
    };
    let full_partial_rotary_factor = full_rope.and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let hidden_dim = tc.get("intermediate_size")?.as_u64()? as usize;

    // MoE config (26B-A4B). Absent / false on dense models (31B).
    let enable_moe_block = tc.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false);
    let moe_intermediate_size = tc.get("moe_intermediate_size")
        .and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let top_k_experts = tc.get("top_k_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    let final_logit_softcapping = tc.get("final_logit_softcapping")
        .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let tie_word_embeddings = tc.get("tie_word_embeddings").and_then(|v| v.as_bool())
        .or_else(|| config.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    let embed_scale = (dim as f32).sqrt();

    let layer_types: Vec<LayerType> = tc.get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| match v.as_str().unwrap_or("sliding_attention") {
            "full_attention" => LayerType::Full,
            _ => LayerType::Sliding,
        }).collect())
        .unwrap_or_else(|| vec![LayerType::Sliding; n_layers]);

    // Multimodal token IDs (top-level in config, not under text_config)
    let has_vision = config.get("vision_config").map(|v| !v.is_null()).unwrap_or(false);
    let image_token_id = config.get("image_token_id").and_then(|v| v.as_u64()).unwrap_or(258880) as u32;
    let boi_token_id = config.get("boi_token_id").and_then(|v| v.as_u64()).unwrap_or(255999) as u32;
    let eoi_token_id = config.get("eoi_token_id").and_then(|v| v.as_u64()).unwrap_or(258882) as u32;
    let audio_token_id = config.get("audio_token_id").and_then(|v| v.as_u64()).unwrap_or(258881) as u32;
    let video_token_id = config.get("video_token_id").and_then(|v| v.as_u64()).unwrap_or(258884) as u32;

    Some(Gemma4Config {
        dim, n_layers, vocab_size, norm_eps,
        bos_token, eos_token, pad_token,
        n_heads,
        sliding_head_dim, sliding_n_kv_heads, sliding_rope_theta, sliding_window,
        full_head_dim, full_n_kv_heads, full_rope_theta, full_rope_type,
        full_partial_rotary_factor, attention_k_eq_v,
        hidden_dim,
        enable_moe_block, moe_intermediate_size, num_experts, top_k_experts,
        final_logit_softcapping, tie_word_embeddings, embed_scale,
        layer_types,
        has_vision,
        image_token_id, boi_token_id, eoi_token_id, audio_token_id, video_token_id,
    })
}

// ─── Weights ────────────────────────────────────────────────────────────

/// Per-layer weights for a SLIDING layer (head_dim=256, 16 KV heads, full RoPE).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,           // [dim]
    pub post_attention_layernorm: GpuTensor,  // [dim]
    pub pre_feedforward_layernorm: GpuTensor, // [dim]
    pub post_feedforward_layernorm: GpuTensor,// [dim]
    pub layer_scalar: GpuTensor,              // [1]
    /// Host-side mirror of layer_scalar. Populated at load time so decode can
    /// call `gpu.scale_f32(x, layer_scalar_host)` without a D2H round-trip.
    pub layer_scalar_host: f32,

    // Attention (sliding — head_dim=256)
    pub q_proj: WeightTensor,   // [n_heads * 256, dim]
    pub k_proj: WeightTensor,   // [16 * 256, dim]
    pub v_proj: WeightTensor,   // [16 * 256, dim]
    pub o_proj: WeightTensor,   // [dim, n_heads * 256]
    pub q_norm: GpuTensor,      // [256]
    pub k_norm: GpuTensor,      // [256]

    // MLP (SwiGLU)
    pub gate_proj: WeightTensor, // [hidden_dim, dim]
    pub up_proj: WeightTensor,   // [hidden_dim, dim]
    pub down_proj: WeightTensor, // [dim, hidden_dim]

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-layer weights for a FULL layer (head_dim=512, 4 KV heads, K=V shared).
///
/// Note: no `v_proj` — V is the pre-k_norm output of k_proj, renormed by
/// weight-less `v_norm`. No `v_norm` tensor either (no_scale — the `with_scale=False`
/// RMSNorm applies only the divide, no learned gain). We reuse the existing
/// rmsnorm kernel with a ones-filled `v_norm_ones` buffer (shared across
/// full-attn layers) to preserve the no-scale semantics.
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar: GpuTensor,
    /// Host-side mirror of layer_scalar. See SlidingLayerWeights for rationale.
    pub layer_scalar_host: f32,

    // Attention (full — head_dim=512, K=V)
    pub q_proj: WeightTensor,   // [n_heads * 512, dim]
    pub k_proj: WeightTensor,   // [4 * 512, dim]
    // no v_proj — V = pre-k_norm output of k_proj
    pub o_proj: WeightTensor,   // [dim, n_heads * 512]
    pub q_norm: GpuTensor,      // [512]
    pub k_norm: GpuTensor,      // [512]
    // no v_norm weight — v_norm is no-scale (divide only)

    // MLP (SwiGLU, same shape as sliding)
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-expert FFN weights for a single MoE expert. 128 of these per layer
/// on 26B-A4B; views into the per-layer pool allocation (so `free_gpu`
/// doesn't free these — the pool owns the bytes).
pub struct MoeExpertWeights {
    /// `[2 * moe_intermediate, dim]` — gate + up fused. Rows [0, mi) are
    /// gate; rows [mi, 2*mi) are up. Quantized as MQ4G256 / MG4G256 when
    /// dim is 256-aligned (it is on 26B-A4B: dim=2816).
    pub gate_up_proj: WeightTensor,
    /// `[dim, moe_intermediate]` — projects per-expert FFN hidden back to dim.
    /// On 26B-A4B, mi=704 isn't 256-aligned so this drops to Q8_0 via the
    /// quantizer fallback chain.
    pub down_proj: WeightTensor,
}

/// MoE branch weights for a Gemma 4 MoE layer (26B-A4B). Present on every
/// layer when `config.enable_moe_block` is set. The branch adds a parallel
/// FFN computation alongside the standard SwiGLU; outputs are summed via
/// sandwich norms then a final post_feedforward_layernorm closes the layer.
pub struct MoeLayerExtras {
    /// `[n_experts, dim]` — projects router input to expert logits.
    pub router_proj: WeightTensor,
    /// `[dim]` — multiplicative scale on router input (`router.scale` in HF).
    pub router_scale: GpuTensor,
    /// `[n_experts]` — per-expert post-`down_proj` scale (`router.per_expert_scale`).
    pub per_expert_scale: GpuTensor,
    /// Host mirror of `per_expert_scale` for fast top-K weight composition.
    pub per_expert_scale_host: Vec<f32>,
    /// `[dim]` — RMSNorm applied to attn_out before the MoE branch.
    pub pre_feedforward_layernorm_2: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_mlp (standard SwiGLU output) BEFORE summing.
    pub post_feedforward_layernorm_1: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_moe (MoE branch output) BEFORE summing.
    pub post_feedforward_layernorm_2: GpuTensor,
    /// Pool allocation for all gate_up tensors. Per-expert WeightTensors
    /// alias into this; `free_gpu` frees the pool, not each WeightTensor.
    pub experts_gate_up_pool: GpuTensor,
    /// Pool allocation for all down tensors. Same aliasing.
    pub experts_down_pool: GpuTensor,
    /// Per-expert views into the pools above.
    pub experts: Vec<MoeExpertWeights>,
    /// `[n_exp]` u64 device pointers — one per expert's gate_up weight
    /// base. Built once at load by reading each expert's pool sub-view
    /// pointer. The indexed MoE kernels read
    /// `expert_ptrs[topk_indices[krank]]` to locate the active expert
    /// weight WITHOUT a D2H sync.
    pub experts_gate_up_ptrs: GpuTensor,
    /// `[n_exp]` u64 device pointers for each expert's down weight base.
    pub experts_down_ptrs: GpuTensor,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

pub struct Gemma4Weights {
    /// Token embedding [vocab_size, dim], Q8F16 to keep the 262144×5376 table manageable.
    /// Aliased as lm_head when tie_word_embeddings is true.
    pub embed_tokens: GpuTensor,
    /// Embed/LM-head format tag for dispatch.
    pub embd_format: EmbeddingFormat,
    /// LM-head projection (shares bytes with embed_tokens when tied).
    pub lm_head: WeightTensor,
    /// Model-final RMSNorm scale [dim].
    pub final_norm: GpuTensor,
    /// Per-layer weights indexed by layer ordinal.
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        // lm_head may alias embed_tokens — skip if so (we rely on the loader
        // to set `lm_head.buf` to an alias and not a separate allocation).
        for l in self.layers {
            match l {
                LayerWeights::Sliding(s) => {
                    for t in [s.input_layernorm, s.post_attention_layernorm,
                              s.pre_feedforward_layernorm, s.post_feedforward_layernorm,
                              s.layer_scalar, s.q_norm, s.k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                    for wt in [s.q_proj.buf, s.k_proj.buf, s.v_proj.buf, s.o_proj.buf,
                               s.gate_proj.buf, s.up_proj.buf, s.down_proj.buf] {
                        let _ = gpu.free_tensor(wt);
                    }
                    if let Some(moe) = s.moe { Self::free_moe(gpu, moe); }
                }
                LayerWeights::Full(f) => {
                    for t in [f.input_layernorm, f.post_attention_layernorm,
                              f.pre_feedforward_layernorm, f.post_feedforward_layernorm,
                              f.layer_scalar, f.q_norm, f.k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                    for wt in [f.q_proj.buf, f.k_proj.buf, f.o_proj.buf,
                               f.gate_proj.buf, f.up_proj.buf, f.down_proj.buf] {
                        let _ = gpu.free_tensor(wt);
                    }
                    if let Some(moe) = f.moe { Self::free_moe(gpu, moe); }
                }
            }
        }
    }

    fn free_moe(gpu: &mut Gpu, moe: MoeLayerExtras) {
        let _ = gpu.free_tensor(moe.router_proj.buf);
        let _ = gpu.free_tensor(moe.router_scale);
        let _ = gpu.free_tensor(moe.per_expert_scale);
        let _ = gpu.free_tensor(moe.pre_feedforward_layernorm_2);
        let _ = gpu.free_tensor(moe.post_feedforward_layernorm_1);
        let _ = gpu.free_tensor(moe.post_feedforward_layernorm_2);
        // per-expert WeightTensors alias into the pools — skip freeing them.
        // free the two pool allocations.
        let _ = gpu.free_tensor(moe.experts_gate_up_pool);
        let _ = gpu.free_tensor(moe.experts_down_pool);
        let _ = gpu.free_tensor(moe.experts_gate_up_ptrs);
        let _ = gpu.free_tensor(moe.experts_down_ptrs);
    }
}

// ─── Loading helpers ───────────────────────────────────────────────────

/// Decode a shape-[n] F16 or F32 tensor from HFQ into an F32 host Vec.
fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> HipResult<Vec<f32>> {
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("tensor not found: {name}"))
    })?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if n != expected_n {
        return Err(hip_bridge::HipError::new(
            0, &format!("shape mismatch for {name}: expected {expected_n}, got {n}"),
        ));
    }
    if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") && data.len() <= 4 && name.contains("layer_scalar") {
        eprintln!("[gemma4] load_f32_vec({name}): qt={}, shape={:?}, raw_bytes={:02x?}",
            info.quant_type, info.shape, &data[..data.len().min(4)]);
    }
    let f32_data = match info.quant_type {
        1 => data.chunks_exact(2)
                 .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                 .collect(),
        2 => data.chunks_exact(4)
                 .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                 .collect(),
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("expected F16/F32 for {name}, got qt={qt}"),
        )),
    };
    Ok(f32_data)
}

/// Load a Gemma 4 RMSNorm weight — `x * weight` form, NO +1 shift.
///
/// Distinct from qwen35::load_norm_weight which shifts by +1 for HF Gemma
/// 2/3-style `x * (1 + weight)`. Gemma 4 uses plain `x * weight` with weights
/// initialized to 1.0 (see modeling_gemma4.py::Gemma4RMSNorm line 157).
fn load_gemma4_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize)
    -> HipResult<GpuTensor>
{
    let f32_data = load_f32_vec(hfq, name, dim)?;
    gpu.upload_f32(&f32_data, &[dim])
}

/// Load a 256-element head-dim Q/K RMSNorm weight. Same semantics as
/// `load_gemma4_norm` but scoped to the attention head_dim (256 on sliding,
/// 512 on full).
fn load_gemma4_head_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, head_dim: usize)
    -> HipResult<GpuTensor>
{
    load_gemma4_norm(hfq, gpu, name, head_dim)
}

/// Load the per-layer `layer_scalar` — shape-[1] BF16/F16 tensor — returning
/// both a GPU-resident [1]-tensor (for potential batched use) and its host-side
/// f32 value (used by the decode path to call `scale_f32(x, cpu_scalar)`).
fn load_layer_scalar(hfq: &HfqFile, gpu: &mut Gpu, name: &str)
    -> HipResult<(GpuTensor, f32)>
{
    let data = load_f32_vec(hfq, name, 1)?;
    let host_val = data[0];
    let gpu_tensor = gpu.upload_f32(&data, &[1])?;
    Ok((gpu_tensor, host_val))
}

/// Load a quantized projection weight. Mirrors qwen35::load_weight_tensor_raw
/// but uses the Gemma 4 tensor-name convention (`model.language_model.<name>`).
fn load_gemma4_weight(hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize)
    -> HipResult<WeightTensor>
{
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("tensor not found: {name}"))
    })?;
    let dtype = match info.quant_type {
        1 => {
            // F16 → upload as f32
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            return Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0, awq_scale: None, paro: None });
        }
        2 => {
            // F32 raw (oracle / --format f32 passthrough .hfq) — upload as-is.
            let buf = gpu.upload_raw(data, &[m, k])?;
            return Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0, awq_scale: None, paro: None });
        }
        3  => DType::Q8_0,
        4  => DType::Q4K,
        6  => DType::HFQ4G256,
        7  => DType::HFQ4G128,
        8  => DType::HFQ6G256,
        9  => DType::HFQ2G256,
        10 => DType::HFQ2G128,
        11 => DType::HFQ3G256,
        12 => DType::HFQ3G128,
        13 => DType::MQ4G256,
        14 => DType::MQ8G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        // MG4-G256 — Magnum-Gemma 4-bit. Same binary layout as MQ4G256 (136 B/group),
        // differs only in calibration policy at quant time. Alias to MQ4G256 so the
        // existing GEMV path handles it without a kernel change. ID was 19 on
        // origin/gemma4 pre-rebase; reassigned to 30 because master shipped
        // MQ2G256Lloyd at 19.
        30 => DType::MQ4G256,
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("unsupported quant_type {qt} for {name}"),
        )),
    };
    let buf = gpu.upload_raw(data, &[data.len()])?;
    let awq_scale = if dtype.supports_awq_sidecar() {
        load_awq_scale(hfq, gpu, name, k)
    } else {
        None
    };
    Ok(WeightTensor { buf, gpu_dtype: dtype, m, k, row_stride: 0, awq_scale, paro: None })
}

/// Load the MoE branch weights for a single Gemma 4 MoE layer (26B-A4B).
/// Builds 128 expert WeightTensors aliased into per-layer gate_up / down
/// pool allocations. Pooling avoids the small-allocation HIP fragmentation
/// that OOM'd at layer 25 on the pre-pool path (origin/gemma4 commit log).
fn load_moe_layer_extras(hfq: &HfqFile, gpu: &mut Gpu, p: &str, config: &Gemma4Config)
    -> HipResult<MoeLayerExtras>
{
    let n_exp = config.num_experts;
    let dim = config.dim;
    let mi = config.moe_intermediate_size;

    let router_proj = load_gemma4_weight(hfq, gpu,
        &format!("{p}.router.proj.weight"), n_exp, dim)?;
    // NOTE: `router.scale` and `router.per_expert_scale` ship WITHOUT the
    // `.weight` suffix in HF's 26B-A4B safetensors (so `should_quantize`
    // returns false → stored as F16). Loader uses bare paths.
    let router_scale = load_gemma4_norm(hfq, gpu, &format!("{p}.router.scale"), dim)?;
    let per_expert_scale_host = load_f32_vec(hfq,
        &format!("{p}.router.per_expert_scale"), n_exp)?;
    let per_expert_scale = {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                per_expert_scale_host.as_ptr() as *const u8,
                per_expert_scale_host.len() * 4,
            )
        };
        gpu.upload_raw(bytes, &[n_exp])?
    };
    let pre_feedforward_layernorm_2 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.pre_feedforward_layernorm_2.weight"), dim)?;
    let post_feedforward_layernorm_1 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.post_feedforward_layernorm_1.weight"), dim)?;
    let post_feedforward_layernorm_2 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.post_feedforward_layernorm_2.weight"), dim)?;

    // Pool all `n_experts` weights of one kind into a single GPU allocation.
    // 128 experts × 2 kinds × 30 layers = 7680 separate hipMalloc on the
    // unpooled path fragmented the HIP heap and OOM'd at layer ~25 even
    // when total memory fit. Pool collapses that to 60 allocs.
    let load_pool = |gpu: &mut Gpu, base: &str|
        -> HipResult<(GpuTensor, DType, usize)>
    {
        // First pass: read first expert to learn quant_type + bytes-per-expert.
        let first_name = format!("{p}.experts.0.{base}.weight");
        let (first_info, first_data) = hfq.tensor_data(&first_name).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {first_name}"))
        })?;
        let bytes_per_expert = first_data.len();
        let dtype = match first_info.quant_type {
            3 => DType::Q8_0, 4 => DType::Q4K,
            6 => DType::HFQ4G256, 7 => DType::HFQ4G128, 8 => DType::HFQ6G256,
            9 => DType::HFQ2G256, 10 => DType::HFQ2G128,
            11 => DType::HFQ3G256, 12 => DType::HFQ3G128,
            // MQ4G256 (13) and MG4G256 (30) share dispatch.
            13 | 30 => DType::MQ4G256,
            14 => DType::MQ8G256, 15 => DType::MQ6G256,
            17 => DType::MQ3G256, 18 => DType::MQ2G256,
            qt => return Err(hip_bridge::HipError::new(
                0, &format!("unsupported MoE expert quant_type {qt} for {first_name}"),
            )),
        };
        // Concat all experts' bytes into one CPU buffer, upload once.
        let mut concat = Vec::with_capacity(bytes_per_expert * n_exp);
        concat.extend_from_slice(first_data);
        for x in 1..n_exp {
            let name = format!("{p}.experts.{x}.{base}.weight");
            let (info, data) = hfq.tensor_data(&name).ok_or_else(|| {
                hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {name}"))
            })?;
            if data.len() != bytes_per_expert {
                return Err(hip_bridge::HipError::new(
                    0, &format!("MoE expert {name} byte size mismatch ({} vs {bytes_per_expert})",
                        data.len()),
                ));
            }
            if info.quant_type != first_info.quant_type {
                return Err(hip_bridge::HipError::new(
                    0, &format!("MoE expert {name} quant_type mismatch ({} vs {})",
                        info.quant_type, first_info.quant_type),
                ));
            }
            concat.extend_from_slice(data);
        }
        let pool = gpu.upload_raw(&concat, &[concat.len()])?;
        Ok((pool, dtype, bytes_per_expert))
    };

    let (gate_up_pool, gate_up_dtype, gate_up_bytes) = load_pool(gpu, "gate_up_proj")?;
    let (down_pool, down_dtype, down_bytes) = load_pool(gpu, "down_proj")?;

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gu_view = gate_up_pool.sub_offset(x * gate_up_bytes, gate_up_bytes);
        let dn_view = down_pool.sub_offset(x * down_bytes, down_bytes);
        experts.push(MoeExpertWeights {
            gate_up_proj: WeightTensor {
                buf: gu_view, gpu_dtype: gate_up_dtype,
                m: 2 * mi, k: dim, row_stride: 0, awq_scale: None, paro: None,
            },
            down_proj: WeightTensor {
                buf: dn_view, gpu_dtype: down_dtype,
                m: dim, k: mi, row_stride: 0, awq_scale: None, paro: None,
            },
        });
    }

    // Build [n_exp] device tensors of u64 weight-base pointers — one for
    // each pool. The indexed MoE kernels read these as
    // `expert_ptrs[topk_indices[krank]]`, eliminating the per-token D2H
    // sync of the legacy CPU per-expert loop. Pointers are stable for the
    // model's lifetime (pool allocations don't move).
    let gate_up_ptr_u64: Vec<u64> = experts.iter()
        .map(|e| e.gate_up_proj.buf.buf.as_ptr() as u64)
        .collect();
    let down_ptr_u64: Vec<u64> = experts.iter()
        .map(|e| e.down_proj.buf.buf.as_ptr() as u64)
        .collect();
    let gate_up_bytes: Vec<u8> = gate_up_ptr_u64.iter()
        .flat_map(|p| p.to_ne_bytes())
        .collect();
    let down_bytes: Vec<u8> = down_ptr_u64.iter()
        .flat_map(|p| p.to_ne_bytes())
        .collect();
    // Each u64 = 8 bytes = 2 f32 slots. The tensor sees [n_exp * 2]
    // f32 entries; the kernel casts the backing buffer to u64* itself.
    let experts_gate_up_ptrs = gpu.upload_raw(&gate_up_bytes, &[n_exp * 2])?;
    let experts_down_ptrs = gpu.upload_raw(&down_bytes, &[n_exp * 2])?;

    Ok(MoeLayerExtras {
        router_proj,
        router_scale,
        per_expert_scale,
        per_expert_scale_host,
        pre_feedforward_layernorm_2,
        post_feedforward_layernorm_1,
        post_feedforward_layernorm_2,
        experts_gate_up_pool: gate_up_pool,
        experts_down_pool: down_pool,
        experts,
        experts_gate_up_ptrs,
        experts_down_ptrs,
    })
}

/// Load Gemma 4 text model weights from an HFQ file.
///
/// Design notes:
///   - `lm_head` aliases the `embed_tokens` GPU bytes (tied weights). We upload
///     the embed data once and create a second WeightTensor whose DeviceBuffer
///     points at the same allocation via `buf.alias()`. `Gemma4Weights::free_gpu`
///     skips freeing the LM head to avoid a double-free.
///   - Vision tensors are skipped here — Phase 7 `gemma4_vision::load_weights`
///     picks those up from the same HFQ file in a separate pass.
///   - The `v_norm_ones_full` ones-filled scratch buffer is populated here so
///     the forward pass never has to manage one-time init state.
pub fn load_weights(hfq: &mut HfqFile, config: &Gemma4Config, gpu: &mut Gpu)
    -> HipResult<Gemma4Weights>
{
    eprintln!("gemma4: loading embed_tokens...");
    let embed_name = "model.language_model.embed_tokens.weight";
    let (embed_info, embed_data) = hfq.tensor_data(embed_name).ok_or_else(|| {
        hip_bridge::HipError::new(0, "embed_tokens not found in HFQ")
    })?;
    let (embed_tokens, embd_format) = match embed_info.quant_type {
        3 => {
            eprintln!("  (Q8_0 / Q8F16, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::Q8_0)
        }
        6 => {
            eprintln!("  (HFQ4-G256, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::HFQ4G256)
        }
        7 => {
            eprintln!("  (HFQ4-G128, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::HFQ4G128)
        }
        1 => {
            eprintln!("  (F16 → F32)");
            let f32_data: Vec<f32> = embed_data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            (gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
        }
        2 => {
            // F32 raw (oracle / --format f32 passthrough .hfq).
            eprintln!("  (F32 raw, {} MB)", embed_data.len() / 1_000_000);
            let f32_data: Vec<f32> = embed_data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
        }
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("unsupported embed quant_type {qt}"),
        )),
    };

    // Tied LM head: WeightTensor whose buffer aliases the embed allocation.
    // free_gpu skips freeing this — embed_tokens owns the bytes.
    let lm_head = {
        let alias_buf = unsafe { embed_tokens.buf.alias() };
        let dtype = match embd_format {
            EmbeddingFormat::Q8_0    => DType::Q8_0,
            EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
            EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
            EmbeddingFormat::F32     => DType::F32,
            EmbeddingFormat::Q4K     => DType::Q4K,
        };
        let alias_tensor = GpuTensor {
            buf: alias_buf,
            shape: embed_tokens.shape.clone(),
            dtype,
        };
        WeightTensor { buf: alias_tensor, gpu_dtype: dtype, m: config.vocab_size, k: config.dim, row_stride: 0, awq_scale: None, paro: None }
    };

    eprintln!("gemma4: loading final norm...");
    let final_norm = load_gemma4_norm(hfq, gpu, "model.language_model.norm.weight", config.dim)?;

    eprintln!("gemma4: loading {} layers...", config.n_layers);
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let p = format!("model.language_model.layers.{i}");
        match config.layer_types[i] {
            LayerType::Sliding => {
                let hd = config.sliding_head_dim;
                let kv_dim = config.sliding_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                if i == 0 { eprintln!("[gemma4] L0 sliding layer_scalar = {layer_scalar_host}"); }
                let moe = if config.enable_moe_block {
                    Some(load_moe_layer_extras(hfq, gpu, &p, config)?)
                } else { None };
                layers.push(LayerWeights::Sliding(SlidingLayerWeights {
                    input_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.input_layernorm.weight"), config.dim)?,
                    post_attention_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_attention_layernorm.weight"), config.dim)?,
                    pre_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.pre_feedforward_layernorm.weight"), config.dim)?,
                    post_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_feedforward_layernorm.weight"), config.dim)?,
                    layer_scalar,
                    layer_scalar_host,
                    q_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.q_proj.weight"), q_dim, config.dim)?,
                    k_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    v_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?,
                    o_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.o_proj.weight"), config.dim, q_dim)?,
                    q_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.q_norm.weight"), hd)?,
                    k_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.k_norm.weight"), hd)?,
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                    moe,
                }));
            }
            LayerType::Full => {
                let hd = config.full_head_dim;
                let kv_dim = config.full_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                if i <= 6 { eprintln!("[gemma4] L{i} full layer_scalar = {layer_scalar_host}"); }
                let moe = if config.enable_moe_block {
                    Some(load_moe_layer_extras(hfq, gpu, &p, config)?)
                } else { None };
                layers.push(LayerWeights::Full(FullLayerWeights {
                    input_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.input_layernorm.weight"), config.dim)?,
                    post_attention_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_attention_layernorm.weight"), config.dim)?,
                    pre_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.pre_feedforward_layernorm.weight"), config.dim)?,
                    post_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_feedforward_layernorm.weight"), config.dim)?,
                    layer_scalar,
                    layer_scalar_host,
                    q_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.q_proj.weight"), q_dim, config.dim)?,
                    k_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    // no v_proj on full layers — V reuses k_proj's pre-norm output.
                    o_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.o_proj.weight"), config.dim, q_dim)?,
                    q_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.q_norm.weight"), hd)?,
                    k_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.k_norm.weight"), hd)?,
                    // no v_norm weight — v_norm is no-scale (ones buffer passed at decode time).
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                    moe,
                }));
            }
        }
    }
    eprintln!("gemma4: loaded all {} layers", config.n_layers);

    Ok(Gemma4Weights {
        embed_tokens,
        embd_format,
        lm_head,
        final_norm,
        layers,
    })
}

/// One-time init for the scratch buffers that must hold a constant value
/// across forward passes (notably the ones-filled `v_norm_ones_full`).
/// Call once after `Gemma4Scratch::new` before the first forward pass.
pub fn init_scratch_constants(gpu: &mut Gpu, scratch: &Gemma4Scratch, full_head_dim: usize)
    -> HipResult<()>
{
    let ones: Vec<f32> = vec![1.0; full_head_dim];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4)
    };
    gpu.hip.memcpy_htod(&scratch.v_norm_ones_full.buf, bytes)?;
    Ok(())
}

// ─── Scratch ────────────────────────────────────────────────────────────

use hip_bridge::DeviceBuffer;

/// Per-decode scratch, sized once at model-load time against the MAX of
/// sliding and full attention dimensions so a single buffer works across
/// layer types. 31B target shapes: sliding Q=[32*256]=8192, full Q=[32*512]=16384
/// → size Q at 16384. Sliding KV=[16*256]=4096, full KV=[4*512]=2048 → size at 4096.
pub struct Gemma4Scratch {
    pub x: GpuTensor,           // [dim] — hidden state
    pub residual: GpuTensor,    // [dim] — saved for sandwich residual
    pub tmp: GpuTensor,         // [dim] — norm output scratch

    /// Position buffer (single i32 on device, updated per decode step).
    pub pos_buf: DeviceBuffer,

    // Attention scratch — sized for max(sliding, full)
    pub q: GpuTensor,           // [max(n_heads*head_dim_sliding, n_heads*head_dim_full)]
    pub k: GpuTensor,           // [max(n_kv_heads*head_dim for each layer type)]
    pub v: GpuTensor,           // [same as k]
    pub attn_out: GpuTensor,    // [same as q]

    // MLP scratch
    pub gate_ffn: GpuTensor,    // [hidden_dim]
    pub up_ffn: GpuTensor,      // [hidden_dim]
    pub ffn_hidden: GpuTensor,  // [hidden_dim]
    pub ffn_out: GpuTensor,     // [dim]

    // Output
    pub logits: GpuTensor,      // [vocab_size]
    pub sample_buf: GpuTensor,  // [2] — (token_id, new_rng_state) for GPU sampling
    pub repeat_buf: GpuTensor,  // [1024] — rolling window for repeat penalty

    // Flash attention tile partials. Sized for the LARGER of the two
    // cache shapes: full-attn uses head_dim=512, max_tiles=max_seq/128.
    // Sliding uses head_dim=256, max_tiles=sliding_window/128 (much smaller).
    pub flash_partials: GpuTensor,

    // No-scale v_norm ones buffer (full-attn layers compute v_norm without
    // a learned weight — we pass this ones-filled tensor to the existing
    // rmsnorm kernel to get no-scale RMS semantics).
    pub v_norm_ones_full: GpuTensor, // [full_head_dim]

    // ── MoE scratch (26B-A4B only). Zero-sized on dense models. ─────────
    pub moe_cur_mlp: GpuTensor,        // [dim] — rmsnorm(ffn_out, post_norm_1)
    pub moe_pre2: GpuTensor,           // [dim] — rmsnorm(attn_out, pre_norm_2)
    pub moe_router_in: GpuTensor,      // [dim] — router input (post-rmsnorm + scale)
    pub moe_router_logits: GpuTensor,  // [n_experts]
    pub moe_topk_indices: GpuTensor,   // [top_k_experts] — i32 packed in f32 slots
    pub moe_topk_weights: GpuTensor,   // [top_k_experts]
    pub moe_cur_moe: GpuTensor,        // [dim] — accumulator across top-K experts
    pub moe_expert_gate_up: GpuTensor, // [2 * moe_intermediate_size]
    pub moe_expert_hidden: GpuTensor,  // [moe_intermediate_size] — gelu(gate) * up
    pub moe_expert_out: GpuTensor,     // [dim] — single expert's down_proj output

    // ── Indexed MoE batched scratch (k_top=8 hardcoded by kernel). ──────
    // These back the device-side fused path that replaces the 8-iteration
    // per-expert CPU loop. Only allocated when the MoE branch is enabled.
    /// `[dim]` — moe_pre2 after one FWHT pass (MQ4 gate_up expects pre-rotated x).
    pub moe_pre2_rot: GpuTensor,
    /// `[k_top × 2 × mi]` — fused gate+up output, one row per top-K rank.
    pub moe_expert_gate_batch: GpuTensor, // [k_top × mi]
    pub moe_expert_up_batch:   GpuTensor, // [k_top × mi]

    // ── Prefill-batch (N tokens at a time) scratch ─────────────────────
    // Sized for max_prefill_batch tokens. Used by forward_prefill_chunk to
    // amortize MoE launch overhead across N tokens via the batched-indexed
    // kernels. None on models with no MoE (n_experts==0).
    pub max_prefill_batch: usize,
    /// `[N × dim]` — per-token attention output after o_proj (before residual).
    pub pb_attn_out: GpuTensor,
    /// `[N × dim]` — per-token dense FFN output (gemma4 26B-A4B-it computes
    /// this in parallel to MoE; both feed the sandwich norm).
    pub pb_ffn_out: GpuTensor,
    /// `[N × dim]` — pre-norm-2(attn_out) for each token.
    pub pb_moe_pre2: GpuTensor,
    /// `[N × dim]` — FWHT-rotated pre2 (MQ4 gate_up expects rotated x).
    pub pb_moe_pre2_rot: GpuTensor,
    /// `[N × dim]` — router input (rmsnorm(attn_out, router_scale) / sqrt(dim)).
    pub pb_moe_router_in: GpuTensor,
    /// `[N × n_experts]` — router logits batched.
    pub pb_moe_router_logits: GpuTensor,
    /// `[N × k_top]` i32 — top-K expert indices per token.
    pub pb_moe_topk_indices: GpuTensor,
    /// `[N × k_top]` — renormalized top-K weights per token.
    pub pb_moe_topk_weights: GpuTensor,
    /// `[n_exp + 1]` i32 — prefix-sum bucket offsets for routing-bucketed MoE GEMM.
    pub pb_moe_expert_offsets: GpuTensor,
    /// `[N × k_top]` i32 — packed (token_idx × k_top + krank), sorted by expert.
    pub pb_moe_expert_token_list: GpuTensor,
    /// `[N × k_top × mi]` — gate output across tokens × experts.
    pub pb_moe_gate_batch: GpuTensor,
    /// `[N × k_top × mi]` — up output across tokens × experts.
    pub pb_moe_up_batch: GpuTensor,
    /// `[N × k_top × mi]` — gelu_tanh(gate) * up.
    pub pb_moe_hidden_batch: GpuTensor,
    /// `[N × dim]` — accumulator for MoE branch output per token.
    pub pb_moe_cur_moe: GpuTensor,
    /// `[N × dim]` — post_norm_1(ffn_out) per token (dense branch).
    pub pb_moe_cur_mlp: GpuTensor,
    /// `[N × dim]` — residual stream per token across the prefill batch.
    pub pb_residual: GpuTensor,
    /// `[N × dim]` — post-rmsnorm input for batched projections.
    pub pb_tmp: GpuTensor,
    /// `[N × max_q_dim]` — batched Q projection output (sized for full layer = n_heads * full_head_dim).
    pub pb_q: GpuTensor,
    /// `[N × max_q_dim]` — batched attention output, kept separate from `pb_q`
    /// so batched flash attention never reads and writes the same buffer.
    pub pb_attn_q: GpuTensor,
    /// `[N × n_heads × max_tiles_full × (2+full_head_dim)]` — batch-sized flash
    /// partials for batched masked prefill attention. The single-query
    /// `flash_partials` overflows with batch_size>1.
    pub pb_flash_partials: GpuTensor,
    /// `[N × max_kv_dim]` — batched K projection output.
    pub pb_k: GpuTensor,
    /// `[N × max_kv_dim]` — batched V projection output.
    pub pb_v: GpuTensor,
    /// `[N × hidden_dim]` — batched gate proj output.
    pub pb_gate: GpuTensor,
    /// `[N × hidden_dim]` — batched up proj output.
    pub pb_up: GpuTensor,
    /// `[N × hidden_dim]` — batched gelu(gate)*up.
    pub pb_ffn_hidden: GpuTensor,
    /// `[N]` — i32 position buffer for batched RoPE.
    pub pb_positions: GpuTensor,
    /// `[k_top × mi]` — gelu_tanh(gate)*up batched over k_top experts.
    pub moe_expert_hidden_batch: GpuTensor,
}

impl Gemma4Scratch {
    pub fn new(gpu: &mut Gpu, config: &Gemma4Config, _max_prefill: usize) -> HipResult<Self> {
        let dim = config.dim;
        let q_dim = (config.n_heads * config.sliding_head_dim).max(config.n_heads * config.full_head_dim);
        let kv_dim = (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim);

        let x = gpu.zeros(&[dim], DType::F32)?;
        let residual = gpu.zeros(&[dim], DType::F32)?;
        let tmp = gpu.zeros(&[dim], DType::F32)?;

        let pos_buf = gpu.hip.malloc(4)?;

        let q = gpu.zeros(&[q_dim], DType::F32)?;
        let k = gpu.zeros(&[kv_dim], DType::F32)?;
        let v = gpu.zeros(&[kv_dim], DType::F32)?;
        let attn_out = gpu.zeros(&[q_dim], DType::F32)?;

        let gate_ffn = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let up_ffn = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let ffn_hidden = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let ffn_out = gpu.zeros(&[dim], DType::F32)?;

        let logits = gpu.zeros(&[config.vocab_size], DType::F32)?;
        let sample_buf = gpu.zeros(&[2], DType::F32)?;
        let repeat_buf = gpu.zeros(&[1024], DType::F32)?;

        // Flash partials sizing. Per-head × max_tiles × (2 + head_dim) floats.
        // Sized for FULL attn (head_dim=512 stride 514, vs sliding 256 stride 258);
        // sliding-layer dispatches use part of the buffer, full-layer dispatches
        // use all of it.
        //
        // Default 32k. The branch name "gemma4-128k-ring-buffer" describes the
        // sliding-window code path (sliding KV is ring-buffered at sliding_window
        // = 1024 slots regardless of context length). The FULL-attention layers
        // (5 of 30 in 26B-A4B-it) still allocate `max_kv_seq` slots — those
        // layers are NOT ring-buffered. At 26B-A4B-it asym3 sizes the full KV
        // budget for 128k is ~970 MB (5 layers × 2 KV heads × 131072 tokens ×
        // 740 B/head), which fits comfortably on a 17 GB card alongside the
        // 14.8 GB model weights. Users who want the full 128k context set
        // `HIPFIRE_KV_SEQ=131072` at daemon launch. Default stays at 32k to
        // match the cross-arch baseline.
        const FALLBACK_KV_SEQ: usize = 32768;
        const TILE_SIZE: usize = 128;
        let max_kv_seq: usize = std::env::var("HIPFIRE_KV_SEQ")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 128 && n <= 524_288)
            .unwrap_or(FALLBACK_KV_SEQ);
        let max_tiles_full = (max_kv_seq + TILE_SIZE - 1) / TILE_SIZE;
        let flash_partials_sz = config.n_heads * max_tiles_full * (2 + config.full_head_dim);
        let flash_partials = gpu.zeros(&[flash_partials_sz], DType::F32)?;

        // (Note 2026-05-19): removed the precomputed sliding/full cos+sin
        // tables that were allocated here but never read by any kernel.
        // `rope_f32` and `rope_partial_halved_f32` compute cos/sin inline
        // from `pos_buf[0]` + `freq_base`; the lookup-table path was wired
        // but never finished. The tables ate 4 * max_kv_seq * head_dim
        // floats — 768 MB at max_kv_seq=131072 — which pushed the full KV
        // cache (~970 MB at 128k) out of VRAM into GTT (PCIe-paged system
        // RAM), causing attention reads to take a slow path.

        // v_norm ones — populated on first use in the forward pass.
        // Allocated up to the LARGER of sliding_head_dim and full_head_dim
        // since both sliding and full apply no-scale v_norm (the post-rebase
        // fix added v_norm to sliding_layer_decode; sliding head_dim=256,
        // full head_dim=512 → max=512 covers both).
        let v_norm_max = config.sliding_head_dim.max(config.full_head_dim);
        let v_norm_ones_full = gpu.zeros(&[v_norm_max], DType::F32)?;

        // MoE scratch. Allocated unconditionally because the buffers are tiny
        // relative to the model; zero-sized on dense models would just complicate
        // the dispatch path. Sized for 26B-A4B: n_experts=128, top_k=8, mi=704.
        let n_exp = config.num_experts.max(1);
        let mi = config.moe_intermediate_size.max(1);
        let k_top = config.top_k_experts.max(1);
        let moe_cur_mlp = gpu.zeros(&[dim], DType::F32)?;
        let moe_pre2 = gpu.zeros(&[dim], DType::F32)?;
        let moe_router_in = gpu.zeros(&[dim], DType::F32)?;
        let moe_router_logits = gpu.zeros(&[n_exp], DType::F32)?;
        let moe_topk_indices = gpu.zeros(&[k_top], DType::F32)?;
        let moe_topk_weights = gpu.zeros(&[k_top], DType::F32)?;
        let moe_cur_moe = gpu.zeros(&[dim], DType::F32)?;
        let moe_expert_gate_up = gpu.zeros(&[2 * mi], DType::F32)?;
        let moe_expert_hidden = gpu.zeros(&[mi], DType::F32)?;
        let moe_expert_out = gpu.zeros(&[dim], DType::F32)?;

        // Indexed-MoE scratch (k_top fixed at 8 by the kernel).
        let moe_pre2_rot = gpu.zeros(&[dim], DType::F32)?;
        let moe_expert_gate_batch = gpu.zeros(&[k_top * mi], DType::F32)?;
        let moe_expert_up_batch   = gpu.zeros(&[k_top * mi], DType::F32)?;
        let moe_expert_hidden_batch = gpu.zeros(&[k_top * mi], DType::F32)?;

        // Prefill-batch scratch (N tokens at once). Larger batches expose
        // more concurrent GPU work — total batch scratch ≈ N*0.16 MB.
        const MAX_PREFILL_BATCH: usize = 128;
        let pb_attn_out          = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_ffn_out           = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_moe_pre2          = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_moe_pre2_rot      = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_moe_router_in     = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_moe_router_logits = gpu.zeros(&[MAX_PREFILL_BATCH, n_exp], DType::F32)?;
        let pb_moe_topk_indices  = gpu.zeros(&[MAX_PREFILL_BATCH, k_top], DType::F32)?;
        let pb_moe_topk_weights  = gpu.zeros(&[MAX_PREFILL_BATCH, k_top], DType::F32)?;
        // Routing-bucket scratch (Phase B). expert_offsets has n_exp+1 entries.
        // expert_token_list has one entry per (token, krank) pair = N × k_top.
        let pb_moe_expert_offsets    = gpu.zeros(&[n_exp + 1], DType::F32)?;  // i32-typed slots
        let pb_moe_expert_token_list = gpu.zeros(&[MAX_PREFILL_BATCH, k_top], DType::F32)?;
        let pb_moe_gate_batch    = gpu.zeros(&[MAX_PREFILL_BATCH, k_top * mi], DType::F32)?;
        let pb_moe_up_batch      = gpu.zeros(&[MAX_PREFILL_BATCH, k_top * mi], DType::F32)?;
        let pb_moe_hidden_batch  = gpu.zeros(&[MAX_PREFILL_BATCH, k_top * mi], DType::F32)?;
        let pb_moe_cur_moe       = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_moe_cur_mlp       = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_residual          = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        let pb_tmp               = gpu.zeros(&[MAX_PREFILL_BATCH, dim], DType::F32)?;
        // Sized for max across sliding/full per-token vector dims.
        // q_dim_max = n_heads * max(sliding_head_dim, full_head_dim)
        let q_dim_max = config.n_heads * config.sliding_head_dim.max(config.full_head_dim);
        let kv_dim_max = (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim);
        let pb_q = gpu.zeros(&[MAX_PREFILL_BATCH, q_dim_max], DType::F32)?;
        let pb_attn_q = gpu.zeros(&[MAX_PREFILL_BATCH, q_dim_max], DType::F32)?;
        let pb_flash_partials = gpu.zeros(&[MAX_PREFILL_BATCH * flash_partials_sz], DType::F32)?;
        let pb_k = gpu.zeros(&[MAX_PREFILL_BATCH, kv_dim_max], DType::F32)?;
        let pb_v = gpu.zeros(&[MAX_PREFILL_BATCH, kv_dim_max], DType::F32)?;
        let pb_gate = gpu.zeros(&[MAX_PREFILL_BATCH, config.hidden_dim], DType::F32)?;
        let pb_up = gpu.zeros(&[MAX_PREFILL_BATCH, config.hidden_dim], DType::F32)?;
        let pb_ffn_hidden = gpu.zeros(&[MAX_PREFILL_BATCH, config.hidden_dim], DType::F32)?;
        let pb_positions = gpu.zeros(&[MAX_PREFILL_BATCH], DType::F32)?; // i32 packed in f32 slots

        Ok(Gemma4Scratch {
            x, residual, tmp, pos_buf,
            q, k, v, attn_out,
            gate_ffn, up_ffn, ffn_hidden, ffn_out,
            logits, sample_buf, repeat_buf,
            flash_partials,
            v_norm_ones_full,
            moe_cur_mlp, moe_pre2, moe_router_in, moe_router_logits,
            moe_topk_indices, moe_topk_weights, moe_cur_moe,
            moe_expert_gate_up, moe_expert_hidden, moe_expert_out,
            moe_pre2_rot,
            moe_expert_gate_batch, moe_expert_up_batch,
            moe_expert_hidden_batch,
            max_prefill_batch: MAX_PREFILL_BATCH,
            pb_attn_out, pb_ffn_out,
            pb_moe_pre2, pb_moe_pre2_rot,
            pb_moe_router_in, pb_moe_router_logits,
            pb_moe_topk_indices, pb_moe_topk_weights,
            pb_moe_expert_offsets, pb_moe_expert_token_list,
            pb_moe_gate_batch, pb_moe_up_batch, pb_moe_hidden_batch,
            pb_moe_cur_moe, pb_moe_cur_mlp, pb_residual,
            pb_tmp, pb_q, pb_attn_q, pb_flash_partials, pb_k, pb_v, pb_gate, pb_up, pb_ffn_hidden,
            pb_positions,
        })
    }

    /// Release every GPU allocation owned by this scratch. Mirrors the
    /// Qwen35Scratch / LlamaScratch pattern so `unload_model` in the daemon
    /// can reclaim VRAM on idle eviction.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.tmp);
        // pos_buf is a DeviceBuffer, not a GpuTensor; rely on Drop.
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.k);
        let _ = gpu.free_tensor(self.v);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.gate_ffn);
        let _ = gpu.free_tensor(self.up_ffn);
        let _ = gpu.free_tensor(self.ffn_hidden);
        let _ = gpu.free_tensor(self.ffn_out);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.sample_buf);
        let _ = gpu.free_tensor(self.repeat_buf);
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.free_tensor(self.v_norm_ones_full);
        let _ = gpu.free_tensor(self.moe_cur_mlp);
        let _ = gpu.free_tensor(self.moe_pre2);
        let _ = gpu.free_tensor(self.moe_router_in);
        let _ = gpu.free_tensor(self.moe_router_logits);
        let _ = gpu.free_tensor(self.moe_topk_indices);
        let _ = gpu.free_tensor(self.moe_topk_weights);
        let _ = gpu.free_tensor(self.moe_cur_moe);
        let _ = gpu.free_tensor(self.moe_expert_gate_up);
        let _ = gpu.free_tensor(self.moe_expert_hidden);
        let _ = gpu.free_tensor(self.moe_expert_out);
        let _ = gpu.free_tensor(self.moe_pre2_rot);
        let _ = gpu.free_tensor(self.moe_expert_gate_batch);
        let _ = gpu.free_tensor(self.moe_expert_up_batch);
        let _ = gpu.free_tensor(self.moe_expert_hidden_batch);
        let _ = gpu.free_tensor(self.pb_attn_out);
        let _ = gpu.free_tensor(self.pb_ffn_out);
        let _ = gpu.free_tensor(self.pb_moe_pre2);
        let _ = gpu.free_tensor(self.pb_moe_pre2_rot);
        let _ = gpu.free_tensor(self.pb_moe_router_in);
        let _ = gpu.free_tensor(self.pb_moe_router_logits);
        let _ = gpu.free_tensor(self.pb_moe_topk_indices);
        let _ = gpu.free_tensor(self.pb_moe_topk_weights);
        let _ = gpu.free_tensor(self.pb_moe_expert_offsets);
        let _ = gpu.free_tensor(self.pb_moe_expert_token_list);
        let _ = gpu.free_tensor(self.pb_moe_gate_batch);
        let _ = gpu.free_tensor(self.pb_moe_up_batch);
        let _ = gpu.free_tensor(self.pb_moe_hidden_batch);
        let _ = gpu.free_tensor(self.pb_moe_cur_moe);
        let _ = gpu.free_tensor(self.pb_moe_cur_mlp);
        let _ = gpu.free_tensor(self.pb_residual);
        let _ = gpu.free_tensor(self.pb_tmp);
        let _ = gpu.free_tensor(self.pb_q);
        let _ = gpu.free_tensor(self.pb_attn_q);
        let _ = gpu.free_tensor(self.pb_flash_partials);
        let _ = gpu.free_tensor(self.pb_k);
        let _ = gpu.free_tensor(self.pb_v);
        let _ = gpu.free_tensor(self.pb_gate);
        let _ = gpu.free_tensor(self.pb_up);
        let _ = gpu.free_tensor(self.pb_ffn_hidden);
        let _ = gpu.free_tensor(self.pb_positions);
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────

/// Apply the Gemma 4 MoE parallel branch (26B-A4B). Called from each layer
/// AFTER `down_proj` produces `scratch.ffn_out`, REPLACING the standalone
/// `post_feedforward_layernorm` call. On exit, `scratch.tmp` holds the
/// combined `post_norm(cur_mlp + cur_moe)`, ready for `x = residual + tmp`.
///
/// Legacy serialized path only (8 experts × 5 launches = 40 launches/layer).
/// The fused indexed-GEMV path (`gemv_hfq4g256_moe_gate_up_k8_indexed`) and
/// fused-down path from origin/gemma4 are NOT yet ported — they require a
/// `rotate_x_mq` + `mq_signs` plumbing the modular crate doesn't have yet.
/// Both produce mathematically identical output; the legacy path is the
/// safety/reference baseline.
///
/// HF reference (modeling_gemma4.py Gemma4MoeBlock + Gemma4MoeMLP):
///   cur_mlp = post_feedforward_layernorm_1(ffn_out)        # standard SwiGLU out, normed
///   pre2    = pre_feedforward_layernorm_2(attn_out)        # MoE branch input
///   router_in    = rmsnorm(attn_out, router_scale) / sqrt(dim)
///   router_logits = router_proj @ router_in
///   topk_idx, topk_w = softmax_topk_renorm(router_logits, k=8)
///   cur_moe = sum_k [ topk_w[k] * per_expert_scale[i_k] *
///                     down_proj_{i_k}( gelu_tanh(gate) * up
///                                      where (gate, up) = split(gate_up_proj_{i_k} @ pre2) ) ]
///   cur_moe = post_feedforward_layernorm_2(cur_moe)
///   tmp     = post_feedforward_layernorm(cur_mlp + cur_moe)
///
/// `attn_out` parameter is the layer's post-attention residual stream
/// (= `scratch.residual` at the call site, since the caller stored
/// `residual = x` after the attention sandwich).
fn apply_moe_branch(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    moe: &MoeLayerExtras,
    post_ffn_norm: &GpuTensor,
    attn_out: &GpuTensor,
) -> HipResult<()> {
    let dim = config.dim;
    let dim_bytes = dim * 4;
    let mi = config.moe_intermediate_size;
    let n_exp = config.num_experts;
    let k_top = config.top_k_experts;
    if k_top != 8 {
        return Err(hip_bridge::HipError::new(
            0, &format!("MoE top_k_experts={k_top} unsupported (kernel hardcoded to 8)"),
        ));
    }

    // 1) cur_mlp = post_feedforward_layernorm_1(ffn_out)
    gpu.rmsnorm_f32(&scratch.ffn_out, &moe.post_feedforward_layernorm_1,
        &scratch.moe_cur_mlp, config.norm_eps)?;

    // 2) pre2 = pre_feedforward_layernorm_2(attn_out)
    gpu.rmsnorm_f32(attn_out, &moe.pre_feedforward_layernorm_2,
        &scratch.moe_pre2, config.norm_eps)?;

    // Dump moe_pre2 on first call
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static PRE2_DUMP: AtomicUsize = AtomicUsize::new(0);
        let pc = PRE2_DUMP.fetch_add(1, Ordering::Relaxed);
        if pc == 0 {
            if let Ok(data) = gpu.download_f32(&scratch.moe_pre2) {
                let sum: f64 = data.iter().map(|&v| v as f64).sum();
                eprintln!("[moe pre2] first4={:?} sum={sum:.4}", &data[..4.min(data.len())]);
            }
        }
    }
    // 3) Router input: rmsnorm(attn_out, router_scale) / sqrt(dim).
    //    Equivalent to ref `rms_norm(x) * router_scale / sqrt(dim)` since
    //    rmsnorm_f32(x, w) = w * x / sqrt(mean(x²) + eps) — elementwise commutative.
    gpu.rmsnorm_f32(attn_out, &moe.router_scale,
        &scratch.moe_router_in, config.norm_eps)?;
    gpu.scale_f32(&scratch.moe_router_in, 1.0 / (dim as f32).sqrt())?;

    // 4) Router GEMV → logits [n_exp]
    weight_gemv(gpu, &moe.router_proj, &scratch.moe_router_in, &scratch.moe_router_logits)?;

    // 5) Top-K softmax + renorm on device. Kernel hardcoded to k_top=8.
    gpu.moe_softmax_topk_renorm_k8(
        &scratch.moe_router_logits,
        &scratch.moe_topk_indices,
        &scratch.moe_topk_weights,
        n_exp,
        true,
    )?;

    // 6) Dispatch top-K experts. Two paths:
    //
    //   (a) Indexed (fast) path: gate_up is MQ4G256/MG4G256 and down is
    //       Q8_0. Two kernel launches replace the 8-iteration CPU loop +
    //       2 D2H syncs/layer (=60 syncs/token on 30-layer Gemma 4 26B).
    //       Whole MoE branch becomes hipGraph-capturable.
    //
    //   (b) Legacy path: any other quant mix. CPU per-expert loop with
    //       the 2 D2H downloads per layer.
    //
    // 26B-A4B-it (Gemma 4): gate_up=MQ4G256 (dim=2816 / 256-aligned),
    // down=Q8_0 (mi=704 not 256-aligned → quantizer falls back to Q8_0).
    // Hits the fast path. Other Gemma 4 variants might land in legacy.
    let first = &moe.experts[0];
    let gate_mq4 = first.gate_up_proj.gpu_dtype == rdna_compute::DType::MQ4G256;
    let gate_q8  = first.gate_up_proj.gpu_dtype == rdna_compute::DType::Q8_0;
    let down_q8  = first.down_proj.gpu_dtype == rdna_compute::DType::Q8_0;
    let down_hfq4g128 = first.down_proj.gpu_dtype == rdna_compute::DType::HFQ4G128;
    let fast = (gate_mq4 || gate_q8) && down_q8;
    let _ = (gate_mq4, down_hfq4g128);
    {
        use std::sync::OnceLock;
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| {
            eprintln!("[gemma4 MoE] dispatch path: {} (gate_up={:?} down={:?})",
                if fast { "indexed-fast" } else { "legacy-cpu-loop" },
                first.gate_up_proj.gpu_dtype, first.down_proj.gpu_dtype);
        });
    }

    if fast {
        // Indexed gate_up: 8 fused GEMVs reading expert IDs from device.
        //   y_gate: [k_top × mi], y_up: [k_top × mi]
        if gate_mq4 {
            // MQ4G256 needs FWHT-rotated input.
            gpu.rotate_x_mq(&scratch.moe_pre2, &scratch.moe_pre2_rot, dim)?;
            gpu.gemv_mq4g256_moe_gate_up_k8_indexed(
                &moe.experts_gate_up_ptrs,
                &scratch.moe_topk_indices,
                &scratch.moe_pre2_rot,
                &scratch.moe_expert_gate_batch,
                &scratch.moe_expert_up_batch,
                2 * mi, dim,
            )?;
        } else {
            // Q8_0 — no rotation needed.
            gpu.gemv_q8_0_moe_gate_up_k8_indexed(
                &moe.experts_gate_up_ptrs,
                &scratch.moe_topk_indices,
                &scratch.moe_pre2,
                &scratch.moe_expert_gate_batch,
                &scratch.moe_expert_up_batch,
                2 * mi, dim,
            )?;
        };

        // Batched gelu_tanh + mul over [k_top × mi].
        gpu.gelu_tanh_f32(&scratch.moe_expert_gate_batch,
            &scratch.moe_expert_hidden_batch, k_top * mi)?;
        gpu.mul_f32(&scratch.moe_expert_hidden_batch,
            &scratch.moe_expert_up_batch,
            &scratch.moe_expert_hidden_batch)?;

        // Zero accumulator (memset is sync but tiny — 11 KB for dim=2816).
        if let Some(s) = gpu.active_stream.as_ref() {
            gpu.hip.memset_async(&scratch.moe_cur_moe.buf, 0, dim_bytes, s)?;
        } else {
            gpu.hip.memset(&scratch.moe_cur_moe.buf, 0, dim_bytes)?;
        }

        // Indexed down + scaled residual: 8 fused GEMVs, atomicAdd into
        // moe_cur_moe with scale = topk_weights[krank] *
        // per_expert_scale[topk_indices[krank]] (all on device). Quant
        // variant picked by the down weight format. Gemma 4 26B-A4B-it's
        // down has K=mi=704 → HFQ4G128. Future Gemma 4 sizes with
        // K%32==0 only could land on Q8_0 instead.
        if down_q8 {
            gpu.gemv_q8_0_moe_down_residual_scaled_k8_indexed(
                &moe.experts_down_ptrs,
                &scratch.moe_topk_indices,
                &scratch.moe_topk_weights,
                &moe.per_expert_scale,
                &scratch.moe_expert_hidden_batch,
                &scratch.moe_cur_moe,
                dim, mi,
            )?;
        } else {
            // down_hfq4g128 path.
            gpu.gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
                &moe.experts_down_ptrs,
                &scratch.moe_topk_indices,
                &scratch.moe_topk_weights,
                &moe.per_expert_scale,
                &scratch.moe_expert_hidden_batch,
                &scratch.moe_cur_moe,
                dim, mi,
            )?;
        }
    } else {
        // ── Legacy CPU per-expert path (quant mix doesn't match the
        //    fast kernels). 60 D2H syncs/token, no graph capture. ──
        let idx_bytes = gpu.download_f32(&scratch.moe_topk_indices)?;
        let topk_indices: Vec<usize> = unsafe {
            std::slice::from_raw_parts(idx_bytes.as_ptr() as *const i32, k_top)
        }.iter().map(|&i| i as usize).collect();
        let topk_weights = gpu.download_f32(&scratch.moe_topk_weights)?;
        for &e in topk_indices.iter().take(k_top) {
            if e >= n_exp {
                return Err(hip_bridge::HipError::new(
                    0, &format!("MoE topk index {e} out of range (n_exp={n_exp})"),
                ));
            }
        }
        if let Some(s) = gpu.active_stream.as_ref() {
            gpu.hip.memset_async(&scratch.moe_cur_moe.buf, 0, dim_bytes, s)?;
        } else {
            gpu.hip.memset(&scratch.moe_cur_moe.buf, 0, dim_bytes)?;
        }
        // Dump router info for first MoE layer
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALL: AtomicUsize = AtomicUsize::new(0);
            let c = CALL.fetch_add(1, Ordering::Relaxed);
            if c == 0 {
            eprintln!("[moe diag] first call: topk_indices={:?} topk_weights={:?}\n  per_expert_scale[0..8]={:?}",
                    &topk_indices[..k_top], &topk_weights[..k_top],
                    &moe.per_expert_scale_host[..8.min(n_exp)]);
        }
        }
        for ki in 0..k_top {
            let e = topk_indices[ki];
            let weight = topk_weights[ki] * moe.per_expert_scale_host[e];
            let expert = &moe.experts[e];
            weight_gemv(gpu, &expert.gate_up_proj, &scratch.moe_pre2, &scratch.moe_expert_gate_up)?;
            let gate = scratch.moe_expert_gate_up.sub_offset(0, mi);
            let up   = scratch.moe_expert_gate_up.sub_offset(mi, mi);
            gpu.gelu_tanh_f32(&gate, &scratch.moe_expert_hidden, mi)?;
            gpu.mul_f32(&scratch.moe_expert_hidden, &up, &scratch.moe_expert_hidden)?;
            weight_gemv(gpu, &expert.down_proj, &scratch.moe_expert_hidden, &scratch.moe_expert_out)?;
            gpu.scaled_add_inplace_cpu_scalar_f32(&scratch.moe_cur_moe, &scratch.moe_expert_out, weight)?;
            // Dump first expert's contribution
            if ki == 0 {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static EXPERT_CALL: AtomicUsize = AtomicUsize::new(0);
                let ec = EXPERT_CALL.fetch_add(1, Ordering::Relaxed);
                if ec == 0 {
                    // gate_up output
                    if let Ok(gu_data) = gpu.download_f32(&scratch.moe_expert_gate_up) {
                        let sum: f64 = gu_data.iter().map(|&v| v as f64).sum();
                        let gate_sum: f64 = gu_data[..mi].iter().map(|&v| v as f64).sum();
                        let up_sum: f64 = gu_data[mi..].iter().map(|&v| v as f64).sum();
                        eprintln!("[moe expert] expert={e} GATE_UP: gate_first4={:?} gate_sum={gate_sum:.4} up_first4={:?} up_sum={up_sum:.4}",
                            &gu_data[..4.min(mi)], &gu_data[mi..mi+4.min(mi)],
                        );
                    }
                    // gate-up shape: gate_up_proj m=2*mi=1408, k=dim=2816
                    eprintln!("[moe expert] expert={e} gate_up_proj m={} k={} dtype={:?}",
                        expert.gate_up_proj.m, expert.gate_up_proj.k,
                        expert.gate_up_proj.gpu_dtype);
                    // down_proj output
                    if let Ok(data) = gpu.download_f32(&scratch.moe_expert_out) {
                        let sum: f64 = data.iter().map(|&v| v as f64).sum();
                        eprintln!("[moe expert] expert={e} weight={weight:.6} down_dtype={:?} expert_out_sum={sum:.4} first4={:?}",
                            expert.down_proj.gpu_dtype,
                            &data[..4.min(data.len())]);
                    }
                }
            }
        }
    }

    // 9) cur_moe = post_feedforward_layernorm_2(cur_moe) — in-place
    // Dump MoE branch intermediates on first call
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static MOE_END: AtomicUsize = AtomicUsize::new(0);
        let mc = MOE_END.fetch_add(1, Ordering::Relaxed);
        if mc == 0 {
            if let Ok(mlp_data) = gpu.download_f32(&scratch.moe_cur_mlp) {
                let sum: f64 = mlp_data.iter().map(|&v| v as f64).sum();
                eprintln!("[moe branch] cur_mlp first4={:?} sum={sum:.4}", &mlp_data[..4.min(mlp_data.len())]);
            }
            if let Ok(moe_data) = gpu.download_f32(&scratch.moe_cur_moe) {
                let sum: f64 = moe_data.iter().map(|&v| v as f64).sum();
                eprintln!("[moe branch] cur_moe (before norm2) first4={:?} sum={sum:.4}", &moe_data[..4.min(moe_data.len())]);
            }
        }
    }
    gpu.rmsnorm_f32(&scratch.moe_cur_moe, &moe.post_feedforward_layernorm_2,
        &scratch.moe_cur_moe, config.norm_eps)?;

    // 10) combined = cur_mlp + cur_moe → scratch.tmp
    gpu.add_f32(&scratch.moe_cur_mlp, &scratch.moe_cur_moe, &scratch.tmp)?;

    // 11) tmp = post_feedforward_layernorm(combined)
    gpu.rmsnorm_f32(&scratch.tmp, post_ffn_norm, &scratch.tmp, config.norm_eps)?;

    Ok(())
}

/// Batched MoE branch — processes N tokens at once. Mirrors `apply_moe_branch`
/// but every per-token tensor becomes a `[N × *]` row-major batch and every
/// kernel call uses the `_batched` variant.
///
/// Inputs (live in `scratch.pb_*`):
///   - pb_attn_out[N × dim]: post-attention output (input to pre_norm_2 and router)
///   - pb_ffn_out [N × dim]: dense FFN output  (input to post_norm_1)
///
/// Output:
///   - pb_moe_pre2[N × dim] holds the post-FFN-layernorm(cur_mlp + cur_moe) —
///     i.e. what the per-token path writes to `scratch.tmp`. Caller adds it
///     to the per-token residual + applies the layer scalar.
///
/// Only the indexed-fast path (MQ4G256 gate_up + HFQ4G128/Q8_0 down) is wired.
/// Falls back to per-token calls when the quant mix doesn't match (slow but
/// correct).
fn apply_moe_branch_batched(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    moe: &MoeLayerExtras,
    post_ffn_norm: &GpuTensor,
    n_batch: usize,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let n_exp = config.num_experts;
    let k_top = config.top_k_experts;
    if k_top != 8 {
        return Err(hip_bridge::HipError::new(
            0, &format!("MoE top_k_experts={k_top} unsupported (kernel hardcoded to 8)"),
        ));
    }
    debug_assert!(n_batch <= scratch.max_prefill_batch,
        "n_batch={n_batch} > MAX_PREFILL_BATCH={}", scratch.max_prefill_batch);

    let first = &moe.experts[0];
    let _gate_dtype = first.gate_up_proj.gpu_dtype;
    let _down_dtype = first.down_proj.gpu_dtype;
    // TODO(Phase 4): fused batched MoE kernels not yet ported.
    // Using per-token expert loop (correct but slow for prefill).

    // 1) cur_mlp_batch = post_feedforward_layernorm_1(pb_ffn_out)
    gpu.rmsnorm_batched(&scratch.pb_ffn_out, &moe.post_feedforward_layernorm_1,
        &scratch.pb_moe_cur_mlp, n_batch, dim, config.norm_eps)?;

    // 2) pre2_batch = pre_feedforward_layernorm_2(pb_attn_out)
    gpu.rmsnorm_batched(&scratch.pb_attn_out, &moe.pre_feedforward_layernorm_2,
        &scratch.pb_moe_pre2, n_batch, dim, config.norm_eps)?;

    // 3) router_in = rmsnorm(attn_out, router_scale) / sqrt(dim)
    gpu.rmsnorm_batched(&scratch.pb_attn_out, &moe.router_scale,
        &scratch.pb_moe_router_in, n_batch, dim, config.norm_eps)?;
    gpu.scale_f32(&scratch.pb_moe_router_in, 1.0 / (dim as f32).sqrt())?;
    // scale_f32 operates on the full tensor — for [N × dim] this scales
    // every element, which is what we want (each row is divided by sqrt(dim)).

    // 4) Router GEMM → logits [N × n_exp]
    run_prefill_gemm(gpu, &moe.router_proj, &scratch.pb_moe_router_in,
        &scratch.pb_moe_router_logits, n_batch)?;

    // 5) Batched top-K softmax + renorm.
    gpu.moe_softmax_topk_renorm_k8_batched(
        &scratch.pb_moe_router_logits,
        &scratch.pb_moe_topk_indices,
        &scratch.pb_moe_topk_weights,
        n_exp,
        true,
        n_batch,
    )?;

    // 5b) Phase B (opt-in): build per-expert buckets so the bucketed GEMV
    // can reuse the weight tile across all tokens routed to the same expert.
    // Measured -5.7% prefill regression on Gemma 4 26B-A4B-it / gfx1201 at
    // N=128 batch (132 → 124 tok/s): the bucketed kernel pre-loads the
    // weight tile into ~48 VGPRs per lane (sc/zp/pk × 16 groups) which
    // cuts occupancy below the indexed_batched kernel that streams weights
    // in the inner loop, and the serialized loop over bucket tokens loses
    // more parallelism than is recovered from launch-overhead savings
    // (180k blocks vs 1.4M). Default OFF; kept behind opt-in for future
    // tuning (smaller MAX_GROUPS, LDS staging, fewer launch_bounds waves).
    let use_bucketed = std::env::var("HIPFIRE_MOE_BUCKETED")
        .ok().map(|v| v == "1").unwrap_or(false);
    if use_bucketed {
        gpu.moe_bucket_build(
            &scratch.pb_moe_topk_indices,
            &scratch.pb_moe_expert_offsets,
            &scratch.pb_moe_expert_token_list,
            n_batch, k_top, n_exp,
        )?;
    }

    // 6-13) Per-token expert loop (Phase 4 fallback until fused kernels ported).
    // For each token: extract per-token topk indices/weights,
    // run 8 expert GEMVs, accumulate into pb_moe_cur_moe.
    let dim_bytes = dim * 4;
    let topk_idx_host = gpu.download_f32(&scratch.pb_moe_topk_indices)?;
    let topk_wt_host = gpu.download_f32(&scratch.pb_moe_topk_weights)?;
    let topk_indices_batch: Vec<Vec<usize>> = (0..n_batch).map(|b| {
        unsafe {
            std::slice::from_raw_parts(
                topk_idx_host.as_ptr().add(b * k_top) as *const i32, k_top)
        }.iter().map(|&i| i as usize).collect()
    }).collect();
    let topk_weights_batch: Vec<Vec<f32>> = (0..n_batch).map(|b| {
        topk_wt_host[b * k_top..(b + 1) * k_top].to_vec()
    }).collect();

    // Zero cur_moe_batch accumulator.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memset_async(&scratch.pb_moe_cur_moe.buf, 0, n_batch * dim_bytes, s)?;
    } else {
        gpu.hip.memset(&scratch.pb_moe_cur_moe.buf, 0, n_batch * dim_bytes)?;
    }

    for b in 0..n_batch {
        for ki in 0..k_top {
            let e = topk_indices_batch[b][ki];
            let weight = topk_weights_batch[b][ki] * moe.per_expert_scale_host[e];
            let expert = &moe.experts[e];

            // Copy this token's pre2 row into scratch.moe_pre2
            if let Some(s) = gpu.active_stream.as_ref() {
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.moe_pre2.buf, 0,
                    &scratch.pb_moe_pre2.buf, b * dim_bytes,
                    dim_bytes, s)?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.moe_pre2.buf, 0,
                    &scratch.pb_moe_pre2.buf, b * dim_bytes,
                    dim_bytes)?;
            }

            // gate_up = expert.gate_up_proj @ pre2
            weight_gemv(gpu, &expert.gate_up_proj, &scratch.moe_pre2,
                &scratch.moe_expert_gate_up)?;
            let gate = scratch.moe_expert_gate_up.sub_offset(0, mi);
            let up   = scratch.moe_expert_gate_up.sub_offset(mi, mi);
            // hidden = gelu_tanh(gate) * up
            gpu.gelu_tanh_f32(&gate, &scratch.moe_expert_hidden, mi)?;
            gpu.mul_f32(&scratch.moe_expert_hidden, &up,
                &scratch.moe_expert_hidden)?;
            // expert_out = expert.down_proj @ hidden
            weight_gemv(gpu, &expert.down_proj, &scratch.moe_expert_hidden,
                &scratch.moe_expert_out)?;
            // scaled_add into the correct row of pb_moe_cur_moe
            if let Some(s) = gpu.active_stream.as_ref() {
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.tmp.buf, 0,
                    &scratch.pb_moe_cur_moe.buf, b * dim_bytes,
                    dim_bytes, s)?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.tmp.buf, 0,
                    &scratch.pb_moe_cur_moe.buf, b * dim_bytes,
                    dim_bytes)?;
            }
            gpu.scaled_add_inplace_cpu_scalar_f32(
                &scratch.tmp, &scratch.moe_expert_out, weight)?;
            if let Some(s) = gpu.active_stream.as_ref() {
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.pb_moe_cur_moe.buf, b * dim_bytes,
                    &scratch.tmp.buf, 0,
                    dim_bytes, s)?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.pb_moe_cur_moe.buf, b * dim_bytes,
                    &scratch.tmp.buf, 0,
                    dim_bytes)?;
            }
        }
    }

    // 11) post_feedforward_layernorm_2(cur_moe) in-place batched.
    gpu.rmsnorm_batched(&scratch.pb_moe_cur_moe, &moe.post_feedforward_layernorm_2,
        &scratch.pb_moe_cur_moe, n_batch, dim, config.norm_eps)?;

    // 12) combined = cur_mlp + cur_moe → reuse pb_moe_pre2 as the combined buffer.
    gpu.add_f32(&scratch.pb_moe_cur_mlp, &scratch.pb_moe_cur_moe,
        &scratch.pb_moe_pre2)?;

    // 13) post_feedforward_layernorm(combined) → batched, in-place.
    gpu.rmsnorm_batched(&scratch.pb_moe_pre2, post_ffn_norm,
        &scratch.pb_moe_pre2, n_batch, dim, config.norm_eps)?;

    Ok(())
}

/// Single-token decode. Phase 3 implementation.
///
/// Precondition: `scratch.sliding_cos/sin` + `scratch.full_cos/sin` +
/// `scratch.v_norm_ones_full` must be populated by the loader before the
/// first forward call (one-time init).
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    token: u32,
    pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;

    // 1) Embedding lookup + sqrt(dim) scale.
    // ALWAYS direct — the captured graph can't bake in `token` (varies per
    // call). The embed lookup fills scratch.x; the rest of the forward
    // (which is graph-capturable now that the MoE branch has no D2H syncs)
    // reads from scratch.x. Same split Qwen35 uses for its captured path.
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0    => gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::F32     => gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, token, dim)?,
        _ => return Err(hip_bridge::HipError::new(0, "unsupported Gemma 4 embed format")),
    }
    gpu.scale_f32(&scratch.x, config.embed_scale)?;

    // hipGraph capture/replay policy.
    //   - DEFAULT-OFF for Gemma 4 (until cross-arch / long-context validation).
    //   - Fixed 2026-05-19 (evening): the earlier diagnosis ("kv_len = pos + 1
    //     is a scalar arg baked at capture") was wrong — attention_flash_*_window
    //     kernels actually compute seq_len = pos_buf[0] + 1 at runtime
    //     (`attention_flash_asym3_tile.hip:47`). The real bugs were three
    //     elementwise kernels (`scale_f32`, `mul_f32`, `add_f32` in
    //     `crates/rdna-compute/src/dispatch.rs`) that used direct `launch_kernel`
    //     instead of `launch_maybe_blob`. `mul_f32` and `add_f32` additionally
    //     ran on the default stream (`None`) instead of `stream_ref()`, so
    //     during graph capture they were NOT recorded at all — every replay
    //     skipped the FFN's `ffn_hidden = gelu(gate) * up` multiply, feeding
    //     wrong tensors into down_proj → token attractor on greedy decode.
    //     `scale_f32` was on the capture stream but using raw `kernelParams`
    //     (stack pointers that dangle by replay under ROCm 7.x loader).
    //     All three converted to `launch_maybe_blob` in the same commit as
    //     this comment. HIPFIRE_GRAPH=1 now produces clean output.
    //   - Compact offset != 0 (TriAttention eviction) still breaks capture
    //     for the same reason as Qwen35 — bail to direct in that case.
    static GRAPH_OVERRIDE_ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let graph_override = *GRAPH_OVERRIDE_ENV.get_or_init(|| {
        match std::env::var("HIPFIRE_GRAPH").ok().as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        }
    });
    let use_graph = graph_override.unwrap_or(false)
        && kv_sliding.compact_offset == 0
        && kv_full.compact_offset == 0;

    if use_graph && gpu.graphs.graph_exec.is_some() {
        // ── Replay path. Update pos_buf via stream_write_value32 (graph-
        //    replay-safe, no host→device copy). The captured graph reads
        //    pos_buf at kernel-launch time, so a fresh `pos` propagates
        //    without recapture. ──
        let stream = gpu.active_stream.as_ref().unwrap();
        gpu.hip.stream_write_value32(stream, &scratch.pos_buf, pos as u32, 0)?;
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
    } else if use_graph && gpu.graphs.graph_exec.is_none() {
        let pos_i32 = pos as i32;
        if !false /* graph-capture-not-wired */ {
            // ── Warmup: direct dispatch so any JIT kernel compiles or lazy
            //    scratch allocations happen OUTSIDE a capture region.
            //    Capturing on the first call hits "hipMalloc not permitted
            //    under stream capture". Same pattern as Qwen35. ──
            /* graph-capture-not-wired */;
            gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            forward_scratch_inner(gpu, weights, config, pos, kv_sliding, kv_full, scratch)?;
        } else {
            // ── First post-warmup call: capture the forward into a graph. ──
            if gpu.active_stream.is_none() {
                gpu.active_stream = Some(gpu.hip.stream_create()?);
            }
            gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
            forward_scratch_inner(gpu, weights, config, pos, kv_sliding, kv_full, scratch)?;
            gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
            // hipStreamCaptureModeGlobal RECORDS — kernels don't execute
            // during capture. Replay once so THIS pos's forward actually
            // runs (KV write, logits update).
            gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
            eprintln!("[gemma4 hipGraph] captured {} blobs, instantiated",
                gpu.graphs.capture_blobs.len());
        }
    } else {
        // ── Direct path (no graph) ──
        let pos_i32 = pos as i32;
        gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_inner(gpu, weights, config, pos, kv_sliding, kv_full, scratch)?;
    }
    Ok(())
}

/// Graph-capturable body of `forward_scratch`. Caller is responsible for:
///   - embed lookup + embed scale (fills scratch.x — varies per call)
///   - pos_buf update (htod for warmup/capture; stream_write_value32 for replay)
fn forward_scratch_inner(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    // #397 Ship 6 — forward-as-pipeline. When HIPFIRE_FORWARD_LOWERED=1,
    // route decode through the lowered super-op executor.
    if forward_lowered_enabled() {
        return forward_scratch_inner_lowered(
            gpu, weights, config, pos, kv_sliding, kv_full, scratch,
        );
    }

    // 3) Per-layer forward (legacy hand path).
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        // Diagnostic: dump residual before layer
        if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") && layer_idx < 2 {
            let data = gpu.download_f32(&scratch.x).unwrap_or_default();
            let sum: f64 = data.iter().map(|&v| v as f64).sum();
            let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("[gemma4 diag] pos={pos} L{layer_idx} before: first4={:?} sum={sum:.4e} min={min:.4} max={max:.4}",
                &data[..4.min(data.len())]);
        }
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                sliding_layer_decode(gpu, config, lw, pos, kv_sliding, sliding_kv_idx, scratch)?;
                sliding_kv_idx += 1;
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                full_layer_decode(gpu, config, lw, pos, kv_full, full_kv_idx, scratch)?;
                full_kv_idx += 1;
            }
            _ => return Err(hip_bridge::HipError::new(
                0,
                &format!("Gemma 4 layer {} type/weights mismatch", layer_idx),
            )),
        }
        // Diagnostic: dump residual after layer
        if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") {
            let data = gpu.download_f32(&scratch.x).unwrap_or_default();
            let sum: f64 = data.iter().map(|&v| v as f64).sum();
            let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("[gemma4 diag] pos={pos} L{layer_idx} hidden: first4={:?} sum={sum:.4e} min={min:.4} max={max:.4}",
                &data[..4.min(data.len())]);
        }
    }

    // 4) Final RMSNorm.
    gpu.rmsnorm_f32(&scratch.x, &weights.final_norm, &scratch.tmp, config.norm_eps)?;

    // Diagnostic: dump hidden state before lm_head
    if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") {
        let data = gpu.download_f32(&scratch.tmp).unwrap_or_default();
        let sum: f64 = data.iter().map(|&v| v as f64).sum();
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        eprintln!("[gemma4 diag] pos={pos} hidden pre-lm_head: first4={:?} sum={sum:.4e} min={min:.4} max={max:.4}",
            &data[..4.min(data.len())]);
    }

    // 5) LM head → logits (reads tied embed bytes via lm_head.buf alias).
    let ctx = DispatchCtx::new(gpu);
    let wr_lm = weights.lm_head.dispatch_ref();
    execute_steps(gpu, &ctx, &[
        Step::Gemv { w: &wr_lm, input: GemvInput::Raw(&scratch.tmp), out: &scratch.logits },
    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

    // 6) Final logit softcap (Gemma 4): logits = tanh(logits / cap) * cap.
    if config.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&scratch.logits, config.vocab_size, config.final_logit_softcapping)?;
    }

    // Diagnostic: dump logits
    if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") {
        let data = gpu.download_f32(&scratch.logits).unwrap_or_default();
        let top5: Vec<(usize, f32)> = {
            let mut indexed: Vec<(usize, f32)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.into_iter().take(5).collect()
        };
        eprintln!("[gemma4 diag] pos={pos} logits top5: {:?}", top5);
    }

    Ok(())
}

/// Single sliding-window attention layer.
///
/// Order matches HF modeling_gemma4.py::Gemma4TextDecoderLayer +
/// Gemma4TextAttention (sliding branch):
///   residual = x
///   x = input_layernorm(x)              — RMSNorm (sandwich pre-attn)
///   q = q_proj(x); q = q_norm(q)        — RMSNorm over head_dim=256
///   k = k_proj(x); k = k_norm(k)
///   v = v_proj(x); v = v_norm(v)         — no_scale RMSNorm (ones buffer)
///   RoPE(q, k) with rotate_half, theta=10000, full head_dim=256
///   write K, V to KV cache at position `pos`
///   attn = flash_attention(q, K, V, window_size=1024, scale=1.0 effective)
///   x = o_proj(attn)
///   x = post_attention_layernorm(x)     — RMSNorm (sandwich post-attn)
///   x = residual + x
///   residual = x
///   x = pre_feedforward_layernorm(x)    — RMSNorm (sandwich pre-FFN)
///   gate = gate_proj(x); up = up_proj(x)
///   ffn = gelu_pytorch_tanh(gate) * up  — SwiGLU
///   x = down_proj(ffn)
///   x = post_feedforward_layernorm(x)   — RMSNorm (sandwich post-FFN)
///   x = residual + x
///   x = x * layer_scalar                — learned per-layer scalar
///
/// Gemma 4 attention uses `scaling=1.0` in HF (see modeling_gemma4.py line 1143).
/// Our flash kernels bake in `scale = 1/sqrt(head_dim)`; we compensate by
/// pre-scaling Q by sqrt(head_dim) so the effective scale is 1.0.
// Sliding-window kernel correctness: the 4 attention_flash_*_window dispatch
// sites here use the kernels modified by commit b608c5f (sliding-window arg
// threaded through 7 attention_flash_*.hip files). Static review of the diff
// is clean — per-position out_of_window predicate writes -1e30 to scores;
// early-tile-skip writes {-1e30, 0, zeros} so reduce sees no contribution;
// all 32 lanes participate in __shfl_xor; window_size=0 = byte-identical
// for Qwen 3.5/3.6. Runtime NRMSE validation: verify_against_torch.rs Phase 2
// (D2.5) — per-layer attention NRMSE against the bf16 PyTorch reference dump.
fn sliding_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    sliding_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, false)
}

/// As `sliding_layer_decode` but with `stop_before_moe=true`, the function
/// returns immediately after the dense FFN computes `scratch.ffn_out`,
/// leaving `scratch.residual` holding the post-attention x. Used by
/// `forward_prefill_chunk` to interleave per-token attention with a
/// batched MoE call across all N tokens.
fn sliding_layer_attn_ffn_only(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    sliding_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, true)
}

fn sliding_layer_decode_impl(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
    stop_before_moe: bool,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.sliding_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.sliding_n_kv_heads;
    let dim_bytes = dim * 4;

    // residual = x
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }
    let _dump_on = std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") && (pos == 0 || pos == 1) && kv_layer_idx == 0;
    if _dump_on { dbg_dump(gpu, "[v1] L0 input scratch.x", &scratch.x, dim); }
    let ctx = DispatchCtx::new(gpu);

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(&scratch.x, &lw.input_layernorm, &scratch.tmp, config.norm_eps)?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after input_norm", &scratch.tmp, dim); }

    // Q/K/V projections: q[n_heads*head_dim], k/v[n_kv*head_dim].
    let wr_q = lw.q_proj.dispatch_ref();
    let wr_k = lw.k_proj.dispatch_ref();
    execute_steps(gpu, &ctx, &[
        Step::Gemv { w: &wr_q, input: GemvInput::Raw(&scratch.tmp), out: &scratch.q },
        Step::Gemv { w: &wr_k, input: GemvInput::Raw(&scratch.tmp), out: &scratch.k },
    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    weight_gemv(gpu, &lw.v_proj, &scratch.tmp, &scratch.v)?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after q_proj", &scratch.q, n_heads * head_dim);
        dbg_dump(gpu, "[v1] L0 after k_proj", &scratch.k, n_kv * head_dim);
        dbg_dump(gpu, "[v1] L0 after v_proj", &scratch.v, n_kv * head_dim);
    }

    // q_norm + k_norm + no-scale v_norm across head_dim (in-place).
    // v_norm matches HF Gemma 4 `value_states = v_norm(v_proj(x))` (no_scale=True
    // RMSNorm — divide-only, ones buffer as weight). Same pattern as full_layer_decode.
    // Omitting v_norm compounds attention-output bias across 48 sliding layers and
    // produces single-token garbage end-to-end while still passing Phase 2 kernel
    // NRMSE (which tests q_norm + k_norm but not v_norm — see d2.5-results.md).
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.v, &scratch.v_norm_ones_full, &scratch.v,
        n_kv, head_dim, config.norm_eps)?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after q_norm", &scratch.q, n_heads * head_dim);
        dbg_dump(gpu, "[v1] L0 after k_norm", &scratch.k, n_kv * head_dim);
        dbg_dump(gpu, "[v1] L0 after v_norm", &scratch.v, n_kv * head_dim);
    }

    // Pre-scale Q by sqrt(head_dim) so the flash-attn kernel's internal
    // 1/sqrt(head_dim) cancels, leaving the effective Gemma 4 scale of 1.0.
    // Only the first n_heads*head_dim elements of scratch.q are live.
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after scale_q", &scratch.q, n_heads * head_dim); }

    // Full rotate_half RoPE, theta=10000, head_dim=256 (all dims rotate).
    gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv, head_dim, config.sliding_rope_theta)?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after rope_q", &scratch.q, n_heads * head_dim);
        dbg_dump(gpu, "[v1] L0 after rope_k", &scratch.k, n_kv * head_dim);
        // Dump raw Q/K/V after RoPE for score analysis
        let q_data = gpu.download_f32(&scratch.q).unwrap_or_default();
        let k_data = gpu.download_f32(&scratch.k).unwrap_or_default();
        let v_data = gpu.download_f32(&scratch.v).unwrap_or_default();
        let _ = std::fs::write("/tmp/gemma4_q_rope.bin", unsafe { std::slice::from_raw_parts(q_data.as_ptr() as *const u8, q_data.len() * 4) });
        let _ = std::fs::write("/tmp/gemma4_k_rope.bin", unsafe { std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4) });
        let _ = std::fs::write("/tmp/gemma4_v.bin", unsafe { std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4) });
        eprintln!("[v1] Dumped post-RoPE Q({}), K({}), V({})", q_data.len(), k_data.len(), v_data.len());
    }

    // KV cache write + flash attention via dispatch framework (Step::Attend).
    // flash_mode=2 (forced) because sliding layers always need flash for window masking.
    let sliding_cap = kv_cache.physical_cap as u32;
    {
        let tier_inputs = KvTierInputs {
            quant_asym4: kv_cache.quant_asym4,
            quant_asym3: kv_cache.quant_asym3,
            quant_asym2: kv_cache.quant_asym2,
            quant_q8: kv_cache.quant_q8,
            quant_fwht: kv_cache.quant_fwht,
            quant_hfq4: false,
            quant_q4: false,
            v_mode_bits: kv_cache.v_mode_bits(),
            pos,
            flash_mode: 2, // forced flash — sliding layers need window masking
            capture_mode: gpu.graphs.capture_mode,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            cache_capacity: sliding_cap,
            head_dim,
            window_size: config.sliding_window as u32,
        };
        let plan = KvTierPlan::derive(tier_inputs)
            .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
        let io = AttnParams {
            q: &scratch.q,
            k: &scratch.k,
            v: &scratch.v,
            k_cache: &kv_cache.k_gpu[kv_layer_idx],
            v_cache: &kv_cache.v_gpu[kv_layer_idx],
            k_scales: None,
            v_scales: None,
            pos_buf: &scratch.pos_buf,
            pos,
            positions: None,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            physical_cap: kv_cache.max_seq,
            cache_capacity: sliding_cap,
            window_size: config.sliding_window as u32,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&scratch.flash_partials),
            givens_cos: kv_cache.givens_cos.as_ref(),
            givens_sin: kv_cache.givens_sin.as_ref(),
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output: &scratch.attn_out,
        };
        execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }
    // Dump attention output regardless of KV cache branch.
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after attention", &scratch.attn_out, n_heads * head_dim);
        let data = gpu.download_f32(&scratch.attn_out).unwrap_or_default();
        let gqa_ratio = n_heads / n_kv;
        for h in 0..n_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            let h_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
            eprintln!("[v1]   head {:2} (kv={}): sum={:+10.4}  first2=[{:.4}, {:.4}]",
                h, h / gqa_ratio, h_sum, data[start], data[start + 1]);
        }
    }

    // o_proj → tmp (reuse tmp, overwriting input_layernorm output).
    let wr_o = lw.o_proj.dispatch_ref();
    execute_steps(gpu, &ctx, &[
        Step::Gemv { w: &wr_o, input: GemvInput::Raw(&scratch.attn_out), out: &scratch.tmp },
    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after o_proj", &scratch.tmp, dim); }

    // Sandwich post-attn norm (in-place on tmp).
    gpu.rmsnorm_f32(&scratch.tmp, &lw.post_attention_layernorm, &scratch.tmp, config.norm_eps)?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after post_attn_norm", &scratch.tmp, dim); }

    // x = residual + tmp. (Reset x first since earlier ops mutated it.)
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after attn_residual", &scratch.x, dim); }

    // residual = x (for the FFN residual stream).
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }

    // Pre-FFN norm.
    gpu.rmsnorm_f32(&scratch.x, &lw.pre_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after pre_ffn_norm", &scratch.tmp, dim); }

    // SwiGLU(gelu_pytorch_tanh): gate_proj, up_proj, gelu_tanh(gate) * up → down_proj.
    let wr_gate = lw.gate_proj.dispatch_ref();
    let wr_up = lw.up_proj.dispatch_ref();
    execute_steps(gpu, &ctx, &[
        Step::Gemv { w: &wr_gate, input: GemvInput::Raw(&scratch.tmp), out: &scratch.gate_ffn },
        Step::Gemv { w: &wr_up, input: GemvInput::Raw(&scratch.tmp), out: &scratch.up_ffn },
    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after gate_proj", &scratch.gate_ffn, config.hidden_dim);
        dbg_dump(gpu, "[v1] L0 after up_proj", &scratch.up_ffn, config.hidden_dim);
    }
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after gelu*up", &scratch.ffn_hidden, config.hidden_dim); }
    let wr_down = lw.down_proj.dispatch_ref();
    execute_steps(gpu, &ctx, &[
        Step::Gemv { w: &wr_down, input: GemvInput::Raw(&scratch.ffn_hidden), out: &scratch.ffn_out },
    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if _dump_on { dbg_dump(gpu, "[v1] L0 after down_proj", &scratch.ffn_out, dim); }

    // Batched prefill hand-off point: caller wants to batch the MoE branch
    // across N tokens. Return now with scratch.ffn_out + scratch.residual
    // populated; caller assembles the batched MoE call externally.
    if stop_before_moe { return Ok(()); }

    // Sandwich post-FFN norm. On MoE layers (26B-A4B) this is folded into
    // apply_moe_branch (which adds the parallel MoE branch + sandwich norms
    // 1 and 2 before this outer norm); on dense layers we just call the
    // standalone post_feedforward_layernorm.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(gpu, config, scratch, moe,
            &lw.post_feedforward_layernorm, &scratch.residual)?,
        _ => gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm,
            &scratch.tmp, config.norm_eps)?,
    }

    // x = residual + tmp (again, reset x from saved residual).
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
}

/// Single full (global) attention layer with K=V weight sharing.
///
/// Key differences from sliding:
///   • head_dim = 512 (global_head_dim), 4 KV heads (vs sliding's 256 / 16).
///   • V is the *pre*-k_norm output of k_proj — CRITICAL ordering (line 1214
///     of modeling_gemma4.py). In Python:
///         key_states = k_proj(x)
///         value_states = v_proj(x) if v_proj else key_states   # bound BEFORE norm
///         key_states   = k_norm(key_states)                    # rebind, value_states holds pre-norm
///         value_states = v_norm(value_states)
///     Our translation: write k_proj output into `scratch.k`, memcpy into
///     `scratch.v`, then apply k_norm in-place on scratch.k.
///   • v_norm is `no_scale=true` RMSNorm — divide only, no learned gain.
///     We call the existing `rmsnorm_batched` with the ones-filled
///     `scratch.v_norm_ones_full` as the weight vector.
///   • RoPE is partial_rotary_factor=0.25 proportional:
///     pairs (i, i+head_dim/2) for i in [0, 64) rotate with theta=1e6;
///     pairs [64, 256) are NoPE (identity). See `rope_partial_halved_f32`.
///   • No sliding window (window_size=0 = full causal).
///   • Attention scale = 1.0 (same as sliding — Gemma 4 sets
///     `self.scaling = 1.0`; we compensate by pre-scaling Q by sqrt(head_dim)).
fn full_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    full_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, false)
}

fn full_layer_attn_ffn_only(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    full_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, true)
}

fn full_layer_decode_impl(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
    stop_before_moe: bool,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.full_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.full_n_kv_heads;
    let dim_bytes = dim * 4;
    let ctx = DispatchCtx::new(gpu);
    let kv_bytes = n_kv * head_dim * 4;

    // residual = x
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(&scratch.x, &lw.input_layernorm, &scratch.tmp, config.norm_eps)?;

    let _fdump = std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") && pos == 1 && kv_layer_idx == 0;
    if _fdump { dbg_dump(gpu, "[FL] L5 input scratch.x", &scratch.x, dim); }
    if _fdump { dbg_dump(gpu, "[FL] L5 post_input_norm", &scratch.tmp, dim); }

    // Q + K projections. V is derived from K's pre-norm output below.
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;
    if _fdump { dbg_dump(gpu, "[FL] L5 post_q_proj", &scratch.q, n_heads * head_dim); }
    if _fdump { dbg_dump(gpu, "[FL] L5 post_k_proj", &scratch.k, n_kv * head_dim); }

    // CRITICAL: capture pre-k_norm bytes as V before applying k_norm.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.v.buf, 0, &scratch.k.buf, 0, kv_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.v.buf, &scratch.k.buf, kv_bytes)?;
    }

    // q_norm, k_norm, and no-scale v_norm (all head_dim = 512).
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.v, &scratch.v_norm_ones_full, &scratch.v,
        n_kv, head_dim, config.norm_eps)?;
    if _fdump { dbg_dump(gpu, "[FL] L5 post_q_norm", &scratch.q, n_heads * head_dim); }
    if _fdump { dbg_dump(gpu, "[FL] L5 post_k_norm", &scratch.k, n_kv * head_dim); }
    if _fdump { dbg_dump(gpu, "[FL] L5 post_v_norm", &scratch.v, n_kv * head_dim); }

    // Pre-scale Q by sqrt(head_dim=512) so the flash kernel's 1/sqrt(head_dim)
    // cancels (Gemma 4 attention scaling is 1.0).
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // Proportional RoPE: rotate_half of the first 64 pairs of every 512-dim head.
    let n_rot_pairs = ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
    gpu.rope_partial_halved_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv, head_dim, n_rot_pairs, config.full_rope_theta)?;
    if _fdump { dbg_dump(gpu, "[FL] L5 post_rope_q", &scratch.q, n_heads * head_dim); }
    if _fdump { dbg_dump(gpu, "[FL] L5 post_rope_k", &scratch.k, n_kv * head_dim); }

    // KV cache write + attention via dispatch framework (Step::Attend).
    // Full-attention layers: window_size=0 (full causal), cache_capacity=0.
    {
        let tier_inputs = KvTierInputs {
            quant_asym4: kv_cache.quant_asym4,
            quant_asym3: kv_cache.quant_asym3,
            quant_asym2: kv_cache.quant_asym2,
            quant_q8: kv_cache.quant_q8,
            quant_fwht: kv_cache.quant_fwht,
            quant_hfq4: false,
            quant_q4: false,
            v_mode_bits: kv_cache.v_mode_bits(),
            pos,
            flash_mode: 2,
            capture_mode: gpu.graphs.capture_mode,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            cache_capacity: 0,
            head_dim,
            window_size: 0,
        };
        let plan = KvTierPlan::derive(tier_inputs)
            .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
        let io = AttnParams {
            q: &scratch.q,
            k: &scratch.k,
            v: &scratch.v,
            k_cache: &kv_cache.k_gpu[kv_layer_idx],
            v_cache: &kv_cache.v_gpu[kv_layer_idx],
            k_scales: None,
            v_scales: None,
            pos_buf: &scratch.pos_buf,
            pos,
            positions: None,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            physical_cap: kv_cache.max_seq,
            cache_capacity: 0,
            window_size: 0,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&scratch.flash_partials),
            givens_cos: kv_cache.givens_cos.as_ref(),
            givens_sin: kv_cache.givens_sin.as_ref(),
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output: &scratch.attn_out,
        };
        execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    // o_proj → tmp.
    weight_gemv(gpu, &lw.o_proj, &scratch.attn_out, &scratch.tmp)?;
    if _fdump { dbg_dump(gpu, "[FL] L5 post_attn_out", &scratch.attn_out, n_heads * head_dim); }
    if _fdump { dbg_dump(gpu, "[FL] L5 post_o_proj", &scratch.tmp, dim); }

    // Sandwich post-attn norm.
    gpu.rmsnorm_f32(&scratch.tmp, &lw.post_attention_layernorm, &scratch.tmp, config.norm_eps)?;

    // x = residual + tmp.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Save new residual.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }

    // Pre-FFN norm.
    gpu.rmsnorm_f32(&scratch.x, &lw.pre_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    // SwiGLU with gelu_pytorch_tanh activation.
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Batched prefill hand-off (see sliding_layer_decode_impl for rationale).
    if stop_before_moe { return Ok(()); }

    // Sandwich post-FFN norm. Same MoE dispatch as sliding_layer_decode.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(gpu, config, scratch, moe,
            &lw.post_feedforward_layernorm, &scratch.residual)?,
        _ => gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm,
            &scratch.tmp, config.norm_eps)?,
    }

    // x = residual + tmp.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
}

/// Token-batched prefill. Processes up to `scratch.max_prefill_batch` tokens
/// at once, amortizing MoE-branch launch overhead across the batch via the
/// batched-indexed kernels (router GEMM, top-K, gate_up, gelu*mul, down).
///
/// Per-token operations (attention, dense projections, RoPE, KV writes)
/// stay per-token in V1 — no batched flash-prefill kernel for asym3
/// sliding window. The win comes from collapsing N per-token MoE calls
/// (~10 launches each) into one batched MoE call per layer.
///
/// On exit: `kv_sliding` / `kv_full` are updated for every token. The KV
/// cache is the source of truth for downstream sampling; this function
/// does NOT compute logits (the caller should follow with a per-token
/// `forward_scratch` for the LAST token if it needs logits).
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    // v2 — batched dense projections + batched MoE. The +55% prefill win
    // from 521161f8 is back: the regressing bug was in `gemm_hfq4g128`'s
    // partial-trailing-group handling (used floor instead of ceil for
    // groups_per_row, silently dropped 64 input dims at K=2112 dense
    // down_proj → garbage). Fixed in kernels/src/gemm_hfq4g128.hip.
    forward_prefill_batch_v2(gpu, weights, config, tokens, start_pos, kv_sliding, kv_full, scratch)
}

fn forward_prefill_batch_v1(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let n_batch = tokens.len();
    if n_batch == 0 { return Ok(()); }
    if n_batch > scratch.max_prefill_batch {
        return Err(hip_bridge::HipError::new(0, &format!(
            "forward_prefill_batch: n_batch={n_batch} > max_prefill_batch={}; \
             caller must chunk", scratch.max_prefill_batch)));
    }
    let dim = config.dim;
    let dim_bytes = dim * 4;

    // ── Step 1: per-token embed + scale into pb_residual[i]. ─────────────
    for (i, &tok) in tokens.iter().enumerate() {
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, tok, dim)?,
            EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, tok, dim)?,
            EmbeddingFormat::Q8_0    => gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, tok, dim)?,
            EmbeddingFormat::F32     => gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, tok, dim)?,
            _ => return Err(hip_bridge::HipError::new(0, "unsupported Gemma 4 embed format")),
        }
        gpu.scale_f32(&scratch.x, config.embed_scale)?;
        // pb_residual[i] = scratch.x
        let stream = gpu.active_stream.as_ref();
        if let Some(s) = stream {
            gpu.hip.memcpy_dtod_async_at(&scratch.pb_residual.buf, i * dim_bytes,
                &scratch.x.buf, 0, dim_bytes, s)?;
        } else {
            gpu.hip.memcpy_dtod_at(&scratch.pb_residual.buf, i * dim_bytes, &scratch.x.buf, 0, dim_bytes)?;
        }
    }

    // ── Step 2: layer loop. Each layer: per-token attn+FFN, then batched MoE. ──
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        // Stage A — per-token attn + FFN. Fills pb_attn_out[i] = post-attn x,
        // pb_ffn_out[i] = dense FFN out. Also writes the per-token KV at
        // position start_pos+i.
        let stream_present = gpu.active_stream.is_some();
        for i in 0..n_batch {
            let pos = start_pos + i;
            let pos_i32 = pos as i32;
            gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            // Load this token's residual into scratch.x.
            if stream_present {
                let s = gpu.active_stream.as_ref().unwrap();
                gpu.hip.memcpy_dtod_async_at(&scratch.x.buf, 0,
                    &scratch.pb_residual.buf, i * dim_bytes, dim_bytes, s)?;
            } else {
                gpu.hip.memcpy_dtod_at(&scratch.x.buf, 0, &scratch.pb_residual.buf, i * dim_bytes, dim_bytes)?;
            }
            match (layer_type, &weights.layers[layer_idx]) {
                (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                    sliding_layer_attn_ffn_only(gpu, config, lw, pos, kv_sliding,
                        sliding_kv_idx, scratch)?;
                }
                (LayerType::Full, LayerWeights::Full(lw)) => {
                    full_layer_attn_ffn_only(gpu, config, lw, pos, kv_full,
                        full_kv_idx, scratch)?;
                }
                _ => return Err(hip_bridge::HipError::new(0, &format!(
                    "Gemma 4 layer {} type/weights mismatch", layer_idx))),
            }
            // Copy outputs into batch slots:
            //   pb_attn_out[i] = scratch.residual (post-attn x; input to MoE pre2/router_in)
            //   pb_ffn_out [i] = scratch.ffn_out  (dense FFN output)
            //   pb_residual[i] = scratch.residual (running residual)
            if stream_present {
                let s = gpu.active_stream.as_ref().unwrap();
                gpu.hip.memcpy_dtod_async_at(&scratch.pb_attn_out.buf, i * dim_bytes,
                    &scratch.residual.buf, 0, dim_bytes, s)?;
                gpu.hip.memcpy_dtod_async_at(&scratch.pb_ffn_out.buf, i * dim_bytes,
                    &scratch.ffn_out.buf, 0, dim_bytes, s)?;
                gpu.hip.memcpy_dtod_async_at(&scratch.pb_residual.buf, i * dim_bytes,
                    &scratch.residual.buf, 0, dim_bytes, s)?;
            } else {
                gpu.hip.memcpy_dtod_at(&scratch.pb_attn_out.buf, i * dim_bytes, &scratch.residual.buf, 0, dim_bytes)?;
                gpu.hip.memcpy_dtod_at(&scratch.pb_ffn_out.buf, i * dim_bytes, &scratch.ffn_out.buf, 0, dim_bytes)?;
                gpu.hip.memcpy_dtod_at(&scratch.pb_residual.buf, i * dim_bytes, &scratch.residual.buf, 0, dim_bytes)?;
            }
        }
        match layer_type {
            LayerType::Sliding => sliding_kv_idx += 1,
            LayerType::Full => full_kv_idx += 1,
        }

        // Stage B — batched MoE branch (or dense post-FFN on layers without MoE).
        let layer_scalar = match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => lw.layer_scalar_host,
            (LayerType::Full, LayerWeights::Full(lw)) => lw.layer_scalar_host,
            _ => unreachable!(),
        };
        let (moe_opt, post_ffn_norm) = match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) =>
                (lw.moe.as_ref(), &lw.post_feedforward_layernorm),
            (LayerType::Full, LayerWeights::Full(lw)) =>
                (lw.moe.as_ref(), &lw.post_feedforward_layernorm),
            _ => unreachable!(),
        };
        match moe_opt {
            Some(moe) => {
                apply_moe_branch_batched(gpu, config, scratch, moe,
                    post_ffn_norm, n_batch)?;
                // Stage C — batched residual add + layer scale.
                // pb_residual[0..N×dim] += pb_moe_pre2[0..N×dim]
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_moe_pre2)?;
                gpu.scale_f32(&scratch.pb_residual, layer_scalar)?;
            }
            None => {
                // Dense path — fall back to per-token finalization to keep
                // semantics identical to single-token forward.
                for i in 0..n_batch {
                    if stream_present {
                        let s = gpu.active_stream.as_ref().unwrap();
                        gpu.hip.memcpy_dtod_async_at(&scratch.x.buf, 0,
                            &scratch.pb_residual.buf, i * dim_bytes, dim_bytes, s)?;
                        gpu.hip.memcpy_dtod_async_at(&scratch.residual.buf, 0,
                            &scratch.pb_residual.buf, i * dim_bytes, dim_bytes, s)?;
                        gpu.hip.memcpy_dtod_async_at(&scratch.ffn_out.buf, 0,
                            &scratch.pb_ffn_out.buf, i * dim_bytes, dim_bytes, s)?;
                    }
                    gpu.rmsnorm_f32(&scratch.ffn_out, post_ffn_norm,
                        &scratch.tmp, config.norm_eps)?;
                    if stream_present {
                        let s = gpu.active_stream.as_ref().unwrap();
                        gpu.hip.memcpy_dtod_async_at(&scratch.x.buf, 0,
                            &scratch.residual.buf, 0, dim_bytes, s)?;
                    }
                    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;
                    gpu.scale_f32(&scratch.x, layer_scalar)?;
                    if stream_present {
                        let s = gpu.active_stream.as_ref().unwrap();
                        gpu.hip.memcpy_dtod_async_at(&scratch.pb_residual.buf, i * dim_bytes,
                            &scratch.x.buf, 0, dim_bytes, s)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// V2 batched prefill — batches both the dense projections AND the MoE
/// branch across N tokens. Per-token RoPE and attention stay sequential
/// because the flash kernels haven't been ported to prefill-batched layout.
///
/// Fixed 2026-05-19 by the gemm_hfq4g128 partial-trailing-group fix
/// (kernels/src/gemm_hfq4g128.hip). The v2 dispatch path is byte-identical
/// to v1 through attention; the divergence was the missing 64-element
/// partial group in the dense down_proj GEMM at K=2112.
fn forward_prefill_batch_v2(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let n_batch = tokens.len();
    if n_batch == 0 { return Ok(()); }
    if n_batch > scratch.max_prefill_batch {
        return Err(hip_bridge::HipError::new(0, &format!(
            "forward_prefill_batch: n_batch={n_batch} > max_prefill_batch={}",
            scratch.max_prefill_batch)));
    }
    let dim = config.dim;
    let dim_bytes = dim * 4;

    // Upload positions array [start_pos, start_pos+1, ..., start_pos+N-1] for batched RoPE.
    let pos_array: Vec<i32> = (0..n_batch).map(|i| (start_pos + i) as i32).collect();
    let pos_bytes: Vec<u8> = pos_array.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&scratch.pb_positions.buf, &pos_bytes)?;

    // Step 1: per-token embed lookup into pb_residual.
    for (i, &tok) in tokens.iter().enumerate() {
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, tok, dim)?,
            EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, tok, dim)?,
            EmbeddingFormat::Q8_0    => gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, tok, dim)?,
            EmbeddingFormat::F32     => gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, tok, dim)?,
            _ => return Err(hip_bridge::HipError::new(0, "unsupported Gemma 4 embed format")),
        }
        gpu.scale_f32(&scratch.x, config.embed_scale)?;
        if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(&scratch.pb_residual.buf, i * dim_bytes, &scratch.x.buf, 0, dim_bytes, _s)?;
                    } else {
                        gpu.hip.memcpy_dtod_at(&scratch.pb_residual.buf, i * dim_bytes, &scratch.x.buf, 0, dim_bytes)?;
                    }
    }

    // Step 2: layer loop.
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        // Invalidate the pointer-keyed F16 conversion cache between layers.
        // The same activation buffer (pb_tmp) is reused with different contents
        // each layer; without invalidation, ensure_fp16_x would skip the
        // F32→F16 conversion and serve stale data from the previous layer.
        gpu.invalidate_fp16_cache();
        let (layer_scalar, post_ffn_norm_ref, moe_opt) =
            match (layer_type, &weights.layers[layer_idx]) {
                (LayerType::Sliding, LayerWeights::Sliding(lw)) =>
                    (lw.layer_scalar_host, &lw.post_feedforward_layernorm, lw.moe.as_ref()),
                (LayerType::Full, LayerWeights::Full(lw)) =>
                    (lw.layer_scalar_host, &lw.post_feedforward_layernorm, lw.moe.as_ref()),
                _ => return Err(hip_bridge::HipError::new(0,
                    &format!("layer {layer_idx} type/weights mismatch"))),
            };

        // ── 2A: batched pre-attn rmsnorm + Q/K/V projections + Q/K/V norms + RoPE. ──
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                let head_dim = config.sliding_head_dim;
                let n_heads = config.n_heads;
                let n_kv = config.sliding_n_kv_heads;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv * head_dim;
                let kv_dim_bytes = kv_dim * 4;
                let q_dim_bytes = q_dim * 4;

                let _dump_on = layer_idx == 0 && start_pos == 0;
                if _dump_on { dbg_dump(gpu, "[v2] L0 input pb_residual[0]", &scratch.pb_residual, dim); }
                gpu.rmsnorm_batched(&scratch.pb_residual, &lw.input_layernorm,
                    &scratch.pb_tmp, n_batch, dim, config.norm_eps)?;
                if _dump_on { dbg_dump(gpu, "[v2] L0 after input_norm", &scratch.pb_tmp, dim); }
                run_prefill_gemm(gpu, &lw.q_proj, &scratch.pb_tmp, &scratch.pb_q, n_batch)?;
                run_prefill_gemm(gpu, &lw.k_proj, &scratch.pb_tmp, &scratch.pb_k, n_batch)?;
                run_prefill_gemm(gpu, &lw.v_proj, &scratch.pb_tmp, &scratch.pb_v, n_batch)?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after q_proj", &scratch.pb_q, q_dim);
                    dbg_dump(gpu, "[v2] L0 after k_proj", &scratch.pb_k, kv_dim);
                    dbg_dump(gpu, "[v2] L0 after v_proj", &scratch.pb_v, kv_dim);
                }
                gpu.rmsnorm_batched(&scratch.pb_q, &lw.q_norm, &scratch.pb_q,
                    n_batch * n_heads, head_dim, config.norm_eps)?;
                gpu.rmsnorm_batched(&scratch.pb_k, &lw.k_norm, &scratch.pb_k,
                    n_batch * n_kv, head_dim, config.norm_eps)?;
                gpu.rmsnorm_batched(&scratch.pb_v, &scratch.v_norm_ones_full, &scratch.pb_v,
                    n_batch * n_kv, head_dim, config.norm_eps)?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after q_norm", &scratch.pb_q, q_dim);
                    dbg_dump(gpu, "[v2] L0 after k_norm", &scratch.pb_k, kv_dim);
                    dbg_dump(gpu, "[v2] L0 after v_norm", &scratch.pb_v, kv_dim);
                }
                gpu.scale_f32(&scratch.pb_q, (head_dim as f32).sqrt())?;
                if _dump_on { dbg_dump(gpu, "[v2] L0 after scale_q", &scratch.pb_q, q_dim); }
                gpu.rope_batched_f32(&scratch.pb_q, &scratch.pb_k, &scratch.pb_positions,
                    n_heads, n_kv, head_dim, config.sliding_rope_theta, n_batch)?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after rope_q", &scratch.pb_q, q_dim);
                    dbg_dump(gpu, "[v2] L0 after rope_k", &scratch.pb_k, kv_dim);
                }

                // Per-token KV write + attention (batched projections, per-token
                // attention). The batched q8 attention path has a bug that
                // corrupts KV when cache_capacity > 0 (ring-buffer mode), so
                // we run attention per-token through the proven decode-path
                // Step::Attend with batch_size=1. This is only 1.7% of GPU
                // time per the rocprof profile, so the overhead is minimal.
                let sliding_cap = config.sliding_window as u32;
                for i in 0..n_batch {
                    let pos = start_pos + i;
                    if let Some(stream) = gpu.active_stream.as_ref() {
                        gpu.hip.stream_write_value32(stream, &scratch.pos_buf, pos as u32, 0)?;
                    } else {
                        gpu.hip.memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(&scratch.q.buf, 0, &scratch.pb_q.buf, i * q_dim_bytes, q_dim_bytes, _s)?;
                        gpu.hip.memcpy_dtod_async_at(&scratch.k.buf, 0, &scratch.pb_k.buf, i * kv_dim_bytes, kv_dim_bytes, _s)?;
                        gpu.hip.memcpy_dtod_async_at(&scratch.v.buf, 0, &scratch.pb_v.buf, i * kv_dim_bytes, kv_dim_bytes, _s)?;
                    } else {
                        gpu.hip.memcpy_dtod_at(&scratch.q.buf, 0, &scratch.pb_q.buf, i * q_dim_bytes, q_dim_bytes)?;
                        gpu.hip.memcpy_dtod_at(&scratch.k.buf, 0, &scratch.pb_k.buf, i * kv_dim_bytes, kv_dim_bytes)?;
                        gpu.hip.memcpy_dtod_at(&scratch.v.buf, 0, &scratch.pb_v.buf, i * kv_dim_bytes, kv_dim_bytes)?;
                    }
                    let tier_inputs = KvTierInputs {
                        quant_asym4: kv_sliding.quant_asym4,
                        quant_asym3: kv_sliding.quant_asym3,
                        quant_asym2: kv_sliding.quant_asym2,
                        quant_q8: kv_sliding.quant_q8,
                        quant_fwht: kv_sliding.quant_fwht,
                        quant_hfq4: false,
                        quant_q4: false,
                        v_mode_bits: kv_sliding.v_mode_bits(),
                        pos,
                        flash_mode: 2,
                        capture_mode: gpu.graphs.capture_mode,
                        batch_size: 1,
                        is_tree: false,
                        is_boundary: false,
                        cache_capacity: sliding_cap,
                        head_dim,
                        window_size: sliding_cap,
                    };
                    let plan = KvTierPlan::derive(tier_inputs)
                        .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
                    let io = AttnParams {
                        q: &scratch.q,
                        k: &scratch.k,
                        v: &scratch.v,
                        k_cache: &kv_sliding.k_gpu[sliding_kv_idx],
                        v_cache: &kv_sliding.v_gpu[sliding_kv_idx],
                        k_scales: None,
                        v_scales: None,
                        pos_buf: &scratch.pos_buf,
                        pos,
                        positions: None,
                        n_heads,
                        n_kv_heads: n_kv,
                        head_dim,
                        physical_cap: kv_sliding.max_seq,
                        cache_capacity: plan.cache_capacity,
                        window_size: plan.window_size,
                        batch_size: 1,
                        max_ctx_len: 0,
                        flash_partials: Some(&scratch.flash_partials),
                        givens_cos: kv_sliding.givens_cos.as_ref(),
                        givens_sin: kv_sliding.givens_sin.as_ref(),
                        tree_bias: None,
                        block_start: 0,
                        block_cols: 0,
                        output: &scratch.attn_out,
                    };
                    let ctx = DispatchCtx::new(gpu);
                    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    // Copy attention output back to batched buffer.
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(&scratch.pb_q.buf, i * q_dim_bytes, &scratch.attn_out.buf, 0, q_dim_bytes, _s)?;
                    } else {
                        gpu.hip.memcpy_dtod_at(&scratch.pb_q.buf, i * q_dim_bytes, &scratch.attn_out.buf, 0, q_dim_bytes)?;
                    }
                }
                sliding_kv_idx += 1;
                if _dump_on { dbg_dump(gpu, "[v2] L0 after attention (pb_q)", &scratch.pb_q, q_dim); }

                run_prefill_gemm(gpu, &lw.o_proj, &scratch.pb_q, &scratch.pb_attn_out, n_batch)?;
                if _dump_on { dbg_dump(gpu, "[v2] L0 after o_proj", &scratch.pb_attn_out, dim); }
                gpu.rmsnorm_batched(&scratch.pb_attn_out, &lw.post_attention_layernorm,
                    &scratch.pb_attn_out, n_batch, dim, config.norm_eps)?;
                if _dump_on { dbg_dump(gpu, "[v2] L0 after post_attn_norm", &scratch.pb_attn_out, dim); }
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_attn_out)?;
                if _dump_on { dbg_dump(gpu, "[v2] L0 after attn_residual", &scratch.pb_residual, dim); }
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                let head_dim = config.full_head_dim;
                let n_heads = config.n_heads;
                let n_kv = config.full_n_kv_heads;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv * head_dim;
                let kv_dim_bytes = kv_dim * 4;
                let q_dim_bytes = q_dim * 4;

                gpu.rmsnorm_batched(&scratch.pb_residual, &lw.input_layernorm,
                    &scratch.pb_tmp, n_batch, dim, config.norm_eps)?;
                run_prefill_gemm(gpu, &lw.q_proj, &scratch.pb_tmp, &scratch.pb_q, n_batch)?;
                run_prefill_gemm(gpu, &lw.k_proj, &scratch.pb_tmp, &scratch.pb_k, n_batch)?;
                if let Some(s) = gpu.active_stream.as_ref() {
                    gpu.hip.memcpy_dtod_async_at(&scratch.pb_v.buf, 0,
                        &scratch.pb_k.buf, 0, n_batch * kv_dim_bytes, s)?;
                } else {
                    gpu.hip.memcpy_dtod_at(&scratch.pb_v.buf, 0,
                        &scratch.pb_k.buf, 0, n_batch * kv_dim_bytes)?;
                }
                gpu.rmsnorm_batched(&scratch.pb_q, &lw.q_norm, &scratch.pb_q,
                    n_batch * n_heads, head_dim, config.norm_eps)?;
                gpu.rmsnorm_batched(&scratch.pb_k, &lw.k_norm, &scratch.pb_k,
                    n_batch * n_kv, head_dim, config.norm_eps)?;
                gpu.rmsnorm_batched(&scratch.pb_v, &scratch.v_norm_ones_full, &scratch.pb_v,
                    n_batch * n_kv, head_dim, config.norm_eps)?;
                gpu.scale_f32(&scratch.pb_q, (head_dim as f32).sqrt())?;
                let n_rot_pairs = ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
                // Per-token partial-halved RoPE (no batched variant yet for this shape).
                for i in 0..n_batch {
                    let pos = start_pos + i;
                    if let Some(stream) = gpu.active_stream.as_ref() {
                        gpu.hip.stream_write_value32(stream, &scratch.pos_buf, pos as u32, 0)?;
                    } else {
                        let pos_i32 = pos as i32;
                        gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(&scratch.q.buf, 0, &scratch.pb_q.buf, i * q_dim_bytes, q_dim_bytes, _s)?;
                    } else {
                        gpu.hip.memcpy_dtod_at(&scratch.q.buf, 0, &scratch.pb_q.buf, i * q_dim_bytes, q_dim_bytes)?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(&scratch.k.buf, 0, &scratch.pb_k.buf, i * kv_dim_bytes, kv_dim_bytes, _s)?;
                    } else {
                        gpu.hip.memcpy_dtod_at(&scratch.k.buf, 0, &scratch.pb_k.buf, i * kv_dim_bytes, kv_dim_bytes)?;
                    }
                    gpu.rope_partial_halved_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
                        n_heads, n_kv, head_dim, n_rot_pairs, config.full_rope_theta)?;
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(&scratch.pb_q.buf, i * q_dim_bytes, &scratch.q.buf, 0, q_dim_bytes, _s)?;
                    } else {
                        gpu.hip.memcpy_dtod_at(&scratch.pb_q.buf, i * q_dim_bytes, &scratch.q.buf, 0, q_dim_bytes)?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(&scratch.pb_k.buf, i * kv_dim_bytes, &scratch.k.buf, 0, kv_dim_bytes, _s)?;
                    } else {
                        gpu.hip.memcpy_dtod_at(&scratch.pb_k.buf, i * kv_dim_bytes, &scratch.k.buf, 0, kv_dim_bytes)?;
                    }
                }

                // Batched full-layer attention (hd=512): all N tokens in ONE
                // masked flash call instead of a per-token loop. Full layers are
                // non-ring (cache_capacity=0) and non-windowed (window_size=0);
                // the asym3 batched-masked kernel derives each query's causal
                // bound from pb_positions, so this reproduces the per-token
                // result with 1 launch/layer instead of N. (Sliding layers stay
                // per-token until the q8 batched ring kernel lands — A.2.)
                // Proportional RoPE was already applied to pb_q/pb_k above.
                let _ = (q_dim_bytes, kv_dim_bytes); // (no per-token staging now)
                let tier_inputs = KvTierInputs {
                    quant_asym4: kv_full.quant_asym4,
                    quant_asym3: kv_full.quant_asym3,
                    quant_asym2: kv_full.quant_asym2,
                    quant_q8: kv_full.quant_q8,
                    quant_fwht: kv_full.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    v_mode_bits: kv_full.v_mode_bits(),
                    pos: start_pos + n_batch - 1,
                    flash_mode: 2,
                    capture_mode: gpu.graphs.capture_mode,
                    batch_size: n_batch,
                    is_tree: false,
                    is_boundary: false,
                    cache_capacity: 0,
                    head_dim,
                    window_size: 0,
                };
                let plan = KvTierPlan::derive(tier_inputs)
                    .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
                let io = AttnParams {
                    q: &scratch.pb_q,
                    k: &scratch.pb_k,
                    v: &scratch.pb_v, // k_eq_v model: V = pre-k_norm k_proj out, v-normed (NOT k-normed/roped pb_k)
                    k_cache: &kv_full.k_gpu[full_kv_idx],
                    v_cache: &kv_full.v_gpu[full_kv_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &scratch.pos_buf,
                    pos: start_pos + n_batch - 1,
                    positions: Some(&scratch.pb_positions),
                    n_heads,
                    n_kv_heads: n_kv,
                    head_dim,
                    physical_cap: kv_full.max_seq,
                    cache_capacity: plan.cache_capacity,
                    window_size: plan.window_size,
                    batch_size: n_batch,
                    max_ctx_len: start_pos + n_batch,
                    flash_partials: Some(&scratch.pb_flash_partials),
                    givens_cos: kv_full.givens_cos.as_ref(),
                    givens_sin: kv_full.givens_sin.as_ref(),
                    tree_bias: None,
                    block_start: 0,
                    block_cols: 0,
                    output: &scratch.pb_attn_q,
                };
                let ctx = DispatchCtx::new(gpu);
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                if std::env::var("HIPFIRE_GEMMA4_ATTN_VERIFY").ok().as_deref() == Some("1") {
                    let batched_out = gpu.download_f32(&scratch.pb_attn_q)?;
                    for i in 0..n_batch {
                        let pos = start_pos + i;
                        gpu.hip.memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
                        gpu.hip.memcpy_dtod_at(&scratch.q.buf, 0, &scratch.pb_q.buf, i * q_dim_bytes, q_dim_bytes)?;
                        gpu.hip.memcpy_dtod_at(&scratch.k.buf, 0, &scratch.pb_k.buf, i * kv_dim_bytes, kv_dim_bytes)?;
                        gpu.hip.memcpy_dtod_at(&scratch.v.buf, 0, &scratch.pb_v.buf, i * kv_dim_bytes, kv_dim_bytes)?;
                        let ti = KvTierInputs {
                            quant_asym4: kv_full.quant_asym4, quant_asym3: kv_full.quant_asym3,
                            quant_asym2: kv_full.quant_asym2, quant_q8: kv_full.quant_q8,
                            quant_fwht: kv_full.quant_fwht, quant_hfq4: false, quant_q4: false,
                            v_mode_bits: kv_full.v_mode_bits(), pos, flash_mode: 2,
                            capture_mode: gpu.graphs.capture_mode, batch_size: 1,
                            is_tree: false, is_boundary: false, cache_capacity: 0,
                            head_dim, window_size: 0,
                        };
                        let p1 = KvTierPlan::derive(ti)
                            .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
                        let io1 = AttnParams {
                            q: &scratch.q, k: &scratch.k, v: &scratch.v,
                            k_cache: &kv_full.k_gpu[full_kv_idx], v_cache: &kv_full.v_gpu[full_kv_idx],
                            k_scales: None, v_scales: None,
                            pos_buf: &scratch.pos_buf, pos, positions: None,
                            n_heads, n_kv_heads: n_kv, head_dim,
                            physical_cap: kv_full.max_seq, cache_capacity: 0, window_size: 0,
                            batch_size: 1, max_ctx_len: 0,
                            flash_partials: Some(&scratch.flash_partials),
                            givens_cos: kv_full.givens_cos.as_ref(), givens_sin: kv_full.givens_sin.as_ref(),
                            tree_bias: None, block_start: 0, block_cols: 0,
                            output: &scratch.attn_out,
                        };
                        let c1 = DispatchCtx::new(gpu);
                        execute_steps(gpu, &c1, &[Step::Attend { plan: p1, io: io1 }])
                            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                        let single = gpu.download_f32(&scratch.attn_out)?;
                        let row = &batched_out[i * q_dim..(i + 1) * q_dim];
                        let mut worst = 0f32; let mut wi = 0usize;
                        for j in 0..q_dim {
                            let d = (single[j] - row[j]).abs();
                            if d > worst { worst = d; wi = j; }
                        }
                        eprintln!("[attn-verify] L{layer_idx} start={start_pos} tok={i} pos={pos} worst={worst:.5} at {wi} single={:.4} batched={:.4}",
                            single[wi], row[wi]);
                    }
                }
                full_kv_idx += 1;

                run_prefill_gemm(gpu, &lw.o_proj, &scratch.pb_attn_q, &scratch.pb_attn_out, n_batch)?;
                gpu.rmsnorm_batched(&scratch.pb_attn_out, &lw.post_attention_layernorm,
                    &scratch.pb_attn_out, n_batch, dim, config.norm_eps)?;
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_attn_out)?;
            }
            _ => unreachable!(),
        }

        // Snapshot post-attn residual for MoE input.
        if let Some(s) = gpu.active_stream.as_ref() {
            gpu.hip.memcpy_dtod_async_at(&scratch.pb_attn_out.buf, 0,
                &scratch.pb_residual.buf, 0, n_batch * dim_bytes, s)?;
        } else {
            gpu.hip.memcpy_dtod_at(&scratch.pb_attn_out.buf, 0,
                &scratch.pb_residual.buf, 0, n_batch * dim_bytes)?;
        }

        // Batched pre-FFN rmsnorm + dense FFN.
        let _dump_ffn = layer_idx == 0 && start_pos == 0;
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                gpu.rmsnorm_batched(&scratch.pb_residual, &lw.pre_feedforward_layernorm,
                    &scratch.pb_tmp, n_batch, dim, config.norm_eps)?;
                if _dump_ffn { dbg_dump(gpu, "[v2] L0 after pre_ffn_norm", &scratch.pb_tmp, dim); }
                run_prefill_gemm(gpu, &lw.gate_proj, &scratch.pb_tmp, &scratch.pb_gate, n_batch)?;
                run_prefill_gemm(gpu, &lw.up_proj, &scratch.pb_tmp, &scratch.pb_up, n_batch)?;
                if _dump_ffn {
                    dbg_dump(gpu, "[v2] L0 after gate_proj", &scratch.pb_gate, config.hidden_dim);
                    dbg_dump(gpu, "[v2] L0 after up_proj", &scratch.pb_up, config.hidden_dim);
                }
                gpu.gelu_tanh_f32(&scratch.pb_gate, &scratch.pb_ffn_hidden,
                    n_batch * config.hidden_dim)?;
                gpu.mul_f32(&scratch.pb_ffn_hidden, &scratch.pb_up, &scratch.pb_ffn_hidden)?;
                if _dump_ffn { dbg_dump(gpu, "[v2] L0 after gelu*up", &scratch.pb_ffn_hidden, config.hidden_dim); }
                run_prefill_gemm(gpu, &lw.down_proj, &scratch.pb_ffn_hidden, &scratch.pb_ffn_out, n_batch)?;
                if _dump_ffn { dbg_dump(gpu, "[v2] L0 after down_proj", &scratch.pb_ffn_out, dim); }
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                gpu.rmsnorm_batched(&scratch.pb_residual, &lw.pre_feedforward_layernorm,
                    &scratch.pb_tmp, n_batch, dim, config.norm_eps)?;
                run_prefill_gemm(gpu, &lw.gate_proj, &scratch.pb_tmp, &scratch.pb_gate, n_batch)?;
                run_prefill_gemm(gpu, &lw.up_proj, &scratch.pb_tmp, &scratch.pb_up, n_batch)?;
                gpu.gelu_tanh_f32(&scratch.pb_gate, &scratch.pb_ffn_hidden,
                    n_batch * config.hidden_dim)?;
                gpu.mul_f32(&scratch.pb_ffn_hidden, &scratch.pb_up, &scratch.pb_ffn_hidden)?;
                run_prefill_gemm(gpu, &lw.down_proj, &scratch.pb_ffn_hidden, &scratch.pb_ffn_out, n_batch)?;
            }
            _ => unreachable!(),
        }

        // MoE branch (or dense fallback). HIPFIRE_MOE_BYPASS=1 forces dense
        // path even on MoE layers (parity with v1 — used to isolate whether
        // a regression lives in apply_moe_branch_batched vs the dense path).
        let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
        match (moe_opt, moe_bypass) {
            (Some(moe), false) => {
                apply_moe_branch_batched(gpu, config, scratch, moe, post_ffn_norm_ref, n_batch)?;
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_moe_pre2)?;
                gpu.scale_f32(&scratch.pb_residual, layer_scalar)?;
            }
            _ => {
                gpu.rmsnorm_batched(&scratch.pb_ffn_out, post_ffn_norm_ref,
                    &scratch.pb_ffn_out, n_batch, dim, config.norm_eps)?;
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_ffn_out)?;
                gpu.scale_f32(&scratch.pb_residual, layer_scalar)?;
            }
        }
        if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") {
            let data = gpu.download_f32(&scratch.pb_residual).unwrap_or_default();
            let last = &data[(n_batch - 1) * dim..n_batch * dim];
            let sum: f64 = last.iter().map(|&v| v as f64).sum();
            let min = last.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = last.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("[v2 diag] L{layer_idx} {:?} last-tok hidden: first4={:?} sum={sum:.4e} min={min:.4} max={max:.4}",
                layer_type, &last[..4.min(last.len())]);
        }
    }

    // ── Final logits: extract last token, run final_norm + lm_head + softcap ──
    // The decode loop samples from scratch.logits, which must be filled
    // with the last token's logits. Copy last position from pb_residual
    // to scratch.tmp, run final_norm, lm_head, and softcap.
    let last_offset = (n_batch - 1) * dim * 4;
    if let Some(_s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(&scratch.tmp.buf, 0, &scratch.pb_residual.buf, last_offset, dim * 4, _s)?;
    } else {
        gpu.hip.memcpy_dtod_at(&scratch.tmp.buf, 0, &scratch.pb_residual.buf, last_offset, dim * 4)?;
    }
    gpu.rmsnorm_f32(&scratch.tmp, &weights.final_norm, &scratch.tmp, config.norm_eps)?;
    let ctx = DispatchCtx::new(gpu);
    let wr_lm = weights.lm_head.dispatch_ref();
    execute_steps(gpu, &ctx, &[
        Step::Gemv { w: &wr_lm, input: GemvInput::Raw(&scratch.tmp), out: &scratch.logits },
    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if config.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&scratch.logits, config.vocab_size, config.final_logit_softcapping)?;
    }

    Ok(())
}

// ── Forward-as-pipeline (#397 Ship 6) — lowered super-op substrate ──────────
//
// Migrates Gemma 4's decode forward from per-token execute_steps resolution
// to pre-resolved LayerPrograms executed via run_layer_program + ForwardBindings.
// Behind HIPFIRE_FORWARD_LOWERED gate (default OFF) until byte-parity validated.
// See docs/plans/gemma4_forward_as_pipeline.md for the full plan.

use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, LayerProgram, OpBinding, SuperOp, SuperOpKind, WeightSlot,
    EscapeKind,
};

// ── Variant enum ──────────────────────────────────────────────────────────

/// The four Gemma 4 decoder-layer shapes. Derived from the layer type
/// (Sliding vs Full) and the presence of MoE extras. Pure → unit-testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gemma4Variant {
    SlidingDense,
    SlidingMoe,
    FullDense,
    FullMoe,
}

fn gemma4_variant_of(layer_type: LayerType, layer: &LayerWeights) -> Gemma4Variant {
    let has_moe = match layer {
        LayerWeights::Sliding(lw) => lw.moe.is_some(),
        LayerWeights::Full(lw) => lw.moe.is_some(),
    };
    match (layer_type, has_moe) {
        (LayerType::Sliding, false) => Gemma4Variant::SlidingDense,
        (LayerType::Sliding, true)  => Gemma4Variant::SlidingMoe,
        (LayerType::Full,    false) => Gemma4Variant::FullDense,
        (LayerType::Full,    true)  => Gemma4Variant::FullMoe,
    }
}

// ── Opcodes ───────────────────────────────────────────────────────────────

/// Arch-local opcodes encoded into `OpBinding.weights[0]` as `WeightSlot(code)`.
/// `SuperOpKind` routes to the `ForwardBindings` method; the opcode disambiguates
/// *which* op of that kind within the layer.
mod g4_op {
    // Proj (projection clusters)
    pub const PROJ_QK_SLIDING: u32 = 0; // sliding: q_proj + k_proj fused (execute_steps)
    pub const PROJ_V_SLIDING:  u32 = 1; // sliding: v_proj standalone (weight_gemv)
    pub const PROJ_Q_FULL:     u32 = 2; // full: q_proj standalone (weight_gemv)
    pub const PROJ_K_FULL:     u32 = 3; // full: k_proj standalone; V←K copied in Attend(FULL)
    pub const PROJ_O:          u32 = 4; // o_proj
    pub const PROJ_GATE_UP:    u32 = 5; // gate_proj + up_proj fused
    pub const PROJ_DOWN:       u32 = 6; // down_proj (folds gelu_tanh + mul activation)

    // Attend
    pub const ATTEND_SLIDING: u32 = 0; // window=1024, head_dim=256, full rope
    pub const ATTEND_FULL: u32 = 1;    // window=0, head_dim=512, partial rope, k_eq_v

    // Norm
    pub const NORM_INPUT:     u32 = 0; // input_layernorm
    pub const NORM_POST_ATTN: u32 = 1; // post_attention_layernorm
    pub const NORM_PRE_FFN:   u32 = 2; // pre_feedforward_layernorm
    pub const NORM_POST_FFN:  u32 = 3; // post_feedforward_layernorm (dense only)

    // ResidualGemv
    pub const RESID_POST_ATTN: u32 = 0; // residual plumbing only (no norm)
    pub const RESID_POST_FFN:  u32 = 1; // residual + scale(layer_scalar)

    // Moe
    pub const MOE_BRANCH: u32 = 0; // encapsulated MoE branch (replaces NORM_POST_FFN)
}

// ── Lowering ──────────────────────────────────────────────────────────────

fn g4_superop(kind: SuperOpKind, code: u32) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: vec![WeightSlot(code)],
            scratch: Vec::new(),
            flavor: superop::OpFlavor::None,
        },
    }
}

fn lower_variant(v: Gemma4Variant) -> LayerProgram {
    use g4_op::*;
    use SuperOpKind::*;
    match v {
        Gemma4Variant::SlidingDense => vec![
            g4_superop(Norm, NORM_INPUT),
            g4_superop(Proj, PROJ_QK_SLIDING),
            g4_superop(Proj, PROJ_V_SLIDING),
            g4_superop(Attend, ATTEND_SLIDING),
            g4_superop(Proj, PROJ_O),
            g4_superop(Norm, NORM_POST_ATTN),
            g4_superop(ResidualGemv, RESID_POST_ATTN),
            g4_superop(Norm, NORM_PRE_FFN),
            g4_superop(Proj, PROJ_GATE_UP),
            g4_superop(Proj, PROJ_DOWN),         // folds gelu_tanh + mul
            g4_superop(Norm, NORM_POST_FFN),     // dense only
            g4_superop(ResidualGemv, RESID_POST_FFN),
        ],
        Gemma4Variant::SlidingMoe => vec![
            g4_superop(Norm, NORM_INPUT),
            g4_superop(Proj, PROJ_QK_SLIDING),
            g4_superop(Proj, PROJ_V_SLIDING),
            g4_superop(Attend, ATTEND_SLIDING),
            g4_superop(Proj, PROJ_O),
            g4_superop(Norm, NORM_POST_ATTN),
            g4_superop(ResidualGemv, RESID_POST_ATTN),
            g4_superop(Norm, NORM_PRE_FFN),
            g4_superop(Proj, PROJ_GATE_UP),
            g4_superop(Proj, PROJ_DOWN),
            // MoE replaces standalone Norm(POST_FFN) — do NOT emit both
            g4_superop(Moe, MOE_BRANCH),
            g4_superop(ResidualGemv, RESID_POST_FFN), // identical to dense path
        ],
        Gemma4Variant::FullDense => vec![
            g4_superop(Norm, NORM_INPUT),
            g4_superop(Proj, PROJ_Q_FULL),
            g4_superop(Proj, PROJ_K_FULL),      // no v_proj; V←K copied in Attend(FULL)
            g4_superop(Attend, ATTEND_FULL),
            g4_superop(Proj, PROJ_O),
            g4_superop(Norm, NORM_POST_ATTN),
            g4_superop(ResidualGemv, RESID_POST_ATTN),
            g4_superop(Norm, NORM_PRE_FFN),
            g4_superop(Proj, PROJ_GATE_UP),
            g4_superop(Proj, PROJ_DOWN),
            g4_superop(Norm, NORM_POST_FFN),
            g4_superop(ResidualGemv, RESID_POST_FFN),
        ],
        Gemma4Variant::FullMoe => vec![
            g4_superop(Norm, NORM_INPUT),
            g4_superop(Proj, PROJ_Q_FULL),
            g4_superop(Proj, PROJ_K_FULL),
            g4_superop(Attend, ATTEND_FULL),
            g4_superop(Proj, PROJ_O),
            g4_superop(Norm, NORM_POST_ATTN),
            g4_superop(ResidualGemv, RESID_POST_ATTN),
            g4_superop(Norm, NORM_PRE_FFN),
            g4_superop(Proj, PROJ_GATE_UP),
            g4_superop(Proj, PROJ_DOWN),
            g4_superop(Moe, MOE_BRANCH),
            g4_superop(ResidualGemv, RESID_POST_FFN),
        ],
    }
}

// ── Bindings struct ───────────────────────────────────────────────────────

fn op_code(op: &OpBinding) -> u32 {
    op.weights.first().map(|w| w.0).unwrap_or(u32::MAX)
}

/// Per-layer execution context for the lowered decode path. Holds the current
/// layer's weights + shared scratch by reference; rebuilt each layer iteration
/// so the borrows stay scoped.
struct Gemma4Bindings<'a> {
    layer: &'a LayerWeights,
    config: &'a Gemma4Config,
    scratch: &'a Gemma4Scratch,
    kv_sliding: &'a mut llama::KvCache,
    kv_full: &'a mut llama::KvCache,
    pos: usize,
    sliding_kv_idx: usize,
    full_kv_idx: usize,
}

impl<'a> ForwardBindings for Gemma4Bindings<'a> {
    fn run_proj(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx, op: &OpBinding) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let s = self.scratch;
        let config = self.config;
        let res: HipResult<()> = match op_code(op) {
            g4_op::PROJ_QK_SLIDING => match self.layer {
                LayerWeights::Sliding(lw) => {
                    let wr_q = lw.q_proj.dispatch_ref();
                    let wr_k = lw.k_proj.dispatch_ref();
                    execute_steps(gpu, ctx, &[
                        Step::Gemv { w: &wr_q, input: GemvInput::Raw(&s.tmp), out: &s.q },
                        Step::Gemv { w: &wr_k, input: GemvInput::Raw(&s.tmp), out: &s.k },
                    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
                }
                _ => Err(hip_bridge::HipError::new(0, "PROJ_QK_SLIDING on non-Sliding layer")),
            },
            g4_op::PROJ_V_SLIDING => match self.layer {
                LayerWeights::Sliding(lw) => weight_gemv(gpu, &lw.v_proj, &s.tmp, &s.v),
                _ => Err(hip_bridge::HipError::new(0, "PROJ_V_SLIDING on non-Sliding layer")),
            },
            g4_op::PROJ_Q_FULL => match self.layer {
                LayerWeights::Full(lw) => weight_gemv(gpu, &lw.q_proj, &s.tmp, &s.q),
                _ => Err(hip_bridge::HipError::new(0, "PROJ_Q_FULL on non-Full layer")),
            },
            g4_op::PROJ_K_FULL => match self.layer {
                LayerWeights::Full(lw) => weight_gemv(gpu, &lw.k_proj, &s.tmp, &s.k),
                _ => Err(hip_bridge::HipError::new(0, "PROJ_K_FULL on non-Full layer")),
            },
            g4_op::PROJ_O => match self.layer {
                LayerWeights::Sliding(lw) => {
                    let wr = lw.o_proj.dispatch_ref();
                    execute_steps(gpu, ctx, &[
                        Step::Gemv { w: &wr, input: GemvInput::Raw(&s.attn_out), out: &s.tmp },
                    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
                }
                LayerWeights::Full(lw) => weight_gemv(gpu, &lw.o_proj, &s.attn_out, &s.tmp),
            },
            g4_op::PROJ_GATE_UP => match self.layer {
                LayerWeights::Sliding(lw) => {
                    let wr_gate = lw.gate_proj.dispatch_ref();
                    let wr_up = lw.up_proj.dispatch_ref();
                    execute_steps(gpu, ctx, &[
                        Step::Gemv { w: &wr_gate, input: GemvInput::Raw(&s.tmp), out: &s.gate_ffn },
                        Step::Gemv { w: &wr_up, input: GemvInput::Raw(&s.tmp), out: &s.up_ffn },
                    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
                }
                LayerWeights::Full(lw) => {
                    let wr_gate = lw.gate_proj.dispatch_ref();
                    let wr_up = lw.up_proj.dispatch_ref();
                    execute_steps(gpu, ctx, &[
                        Step::Gemv { w: &wr_gate, input: GemvInput::Raw(&s.tmp), out: &s.gate_ffn },
                        Step::Gemv { w: &wr_up, input: GemvInput::Raw(&s.tmp), out: &s.up_ffn },
                    ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
                }
            },
            g4_op::PROJ_DOWN => {
                // Folded activation: gelu_tanh(gate) * up, then down_proj.
                let hidden_dim = config.hidden_dim;
                gpu.gelu_tanh_f32(&s.gate_ffn, &s.ffn_hidden, hidden_dim)
                    .map_err(|e| hipfire_dispatch::types::DispatchError::Hip(e.to_string()))?;
                gpu.mul_f32(&s.ffn_hidden, &s.up_ffn, &s.ffn_hidden)
                    .map_err(|e| hipfire_dispatch::types::DispatchError::Hip(e.to_string()))?;
                match self.layer {
                    LayerWeights::Sliding(lw) => {
                        let wr = lw.down_proj.dispatch_ref();
                        execute_steps(gpu, ctx, &[
                            Step::Gemv { w: &wr, input: GemvInput::Raw(&s.ffn_hidden), out: &s.ffn_out },
                        ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
                    }
                    LayerWeights::Full(lw) => weight_gemv(gpu, &lw.down_proj, &s.ffn_hidden, &s.ffn_out),
                }
            }
            other => Err(hip_bridge::HipError::new(0, &format!("unknown PROJ opcode {other}"))),
        };
        res.map_err(|e| hipfire_dispatch::types::DispatchError::Hip(e.to_string()))
    }
    fn run_residual_gemv(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, op: &OpBinding) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let s = self.scratch;
        let dim = self.config.dim;
        let dim_bytes = dim * 4;
        let hip_to_dispatch = |e: hip_bridge::HipError| hipfire_dispatch::types::DispatchError::Hip(e.to_string());

        let res: HipResult<()> = match op_code(op) {
            g4_op::RESID_POST_ATTN => {
                // First: save x → residual (this is the first time residual is set
                // in this layer — Norm(INPUT) wrote to tmp, x is still intact).
                if let Some(stream) = gpu.active_stream.as_ref() {
                    gpu.hip.memcpy_dtod_async_at(&s.residual.buf, 0, &s.x.buf, 0, dim_bytes, stream).map_err(hip_to_dispatch)?;
                } else {
                    gpu.hip.memcpy_dtod(&s.residual.buf, &s.x.buf, dim_bytes).map_err(hip_to_dispatch)?;
                }
                // x = residual + tmp (tmp holds post_attn_norm output).
                if let Some(stream) = gpu.active_stream.as_ref() {
                    gpu.hip.memcpy_dtod_async_at(&s.x.buf, 0, &s.residual.buf, 0, dim_bytes, stream).map_err(hip_to_dispatch)?;
                } else {
                    gpu.hip.memcpy_dtod(&s.x.buf, &s.residual.buf, dim_bytes).map_err(hip_to_dispatch)?;
                }
                gpu.add_inplace_f32(&s.x, &s.tmp).map_err(hip_to_dispatch)?;
                // Save x → residual for the FFN residual stream.
                if let Some(stream) = gpu.active_stream.as_ref() {
                    gpu.hip.memcpy_dtod_async_at(&s.residual.buf, 0, &s.x.buf, 0, dim_bytes, stream).map_err(hip_to_dispatch)?;
                } else {
                    gpu.hip.memcpy_dtod(&s.residual.buf, &s.x.buf, dim_bytes).map_err(hip_to_dispatch)?;
                }
                Ok(())
            }
            g4_op::RESID_POST_FFN => {
                // x = residual + tmp; x *= layer_scalar.
                // Identical for dense and MoE — tmp already holds normalized output.
                if let Some(stream) = gpu.active_stream.as_ref() {
                    gpu.hip.memcpy_dtod_async_at(&s.x.buf, 0, &s.residual.buf, 0, dim_bytes, stream).map_err(hip_to_dispatch)?;
                } else {
                    gpu.hip.memcpy_dtod(&s.x.buf, &s.residual.buf, dim_bytes).map_err(hip_to_dispatch)?;
                }
                gpu.add_inplace_f32(&s.x, &s.tmp).map_err(hip_to_dispatch)?;
                let layer_scalar = match self.layer {
                    LayerWeights::Sliding(lw) => lw.layer_scalar_host,
                    LayerWeights::Full(lw) => lw.layer_scalar_host,
                };
                gpu.scale_f32(&s.x, layer_scalar).map_err(hip_to_dispatch)?;
                Ok(())
            }
            other => Err(hip_bridge::HipError::new(0, &format!("unknown RESID_GEMV opcode {other}"))),
        };
        res.map_err(|e| hipfire_dispatch::types::DispatchError::Hip(e.to_string()))
    }
    fn run_norm(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, op: &OpBinding) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let s = self.scratch;
        let config = self.config;
        let res: HipResult<()> = match op_code(op) {
            g4_op::NORM_INPUT => match self.layer {
                LayerWeights::Sliding(lw) => gpu.rmsnorm_f32(&s.x, &lw.input_layernorm, &s.tmp, config.norm_eps),
                LayerWeights::Full(lw) => gpu.rmsnorm_f32(&s.x, &lw.input_layernorm, &s.tmp, config.norm_eps),
            },
            g4_op::NORM_POST_ATTN => match self.layer {
                LayerWeights::Sliding(lw) => gpu.rmsnorm_f32(&s.tmp, &lw.post_attention_layernorm, &s.tmp, config.norm_eps),
                LayerWeights::Full(lw) => gpu.rmsnorm_f32(&s.tmp, &lw.post_attention_layernorm, &s.tmp, config.norm_eps),
            },
            g4_op::NORM_PRE_FFN => match self.layer {
                LayerWeights::Sliding(lw) => gpu.rmsnorm_f32(&s.x, &lw.pre_feedforward_layernorm, &s.tmp, config.norm_eps),
                LayerWeights::Full(lw) => gpu.rmsnorm_f32(&s.x, &lw.pre_feedforward_layernorm, &s.tmp, config.norm_eps),
            },
            g4_op::NORM_POST_FFN => match self.layer {
                // Dense-only norm. MoE layers use Moe(MOE_BRANCH) instead.
                LayerWeights::Sliding(lw) => gpu.rmsnorm_f32(&s.ffn_out, &lw.post_feedforward_layernorm, &s.tmp, config.norm_eps),
                LayerWeights::Full(lw) => gpu.rmsnorm_f32(&s.ffn_out, &lw.post_feedforward_layernorm, &s.tmp, config.norm_eps),
            },
            other => return Err(hipfire_dispatch::types::DispatchError::Hip(format!("unknown NORM opcode {other}"))),
        };
        res.map_err(|e| hipfire_dispatch::types::DispatchError::Hip(e.to_string()))
    }
    fn run_attend(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, op: &OpBinding) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let s = self.scratch;
        let config = self.config;
        let pos = self.pos;
        let hip_to_dispatch = |e: hip_bridge::HipError| hipfire_dispatch::types::DispatchError::Hip(e.to_string());

        let res: HipResult<()> = match op_code(op) {
            g4_op::ATTEND_SLIDING => {
                let lw = match self.layer {
                    LayerWeights::Sliding(lw) => lw,
                    _ => return Err(hipfire_dispatch::types::DispatchError::Hip("ATTEND_SLIDING on non-Sliding layer".into())),
                };
                let head_dim = config.sliding_head_dim;
                let n_heads = config.n_heads;
                let n_kv = config.sliding_n_kv_heads;

                // q/k/v norms (v_norm is no-scale — ones buffer).
                gpu.rmsnorm_batched(&s.q, &lw.q_norm, &s.q, n_heads, head_dim, config.norm_eps).map_err(hip_to_dispatch)?;
                gpu.rmsnorm_batched(&s.k, &lw.k_norm, &s.k, n_kv, head_dim, config.norm_eps).map_err(hip_to_dispatch)?;
                gpu.rmsnorm_batched(&s.v, &s.v_norm_ones_full, &s.v, n_kv, head_dim, config.norm_eps).map_err(hip_to_dispatch)?;

                // Pre-scale Q by sqrt(head_dim) — Gemma 4 attention scale is 1.0.
                gpu.scale_f32(&s.q, (head_dim as f32).sqrt()).map_err(hip_to_dispatch)?;

                // Full rotate_half RoPE (all dims rotate).
                gpu.rope_f32(&s.q, &s.k, &s.pos_buf,
                    n_heads, n_kv, head_dim, config.sliding_rope_theta).map_err(hip_to_dispatch)?;

                // KV write + flash attention via dispatch.
                let kv = &mut *self.kv_sliding;
                let kv_layer_idx = self.sliding_kv_idx;
                let sliding_cap = kv.physical_cap as u32;
                let ctx = DispatchCtx::new(gpu);
                let tier_inputs = KvTierInputs {
                    quant_asym4: kv.quant_asym4,
                    quant_asym3: kv.quant_asym3,
                    quant_asym2: kv.quant_asym2,
                    quant_q8: kv.quant_q8,
                    quant_fwht: kv.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    v_mode_bits: kv.v_mode_bits(),
                    pos,
                    flash_mode: 2,
                    capture_mode: gpu.graphs.capture_mode,
                    batch_size: 1,
                    is_tree: false,
                    is_boundary: false,
                    cache_capacity: sliding_cap,
                    head_dim,
                    window_size: config.sliding_window as u32,
                };
                let plan = KvTierPlan::derive(tier_inputs)
                    .map_err(|e| hipfire_dispatch::types::DispatchError::Hip(format!("{:?}", e)))?;
                let io = AttnParams {
                    q: &s.q, k: &s.k, v: &s.v,
                    k_cache: &kv.k_gpu[kv_layer_idx],
                    v_cache: &kv.v_gpu[kv_layer_idx],
                    k_scales: None, v_scales: None,
                    pos_buf: &s.pos_buf, pos, positions: None,
                    n_heads, n_kv_heads: n_kv, head_dim,
                    physical_cap: kv.max_seq,
                    cache_capacity: sliding_cap,
                    window_size: config.sliding_window as u32,
                    batch_size: 1, max_ctx_len: 0,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv.givens_cos.as_ref(),
                    givens_sin: kv.givens_sin.as_ref(),
                    tree_bias: None,
                    block_start: 0, block_cols: 0,
                    output: &s.attn_out,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
            }
            g4_op::ATTEND_FULL => {
                let lw = match self.layer {
                    LayerWeights::Full(lw) => lw,
                    _ => return Err(hipfire_dispatch::types::DispatchError::Hip("ATTEND_FULL on non-Full layer".into())),
                };
                let head_dim = config.full_head_dim;
                let n_heads = config.n_heads;
                let n_kv = config.full_n_kv_heads;
                let kv_bytes = n_kv * head_dim * 4;

                // CRITICAL: capture pre-k_norm K as V before applying k_norm.
                if let Some(stream) = gpu.active_stream.as_ref() {
                    gpu.hip.memcpy_dtod_async_at(&s.v.buf, 0, &s.k.buf, 0, kv_bytes, stream).map_err(hip_to_dispatch)?;
                } else {
                    gpu.hip.memcpy_dtod(&s.v.buf, &s.k.buf, kv_bytes).map_err(hip_to_dispatch)?;
                }

                // q/k/v norms (v_norm is no-scale — ones buffer).
                gpu.rmsnorm_batched(&s.q, &lw.q_norm, &s.q, n_heads, head_dim, config.norm_eps).map_err(hip_to_dispatch)?;
                gpu.rmsnorm_batched(&s.k, &lw.k_norm, &s.k, n_kv, head_dim, config.norm_eps).map_err(hip_to_dispatch)?;
                gpu.rmsnorm_batched(&s.v, &s.v_norm_ones_full, &s.v, n_kv, head_dim, config.norm_eps).map_err(hip_to_dispatch)?;

                // Pre-scale Q by sqrt(head_dim).
                gpu.scale_f32(&s.q, (head_dim as f32).sqrt()).map_err(hip_to_dispatch)?;

                // Proportional partial RoPE (only first n_rot_pairs pairs rotate).
                let n_rot_pairs = ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
                gpu.rope_partial_halved_f32(&s.q, &s.k, &s.pos_buf,
                    n_heads, n_kv, head_dim, n_rot_pairs, config.full_rope_theta).map_err(hip_to_dispatch)?;

                // KV write + flash attention via dispatch.
                let kv = &mut *self.kv_full;
                let kv_layer_idx = self.full_kv_idx;
                let ctx = DispatchCtx::new(gpu);
                let tier_inputs = KvTierInputs {
                    quant_asym4: kv.quant_asym4,
                    quant_asym3: kv.quant_asym3,
                    quant_asym2: kv.quant_asym2,
                    quant_q8: kv.quant_q8,
                    quant_fwht: kv.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    v_mode_bits: kv.v_mode_bits(),
                    pos,
                    flash_mode: 2,
                    capture_mode: gpu.graphs.capture_mode,
                    batch_size: 1,
                    is_tree: false,
                    is_boundary: false,
                    cache_capacity: 0,
                    head_dim,
                    window_size: 0,
                };
                let plan = KvTierPlan::derive(tier_inputs)
                    .map_err(|e| hipfire_dispatch::types::DispatchError::Hip(format!("{:?}", e)))?;
                let io = AttnParams {
                    q: &s.q, k: &s.k, v: &s.v,
                    k_cache: &kv.k_gpu[kv_layer_idx],
                    v_cache: &kv.v_gpu[kv_layer_idx],
                    k_scales: None, v_scales: None,
                    pos_buf: &s.pos_buf, pos, positions: None,
                    n_heads, n_kv_heads: n_kv, head_dim,
                    physical_cap: kv.max_seq,
                    cache_capacity: 0,
                    window_size: 0,
                    batch_size: 1, max_ctx_len: 0,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv.givens_cos.as_ref(),
                    givens_sin: kv.givens_sin.as_ref(),
                    tree_bias: None,
                    block_start: 0, block_cols: 0,
                    output: &s.attn_out,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))
            }
            other => Err(hip_bridge::HipError::new(0, &format!("unknown ATTEND opcode {other}"))),
        };
        res.map_err(|e| hipfire_dispatch::types::DispatchError::Hip(e.to_string()))
    }
    fn run_moe(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let moe = match self.layer {
            LayerWeights::Sliding(lw) => lw.moe.as_ref(),
            LayerWeights::Full(lw) => lw.moe.as_ref(),
        };
        let moe = match moe {
            Some(m) => m,
            None => return Err(hipfire_dispatch::types::DispatchError::Hip(
                "MOE_BRANCH on layer without MoE extras".into())),
        };
        let post_ffn_norm = match self.layer {
            LayerWeights::Sliding(lw) => &lw.post_feedforward_layernorm,
            LayerWeights::Full(lw) => &lw.post_feedforward_layernorm,
        };
        apply_moe_branch(gpu, self.config, self.scratch, moe,
            post_ffn_norm, &self.scratch.residual)
            .map_err(|e| hipfire_dispatch::types::DispatchError::Hip(e.to_string()))
    }
    fn run_recurrent(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), hipfire_dispatch::types::DispatchError> {
        Err(hipfire_dispatch::types::DispatchError::Hip("gemma4 has no recurrent ops".into()))
    }
    fn run_conv(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), hipfire_dispatch::types::DispatchError> {
        Err(hipfire_dispatch::types::DispatchError::Hip("gemma4 has no conv ops".into()))
    }
    fn run_escape(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding, _kind: EscapeKind) -> Result<(), hipfire_dispatch::types::DispatchError> {
        Err(hipfire_dispatch::types::DispatchError::Hip("gemma4 escape not yet implemented (Step 6)".into()))
    }
}

// ── Gate + lowered forward ───────────────────────────────────────────────

/// Cached `HIPFIRE_FORWARD_LOWERED` toggle. Default OFF until byte-parity
/// validated. Escape hatch: `HIPFIRE_FORWARD_LOWERED=1` to opt in.
fn forward_lowered_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        // Default ON (byte-parity validated 2026-06-08).
        // Set HIPFIRE_FORWARD_LOWERED=0 to force legacy hand path.
        std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0")
    })
}

/// Lowered (#397 Ship 6) decode layer loop. Behaviorally equivalent to
/// `forward_scratch_inner`'s hand arms. Builds a coarse-super-op `LayerProgram`
/// per layer and runs it through the dispatch substrate's executor.
fn forward_scratch_inner_lowered(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    pos: usize,
    kv_sliding: &mut llama::KvCache,
    kv_full: &mut llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;

    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        let layer = &weights.layers[layer_idx];
        let variant = gemma4_variant_of(layer_type, layer);
        let program = lower_variant(variant);
        {
            let mut bind = Gemma4Bindings {
                layer,
                config,
                scratch,
                kv_sliding,
                kv_full,
                pos,
                sliding_kv_idx,
                full_kv_idx,
            };
            superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        }
        match layer_type {
            LayerType::Sliding => sliding_kv_idx += 1,
            LayerType::Full => full_kv_idx += 1,
        }
    }

    // Output stage: final norm + lm_head + softcap (same as hand path).
    gpu.rmsnorm_f32(&scratch.x, &weights.final_norm, &scratch.tmp, config.norm_eps)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = weights.lm_head.dispatch_ref();
        execute_steps(gpu, &ctx, &[
            Step::Gemv { w: &wr, input: GemvInput::Raw(&scratch.tmp), out: &scratch.logits },
        ]).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }
    if config.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&scratch.logits, config.vocab_size, config.final_logit_softcapping)?;
    }
    Ok(())
}
