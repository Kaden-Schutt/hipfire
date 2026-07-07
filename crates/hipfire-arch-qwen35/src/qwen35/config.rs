// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 model config: layer typing, weight-load modes, and construction
//! from an .hfq container or a safetensors source.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    LinearAttention, // DeltaNet
    FullAttention,   // Standard MHA with gated output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum F16LmHeadMode {
    Native,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bf16WeightLoadMode {
    Auto,
    Native,
    F16,
    F32,
}

pub(crate) fn parse_f16_lm_head_mode(value: Option<&str>) -> F16LmHeadMode {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if matches!(v.as_str(), "0" | "f32" | "fp32" | "legacy") => F16LmHeadMode::F32,
        _ => F16LmHeadMode::Native,
    }
}

pub(crate) fn parse_bf16_weight_load_mode(value: Option<&str>) -> Bf16WeightLoadMode {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Bf16WeightLoadMode::Auto,
        Some(v) if matches!(v.as_str(), "auto" | "") => Bf16WeightLoadMode::Auto,
        Some(v) if matches!(v.as_str(), "native" | "bf16") => Bf16WeightLoadMode::Native,
        Some(v) if matches!(v.as_str(), "f16" | "fp16") => Bf16WeightLoadMode::F16,
        Some(v) if matches!(v.as_str(), "0" | "f32" | "fp32" | "legacy") => Bf16WeightLoadMode::F32,
        _ => Bf16WeightLoadMode::Auto,
    }
}

pub(crate) fn f16_lm_head_mode_from_env() -> F16LmHeadMode {
    let value = std::env::var("HIPFIRE_LM_HEAD_F16").ok();
    parse_f16_lm_head_mode(value.as_deref())
}

pub(crate) fn bf16_weight_load_mode_from_env() -> Bf16WeightLoadMode {
    let value = std::env::var("HIPFIRE_BF16_WEIGHTS").ok();
    parse_bf16_weight_load_mode(value.as_deref())
}

pub(crate) fn bf16_native_weight_arch(arch: &str) -> bool {
    arch.starts_with("gfx11") || arch.starts_with("gfx12")
}

pub(crate) fn resolve_bf16_weight_load_mode(
    requested: Bf16WeightLoadMode,
    arch: &str,
) -> Bf16WeightLoadMode {
    match requested {
        Bf16WeightLoadMode::Auto if bf16_native_weight_arch(arch) => Bf16WeightLoadMode::Native,
        Bf16WeightLoadMode::Auto => Bf16WeightLoadMode::F16,
        other => other,
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub eos_token: u32,

    // Full attention params
    pub n_heads: usize,    // 8
    pub n_kv_heads: usize, // 2
    pub head_dim: usize,   // 256
    pub rope_theta: f32,
    pub partial_rotary_factor: f32, // 0.25 — only 64/256 dims get RoPE
    /// Qwen3.5 FullAttention checkpoints use a doubled Q projection that
    /// interleaves Q and an attention-output gate. Some routed-only Qwen3
    /// artifacts set this false and store plain Q only.
    pub attn_output_gate: bool,
    /// True when a composite Qwen3.5-VL checkpoint is being used as a
    /// text-only model through its nested `text_config`.
    pub is_vl_text: bool,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],

    // DeltaNet params
    pub linear_num_key_heads: usize,   // 16
    pub linear_num_value_heads: usize, // 16
    pub linear_key_head_dim: usize,    // 128
    pub linear_value_head_dim: usize,  // 128
    pub conv_kernel_dim: usize,        // 4

    // FFN — dense; for MoE see num_experts below
    pub hidden_dim: usize, // 3584 (dense) or unused when num_experts > 0

    // MoE (qwen3_5_moe / A3B). num_experts == 0 means plain dense (qwen3_5).
    pub num_experts: usize,                     // 256 for A3B
    pub num_experts_per_tok: usize,             // 8 for A3B
    pub moe_intermediate_size: usize,           // 512 for A3B (per-routed-expert FFN)
    pub shared_expert_intermediate_size: usize, // 512 for A3B
    pub has_shared_expert: bool,                // true for A3B (always-on shared expert)
    /// If true, top-K routing weights are re-normalized to sum to 1 after
    /// softmax + top-K selection. Qwen convention (matches HF
    /// `modeling_qwen3_5_moe.py`). DeepSeek-v1 uses false.
    pub norm_topk_prob: bool,

    // Per-layer type dispatch
    pub layer_types: Vec<LayerType>,

    // ── Weight pager (MAD-93 v0.1) ───────────────────────────────────
    /// If true, MoE expert weights are managed by [`hipfire_runtime::weight_pager::WeightPager`]
    /// and only the active top-k experts per layer are guaranteed resident in
    /// VRAM. Default false (all experts resident, today's behavior).
    ///
    /// Off-switch for the v0.1 PR: when false there is no behavior change
    /// vs main; when true the forward path takes the paged code path which
    /// uses a CPU-side router replica + on-demand H2D transfers.
    pub paged_experts: bool,

    /// Soft cap on VRAM bytes the weight pager is allowed to hold for paged
    /// expert weights. Only meaningful when `paged_experts == true`. Defaults
    /// to `u64::MAX` (no eviction — tested when VRAM is unlimited or we just
    /// want to verify the routing path works without eviction pressure).
    pub vram_budget_bytes: u64,
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Qwen35Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);
    let is_vl_text = config.get("text_config").is_some() && config.get("vision_config").is_some();

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = tc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    // Dense FFN intermediate dim. MoE configs (qwen3_5_moe / A3B) replace this
    // with `moe_intermediate_size` and don't ship `intermediate_size`, so don't
    // hard-fail here — we still need to load the rest of the config to detect
    // is_moe and route accordingly.
    let hidden_dim = tc
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let norm_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;

    let rope_params = tc.get("rope_parameters");
    let rope_theta = rope_params
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000_000.0) as f32;
    let partial_rotary_factor = tc
        .get("partial_rotary_factor")
        .or_else(|| rope_params.and_then(|r| r.get("partial_rotary_factor")))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.25) as f32;
    let q_dim = n_heads * head_dim;
    let metadata_attn_output_gate = tc
        .get("attn_output_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let attn_output_gate =
        infer_attn_output_gate_from_hfq(hfq, n_layers, q_dim).unwrap_or(metadata_attn_output_gate);
    let mrope_interleaved = rope_params
        .and_then(|r| r.get("mrope_interleaved"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut mrope_section = [11usize, 11usize, 10usize];
    if let Some(arr) = rope_params
        .and_then(|r| r.get("mrope_section"))
        .and_then(|v| v.as_array())
    {
        for (dst, src) in mrope_section.iter_mut().zip(arr.iter().take(3)) {
            if let Some(v) = src.as_u64() {
                *dst = v as usize;
            }
        }
    }

    let eos_token = tc
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(248044) as u32;

    let linear_num_key_heads = tc
        .get("linear_num_key_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;
    let linear_num_value_heads = tc
        .get("linear_num_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;
    let linear_key_head_dim = tc
        .get("linear_key_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let linear_value_head_dim = tc
        .get("linear_value_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let conv_kernel_dim = tc
        .get("linear_conv_kernel_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;

    let layer_types: Vec<LayerType> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| match v.as_str().unwrap_or("full_attention") {
                    "linear_attention" => LayerType::LinearAttention,
                    _ => LayerType::FullAttention,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![LayerType::FullAttention; n_layers]);

    // MoE config (zeros = dense fallback). Qwen3.5-MoE / A3B sets these.
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_experts_per_tok = tc
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let moe_intermediate_size = tc
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let shared_expert_intermediate_size = tc
        .get("shared_expert_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let has_shared_expert = shared_expert_intermediate_size > 0;
    // Qwen convention: re-normalize top-K routing weights to sum to 1.
    // Absent from some configs (including the shipped A3B HFQ); default on
    // for Qwen3.5-MoE / A3B to match the HF reference.
    let norm_topk_prob = tc
        .get("norm_topk_prob")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Some(Qwen35Config {
        dim,
        n_layers,
        vocab_size,
        norm_eps,
        eos_token,
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor,
        attn_output_gate,
        is_vl_text,
        mrope_interleaved,
        mrope_section,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        conv_kernel_dim,
        hidden_dim,
        layer_types,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        shared_expert_intermediate_size,
        has_shared_expert,
        norm_topk_prob,
        paged_experts: qwen35_paged_experts_enabled(num_experts),
        vram_budget_bytes: qwen35_expert_cache_budget_bytes(),
    })
}

/// Parse Qwen35Config from a SafetensorsSource (or any ModelSource).
/// Delegates to the same JSON parser as config_from_hfq — the SafetensorsSource
/// builds compatible metadata JSON from config.json.
pub fn config_from_safetensors(source: &dyn ModelSource) -> Option<Qwen35Config> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json()).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = tc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let hidden_dim = tc
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let norm_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let rope_params = tc.get("rope_parameters");
    let rope_theta = rope_params
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000_000.0) as f32;
    let partial_rotary_factor = tc
        .get("partial_rotary_factor")
        .or_else(|| rope_params.and_then(|r| r.get("partial_rotary_factor")))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.25) as f32;
    let attn_output_gate = tc
        .get("attn_output_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let is_vl_text = config.get("text_config").is_some() && config.get("vision_config").is_some();
    let mrope_interleaved = rope_params
        .and_then(|r| r.get("mrope_interleaved"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut mrope_section = [11usize, 11usize, 10usize];
    if let Some(arr) = rope_params
        .and_then(|r| r.get("mrope_section"))
        .and_then(|v| v.as_array())
    {
        for (dst, src) in mrope_section.iter_mut().zip(arr.iter().take(3)) {
            if let Some(v) = src.as_u64() {
                *dst = v as usize;
            }
        }
    }
    let eos_token = tc
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(248044) as u32;
    let linear_num_key_heads = tc
        .get("linear_num_key_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;
    let linear_num_value_heads = tc
        .get("linear_num_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;
    let linear_key_head_dim = tc
        .get("linear_key_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let linear_value_head_dim = tc
        .get("linear_value_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let conv_kernel_dim = tc
        .get("linear_conv_kernel_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;
    let layer_types: Vec<LayerType> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| match v.as_str().unwrap_or("full_attention") {
                    "linear_attention" => LayerType::LinearAttention,
                    _ => LayerType::FullAttention,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![LayerType::FullAttention; n_layers]);
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_experts_per_tok = tc
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let moe_intermediate_size = tc
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let shared_expert_intermediate_size = tc
        .get("shared_expert_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let has_shared_expert = shared_expert_intermediate_size > 0;
    let norm_topk_prob = tc
        .get("norm_topk_prob")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    Some(Qwen35Config {
        dim,
        n_layers,
        vocab_size,
        norm_eps,
        eos_token,
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor,
        attn_output_gate,
        is_vl_text,
        mrope_interleaved,
        mrope_section,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        conv_kernel_dim,
        hidden_dim,
        layer_types,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        shared_expert_intermediate_size,
        has_shared_expert,
        norm_topk_prob,
        paged_experts: qwen35_paged_experts_enabled(num_experts),
        vram_budget_bytes: qwen35_expert_cache_budget_bytes(),
    })
}
