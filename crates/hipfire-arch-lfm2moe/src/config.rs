// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-MoE config, parsed from the HFQ `metadata_json` envelope (which
//! carries the source `config.json` wholesale under the `config` key).
//!
//! Ground truth: LiquidAI/LFM2.5-8B-A1B config.json + transformers
//! `Lfm2MoeConfig`/`modeling_lfm2_moe.py` (read 2026-05-29):
//!   hidden 2048, 24 layers, 32 q-heads / 8 kv-heads, head_dim 64,
//!   layer_types interleave 18 "conv" + 6 "full_attention",
//!   num_dense_layers 2 (dense SwiGLU MLP), the rest top-4 MoE (32 experts,
//!   moe_intermediate 1792, sigmoid+expert_bias routing, norm_topk_prob),
//!   conv_L_cache 3 (depthwise causal short-conv), rope_theta 5e6, eps 1e-5,
//!   standard RMSNorm (weight * x̂, no +1), tie_word_embeddings.

use hipfire_runtime::hfq::HfqFile;
use serde::Deserialize;

/// Per-layer mixer kind, decoded from `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    /// Double-gated LIV short-convolution (depthwise causal, kernel = conv_L_cache).
    Conv,
    /// GQA attention with per-head QK-norm + full-dim rotate_half RoPE.
    Attention,
}

/// Typed LFM2.5-MoE shape constants.
#[derive(Clone, Debug)]
pub struct Lfm2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Short-conv kernel size (HF `conv_L_cache`); decode conv-state holds K-1.
    pub conv_kernel_size: usize,
    /// Dense-MLP FFN intermediate size (HF `intermediate_size`).
    pub intermediate_size: usize,
    /// Expert (MoE) FFN intermediate size (HF `moe_intermediate_size`).
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// The first `num_dense_layers` MLP layers are dense SwiGLU; the rest MoE.
    pub num_dense_layers: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// Renormalize the top-k gathered router weights to sum 1.
    pub norm_topk_prob: bool,
    /// Router adds `expert_bias` for selection only (aux-loss-free routing).
    pub use_expert_bias: bool,
    /// Scale applied to the combined expert output (HF `routed_scaling_factor`).
    pub routed_scaling_factor: f32,
    /// lm_head shares embed_tokens.
    pub tie_word_embeddings: bool,
    /// Per-layer mixer choice (length == num_hidden_layers).
    pub layer_types: Vec<MixerKind>,
}

#[derive(Deserialize)]
struct RawRope {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
}

#[derive(Deserialize)]
struct RawLfm2MoeConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_conv_l")]
    conv_L_cache: usize,
    intermediate_size: usize,
    // MoE fields — present on lfm2_moe (A1B), ABSENT on dense lfm2 (350M/1.2B).
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    num_dense_layers: usize,
    // LFM2 SwiGLU FFN auto-adjustment (LLaMA-style): the REAL dense FFN dim is
    // round_to(block_multiple_of, multiplier * 2/3 * block_ff_dim), NOT the raw
    // intermediate_size. e.g. 350M: round_256(2/3 * 6656) = 4608, not 6656.
    #[serde(default)]
    block_auto_adjust_ff_dim: bool,
    #[serde(default = "default_multiple_of")]
    block_multiple_of: usize,
    #[serde(default)]
    block_ffn_dim_multiplier: Option<f32>,
    #[serde(default)]
    block_ff_dim: Option<usize>,
    #[serde(default)]
    rope_parameters: Option<RawRope>,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_eps")]
    norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default = "default_true")]
    use_expert_bias: bool,
    #[serde(default = "default_routed_scale")]
    routed_scaling_factor: f32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    /// "conv" | "full_attention" per layer.
    layer_types: Vec<String>,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_multiple_of() -> usize {
    256
}
fn default_conv_l() -> usize {
    3
}
fn default_eps() -> f32 {
    1e-5
}
fn default_max_pos() -> usize {
    128_000
}
fn default_true() -> bool {
    true
}
fn default_routed_scale() -> f32 {
    1.0
}

impl Lfm2MoeConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("lfm2moe: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "lfm2moe: metadata_json missing `config` wrapper".to_string())?;
        Self::from_config_value(inner)
    }

    /// Parse from a raw `config.json` Value (the inner `config` blob).
    pub fn from_config_value(inner: &serde_json::Value) -> Result<Self, String> {
        let raw: RawLfm2MoeConfig = serde_json::from_value(inner.clone())
            .map_err(|e| format!("lfm2moe: parsing config failed: {e}"))?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        // rope_theta lives under rope_parameters in LFM2.5; fall back to a flat
        // rope_theta (older configs) then the default.
        let rope_theta = raw
            .rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .unwrap_or(raw.rope_theta);
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(format!(
                "lfm2moe: layer_types len {} != num_hidden_layers {}",
                raw.layer_types.len(),
                raw.num_hidden_layers
            ));
        }
        let layer_types = raw
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" | "attention" | "attn" => Ok(MixerKind::Attention),
                "conv" | "short_conv" | "conv1d" => Ok(MixerKind::Conv),
                other => Err(format!("lfm2moe: unknown layer_type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Dense SwiGLU FFN dim: LFM2 auto-adjusts intermediate_size LLaMA-style
        // (the loaded w1/w2/w3 tensors use this dim, NOT raw intermediate_size).
        // Idempotent for configs that already carry the adjusted value or set
        // block_auto_adjust_ff_dim=false (e.g. A1B), so MoE behavior is preserved.
        let intermediate_size = {
            let base = raw.block_ff_dim.unwrap_or(raw.intermediate_size);
            if raw.block_auto_adjust_ff_dim {
                let mut ff = (2 * base) / 3;
                if let Some(m) = raw.block_ffn_dim_multiplier {
                    ff = (m * ff as f32) as usize;
                }
                let mo = raw.block_multiple_of.max(1);
                mo * ((ff + mo - 1) / mo)
            } else {
                raw.intermediate_size
            }
        };
        // Dense lfm2 (Lfm2ForCausalLM, no experts) → every layer is dense SwiGLU.
        let num_dense_layers = if raw.num_experts == 0 {
            raw.num_hidden_layers
        } else {
            raw.num_dense_layers
        };
        Ok(Lfm2MoeConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            conv_kernel_size: raw.conv_L_cache,
            intermediate_size,
            moe_intermediate_size: raw.moe_intermediate_size,
            num_experts: raw.num_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            num_dense_layers,
            rope_theta,
            rms_norm_eps: raw.norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            norm_topk_prob: raw.norm_topk_prob,
            use_expert_bias: raw.use_expert_bias,
            routed_scaling_factor: raw.routed_scaling_factor,
            tie_word_embeddings: raw.tie_word_embeddings,
            layer_types,
        })
    }

    /// q projection output width (n_heads * head_dim).
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    /// k/v projection output width (n_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn mixer(&self, layer: usize) -> MixerKind {
        self.layer_types[layer]
    }
    pub fn is_attention(&self, layer: usize) -> bool {
        self.layer_types[layer] == MixerKind::Attention
    }
    /// The first `num_dense_layers` FFN blocks are dense SwiGLU; the rest MoE.
    pub fn is_dense_ffn(&self, layer: usize) -> bool {
        layer < self.num_dense_layers
    }
    /// Number of attention layers (== KvCache slots needed).
    pub fn num_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == MixerKind::Attention)
            .count()
    }
    /// Number of conv layers (== conv-state cache slots needed).
    pub fn num_conv_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == MixerKind::Conv)
            .count()
    }
}
