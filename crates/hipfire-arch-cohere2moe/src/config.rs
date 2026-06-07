// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2Moe config, parsed from the HFQ `metadata_json` envelope (which
//! carries the source `config.json` wholesale under the `config` key).
//!
//! Ground truth: CohereLabs/BLS-Mini-Code-1.0 config.json + transformers
//! `Cohere2MoeConfig` / `modeling_cohere2_moe.py` (read 2026-06-07):
//!   hidden 2048, 49 layers, 32 q-heads / 4 kv-heads, head_dim 128,
//!   `first_k_dense_replace` = 1 (layer 0 dense SwiGLU @ inter 3072; layers
//!   1..48 = 128-expert top-8 MoE @ inter 768), sigmoid expert selection,
//!   `norm_topk_prob` = false (NO top-k renormalization), no routing bias,
//!   no shared experts, **parallel decoder block** (single input_layernorm
//!   shared by attention + MLP, both summed into the residual), **standard
//!   RMSNorm** (`weight * x̂`, no +1; the model sets `rms_norm_eps` so the
//!   reference picks `Cohere2MoeRMSNorm` not the mean-centered LayerNorm),
//!   **interleaved RoPE** (rotate pairs (2i,2i+1), full head_dim) theta 5e4,
//!   1:3 full:sliding attention cadence (`sliding_window` 4096), tied
//!   embeddings, `logit_scale` 1.0 (no-op here), `use_qk_norm` false.

use hipfire_runtime::hfq::HfqFile;
use serde::Deserialize;

/// Per-layer attention kind, decoded from `layer_types`.
///
/// NOTE: sliding-window attention is **not yet implemented** — both kinds
/// currently run full causal attention in the forward (correct for any prompt
/// shorter than `sliding_window` = 4096). The kind is parsed and stored so the
/// windowed-KV path can be added later without a config change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    Full,
    Sliding,
}

/// Typed Cohere2Moe shape constants.
#[derive(Clone, Debug)]
pub struct Cohere2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Expert (MoE) FFN intermediate size (HF `intermediate_size`).
    pub moe_intermediate_size: usize,
    /// Dense-MLP FFN intermediate size for the prefix dense layers
    /// (HF `prefix_dense_intermediate_size`).
    pub dense_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// The first `first_k_dense_replace` FFN blocks are dense SwiGLU; the rest
    /// are MoE.
    pub first_k_dense_replace: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    /// When 1, the prefix dense layers force RoPE even though they are
    /// full-attention (HF `prefix_dense_sliding_window_pattern`; drives
    /// `force_rope`). See `apply_rope`.
    pub prefix_dense_sliding_window_pattern: usize,
    /// Renormalize the top-k gathered router weights to sum 1 (HF
    /// `norm_topk_prob`). Cohere2Moe = false.
    pub norm_topk_prob: bool,
    /// Router score activation; Cohere2Moe = "sigmoid".
    pub expert_selection_fn: String,
    /// Final-logit multiplier (HF `logit_scale`). 1.0 for this checkpoint.
    pub logit_scale: f32,
    /// Per-layer QK-norm (Cohere2Moe = false for this checkpoint).
    pub use_qk_norm: bool,
    /// Parallel decoder block: attention + MLP read ONE shared input_layernorm
    /// and are BOTH summed into the residual. Cohere2Moe = true.
    pub use_parallel_block: bool,
    /// lm_head shares embed_tokens.
    pub tie_word_embeddings: bool,
    /// Per-layer attention kind (length == num_hidden_layers).
    pub layer_types: Vec<AttnKind>,
}

#[derive(Deserialize)]
struct RawCohere2MoeConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    intermediate_size: usize,
    #[serde(default)]
    prefix_dense_intermediate_size: Option<usize>,
    num_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default = "default_first_k_dense")]
    first_k_dense_replace: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default = "default_one")]
    prefix_dense_sliding_window_pattern: usize,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default = "default_selection")]
    expert_selection_fn: String,
    #[serde(default = "default_logit_scale")]
    logit_scale: f32,
    #[serde(default)]
    use_qk_norm: bool,
    #[serde(default = "default_true")]
    use_parallel_block: bool,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    /// "full_attention" | "sliding_attention" per layer.
    #[serde(default)]
    layer_types: Vec<String>,
}

fn default_first_k_dense() -> usize {
    1
}
fn default_rope_theta() -> f32 {
    50_000.0
}
fn default_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    500_000
}
fn default_sliding_window() -> usize {
    4096
}
fn default_one() -> usize {
    1
}
fn default_selection() -> String {
    "sigmoid".to_string()
}
fn default_logit_scale() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}

impl Cohere2MoeConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("cohere2moe: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "cohere2moe: metadata_json missing `config` wrapper".to_string())?;
        Self::from_config_value(inner)
    }

    /// Parse from a raw `config.json` Value (the inner `config` blob).
    pub fn from_config_value(inner: &serde_json::Value) -> Result<Self, String> {
        let raw: RawCohere2MoeConfig = serde_json::from_value(inner.clone())
            .map_err(|e| format!("cohere2moe: parsing config failed: {e}"))?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        let dense_intermediate_size = raw
            .prefix_dense_intermediate_size
            .unwrap_or(raw.intermediate_size);
        // layer_types is optional in flat configs; default every layer to full
        // attention when absent (SWA is deferred anyway).
        let layer_types = if raw.layer_types.is_empty() {
            vec![AttnKind::Full; raw.num_hidden_layers]
        } else {
            if raw.layer_types.len() != raw.num_hidden_layers {
                return Err(format!(
                    "cohere2moe: layer_types len {} != num_hidden_layers {}",
                    raw.layer_types.len(),
                    raw.num_hidden_layers
                ));
            }
            raw.layer_types
                .iter()
                .map(|s| match s.as_str() {
                    "full_attention" | "full" | "attention" => Ok(AttnKind::Full),
                    "sliding_attention" | "sliding" | "swa" => Ok(AttnKind::Sliding),
                    other => Err(format!("cohere2moe: unknown layer_type {other:?}")),
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Cohere2MoeConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            moe_intermediate_size: raw.intermediate_size,
            dense_intermediate_size,
            num_experts: raw.num_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            first_k_dense_replace: raw.first_k_dense_replace,
            rope_theta: raw.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window: raw.sliding_window,
            prefix_dense_sliding_window_pattern: raw.prefix_dense_sliding_window_pattern,
            norm_topk_prob: raw.norm_topk_prob,
            expert_selection_fn: raw.expert_selection_fn,
            logit_scale: raw.logit_scale,
            use_qk_norm: raw.use_qk_norm,
            use_parallel_block: raw.use_parallel_block,
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
    /// The first `first_k_dense_replace` FFN blocks are dense SwiGLU; the rest MoE.
    pub fn is_dense_ffn(&self, layer: usize) -> bool {
        layer < self.first_k_dense_replace
    }
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        self.layer_types[layer]
    }

    /// Whether RoPE is applied at `layer`. Cohere2Moe is **NoPE on
    /// full-attention (global) layers** and RoPE on sliding-window (local)
    /// layers — except the prefix dense layers force RoPE when
    /// `prefix_dense_sliding_window_pattern == 1`. Mirrors the reference gate
    /// `if self.sliding_window is not None or self.force_rope`.
    pub fn apply_rope(&self, layer: usize) -> bool {
        match self.attn_kind(layer) {
            AttnKind::Sliding => true,
            AttnKind::Full => {
                self.is_dense_ffn(layer) && self.prefix_dense_sliding_window_pattern == 1
            }
        }
    }
}
