// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE config, parsed from the HFQ `metadata_json` envelope (which
//! carries the source `config.json` wholesale under the `config` key).
//!
//! Ground truth: CohereLabs/North-Mini-Code-1.0 `config.json`
//! (`architectures: ["Cohere2MoeForCausalLM"]`, `model_type: cohere2_moe`,
//! transformers 5.8.0) + HF `modeling_cohere2`:
//!   hidden 2048, 49 layers, 32 q-heads / 4 kv-heads, head_dim 128 (explicit),
//!   128 experts top-8 (sigmoid selection, `norm_topk_prob=false`, 0 shared),
//!   `first_k_dense_replace=1` (layer 0 is a dense SwiGLU MLP,
//!   intermediate = `prefix_dense_intermediate_size` 3072; experts use
//!   `intermediate_size` 768), vocab 262144, rope_theta 50000,
//!   sliding_window 4096, max_position 500000.
//!
//! Three structural traits distinguish this arch from every other hipfire MoE:
//!   1. **Parallel block** (`use_parallel_block=true`): a SINGLE
//!      `input_layernorm` feeds BOTH the attention and the MoE/dense branch,
//!      and both add into the residual — `h = h + attn(LN(h)) + moe(LN(h))`
//!      (no separate `post_attention_layernorm`).
//!   2. **Interleaved attention** from `layer_types`: `full_attention`
//!      (global, **NoPE** — no positional embedding) vs `sliding_attention`
//!      (window 4096, **RoPE**). Every 4th layer (0,4,…,48) is full.
//!   3. **RMSNorm** (`LlamaRMSNorm`, no bias) — cohere2_moe replaced base
//!      Cohere2's mean-centered LayerNorm; uses `rms_norm_eps` (1e-6).
//!
//! No QK-norm, no attention bias, `logit_scale=1.0` (no-op), tied embeddings.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::model_source::ModelSource;
use serde::Deserialize;

/// Per-layer attention kind, decoded from `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    /// Global attention over the full context. **NoPE** — no RoPE applied
    /// (Cohere2 sets `sliding_window=None` on these layers, which gates off
    /// the rotary embedding in the HF attention forward).
    Full,
    /// Local sliding-window attention (window = `sliding_window`). **RoPE**
    /// applied over the full head_dim.
    Sliding,
}

/// Typed Cohere2-MoE shape constants.
#[derive(Clone, Debug)]
pub struct Cohere2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit per-head dim (HF `head_dim` = 128); NOT hidden/n_heads
    /// (32*128 = 4096 q-dim ≠ hidden 2048).
    pub head_dim: usize,
    /// Routed-expert FFN intermediate size (HF `intermediate_size` = 768).
    pub moe_intermediate_size: usize,
    /// Dense prefix-layer FFN intermediate size
    /// (HF `prefix_dense_intermediate_size` = 3072).
    pub dense_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// The first `first_k_dense_replace` layers are dense SwiGLU MLPs; the
    /// rest are MoE. North-Mini-Code: 1 (layer 0 dense).
    pub first_k_dense_replace: usize,
    /// HF `prefix_dense_sliding_window_pattern` (North: 1). When 1, the dense
    /// prefix layers (`l < first_k_dense_replace`) get RoPE via `force_rope`
    /// even though they are `full_attention` — matches Cohere2MoeAttention.
    pub prefix_dense_sliding_window_pattern: usize,
    /// Shared experts run on every token alongside the routed ones. 0 for
    /// North-Mini-Code (`num_shared_experts=0`).
    pub num_shared_experts: usize,
    pub rope_theta: f32,
    /// Epsilon for the legacy mean-centered `Cohere2LayerNorm` fallback (HF
    /// `layer_norm_eps`). NOTE: cohere2_moe (North) sets `rms_norm_eps`, so the
    /// reference normalizes with plain RMSNorm — see `rms_norm_eps`.
    pub layer_norm_eps: f32,
    /// Epsilon for the per-layer + final **RMSNorm** (HF `rms_norm_eps` = 1e-6).
    /// The cohere2_moe modular file replaces base Cohere2's mean-centered
    /// LayerNorm with plain `LlamaRMSNorm` when this is set (it is, for North):
    /// `Cohere2MoeRMSNorm(eps=rms_norm_eps) if rms_norm_eps is not None`.
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// Local-attention window for `sliding_attention` layers (HF
    /// `sliding_window` = 4096). For sequences ≤ this, sliding == full causal.
    pub sliding_window: usize,
    /// Renormalize the top-k gathered router weights to sum 1. **false** for
    /// North-Mini-Code (`norm_topk_prob=false` — raw sigmoid scores are used
    /// as combine weights).
    pub norm_topk_prob: bool,
    /// Output-logit multiplier (HF `logit_scale`); 1.0 = no-op for this model.
    pub logit_scale: f32,
    /// lm_head shares embed_tokens (Cohere ties; no `lm_head.weight` in the
    /// checkpoint).
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
    /// Routed-expert FFN size.
    intermediate_size: usize,
    #[serde(default)]
    prefix_dense_intermediate_size: Option<usize>,
    num_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default = "default_first_k_dense")]
    first_k_dense_replace: usize,
    #[serde(default = "default_one")]
    prefix_dense_sliding_window_pattern: usize,
    #[serde(default)]
    num_shared_experts: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_ln_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default = "default_logit_scale")]
    logit_scale: f32,
    /// Absent on this checkpoint (Cohere ties by default).
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    /// "full_attention" | "sliding_attention" per layer.
    layer_types: Vec<String>,
}

fn default_first_k_dense() -> usize {
    1
}
fn default_one() -> usize {
    1
}
fn default_rope_theta() -> f32 {
    50_000.0
}
fn default_ln_eps() -> f32 {
    1e-5
}
fn default_rms_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    500_000
}
fn default_sliding_window() -> usize {
    4096
}
fn default_logit_scale() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}

impl Cohere2MoeConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        Self::from_metadata_json(&hfq.metadata_json)
    }

    /// Parse from a `metadata_json` string (HFQ metadata OR a SafetensorsSource's
    /// config.json, both of which embed the HF config under the `config` key).
    /// Shared by `from_hfq` and `from_safetensors`.
    pub fn from_metadata_json(metadata_json: &str) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|e| format!("cohere2moe: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "cohere2moe: metadata_json missing `config` wrapper".to_string())?;
        Self::from_config_value(inner)
    }

    /// Parse the config from a safetensors directory source (the transparent
    /// Dir loading path). Mirrors `qwen35::config_from_safetensors`. The
    /// `SafetensorsSource` embeds config.json under the same `config` wrapper
    /// key as HFQ metadata, so this reuses `from_metadata_json`.
    pub fn from_safetensors(source: &dyn ModelSource) -> Result<Self, String> {
        Self::from_metadata_json(source.metadata_json())
    }

    /// Parse from a raw `config.json` Value (the inner `config` blob).
    pub fn from_config_value(inner: &serde_json::Value) -> Result<Self, String> {
        let raw: RawCohere2MoeConfig = serde_json::from_value(inner.clone())
            .map_err(|e| format!("cohere2moe: parsing config failed: {e}"))?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(format!(
                "cohere2moe: layer_types len {} != num_hidden_layers {}",
                raw.layer_types.len(),
                raw.num_hidden_layers
            ));
        }
        let layer_types = raw
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" | "full" | "global" => Ok(AttnKind::Full),
                "sliding_attention" | "sliding" | "swa" => Ok(AttnKind::Sliding),
                other => Err(format!("cohere2moe: unknown layer_type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Dense prefix FFN dim: `prefix_dense_intermediate_size` if present,
        // else fall back to the expert intermediate (degenerate but safe).
        let dense_intermediate_size = raw
            .prefix_dense_intermediate_size
            .unwrap_or(raw.intermediate_size);
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
            prefix_dense_sliding_window_pattern: raw.prefix_dense_sliding_window_pattern,
            num_shared_experts: raw.num_shared_experts,
            rope_theta: raw.rope_theta,
            layer_norm_eps: raw.layer_norm_eps,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window: raw.sliding_window,
            norm_topk_prob: raw.norm_topk_prob,
            logit_scale: raw.logit_scale,
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
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        self.layer_types[layer]
    }
    /// Whether RoPE is applied on this layer. Sliding layers use RoPE; global
    /// (full_attention) layers are **NoPE** (no positional embedding).
    pub fn uses_rope(&self, layer: usize) -> bool {
        self.layer_types[layer] == AttnKind::Sliding
    }
    /// Whether this layer's FFN is dense SwiGLU (the `first_k_dense_replace`
    /// prefix) rather than routed MoE.
    pub fn is_dense_ffn(&self, layer: usize) -> bool {
        layer < self.first_k_dense_replace
    }
    /// The FFN intermediate size for a given layer (dense prefix vs experts).
    pub fn ffn_intermediate(&self, layer: usize) -> usize {
        if self.is_dense_ffn(layer) {
            self.dense_intermediate_size
        } else {
            self.moe_intermediate_size
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    /// The transparent Dir path: SafetensorsSource::metadata_json embeds
    /// config.json under the `config` key (same as HFQ metadata), so
    /// from_metadata_json (which from_safetensors delegates to) must parse it.
    #[test]
    fn from_metadata_json_parses_dir_shape() {
        let json = r#"{"config":{
            "vocab_size": 256000, "hidden_size": 2048, "num_hidden_layers": 4,
            "num_attention_heads": 16, "num_key_value_heads": 4,
            "intermediate_size": 1024, "num_experts": 8, "num_experts_per_tok": 2,
            "sliding_window": 4096,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","full_attention"]
        }}"#;
        let cfg = Cohere2MoeConfig::from_metadata_json(json).expect("parse Dir config");
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.num_experts, 8);
        assert_eq!(cfg.sliding_window, 4096);
        assert_eq!(cfg.layer_types.len(), 4);
        assert_eq!(cfg.layer_types[3], AttnKind::Full);
        assert_eq!(cfg.layer_types[0], AttnKind::Sliding);
    }

    #[test]
    fn missing_config_wrapper_errs() {
        // A bare config.json (no `config` wrapper) is rejected with a clear error.
        assert!(Cohere2MoeConfig::from_metadata_json(r#"{"hidden_size":2048}"#).is_err());
    }
}
