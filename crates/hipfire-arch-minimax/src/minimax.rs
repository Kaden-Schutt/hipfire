// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 config / weights / state.
//!
//! Config is parsed from the HFQ `metadata_json` envelope
//! (`{"architecture":..., "config":{...HF config...}, "tokenizer":...}`),
//! same as deepseek4/qwen35. Weights/State are SCAFFOLD STUBS for now —
//! the real loader (mirroring `deepseek4::arch` upload helpers + 256
//! routed experts) and the KV/MoE scratch land with the forward bring-up.

use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{Gpu, GpuTensor};
use serde::Deserialize;

/// Typed MiniMax-M2 shape constants.
#[derive(Clone, Debug)]
pub struct MiniMaxConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Expert (MoE) FFN intermediate size (HF `intermediate_size`).
    pub intermediate_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    /// Rotated-dim count for partial RoPE (`rotary_dim`, < head_dim).
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// Per-layer QK-norm on the flat q/k projection (RMSNorm pre-reshape).
    pub use_qk_norm: bool,
    /// Router uses `e_score_correction_bias` for top-k selection.
    pub use_routing_bias: bool,
    /// Router score activation; MiniMax-M2 = "sigmoid".
    pub scoring_func: String,
    /// MTP draft modules (spec-decode; 0 for the base forward).
    pub num_mtp_modules: usize,
}

/// Raw deserialize of the inner HF `config` object. `#[serde(default)]`
/// on the kwarg-style fields so missing keys fall back to MiniMax-M2
/// defaults rather than erroring.
#[derive(Deserialize)]
struct RawMiniMaxConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    intermediate_size: usize,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default = "default_rotary_dim")]
    rotary_dim: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default)]
    use_qk_norm: bool,
    #[serde(default)]
    use_routing_bias: bool,
    #[serde(default = "default_scoring")]
    scoring_func: String,
    #[serde(default)]
    num_mtp_modules: usize,
}

fn default_rotary_dim() -> usize {
    64
}
fn default_rope_theta() -> f32 {
    5_000_000.0
}
fn default_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    196_608
}
fn default_scoring() -> String {
    "sigmoid".to_string()
}

impl MiniMaxConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("minimax: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "minimax: metadata_json missing `config` wrapper".to_string())?;
        let raw: RawMiniMaxConfig = serde_json::from_value(inner.clone())
            .map_err(|e| format!("minimax: parsing inner config failed: {e}"))?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        Ok(MiniMaxConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            intermediate_size: raw.intermediate_size,
            num_local_experts: raw.num_local_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            rotary_dim: raw.rotary_dim,
            rope_theta: raw.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            use_qk_norm: raw.use_qk_norm,
            use_routing_bias: raw.use_routing_bias,
            scoring_func: raw.scoring_func,
            num_mtp_modules: raw.num_mtp_modules,
        })
    }

    /// Heads-per-KV-group (GQA repeat factor).
    pub fn num_key_value_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Per-layer GPU-resident weights. `Option<GpuTensor>` slots so the
/// host-walk / progressive-upload paths can populate incrementally
/// (mirrors `DeepseekV4LayerWeights`). The routed-expert blobs +
/// pointer tables are added with the real loader.
#[derive(Default)]
pub struct MiniMaxLayerWeights {
    pub input_norm: Option<GpuTensor>,
    pub post_attn_norm: Option<GpuTensor>,
    /// Flat per-layer QK-norm weights: q_norm [n_heads*head_dim],
    /// k_norm [n_kv*head_dim].
    pub q_norm: Option<GpuTensor>,
    pub k_norm: Option<GpuTensor>,
    pub q_proj: Option<GpuTensor>,
    pub k_proj: Option<GpuTensor>,
    pub v_proj: Option<GpuTensor>,
    pub o_proj: Option<GpuTensor>,
    /// MoE router gate [n_experts, hidden] + e_score_correction_bias [n_experts].
    pub gate: Option<GpuTensor>,
    pub e_score_bias: Option<GpuTensor>,
    // routed experts (gate/up/down blobs + ptr tables) — filled by the
    // real loader, stubbed here.
}

/// Model weights. SCAFFOLD: slots present, real upload TODO (P3 wiring).
pub struct MiniMaxWeights {
    pub embed: Option<GpuTensor>,
    pub layers: Vec<MiniMaxLayerWeights>,
    pub final_norm: Option<GpuTensor>,
    pub lm_head: Option<GpuTensor>,
}

impl MiniMaxWeights {
    /// SCAFFOLD STUB — allocates empty per-layer slots so the crate
    /// compiles and the daemon can be wired. The real loader mirrors
    /// `deepseek4::arch::{upload_quant_or_f16, upload_layer_routed_experts}`
    /// (per-layer attn/norm/router + 256 routed SwiGLU experts) and is
    /// implemented during forward bring-up.
    pub fn load(_hfq: &mut HfqFile, cfg: &MiniMaxConfig, _gpu: &mut Gpu) -> Result<Self, String> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| MiniMaxLayerWeights::default())
            .collect();
        Ok(MiniMaxWeights {
            embed: None,
            layers,
            final_norm: None,
            lm_head: None,
        })
    }
}

/// Per-decode GPU scratch (KV cache, attention workspace, MoE workspace).
/// SCAFFOLD: tracks position only; buffers allocated during bring-up.
pub struct MiniMaxState {
    pub n_tokens: usize,
}

impl MiniMaxState {
    /// SCAFFOLD STUB. Real allocation (GQA KV cache sized by
    /// num_key_value_heads*head_dim*max_seq, attention scratch, MoE
    /// scatter/expert workspace) lands with the forward.
    pub fn new(_gpu: &mut Gpu, _cfg: &MiniMaxConfig) -> Result<Self, String> {
        Ok(MiniMaxState { n_tokens: 0 })
    }
}
