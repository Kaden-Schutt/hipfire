// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `zaya` architecture support (Zyphra ZAYA1 family, arch_id 16) — a **uniform
//! stack of hybrid decoder blocks**, each one: CCA attention mixer + EDA/MoD-routed
//! MoE FFN, joined by learned (fp32) residual rescaling.
//!
//! The published ZAYA1-8B checkpoint ships in the original Megatron
//! **alternating-layer** layout: `2*num_hidden_layers` does not apply — instead
//! `num_hidden_layers` (e.g. 80) counts the alternating attention/MoE half-layers,
//! where **even index = CCA attention, odd index = MoE**. They pair into
//! `num_hidden_layers / 2` (e.g. 40) hybrid decoder blocks. See the upstream
//! `convert_zaya_weights_to_hf.py` (`new_layer_idx = old // 2`) — this is the
//! ground truth this crate mirrors, vendored at
//! `third_party/transformers` (origin Zyphra/transformers @ `zaya1`).
//!
//! Z0 (this module): the config + block taxonomy only — pure, GPU-free, parsed
//! from the HF `config.json`. The CCA mixer (two causal conv1d over q/k, L2 qk-norm
//! with learned key temperature, 1-token delayed value composition, partial RoPE,
//! GQA attention), the EDA MLP router (top-1 of `num_experts + 1`, the extra
//! "skip" expert being MoD), the SwiGLU experts, the per-block residual rescaling,
//! the weight loader, and the serving impls land in later loop iterations (Z1+).

use serde::Deserialize;

pub mod arch;
pub mod calibration;
pub mod cpu;
pub mod gpu;
pub mod weights;

pub use hipfire_model::ARCH_ID_ZAYA;

/// Per-half-layer mixer/FFN role in the **alternating** checkpoint layout. The
/// loader pairs one `Attention` half-layer (even index) with the following `Moe`
/// half-layer (odd index) into a single [`ZayaConfig`] hybrid block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfLayerKind {
    /// Even index — CCA attention mixer (`self_attn.*`).
    Attention,
    /// Odd index — EDA/MoD-routed MoE FFN (`zaya_block.*`).
    Moe,
}

impl HalfLayerKind {
    /// The role of half-layer `idx` in the alternating layout.
    pub fn at(idx: usize) -> Self {
        if idx % 2 == 0 {
            HalfLayerKind::Attention
        } else {
            HalfLayerKind::Moe
        }
    }
}

/// Per-block attention type. ZAYA hybrid blocks are full-attention; sliding-window
/// blocks (`hybrid_sliding`) exist in the upstream config space but ZAYA1-8B uses
/// none (`sliding_window: null`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnWindow {
    /// Full causal attention.
    Full,
    /// Sliding-window causal attention (span = [`ZayaConfig::sliding_window`]).
    Sliding,
}

/// Resolved ZAYA model shape, after collapsing the alternating half-layers into
/// hybrid decoder blocks. All dims are GPU-free constants parsed from `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct ZayaConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    /// Number of hybrid decoder blocks (= `num_hidden_layers / 2`).
    pub num_blocks: usize,
    /// Raw alternating half-layer count from `config.json` (`num_hidden_layers`).
    pub num_half_layers: usize,
    pub rms_norm_eps: f32,
    /// lm_head tied to `embed_tokens` (`tie_word_embeddings`, default true for ZAYA).
    pub tie_word_embeddings: bool,
    /// Whether the lm_head carries a bias (`lm_head_bias`; ZAYA1-8B: false).
    pub lm_head_bias: bool,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub pad_token_id: u32,
    pub max_position_embeddings: usize,

    pub attn: ZayaAttnConfig,
    pub moe: ZayaMoeConfig,
    /// Per-block attention window (length `num_blocks`). ZAYA1-8B: all `Full`.
    pub windows: Vec<AttnWindow>,
    /// Sliding-window span (total local span incl. current token), if any block
    /// is `Sliding`.
    pub sliding_window: Option<usize>,
}

/// CCA attention shape. The mixer feeds a *standard* GQA attention after a
/// ZAYA-specific projection (two causal conv1d over concatenated q/k, learned
/// per-kv-head key temperature, 1-token delayed value composition).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZayaAttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dim count: `round(head_dim * partial_rotary_factor)` (ZAYA: 64).
    pub n_rot: usize,
    pub rope_theta: f32,
    /// Rope base for sliding-window blocks (upstream default 10_000), if present.
    pub swa_rope_theta: f32,
    pub attention_bias: bool,
    /// Depthwise conv kernel width over concat(q,k) (`cca_time0`, default 2).
    pub conv_depthwise_kernel: usize,
    /// Grouped conv kernel width over concat(q,k) (`cca_time1`, default 2).
    pub conv_grouped_kernel: usize,
}

impl ZayaAttnConfig {
    /// Channels of the concatenated q/k conv stream:
    /// `(num_heads + num_kv_heads) * head_dim`.
    pub fn conv_channels(&self) -> usize {
        (self.num_heads + self.num_kv_heads) * self.head_dim
    }

    /// Cached convolution-state width during decode:
    /// `(conv_depthwise_kernel - 1) + (conv_grouped_kernel - 1)`.
    pub fn conv_state_len(&self) -> usize {
        (self.conv_depthwise_kernel - 1) + (self.conv_grouped_kernel - 1)
    }
}

/// EDA/MoD-routed MoE shape. SwiGLU experts; a 3-linear MLP router operating in a
/// `router_hidden_size` space that emits `num_experts + 1` logits (top-1). The
/// extra expert is the **MoD skip** route (token bypasses the FFN).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZayaMoeConfig {
    /// Real (non-skip) routed experts (`num_experts`, e.g. 16).
    pub num_experts: usize,
    /// Experts selected per token (`moe_router_topk`; ZAYA: 1).
    pub top_k: usize,
    /// Per-expert SwiGLU gate/up width (`ffn_hidden_size / 2`).
    pub moe_intermediate_size: usize,
    /// Router MLP hidden width (`router_hidden_size` / `zaya_mlp_expansion`; 256).
    pub router_hidden_size: usize,
    /// Cross-layer router state recurrence (`zaya_use_eda`).
    pub use_eda: bool,
    /// Mixture-of-Depths skip route via the extra router expert (`zaya_use_mod`).
    pub use_mod: bool,
}

impl ZayaMoeConfig {
    /// Router output width including the trailing MoD skip expert.
    pub fn num_router_experts(&self) -> usize {
        self.num_experts + 1
    }
}

impl ZayaConfig {
    /// Parse the resolved config from a ZAYA `config.json` value. Accepts **both**
    /// layouts and normalizes to hybrid-block dims:
    /// - **Native** (post `convert_zaya_weights_to_hf.py`): `num_hidden_layers`
    ///   already counts hybrid blocks (e.g. 40); `moe_intermediate_size`,
    ///   `rms_norm_eps`, `router_hidden_size` present.
    /// - **Megatron** (published ZAYA1-8B): `num_hidden_layers` counts alternating
    ///   attn/MoE half-layers (e.g. 80 → 40 blocks); `ffn_hidden_size` (fused
    ///   gate+up), `norm_epsilon`, `zaya_mlp_expansion` present.
    ///
    /// The discriminator is `ffn_hidden_size` (Megatron-only; native uses
    /// `moe_intermediate_size`).
    pub fn from_json(c: &serde_json::Value) -> Result<Self, String> {
        let raw: RawConfig =
            serde_json::from_value(c.clone()).map_err(|e| format!("zaya config: {e}"))?;

        if raw.hidden_size == 0 || raw.vocab_size == 0 {
            return Err("zaya config: zero hidden_size/vocab_size".to_string());
        }
        if raw.num_attention_heads % raw.num_key_value_heads != 0 {
            return Err(format!(
                "zaya config: num_attention_heads {} not a multiple of num_key_value_heads {}",
                raw.num_attention_heads, raw.num_key_value_heads
            ));
        }
        if raw.moe_router_topk != 1 {
            return Err(format!(
                "zaya config: only moe_router_topk=1 is supported, got {}",
                raw.moe_router_topk
            ));
        }
        // The forward implements full causal attention only; reject sliding-window
        // (`hybrid_sliding`) checkpoints rather than silently computing full attn.
        if raw.sliding_window.is_some() {
            return Err("zaya config: sliding-window attention is not implemented".to_string());
        }

        let is_megatron = raw.ffn_hidden_size.is_some();
        let num_blocks = if is_megatron {
            if raw.num_hidden_layers == 0 || raw.num_hidden_layers % 2 != 0 {
                return Err(format!(
                    "zaya Megatron config: num_hidden_layers must be a positive even (alternating attn/MoE) count, got {}",
                    raw.num_hidden_layers
                ));
            }
            raw.num_hidden_layers / 2
        } else {
            if raw.num_hidden_layers == 0 {
                return Err("zaya config: zero num_hidden_layers".to_string());
            }
            raw.num_hidden_layers
        };
        // Per-expert SwiGLU hidden (gate == up width): native gives it directly;
        // Megatron's `ffn_hidden_size` is the fused gate+up width, so halve it.
        let moe_intermediate_size = raw
            .moe_intermediate_size
            .or(raw.ffn_hidden_size.map(|f| f / 2))
            .ok_or("zaya config: missing moe_intermediate_size / ffn_hidden_size")?;
        let n_rot = ((raw.head_dim as f32) * raw.partial_rotary_factor).round() as usize;
        // Native names it `router_hidden_size`; Megatron carries it as
        // `zaya_mlp_expansion`. Prefer the explicit one.
        let router_hidden_size = raw
            .router_hidden_size
            .or(raw.zaya_mlp_expansion)
            .unwrap_or(256);

        let windows = vec![AttnWindow::Full; num_blocks];

        Ok(Self {
            hidden_size: raw.hidden_size,
            vocab_size: raw.vocab_size,
            num_blocks,
            num_half_layers: raw.num_hidden_layers,
            rms_norm_eps: raw.norm_epsilon,
            tie_word_embeddings: raw.tie_word_embeddings,
            lm_head_bias: raw.lm_head_bias,
            eos_token_id: raw.eos_token_id,
            bos_token_id: raw.bos_token_id,
            pad_token_id: raw.pad_token_id,
            max_position_embeddings: raw.max_position_embeddings,
            attn: ZayaAttnConfig {
                num_heads: raw.num_attention_heads,
                num_kv_heads: raw.num_key_value_heads,
                head_dim: raw.head_dim,
                n_rot,
                rope_theta: raw.rope_theta,
                swa_rope_theta: raw.swa_rope_theta,
                attention_bias: raw.attention_bias,
                conv_depthwise_kernel: raw.cca_time0,
                conv_grouped_kernel: raw.cca_time1,
            },
            moe: ZayaMoeConfig {
                num_experts: raw.num_experts,
                top_k: raw.moe_router_topk,
                moe_intermediate_size,
                router_hidden_size,
                use_eda: raw.zaya_use_eda,
                use_mod: raw.zaya_use_mod,
            },
            windows,
            sliding_window: raw.sliding_window,
        })
    }

    /// GQA expansion factor (`num_heads / num_kv_heads`).
    pub fn num_kv_groups(&self) -> usize {
        self.attn.num_heads / self.attn.num_kv_heads
    }
}

/// Serde shape of the published (Megatron-named) ZAYA `config.json`. Fields the
/// runtime does not consume (`activation_func`, `normalization`, `mamba_cache_dtype`,
/// the fusion flags) are intentionally omitted — serde ignores unknown keys.
#[derive(Deserialize)]
struct RawConfig {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    #[serde(default = "default_eps", alias = "rms_norm_eps")]
    norm_epsilon: f32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    lm_head_bias: bool,
    #[serde(default = "default_eos")]
    eos_token_id: u32,
    #[serde(default = "default_bos")]
    bos_token_id: u32,
    #[serde(default)]
    pad_token_id: u32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_swa_rope_theta")]
    swa_rope_theta: f32,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default = "default_cca_time")]
    cca_time0: usize,
    #[serde(default = "default_cca_time")]
    cca_time1: usize,

    /// Megatron-only: fused gate+up width. Absent in native configs.
    #[serde(default)]
    ffn_hidden_size: Option<usize>,
    /// Native-only: per-expert SwiGLU gate/up width.
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    num_experts: usize,
    #[serde(default = "default_topk", alias = "num_experts_per_tok")]
    moe_router_topk: usize,
    /// HF-native name for the router hidden width.
    #[serde(default)]
    router_hidden_size: Option<usize>,
    /// Megatron name for the router hidden width.
    #[serde(default)]
    zaya_mlp_expansion: Option<usize>,
    #[serde(default = "default_true")]
    zaya_use_eda: bool,
    #[serde(default = "default_true")]
    zaya_use_mod: bool,

    #[serde(default)]
    sliding_window: Option<usize>,
}

fn default_eps() -> f32 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_eos() -> u32 {
    106
}
fn default_bos() -> u32 {
    2
}
fn default_max_pos() -> usize {
    131072
}
fn default_partial_rotary() -> f32 {
    0.5
}
fn default_rope_theta() -> f32 {
    5_000_000.0
}
fn default_swa_rope_theta() -> f32 {
    10_000.0
}
fn default_cca_time() -> usize {
    2
}
fn default_topk() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published Zyphra/ZAYA1-8B `config.json` (verbatim shape).
    fn zaya1_8b_config() -> serde_json::Value {
        serde_json::json!({
            "activation_func": "swiglu",
            "add_bias_linear": false,
            "architectures": ["ZayaForCausalLM"],
            "attention_bias": false,
            "bos_token_id": 2,
            "cca": true,
            "dtype": "bfloat16",
            "eos_token_id": 106,
            "ffn_hidden_size": 4096,
            "gated_linear_unit": true,
            "hidden_size": 2048,
            "head_dim": 128,
            "kv_channels": 128,
            "lm_head_bias": false,
            "mamba_cache_dtype": "float32",
            "max_position_embeddings": 131072,
            "model_type": "zaya",
            "moe_router_topk": 1,
            "norm_epsilon": 1e-05,
            "normalization": "RMSNorm",
            "num_attention_heads": 8,
            "num_experts": 16,
            "num_hidden_layers": 80,
            "num_key_value_heads": 2,
            "num_query_groups": 2,
            "pad_token_id": 0,
            "partial_rotary_factor": 0.5,
            "residual_in_fp32": true,
            "rope_scaling": false,
            "rope_theta": 5000000,
            "scale_residual_merge": true,
            "sliding_window": null,
            "vocab_size": 262272,
            "zaya_mlp_expansion": 256,
            "zaya_use_eda": true,
            "zaya_use_mod": true
        })
    }

    #[test]
    fn parses_zaya1_8b() {
        let cfg = ZayaConfig::from_json(&zaya1_8b_config()).expect("parse");

        // Alternating 80 half-layers collapse to 40 hybrid blocks.
        assert_eq!(cfg.num_half_layers, 80);
        assert_eq!(cfg.num_blocks, 40);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.vocab_size, 262272);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert!(cfg.tie_word_embeddings);
        assert!(!cfg.lm_head_bias);
        assert_eq!(cfg.eos_token_id, 106);
        assert_eq!(cfg.windows.len(), 40);
        assert!(cfg.windows.iter().all(|&w| w == AttnWindow::Full));
        assert_eq!(cfg.sliding_window, None);
    }

    #[test]
    fn cca_attention_dims() {
        let cfg = ZayaConfig::from_json(&zaya1_8b_config()).expect("parse");
        let a = cfg.attn;
        assert_eq!(a.num_heads, 8);
        assert_eq!(a.num_kv_heads, 2);
        assert_eq!(a.head_dim, 128);
        assert_eq!(cfg.num_kv_groups(), 4);
        // partial rotary 0.5 over head_dim 128.
        assert_eq!(a.n_rot, 64);
        assert_eq!(a.rope_theta, 5_000_000.0);
        // q(8*128=1024) + k(2*128=256) = 1280 conv channels.
        assert_eq!(a.conv_channels(), 1280);
        // kernel 2 + 2 -> cached conv state width (2-1)+(2-1) = 2.
        assert_eq!(a.conv_depthwise_kernel, 2);
        assert_eq!(a.conv_grouped_kernel, 2);
        assert_eq!(a.conv_state_len(), 2);
    }

    #[test]
    fn eda_mod_moe_dims() {
        let cfg = ZayaConfig::from_json(&zaya1_8b_config()).expect("parse");
        let m = cfg.moe;
        assert_eq!(m.num_experts, 16);
        assert_eq!(m.top_k, 1);
        // ffn_hidden_size 4096 is fused gate+up; per-expert SwiGLU width is 2048.
        assert_eq!(m.moe_intermediate_size, 2048);
        // router hidden width carried as zaya_mlp_expansion.
        assert_eq!(m.router_hidden_size, 256);
        assert!(m.use_eda);
        assert!(m.use_mod);
        // 16 real experts + 1 MoD skip route = 17 router logits.
        assert_eq!(m.num_router_experts(), 17);
    }

    /// The native (post-conversion) config.json shape, with `num_hidden_layers`
    /// already counting hybrid blocks and the HF-native field names.
    fn zaya1_8b_native_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["ZayaForCausalLM"],
            "attention_bias": false,
            "bos_token_id": 2,
            "cca_time0": 2,
            "cca_time1": 2,
            "dtype": "bfloat16",
            "eos_token_id": 106,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "lm_head_bias": false,
            "max_position_embeddings": 131072,
            "model_type": "zaya",
            "moe_intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_experts": 16,
            "num_experts_per_tok": 1,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-05,
            "router_hidden_size": 256,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "vocab_size": 262272
        })
    }

    #[test]
    fn native_and_megatron_agree() {
        let mega = ZayaConfig::from_json(&zaya1_8b_config()).expect("megatron parse");
        let native = ZayaConfig::from_json(&zaya1_8b_native_config()).expect("native parse");
        // Both layouts must resolve to the same hybrid-block shape.
        assert_eq!(native.num_blocks, 40);
        assert_eq!(native.num_half_layers, 40); // native carries no half-layer notion
        assert_eq!(mega.num_blocks, native.num_blocks);
        assert_eq!(mega.hidden_size, native.hidden_size);
        assert_eq!(mega.rms_norm_eps, native.rms_norm_eps);
        assert_eq!(mega.attn, native.attn);
        assert_eq!(mega.moe, native.moe);
        assert_eq!(mega.windows, native.windows);
        assert_eq!(native.moe.moe_intermediate_size, 2048);
        assert_eq!(native.moe.num_router_experts(), 17);
        assert_eq!(native.attn.n_rot, 64);
    }

    #[test]
    fn half_layer_roles_alternate() {
        assert_eq!(HalfLayerKind::at(0), HalfLayerKind::Attention);
        assert_eq!(HalfLayerKind::at(1), HalfLayerKind::Moe);
        assert_eq!(HalfLayerKind::at(78), HalfLayerKind::Attention);
        assert_eq!(HalfLayerKind::at(79), HalfLayerKind::Moe);
    }

    #[test]
    fn rejects_odd_layer_count() {
        let mut c = zaya1_8b_config();
        c["num_hidden_layers"] = serde_json::json!(79);
        assert!(ZayaConfig::from_json(&c).is_err());
    }

    #[test]
    fn rejects_unsupported_topk() {
        let mut c = zaya1_8b_config();
        c["moe_router_topk"] = serde_json::json!(2);
        assert!(ZayaConfig::from_json(&c).is_err());
    }
}
