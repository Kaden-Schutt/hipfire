// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 dense (text-only) config.
//!
//! Parsed from the HFQ `metadata_json.config.text_config.*` envelope for
//! `google/gemma-4-12B-it` (`model_type = gemma4_unified`, text config
//! `gemma4_unified_text`). This is the DENSE 12B port: MoE
//! (`enable_moe_block`), the E-series per-layer-embedding / KV-sharing /
//! double-wide-MLP features, and vision/audio are all DROPPED — the 12B uses
//! none of them. The remaining hybrid structure (5:1 sliding/full attention,
//! K=V sharing on full layers, sandwich RMSNorm, dual RoPE) is preserved.

use hipfire_runtime::hfq::HfqFile;

// ─── Layer / RoPE types ─────────────────────────────────────────────────

/// Per-layer attention type. Gemma 4 12B alternates 5 sliding + 1 full
/// (every 6th layer is full; 8 full layers over 48). Read from the
/// `layer_types` config array; do not assume the period.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Sliding-window causal attention (window = 1024, head_dim = 256).
    Sliding,
    /// Full (global) causal attention (head_dim = 512, K=V sharing).
    Full,
}

/// RoPE flavour for the full-attention layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Standard RoPE: every head_dim position rotates.
    Default,
    /// Proportional / partial RoPE (Gemma 4 full layers): only the first
    /// `partial_rotary_factor × head_dim` positions rotate; the rest are NoPE.
    Proportional,
}

// ─── Config ─────────────────────────────────────────────────────────────

/// Typed Gemma 4 dense-text shape constants.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    // Common
    pub dim: usize,        // hidden_size = 3840 on 12B
    pub n_layers: usize,   // num_hidden_layers = 48
    pub vocab_size: usize, // 262144
    pub norm_eps: f32,     // rms_norm_eps = 1e-6
    pub bos_token: u32,    // 2
    pub eos_token: u32,    // 1 (also 106 = <end_of_turn>)
    pub pad_token: u32,    // 0

    // Attention heads (same count for sliding + full)
    pub n_heads: usize, // num_attention_heads = 16

    // Sliding-window attention
    pub sliding_head_dim: usize,   // head_dim = 256
    pub sliding_n_kv_heads: usize, // num_key_value_heads = 8
    pub sliding_rope_theta: f32,   // 10000.0
    pub sliding_window: usize,     // 1024

    // Full (global) attention
    pub full_head_dim: usize,            // global_head_dim = 512
    pub full_n_kv_heads: usize,          // num_global_key_value_heads (def = sliding)
    pub full_rope_theta: f32,            // 1_000_000.0
    pub full_rope_type: RopeType,        // Proportional
    pub full_partial_rotary_factor: f32, // 0.25
    pub attention_k_eq_v: bool,          // true — full layers' V = pre-k_norm K

    // FFN (SwiGLU, gelu_pytorch_tanh)
    pub hidden_dim: usize, // intermediate_size = 15360

    // Output
    pub final_logit_softcapping: f32, // 30.0 — tanh(x/30)*30
    pub tie_word_embeddings: bool,    // true — lm_head aliases embed_tokens
    pub embed_scale: f32,             // sqrt(dim), applied at embed lookup

    pub max_position_embeddings: usize, // 262144

    // Per-layer dispatch (len == n_layers)
    pub layer_types: Vec<LayerType>,

    /// When set (via `HIPFIRE_GEMMA4_NORM_PLUS_ONE=1`), all RMSNorm weights are
    /// baked with the Gemma-2/3 `x * (1 + w)` convention at load time. Gemma 4
    /// (per the old branch's `load_gemma4_norm` + modeling_gemma4.py) uses plain
    /// `x * w`, so this is OFF by default; the toggle lets us flip if the 12B
    /// turns out to differ.
    pub norm_plus_one: bool,
}

impl Gemma4Config {
    /// Parse from the HFQ metadata envelope. Mirrors the old branch's
    /// `config_from_hfq`, scoped to the dense-text fields.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("gemma4: metadata_json not valid JSON: {e}"))?;
        let config = meta
            .get("config")
            .ok_or_else(|| "gemma4: metadata_json missing `config` wrapper".to_string())?;
        // text_config of gemma4_unified; fall back to top-level for flat configs.
        let tc = config.get("text_config").unwrap_or(config);

        let getu = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let getf = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let getb = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_bool());

        let dim = getu(tc, "hidden_size").ok_or("gemma4: missing hidden_size")? as usize;
        let n_layers =
            getu(tc, "num_hidden_layers").ok_or("gemma4: missing num_hidden_layers")? as usize;
        let vocab_size = getu(tc, "vocab_size").ok_or("gemma4: missing vocab_size")? as usize;
        let norm_eps = getf(tc, "rms_norm_eps").unwrap_or(1e-6) as f32;
        let bos_token = getu(tc, "bos_token_id").unwrap_or(2) as u32;
        // eos_token_id may be an array ([1, 106]); take the first as the primary.
        let eos_token = match tc.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => {
                a.first().and_then(|x| x.as_u64()).unwrap_or(1) as u32
            }
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(1) as u32,
            _ => 1,
        };
        let pad_token = getu(tc, "pad_token_id").unwrap_or(0) as u32;

        let n_heads =
            getu(tc, "num_attention_heads").ok_or("gemma4: missing num_attention_heads")? as usize;

        // Sliding attention.
        let sliding_head_dim = getu(tc, "head_dim").map(|v| v as usize).unwrap_or(dim / n_heads);
        let sliding_n_kv_heads =
            getu(tc, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
        let sliding_window = getu(tc, "sliding_window").unwrap_or(1024) as usize;

        // Full attention (may differ from sliding).
        let full_head_dim = getu(tc, "global_head_dim")
            .map(|v| v as usize)
            .unwrap_or(sliding_head_dim);
        let full_n_kv_heads =
            getu(tc, "num_global_key_value_heads").unwrap_or(sliding_n_kv_heads as u64) as usize;
        let attention_k_eq_v = getb(tc, "attention_k_eq_v").unwrap_or(false);

        // rope_parameters: {sliding_attention:{...}, full_attention:{...}}.
        let rope_params = tc.get("rope_parameters");
        let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
        let full_rope = rope_params.and_then(|r| r.get("full_attention"));

        let sliding_rope_theta = sliding_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(10_000.0) as f32;
        let full_rope_theta = full_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(1_000_000.0) as f32;
        let full_rope_type = match full_rope
            .and_then(|r| r.get("rope_type"))
            .and_then(|v| v.as_str())
        {
            Some("proportional") => RopeType::Proportional,
            _ => RopeType::Default,
        };
        let full_partial_rotary_factor = full_rope
            .and_then(|r| getf(r, "partial_rotary_factor"))
            .unwrap_or(1.0) as f32;

        let hidden_dim =
            getu(tc, "intermediate_size").ok_or("gemma4: missing intermediate_size")? as usize;

        let final_logit_softcapping = getf(tc, "final_logit_softcapping").unwrap_or(0.0) as f32;
        let tie_word_embeddings = getb(tc, "tie_word_embeddings")
            .or_else(|| getb(config, "tie_word_embeddings"))
            .unwrap_or(true);

        let embed_scale = (dim as f32).sqrt();
        let max_position_embeddings =
            getu(tc, "max_position_embeddings").unwrap_or(262_144) as usize;

        // layer_types: array of "sliding_attention" / "full_attention". READ it
        // — do not assume the 5:1 period.
        let layer_types: Vec<LayerType> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| match v.as_str().unwrap_or("sliding_attention") {
                        "full_attention" => LayerType::Full,
                        _ => LayerType::Sliding,
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![LayerType::Sliding; n_layers]);

        let norm_plus_one =
            std::env::var("HIPFIRE_GEMMA4_NORM_PLUS_ONE").ok().as_deref() == Some("1");

        Ok(Gemma4Config {
            dim,
            n_layers,
            vocab_size,
            norm_eps,
            bos_token,
            eos_token,
            pad_token,
            n_heads,
            sliding_head_dim,
            sliding_n_kv_heads,
            sliding_rope_theta,
            sliding_window,
            full_head_dim,
            full_n_kv_heads,
            full_rope_theta,
            full_rope_type,
            full_partial_rotary_factor,
            attention_k_eq_v,
            hidden_dim,
            final_logit_softcapping,
            tie_word_embeddings,
            embed_scale,
            max_position_embeddings,
            layer_types,
            norm_plus_one,
        })
    }

    /// Number of full (global) attention layers — sizes the full KV cache.
    pub fn n_full_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == LayerType::Full)
            .count()
    }

    /// Number of sliding-window attention layers — sizes the sliding KV cache.
    pub fn n_sliding_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == LayerType::Sliding)
            .count()
    }

    /// Max head_dim across the two attention flavours (scratch sizing).
    pub fn max_head_dim(&self) -> usize {
        self.sliding_head_dim.max(self.full_head_dim)
    }

    /// Max q projection width across layer types (scratch sizing).
    pub fn max_q_dim(&self) -> usize {
        (self.n_heads * self.sliding_head_dim).max(self.n_heads * self.full_head_dim)
    }

    /// Max k/v projection width across layer types (scratch sizing).
    pub fn max_kv_dim(&self) -> usize {
        (self.sliding_n_kv_heads * self.sliding_head_dim)
            .max(self.full_n_kv_heads * self.full_head_dim)
    }
}
