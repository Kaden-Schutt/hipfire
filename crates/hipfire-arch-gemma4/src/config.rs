// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 dense (text-only) config.
//!
//! Parsed from the HFQ `metadata_json.config.text_config.*` envelope. Stage-1
//! is dense text-only: MoE and multimodal towers are out of scope, while dense
//! Gemma4 features such as KV sharing, double-wide MLP layers, and per-layer
//! inputs are represented in the config so the forward path can opt in safely.

// ─── Layer / RoPE types ─────────────────────────────────────────────────

/// Per-layer attention type, read directly from the `layer_types` array.
/// Different Gemma 4 products use different sliding/full periods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Sliding-window causal attention.
    Sliding,
    /// Full (global) causal attention (head_dim = 512 for supported E-series).
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

/// E-series text towers supported by the first Gemma 4 runtime increment.
///
/// This is deliberately shape-derived. Model names are packaging metadata and
/// are not sufficient to select kernels or attach an E2B-only assistant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4ESeriesVariant {
    E2B,
    E4B,
}

// ─── Config ─────────────────────────────────────────────────────────────

/// Typed Gemma 4 dense-text shape constants.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    // Common
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,

    // Attention heads (same count for sliding + full)
    pub n_heads: usize,

    // Sliding-window attention
    pub sliding_head_dim: usize,
    pub sliding_n_kv_heads: usize,
    pub sliding_rope_theta: f32,
    pub sliding_rope_type: RopeType,
    pub sliding_window: usize,

    // Full (global) attention
    pub full_head_dim: usize,            // global_head_dim = 512
    pub full_n_kv_heads: usize,          // num_global_key_value_heads (def = sliding)
    pub full_rope_theta: f32,            // 1_000_000.0
    pub full_rope_type: RopeType,        // Proportional
    pub full_partial_rotary_factor: f32, // 0.25
    pub attention_k_eq_v: bool,          // false for the supported E-series checkpoints

    // FFN (SwiGLU, gelu_pytorch_tanh)
    pub hidden_dim: usize, // intermediate_size; doubled on shared layers when enabled
    pub use_double_wide_mlp: bool,

    // Per-layer input embedding (PLE) support. A nonzero value means the model
    // carries embed_tokens_per_layer / per_layer_* weights and each decoder
    // layer needs an additional residual branch.
    pub hidden_size_per_layer_input: usize,
    pub vocab_size_per_layer_input: usize,

    // KV sharing. Layers at or after `n_layers - num_kv_shared_layers` reuse
    // the last non-shared layer of the same attention type for K/V.
    pub num_kv_shared_layers: usize,

    // Output
    pub final_logit_softcapping: f32, // 30.0 — tanh(x/30)*30
    pub tie_word_embeddings: bool,    // true — lm_head aliases embed_tokens
    pub embed_scale: f32,             // sqrt(dim), applied at embed lookup

    pub max_position_embeddings: usize,

    // Per-layer dispatch (len == n_layers)
    pub layer_types: Vec<LayerType>,

    /// When set (via `HIPFIRE_GEMMA4_NORM_PLUS_ONE=1`), all RMSNorm weights are
    /// baked with the Gemma-2/3 `x * (1 + w)` convention at load time. The
    /// supported Gemma4 E-series checkpoints use plain `x * w`, so this is OFF
    /// by default and remains a developer diagnostic only.
    pub norm_plus_one: bool,
}

impl Gemma4Config {
    pub fn from_hfq(hfq: &hipfire_runtime::hfq::HfqFile) -> Result<Self, String> {
        Self::from_metadata_json(&hfq.metadata_json)
    }

    /// Parse the HFQ metadata envelope without opening an HFQ payload.
    ///
    /// Quantization and loader tests use this entry point so config admission
    /// can be checked independently of GPU resources and packed weights.
    pub fn from_metadata_json(metadata_json: &str) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json)
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
        if n_heads == 0 {
            return Err("gemma4: num_attention_heads must be nonzero".into());
        }

        // Sliding attention.
        let sliding_head_dim = getu(tc, "head_dim")
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let sliding_n_kv_heads = getu(tc, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
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
        let sliding_rope_type = match sliding_rope
            .and_then(|r| r.get("rope_type"))
            .and_then(|v| v.as_str())
        {
            Some("proportional") => RopeType::Proportional,
            Some("default") | None => RopeType::Default,
            Some(other) => {
                return Err(format!(
                    "gemma4: unsupported sliding_attention rope_type={other:?}"
                ));
            }
        };
        let full_rope_theta = full_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(1_000_000.0) as f32;
        let full_rope_type = match full_rope
            .and_then(|r| r.get("rope_type"))
            .and_then(|v| v.as_str())
        {
            Some("proportional") => RopeType::Proportional,
            Some("default") | None => RopeType::Default,
            Some(other) => {
                return Err(format!(
                    "gemma4: unsupported full_attention rope_type={other:?}"
                ));
            }
        };
        let full_partial_rotary_factor = full_rope
            .and_then(|r| getf(r, "partial_rotary_factor"))
            .unwrap_or(1.0) as f32;

        let hidden_dim =
            getu(tc, "intermediate_size").ok_or("gemma4: missing intermediate_size")? as usize;
        let use_double_wide_mlp = getb(tc, "use_double_wide_mlp").unwrap_or(false);
        let hidden_size_per_layer_input =
            getu(tc, "hidden_size_per_layer_input").unwrap_or(0) as usize;
        let vocab_size_per_layer_input =
            getu(tc, "vocab_size_per_layer_input").unwrap_or(vocab_size as u64) as usize;
        let num_kv_shared_layers = getu(tc, "num_kv_shared_layers").unwrap_or(0) as usize;

        let final_logit_softcapping = getf(tc, "final_logit_softcapping").unwrap_or(0.0) as f32;
        let tie_word_embeddings = getb(tc, "tie_word_embeddings")
            .or_else(|| getb(config, "tie_word_embeddings"))
            .unwrap_or(true);

        let embed_scale = (dim as f32).sqrt();
        let max_position_embeddings =
            getu(tc, "max_position_embeddings").unwrap_or(262_144) as usize;

        // layer_types: array of "sliding_attention" / "full_attention". READ it
        // — do not assume the 5:1 period.
        let layer_values = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "gemma4: missing layer_types array".to_string())?;
        let layer_types: Vec<LayerType> = layer_values
            .iter()
            .enumerate()
            .map(|(layer_idx, value)| match value.as_str() {
                Some("sliding_attention") => Ok(LayerType::Sliding),
                Some("full_attention") => Ok(LayerType::Full),
                Some(other) => Err(format!(
                    "gemma4: unsupported layer_types[{layer_idx}]={other:?}"
                )),
                None => Err(format!("gemma4: layer_types[{layer_idx}] must be a string")),
            })
            .collect::<Result<_, _>>()?;
        validate_layer_metadata(n_layers, num_kv_shared_layers, &layer_types)?;

        let norm_plus_one = std::env::var("HIPFIRE_GEMMA4_NORM_PLUS_ONE")
            .ok()
            .as_deref()
            == Some("1");

        let cfg = Gemma4Config {
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
            sliding_rope_type,
            sliding_window,
            full_head_dim,
            full_n_kv_heads,
            full_rope_theta,
            full_rope_type,
            full_partial_rotary_factor,
            attention_k_eq_v,
            hidden_dim,
            use_double_wide_mlp,
            hidden_size_per_layer_input,
            vocab_size_per_layer_input,
            num_kv_shared_layers,
            final_logit_softcapping,
            tie_word_embeddings,
            embed_scale,
            max_position_embeddings,
            layer_types,
            norm_plus_one,
        };
        validate_common_shapes(&cfg)?;
        Ok(cfg)
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

    /// Number of physical full-attention KV slots. E-series shared layers
    /// consume a preceding same-type slot instead of allocating their own.
    pub fn n_full_kv_slots(&self) -> usize {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(layer_idx, layer_type)| {
                **layer_type == LayerType::Full && !self.is_kv_shared_layer(*layer_idx)
            })
            .count()
    }

    /// Number of physical sliding-attention KV slots.
    pub fn n_sliding_kv_slots(&self) -> usize {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(layer_idx, layer_type)| {
                **layer_type == LayerType::Sliding && !self.is_kv_shared_layer(*layer_idx)
            })
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

    pub fn first_kv_shared_layer_idx(&self) -> Option<usize> {
        if self.num_kv_shared_layers == 0 || self.num_kv_shared_layers >= self.n_layers {
            None
        } else {
            Some(self.n_layers - self.num_kv_shared_layers)
        }
    }

    pub fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        self.first_kv_shared_layer_idx()
            .is_some_and(|first| layer_idx >= first)
    }

    pub fn kv_shared_source_layer_idx(&self, layer_idx: usize) -> Option<usize> {
        let first = self.first_kv_shared_layer_idx()?;
        if layer_idx < first || layer_idx >= self.n_layers {
            return None;
        }
        let layer_type = *self.layer_types.get(layer_idx)?;
        self.layer_types[..first]
            .iter()
            .rposition(|&prev_type| prev_type == layer_type)
    }

    pub fn ffn_hidden_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.use_double_wide_mlp && self.is_kv_shared_layer(layer_idx) {
            self.hidden_dim * 2
        } else {
            self.hidden_dim
        }
    }

    pub fn max_ffn_hidden_dim(&self) -> usize {
        if self.use_double_wide_mlp && self.first_kv_shared_layer_idx().is_some() {
            self.hidden_dim * 2
        } else {
            self.hidden_dim
        }
    }

    /// Identify an E-series topology that has an implemented runtime contract.
    ///
    /// The checks are intentionally strict. Unknown Gemma 4 variants remain
    /// parseable for tooling, but must not enter the E-series execution path.
    pub fn e_series_variant(&self) -> Result<Gemma4ESeriesVariant, String> {
        let common = self.vocab_size == 262_144
            && self.n_heads == 8
            && self.sliding_head_dim == 256
            && self.full_head_dim == 512
            && !self.attention_k_eq_v
            && self.sliding_rope_type == RopeType::Default
            && self.full_rope_type == RopeType::Proportional
            && approximately_equal(self.sliding_rope_theta, 10_000.0)
            && approximately_equal(self.full_rope_theta, 1_000_000.0)
            && approximately_equal(self.full_partial_rotary_factor, 0.25)
            && self.hidden_size_per_layer_input == 256
            && self.vocab_size_per_layer_input == 262_144
            && self.sliding_window == 512
            && self.max_position_embeddings == 131_072
            && self.tie_word_embeddings
            && (self.final_logit_softcapping - 30.0).abs() <= f32::EPSILON;

        if common
            && self.dim == 1536
            && self.n_layers == 35
            && self.sliding_n_kv_heads == 1
            && self.full_n_kv_heads == 1
            && self.hidden_dim == 6144
            && self.num_kv_shared_layers == 20
            && self.use_double_wide_mlp
            && matches_attention_period(&self.layer_types, 5)
        {
            return Ok(Gemma4ESeriesVariant::E2B);
        }

        if common
            && self.dim == 2560
            && self.n_layers == 42
            && self.sliding_n_kv_heads == 2
            && self.full_n_kv_heads == 2
            && self.hidden_dim == 10_240
            && self.num_kv_shared_layers == 18
            && !self.use_double_wide_mlp
            && matches_attention_period(&self.layer_types, 6)
        {
            return Ok(Gemma4ESeriesVariant::E4B);
        }

        Err(format!(
            "gemma4: unsupported E-series topology: dim={} layers={} q_heads={} kv_heads={}/{} hidden_dim={} shared_layers={} double_wide_mlp={}",
            self.dim,
            self.n_layers,
            self.n_heads,
            self.sliding_n_kv_heads,
            self.full_n_kv_heads,
            self.hidden_dim,
            self.num_kv_shared_layers,
            self.use_double_wide_mlp
        ))
    }
}

fn approximately_equal(value: f32, expected: f32) -> bool {
    let scale = expected.abs().max(1.0);
    (value - expected).abs() <= scale * 1e-6
}

fn matches_attention_period(layer_types: &[LayerType], period: usize) -> bool {
    period != 0
        && layer_types
            .iter()
            .enumerate()
            .all(|(layer_idx, layer_type)| {
                let expected = if (layer_idx + 1) % period == 0 {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                };
                *layer_type == expected
            })
}

fn validate_common_shapes(cfg: &Gemma4Config) -> Result<(), String> {
    if cfg.dim == 0 || cfg.hidden_dim == 0 || cfg.n_heads == 0 {
        return Err("gemma4: hidden, intermediate, and head dimensions must be nonzero".into());
    }
    for (label, kv_heads) in [
        ("num_key_value_heads", cfg.sliding_n_kv_heads),
        ("num_global_key_value_heads", cfg.full_n_kv_heads),
    ] {
        if kv_heads == 0 || cfg.n_heads % kv_heads != 0 {
            return Err(format!(
                "gemma4: num_attention_heads={} must be divisible by {label}={kv_heads}",
                cfg.n_heads
            ));
        }
    }
    Ok(())
}

fn validate_layer_metadata(
    n_layers: usize,
    num_kv_shared_layers: usize,
    layer_types: &[LayerType],
) -> Result<(), String> {
    if layer_types.len() != n_layers {
        return Err(format!(
            "gemma4: layer_types length {} does not match num_hidden_layers {}",
            layer_types.len(),
            n_layers
        ));
    }
    if num_kv_shared_layers >= n_layers && num_kv_shared_layers != 0 {
        return Err(format!(
            "gemma4: num_kv_shared_layers={} must be in 0..num_hidden_layers ({})",
            num_kv_shared_layers, n_layers
        ));
    }
    if num_kv_shared_layers != 0 {
        let first_shared = n_layers - num_kv_shared_layers;
        for layer_idx in first_shared..n_layers {
            let layer_type = layer_types[layer_idx];
            let has_source = layer_types[..first_shared]
                .iter()
                .any(|&prev_type| prev_type == layer_type);
            if !has_source {
                return Err(format!(
                    "gemma4: shared layer {layer_idx} ({layer_type:?}) has no non-shared source layer of the same type"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn periodic_layer_types(n_layers: usize, period: usize) -> Vec<LayerType> {
        (0..n_layers)
            .map(|layer_idx| {
                if (layer_idx + 1) % period == 0 {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                }
            })
            .collect()
    }

    fn base_cfg() -> Gemma4Config {
        Gemma4Config {
            dim: 1536,
            n_layers: 35,
            vocab_size: 262144,
            norm_eps: 1e-6,
            bos_token: 2,
            eos_token: 1,
            pad_token: 0,
            n_heads: 8,
            sliding_head_dim: 256,
            sliding_n_kv_heads: 1,
            sliding_rope_theta: 10_000.0,
            sliding_rope_type: RopeType::Default,
            sliding_window: 512,
            full_head_dim: 512,
            full_n_kv_heads: 1,
            full_rope_theta: 1_000_000.0,
            full_rope_type: RopeType::Proportional,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: false,
            hidden_dim: 6144,
            use_double_wide_mlp: true,
            hidden_size_per_layer_input: 256,
            vocab_size_per_layer_input: 262144,
            num_kv_shared_layers: 20,
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            embed_scale: (1536.0f32).sqrt(),
            max_position_embeddings: 131072,
            layer_types: vec![LayerType::Sliding; 35],
            norm_plus_one: false,
        }
    }

    #[test]
    fn kv_shared_layers_drive_double_wide_ffn_dims() {
        let cfg = base_cfg();
        assert_eq!(cfg.first_kv_shared_layer_idx(), Some(15));
        assert!(!cfg.is_kv_shared_layer(14));
        assert!(cfg.is_kv_shared_layer(15));
        assert_eq!(cfg.kv_shared_source_layer_idx(14), None);
        assert_eq!(cfg.kv_shared_source_layer_idx(15), Some(14));
        assert_eq!(cfg.kv_shared_source_layer_idx(34), Some(14));
        assert_eq!(cfg.ffn_hidden_dim_for_layer(14), 6144);
        assert_eq!(cfg.ffn_hidden_dim_for_layer(15), 12288);
        assert_eq!(cfg.max_ffn_hidden_dim(), 12288);
    }

    #[test]
    fn kv_shared_source_tracks_last_non_shared_layer_of_same_type() {
        let mut cfg = base_cfg();
        cfg.layer_types = vec![
            LayerType::Sliding,
            LayerType::Full,
            LayerType::Sliding,
            LayerType::Full,
            LayerType::Sliding,
            LayerType::Full,
        ];
        cfg.n_layers = cfg.layer_types.len();
        cfg.num_kv_shared_layers = 2;
        assert_eq!(cfg.first_kv_shared_layer_idx(), Some(4));
        assert_eq!(cfg.kv_shared_source_layer_idx(4), Some(2));
        assert_eq!(cfg.kv_shared_source_layer_idx(5), Some(3));
    }

    #[test]
    fn double_wide_disabled_keeps_base_ffn_dims() {
        let mut cfg = base_cfg();
        cfg.use_double_wide_mlp = false;
        assert_eq!(cfg.ffn_hidden_dim_for_layer(15), 6144);
        assert_eq!(cfg.max_ffn_hidden_dim(), 6144);
    }

    #[test]
    fn validate_layer_metadata_rejects_malformed_shared_configs() {
        assert!(validate_layer_metadata(4, 0, &[LayerType::Sliding; 3]).is_err());
        assert!(validate_layer_metadata(4, 4, &[LayerType::Sliding; 4]).is_err());
        assert!(validate_layer_metadata(
            4,
            1,
            &[
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Full,
            ],
        )
        .is_err());
        assert!(validate_layer_metadata(
            4,
            1,
            &[
                LayerType::Sliding,
                LayerType::Full,
                LayerType::Sliding,
                LayerType::Full,
            ],
        )
        .is_ok());
    }

    #[test]
    fn recognizes_e2b_from_exact_text_topology() {
        let mut cfg = base_cfg();
        cfg.layer_types = periodic_layer_types(35, 5);

        assert_eq!(cfg.e_series_variant(), Ok(Gemma4ESeriesVariant::E2B));
        assert_eq!(cfg.first_kv_shared_layer_idx(), Some(15));
        assert_eq!(cfg.kv_shared_source_layer_idx(15), Some(13));
        assert_eq!(cfg.kv_shared_source_layer_idx(19), Some(14));
        assert_eq!(cfg.n_sliding_kv_slots(), 12);
        assert_eq!(cfg.n_full_kv_slots(), 3);
    }

    #[test]
    fn parses_and_recognizes_e4b_text_config() {
        let layer_types: Vec<&str> = (0..42)
            .map(|layer_idx| {
                if (layer_idx + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let metadata = serde_json::json!({
            "architecture": "gemma4",
            "config": {
                "model_type": "gemma4",
                "tie_word_embeddings": true,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 2560,
                    "num_hidden_layers": 42,
                    "vocab_size": 262144,
                    "rms_norm_eps": 1e-6,
                    "bos_token_id": 2,
                    "eos_token_id": 1,
                    "pad_token_id": 0,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "num_key_value_heads": 2,
                    "global_head_dim": 512,
                    "num_global_key_value_heads": null,
                    "sliding_window": 512,
                    "intermediate_size": 10240,
                    "hidden_size_per_layer_input": 256,
                    "vocab_size_per_layer_input": 262144,
                    "num_kv_shared_layers": 18,
                    "use_double_wide_mlp": false,
                    "final_logit_softcapping": 30.0,
                    "tie_word_embeddings": true,
                    "max_position_embeddings": 131072,
                    "layer_types": layer_types,
                    "rope_parameters": {
                        "sliding_attention": {
                            "rope_theta": 10000.0,
                            "rope_type": "default"
                        },
                        "full_attention": {
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional",
                            "partial_rotary_factor": 0.25
                        }
                    }
                }
            }
        })
        .to_string();

        let cfg = Gemma4Config::from_metadata_json(&metadata).unwrap();
        assert_eq!(cfg.e_series_variant(), Ok(Gemma4ESeriesVariant::E4B));
        assert_eq!(cfg.first_kv_shared_layer_idx(), Some(24));
        assert_eq!(cfg.kv_shared_source_layer_idx(24), Some(22));
        assert_eq!(cfg.kv_shared_source_layer_idx(29), Some(23));
        assert_eq!(cfg.ffn_hidden_dim_for_layer(41), 10_240);
        assert_eq!(cfg.max_ffn_hidden_dim(), 10_240);
        assert_eq!(cfg.n_sliding_kv_slots(), 20);
        assert_eq!(cfg.n_full_kv_slots(), 4);
    }

    #[test]
    fn parses_dense_12b_without_selecting_e_series() {
        let layer_types: Vec<&str> = (0..48)
            .map(|layer_idx| {
                if (layer_idx + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let metadata = serde_json::json!({
            "config": {
                "model_type": "gemma4_unified",
                "tie_word_embeddings": true,
                "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 3840,
                    "num_hidden_layers": 48,
                    "vocab_size": 262144,
                    "rms_norm_eps": 1e-6,
                    "bos_token_id": 2,
                    "eos_token_id": [1, 106],
                    "pad_token_id": 0,
                    "num_attention_heads": 16,
                    "head_dim": 256,
                    "num_key_value_heads": 8,
                    "global_head_dim": 512,
                    "num_global_key_value_heads": 8,
                    "sliding_window": 1024,
                    "intermediate_size": 15360,
                    "attention_k_eq_v": true,
                    "final_logit_softcapping": 30.0,
                    "max_position_embeddings": 262144,
                    "layer_types": layer_types,
                    "rope_parameters": {
                        "sliding_attention": {
                            "rope_theta": 10000.0,
                            "rope_type": "default"
                        },
                        "full_attention": {
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional",
                            "partial_rotary_factor": 0.25
                        }
                    }
                }
            }
        })
        .to_string();

        let cfg = Gemma4Config::from_metadata_json(&metadata).unwrap();
        assert_eq!(cfg.dim, 3840);
        assert_eq!(cfg.n_layers, 48);
        assert_eq!(cfg.n_full_layers(), 8);
        assert_eq!(cfg.hidden_size_per_layer_input, 0);
        assert_eq!(cfg.num_kv_shared_layers, 0);
        assert!(!cfg.use_double_wide_mlp);
        assert!(cfg.e_series_variant().is_err());
    }

    #[test]
    fn e_series_detection_fails_closed_on_attention_pattern_change() {
        let mut cfg = base_cfg();
        cfg.layer_types = periodic_layer_types(35, 5);
        cfg.layer_types.swap(3, 4);

        assert!(cfg.e_series_variant().is_err());
    }

    #[test]
    fn e_series_detection_fails_closed_on_full_attention_semantic_changes() {
        let mut cfg = base_cfg();
        cfg.layer_types = periodic_layer_types(35, 5);

        cfg.attention_k_eq_v = true;
        assert!(cfg.e_series_variant().is_err());

        cfg.attention_k_eq_v = false;
        cfg.full_rope_type = RopeType::Default;
        assert!(cfg.e_series_variant().is_err());

        cfg.full_rope_type = RopeType::Proportional;
        cfg.full_partial_rotary_factor = 1.0;
        assert!(cfg.e_series_variant().is_err());

        cfg.full_partial_rotary_factor = 0.25;
        cfg.full_rope_theta = 10_000.0;
        assert!(cfg.e_series_variant().is_err());

        cfg.full_rope_theta = 1_000_000.0;
        cfg.sliding_rope_theta = 1_000_000.0;
        assert!(cfg.e_series_variant().is_err());

        cfg.sliding_rope_theta = 10_000.0;
        cfg.sliding_rope_type = RopeType::Proportional;
        assert!(cfg.e_series_variant().is_err());
    }

    #[test]
    fn metadata_parser_rejects_unknown_layer_type() {
        let metadata = serde_json::json!({
            "config": {
                "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 1,
                    "head_dim": 8,
                    "num_key_value_heads": 1,
                    "intermediate_size": 16,
                    "layer_types": ["future_attention"]
                }
            }
        })
        .to_string();

        let error = Gemma4Config::from_metadata_json(&metadata).unwrap_err();
        assert!(error.contains("unsupported layer_types[0]"));
    }

    #[test]
    fn metadata_parser_rejects_unknown_rope_type() {
        let metadata = serde_json::json!({
            "config": {
                "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 1,
                    "head_dim": 8,
                    "num_key_value_heads": 1,
                    "intermediate_size": 16,
                    "layer_types": ["sliding_attention"],
                    "rope_parameters": {
                        "sliding_attention": {
                            "rope_type": "future_rope"
                        }
                    }
                }
            }
        })
        .to_string();

        let error = Gemma4Config::from_metadata_json(&metadata).unwrap_err();
        assert!(error.contains("unsupported sliding_attention rope_type"));
    }

    #[test]
    fn metadata_parser_rejects_zero_query_heads_without_panicking() {
        let metadata = serde_json::json!({
            "config": {
                "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                    "num_attention_heads": 0,
                    "intermediate_size": 16,
                    "layer_types": ["sliding_attention"]
                }
            }
        })
        .to_string();

        let error = Gemma4Config::from_metadata_json(&metadata).unwrap_err();
        assert_eq!(error, "gemma4: num_attention_heads must be nonzero");
    }
}
