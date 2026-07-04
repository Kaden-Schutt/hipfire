// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 config parsing. See LICENSE / NOTICE.

//! [`Gemma3Config`] and the HFQ-metadata parser.
//!
//! The quantizer embeds the full original `config.json` under `metadata.config`
//! and tags `metadata.architecture` (`"gemma3_text"` for `Gemma3ForCausalLM`, or
//! `"gemma3"` for the multimodal `Gemma3ForConditionalGeneration` wrapper, whose
//! text fields live under `config.text_config`). We read whichever applies, so
//! one parser covers both the text-only SKUs (medgemma-27b-text-it) and the
//! text tower of the multimodal SKUs.

use hipfire_runtime::hfq::HfqFile;

/// Gemma3 text-decoder shape constants. Cheap to clone, `Send`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Independent of `hidden_size / num_attention_heads` (128 @27b, 256 @4b).
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    /// RoPE base for the **global** (full-causal) layers.
    pub rope_theta: f32,
    /// RoPE base for the **local** (sliding-window) layers.
    pub rope_local_base_freq: f32,
    /// Sliding-window span for local layers (causal mask width).
    pub sliding_window: usize,
    /// Layer-type period: every `sliding_window_pattern`-th layer is global,
    /// the rest are local sliding-window. Gemma3 default 6 (5 local : 1 global).
    pub sliding_window_pattern: usize,
    /// Query pre-attention scalar `s`; the attention score scale is `s^-0.5`
    /// (NOT `1/√head_dim` — they differ on the 27b, where `s = 168 ≠ 128`).
    pub query_pre_attn_scalar: f32,
    /// Expected `"gelu_pytorch_tanh"` (GeGLU). Retained for validation.
    pub hidden_activation: String,
    /// Gemma3 ties the lm_head to the embedding table (default true); the
    /// loader re-uploads the embedding bytes as the output projection.
    pub tie_word_embeddings: bool,
    /// The `(1+w)` offset the quantizer baked into the norm weights at ingest
    /// (`1.0` for a correctly-ingested Gemma3 artifact). Stored for provenance;
    /// the forward applies no further offset because it is already in the
    /// stored weights.
    pub gemma_norm_offset: f32,
    /// Generation stop token. Gemma3-it ends a turn with `<end_of_turn>` (106),
    /// not the base `<eos>` (1); config.json may give either a scalar or a
    /// `[1, 106]` array. The serving seam stops on this id.
    pub eos_token_id: u32,
}

impl Gemma3Config {
    /// Embedding normalizer — Gemma scales the looked-up embedding by
    /// `√hidden_size` before the first decoder layer.
    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    /// Attention score scale: `query_pre_attn_scalar^-0.5`. Applied to QK^T
    /// (replaces the usual `1/√head_dim`).
    pub fn attn_scale(&self) -> f32 {
        self.query_pre_attn_scalar.powf(-0.5)
    }

    /// Pre-scale applied to Q so the attention kernel's built-in `1/√head_dim`
    /// softmax scale equals Gemma's `1/√query_pre_attn_scalar`:
    /// `√(head_dim / query_pre_attn_scalar)`. It is `1.0` when
    /// `query_pre_attn_scalar == head_dim` (e.g. gemma3-4b). The loader bakes
    /// this into the `q_norm` weights so no per-step scale launch is needed.
    pub fn q_prescale(&self) -> f32 {
        (self.head_dim as f32 / self.query_pre_attn_scalar).sqrt()
    }

    /// True if layer `layer_idx` (0-based) uses **global** full-causal attention
    /// (θ=`rope_theta`); false ⇒ a **local** sliding-window layer
    /// (θ=`rope_local_base_freq`, mask width `sliding_window`). Gemma3 makes
    /// every `sliding_window_pattern`-th layer global: HF's
    /// `is_sliding = (layer_idx + 1) % sliding_window_pattern != 0`.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        self.sliding_window_pattern > 0
            && (layer_idx + 1).is_multiple_of(self.sliding_window_pattern)
    }

    /// Per-layer RoPE base, selecting the global vs local theta.
    pub fn rope_base_for_layer(&self, layer_idx: usize) -> f32 {
        if self.is_global_layer(layer_idx) {
            self.rope_theta
        } else {
            self.rope_local_base_freq
        }
    }
}

/// Parse a [`Gemma3Config`] from an HFQ file's embedded metadata.
pub fn config_from_hfq(hfq: &HfqFile) -> Option<Gemma3Config> {
    config_from_metadata_json(&hfq.metadata_json)
}

/// Inner parser, decoupled from `HfqFile` for unit testability.
pub fn config_from_metadata_json(metadata_json: &str) -> Option<Gemma3Config> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let config = meta.get("config")?;
    // `gemma3` (multimodal) nests the decoder shape under `text_config`;
    // `gemma3_text` keeps it top-level. Prefer the nested block when present.
    let tc = config.get("text_config").unwrap_or(config);

    let hidden_size = tc.get("hidden_size")?.as_u64()? as usize;
    let num_hidden_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let num_attention_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
    let num_key_value_heads = tc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads as u64) as usize;
    // Gemma3 sets head_dim explicitly and it is NOT hidden_size/n_heads.
    let head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = tc.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let max_position_embeddings = tc
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(131072) as usize;
    let rms_norm_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let rope_theta = tc
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;
    let rope_local_base_freq = tc
        .get("rope_local_base_freq")
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0) as f32;
    let sliding_window = tc
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;
    let sliding_window_pattern = tc
        .get("sliding_window_pattern")
        .and_then(|v| v.as_u64())
        .unwrap_or(6) as usize;
    // query_pre_attn_scalar defaults to head_dim (the standard 1/√head_dim
    // scale) when absent.
    let query_pre_attn_scalar = tc
        .get("query_pre_attn_scalar")
        .and_then(|v| v.as_f64())
        .unwrap_or(head_dim as f64) as f32;
    let hidden_activation = tc
        .get("hidden_activation")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu_pytorch_tanh")
        .to_string();
    let tie_word_embeddings = tc
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    // Baked at ingest; default 0.0 for a non-Gemma / legacy artifact so the
    // absence is visible rather than silently assumed.
    let gemma_norm_offset = meta
        .get("gemma_norm_offset")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    // Gemma3-it turn-ender is `<end_of_turn>` (106). config.json `eos_token_id`
    // can be a scalar (`1`) or an array (`[1, 106]`); when an array, prefer the
    // turn-ender (106) so chat generation actually stops, else take the first.
    let eos_token_id = tc
        .get("eos_token_id")
        .and_then(|v| match v {
            serde_json::Value::Array(a) => {
                let ids: Vec<u64> = a.iter().filter_map(|x| x.as_u64()).collect();
                ids.iter()
                    .copied()
                    .find(|&id| id == 106)
                    .or_else(|| ids.first().copied())
            }
            other => other.as_u64(),
        })
        .unwrap_or(106) as u32;

    Some(Gemma3Config {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        rope_local_base_freq,
        sliding_window,
        sliding_window_pattern,
        query_pre_attn_scalar,
        hidden_activation,
        tie_word_embeddings,
        gemma_norm_offset,
        eos_token_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// medgemma-27b-text-it config.json values (verified 2026-06-19), wrapped
    /// in the HFQ metadata envelope the quantizer emits.
    fn medgemma_27b_text_metadata() -> String {
        serde_json::json!({
            "architecture": "gemma3_text",
            "gemma_norm_offset": 1.0,
            "config": {
                "model_type": "gemma3_text",
                "hidden_size": 5376,
                "num_hidden_layers": 62,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "head_dim": 128,
                "intermediate_size": 21504,
                "vocab_size": 262144,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
                "rope_local_base_freq": 10_000.0,
                "sliding_window": 1024,
                "sliding_window_pattern": 6,
                "query_pre_attn_scalar": 168,
                "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 131072,
                "eos_token_id": [1, 106]
            }
        })
        .to_string()
    }

    #[test]
    fn parses_medgemma_27b_text() {
        let cfg = config_from_metadata_json(&medgemma_27b_text_metadata()).unwrap();
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.num_hidden_layers, 62);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 16);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 262144);
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert_eq!(cfg.query_pre_attn_scalar, 168.0);
        assert_eq!(cfg.hidden_activation, "gelu_pytorch_tanh");
        assert_eq!(cfg.gemma_norm_offset, 1.0);
        // `[1, 106]` → prefer the `<end_of_turn>` turn-ender so chat stops.
        assert_eq!(cfg.eos_token_id, 106);
    }

    #[test]
    fn eos_defaults_to_end_of_turn_when_absent() {
        let meta = serde_json::json!({
            "architecture": "gemma3_text",
            "config": { "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 32 }
        })
        .to_string();
        let cfg = config_from_metadata_json(&meta).unwrap();
        assert_eq!(cfg.eos_token_id, 106);
    }

    #[test]
    fn custom_attn_scale_differs_from_inv_sqrt_head_dim() {
        let cfg = config_from_metadata_json(&medgemma_27b_text_metadata()).unwrap();
        // 27b: query_pre_attn_scalar=168, head_dim=128 — the scales must differ.
        let custom = cfg.attn_scale();
        let naive = (cfg.head_dim as f32).powf(-0.5);
        assert!((custom - 168f32.powf(-0.5)).abs() < 1e-9);
        assert!((custom - naive).abs() > 1e-4, "scales should differ on 27b");
    }

    #[test]
    fn embed_scale_is_sqrt_hidden() {
        let cfg = config_from_metadata_json(&medgemma_27b_text_metadata()).unwrap();
        assert!((cfg.embed_scale() - (5376f32).sqrt()).abs() < 1e-3);
    }

    #[test]
    fn sliding_window_interleave_5_local_1_global() {
        let cfg = config_from_metadata_json(&medgemma_27b_text_metadata()).unwrap();
        // Pattern 6: layers 5, 11, 17, ... (0-based) are global; the rest local.
        let global: Vec<usize> = (0..12).filter(|&i| cfg.is_global_layer(i)).collect();
        assert_eq!(global, vec![5, 11]);
        assert_eq!(cfg.rope_base_for_layer(0), 10_000.0); // local
        assert_eq!(cfg.rope_base_for_layer(5), 1_000_000.0); // global
    }
}
