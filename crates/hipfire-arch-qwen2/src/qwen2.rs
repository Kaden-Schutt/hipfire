//! Qwen2 model types: Config / Weights / State.
//!
//! Rev 0 status:
//! - `Qwen2Config::from_hfq` — real metadata parser (this commit).
//! - `Qwen2Weights::load` — stub (next commit).
//! - `Qwen2State::new` — stub (next commit).
//! - Forward pass — not yet present; to be ported from
//!   `hipfire-arch-qwen35::qwen35` per plan §5 phase 1.
//!
//! See `docs/plans/qwen_2.5_vlm.md` phase 1 for the full plan.

use hipfire_runtime::hfq::HfqFile;

/// Qwen2 model-shape constants parsed from `HfqFile::metadata_json`.
///
/// At rev 0 these are populated with Qwen2-1.5B-Instruct defaults; the
/// real `from_hfq` walks `hfq.metadata_json` (a JSON blob) and reads the
/// per-model values. See `hipfire_arch_qwen35::qwen35::config_from_hfq`
/// for the production pattern.
///
/// # Field notes
///
/// - `attention_bias`: Qwen2 modeling-code default is `true`. Many Qwen2
///   HF configs omit the field; treat missing as `true`.
/// - `tie_word_embeddings`: differs across Qwen2 checkpoints. 1.5B-Instruct
///   has `true` (no separate lm_head on disk); dots.ocr's Qwen2 backbone
///   has `false`. Loader must detect and handle both.
/// - `rope_theta`: 1_000_000 for all Qwen2 variants seen so far.
/// - `rms_norm_eps`: 1e-6.
#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
}

/// Parse a Qwen2 config out of an HFQ file's metadata. Free-function form
/// (mirrors `qwen35::config_from_hfq`); the trait impl in `arch.rs` wraps
/// `None` into a `Result::Err`.
///
/// Defaults applied for fields that are missing from common Qwen2 configs:
/// - `head_dim` ← `hidden_size / num_attention_heads`
/// - `num_key_value_heads` ← `num_attention_heads` (i.e. MHA fallback)
/// - `attention_bias` ← `true` (Qwen2 modeling default; both
///   Qwen2-1.5B-Instruct and dots.ocr's text backbone match this even
///   when the field is absent from `config.json`)
/// - `tie_word_embeddings` ← `false` (HF transformers default)
/// - `rms_norm_eps` ← `1e-6`
/// - `rope_theta` ← `1_000_000` (Qwen2 default)
/// - `eos_token_id` ← `151645` (`<|im_end|>` in the Qwen2 vocab)
pub fn config_from_hfq(hfq: &HfqFile) -> Option<Qwen2Config> {
    config_from_metadata_json(&hfq.metadata_json)
}

/// Inner parser, decoupled from `HfqFile` for unit testability. Takes the
/// raw `metadata_json` blob and emits `Some(Qwen2Config)` on a recognised
/// Qwen2 shape, `None` otherwise.
pub fn config_from_metadata_json(metadata_json: &str) -> Option<Qwen2Config> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    // Qwen2's HF config.json is flat (no `text_config` nesting like Qwen3.5-VL).
    // Fall back to `config` itself for forward-compat with future nested layouts.
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let hidden_size = tc.get("hidden_size")?.as_u64()? as usize;
    let num_hidden_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let num_attention_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
    let num_key_value_heads = tc.get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads as u64) as usize;
    let head_dim = tc.get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = tc.get("intermediate_size")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let max_position_embeddings = tc.get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(32768) as usize;
    let rope_theta = tc.get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;
    let rms_norm_eps = tc.get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let attention_bias = tc.get("attention_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let tie_word_embeddings = tc.get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // eos_token_id may be a single int (Qwen2-1.5B-Instruct) or an array
    // (dots.ocr). Accept either; pick the first int from an array.
    let eos_token_id = tc.get("eos_token_id")
        .and_then(|v| v.as_u64().or_else(|| {
            v.as_array().and_then(|arr| arr.first().and_then(|e| e.as_u64()))
        }))
        .unwrap_or(151645) as u32;

    Some(Qwen2Config {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rope_theta,
        rms_norm_eps,
        attention_bias,
        tie_word_embeddings,
        eos_token_id,
    })
}

impl Qwen2Config {
    /// Convenience: parse and lift `Option` into `Result`. The trait
    /// impl uses the free-function `config_from_hfq` directly.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        config_from_hfq(hfq)
            .ok_or_else(|| "qwen2: failed to parse config from HFQ metadata".to_string())
    }
}

/// Qwen2 GPU-resident weight handles.
///
/// Rev 0: stub placeholder. Real implementation will hold per-layer
/// `WeightTensor` arrays (attention QKV with bias, output proj, FFN
/// gate/up/down, layernorm scales, embeddings, optional lm_head). See
/// `Qwen35Weights` in `hipfire-arch-qwen35` for the production shape.
pub struct Qwen2Weights {
    /// Placeholder: an embedding tensor sized by config. Replaced in
    /// phase 1 with proper `WeightTensor` handles for every projection.
    pub embeddings: Vec<f32>,
}

impl Qwen2Weights {
    /// Stub loader (rev 0): allocates a zero-initialised embedding vec.
    /// Real implementation walks `hfq.tensor_info(name)` for every weight
    /// tensor and dispatches to `WeightTensor::from_hfq_tensor`. See
    /// plan §5 phase 1.
    pub fn load(_hfq: &HfqFile, cfg: &Qwen2Config) -> Result<Self, String> {
        Ok(Qwen2Weights {
            embeddings: vec![0.0; cfg.vocab_size * cfg.hidden_size],
        })
    }
}

/// Qwen2 per-decode GPU scratch (KV cache + attention workspace).
///
/// Rev 0: stub. Real implementation allocates GPU buffers for the KV
/// cache (sized by num_key_value_heads × head_dim × max_seq_len × layers)
/// and the per-layer attention workspace. See `ForwardScratch::new` in
/// `hipfire-runtime::llama` for the dense-FA reference shape.
pub struct Qwen2State {
    pub token_count: usize,
}

impl Qwen2State {
    /// Stub state init (rev 0): bare counter. Real implementation
    /// allocates KV-cache and attention-scratch GPU buffers sized by
    /// `cfg`.
    pub fn new(_cfg: &Qwen2Config) -> Result<Self, String> {
        Ok(Qwen2State { token_count: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal metadata blob shaped like what `hipfire-quantize` emits:
    /// `{"config": <flat Qwen2 HF config>}`. Values reproduce
    /// Qwen2-1.5B-Instruct's `config.json` exactly (verified
    /// 2026-05-19 against
    /// `/home/kread/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/`).
    const QWEN2_1P5B_METADATA: &str = r#"{
        "config": {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": true,
            "hidden_act": "silu",
            "eos_token_id": 151645,
            "torch_dtype": "bfloat16"
        }
    }"#;

    /// Metadata blob mimicking dots.ocr's HF config: `attention_bias=true`
    /// explicit, `tie_word_embeddings=false`, multi-element `eos_token_id`
    /// array. Shape only — vision_config omitted (dots.ocr crate handles
    /// that, not qwen2).
    const DOTS_OCR_TEXT_METADATA: &str = r#"{
        "config": {
            "architectures": ["DotsOCRForCausalLM"],
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "vocab_size": 151936,
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-06,
            "attention_bias": true,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "eos_token_id": [151643, 151673],
            "torch_dtype": "bfloat16"
        }
    }"#;

    #[test]
    fn parses_qwen2_1p5b_instruct_config() {
        let cfg = config_from_metadata_json(QWEN2_1P5B_METADATA)
            .expect("parser returned None on a valid Qwen2-1.5B-Instruct config");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 128, "head_dim should derive from 1536/12 when absent");
        assert_eq!(cfg.intermediate_size, 8960);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);
        // attention_bias missing from Qwen2-1.5B config.json → defaults to true
        // (matches both the modeling-code default and the on-disk weight reality:
        // q/k/v_proj.bias tensors are present in the safetensors).
        assert!(cfg.attention_bias);
        assert!(cfg.tie_word_embeddings, "1.5B-Instruct has tie_word_embeddings=true");
        assert_eq!(cfg.eos_token_id, 151645);
    }

    #[test]
    fn parses_dots_ocr_text_config() {
        let cfg = config_from_metadata_json(DOTS_OCR_TEXT_METADATA)
            .expect("parser returned None on a valid dots.ocr text config");
        assert!(cfg.attention_bias, "dots.ocr explicitly sets attention_bias=true");
        assert!(!cfg.tie_word_embeddings, "dots.ocr does not tie embeddings");
        // First element of the eos_token_id array.
        assert_eq!(cfg.eos_token_id, 151643);
        assert_eq!(cfg.max_position_embeddings, 131072);
    }

    #[test]
    fn missing_required_field_returns_none() {
        let bad = r#"{"config": {"hidden_size": 1536}}"#; // missing num_hidden_layers etc.
        assert!(config_from_metadata_json(bad).is_none());
    }

    #[test]
    fn missing_optional_fields_get_defaults() {
        // Bare minimum required keys; all optional keys absent.
        let minimal = r#"{
            "config": {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "vocab_size": 32000
            }
        }"#;
        let cfg = config_from_metadata_json(minimal).expect("minimal config should parse");
        assert_eq!(cfg.num_key_value_heads, 12, "fallback to MHA");
        assert_eq!(cfg.head_dim, 64, "fallback to hidden/heads");
        assert!(cfg.attention_bias, "Qwen2 modeling default is true");
        assert!(!cfg.tie_word_embeddings, "HF transformers default is false");
        assert_eq!(cfg.eos_token_id, 151645, "default <|im_end|>");
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    }
}
