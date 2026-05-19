//! Qwen2 model types: Config / Weights / State.
//!
//! Skeleton at rev 0: struct shapes reflect Qwen2 reality but the loaders
//! are stubs ported from the toy template. The forward pass lives outside
//! the trait (free functions on this module — to be added in phase 1) and
//! is dispatched statically by the daemon's generation loop. See
//! `docs/plans/qwen_2.5_vlm.md` phase 1 for the implementation plan.

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
}

impl Qwen2Config {
    /// Stub loader (rev 0): returns Qwen2-1.5B-Instruct defaults regardless
    /// of input. Real implementation walks `hfq.metadata_json` and reads
    /// the per-model values; see plan §5 phase 1.
    pub fn from_hfq(_hfq: &HfqFile) -> Result<Self, String> {
        Ok(Qwen2Config {
            hidden_size: 1536,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            head_dim: 128,
            intermediate_size: 8960,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            attention_bias: true,
            tie_word_embeddings: true,
        })
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
