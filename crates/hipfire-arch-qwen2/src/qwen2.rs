//! Qwen2 model types: Config / Weights / State.
//!
//! Rev 1 status:
//! - `Qwen2Config::from_hfq` — real metadata parser.
//! - `Qwen2Weights::load` — real loader for HFQ4G256 + F16 quant types.
//!   Supports tied-embeddings (no `lm_head` on disk) and Q/K/V bias.
//!   Other quant types (HFQ4G128, MQ4, MQ3, etc.) panic with a clear
//!   error — extend as needed.
//! - `Qwen2State::new` — stub; real KV cache allocation comes with the
//!   forward pass port.
//! - Forward pass — not yet present.
//!
//! See `docs/plans/qwen_2.5_vlm.md` phase 1 for the full plan.
//!
//! # TODO(transformer-extraction)
//!
//! The helpers in this module (`load_norm_weight_raw`,
//! `load_bias_f32`, `load_weight_tensor`) duplicate logic from
//! `hipfire-arch-qwen35::qwen35`. The Transformer-extraction PR will
//! pull these into `hipfire_runtime::transformer::*` so every arch
//! crate shares one implementation. Marked individually below.

use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Qwen2 model-shape constants parsed from `HfqFile::metadata_json`.
///
/// # Field notes
///
/// - `attention_bias`: Qwen2 modeling-code default is `true`. Many Qwen2
///   HF configs omit the field; treat missing as `true`.
/// - `tie_word_embeddings`: differs across Qwen2 checkpoints. 1.5B-Instruct
///   has `true` (no separate lm_head on disk); dots.ocr's Qwen2 backbone
///   has `false`. Loader handles both.
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

/// Parse a Qwen2 config out of an HFQ file's metadata.
pub fn config_from_hfq(hfq: &HfqFile) -> Option<Qwen2Config> {
    config_from_metadata_json(&hfq.metadata_json)
}

/// Inner parser, decoupled from `HfqFile` for unit testability.
pub fn config_from_metadata_json(metadata_json: &str) -> Option<Qwen2Config> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
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
    /// Convenience: parse and lift `Option` into `Result`.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        config_from_hfq(hfq)
            .ok_or_else(|| "qwen2: failed to parse config from HFQ metadata".to_string())
    }
}

// ─── Weight structs ─────────────────────────────────────────────────────

/// Per-layer Qwen2 dense weights.
///
/// All Qwen2 layers are full-attention dense FFN (no MoE, no hybrid LA).
/// Q/K/V projections carry a bias tensor (`attention_bias=true` in
/// modeling default); `o_proj` and the FFN linears do not.
pub struct Qwen2LayerWeights {
    pub attn_norm: GpuTensor,         // input_layernorm.weight, F32 on GPU
    pub wq: WeightTensor,             // q_proj.weight  [n_heads*head_dim, hidden]
    pub wq_bias: GpuTensor,           // q_proj.bias    [n_heads*head_dim], F32
    pub wk: WeightTensor,             // k_proj.weight  [n_kv_heads*head_dim, hidden]
    pub wk_bias: GpuTensor,           // k_proj.bias    [n_kv_heads*head_dim], F32
    pub wv: WeightTensor,             // v_proj.weight
    pub wv_bias: GpuTensor,           // v_proj.bias
    pub wo: WeightTensor,             // o_proj.weight  (no bias)
    pub ffn_norm: GpuTensor,          // post_attention_layernorm.weight, F32
    pub w_gate: WeightTensor,         // mlp.gate_proj.weight  (no bias)
    pub w_up: WeightTensor,           // mlp.up_proj.weight
    pub w_down: WeightTensor,         // mlp.down_proj.weight
}

/// GPU-resident Qwen2 model weights.
pub struct Qwen2Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<Qwen2LayerWeights>,
    /// True when the model uses tied embeddings and `output` aliases the
    /// embedding table (no separate `lm_head.weight` on disk).
    pub tied_lm_head: bool,
}

impl Qwen2Weights {
    /// Load every tensor from `hfq` to GPU.
    ///
    /// Supports HFQ4G256 (qt=6), HFQ4G128 (qt=7), and F16 (qt=1) on linear
    /// weights; F16/F32 on norm and bias tensors. Other quant types panic
    /// with a clear message — extend as needed.
    pub fn load(hfq: &mut HfqFile, cfg: &Qwen2Config, gpu: &mut Gpu) -> Result<Self, String> {
        load_weights(hfq, cfg, gpu)
            .map_err(|e| format!("qwen2: load_weights failed: {e:?}"))
    }
}

/// Free-function loader, takes a borrowed `Gpu` so the trait impl in
/// `arch.rs` can pass through the runtime-provided handle.
pub fn load_weights(
    hfq: &mut HfqFile,
    cfg: &Qwen2Config,
    gpu: &mut Gpu,
) -> HipResult<Qwen2Weights> {
    #[cfg(unix)]
    hfq.drop_mmap();

    eprintln!("qwen2: loading token_embd...");
    let (embd_token, embd_format) = load_embed_tokens(hfq, gpu, cfg)?;

    eprintln!("qwen2: loading model.norm...");
    let output_norm = load_norm_weight_raw(hfq, gpu, "model.norm.weight", cfg.hidden_size)?;

    eprintln!("qwen2: loading lm_head...");
    let (output, tied_lm_head) = load_lm_head(hfq, gpu, cfg, &embd_token, embd_format)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        eprintln!("qwen2: loading layer {}/{}...", i + 1, cfg.num_hidden_layers);
        layers.push(load_layer(hfq, gpu, cfg, i)?);
    }

    Ok(Qwen2Weights {
        token_embd: embd_token,
        embd_format,
        output_norm,
        output,
        layers,
        tied_lm_head,
    })
}

// ─── Per-tensor loaders ─────────────────────────────────────────────────

fn load_embed_tokens(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Qwen2Config,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    let name = "model.embed_tokens.weight";
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    match info.quant_type {
        6 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok((buf, EmbeddingFormat::HFQ4G256))
        }
        1 => {
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let buf = gpu.upload_f32(&f32_data, &[cfg.vocab_size, cfg.hidden_size])?;
            Ok((buf, EmbeddingFormat::F32))
        }
        qt => panic!("qwen2: unsupported embedding quant_type {qt}; \
                     expected 1 (F16) or 6 (HFQ4G256). Extend load_embed_tokens to handle this format."),
    }
}

/// Load the lm_head. For tied-embedding configs, re-upload the embedding
/// bytes as a separate GPU allocation (matches qwen35's pattern at
/// `qwen35.rs:1384-1410`; `GpuTensor` is not `Clone` so we can't alias).
/// For untied configs, load the separate `lm_head.weight` tensor.
///
/// TODO(transformer-extraction): the tied-embedding re-upload and the
/// DType↔EmbeddingFormat mapping below are cross-arch primitives that
/// also exist in `hipfire-arch-qwen35::qwen35::load_weights`. Move into
/// `hipfire_runtime::transformer::lm_head` during consolidation; consider
/// adding a `GpuTensor::shallow_clone` or moving to `Arc<GpuTensor>` so
/// tied embeddings stop double-allocating VRAM.
fn load_lm_head(
    hfq: &HfqFile,
    gpu: &Gpu,
    cfg: &Qwen2Config,
    _embd_token: &GpuTensor,
    embd_format: EmbeddingFormat,
) -> HipResult<(WeightTensor, bool)> {
    if cfg.tie_word_embeddings {
        let name = "model.embed_tokens.weight";
        let (info, data) = hfq.tensor_data_vec(name)
            .unwrap_or_else(|| panic!("qwen2: tensor not found for tied lm_head: {name}"));
        let dtype = match embd_format {
            EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
            EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
            EmbeddingFormat::Q8_0 => DType::Q8_0,
            EmbeddingFormat::F32 => DType::F32,
            EmbeddingFormat::Q4K => panic!("qwen2: tied embeddings with Q4K not supported"),
        };
        let buf = match info.quant_type {
            6 | 7 | 3 => gpu.upload_raw(&data, &[data.len()])?,
            1 => {
                // F16 → upload raw; the kernel handles F16 natively.
                gpu.upload_raw(&data, &[data.len()])?
            }
            qt => panic!("qwen2: unsupported tied embedding quant_type {qt}"),
        };
        let wt = WeightTensor {
            buf,
            gpu_dtype: dtype,
            m: cfg.vocab_size,
            k: cfg.hidden_size,
            row_stride: 0,
            awq_scale: None,
        };
        Ok((wt, true))
    } else {
        let wt = load_weight_tensor(hfq, gpu, "lm_head.weight", cfg.vocab_size, cfg.hidden_size)?;
        Ok((wt, false))
    }
}

fn load_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Qwen2Config,
    i: usize,
) -> HipResult<Qwen2LayerWeights> {
    let p = format!("model.layers.{i}");
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    let attn_norm = load_norm_weight_raw(hfq, gpu, &format!("{p}.input_layernorm.weight"), cfg.hidden_size)?;

    let wq = load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_dim, cfg.hidden_size)?;
    let wq_bias = load_bias_f32(hfq, gpu, &format!("{p}.self_attn.q_proj.bias"), q_dim)?;
    let wk = load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, cfg.hidden_size)?;
    let wk_bias = load_bias_f32(hfq, gpu, &format!("{p}.self_attn.k_proj.bias"), kv_dim)?;
    let wv = load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, cfg.hidden_size)?;
    let wv_bias = load_bias_f32(hfq, gpu, &format!("{p}.self_attn.v_proj.bias"), kv_dim)?;
    let wo = load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), cfg.hidden_size, q_dim)?;

    let ffn_norm = load_norm_weight_raw(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), cfg.hidden_size)?;

    let w_gate = load_weight_tensor(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), cfg.intermediate_size, cfg.hidden_size)?;
    let w_up = load_weight_tensor(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), cfg.intermediate_size, cfg.hidden_size)?;
    let w_down = load_weight_tensor(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), cfg.hidden_size, cfg.intermediate_size)?;

    Ok(Qwen2LayerWeights {
        attn_norm,
        wq, wq_bias, wk, wk_bias, wv, wv_bias, wo,
        ffn_norm,
        w_gate, w_up, w_down,
    })
}

// ─── Helpers (duplicated from qwen35 with Qwen2 conventions) ────────────

/// TODO(transformer-extraction): duplicates `load_norm_weight_raw` in
/// `hipfire-arch-qwen35::qwen35`. Differences from the qwen35 version:
///
/// - **No `+= 1.0` offset** — Qwen2 uses standard RMSNorm
///   `weight * x * rsqrt(...)`, whereas Qwen3.5 uses `(1 + weight) * ...`.
///   The qwen35 crate has two helpers (`load_norm_weight` with offset,
///   `load_norm_weight_raw` without); Qwen2 only ever needs the raw form.
/// - **No `model.language_model.` name prefix** — Qwen2 stores norms as
///   `model.{...}` directly, not the VL-friendly `model.language_model.`
///   that qwen35 uses.
///
/// Both deltas would be parameters if this lived in
/// `hipfire_runtime::transformer::norm`. Pull during the
/// Transformer-extraction PR.
fn load_norm_weight_raw(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        qt => panic!("qwen2: expected F16/F32 for norm {name}, got qt={qt}"),
    };
    gpu.upload_f32(&f32_data, &[n])
}

/// Load a bias tensor (Q/K/V projection bias) as F32 on GPU.
///
/// TODO(transformer-extraction): qwen35 has no equivalent because Qwen3
/// uses `attention_bias=false` — qwen35's QKV linears have no bias. This
/// helper is unique to Qwen2-family arches (Qwen2 + dots.ocr's Qwen2
/// backbone). When the Transformer-extraction PR lands, this can live
/// next to `load_norm_weight` as a sibling F32-uploader keyed by tensor
/// element count.
fn load_bias_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        qt => panic!("qwen2: expected F16/F32 for bias {name}, got qt={qt}"),
    };
    assert_eq!(f32_data.len(), n,
        "qwen2: bias {name} has {} elements, expected {n}", f32_data.len());
    gpu.upload_f32(&f32_data, &[n])
}

/// TODO(transformer-extraction): duplicates `load_weight_tensor` +
/// `load_weight_tensor_raw` in `hipfire-arch-qwen35::qwen35`. The qwen35
/// version handles ~14 quant_types; this rev-1 starter only covers the
/// two we've actually shipped HFQ files for (HFQ4G256, F16). Extend as
/// needed, or wait for the consolidation PR to pick up the qwen35
/// implementation.
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    match info.quant_type {
        6 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G256, m, k, row_stride: 0, awq_scale: None })
        }
        7 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G128, m, k, row_stride: 0, awq_scale: None })
        }
        1 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F16, m, k, row_stride: 0, awq_scale: None })
        }
        qt => panic!("qwen2: unsupported weight quant_type {qt} for {name}. \
                     This rev-1 loader handles qt ∈ {{1 (F16), 6 (HFQ4G256), 7 (HFQ4G128)}}. \
                     Extend load_weight_tensor or wait for the Transformer-extraction PR \
                     to pick up qwen35's full quant_type matrix."),
    }
}

// ─── State ───────────────────────────────────────────────────────────────

/// Qwen2 per-decode GPU scratch (KV cache + attention workspace).
///
/// Rev 0: stub. Real implementation allocates GPU buffers for the KV
/// cache (sized by `num_key_value_heads × head_dim × max_seq_len ×
/// num_hidden_layers`) and the per-layer attention workspace. See
/// `ForwardScratch::new` in `hipfire-runtime::llama` for the dense-FA
/// reference shape.
pub struct Qwen2State {
    pub token_count: usize,
}

impl Qwen2State {
    pub fn new(_cfg: &Qwen2Config) -> Result<Self, String> {
        Ok(Qwen2State { token_count: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 8960);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert!(cfg.attention_bias);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.eos_token_id, 151645);
    }

    #[test]
    fn parses_dots_ocr_text_config() {
        let cfg = config_from_metadata_json(DOTS_OCR_TEXT_METADATA)
            .expect("parser returned None on a valid dots.ocr text config");
        assert!(cfg.attention_bias);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.eos_token_id, 151643);
        assert_eq!(cfg.max_position_embeddings, 131072);
    }

    #[test]
    fn missing_required_field_returns_none() {
        let bad = r#"{"config": {"hidden_size": 1536}}"#;
        assert!(config_from_metadata_json(bad).is_none());
    }

    #[test]
    fn missing_optional_fields_get_defaults() {
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
        assert_eq!(cfg.num_key_value_heads, 12);
        assert_eq!(cfg.head_dim, 64);
        assert!(cfg.attention_bias);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.eos_token_id, 151645);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    }
}
