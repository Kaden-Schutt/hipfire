//! Qwen2 model types: Config / Weights / State, plus the
//! [`forward_step`] / [`forward_step_greedy`] hot-path entry points.
//!
//! Implementation status:
//! - [`Qwen2Config::from_hfq`] — full HFQ-metadata parser; handles
//!   scalar + array `eos_token_id`, optional `head_dim`, the
//!   `attention_bias` default, and `text_config` nesting.
//! - [`Qwen2Weights::load`] — loads embed_tokens + final norm + lm_head
//!   (tied or untied; F16-tied path host-expands to F32) + 28 layers.
//!   Supports HFQ4G256 / HFQ4G128 / Q8F16 / F16 weight quant types.
//! - [`Qwen2State`] — full per-step scratch graph + F32 KV cache.
//!   `new_with_max_seq` for explicit KV budget; `reset()` for cheap
//!   between-turn rewind.
//! - [`forward_step`] — one decode step through 28 layers (RMSNorm →
//!   fused QKV + 3× bias_add → RoPE → KV write → attention → o_proj →
//!   residual → FFN norm → SwiGLU → residual). End-to-end validated
//!   16/16 top-1 match vs HF F32 reference at Q8F16 precision.
//!
//! See `docs/plans/dots-ocr-prd.md` phase 1 for the
//! bring-up plan and `lib.rs` for the rev-3 status summary.
//!
//! # TODO(transformer-extraction)
//!
//! The helpers in this module (`load_norm_weight_raw`,
//! `load_bias_f32`, `load_weight_tensor`) duplicate logic from
//! `hipfire-arch-qwen35::qwen35`. The Transformer-extraction PR will
//! pull these into `hipfire_runtime::transformer::*` so every arch
//! crate shares one implementation. Marked individually below.

use hip_bridge::{DeviceBuffer, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps_mesh, GemvInput, Step};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_hardware::DeviceMesh;
use hipfire_runtime::arch_spec::{dense_forward, DenseArch, DenseKnobs, DenseLayer, DenseScratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use hipfire_runtime::llama::{gemv_family, weight_gemm, EmbeddingFormat, WeightTensor};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::weight_backend::{
    dequant_norm, dequant_weight_raw, flat_name_candidates, load_embedding, resolve_lm_head,
    HfqBackend, WeightBackend,
};
use hipfire_runtime::{screen_weight_tensor, MmqScreenable};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;

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
/// - `eos_token_id` / `eos_token_ids`: HF stores either a scalar or an
///   array. `eos_token_id` is the first/primary element (back-compat
///   accessor); `eos_token_ids` carries the full set so the runtime
///   can build a multi-element stop-set (e.g. dots.ocr's
///   `[151643, 151673]` — without both, streaming EOS misses one).
///   Lookup order: `config.eos_token_id` (scalar or array) →
///   `generation_config.eos_token_id` → default `[151645]` (ChatML
///   `<|im_end|>`). dots.ocr's `config.json` carries no EOS at all;
///   the array lives only in `generation_config.json`, which the
///   quantiser packs into HFQ metadata as of R5. See
///   `docs/plans/dots-ocr-devlog.md` §7 (R5).
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
    /// Primary EOS for back-compat with the daemon's scalar consumer.
    /// Equal to `eos_token_ids[0]` when the array form is present.
    pub eos_token_id: u32,
    /// Full EOS set. Single-element vec for scalar configs; multi-element
    /// for array configs (Qwen2-1.5B: `[151645, 151643]`; dots.ocr:
    /// `[151643, 151673]`). Always non-empty.
    pub eos_token_ids: Vec<u32>,
}

/// Serde mirror of the Qwen2 `config` (or nested `text_config`) blob.
///
/// Required model-shape fields are non-`Option` so serde hard-errors on a
/// missing or wrong-typed key. Derived fields (`num_key_value_heads`,
/// `head_dim`) are `Option` and finalized below. The multi-source EOS
/// resolution needs the root `meta`, so it is NOT a field here — it runs in
/// [`from_config_value`] / [`config_from_metadata_json`].
#[derive(Deserialize)]
struct RawQwen2Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_attention_bias")]
    attention_bias: bool,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_max_pos() -> usize {
    32768
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_attention_bias() -> bool {
    true
}

/// Parse a Qwen2 config out of an HFQ file's metadata.
pub fn config_from_hfq(hfq: &HfqFile) -> Result<Qwen2Config, String> {
    config_from_metadata_json(&hfq.metadata_json)
}

/// `eos_token_id` may be a scalar Number or an Array of numbers; both
/// flatten to a `Vec<u32>` (Array→`filter_map` as_u64, Number→single).
fn parse_eos(val: &serde_json::Value) -> Vec<u32> {
    match val {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|e| e.as_u64().map(|n| n as u32))
            .collect(),
        serde_json::Value::Number(n) => n.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Inner parser, decoupled from `HfqFile` for unit testability.
///
/// Unwraps the `{config}` envelope, descends into `text_config` if present,
/// then delegates to [`from_config_value`] with the root `meta` (needed for
/// the `generation_config.eos_token_id` fallback).
pub fn config_from_metadata_json(metadata_json: &str) -> Result<Qwen2Config, String> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| format!("qwen2: metadata_json not valid JSON: {e}"))?;
    let config = meta
        .get("config")
        .ok_or_else(|| "qwen2: metadata_json missing `config` wrapper".to_string())?;
    let tc = config.get("text_config").unwrap_or(config);
    from_config_value(tc, Some(&meta))
}

/// Parse from a raw config Value (the inner `config` / `text_config` node).
///
/// `meta` is the root envelope, consulted only for the
/// `generation_config.eos_token_id` fallback; pass `None` when no envelope
/// exists (EOS then resolves from `inner` + default).
///
/// EOS lookup order:
///   1. `inner.eos_token_id` (scalar or array)
///   2. root `generation_config.eos_token_id` (dots.ocr's [151643, 151673]
///      lives only here per R5 — the quantiser packs this sibling JSON into
///      HFQ metadata so this fallback is reachable)
///   3. Default [151645] (ChatML `<|im_end|>`)
pub fn from_config_value(
    inner: &serde_json::Value,
    meta: Option<&serde_json::Value>,
) -> Result<Qwen2Config, String> {
    let raw: RawQwen2Config = serde_json::from_value(inner.clone())
        .map_err(|e| format!("qwen2: parsing config failed: {e}"))?;

    let num_key_value_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
    let head_dim = raw
        .head_dim
        .unwrap_or(raw.hidden_size / raw.num_attention_heads);

    let mut eos_token_ids: Vec<u32> = inner.get("eos_token_id").map(parse_eos).unwrap_or_default();
    if eos_token_ids.is_empty() {
        if let Some(gc_eos) = meta
            .and_then(|m| m.get("generation_config"))
            .and_then(|gc| gc.get("eos_token_id"))
        {
            eos_token_ids = parse_eos(gc_eos);
        }
    }
    let eos_token_ids = if eos_token_ids.is_empty() {
        vec![151645]
    } else {
        eos_token_ids
    };
    let eos_token_id = eos_token_ids[0];

    Ok(Qwen2Config {
        hidden_size: raw.hidden_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size: raw.intermediate_size,
        vocab_size: raw.vocab_size,
        max_position_embeddings: raw.max_position_embeddings,
        rope_theta: raw.rope_theta,
        rms_norm_eps: raw.rms_norm_eps,
        attention_bias: raw.attention_bias,
        tie_word_embeddings: raw.tie_word_embeddings,
        eos_token_id,
        eos_token_ids,
    })
}

impl Qwen2Config {
    /// Convenience wrapper around [`config_from_hfq`].
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        config_from_hfq(hfq)
    }
}

// ─── Weight structs ─────────────────────────────────────────────────────

/// Per-layer Qwen2 dense weights.
///
/// All Qwen2 layers are full-attention dense FFN (no MoE, no hybrid LA).
/// Q/K/V projections carry a bias tensor (`attention_bias=true` in
/// modeling default); `o_proj` and the FFN linears do not.
pub struct Qwen2LayerWeights {
    pub attn_norm: GpuTensor, // input_layernorm.weight, F32 on GPU
    pub wq: WeightTensor,     // q_proj.weight  [n_heads*head_dim, hidden]
    pub wq_bias: GpuTensor,   // q_proj.bias    [n_heads*head_dim], F32
    pub wk: WeightTensor,     // k_proj.weight  [n_kv_heads*head_dim, hidden]
    pub wk_bias: GpuTensor,   // k_proj.bias    [n_kv_heads*head_dim], F32
    pub wv: WeightTensor,     // v_proj.weight
    pub wv_bias: GpuTensor,   // v_proj.bias
    pub wo: WeightTensor,     // o_proj.weight  (no bias)
    pub ffn_norm: GpuTensor,  // post_attention_layernorm.weight, F32
    pub w_gate: WeightTensor, // mlp.gate_proj.weight  (no bias)
    pub w_up: WeightTensor,   // mlp.up_proj.weight
    pub w_down: WeightTensor, // mlp.down_proj.weight
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
        load_weights(hfq, cfg, gpu).map_err(|e| format!("qwen2: load_weights failed: {e:?}"))
    }

    /// Release every GPU buffer back to the pool. Consumes self.
    /// Mirrors `LlamaWeights::free_gpu` and `Qwen35Weights::free_gpu`
    /// — the daemon calls this on unload to actually return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        if !self.tied_lm_head {
            let _ = gpu.free_tensor(self.output.buf);
        }
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wq_bias);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wk_bias);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wv_bias);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
    }
}

impl MmqScreenable for Qwen2Weights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.output, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            for weight in [
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ] {
                screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
            }
        }
        (safe, unsafe_count)
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
        eprintln!(
            "qwen2: loading layer {}/{}...",
            i + 1,
            cfg.num_hidden_layers
        );
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
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    load_embedding(gpu, info.quant_type, &data, cfg.vocab_size, cfg.hidden_size)
}

/// Load the lm_head. For tied-embedding configs, re-upload the embedding
/// bytes as a separate GPU allocation (matches qwen35's pattern at
/// `qwen35.rs:1414-1448`; `GpuTensor` is not `Clone` so we can't alias).
/// For untied configs, load the separate `lm_head.weight` tensor.
///
/// **F16 source caveat:** `EmbeddingFormat` has no `F16` variant
/// (`hipfire_runtime::llama::EmbeddingFormat` is F32 / Q4K / HFQ4G256 /
/// HFQ4G128 / Q8_0). `load_embed_tokens` promotes F16 source to F32 on
/// the host before upload; the tied-lm_head path here must do the
/// same. Uploading raw F16 bytes while tagging `gpu_dtype = F32`
/// produces a corrupted matmul (kernel reads F16 bytes as F32 values).
/// See R4 in `docs/plans/dots-ocr-devlog.md` §7 for the catch history.
///
fn load_lm_head(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Qwen2Config,
    embd_token: &GpuTensor,
    embd_format: EmbeddingFormat,
) -> HipResult<(WeightTensor, bool)> {
    resolve_lm_head(
        gpu,
        !cfg.tie_word_embeddings, // has_separate
        true,                     // single-GPU: alias when tied
        embd_token,
        embd_format,
        cfg.vocab_size,
        cfg.hidden_size,
        |gpu| {
            load_weight_tensor(
                hfq,
                gpu,
                "lm_head.weight",
                cfg.vocab_size,
                cfg.hidden_size,
                flat_name_candidates,
            )
        },
        // qwen2 is single-GPU; the reupload arm is never taken.
        |_gpu| unreachable!("qwen2 is single-GPU; tied lm_head always aliases"),
    )
}

fn load_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Qwen2Config,
    i: usize,
) -> HipResult<Qwen2LayerWeights> {
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    let mut b = HfqBackend {
        hfq,
        gpu,
        norm_bias: 0.0,
        candidates: flat_name_candidates,
        read_proj: load_weight_tensor,
        layer: i,
    };

    Ok(Qwen2LayerWeights {
        attn_norm: b.norm("input_layernorm.weight", &[cfg.hidden_size])?,
        wq: b.proj("self_attn.q_proj", q_dim, cfg.hidden_size)?,
        wq_bias: b.bias("self_attn.q_proj.bias", q_dim)?,
        wk: b.proj("self_attn.k_proj", kv_dim, cfg.hidden_size)?,
        wk_bias: b.bias("self_attn.k_proj.bias", kv_dim)?,
        wv: b.proj("self_attn.v_proj", kv_dim, cfg.hidden_size)?,
        wv_bias: b.bias("self_attn.v_proj.bias", kv_dim)?,
        wo: b.proj("self_attn.o_proj", cfg.hidden_size, q_dim)?,
        ffn_norm: b.norm("post_attention_layernorm.weight", &[cfg.hidden_size])?,
        w_gate: b.proj("mlp.gate_proj", cfg.intermediate_size, cfg.hidden_size)?,
        w_up: b.proj("mlp.up_proj", cfg.intermediate_size, cfg.hidden_size)?,
        w_down: b.proj("mlp.down_proj", cfg.hidden_size, cfg.intermediate_size)?,
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
fn load_norm_weight_raw(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    let t = dequant_norm(gpu, info.quant_type, &data, &[n], 0.0)?;
    Ok(t)
}

/// TODO(transformer-extraction): duplicates `load_weight_tensor` +
/// `load_weight_tensor_raw` in `hipfire-arch-qwen35::qwen35`. The qwen35
/// version handles ~14 quant_types; this rev-1 starter only covers the
/// two we've actually shipped HFQ files for (HFQ4G256, F16). Extend as
/// needed, or wait for the consolidation PR to pick up the qwen35
/// implementation.
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
    candidates: fn(&str) -> Vec<String>,
) -> HipResult<WeightTensor> {
    for cand in candidates(name) {
        if let Some((info, data)) = hfq.tensor_data_vec(&cand) {
            return dequant_weight_raw(gpu, info.quant_type, &data, m, k);
        }
    }
    panic!("qwen2: tensor not found: {name}");
}

// ─── Source-based loaders (from &dyn ModelSource) ──────────────────────

/// Parse Qwen2Config from a ModelSource (safetensors or HFQ).
pub fn config_from_source(source: &dyn ModelSource) -> Option<Qwen2Config> {
    config_from_metadata_json(source.metadata_json()).ok()
}

/// Convert BF16 bytes to a vector of F32 values.
/// BF16 has the same 8-bit exponent as F32 but only 7 mantissa bits.
fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Convert source dtype bytes to a F16 byte stream ready for GPU upload.
/// Handles F16 (passthrough), BF16 (→F16), and F32 (→F16).
fn source_bytes_to_f16_stream(source_dtype: &str, data: &[u8], n_elements: usize) -> Vec<u8> {
    match source_dtype {
        "F16" => {
            assert_eq!(
                data.len(),
                2 * n_elements,
                "qwen2: F16 tensor has {} bytes, expected 2 * {n_elements}",
                data.len()
            );
            data.to_vec()
        }
        "BF16" => {
            let f32_vals = bf16_bytes_to_f32(data);
            assert_eq!(f32_vals.len(), n_elements);
            f32_vals
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect()
        }
        "F32" => {
            let f32_vals: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(f32_vals.len(), n_elements);
            f32_vals
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect()
        }
        other => {
            panic!("qwen2: unsupported source dtype '{other}' for tensor (expected F16/BF16/F32)")
        }
    }
}

/// Convert source dtype bytes to an F32 vector (for norms/biases).
fn source_bytes_to_f32_vec(source_dtype: &str, data: &[u8], n: usize) -> Vec<f32> {
    match source_dtype {
        "F16" => {
            let v: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            assert_eq!(v.len(), n);
            v
        }
        "BF16" => {
            let v = bf16_bytes_to_f32(data);
            assert_eq!(v.len(), n);
            v
        }
        "F32" => {
            let v: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(v.len(), n);
            v
        }
        other => {
            panic!("qwen2: unsupported source dtype '{other}' for tensor (expected F16/BF16/F32)")
        }
    }
}

/// Load a norm weight tensor from ModelSource (always F32 on GPU).
fn load_norm_weight_raw_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> HipResult<GpuTensor> {
    let (info, data) = source
        .tensor_data(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    let f32_data = source_bytes_to_f32_vec(&info.dtype, data, n);
    gpu.upload_f32(&f32_data, &[n])
}

/// Load a bias tensor from ModelSource (always F32 on GPU).
fn load_bias_f32_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> HipResult<GpuTensor> {
    let (info, data) = source
        .tensor_data(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    let f32_data = source_bytes_to_f32_vec(&info.dtype, data, n);
    gpu.upload_f32(&f32_data, &[n])
}

/// Load a linear weight tensor from ModelSource as F16 on GPU.
fn load_weight_tensor_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    // Clean error (not panic) on a missing tensor — e.g. a ParoQuant qwen2 dir
    // that ships only `.qweight` (no F16 `.weight`). The caller turns this into
    // a load failure instead of aborting the daemon.
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| hip_bridge::HipError::new(0, &format!("qwen2: tensor not found: {name}")))?;
    let n_elements = m * k;
    let f16_bytes = source_bytes_to_f16_stream(&info.dtype, data, n_elements);
    let buf = gpu.upload_raw(&f16_bytes, &[n_elements])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::F16,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

/// Load embedding tokens from ModelSource as F32 on GPU.
fn load_embed_tokens_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    cfg: &Qwen2Config,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    let name = "model.embed_tokens.weight";
    let (info, data) = source
        .tensor_data(name)
        .unwrap_or_else(|| panic!("qwen2: tensor not found: {name}"));
    let n = cfg.vocab_size * cfg.hidden_size;
    let f32_data = source_bytes_to_f32_vec(&info.dtype, data, n);
    let buf = gpu.upload_f32(&f32_data, &[cfg.vocab_size, cfg.hidden_size])?;
    Ok((buf, EmbeddingFormat::F32))
}

/// Load lm_head from ModelSource. For untied configs, loads separate
/// lm_head.weight. For tied configs, re-uses embed_tokens (also F32).
fn load_lm_head_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    cfg: &Qwen2Config,
    _embd_token: &GpuTensor,
    _embd_format: EmbeddingFormat,
) -> HipResult<(WeightTensor, bool)> {
    if cfg.tie_word_embeddings {
        let name = "model.embed_tokens.weight";
        let (info, data) = source
            .tensor_data(name)
            .unwrap_or_else(|| panic!("qwen2: tensor not found for tied lm_head: {name}"));
        let n = cfg.vocab_size * cfg.hidden_size;
        let f32_data = source_bytes_to_f32_vec(&info.dtype, data, n);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu.upload_raw(bytes, &[cfg.vocab_size, cfg.hidden_size])?;
        let wt = WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m: cfg.vocab_size,
            k: cfg.hidden_size,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        Ok((wt, true))
    } else {
        let wt = load_weight_tensor_from_source(
            source,
            gpu,
            "lm_head.weight",
            cfg.vocab_size,
            cfg.hidden_size,
        )?;
        Ok((wt, false))
    }
}

/// Load a single Qwen2 layer from ModelSource.
fn load_layer_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    cfg: &Qwen2Config,
    i: usize,
) -> HipResult<Qwen2LayerWeights> {
    let p = format!("model.layers.{i}");
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    let attn_norm = load_norm_weight_raw_from_source(
        source,
        gpu,
        &format!("{p}.input_layernorm.weight"),
        cfg.hidden_size,
    )?;

    let wq = load_weight_tensor_from_source(
        source,
        gpu,
        &format!("{p}.self_attn.q_proj.weight"),
        q_dim,
        cfg.hidden_size,
    )?;
    let wq_bias =
        load_bias_f32_from_source(source, gpu, &format!("{p}.self_attn.q_proj.bias"), q_dim)?;
    let wk = load_weight_tensor_from_source(
        source,
        gpu,
        &format!("{p}.self_attn.k_proj.weight"),
        kv_dim,
        cfg.hidden_size,
    )?;
    let wk_bias =
        load_bias_f32_from_source(source, gpu, &format!("{p}.self_attn.k_proj.bias"), kv_dim)?;
    let wv = load_weight_tensor_from_source(
        source,
        gpu,
        &format!("{p}.self_attn.v_proj.weight"),
        kv_dim,
        cfg.hidden_size,
    )?;
    let wv_bias =
        load_bias_f32_from_source(source, gpu, &format!("{p}.self_attn.v_proj.bias"), kv_dim)?;
    let wo = load_weight_tensor_from_source(
        source,
        gpu,
        &format!("{p}.self_attn.o_proj.weight"),
        cfg.hidden_size,
        q_dim,
    )?;

    let ffn_norm = load_norm_weight_raw_from_source(
        source,
        gpu,
        &format!("{p}.post_attention_layernorm.weight"),
        cfg.hidden_size,
    )?;

    let w_gate = load_weight_tensor_from_source(
        source,
        gpu,
        &format!("{p}.mlp.gate_proj.weight"),
        cfg.intermediate_size,
        cfg.hidden_size,
    )?;
    let w_up = load_weight_tensor_from_source(
        source,
        gpu,
        &format!("{p}.mlp.up_proj.weight"),
        cfg.intermediate_size,
        cfg.hidden_size,
    )?;
    let w_down = load_weight_tensor_from_source(
        source,
        gpu,
        &format!("{p}.mlp.down_proj.weight"),
        cfg.hidden_size,
        cfg.intermediate_size,
    )?;

    Ok(Qwen2LayerWeights {
        attn_norm,
        wq,
        wq_bias,
        wk,
        wk_bias,
        wv,
        wv_bias,
        wo,
        ffn_norm,
        w_gate,
        w_up,
        w_down,
    })
}

/// Load full Qwen2 weights from a ModelSource (safetensors or compatible).
pub fn load_weights_from_source(
    source: &dyn ModelSource,
    cfg: &Qwen2Config,
    gpu: &mut Gpu,
) -> HipResult<Qwen2Weights> {
    eprintln!("qwen2: loading token_embd (from source)...");
    let (embd_token, embd_format) = load_embed_tokens_from_source(source, gpu, cfg)?;

    eprintln!("qwen2: loading model.norm...");
    let output_norm =
        load_norm_weight_raw_from_source(source, gpu, "model.norm.weight", cfg.hidden_size)?;

    eprintln!("qwen2: loading lm_head...");
    let (output, tied_lm_head) =
        load_lm_head_from_source(source, gpu, cfg, &embd_token, embd_format)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        eprintln!(
            "qwen2: loading layer {}/{} (from source)...",
            i + 1,
            cfg.num_hidden_layers
        );
        layers.push(load_layer_from_source(source, gpu, cfg, i)?);
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

// ─── State ───────────────────────────────────────────────────────────────

/// Qwen2 per-decode GPU scratch (KV cache + per-step workspace).
///
/// Rev 3: real. Mirrors `hipfire_runtime::llama::ForwardScratch` with
/// three deltas:
///
/// - **F32 KV cache** only. The bring-up validation path is greedy
///   decode against an HF F32 reference, so any KV quantisation would
///   add a confound to top-1 match debugging. Quantised KV (HFQ4 /
///   HFQ8 / asym-N / Q8) is a phase-1.5 follow-on under the existing
///   `kv_mode` story.
/// - **No sampler scratch.** `sample_buf` / `repeat_buf` are unused
///   because we drive validation with `argmax_f32` (greedy). Sampling
///   wiring is a follow-on when the daemon arm is added (R3).
/// - **`x_rot` scratch.** Sized `max(dim, intermediate_size)`, used by
///   `RmsnormAutomatic` as the rotation output buffer. For HFQ4/Q8_0
///   (rotation-free dtypes), `RmsnormAutomatic(None)` writes plain rmsnorm
///   output here; for MQ-family dtypes it would hold FWHT-rotated activations.
///
/// Sizes:
/// - `x`, `tmp`, `o`, `ffn_out` : `hidden_size` (residual stream)
/// - `q`, `attn_out`            : `n_heads × head_dim`
/// - `k`, `v`                   : `n_kv_heads × head_dim`
/// - `gate`, `up`, `ffn_hidden` : `intermediate_size`
/// - `logits`                   : `vocab_size`
/// - `k_cache[layer]`, `v_cache[layer]`: `max_seq × n_kv_heads × head_dim`
/// - `pos_buf`                  : 4 bytes (single i32, device-side
///   position counter for `rope_f32` / `kv_cache_write` / `attention_f32`)
///
/// `max_seq` is the KV cache budget set at allocation time. Bring-up
/// uses 512 which fits the smoke prompt + 32-token continuation with
/// headroom; bump via `Qwen2State::new_with_max_seq` for longer runs.
pub struct Qwen2State {
    pub x: GpuTensor,
    pub tmp: GpuTensor,
    pub x_rot: GpuTensor, // rotation output scratch (rmsnorm output for non-MQ dtypes)
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub attn_out: GpuTensor,
    pub o: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub ffn_out: GpuTensor,
    pub logits: GpuTensor,
    /// Scratch for the split-K flash decode attention (`attention_flash`):
    /// per-(head, chunk) partials `[n_heads * ceil(max_seq/128) * (2 + head_dim)]`.
    pub attn_partials: GpuTensor,
    pub pos_buf: DeviceBuffer,
    pub k_cache: Vec<GpuTensor>,
    pub v_cache: Vec<GpuTensor>,
    pub max_seq: usize,
    /// Tracks the next free KV slot — i.e. the absolute position the
    /// next forward step will write. Bumped by [`forward_step`].
    pub next_pos: usize,
}

/// Default KV budget for the bring-up validation path. Smoke prompt is
/// 15 tokens + 32-token continuation = 47 positions consumed; 512 leaves
/// 10× headroom and only costs `28 × 2 × 512 × 256 × 4 ≈ 28 MB` VRAM
/// at f32 KV (28 layers, k+v, kv_dim=256, f32).
pub const DEFAULT_MAX_SEQ: usize = 512;

impl Qwen2State {
    /// Construct with the default KV budget. Wraps the trait surface.
    pub fn new(gpu: &mut Gpu, cfg: &Qwen2Config) -> Result<Self, String> {
        Self::new_with_max_seq(gpu, cfg, DEFAULT_MAX_SEQ)
            .map_err(|e| format!("qwen2: Qwen2State::new failed: {e:?}"))
    }

    /// Allocate the full scratch graph + KV cache at the given seq budget.
    pub fn new_with_max_seq(gpu: &mut Gpu, cfg: &Qwen2Config, max_seq: usize) -> HipResult<Self> {
        let dim = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let hidden_dim = cfg.intermediate_size;

        let mut k_cache = Vec::with_capacity(cfg.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            k_cache.push(gpu.zeros(&[max_seq * kv_dim], DType::F32)?);
            v_cache.push(gpu.zeros(&[max_seq * kv_dim], DType::F32)?);
        }

        // Flash-decode partials: chunk_size is 128 for seq_len > 128, so the
        // max chunk count is ceil(max_seq/128); stride is (max, sum, head_dim).
        let n_chunks_max = (max_seq + 127) / 128;
        let attn_partials_len = cfg.num_attention_heads * n_chunks_max * (2 + cfg.head_dim);

        // x_rot scratch: must fit both attention (dim) and FFN (hidden_dim)
        // rmsnorm/rotation output.
        let x_rot_len = dim.max(hidden_dim);

        Ok(Self {
            x: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            x_rot: gpu.alloc_tensor(&[x_rot_len], DType::F32)?,
            q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            up: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)?,
            attn_partials: gpu.alloc_tensor(&[attn_partials_len], DType::F32)?,
            pos_buf: gpu.hip.malloc(4)?,
            k_cache,
            v_cache,
            max_seq,
            next_pos: 0,
        })
    }

    /// Rewind the position cursor to 0 so the next [`forward_step`]
    /// begins a fresh conversation. The KV cache buffers are not zeroed
    /// — slots get overwritten in place as `forward_step` writes at the
    /// new positions — so reset is O(1). The daemon calls this from the
    /// `reset` event handler and from the `bench_prefill` cold-start
    /// path; callers driving multi-turn chat through a long-running
    /// session should call it whenever they want to discard prior
    /// context.
    pub fn reset(&mut self) {
        self.next_pos = 0;
    }

    /// Release every GPU buffer back to the pool. Consumes self.
    /// Mirrors `ForwardScratch::free_gpu` in `hipfire_runtime::llama`.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x,
            self.tmp,
            self.x_rot,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.o,
            self.gate,
            self.up,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.attn_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for t in self.k_cache {
            let _ = gpu.free_tensor(t);
        }
        for t in self.v_cache {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────

/// Single-token decode step. Reads `token` at `state.next_pos`, runs
/// the full 28-layer stack, writes K/V into the cache at the same
/// position, and leaves the final logits in `state.logits`. Bumps
/// `state.next_pos` by 1.
///
/// Returns Ok(()) on success; `state.logits` holds the f32 vocab-sized
/// distribution and the caller drives sampling (e.g. via
/// [`forward_step_greedy`], or future top-p / repeat-penalty paths).
///
/// Layer body, in order:
///
/// 1. RMSNorm(x → tmp) with `attn_norm`
/// 2. `fused_qkv_hfq4g256(tmp → q,k,v)` (assumes HFQ4G256 attn weights;
///    other dtypes fall back to three `weight_gemv` calls)
/// 3. `bias_add_f32` on each of q, k, v (Qwen2 has attention_bias=true)
/// 4. RoPE on q,k (1-D, theta=cfg.rope_theta)
/// 5. KV cache write at `next_pos`
/// 6. `attention_f32` (GQA via `n_heads` vs `n_kv_heads`)
/// 7. `o_proj` via `weight_gemv` → o
/// 8. Residual add x += o
/// 9. RMSNorm(x → tmp) with `ffn_norm`
/// 10. SwiGLU: `gate = w_gate(tmp)`, `up = w_up(tmp)`,
///     `ffn_hidden = silu(gate) * up`, `ffn_out = w_down(ffn_hidden)`
/// 11. Residual add x += ffn_out
///
/// Then final RMSNorm + lm_head GEMV → logits.
///
/// What this does NOT do:
/// - Prefill batching (we run one token at a time; prefill = N
///   sequential calls). Adequate for greedy validation on short prompts;
///   prefill batching is a follow-on for serving perf.
/// - KV quantisation (cache is F32; see Qwen2State doc for rationale).
/// - Sampling (caller picks argmax or top-p).
pub fn forward_step(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    token: u32,
) -> HipResult<()> {
    let pos = forward_step_prelude(gpu, state)?;

    // Embedding lookup → state.x.
    let dim = cfg.hidden_size;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::Q4K => {
            gpu.embedding_lookup_q4k(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &state.x, token, dim)?,
    }

    forward_step_after_x(gpu, weights, cfg, state, pos)
}

/// Variant of [`forward_step`] that consumes a pre-built F32 embedding row
/// instead of looking one up from `weights.token_embd`. Used by VLM splice
/// paths (dots-ocr, qwen2-vl) to insert vision-tower merger outputs at
/// `<|imgpad|>` positions during prefill.
///
/// `embedding` is a host-side slice of exactly `cfg.hidden_size` F32 values
/// (one row of the merger output). It is uploaded directly into `state.x.buf`,
/// after which the layer loop runs identically to [`forward_step`].
pub fn forward_step_with_embed(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    embedding: &[f32],
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    if embedding.len() != dim {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "qwen2: forward_step_with_embed expects {dim} F32s (hidden_size); \
                 got {} — caller probably sliced the wrong row of merger output",
                embedding.len(),
            ),
        ));
    }
    let pos = forward_step_prelude(gpu, state)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4) };
    gpu.hip.memcpy_htod(&state.x.buf, bytes)?;
    forward_step_after_x(gpu, weights, cfg, state, pos)
}

/// Embed one token to a host F32 row (`hidden_size`) for batched-prefill
/// splice: lets a VLM example build a [batch, dim] embeds matrix with vision
/// rows interleaved at IMGPAD slots and text rows here. Uses `state.x` scratch.
pub fn embed_token_row(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    token: u32,
) -> HipResult<Vec<f32>> {
    let dim = cfg.hidden_size;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::Q4K => {
            gpu.embedding_lookup_q4k(&weights.token_embd, &state.x, token, dim)?
        }
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &state.x, token, dim)?,
    }
    gpu.download_f32(&state.x)
}

/// Common prefix: bounds-check, upload pos. Returns the position used.
fn forward_step_prelude(gpu: &mut Gpu, state: &Qwen2State) -> HipResult<usize> {
    let pos = state.next_pos;
    if pos >= state.max_seq {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "qwen2: forward_step pos={pos} >= max_seq={}; \
                 rebuild Qwen2State with a larger budget via \
                 Qwen2State::new_with_max_seq",
                state.max_seq
            ),
        ));
    }
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&state.pos_buf, &pos_i32.to_ne_bytes())?;
    Ok(pos)
}

/// Common tail: 28-layer decoder + final RMSNorm + lm_head + bump next_pos.
/// Assumes `state.x` already holds the embedding for `pos` and `state.pos_buf`
/// has been uploaded.
fn forward_step_after_x(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    pos: usize,
) -> HipResult<()> {
    // #397 Ship 6 — forward-as-pipeline. HIPFIRE_FORWARD_LOWERED=1 routes the
    // per-layer decode through the super-op executor (run_layer_program). Default
    // off until fleet byte-parity validated on gfx1100 + gfx1201.
    if qwen2_forward_lowered_enabled() {
        return forward_step_after_x_lowered(gpu, weights, cfg, state, pos);
    }

    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let ctx = DispatchCtx::new(gpu);
    // P-A: single (1×1) device mesh threaded to the dispatch chokepoint.
    let mesh = DeviceMesh::single();

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        // (1–2) RMSNorm + QKV projection via execute_steps.
        // The interpreter selects FusedQkvHfq4G256 / fused-MQ / per-op
        // based on dtype — no model-side branching.
        let qkv_rot = dtype_rotation_plan(layer.wq.gpu_dtype);
        let wrq = layer.wq.dispatch_ref();
        let wrk = layer.wk.dispatch_ref();
        let wrv = layer.wv.dispatch_ref();
        execute_steps_mesh(
            &mesh,
            gpu,
            &ctx,
            &[
                Step::RmsnormAutomatic {
                    x: &state.x,
                    norm_weight: &layer.attn_norm,
                    x_plain: &state.tmp,
                    out: &state.x_rot,
                    awq_scale: layer.wq.awq_scale.as_ref(),
                    k: layer.wq.k,
                    eps: cfg.rms_norm_eps,
                    rotation: qkv_rot,
                },
                Step::Gemv {
                    w: &wrq,
                    input: GemvInput::Prerotated(&state.x_rot),
                    out: &state.q,
                },
                Step::Gemv {
                    w: &wrk,
                    input: GemvInput::Prerotated(&state.x_rot),
                    out: &state.k,
                },
                Step::Gemv {
                    w: &wrv,
                    input: GemvInput::Prerotated(&state.x_rot),
                    out: &state.v,
                },
            ],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

        // (3) QKV bias. attention_bias=true on Qwen2 — three small adds
        // per layer (batch=1, n=q_dim or kv_dim). This is **option (a)**
        // from the plan §5 (3 launches per layer × 28 = 84 launches per
        // decode step), not option (c) as an earlier comment claimed.
        // The plan's preferred (c) is a single batched bias-add of
        // Q/K/V per layer (~28 launches per decode step); reaching (c)
        // needs either a kernel that takes three (buf, bias, n) triples
        // or a refactor of `bias_add_f32` to accept multi-row inputs.
        // Promote to (c) / (b) under the Δ ≥ 5% rule.
        gpu.bias_add_f32(&state.q, &layer.wq_bias, 1, q_dim)?;
        gpu.bias_add_f32(&state.k, &layer.wk_bias, 1, kv_dim)?;
        gpu.bias_add_f32(&state.v, &layer.wv_bias, 1, kv_dim)?;

        // (4) RoPE on q,k (1-D, theta from config). Qwen2 does NOT apply
        // q/k RMSNorm pre-RoPE (Qwen3-only — see lib.rs doc).
        gpu.rope_f32(
            &state.q,
            &state.k,
            &state.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_theta,
        )?;

        // (5) KV cache write at pos.
        gpu.kv_cache_write(&state.k_cache[layer_idx], &state.k, &state.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&state.v_cache[layer_idx], &state.v, &state.pos_buf, kv_dim)?;

        // (6) Attention — split-K flash decode (`attention_flash`): grid
        // [n_heads, n_chunks] saturates the GPU vs the naive single-token
        // attention_f32 (grid [n_heads] = ~14% CU occupancy, 71% of decode
        // GPU time per rocprof). GQA via n_heads / n_kv_heads. F32 KV cache.
        // GQA-aware split-K when there's a group to share K/V loads and the
        // context is long enough to fill the grid (n_kv_heads×n_chunks); else
        // the per-head flash. Both bit-identical; gqa is ~15-23% faster at
        // OCR decode lengths (5k-11k). Falls to flash for short/non-GQA.
        //
        // Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch
        // per layer, no partials buffer, no reduce. Grid = n_kv_heads only
        // (2 for dots.ocr), so lower occupancy but eliminates the partials
        // DRAM round-trip + reduce dispatch. Probe of launch-overhead vs
        // occupancy tradeoff.
        let use_fused = hipfire_config::developer_var("HIPFIRE_GQA_FUSED")
            .map(|v| v == "1")
            .unwrap_or(false);
        if use_fused && n_kv_heads < n_heads {
            gpu.attention_flash_gqa_fused(
                &state.q,
                &state.k_cache[layer_idx],
                &state.v_cache[layer_idx],
                &state.attn_out,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                state.max_seq,
            )?;
        } else if n_kv_heads < n_heads && head_dim == 128 && pos + 1 >= 4096 {
            gpu.attention_gqa_warp(
                &state.q,
                &state.k_cache[layer_idx],
                &state.v_cache[layer_idx],
                &state.attn_out,
                &state.attn_partials,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                state.max_seq,
            )?;
        } else if n_kv_heads < n_heads && pos + 1 >= 4096 {
            Gpu::attention_flash_gqa(
                gpu,
                &state.q,
                &state.k_cache[layer_idx],
                &state.v_cache[layer_idx],
                &state.attn_out,
                &state.attn_partials,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                state.max_seq,
            )?;
        } else {
            Gpu::attention_flash(
                gpu,
                &state.q,
                &state.k_cache[layer_idx],
                &state.v_cache[layer_idx],
                &state.attn_out,
                &state.attn_partials,
                pos + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                state.max_seq,
            )?;
        }

        // (7–8) o_proj + residual via execute_steps.
        let wro = layer.wo.dispatch_ref();
        execute_steps_mesh(
            &mesh,
            gpu,
            &ctx,
            &[Step::GemvResidual {
                w: &wro,
                input: GemvInput::Raw(&state.attn_out),
                residual: &state.x,
                out: &state.o,
            }],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

        // (9–10) FFN norm + gate/up via execute_steps.
        // The interpreter selects FusedGateUpQ8_0 / FusedGateUpHfq4G256 / per-op.
        let ffn_rot = dtype_rotation_plan(layer.w_gate.gpu_dtype);
        let wrg = layer.w_gate.dispatch_ref();
        let wru = layer.w_up.dispatch_ref();
        execute_steps_mesh(
            &mesh,
            gpu,
            &ctx,
            &[
                Step::RmsnormAutomatic {
                    x: &state.x,
                    norm_weight: &layer.ffn_norm,
                    x_plain: &state.tmp,
                    out: &state.x_rot,
                    awq_scale: layer.w_gate.awq_scale.as_ref(),
                    k: layer.w_gate.k,
                    eps: cfg.rms_norm_eps,
                    rotation: ffn_rot,
                },
                Step::Gemv {
                    w: &wrg,
                    input: GemvInput::Prerotated(&state.x_rot),
                    out: &state.gate,
                },
                Step::Gemv {
                    w: &wru,
                    input: GemvInput::Prerotated(&state.x_rot),
                    out: &state.up,
                },
            ],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

        // SwiGLU activation + w_down + residual.
        gpu.silu_mul_f32(&state.gate, &state.up, &state.ffn_hidden)?;
        let wrd = layer.w_down.dispatch_ref();
        execute_steps_mesh(
            &mesh,
            gpu,
            &ctx,
            &[Step::GemvResidual {
                w: &wrd,
                input: GemvInput::Raw(&state.ffn_hidden),
                residual: &state.x,
                out: &state.ffn_out,
            }],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    // Final RMSNorm + lm_head.
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, cfg.rms_norm_eps)?;
    let wr_out = weights.output.dispatch_ref();
    execute_steps_mesh(
        &mesh,
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &wr_out,
            input: GemvInput::Raw(&state.tmp),
            out: &state.logits,
        }],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

    state.next_pos = pos + 1;
    Ok(())
}

/// Convenience: run [`forward_step`] then greedy-argmax the logits.
/// Returns the next token id.
pub fn forward_step_greedy(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    token: u32,
) -> HipResult<u32> {
    forward_step(gpu, weights, cfg, state, token)?;
    gpu.argmax_f32(&state.logits, cfg.vocab_size)
}

/// Batched prefill over a pre-built `[batch × dim]` embedding matrix
/// (row-major F32). The caller resolves every prompt position to an
/// embedding row — token-embedding lookups for text positions, spliced
/// vision-merger rows for image positions — so this path works for both
/// plain Qwen2 and the dots.ocr VL splice without an embedding-lookup
/// branch inside the hot loop.
///
/// Processes the whole prompt in one pass with batched GEMM / RoPE /
/// causal-attention kernels instead of the per-token `forward_step`
/// loop. Fills `state.k_cache` / `state.v_cache` for positions
/// `[next_pos, next_pos + batch)`, advances `state.next_pos`, and leaves
/// the LAST position's logits in `state.logits` (ready for argmax →
/// first generated token). Decode then continues with `forward_step`.
///
/// KV-cache writes use a per-position F32 loop (the cache is F32 and only
/// a single-position `kv_cache_write` exists); a batched-F32 KV kernel is
/// the follow-up if that loop dominates prefill time.
pub fn forward_prefill_batch_embeds(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    embeds: &[f32],
) -> HipResult<()> {
    let dim = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let hidden_dim = cfg.intermediate_size;

    let gemv = gemv_family();
    let ctx = DispatchCtx::new(gpu);

    assert_eq!(
        embeds.len() % dim,
        0,
        "forward_prefill_batch_embeds: embeds.len()={} not a multiple of dim={dim}",
        embeds.len()
    );
    let batch = embeds.len() / dim;
    let base = state.next_pos;
    if base + batch > state.max_seq {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "qwen2: prefill batch={batch} + next_pos={base} > max_seq={}",
                state.max_seq
            ),
        ));
    }

    // Batched projection: Q8 weights get a true batched GEMM; other dtypes
    // fall back to weight_gemm (HFQ4 batched, else a per-row GEMV loop).
    fn proj(
        gpu: &mut Gpu,
        w: &WeightTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        batch: usize,
    ) -> HipResult<()> {
        match w.gpu_dtype {
            DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(&w.buf, x, y, w.m, w.k, batch),
            _ => weight_gemm(gpu, w, x, y, batch),
        }
    }

    let x_batch = gpu.upload_f32(embeds, &[batch, dim])?;
    let tmp_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    let q_batch = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
    let k_batch = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
    let v_batch = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
    let attn_out_batch = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
    let o_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    let gate_batch = gpu.alloc_tensor(&[batch, hidden_dim], DType::F32)?;
    let up_batch = gpu.alloc_tensor(&[batch, hidden_dim], DType::F32)?;
    let ffn_hidden_batch = gpu.alloc_tensor(&[batch, hidden_dim], DType::F32)?;
    let ffn_out_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;

    let use_wmma_causal = (gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12())
        && head_dim == 128
        && batch >= 64;
    let (k_f16_batch, v_f16_batch) = if use_wmma_causal {
        let k16 = gpu.alloc_tensor(&[batch, kv_dim], DType::F16)?;
        let v16 = gpu.alloc_tensor(&[batch, kv_dim], DType::F16)?;
        (Some(k16), Some(v16))
    } else {
        (None, None)
    };

    // Absolute positions [base .. base+batch) for batched RoPE.
    let pos_bytes: Vec<u8> = (0..batch as i32)
        .flat_map(|i| (i + base as i32).to_ne_bytes())
        .collect();
    let pos_array = gpu.alloc_tensor(&[batch], DType::F32)?; // i32 payload, same width
    gpu.hip.memcpy_htod(&pos_array.buf, &pos_bytes)?;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        gpu.rmsnorm_batched(
            &x_batch,
            &layer.attn_norm,
            &tmp_batch,
            batch,
            dim,
            cfg.rms_norm_eps,
        )?;

        proj(gpu, &layer.wq, &tmp_batch, &q_batch, batch)?;
        proj(gpu, &layer.wk, &tmp_batch, &k_batch, batch)?;
        proj(gpu, &layer.wv, &tmp_batch, &v_batch, batch)?;

        gpu.bias_add_f32(&q_batch, &layer.wq_bias, batch, q_dim)?;
        gpu.bias_add_f32(&k_batch, &layer.wk_bias, batch, kv_dim)?;
        gpu.bias_add_f32(&v_batch, &layer.wv_bias, batch, kv_dim)?;

        gpu.rope_batched_f32(
            &q_batch,
            &k_batch,
            &pos_array,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_theta,
            batch,
        )?;

        // Persist post-RoPE K/V to the F32 cache at absolute positions
        // (pos_array = [base..base+batch)) in one launch each, so decode
        // (forward_step) can attend to the whole prompt.
        gpu.kv_cache_write_f32_batched(
            &state.k_cache[layer_idx],
            &k_batch,
            &pos_array,
            kv_dim,
            batch,
        )?;
        gpu.kv_cache_write_f32_batched(
            &state.v_cache[layer_idx],
            &v_batch,
            &pos_array,
            kv_dim,
            batch,
        )?;

        // Attention: WMMA causal flash when head_dim=128 and batch is
        // large enough to fill the M=64 tile. gfx11 and gfx12 use separate
        // kernel siblings because their WMMA operand layouts differ.
        // Keep v3-causal in production: the v4-causal V_lds transpose variant
        // is bench-only until it is fixed for non-128-token prompt lengths.
        if let (Some(k16), Some(v16)) = (&k_f16_batch, &v_f16_batch) {
            gpu.cast_f32_to_f16(&k_batch, k16)?;
            gpu.cast_f32_to_f16(&v_batch, v16)?;
            gpu.attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
                &q_batch,
                k16,
                v16,
                &attn_out_batch,
                batch,
                batch,
                n_heads,
                n_kv_heads,
                head_dim,
            )?;
        } else {
            gpu.attention_causal_batched(
                &q_batch,
                &k_batch,
                &v_batch,
                &attn_out_batch,
                batch,
                n_heads,
                n_kv_heads,
                head_dim,
            )?;
        }

        proj(gpu, &layer.wo, &attn_out_batch, &o_batch, batch)?;
        gpu.add_inplace_f32(&x_batch, &o_batch)?;

        gpu.rmsnorm_batched(
            &x_batch,
            &layer.ffn_norm,
            &tmp_batch,
            batch,
            dim,
            cfg.rms_norm_eps,
        )?;

        proj(gpu, &layer.w_gate, &tmp_batch, &gate_batch, batch)?;
        proj(gpu, &layer.w_up, &tmp_batch, &up_batch, batch)?;
        gpu.silu_mul_f32(&gate_batch, &up_batch, &ffn_hidden_batch)?;
        proj(gpu, &layer.w_down, &ffn_hidden_batch, &ffn_out_batch, batch)?;
        gpu.add_inplace_f32(&x_batch, &ffn_out_batch)?;
    }

    // Final norm + lm_head for the LAST position only → state.logits.
    let last_off = (batch - 1) * dim * 4;
    gpu.hip
        .memcpy_dtod_at(&state.x.buf, 0, &x_batch.buf, last_off, dim * 4)?;
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, cfg.rms_norm_eps)?;
    gemv.run_auto(
        &ctx,
        gpu,
        &weights.output.dispatch_ref(),
        &state.tmp,
        &state.logits,
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

    state.next_pos = base + batch;

    for t in [
        x_batch,
        tmp_batch,
        q_batch,
        k_batch,
        v_batch,
        attn_out_batch,
        o_batch,
        gate_batch,
        up_batch,
        ffn_hidden_batch,
        ffn_out_batch,
        pos_array,
    ] {
        gpu.free_tensor(t)?;
    }
    if let Some(k16) = k_f16_batch {
        gpu.free_tensor(k16)?;
    }
    if let Some(v16) = v_f16_batch {
        gpu.free_tensor(v16)?;
    }
    Ok(())
}

/// Block-parallel spec-decode verify: run a whole `block` of candidate tokens
/// through one batched forward at absolute positions `[position .. position+B)`
/// and return the per-slot greedy argmax (the verifier's pick AFTER each block
/// token). Slot `i` is `argmax(logits)` for query position `position+i`, which
/// is exactly what the sequential `forward_step(block[i])` loop returns — but in
/// one batched layer loop instead of B sequential decodes.
///
/// Correctness hinges on the attention seeing the FULL history: unlike
/// `attention_causal_batched` (intra-block only, blind to prompt/accepted KV),
/// this writes the block's post-RoPE K/V into the F32 cache at absolute
/// positions first, then runs `attention_decode_batched_history`, so query row
/// `i` attends to absolute positions `[0 .. position+i]` (prompt + accepted
/// prefix + causal-within-block). That makes the result byte-equivalent to the
/// sequential verify's per-step flash-decode attention.
///
/// Leaves `state.next_pos = position + B` and `state.k_cache`/`v_cache`
/// populated for `[position .. position+B)` (the spec loop's `commit_prefix` is
/// a no-op for pure attention; the rejected tail is overwritten by the next
/// verify, exactly as in the sequential path).
pub fn forward_verify_block_batched(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    block: &[u32],
    position: usize,
) -> HipResult<Vec<u32>> {
    let dim = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let hidden_dim = cfg.intermediate_size;
    let batch = block.len();

    if position + batch > state.max_seq {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "qwen2: verify block={batch} + position={position} > max_seq={}",
                state.max_seq
            ),
        ));
    }

    fn proj(
        gpu: &mut Gpu,
        w: &WeightTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        batch: usize,
    ) -> HipResult<()> {
        match w.gpu_dtype {
            DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(&w.buf, x, y, w.m, w.k, batch),
            _ => weight_gemm(gpu, w, x, y, batch),
        }
    }

    // Build the [batch × dim] F32 embedding matrix. Q8/HFQ4G256 tables get a
    // single batched GPU lookup (no host round-trip); other formats fall back
    // to the per-token host loop (block is small, so this is cheap and rare).
    let x_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::Q8_0 | EmbeddingFormat::HFQ4G256 => {
            let tok_ids: Vec<i32> = block.iter().map(|&t| t as i32).collect();
            let tok_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tok_ids.as_ptr() as *const u8, batch * 4) };
            let tok_buf = gpu.alloc_tensor(&[batch], DType::F32)?; // i32 payload
            gpu.hip.memcpy_htod(&tok_buf.buf, tok_bytes)?;
            match weights.embd_format {
                EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
                    &weights.token_embd,
                    &x_batch,
                    &tok_buf,
                    batch,
                    dim,
                )?,
                EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
                    &weights.token_embd,
                    &x_batch,
                    &tok_buf,
                    batch,
                    dim,
                )?,
                _ => unreachable!(),
            }
            gpu.free_tensor(tok_buf)?;
        }
        _ => {
            let mut embeds = Vec::with_capacity(batch * dim);
            for &tok in block {
                let row = embed_token_row(gpu, weights, cfg, state, tok)?;
                embeds.extend_from_slice(&row);
            }
            gpu.hip.memcpy_htod(&x_batch.buf, unsafe {
                std::slice::from_raw_parts(embeds.as_ptr() as *const u8, embeds.len() * 4)
            })?;
        }
    }
    let tmp_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    let q_batch = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
    let k_batch = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
    let v_batch = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
    let attn_out_batch = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
    let o_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    let gate_batch = gpu.alloc_tensor(&[batch, hidden_dim], DType::F32)?;
    let up_batch = gpu.alloc_tensor(&[batch, hidden_dim], DType::F32)?;
    let ffn_hidden_batch = gpu.alloc_tensor(&[batch, hidden_dim], DType::F32)?;
    let ffn_out_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
    // Per-row logits + argmax (verifier's pick at each slot).
    let logits_batch = gpu.alloc_tensor(&[batch, cfg.vocab_size], DType::F32)?;
    // i32 payload in an F32 tensor (4-byte width; matches argmax_f32_batched
    // output convention used across the spec-decode paths).
    let argmax_batch = gpu.alloc_tensor(&[batch], DType::F32)?;

    // Absolute positions [position .. position+batch) for batched RoPE and the
    // F32 KV cache writes.
    let pos_bytes: Vec<u8> = (0..batch as i32)
        .flat_map(|i| (i + position as i32).to_ne_bytes())
        .collect();
    let pos_array = gpu.alloc_tensor(&[batch], DType::F32)?; // i32 payload, same width
    gpu.hip.memcpy_htod(&pos_array.buf, &pos_bytes)?;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        gpu.rmsnorm_batched(
            &x_batch,
            &layer.attn_norm,
            &tmp_batch,
            batch,
            dim,
            cfg.rms_norm_eps,
        )?;

        proj(gpu, &layer.wq, &tmp_batch, &q_batch, batch)?;
        proj(gpu, &layer.wk, &tmp_batch, &k_batch, batch)?;
        proj(gpu, &layer.wv, &tmp_batch, &v_batch, batch)?;

        gpu.bias_add_f32(&q_batch, &layer.wq_bias, batch, q_dim)?;
        gpu.bias_add_f32(&k_batch, &layer.wk_bias, batch, kv_dim)?;
        gpu.bias_add_f32(&v_batch, &layer.wv_bias, batch, kv_dim)?;

        gpu.rope_batched_f32(
            &q_batch,
            &k_batch,
            &pos_array,
            n_heads,
            n_kv_heads,
            head_dim,
            cfg.rope_theta,
            batch,
        )?;

        // Persist the block's post-RoPE K/V into the F32 cache at absolute
        // positions [position..position+batch). This makes the in-block causal
        // prefix visible to the history-aware attention below AND overwrites any
        // rejected-tail KV left by a prior verify window.
        gpu.kv_cache_write_f32_batched(
            &state.k_cache[layer_idx],
            &k_batch,
            &pos_array,
            kv_dim,
            batch,
        )?;
        gpu.kv_cache_write_f32_batched(
            &state.v_cache[layer_idx],
            &v_batch,
            &pos_array,
            kv_dim,
            batch,
        )?;

        // Block-parallel decode-with-history: row i attends to [0..position+i].
        gpu.attention_decode_batched_history(
            &q_batch,
            &state.k_cache[layer_idx],
            &state.v_cache[layer_idx],
            &attn_out_batch,
            batch,
            position,
            n_heads,
            n_kv_heads,
            head_dim,
        )?;

        proj(gpu, &layer.wo, &attn_out_batch, &o_batch, batch)?;
        gpu.add_inplace_f32(&x_batch, &o_batch)?;

        gpu.rmsnorm_batched(
            &x_batch,
            &layer.ffn_norm,
            &tmp_batch,
            batch,
            dim,
            cfg.rms_norm_eps,
        )?;

        proj(gpu, &layer.w_gate, &tmp_batch, &gate_batch, batch)?;
        proj(gpu, &layer.w_up, &tmp_batch, &up_batch, batch)?;
        gpu.silu_mul_f32(&gate_batch, &up_batch, &ffn_hidden_batch)?;
        proj(gpu, &layer.w_down, &ffn_hidden_batch, &ffn_out_batch, batch)?;
        gpu.add_inplace_f32(&x_batch, &ffn_out_batch)?;
    }

    // Final norm + lm_head for EVERY row (verify needs per-slot logits).
    gpu.rmsnorm_batched(
        &x_batch,
        &weights.output_norm,
        &tmp_batch,
        batch,
        dim,
        cfg.rms_norm_eps,
    )?;
    // lm_head over the whole [batch × dim] block in one batched GEMM.
    proj(gpu, &weights.output, &tmp_batch, &logits_batch, batch)?;

    gpu.argmax_f32_batched(&logits_batch, &argmax_batch, cfg.vocab_size, batch)?;
    let mut argmax_host: Vec<i32> = vec![0; batch];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, batch * 4)
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_batch.buf)?;
    }
    let out: Vec<u32> = argmax_host.iter().map(|&v| v as u32).collect();

    // Leave the last row's logits in state.logits for callers that want the
    // tail prediction (mirrors the sequential verify's final-step residue).
    let last_off = (batch - 1) * cfg.vocab_size * 4;
    gpu.hip.memcpy_dtod_at(
        &state.logits.buf,
        0,
        &logits_batch.buf,
        last_off,
        cfg.vocab_size * 4,
    )?;

    state.next_pos = position + batch;

    for t in [
        x_batch,
        tmp_batch,
        q_batch,
        k_batch,
        v_batch,
        attn_out_batch,
        o_batch,
        gate_batch,
        up_batch,
        ffn_hidden_batch,
        ffn_out_batch,
        logits_batch,
        argmax_batch,
        pos_array,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok(out)
}

/// Lowered decode is DEFAULT ON (fleet-standard `!= Some("0")`, same
/// convention as qwen35/minimax/deepseek4/lfm2moe); opt out with
/// HIPFIRE_FORWARD_LOWERED=0. Byte-parity validated on gfx1100 (Kevin:
/// short/GQA_FUSED/long-ctx/perf, all identical) and gfx1201 (2026-06-09:
/// short 128/128 ids, GQA_FUSED, long-ctx 5550-tok prompt pos>=4096
/// gqa_warp, default check, perf 266.7 tok/s both paths Δ<=0.2%;
/// qwen2-1.5b hfq4, temp 0). dots-ocr text decode rides this same path.
fn qwen2_forward_lowered_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FORWARD_LOWERED")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Lowered (#397 Ship 6) per-layer decode loop + final norm/head. Behaviorally
/// equivalent to the hand loop in `forward_step_after_x` (validated via
/// FORWARD_LOWERED=0-vs-=1 committed-token md5 A/B). Both `forward_step` and
/// `forward_step_with_embed` funnel through `forward_step_after_x`, so both
/// entry points are covered.
fn forward_step_after_x_lowered(
    gpu: &mut Gpu,
    weights: &Qwen2Weights,
    cfg: &Qwen2Config,
    state: &mut Qwen2State,
    pos: usize,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    // P-A: single (1×1) device mesh threaded to the dispatch chokepoint.
    let mesh = DeviceMesh::single();
    let knobs = DenseKnobs {
        attn_bias: true,
        qk_norm: false,
        rope_theta: cfg.rope_theta,
        norm_eps: cfg.rms_norm_eps,
        n_heads: cfg.num_attention_heads,
        n_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        q_dim: cfg.num_attention_heads * cfg.head_dim,
        kv_dim: cfg.num_key_value_heads * cfg.head_dim,
    };
    let arch = Qwen2Dense {
        weights,
        state: &*state,
        knobs,
        seq_len: pos + 1,
    };
    dense_forward(gpu, &ctx, &arch)?;

    // Final RMSNorm + lm_head (outside layer loop).
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, cfg.rms_norm_eps)?;
    let wr_out = weights.output.dispatch_ref();
    execute_steps_mesh(
        &mesh,
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &wr_out,
            input: GemvInput::Raw(&state.tmp),
            out: &state.logits,
        }],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    state.next_pos = pos + 1;
    Ok(())
}

/// Per-forward [`DenseArch`] wrapper for qwen2-family models (qwen2 / qwen2.5 /
/// dots-ocr text). Routes the projection/FFN body through the shared
/// `dense_forward`; attention stays qwen2's plain-F32 KV write + flash/gqa
/// selector via [`Self::attend`]. Replaced the SuperOp lowered path
/// (`qwen2_lower_program` + `Qwen2Bindings`), now removed.
struct Qwen2Dense<'a> {
    weights: &'a Qwen2Weights,
    state: &'a Qwen2State,
    knobs: DenseKnobs,
    seq_len: usize,
}

impl DenseArch for Qwen2Dense<'_> {
    fn n_layers(&self) -> usize {
        self.weights.layers.len()
    }
    fn knobs(&self) -> &DenseKnobs {
        &self.knobs
    }
    fn scratch(&self) -> DenseScratch<'_> {
        let s = self.state;
        DenseScratch {
            x: &s.x,
            tmp: &s.tmp,
            x_rot: &s.x_rot,
            q: &s.q,
            k: &s.k,
            v: &s.v,
            attn_out: &s.attn_out,
            o: &s.o,
            gate: &s.gate,
            up: &s.up,
            ffn_hidden: &s.ffn_hidden,
            ffn_out: &s.ffn_out,
            pos_buf: &s.pos_buf,
        }
    }
    fn layer(&self, l: usize) -> DenseLayer<'_> {
        let lw = &self.weights.layers[l];
        DenseLayer {
            attn_norm: &lw.attn_norm,
            ffn_norm: &lw.ffn_norm,
            wq: lw.wq.dispatch_ref(),
            wk: lw.wk.dispatch_ref(),
            wv: lw.wv.dispatch_ref(),
            wo: lw.wo.dispatch_ref(),
            w_gate: lw.w_gate.dispatch_ref(),
            w_up: lw.w_up.dispatch_ref(),
            w_down: lw.w_down.dispatch_ref(),
            wq_bias: Some(&lw.wq_bias),
            wk_bias: Some(&lw.wk_bias),
            wv_bias: Some(&lw.wv_bias),
            q_norm: None,
            k_norm: None,
            qkv_rot: dtype_rotation_plan(lw.wq.gpu_dtype),
            ffn_rot: dtype_rotation_plan(lw.w_gate.gpu_dtype),
            qkv_awq: lw.wq.awq_scale.as_ref(),
            ffn_awq: lw.w_gate.awq_scale.as_ref(),
            qkv_k: lw.wq.k,
            ffn_k: lw.w_gate.k,
        }
    }
    fn attend(&self, gpu: &mut Gpu, l: usize) -> HipResult<()> {
        let st = self.state;
        let k = &self.knobs;
        // KV write (plain F32) — bias + RoPE already applied by dense_forward.
        gpu.kv_cache_write(&st.k_cache[l], &st.k, &st.pos_buf, k.kv_dim)?;
        gpu.kv_cache_write(&st.v_cache[l], &st.v, &st.pos_buf, k.kv_dim)?;
        // Attention — 4-way select (exact mirror of the SuperOp run_attend path).
        let use_fused = hipfire_config::developer_var("HIPFIRE_GQA_FUSED")
            .map(|v| v == "1")
            .unwrap_or(false);
        if use_fused && k.n_kv_heads < k.n_heads {
            gpu.attention_flash_gqa_fused(
                &st.q,
                &st.k_cache[l],
                &st.v_cache[l],
                &st.attn_out,
                self.seq_len,
                k.n_heads,
                k.n_kv_heads,
                k.head_dim,
                st.max_seq,
            )
        } else if k.n_kv_heads < k.n_heads && k.head_dim == 128 && self.seq_len >= 4096 {
            gpu.attention_gqa_warp(
                &st.q,
                &st.k_cache[l],
                &st.v_cache[l],
                &st.attn_out,
                &st.attn_partials,
                self.seq_len,
                k.n_heads,
                k.n_kv_heads,
                k.head_dim,
                st.max_seq,
            )
        } else if k.n_kv_heads < k.n_heads && self.seq_len >= 4096 {
            Gpu::attention_flash_gqa(
                gpu,
                &st.q,
                &st.k_cache[l],
                &st.v_cache[l],
                &st.attn_out,
                &st.attn_partials,
                self.seq_len,
                k.n_heads,
                k.n_kv_heads,
                k.head_dim,
                st.max_seq,
            )
        } else {
            Gpu::attention_flash(
                gpu,
                &st.q,
                &st.k_cache[l],
                &st.v_cache[l],
                &st.attn_out,
                &st.attn_partials,
                self.seq_len,
                k.n_heads,
                k.n_kv_heads,
                k.head_dim,
                st.max_seq,
            )
        }
    }
    fn attend_plan(
        &self,
        l: usize,
    ) -> HipResult<
        Option<(
            hipfire_dispatch::families::kv_tier::KvTierPlan,
            hipfire_dispatch::families::attention::AttnParams<'_>,
        )>,
    > {
        use hipfire_dispatch::families::kv_tier::{F32AttnPolicy, KvTierInputs, KvTierPlan};
        let st = self.state;
        let k = &self.knobs;
        // Read HIPFIRE_GQA_FUSED here, mirroring the hand attend() (which also
        // reads it per layer) — same result, strict no-op.
        let fused = hipfire_config::developer_var("HIPFIRE_GQA_FUSED")
            .map(|v| v == "1")
            .unwrap_or(false);
        let pos = self.seq_len - 1;
        let plan = KvTierPlan::derive(KvTierInputs {
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_q8: false,
            quant_fwht: false,
            quant_hfq4: false,
            quant_q4: false,
            quant_int8: false,
            quant_hfq8: false,
            f32_policy: F32AttnPolicy::Gqa {
                n_heads: k.n_heads,
                n_kv_heads: k.n_kv_heads,
                head_dim: k.head_dim,
                fused,
            },
            v_mode_bits: 8,
            pos,
            flash_mode: 0,
            capture_mode: false,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            q8_windowed: false,
            window: 0,
        })
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        let io = hipfire_dispatch::families::attention::AttnParams {
            q: &st.q,
            k: &st.k,
            v: &st.v,
            k_cache: &st.k_cache[l],
            v_cache: &st.v_cache[l],
            k_scales: None,
            v_scales: None,
            pos_buf: &st.pos_buf,
            pos,
            positions: None,
            n_heads: k.n_heads,
            n_kv_heads: k.n_kv_heads,
            head_dim: k.head_dim,
            // KILL 3: the GQA kernels take max_seq as their last arg; the dispatch
            // arm forwards io.physical_cap there, so it MUST be state.max_seq.
            physical_cap: st.max_seq,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&st.attn_partials),
            givens_cos: None,
            givens_sin: None,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &st.attn_out,
        };
        Ok(Some((plan, io)))
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
            .expect("parser errored on a valid Qwen2-1.5B-Instruct config");
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
        assert_eq!(cfg.eos_token_ids, vec![151645]);
    }

    #[test]
    fn parses_dots_ocr_text_config() {
        let cfg = config_from_metadata_json(DOTS_OCR_TEXT_METADATA)
            .expect("parser errored on a valid dots.ocr text config");
        assert!(cfg.attention_bias);
        assert!(!cfg.tie_word_embeddings);
        // The array form is preserved; scalar is the first element.
        // dots.ocr's real `eos_token_id: [151643, 151673]` — both
        // tokens must end up in the stop-set so streaming EOS doesn't
        // miss the `<|endofassistant|>` 151673 case. The test fixture
        // mimics what would happen if `generation_config.json` got
        // merged into the metadata (which it currently doesn't — see
        // R5 in the plan).
        assert_eq!(cfg.eos_token_id, 151643);
        assert_eq!(cfg.eos_token_ids, vec![151643, 151673]);
        assert_eq!(cfg.max_position_embeddings, 131072);
    }

    #[test]
    fn missing_required_field_returns_err() {
        let bad = r#"{"config": {"hidden_size": 1536}}"#;
        assert!(config_from_metadata_json(bad).is_err());
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
        // Missing eos falls back to the ChatML scalar [151645].
        assert_eq!(cfg.eos_token_id, 151645);
        assert_eq!(cfg.eos_token_ids, vec![151645]);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn eos_array_preserves_full_set() {
        // Qwen2-1.5B-Instruct's generation_config has [151645, 151643]
        // (note order differs from dots.ocr). Verify the parser
        // preserves order and arity, not just the scalar accessor.
        let with_array = r#"{
            "config": {
                "hidden_size": 1536,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "intermediate_size": 8960,
                "vocab_size": 151936,
                "eos_token_id": [151645, 151643]
            }
        }"#;
        let cfg = config_from_metadata_json(with_array).expect("array eos should parse");
        assert_eq!(cfg.eos_token_id, 151645);
        assert_eq!(cfg.eos_token_ids, vec![151645, 151643]);
    }

    #[test]
    fn eos_falls_back_to_generation_config_when_absent_from_config() {
        // dots.ocr's real shape: config.json carries NO eos_token_id at
        // all; the [151643, 151673] array lives only in
        // generation_config.json. The quantiser-side R5 fix packs
        // generation_config into HFQ metadata so the parser can find
        // it. Without this fallback the parser would default to the
        // ChatML [151645] which never fires on a correct dots.ocr
        // response (151645 `<|im_end|>` is not in the dots.ocr template).
        let json = r#"{
            "config": {
                "hidden_size": 1536,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "intermediate_size": 8960,
                "vocab_size": 151936
            },
            "generation_config": {
                "eos_token_id": [151643, 151673],
                "pad_token_id": 151643
            }
        }"#;
        let cfg = config_from_metadata_json(json).expect("config + generation_config should parse");
        assert_eq!(cfg.eos_token_id, 151643);
        assert_eq!(cfg.eos_token_ids, vec![151643, 151673]);
    }

    #[test]
    fn eos_in_config_takes_precedence_over_generation_config() {
        // If config.eos_token_id IS set, it wins — generation_config
        // is only consulted as a fallback. Guards against a future
        // quantiser that packs both with conflicting values.
        let json = r#"{
            "config": {
                "hidden_size": 1536,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "intermediate_size": 8960,
                "vocab_size": 151936,
                "eos_token_id": 151645
            },
            "generation_config": {
                "eos_token_id": [151643, 151673]
            }
        }"#;
        let cfg = config_from_metadata_json(json).expect("should parse");
        assert_eq!(cfg.eos_token_id, 151645);
        assert_eq!(cfg.eos_token_ids, vec![151645]);
    }
}
