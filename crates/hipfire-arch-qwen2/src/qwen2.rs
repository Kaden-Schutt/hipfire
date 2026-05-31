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
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, weight_gemv, weight_gemm, EmbeddingFormat, WeightTensor};
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
    // Build the full EOS set first, then the scalar accessor is its
    // first element. Both array and scalar config layouts are accepted.
    //
    // Lookup order:
    //   1. text_config.eos_token_id / config.eos_token_id (Qwen2-1.5B carries this)
    //   2. generation_config.eos_token_id (dots.ocr's [151643, 151673]
    //      lives only here per R5 — the quantiser now packs this sibling
    //      JSON into HFQ metadata so this fallback is reachable)
    //   3. Default [151645] (ChatML `<|im_end|>`)
    let parse_eos = |val: &serde_json::Value| -> Vec<u32> {
        match val {
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|e| e.as_u64().map(|n| n as u32))
                .collect(),
            serde_json::Value::Number(n) => n.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
            _ => Vec::new(),
        }
    };
    let mut eos_token_ids: Vec<u32> = tc.get("eos_token_id")
        .map(parse_eos)
        .unwrap_or_default();
    if eos_token_ids.is_empty() {
        if let Some(gc_eos) = meta.get("generation_config").and_then(|gc| gc.get("eos_token_id")) {
            eos_token_ids = parse_eos(gc_eos);
        }
    }
    let eos_token_ids = if eos_token_ids.is_empty() {
        vec![151645]
    } else {
        eos_token_ids
    };
    let eos_token_id = eos_token_ids[0];

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
        eos_token_ids,
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

    /// Release every GPU buffer back to the pool. Consumes self.
    /// Mirrors `LlamaWeights::free_gpu` and `Qwen35Weights::free_gpu`
    /// — the daemon calls this on unload to actually return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        let _ = gpu.free_tensor(self.output.buf);
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
    // Quant-type coverage matches `load_lm_head` tied branch above, so a
    // tied-embeddings model produces consistent embed + lm_head paths.
    match info.quant_type {
        6 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok((buf, EmbeddingFormat::HFQ4G256))
        }
        7 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok((buf, EmbeddingFormat::HFQ4G128))
        }
        3 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok((buf, EmbeddingFormat::Q8_0))
        }
        1 => {
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let buf = gpu.upload_f32(&f32_data, &[cfg.vocab_size, cfg.hidden_size])?;
            Ok((buf, EmbeddingFormat::F32))
        }
        qt => panic!("qwen2: unsupported embedding quant_type {qt}; \
                     handled: 1 (F16→F32), 3 (Q8_0), 6 (HFQ4G256), 7 (HFQ4G128). \
                     Extend load_embed_tokens to handle this format."),
    }
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
                // F16 source: load_embed_tokens promoted to F32 on host.
                // We must do the same so gpu_dtype=F32 matches the actual
                // buffer contents. Mirror qwen35.rs:1438-1447.
                let f32_data: Vec<f32> = data.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        f32_data.as_ptr() as *const u8,
                        f32_data.len() * 4,
                    )
                };
                gpu.upload_raw(bytes, &[cfg.vocab_size, cfg.hidden_size])?
            }
            qt => panic!("qwen2: unsupported tied embedding quant_type {qt}"),
        };
        let wt = WeightTensor {
            buf,
            gpu_dtype: dtype,
            m: cfg.vocab_size,
            k: cfg.hidden_size,
            row_stride: 0,
            paro: None,
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
    // Harmonised with `hipfire-arch-dots-ocr::dots_ocr::load_norm_weight_raw`.
    // Catches a manifest-shape vs caller-arg mismatch (e.g. norm tensor on
    // disk is `[hidden_size]` but caller passed `head_dim`) at upload
    // time rather than letting a wrong-shape GpuTensor cascade through
    // the forward pass.
    assert_eq!(
        f32_data.len(), n,
        "qwen2: norm {name} has {} elements, expected {n}", f32_data.len(),
    );
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
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G256, m, k, row_stride: 0, paro: None, awq_scale: None })
        }
        7 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G128, m, k, row_stride: 0, paro: None, awq_scale: None })
        }
        3 => {
            // Q8F16 (= GGML Q8_0 layout): [F16 scale ‖ 32× INT8]. The fused
            // qkv_hfq4g256 fast path doesn't apply here; forward_step
            // falls back to three weight_gemv calls per layer (which
            // dispatches to gpu.gemv_q8_0 for this gpu_dtype). Used by
            // the high-precision sweep (`--format q8`) to discriminate
            // forward-pass correctness from HFQ4 quant noise.
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::Q8_0, m, k, row_stride: 0, paro: None, awq_scale: None })
        }
        1 => {
            let buf = gpu.upload_raw(&data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F16, m, k, row_stride: 0, paro: None, awq_scale: None })
        }
        qt => panic!("qwen2: unsupported weight quant_type {qt} for {name}. \
                     This loader handles qt ∈ {{1 (F16), 3 (Q8F16), 6 (HFQ4G256), 7 (HFQ4G128)}}. \
                     Extend load_weight_tensor or wait for the Transformer-extraction PR \
                     to pick up qwen35's full quant_type matrix."),
    }
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
/// - **No `x_rot`.** Qwen2 uses HFQ4 weights with no FWHT rotation
///   per row; the MagnumQuant `x_rot` scratch in `ForwardScratch` is
///   dead weight here.
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
    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Qwen2Config,
        max_seq: usize,
    ) -> HipResult<Self> {
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

        Ok(Self {
            x:           gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp:         gpu.alloc_tensor(&[dim], DType::F32)?,
            q:           gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k:           gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v:           gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out:    gpu.alloc_tensor(&[q_dim], DType::F32)?,
            o:           gpu.alloc_tensor(&[dim], DType::F32)?,
            gate:        gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            up:          gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            ffn_hidden:  gpu.alloc_tensor(&[hidden_dim], DType::F32)?,
            ffn_out:     gpu.alloc_tensor(&[dim], DType::F32)?,
            logits:      gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)?,
            attn_partials: gpu.alloc_tensor(&[attn_partials_len], DType::F32)?,
            pos_buf:     gpu.hip.malloc(4)?,
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
        for t in [self.x, self.tmp, self.q, self.k, self.v, self.attn_out,
                  self.o, self.gate, self.up, self.ffn_hidden,
                  self.ffn_out, self.logits, self.attn_partials] {
            let _ = gpu.free_tensor(t);
        }
        for t in self.k_cache { let _ = gpu.free_tensor(t); }
        for t in self.v_cache { let _ = gpu.free_tensor(t); }
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
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &state.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &state.x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &state.x, token, dim)?,
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &state.x, token, dim)?,
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
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4)
    };
    gpu.hip.memcpy_htod(&state.x.buf, bytes)?;
    forward_step_after_x(gpu, weights, cfg, state, pos)
}

/// Embed one token to a host F32 row (`hidden_size`) for batched-prefill
/// splice: lets a VLM example build a [batch, dim] embeds matrix with vision
/// rows interleaved at IMGPAD slots and text rows here. Uses `state.x` scratch.
pub fn embed_token_row(
    gpu: &mut Gpu, weights: &Qwen2Weights, cfg: &Qwen2Config,
    state: &mut Qwen2State, token: u32,
) -> HipResult<Vec<f32>> {
    let dim = cfg.hidden_size;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &state.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &state.x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &state.x, token, dim)?,
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, &state.x, token, dim)?,
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
    gpu.hip.memcpy_htod(&state.pos_buf, &pos_i32.to_ne_bytes())?;
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
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];

        // (1) RMSNorm(x → tmp) with input_layernorm.
        gpu.rmsnorm_f32(&state.x, &layer.attn_norm, &state.tmp, cfg.rms_norm_eps)?;

        // (2) QKV projection. fused_qkv_hfq4g256 expects all three weights
        // to be HFQ4G256; otherwise fall through to three individual
        // weight_gemv calls. This keeps the bring-up path open for F16
        // weights (qt=1) while the HFQ4G256 fast path is the default.
        let all_hfq4g256 = layer.wq.gpu_dtype == DType::HFQ4G256
            && layer.wk.gpu_dtype == DType::HFQ4G256
            && layer.wv.gpu_dtype == DType::HFQ4G256;
        if all_hfq4g256 {
            gpu.fused_qkv_hfq4g256(
                &layer.wq.buf, &layer.wk.buf, &layer.wv.buf,
                &state.tmp,
                &state.q, &state.k, &state.v,
                layer.wq.m, layer.wk.m, layer.wv.m,
                layer.wq.k,
            )?;
        } else {
            weight_gemv(gpu, &layer.wq, &state.tmp, &state.q)?;
            weight_gemv(gpu, &layer.wk, &state.tmp, &state.k)?;
            weight_gemv(gpu, &layer.wv, &state.tmp, &state.v)?;
        }

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
        gpu.rope_f32(&state.q, &state.k, &state.pos_buf, n_heads, n_kv_heads, head_dim, cfg.rope_theta)?;

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
        let use_fused = std::env::var("HIPFIRE_GQA_FUSED")
            .map(|v| v == "1").unwrap_or(false);
        if use_fused && n_kv_heads < n_heads {
            gpu.attention_flash_gqa_fused(
                &state.q, &state.k_cache[layer_idx], &state.v_cache[layer_idx],
                &state.attn_out,
                pos + 1, n_heads, n_kv_heads, head_dim, state.max_seq,
            )?;
        } else if n_kv_heads < n_heads && head_dim == 128 && pos + 1 >= 4096 {
            gpu.attention_gqa_warp(
                &state.q, &state.k_cache[layer_idx], &state.v_cache[layer_idx],
                &state.attn_out, &state.attn_partials,
                pos + 1, n_heads, n_kv_heads, head_dim, state.max_seq,
            )?;
        } else if n_kv_heads < n_heads && pos + 1 >= 4096 {
            Gpu::attention_flash_gqa(gpu,
                &state.q, &state.k_cache[layer_idx], &state.v_cache[layer_idx],
                &state.attn_out, &state.attn_partials,
                pos + 1, n_heads, n_kv_heads, head_dim, state.max_seq,
            )?;
        } else {
            Gpu::attention_flash(gpu,
                &state.q, &state.k_cache[layer_idx], &state.v_cache[layer_idx],
                &state.attn_out, &state.attn_partials,
                pos + 1, n_heads, n_kv_heads, head_dim, state.max_seq,
            )?;
        }

        // (7) o_proj (no bias) + (8) residual.
        weight_gemv(gpu, &layer.wo, &state.attn_out, &state.o)?;
        gpu.add_inplace_f32(&state.x, &state.o)?;

        // (9) FFN norm.
        gpu.rmsnorm_f32(&state.x, &layer.ffn_norm, &state.tmp, cfg.rms_norm_eps)?;

        // (10) SwiGLU: gate = silu(w_gate(x)) * w_up(x); down(...).
        if layer.w_gate.gpu_dtype == DType::Q8_0
            && layer.w_up.gpu_dtype == DType::Q8_0
            && layer.w_gate.k == layer.w_up.k
        {
            gpu.fused_gate_up_q8_0(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &state.tmp,
                &state.gate,
                &state.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        } else {
            weight_gemv(gpu, &layer.w_gate, &state.tmp, &state.gate)?;
            weight_gemv(gpu, &layer.w_up, &state.tmp, &state.up)?;
        }
        gpu.silu_mul_f32(&state.gate, &state.up, &state.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &state.ffn_hidden, &state.ffn_out)?;

        // (11) Residual.
        gpu.add_inplace_f32(&state.x, &state.ffn_out)?;
    }

    // Final RMSNorm + lm_head.
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, cfg.rms_norm_eps)?;
    weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;

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

    assert_eq!(embeds.len() % dim, 0,
        "forward_prefill_batch_embeds: embeds.len()={} not a multiple of dim={dim}", embeds.len());
    let batch = embeds.len() / dim;
    let base = state.next_pos;
    if base + batch > state.max_seq {
        return Err(hip_bridge::HipError::new(0, &format!(
            "qwen2: prefill batch={batch} + next_pos={base} > max_seq={}", state.max_seq)));
    }

    // Batched projection: Q8 weights get a true batched GEMM; other dtypes
    // fall back to weight_gemm (HFQ4 batched, else a per-row GEMV loop).
    fn proj(gpu: &mut Gpu, w: &WeightTensor, x: &GpuTensor, y: &GpuTensor, batch: usize) -> HipResult<()> {
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

    let use_wmma_causal =
        (gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12())
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

        gpu.rmsnorm_batched(&x_batch, &layer.attn_norm, &tmp_batch, batch, dim, cfg.rms_norm_eps)?;

        proj(gpu, &layer.wq, &tmp_batch, &q_batch, batch)?;
        proj(gpu, &layer.wk, &tmp_batch, &k_batch, batch)?;
        proj(gpu, &layer.wv, &tmp_batch, &v_batch, batch)?;

        gpu.bias_add_f32(&q_batch, &layer.wq_bias, batch, q_dim)?;
        gpu.bias_add_f32(&k_batch, &layer.wk_bias, batch, kv_dim)?;
        gpu.bias_add_f32(&v_batch, &layer.wv_bias, batch, kv_dim)?;

        gpu.rope_batched_f32(&q_batch, &k_batch, &pos_array,
            n_heads, n_kv_heads, head_dim, cfg.rope_theta, batch)?;

        // Persist post-RoPE K/V to the F32 cache at absolute positions
        // (pos_array = [base..base+batch)) in one launch each, so decode
        // (forward_step) can attend to the whole prompt.
        gpu.kv_cache_write_f32_batched(&state.k_cache[layer_idx], &k_batch, &pos_array, kv_dim, batch)?;
        gpu.kv_cache_write_f32_batched(&state.v_cache[layer_idx], &v_batch, &pos_array, kv_dim, batch)?;

        // Attention: WMMA causal flash when head_dim=128 and batch is
        // large enough to fill the M=64 tile. gfx11 and gfx12 use separate
        // kernel siblings because their WMMA operand layouts differ.
        // Keep v3-causal in production: the v4-causal V_lds transpose variant
        // is bench-only until it is fixed for non-128-token prompt lengths.
        if let (Some(k16), Some(v16)) = (&k_f16_batch, &v_f16_batch) {
            gpu.cast_f32_to_f16(&k_batch, k16)?;
            gpu.cast_f32_to_f16(&v_batch, v16)?;
            gpu.attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
                &q_batch, k16, v16, &attn_out_batch,
                batch, batch, n_heads, n_kv_heads, head_dim,
            )?;
        } else {
            gpu.attention_causal_batched(&q_batch, &k_batch, &v_batch, &attn_out_batch,
                batch, n_heads, n_kv_heads, head_dim)?;
        }

        proj(gpu, &layer.wo, &attn_out_batch, &o_batch, batch)?;
        gpu.add_inplace_f32(&x_batch, &o_batch)?;

        gpu.rmsnorm_batched(&x_batch, &layer.ffn_norm, &tmp_batch, batch, dim, cfg.rms_norm_eps)?;

        proj(gpu, &layer.w_gate, &tmp_batch, &gate_batch, batch)?;
        proj(gpu, &layer.w_up, &tmp_batch, &up_batch, batch)?;
        gpu.silu_mul_f32(&gate_batch, &up_batch, &ffn_hidden_batch)?;
        proj(gpu, &layer.w_down, &ffn_hidden_batch, &ffn_out_batch, batch)?;
        gpu.add_inplace_f32(&x_batch, &ffn_out_batch)?;
    }

    // Final norm + lm_head for the LAST position only → state.logits.
    let last_off = (batch - 1) * dim * 4;
    gpu.hip.memcpy_dtod_at(&state.x.buf, 0, &x_batch.buf, last_off, dim * 4)?;
    gpu.rmsnorm_f32(&state.x, &weights.output_norm, &state.tmp, cfg.rms_norm_eps)?;
    weight_gemv(gpu, &weights.output, &state.tmp, &state.logits)?;

    state.next_pos = base + batch;

    for t in [x_batch, tmp_batch, q_batch, k_batch, v_batch, attn_out_batch,
              o_batch, gate_batch, up_batch, ffn_hidden_batch, ffn_out_batch,
              pos_array] {
        gpu.free_tensor(t)?;
    }
    if let Some(k16) = k_f16_batch { gpu.free_tensor(k16)?; }
    if let Some(v16) = v_f16_batch { gpu.free_tensor(v16)?; }
    Ok(())
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
        assert_eq!(cfg.eos_token_ids, vec![151645]);
    }

    #[test]
    fn parses_dots_ocr_text_config() {
        let cfg = config_from_metadata_json(DOTS_OCR_TEXT_METADATA)
            .expect("parser returned None on a valid dots.ocr text config");
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
        let cfg = config_from_metadata_json(json)
            .expect("config + generation_config should parse");
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
