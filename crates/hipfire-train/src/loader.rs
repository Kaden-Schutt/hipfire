// SPDX-License-Identifier: Apache-2.0
//! Load a dense LLaMA safetensors checkpoint into fp32 GPU tensors.
//!
//! Uses `SafetensorsSource` purely as a name→raw-bytes mmap (no quantizer
//! involvement, per Phase 0 plan §2). Every weight is converted to fp32 on
//! upload — Supra-50M ships bf16. Weights are the *frozen base*; the trainable
//! LoRA adapters are created separately.

use crate::config::LlamaConfig;
use hipfire_model::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{Gpu, GpuTensor};
use std::path::Path;

/// Per-layer frozen weights (HF row-major `[out, in]`, ready for
/// `gemm_f32_train` with `trans_b=true`).
pub struct LlamaLayerF32 {
    pub input_layernorm: GpuTensor,          // [hidden]
    pub q_proj: GpuTensor,                   // [q_dim, hidden]
    pub k_proj: GpuTensor,                   // [kv_dim, hidden]
    pub v_proj: GpuTensor,                   // [kv_dim, hidden]
    pub o_proj: GpuTensor,                   // [hidden, q_dim]
    pub post_attention_layernorm: GpuTensor, // [hidden]
    pub gate_proj: GpuTensor,                // [inter, hidden]
    pub up_proj: GpuTensor,                  // [inter, hidden]
    pub down_proj: GpuTensor,                // [hidden, inter]
}

pub struct LlamaWeightsF32 {
    pub embed_tokens: GpuTensor, // [vocab, hidden]
    pub layers: Vec<LlamaLayerF32>,
    pub final_norm: GpuTensor, // [hidden]
    /// `None` when `tie_word_embeddings` — logits use `embed_tokens`.
    pub lm_head: Option<GpuTensor>, // [vocab, hidden]
}

/// Open `dir`, parse config, and upload all weights as fp32.
pub fn load_llama_fp32(
    gpu: &mut Gpu,
    dir: &Path,
) -> Result<(LlamaConfig, LlamaWeightsF32), String> {
    let cfg = LlamaConfig::from_dir(dir)?;
    let src = SafetensorsSource::open(dir).map_err(|e| format!("open safetensors: {e}"))?;

    let load = |gpu: &mut Gpu, name: &str, want: &[usize]| -> Result<GpuTensor, String> {
        load_tensor_f32(gpu, &src, name, want)
    };

    let h = cfg.hidden_size;
    let q = cfg.q_dim();
    let kv = cfg.kv_dim();
    let inter = cfg.intermediate_size;

    let embed_tokens = load(gpu, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
    let final_norm = load(gpu, "model.norm.weight", &[h])?;
    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load(gpu, "lm_head.weight", &[cfg.vocab_size, h])?)
    };

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        layers.push(LlamaLayerF32 {
            input_layernorm: load(gpu, &format!("{p}.input_layernorm.weight"), &[h])?,
            q_proj: load(gpu, &format!("{p}.self_attn.q_proj.weight"), &[q, h])?,
            k_proj: load(gpu, &format!("{p}.self_attn.k_proj.weight"), &[kv, h])?,
            v_proj: load(gpu, &format!("{p}.self_attn.v_proj.weight"), &[kv, h])?,
            o_proj: load(gpu, &format!("{p}.self_attn.o_proj.weight"), &[h, q])?,
            post_attention_layernorm: load(
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            )?,
            gate_proj: load(gpu, &format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
            up_proj: load(gpu, &format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
            down_proj: load(gpu, &format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
        });
    }

    Ok((
        cfg,
        LlamaWeightsF32 {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        },
    ))
}

/// Fetch a tensor's raw bytes, convert to fp32, validate shape, upload.
fn load_tensor_f32(
    gpu: &mut Gpu,
    src: &SafetensorsSource,
    name: &str,
    want_shape: &[usize],
) -> Result<GpuTensor, String> {
    let (info, bytes) = src
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.shape != want_shape {
        return Err(format!(
            "tensor {name}: shape {:?} != expected {:?}",
            info.shape, want_shape
        ));
    }
    let f32s = bytes_to_f32(&info.dtype, bytes).map_err(|e| format!("tensor {name}: {e}"))?;
    let expected: usize = want_shape.iter().product();
    if f32s.len() != expected {
        return Err(format!(
            "tensor {name}: {} elems != {} from shape",
            f32s.len(),
            expected
        ));
    }
    gpu.upload_f32(&f32s, want_shape)
        .map_err(|e| format!("upload {name}: {e:?}"))
}

/// Decode an HFQM tensor's bytes (by `quant_type`) to fp32 — the layer-1 runtime
/// unification: training loads its base from the *exact served artifact*.
/// Handles BF16(16)/F32(2)/Q8F16(3) now; Qtip3G256(31) is a clear TODO (qtip2-sim
/// `.hfq` is all bf16, so this covers the 2-bit path today).
fn decode_hfq_tensor(quant_type: u8, data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    match quant_type {
        2 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        16 => Ok(data
            .chunks_exact(2)
            .map(|c| crate::hfq_patch::bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        3 => Ok(hipfire_runtime::quant::dequant_q8f16(data, n)),
        31 => Err(
            "Qtip3G256 .hfq decode not yet implemented in hipfire-train \
                   (use a bf16/qtip2-sim .hfq, or load from the source safetensors)"
                .to_string(),
        ),
        other => Err(format!("unsupported quant_type {other} for hfq decode")),
    }
}

/// Load a dense LLaMA model's base weights directly from a `.hfq` artifact,
/// decoded to fp32 — so the training "student" IS the served model (no
/// re-quantize / format-matching). Config comes from the HFQM metadata.
pub fn load_llama_from_hfq(
    gpu: &mut Gpu,
    path: &Path,
) -> Result<(LlamaConfig, LlamaWeightsF32), String> {
    use std::collections::HashMap;
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let (entries, meta) = crate::hfq_patch::parse_hfq(&bytes)?;
    let cfg = LlamaConfig::from_hfq_metadata(&meta)?;
    let map: HashMap<&str, &crate::hfq_patch::HfqEntry> =
        entries.iter().map(|e| (e.name.as_str(), e)).collect();

    let load = |gpu: &mut Gpu, name: &str, want: &[usize]| -> Result<GpuTensor, String> {
        let e = map
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let data = &bytes[e.data_offset..e.data_offset + e.data_size];
        let n: usize = want.iter().product();
        let f32s = decode_hfq_tensor(e.quant_type, data, n).map_err(|x| format!("{name}: {x}"))?;
        if f32s.len() != n {
            return Err(format!("{name}: {} elems != {n}", f32s.len()));
        }
        gpu.upload_f32(&f32s, want)
            .map_err(|x| format!("upload {name}: {x:?}"))
    };

    let (h, q, kv, inter) = (
        cfg.hidden_size,
        cfg.q_dim(),
        cfg.kv_dim(),
        cfg.intermediate_size,
    );
    let embed_tokens = load(gpu, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
    let final_norm = load(gpu, "model.norm.weight", &[h])?;
    let lm_head = if map.contains_key("lm_head.weight") {
        Some(load(gpu, "lm_head.weight", &[cfg.vocab_size, h])?)
    } else {
        None
    };
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        layers.push(LlamaLayerF32 {
            input_layernorm: load(gpu, &format!("{p}.input_layernorm.weight"), &[h])?,
            q_proj: load(gpu, &format!("{p}.self_attn.q_proj.weight"), &[q, h])?,
            k_proj: load(gpu, &format!("{p}.self_attn.k_proj.weight"), &[kv, h])?,
            v_proj: load(gpu, &format!("{p}.self_attn.v_proj.weight"), &[kv, h])?,
            o_proj: load(gpu, &format!("{p}.self_attn.o_proj.weight"), &[h, q])?,
            post_attention_layernorm: load(
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            )?,
            gate_proj: load(gpu, &format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
            up_proj: load(gpu, &format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
            down_proj: load(gpu, &format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
        });
    }
    Ok((
        cfg,
        LlamaWeightsF32 {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        },
    ))
}

/// Convert little-endian safetensors bytes of the given dtype to fp32.
fn bytes_to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => {
            if bytes.len() % 4 != 0 {
                return Err("F32 byte len not /4".into());
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        other => Err(format!("unsupported dtype {other} for fp32 training load")),
    }
}
