//! SafetensorsSource: load HuggingFace safetensors models directly.
//!
//! Supports ParoQuant, AWQ, and unquantized safetensors models.
//! Reads config.json for architecture detection and quantization config.
//! Mmaps .safetensors files and serves tensor data by name.

use crate::model_source::{ModelSource, QuantConfig, TensorInfo};
use half::bf16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
}

pub struct SafetensorsSource {
    dir: PathBuf,
    files: Vec<SafetensorsFile>,
    tensors: Vec<TensorInfo>,
    tensor_map: HashMap<String, (usize, usize)>, // name -> (file_idx, tensor_idx)
    metadata_json_cached: String,
    arch_id: u32,
    quant_config: Option<QuantConfig>,
}

impl SafetensorsSource {
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        // Read config.json
        let config_path = dir.join("config.json");
        let mut config_str = String::new();
        File::open(&config_path)?.read_to_string(&mut config_str)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Derive arch_id from architectures field
        let arch_id = derive_arch_id(&config);

        // Parse quantization config
        let quant_config = parse_quant_config(&config);

        // Build metadata JSON in HFQ-compatible format
        let metadata_json_cached = build_metadata_json(&config, &config_str);

        // Find and open all .safetensors files
        let mut st_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        st_paths.sort();

        if st_paths.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{}: no .safetensors files found", dir.display()),
            ));
        }

        let mut files = Vec::new();
        let mut tensors = Vec::new();
        let mut tensor_map = HashMap::new();

        for (file_idx, st_path) in st_paths.iter().enumerate() {
            let file = File::open(st_path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            let parsed = SafeTensors::deserialize(&mmap)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let mmap_start = mmap.as_ptr() as usize;
            for (name, view) in parsed.iter() {
                let data_offset = (view.data().as_ptr() as usize)
                    .checked_sub(mmap_start)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("safetensors tensor {name} is outside its mmap"),
                        )
                    })?;
                let tensor_idx = tensors.len();
                let info = TensorInfo {
                    name: name.to_string(),
                    dtype: view.dtype().to_string(),
                    shape: view.shape().to_vec(),
                    quant_type: 0xFF, // not an HFQ quant_type
                    data_offset,
                    data_size: view.data().len(),
                };
                tensors.push(info);
                tensor_map.insert(name.to_string(), (file_idx, tensor_idx));
            }

            files.push(SafetensorsFile { _file: file, mmap });
        }

        tracing::debug!(
            model_dir = %dir.display(),
            shard_count = files.len(),
            tensor_count = tensors.len(),
            arch_id,
            quantized = quant_config.is_some(),
            "opened safetensors model source"
        );

        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            tensors,
            tensor_map,
            metadata_json_cached,
            arch_id,
            quant_config,
        })
    }

    /// Public accessor so `loader_api` doesn't need the `ModelSource` trait in scope.
    pub fn arch_id(&self) -> u32 {
        self.arch_id
    }
}

impl ModelSource for SafetensorsSource {
    fn metadata_json(&self) -> &str {
        &self.metadata_json_cached
    }

    fn arch_id(&self) -> u32 {
        self.arch_id
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        self.quant_config.as_ref()
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        let &(file_idx, tensor_idx) = self.tensor_map.get(name)?;
        let info = &self.tensors[tensor_idx];
        let mmap = &self.files[file_idx].mmap;
        Some((
            info,
            &mmap[info.data_offset..info.data_offset + info.data_size],
        ))
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let &(_file_idx, tensor_idx) = self.tensor_map.get(name)?;
        Some(&self.tensors[tensor_idx])
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    fn tokenizer_json_path(&self) -> Option<PathBuf> {
        let p = self.dir.join("tokenizer.json");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    fn chat_template(&self) -> Option<String> {
        // Newer HF convention (transformers 5.x): a standalone
        // `chat_template.jinja` file (e.g. North-Mini-Code / cohere2_moe).
        let jinja = self.dir.join("chat_template.jinja");
        if let Ok(mut f) = File::open(&jinja) {
            let mut s = String::new();
            if f.read_to_string(&mut s).is_ok() && !s.trim().is_empty() {
                return Some(s);
            }
        }
        // Older convention: a `chat_template` field in tokenizer_config.json.
        let p = self.dir.join("tokenizer_config.json");
        let mut s = String::new();
        File::open(p).ok()?.read_to_string(&mut s).ok()?;
        let v: serde_json::Value = serde_json::from_str(&s).ok()?;
        v.get("chat_template")?.as_str().map(|s| s.to_string())
    }
}

fn derive_arch_id(config: &serde_json::Value) -> u32 {
    let archs = config
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    // Check text_config for MoE indicators
    let text_config = config.get("text_config").unwrap_or(config);
    let has_experts = text_config
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        > 0;

    for arch in &archs {
        let arch_lower = arch.to_lowercase();
        if arch_lower.contains("qwen3_5")
            || arch_lower.contains("qwen3.5")
            || arch_lower.contains("qwen3_6")
            || arch_lower.contains("qwen3.6")
        {
            return if has_experts { 6 } else { 5 };
        }
        // qwen2 → arch_id=7 (Qwen2Carrier loads the Q/K/V attention biases the
        // llama-family Dir loader drops); qwen3 → arch_id=1 (LlamaCarrier).
        if arch_lower.contains("qwen2") {
            return 7;
        }
        if arch_lower.contains("qwen3") {
            return 1;
        }
        if arch_lower.contains("llama") || arch_lower.contains("mistral") {
            return 0;
        }
        if arch_lower.contains("gemma4_unified_assistant") {
            return 22;
        }
        if arch_lower.contains("gemma4") {
            return 13;
        }
        if arch_lower.contains("muse_glimmer_assistant") {
            return 23;
        }
        if arch_lower.contains("muse_glimmer") {
            return 14;
        }
    }

    // Fallback: check model_type
    let model_type = config
        .get("model_type")
        .or_else(|| text_config.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match model_type {
        "qwen3_5" | "qwen3.5" | "qwen3_6" | "qwen3.6" => {
            if has_experts {
                6
            } else {
                5
            }
        }
        "qwen3" => 1,
        // qwen2 dirs route to arch_id=7 (Qwen2Carrier / hipfire-arch-qwen2) so
        // the Q/K/V `attention_bias=true` biases load — the llama-family Dir
        // loader (arch_id=1) drops them and produces garbage.
        "qwen2" => 7,
        "llama" | "mistral" => 0,
        // Per-expert / VLM arches whose safetensors Dir paths route to their
        // dedicated carriers. model_type strings mirror the quantizer ingest
        // map (hipfire-quantize/src/main.rs auto_arch_id).
        "dots_ocr" => 8,
        "deepseek_v4" => 9,
        "minimax_m2" => 10,
        "lfm2_moe" | "lfm2" => 11,
        "cohere2_moe" => 12,
        "gemma4_text" | "gemma4" => 13,
        "gemma4_unified_assistant" => 22,
        "muse_glimmer" | "muse_glimmer_text" => 14,
        "muse_glimmer_assistant" => 23,
        _ => {
            // C1: unrecognized model_type → an explicit unclaimed sentinel that NO
            // carrier matches, so `load_model` fails cleanly with "no carrier for
            // <dir>" instead of silently mis-routing to Qwen35 (arch_id=5) and dying
            // deep in weight loading with a confusing error.
            eprintln!(
                "warning: unrecognized model_type '{model_type}'; no carrier claims it \
                 (add a carrier or extend derive_arch_id's model_type mapping)"
            );
            UNCLAIMED_ARCH_ID
        }
    }
}

/// Sentinel `arch_id` emitted by [`derive_arch_id`] for an unrecognized
/// `model_type`. No carrier's `claims_arch_id` matches it, so routing fails
/// loudly with a clean "no carrier" error rather than silently defaulting to
/// Qwen35. Far outside the assigned range (0..=64) the registry tests sweep.
const UNCLAIMED_ARCH_ID: u32 = u32::MAX;

fn parse_quant_config(config: &serde_json::Value) -> Option<QuantConfig> {
    let qc = config.get("quantization_config")?;
    let method = qc.get("quant_method")?.as_str()?.to_string();
    let bits = qc.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as u8;
    let group_size = qc.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as u32;
    let krot = qc.get("krot").and_then(|v| v.as_u64()).unwrap_or(0) as u8;

    let dynamic_excludes = qc
        .get("dynamic")
        .and_then(|d| d.as_object())
        .map(|obj| {
            obj.keys()
                .filter(|k| k.starts_with("-:"))
                .map(|k| k.strip_prefix("-:").unwrap_or(k).to_string())
                .collect()
        })
        .unwrap_or_default();

    Some(QuantConfig {
        method,
        bits,
        group_size,
        krot,
        dynamic_excludes,
    })
}

fn build_metadata_json(config: &serde_json::Value, raw_config: &str) -> String {
    // Build HFQ-compatible metadata: { "architecture": "...", "config": {...} }
    // The Qwen35 config parser expects metadata_json to contain a "config" key.
    let mut meta = serde_json::Map::new();

    // Determine architecture string
    let text_config = config.get("text_config").unwrap_or(config);
    let model_type = text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    meta.insert(
        "architecture".to_string(),
        serde_json::Value::String(model_type.to_string()),
    );

    // Embed the full config.json as the "config" key
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw_config) {
        meta.insert("config".to_string(), parsed);
    }

    serde_json::to_string(&serde_json::Value::Object(meta)).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// BF16 (bfloat16) decode helpers
// ---------------------------------------------------------------------------

/// Widen a BF16 (bfloat16) value to F32.
/// BF16 is the upper 16 bits of an IEEE-754 F32 number — widening is left-shifting by 16.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// Convert BF16 byte slice to F16 byte vector (owned).
/// Each BF16 value is widened to F32, then narrowed to F16.
pub fn bf16_bytes_to_f16(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(2)
        .map(|c| {
            let bf16 = u16::from_le_bytes([c[0], c[1]]);
            let f32_val = f32::from_bits((bf16 as u32) << 16);
            crate::llama::f32_to_f16(f32_val).to_le_bytes()
        })
        .flatten()
        .collect()
}

/// Convert BF16 byte slice to F32 vector.
pub fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bf16 = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bf16 as u32) << 16)
        })
        .collect()
}

/// Convert tensor bytes to F16 bytes based on dtype string.
/// Handles F16 (passthrough), BF16 (decode), F32 (narrow).
/// Panics on unknown dtype (fail-fast over silent wrong results).
/// NOTE: n_elements validation removed — caller responsibility.
pub fn source_bytes_to_f16_stream(source_dtype: &str, data: &[u8]) -> Vec<u8> {
    match source_dtype {
        "F16" => data.to_vec(),
        "BF16" => bf16_bytes_to_f16(data),
        "F32" => data
            .chunks_exact(4)
            .map(|c| {
                let f32_val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                crate::llama::f32_to_f16(f32_val).to_le_bytes()
            })
            .flatten()
            .collect(),
        other => panic!(
            "unsupported source dtype '{other}' for fp-to-f16 conversion (expected F16/BF16/F32)"
        ),
    }
}

/// Convert tensor bytes to F32 vector based on dtype string.
/// Panics on unknown dtype.
pub fn source_bytes_to_f32_vec(source_dtype: &str, data: &[u8]) -> Vec<f32> {
    match source_dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => bf16_bytes_to_f32(data),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!(
            "unsupported source dtype '{other}' for fp-to-f32 conversion (expected F16/BF16/F32)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn known_model_types_route_as_expected() {
        assert_eq!(derive_arch_id(&json!({ "model_type": "llama" })), 0);
        assert_eq!(derive_arch_id(&json!({ "model_type": "mistral" })), 0);
        // qwen2 → 7 (Qwen2Carrier, loads attn biases); qwen3 → 1 (LlamaCarrier).
        assert_eq!(derive_arch_id(&json!({ "model_type": "qwen2" })), 7);
        assert_eq!(derive_arch_id(&json!({ "model_type": "qwen3" })), 1);
        assert_eq!(
            derive_arch_id(&json!({ "architectures": ["Qwen2ForCausalLM"] })),
            7
        );
        assert_eq!(
            derive_arch_id(&json!({ "architectures": ["Qwen3ForCausalLM"] })),
            1
        );
        assert_eq!(derive_arch_id(&json!({ "model_type": "qwen3.5" })), 5);
        assert_eq!(
            derive_arch_id(&json!({ "model_type": "qwen3.5", "num_experts": 8 })),
            6
        );
        assert_eq!(derive_arch_id(&json!({ "model_type": "minimax_m2" })), 10);
        // Per-expert / VLM Dir arches routed to dedicated carriers (mirrors the
        // quantizer ingest map). Lock the strings so a rename can't silently
        // unclaim a real checkpoint.
        assert_eq!(derive_arch_id(&json!({ "model_type": "dots_ocr" })), 8);
        assert_eq!(derive_arch_id(&json!({ "model_type": "deepseek_v4" })), 9);
        assert_eq!(derive_arch_id(&json!({ "model_type": "lfm2_moe" })), 11);
        assert_eq!(derive_arch_id(&json!({ "model_type": "lfm2" })), 11);
        assert_eq!(derive_arch_id(&json!({ "model_type": "cohere2_moe" })), 12);
    }

    /// C1: an unrecognized model_type must NOT silently become Qwen35 (arch_id=5).
    /// It returns the unclaimed sentinel so routing fails with a clean "no carrier".
    #[test]
    fn unrecognized_model_type_is_unclaimed_not_qwen35() {
        let id = derive_arch_id(&json!({ "model_type": "totally_unknown_arch" }));
        assert_eq!(id, UNCLAIMED_ARCH_ID);
        assert_ne!(id, 5, "must not default to Qwen35");
    }

    #[test]
    fn bf16_to_f32_basic_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0f32); // normal
        assert_eq!(bf16_to_f32(0xC000), -2.0f32); // normal negative
        assert_eq!(bf16_to_f32(0x0000), 0.0f32); // zero
        assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits()); // neg zero
        assert!(bf16_to_f32(0x0001) > 0.0); // subnormal
        assert!(bf16_to_f32(0x7FC0).is_nan()); // NaN (quiet)
        assert!(bf16_to_f32(0x7F81).is_nan()); // NaN (signaling)
        assert!(bf16_to_f32(0xFFC0).is_nan()); // negative NaN
    }

    #[test]
    fn source_bytes_roundtrip() {
        // F16 passthrough: F16 1.0 → stays F16 1.0
        let f16_data = vec![0x00u8, 0x3C];
        let result = source_bytes_to_f16_stream("F16", &f16_data);
        assert_eq!(result, f16_data);

        // BF16→F16: BF16 1.0 (0x3F80 LE) → F16 1.0 (0x3C00 LE)
        let bf16_data = vec![0x80u8, 0x3F];
        let result = source_bytes_to_f16_stream("BF16", &bf16_data);
        assert_eq!(result, vec![0x00, 0x3C]);

        // F32→F16: F32 1.0 → F16 1.0
        let f32_data = vec![0x00u8, 0x00, 0x80, 0x3F];
        let result = source_bytes_to_f16_stream("F32", &f32_data);
        assert_eq!(result, vec![0x00, 0x3C]);

        // BF16→F32: BF16 -2.0 (0xC000 LE) → -2.0 F32
        let bf16_data = vec![0x00u8, 0xC0];
        let result = source_bytes_to_f32_vec("BF16", &bf16_data);
        assert_eq!(result, vec![-2.0f32]);
    }

    #[test]
    #[should_panic(expected = "unsupported source dtype")]
    fn source_bytes_to_f16_unknown_dtype_panics() {
        source_bytes_to_f16_stream("FP8", &[0u8; 4]);
    }
}
