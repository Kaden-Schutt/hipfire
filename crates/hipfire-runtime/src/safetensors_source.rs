//! SafetensorsSource: load HuggingFace safetensors models directly.
//!
//! Supports ParoQuant, AWQ, and unquantized safetensors models.
//! Reads config.json for architecture detection and quantization config.
//! Mmaps .safetensors files and serves tensor data by name.

use crate::model_source::{ModelSource, QuantConfig, TensorInfo};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
    header_size: usize,
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

            // Parse safetensors header
            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let raw: serde_json::Value = serde_json::from_str(header_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let header_size = 8 + header_len;

            if let serde_json::Value::Object(map) = raw {
                for (name, meta) in map {
                    if name == "__metadata__" {
                        continue;
                    }
                    let dtype = meta["dtype"].as_str().unwrap_or("F16").to_string();
                    let shape: Vec<usize> = meta["shape"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                        .unwrap_or_default();
                    let offsets = meta["data_offsets"]
                        .as_array()
                        .map(|a| {
                            let start = a[0].as_u64().unwrap_or(0) as usize;
                            let end = a[1].as_u64().unwrap_or(0) as usize;
                            (start, end)
                        })
                        .unwrap_or((0, 0));

                    let tensor_idx = tensors.len();
                    let info = TensorInfo {
                        name: name.clone(),
                        dtype,
                        shape,
                        quant_type: 0xFF, // not an HFQ quant_type
                        data_offset: header_size + offsets.0,
                        data_size: offsets.1 - offsets.0,
                    };
                    tensors.push(info);
                    tensor_map.insert(name, (file_idx, tensor_idx));
                }
            }

            files.push(SafetensorsFile {
                _file: file,
                mmap,
                header_size,
            });
        }

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
        Some((info, &mmap[info.data_offset..info.data_offset + info.data_size]))
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
        if p.exists() { Some(p) } else { None }
    }

    fn chat_template(&self) -> Option<String> {
        let p = self.dir.join("tokenizer_config.json");
        let mut s = String::new();
        File::open(p).ok()?.read_to_string(&mut s).ok()?;
        let v: serde_json::Value = serde_json::from_str(&s).ok()?;
        v.get("chat_template")?.as_str().map(|s| s.to_string())
    }
}

fn derive_arch_id(config: &serde_json::Value) -> u32 {
    let archs = config.get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    // Check text_config for MoE indicators
    let text_config = config.get("text_config").unwrap_or(config);
    let has_experts = text_config.get("num_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) > 0;

    for arch in &archs {
        let arch_lower = arch.to_lowercase();
        if arch_lower.contains("qwen3_5") || arch_lower.contains("qwen3.5")
            || arch_lower.contains("qwen3_6") || arch_lower.contains("qwen3.6")
        {
            return if has_experts { 6 } else { 5 };
        }
        if arch_lower.contains("qwen3") || arch_lower.contains("qwen2") {
            return 1;
        }
        if arch_lower.contains("llama") || arch_lower.contains("mistral") {
            return 0;
        }
    }

    // Fallback: check model_type
    let model_type = config.get("model_type")
        .or_else(|| text_config.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match model_type {
        "qwen3_5" | "qwen3.5" | "qwen3_6" | "qwen3.6" => {
            if has_experts { 6 } else { 5 }
        }
        "qwen3" | "qwen2" => 1,
        "llama" | "mistral" => 0,
        _ => {
            eprintln!("warning: unknown model_type '{model_type}', defaulting to arch_id=5 (Qwen3.5)");
            5
        }
    }
}

fn parse_quant_config(config: &serde_json::Value) -> Option<QuantConfig> {
    let qc = config.get("quantization_config")?;
    let method = qc.get("quant_method")?.as_str()?.to_string();
    let bits = qc.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as u8;
    let group_size = qc.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as u32;
    let krot = qc.get("krot").and_then(|v| v.as_u64()).unwrap_or(0) as u8;

    let dynamic_excludes = qc.get("dynamic")
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
    let model_type = text_config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    meta.insert("architecture".to_string(), serde_json::Value::String(model_type.to_string()));

    // Embed the full config.json as the "config" key
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw_config) {
        meta.insert("config".to_string(), parsed);
    }

    serde_json::to_string(&serde_json::Value::Object(meta)).unwrap_or_default()
}
