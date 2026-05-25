//! ModelSource: unified interface for loading model weights from different
//! container formats (HFQ, safetensors, etc.).
//!
//! The Architecture trait's `load_weights` and `config_from_source` consume
//! `&dyn ModelSource` so the same loading code works for hipfire's native HFQ
//! format and HuggingFace safetensors (ParoQuant, AWQ, etc.).

use std::collections::HashMap;

/// Metadata about a single tensor in a model file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Safetensors dtype string: "F16", "F32", "I32", "I16", "BF16", etc.
    pub dtype: String,
    pub shape: Vec<usize>,
    /// For HFQ: the quant_type byte. For safetensors: 0xFF (use dtype instead).
    pub quant_type: u8,
    /// Byte offset into the backing store.
    pub data_offset: usize,
    /// Byte size of the tensor data.
    pub data_size: usize,
}

/// Quantization config parsed from HFQ metadata or HF config.json.
#[derive(Debug, Clone, Default)]
pub struct QuantConfig {
    pub method: String,        // "paroquant", "awq", "gptq", "" (HFQ native)
    pub bits: u8,              // 4
    pub group_size: u32,       // 128
    pub krot: u8,              // 8 for ParoQuant, 0 otherwise
    /// Regex patterns for layers excluded from quantization (kept FP16).
    pub dynamic_excludes: Vec<String>,
}

/// Unified interface for reading model data from HFQ files or safetensors
/// directories. Both `HfqFile` and `SafetensorsSource` implement this.
pub trait ModelSource {
    /// JSON metadata blob. For HFQ: the embedded metadata. For safetensors:
    /// the contents of config.json formatted as HFQ-compatible metadata.
    fn metadata_json(&self) -> &str;

    /// Architecture ID for dispatch.
    /// 0 = LLaMA/Mistral, 1 = Qwen3/Qwen2, 5 = Qwen3.5 dense, 6 = MoE.
    fn arch_id(&self) -> u32;

    /// Quantization config (if detected from metadata).
    fn quant_config(&self) -> Option<&QuantConfig>;

    /// Look up a tensor by name. Returns metadata + byte slice.
    /// Returns None if the tensor doesn't exist or the mmap was dropped.
    fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])>;

    /// Look up tensor metadata without data (for pre-screening).
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo>;

    /// All tensor names in the source.
    fn tensor_names(&self) -> Vec<&str>;

    /// Path to the model directory or file (for weight pager, logging).
    fn path(&self) -> &std::path::Path;

    /// Path to tokenizer.json (if available in the model directory).
    /// HFQ embeds the tokenizer in metadata; safetensors models ship it
    /// as a separate file.
    fn tokenizer_json_path(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Chat template string (Jinja) if available.
    fn chat_template(&self) -> Option<String> {
        None
    }
}

/// Open a model from a path, auto-detecting the format.
/// - If path is a directory with config.json: opens as SafetensorsSource
/// - If path ends in .hfq: opens as HfqFile
/// - Otherwise: tries HfqFile first, then directory
pub fn open_model(path: &std::path::Path) -> Result<Box<dyn ModelSource>, String> {
    if path.is_dir() {
        let config_path = path.join("config.json");
        if config_path.exists() {
            let source = crate::safetensors_source::SafetensorsSource::open(path)
                .map_err(|e| format!("safetensors open failed: {e}"))?;
            Ok(Box::new(source))
        } else {
            Err(format!("{}: directory has no config.json", path.display()))
        }
    } else {
        let hfq = crate::hfq::HfqFile::open(path)
            .map_err(|e| format!("{e}"))?;
        Ok(Box::new(hfq))
    }
}
