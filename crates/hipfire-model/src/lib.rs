// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared model artifact identity helpers and model-source contracts.

pub mod gguf;
/// Generated model-support tables (`ARCH_ROWS`/`QUANT_TABLE`/`GATE_TABLE`).
/// Source of truth: `docs/model-support.toml`; regenerate with
/// `cargo run -p hipfire-cli -- gen-model-support`.
pub mod model_support_generated;
pub mod tokenizer;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use hipfire_hash::{file_hash, stable_hash_bytes};

/// Extension preferred order for fuzzy model discovery.
pub const QUANT_PREFERENCE: &[&str] =
    &["-mq4", "-hf4", "-mq3", "-lloyd-mq2", "-mq6", "-hf6", "-q8"];

/// Metadata about a single tensor in a model source.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuantConfig {
    pub method: String,
    pub bits: u8,
    pub group_size: u32,
    pub krot: u8,
    /// Regex patterns for layers excluded from quantization (kept FP16).
    pub dynamic_excludes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifestEntry {
    pub role: String,
    pub identifier: String,
    pub path_exists: bool,
    pub file_size: Option<u64>,
    pub file_hash: Option<String>,
    pub tag_hash: Option<String>,
    pub hfq_arch_id: Option<u32>,
    pub hfq_metadata_hash: Option<String>,
    pub quantization_hash: Option<Value>,
    pub metadata_status: String,
    pub metadata_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfqMetadata {
    pub arch_id: u32,
    pub metadata_json: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HfqTokenizerMetadata {
    HfJson(String),
    GgufMeta(Value),
}

/// Stable identity for routing requests to a compatible loaded model worker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWorkerKey {
    pub artifact_path: String,
    pub artifact_digest: Option<String>,
    pub arch_id: String,
    pub quant_family: String,
    pub state_mode: String,
    pub max_seq_bucket: usize,
    pub accelerator_kind: Option<String>,
    pub device_id: Option<String>,
    pub feature_flags: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWorkerId {
    pub value: String,
}

impl ModelWorkerId {
    pub fn from_runtime_parts(arch_id: u32, pp: usize, kv_mode: Option<&str>) -> Self {
        Self {
            value: format!(
                "worker:arch{}:pp{}:{}",
                arch_id,
                pp,
                kv_mode.unwrap_or("unknown")
            ),
        }
    }
}

pub fn parse_model_worker_id(msg: &Value, default_worker_id: &str) -> ModelWorkerId {
    let value = msg
        .get("worker_id")
        .or_else(|| msg.get("worker_key_id"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(default_worker_id)
        .to_string();
    ModelWorkerId { value }
}

pub fn has_worker_or_model_identity(msg: &Value) -> bool {
    msg.get("worker_key_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .is_some()
        || msg
            .get("model")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .is_some()
}

pub const ARCH_ID_LLAMA_MISTRAL: u32 = 0;
pub const ARCH_ID_QWEN3_QWEN2_LEGACY: u32 = 1;
pub const ARCH_ID_QWEN35_DENSE: u32 = 5;
pub const ARCH_ID_QWEN35_MOE: u32 = 6;
pub const ARCH_ID_QWEN2: u32 = 7;
pub const ARCH_ID_DOTS_OCR: u32 = 8;
pub const ARCH_ID_DEEPSEEK4_FLASH: u32 = 9;
pub const ARCH_ID_MINIMAX_M2: u32 = 10;
pub const ARCH_ID_LFM2_MOE: u32 = 11;
pub const ARCH_ID_GEMMA3_TEXT: u32 = 12;
pub const ARCH_ID_GEMMA3_VL: u32 = 13;
pub const ARCH_ID_NEMOTRON_H: u32 = 14;
pub const ARCH_ID_MAMBA2: u32 = 15;

/// Runtime model arch IDs that must appear in `docs/model-support.toml`.
pub const KNOWN_RUNTIME_ARCH_IDS: &[(u32, &str)] = &[
    (ARCH_ID_LLAMA_MISTRAL, "llama"),
    (ARCH_ID_QWEN3_QWEN2_LEGACY, "qwen3-legacy"),
    (ARCH_ID_QWEN35_DENSE, "qwen3.5-dense"),
    (ARCH_ID_QWEN35_MOE, "qwen3.5-moe"),
    (ARCH_ID_QWEN2, "qwen2"),
    (ARCH_ID_DOTS_OCR, "dots-ocr"),
    (ARCH_ID_DEEPSEEK4_FLASH, "deepseek4"),
    (ARCH_ID_MINIMAX_M2, "minimax"),
    (ARCH_ID_LFM2_MOE, "lfm2-moe"),
    (ARCH_ID_GEMMA3_TEXT, "gemma3"),
    (ARCH_ID_GEMMA3_VL, "gemma3-vl"),
    (ARCH_ID_NEMOTRON_H, "nemotron_h"),
    (ARCH_ID_MAMBA2, "mamba2"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelArchFamily {
    LlamaMistral,
    Qwen3Qwen2Legacy,
    Qwen35Dense,
    Qwen35Moe,
    Qwen2,
    DotsOcr,
    DeepSeek4Flash,
    MiniMaxM2,
    Lfm2Moe,
    Gemma3Text,
    Gemma3Vl,
    NemotronH,
    Mamba2,
    Unknown,
}

pub fn model_arch_family(arch_id: u32) -> ModelArchFamily {
    match arch_id {
        ARCH_ID_LLAMA_MISTRAL => ModelArchFamily::LlamaMistral,
        ARCH_ID_QWEN3_QWEN2_LEGACY => ModelArchFamily::Qwen3Qwen2Legacy,
        ARCH_ID_QWEN35_DENSE => ModelArchFamily::Qwen35Dense,
        ARCH_ID_QWEN35_MOE => ModelArchFamily::Qwen35Moe,
        ARCH_ID_QWEN2 => ModelArchFamily::Qwen2,
        ARCH_ID_DOTS_OCR => ModelArchFamily::DotsOcr,
        ARCH_ID_DEEPSEEK4_FLASH => ModelArchFamily::DeepSeek4Flash,
        ARCH_ID_MINIMAX_M2 => ModelArchFamily::MiniMaxM2,
        ARCH_ID_LFM2_MOE => ModelArchFamily::Lfm2Moe,
        ARCH_ID_GEMMA3_TEXT => ModelArchFamily::Gemma3Text,
        ARCH_ID_GEMMA3_VL => ModelArchFamily::Gemma3Vl,
        ARCH_ID_NEMOTRON_H => ModelArchFamily::NemotronH,
        ARCH_ID_MAMBA2 => ModelArchFamily::Mamba2,
        _ => ModelArchFamily::Unknown,
    }
}

pub fn is_qwen35_dense_arch_id(arch_id: u32) -> bool {
    model_arch_family(arch_id) == ModelArchFamily::Qwen35Dense
}

pub fn is_qwen35_moe_arch_id(arch_id: u32) -> bool {
    model_arch_family(arch_id) == ModelArchFamily::Qwen35Moe
}

pub fn is_qwen35_family_arch_id(arch_id: u32) -> bool {
    matches!(
        model_arch_family(arch_id),
        ModelArchFamily::Qwen35Dense | ModelArchFamily::Qwen35Moe
    )
}

pub fn normalize_feature_flags(flags: &[String]) -> Vec<String> {
    let mut flags = flags.to_vec();
    flags.sort();
    flags.dedup();
    flags
}

pub fn normalize_model_worker_key(key: &ModelWorkerKey) -> ModelWorkerKey {
    ModelWorkerKey {
        artifact_path: key.artifact_path.clone(),
        artifact_digest: key.artifact_digest.clone(),
        arch_id: key.arch_id.clone(),
        quant_family: key.quant_family.clone(),
        state_mode: key.state_mode.clone(),
        max_seq_bucket: key.max_seq_bucket,
        accelerator_kind: Some(
            key.accelerator_kind
                .clone()
                .unwrap_or_else(|| "hip".to_string()),
        ),
        device_id: Some(key.device_id.clone().unwrap_or_else(|| "0".to_string())),
        feature_flags: normalize_feature_flags(&key.feature_flags),
    }
}

pub fn model_worker_key_id(key: &ModelWorkerKey) -> String {
    let normalized = normalize_model_worker_key(key);
    [
        normalized
            .artifact_digest
            .unwrap_or(normalized.artifact_path),
        normalized.arch_id,
        normalized.quant_family,
        normalized.state_mode,
        normalized.max_seq_bucket.to_string(),
        normalized
            .accelerator_kind
            .unwrap_or_else(|| "hip".to_string()),
        normalized.device_id.unwrap_or_else(|| "0".to_string()),
        normalized.feature_flags.join("+"),
    ]
    .join("|")
}

pub fn same_model_worker_key(a: &ModelWorkerKey, b: &ModelWorkerKey) -> bool {
    model_worker_key_id(a) == model_worker_key_id(b)
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorInventory {
    pub source: String,
    pub devices: Vec<AcceleratorDeviceInfo>,
}

impl AcceleratorInventory {
    pub fn not_probed() -> Self {
        Self {
            source: "not_probed".to_string(),
            devices: Vec::new(),
        }
    }

    pub fn from_devices(source: impl Into<String>, devices: Vec<AcceleratorDeviceInfo>) -> Self {
        Self {
            source: source.into(),
            devices,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorDeviceInfo {
    pub kind: String,
    pub device_id: String,
    pub ordinal: Option<usize>,
    pub arch: Option<String>,
    pub name: Option<String>,
    pub total_memory_bytes: Option<u64>,
    pub integrated: Option<bool>,
    pub runtime: Option<String>,
    pub available: bool,
    pub selected: bool,
    pub reason: Option<String>,
}

impl AcceleratorDeviceInfo {
    pub fn hip(
        device_id: impl Into<String>,
        ordinal: usize,
        arch: Option<String>,
        total_memory_bytes: Option<u64>,
        integrated: Option<bool>,
        runtime: Option<String>,
    ) -> Self {
        Self {
            kind: "hip".to_string(),
            device_id: device_id.into(),
            ordinal: Some(ordinal),
            arch,
            total_memory_bytes,
            integrated,
            runtime,
            available: true,
            ..Default::default()
        }
    }

    pub fn npu_xdna1(
        device_id: impl Into<String>,
        ordinal: Option<usize>,
        runtime: Option<String>,
        available: bool,
        reason: Option<String>,
    ) -> Self {
        Self {
            kind: "npu".to_string(),
            device_id: device_id.into(),
            ordinal,
            arch: Some("xdna1".to_string()),
            name: Some("XDNA1 NPU".to_string()),
            runtime,
            available,
            reason,
            ..Default::default()
        }
    }

    pub fn device_class(&self) -> &'static str {
        match self.integrated {
            Some(true) => "integrated",
            Some(false) => "discrete",
            None => "unknown",
        }
    }
}

pub fn accelerator_inventory_json(inventory: &AcceleratorInventory) -> serde_json::Value {
    let devices = inventory
        .devices
        .iter()
        .map(accelerator_device_info_json)
        .collect::<Vec<_>>();
    serde_json::json!({
        "source": inventory.source,
        "device_count": inventory.devices.len(),
        "devices": devices,
    })
}

pub fn accelerator_device_info_json(device: &AcceleratorDeviceInfo) -> serde_json::Value {
    serde_json::json!({
        "kind": device.kind,
        "device_id": device.device_id,
        "ordinal": device.ordinal,
        "arch": device.arch,
        "name": device.name,
        "total_memory_bytes": device.total_memory_bytes,
        "integrated": device.integrated,
        "device_class": device.device_class(),
        "runtime": device.runtime,
        "available": device.available,
        "selected": device.selected,
        "reason": device.reason,
    })
}

/// Unified interface for reading model data from HFQ files or safetensors
/// directories. Concrete loaders live in backend/runtime crates.
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
    fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])>;

    /// Look up tensor metadata without data (for pre-screening).
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo>;

    /// All tensor names in the source.
    fn tensor_names(&self) -> Vec<&str>;

    /// Path to the model directory or file (for weight pager, logging).
    fn path(&self) -> &Path;

    /// Path to tokenizer.json (if available in the model directory).
    /// HFQ embeds the tokenizer in metadata; safetensors models ship it
    /// as a separate file.
    fn tokenizer_json_path(&self) -> Option<PathBuf> {
        None
    }

    /// Chat template string (Jinja) if available.
    fn chat_template(&self) -> Option<String> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArtifactFormat {
    Hfq,
    Gguf,
    SafetensorsDirectory,
    Unknown,
}

pub fn detect_model_artifact_format(path: &Path) -> ModelArtifactFormat {
    if path.is_dir() {
        if path.join("config.json").exists() {
            ModelArtifactFormat::SafetensorsDirectory
        } else {
            ModelArtifactFormat::Unknown
        }
    } else {
        match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
        {
            Some("hfq") => ModelArtifactFormat::Hfq,
            Some("gguf") => ModelArtifactFormat::Gguf,
            _ => ModelArtifactFormat::Unknown,
        }
    }
}

/// Apply the shared model-source opening policy while concrete loader crates
/// provide the actual HFQ and safetensors constructors.
pub fn open_model_source_with<T, HfqErr, SafetensorsErr>(
    path: &Path,
    open_hfq: impl FnOnce(&Path) -> Result<T, HfqErr>,
    open_safetensors: impl FnOnce(&Path) -> Result<T, SafetensorsErr>,
) -> Result<T, String>
where
    HfqErr: Display,
    SafetensorsErr: Display,
{
    if path.is_dir() {
        let config_path = path.join("config.json");
        if config_path.exists() {
            open_safetensors(path).map_err(|e| format!("safetensors open failed: {e}"))
        } else {
            Err(format!("{}: directory has no config.json", path.display()))
        }
    } else {
        open_hfq(path).map_err(|e| format!("{e}"))
    }
}

/// Stable tokenizer fingerprint used for compatibility checks between target
/// and draft tokenizers. The caller supplies vocabulary in token-id order and
/// special tokens in the tokenizer's canonical matching order.
pub fn tokenizer_signature(
    vocab: &[String],
    special_tokens: &[(String, u32)],
    bos_id: u32,
    eos_id: u32,
    eot_id: Option<u32>,
) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    };

    for tok in vocab {
        mix(tok.as_bytes());
    }
    for (token, id) in special_tokens {
        mix(token.as_bytes());
        mix(&id.to_le_bytes());
    }
    mix(&bos_id.to_le_bytes());
    mix(&eos_id.to_le_bytes());
    mix(&eot_id.unwrap_or(u32::MAX).to_le_bytes());
    h
}

/// Extract the tokenizer payload embedded in HFQ metadata.
///
/// Safetensors-origin HFQ files store a HuggingFace `tokenizer.json` blob in
/// `tokenizer`; GGUF-origin HFQ files preserve the original GGUF tokenizer
/// metadata under `gguf_meta`. Runtime still owns the actual encoder, but the
/// model crate owns this artifact metadata selection policy.
pub fn hfq_tokenizer_metadata(
    metadata_json: &str,
) -> Result<Option<HfqTokenizerMetadata>, serde_json::Error> {
    let meta: Value = serde_json::from_str(metadata_json)?;
    if let Some(tok_str) = meta.get("tokenizer").and_then(Value::as_str) {
        return Ok(Some(HfqTokenizerMetadata::HfJson(tok_str.to_string())));
    }
    if let Some(gguf_meta) = meta.get("gguf_meta") {
        return Ok(Some(HfqTokenizerMetadata::GgufMeta(gguf_meta.clone())));
    }
    Ok(None)
}

/// Extract the upstream HuggingFace Jinja chat template from HFQ metadata.
pub fn hfq_chat_template(metadata_json: &str) -> Option<String> {
    let meta: Value = serde_json::from_str(metadata_json).ok()?;
    meta.get("tokenizer_config")?
        .get("chat_template")?
        .as_str()
        .map(ToString::to_string)
}

/// Read an optional HuggingFace `tokenizer.json` file from a model directory.
///
/// Safetensors directories carry tokenizer configuration as a sidecar file.
/// Existing runtime behavior treats a missing or unreadable sidecar as absent
/// and reserves hard errors for successfully-read but malformed tokenizer
/// content.
pub fn read_optional_tokenizer_json(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// Normalize a user-facing model tag into the fuzzy filename search stem.
pub fn normalize_tag_stem(tag: &str) -> String {
    tag.replace(':', "-").to_lowercase()
}

/// Return whether a filename is a role sidecar rather than a primary model.
pub fn is_role_sidecar_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.ends_with(".triattn.hfq")
        || name.ends_with(".dflash.hfq")
        || name.ends_with(".mtp.hfq")
        || name.ends_with(".hfqm2.hfq")
        || name.ends_with(".hfqm2.hfq.tmp")
}

/// Rank a model filename by the repo's preferred quant fallback order.
pub fn quant_preference_rank(path: &Path) -> usize {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    QUANT_PREFERENCE
        .iter()
        .position(|q| name.contains(q))
        .unwrap_or(QUANT_PREFERENCE.len())
}

/// Derive a display name (tag) from a model file path.
pub fn model_display_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .trim_end_matches(".hfq")
        .trim_end_matches(".tmp")
        .to_string()
}

// ---------------------------------------------------------------------------
// Model card: shared, GPU-free capability + artifact description
//
// Single source of truth for "what can this model do / what ships with it",
// consumed by `hipfire list` (display) and serving admission (gating). The
// arch-feature matrix mirrors MODEL-SUPPORT.md; keep the two in sync.
// ---------------------------------------------------------------------------

/// Tri-state support level for an arch capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSupport {
    /// Implemented and exercised on this arch.
    Full,
    /// Partial / limited / single-token-only path.
    Partial,
    /// Not implemented or not applicable.
    None,
    /// Arch id was unreadable / unrecognized.
    Unknown,
}

impl FeatureSupport {
    /// Compact ASCII mark: `y`/`~`/`-`/`?`.
    pub fn mark(self) -> &'static str {
        match self {
            FeatureSupport::Full => "y",
            FeatureSupport::Partial => "~",
            FeatureSupport::None => "-",
            FeatureSupport::Unknown => "?",
        }
    }
    /// True only for a fully-supported capability — the conservative answer for
    /// admission gating (Partial/Unknown do not pass).
    pub fn is_full(self) -> bool {
        matches!(self, FeatureSupport::Full)
    }
}

/// Per-arch capability summary, keyed by HFQ arch_id. Mirrors the feature
/// matrix in MODEL-SUPPORT.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArchFeatures {
    pub label: &'static str,
    /// Batched (multi-token) prefill.
    pub prefill: FeatureSupport,
    /// DFlash speculative decode.
    pub dflash: FeatureSupport,
    /// MTP speculative decode.
    pub mtp: FeatureSupport,
    /// KV-quant menu, e.g. "full" / "fp32+q8" / "fp32".
    pub kv: &'static str,
    /// Vision / multimodal input.
    pub vision: FeatureSupport,
}

/// Look up the capability summary for an HFQ arch_id. Backed by the generated
/// `ARCH_ROWS` table (source of truth: `docs/model-support.toml`, kept in sync
/// by `hipfire gen-model-support` + the no-gpu-ci `--check` drift gate).
pub fn arch_features(arch_id: u32) -> ArchFeatures {
    for row in model_support_generated::ARCH_ROWS {
        if row.ids.contains(&arch_id) {
            return row.features;
        }
    }
    ArchFeatures {
        label: "unknown",
        prefill: FeatureSupport::Unknown,
        dflash: FeatureSupport::Unknown,
        mtp: FeatureSupport::Unknown,
        kv: "?",
        vision: FeatureSupport::Unknown,
    }
}

/// On-disk companion artifacts bundled with or sitting beside a model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sidecars {
    /// Chat template present in the HFQ metadata header.
    pub template: bool,
    /// MTP draft: `<base>.mtp.hfq` sidecar or bundled (`+mtp`).
    pub mtp: bool,
    /// DFlash draft: `<base>.dflash.hfq` sidecar or bundled (`+dflash`).
    pub dflash: bool,
    /// TriAttention sidecar: `<base>.triattn.hfq`.
    pub triattn: bool,
    /// Hessian / calibration sidecar: `<base>.calib.hfq`.
    pub hessian: bool,
}

/// Quant/format token of a model display name. Scans dotted/dashed groups for
/// the canonical artifact quant token (`mq*`, `oq*`, `bf16`, `f16`, `fp16`) and
/// falls back to the last segment for unknown local artifacts.
pub fn quant_token(display: &str) -> String {
    parse_canonical_model_artifact_name(display)
        .map(|name| name.quant)
        .unwrap_or_else(|| {
            display
                .rsplit(['-', '.'])
                .next()
                .unwrap_or(display)
                .split('+')
                .next()
                .unwrap_or(display)
                .to_ascii_lowercase()
        })
}

/// Detect template + sidecar artifacts for a primary model file (GPU-free:
/// reads only the metadata header and stats sibling files).
pub fn detect_sidecars(path: &Path) -> Sidecars {
    let full = path.to_string_lossy();
    let base = full.strip_suffix(".hfq").unwrap_or(&full);
    let sib = |role: &str| Path::new(&format!("{base}.{role}.hfq")).exists();

    // Bundled features ride the `+feature` filename tokens (mq4+mtp, …).
    let display = model_display_name(path);
    let bundled: Vec<&str> = display.split('+').skip(1).collect();
    let has_bundled = |needle: &str| bundled.iter().any(|b| b.contains(needle));

    // Chat template lives in tokenizer_config.chat_template (HF convention);
    // some artifacts stash it top-level. Present + non-empty in either place.
    let template = read_hfq_metadata(path)
        .ok()
        .and_then(|m| serde_json::from_str::<Value>(&m.metadata_json).ok())
        .map(|v| {
            let nonempty = |t: Option<&Value>| {
                t.and_then(|x| x.as_str())
                    .map(|s| !s.trim().is_empty())
                    .unwrap_or(false)
            };
            nonempty(v.get("chat_template"))
                || nonempty(
                    v.get("tokenizer_config")
                        .and_then(|tc| tc.get("chat_template")),
                )
        })
        .unwrap_or(false);

    Sidecars {
        template,
        mtp: sib("mtp") || has_bundled("mtp"),
        dflash: sib("dflash") || has_bundled("dflash"),
        triattn: sib("triattn"),
        hessian: sib("calib"),
    }
}

/// Full GPU-free description of a local model: identity, quant, arch capability,
/// and bundled/sidecar artifacts. The shared card consumed by `hipfire list`
/// (display) and serving admission (gating).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelCard {
    pub name: String,
    pub quant: String,
    /// HFQ arch_id, or `None` if the metadata header was unreadable.
    pub arch_id: Option<u32>,
    pub features: ArchFeatures,
    pub sidecars: Sidecars,
}

/// Build a [`ModelCard`] for a primary model file.
pub fn model_card(path: &Path) -> ModelCard {
    let name = model_display_name(path);
    let arch_id = read_hfq_metadata(path).ok().map(|m| m.arch_id);
    ModelCard {
        quant: quant_token(&name),
        features: arch_features(arch_id.unwrap_or(u32::MAX)),
        sidecars: detect_sidecars(path),
        arch_id,
        name,
    }
}

pub fn openai_model_list_json<I, P>(models: I) -> Value
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let data: Vec<Value> = models
        .into_iter()
        .map(|path| {
            serde_json::json!({
                "id": model_display_name(path.as_ref()),
                "object": "model",
            })
        })
        .collect();

    serde_json::json!({ "object": "list", "data": data })
}

/// Derive a filesystem-safe identity stem from a model path or tag.
pub fn model_artifact_stem(model: &str) -> String {
    let name = Path::new(model)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(model);
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    sanitized.trim_matches('-').to_string()
}

pub fn model_manifest_entry(role: &str, identifier: &str) -> ModelManifestEntry {
    let path = Path::new(identifier);
    let path_exists = path.exists();
    let file_size = if path_exists {
        fs::metadata(path).ok().map(|m| m.len())
    } else {
        None
    };
    let file_hash = if path_exists {
        model_hash(identifier)
    } else {
        None
    };
    let tag_hash = if path_exists {
        None
    } else {
        Some(format!("tag:{}", stable_hash_bytes(identifier.as_bytes())))
    };
    let (hfq_arch_id, hfq_metadata_hash, quantization_hash, metadata_status, metadata_reason) =
        if path_exists {
            match read_hfq_metadata(path) {
                Ok(meta) => {
                    let parsed: Value =
                        serde_json::from_str(&meta.metadata_json).unwrap_or(Value::Null);
                    (
                        Some(meta.arch_id),
                        Some(stable_hash_bytes(meta.metadata_json.as_bytes())),
                        parsed.get("quantization_hash").cloned(),
                        "pass".to_string(),
                        None,
                    )
                }
                Err(reason) => (None, None, None, "skip".to_string(), Some(reason)),
            }
        } else {
            (
                None,
                None,
                None,
                "skip".to_string(),
                Some("identifier is not a local file path; treating as model tag".to_string()),
            )
        };

    ModelManifestEntry {
        role: role.to_string(),
        identifier: identifier.to_string(),
        path_exists,
        file_size,
        file_hash,
        tag_hash,
        hfq_arch_id,
        hfq_metadata_hash,
        quantization_hash,
        metadata_status,
        metadata_reason,
    }
}

#[allow(clippy::type_complexity)]
fn model_hash_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<(String, u64, u64), Option<String>>> {
    static C: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(String, u64, u64), Option<String>>>,
    > = std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Pull the producer-stamped content hash out of an HFQ container's metadata
/// (`quantization_hash`, an xxh64 over the tensor index + payload). This reads
/// only the metadata span, not the multi-GB payload, so it is effectively free
/// vs `file_hash`. `None` for non-HFQ files or older artifacts lacking the field.
fn hfq_embedded_hash(path: &Path) -> Option<String> {
    let meta = read_hfq_metadata(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&meta.metadata_json).ok()?;
    let qh = v.get("quantization_hash")?;
    let value = qh.get("value")?.as_str()?;
    let algo = qh
        .get("algorithm")
        .and_then(|a| a.as_str())
        .unwrap_or("hash");
    // Namespaced so it never collides with a raw file_hash hex or a tag: id.
    Some(format!("hfq:{algo}:{value}"))
}

pub fn model_hash(model: &str) -> Option<String> {
    let p = Path::new(model);
    if !p.exists() {
        return Some(format!("tag:{}", stable_hash_bytes(model.as_bytes())));
    }
    // Prefer the content hash the producer already stamped into the HFQ metadata
    // (cheap: metadata-span read only). Falls back to a full file hash for
    // non-HFQ files or older artifacts without the field.
    if let Some(h) = hfq_embedded_hash(p) {
        return Some(h);
    }
    // `file_hash` reads the whole (multi-GB) model; callers hash the same model
    // repeatedly within one process (per-battery cache keys, model sweeps), so
    // memoize by (path, mtime, size) — recompute only if the file changes.
    let meta = std::fs::metadata(p).ok();
    let mtime = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let key = (model.to_string(), mtime, size);
    if let Some(cached) = model_hash_cache().lock().unwrap().get(&key) {
        return cached.clone();
    }
    let hash = file_hash(p);
    model_hash_cache().lock().unwrap().insert(key, hash.clone());
    hash
}

pub fn read_hfq_metadata(path: &Path) -> Result<HfqMetadata, String> {
    let mut f = File::open(path).map_err(|e| format!("open model: {e}"))?;
    let mut header = [0u8; 32];
    f.read_exact(&mut header)
        .map_err(|e| format!("read HFQ header: {e}"))?;
    if &header[0..4] != b"HFQM" {
        return Err("not an HFQ container".to_string());
    }
    let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
    let span_len = data_offset.saturating_sub(metadata_offset);
    if span_len == 0 || span_len > 256 * 1024 * 1024 {
        return Err(format!(
            "invalid or too-large metadata span: {metadata_offset}..{data_offset}"
        ));
    }
    f.seek(SeekFrom::Start(metadata_offset as u64))
        .map_err(|e| format!("seek HFQ metadata span: {e}"))?;
    let mut span = vec![0u8; span_len];
    f.read_exact(&mut span)
        .map_err(|e| format!("read HFQ metadata span: {e}"))?;
    let json_end = find_json_object_end(&span)
        .ok_or_else(|| "HFQ metadata JSON object was not terminated".to_string())?;
    let metadata_json = String::from_utf8(span[..json_end].to_vec())
        .map_err(|e| format!("HFQ metadata is not UTF-8: {e}"))?;
    Ok(HfqMetadata {
        arch_id,
        metadata_json,
    })
}

/// Resolve a model identifier to a file path using the standard Hipfire local
/// lookup order.
pub fn find_model_in(arg: &str, models_dir: &Path, aliases_path: Option<&Path>) -> Option<PathBuf> {
    let direct = PathBuf::from(arg);
    if direct.exists() {
        return Some(direct);
    }

    let in_models = models_dir.join(arg);
    if in_models.exists() {
        return Some(in_models);
    }

    let with_ext = models_dir.join(format!("{arg}.hfq"));
    if with_ext.exists() {
        return Some(with_ext);
    }

    if let Some(aliases_path) = aliases_path {
        if let Ok(s) = std::fs::read_to_string(aliases_path) {
            if let Ok(map) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(path_str) = map.get(arg).and_then(|v| v.as_str()) {
                    let p = PathBuf::from(path_str);
                    if p.exists() {
                        return Some(p);
                    }
                }
            }
        }
    }

    let tag_stem = normalize_tag_stem(arg);
    let mut candidates = scan_models_dir(models_dir, &tag_stem);
    candidates.sort_by_key(|p| quant_preference_rank(p));
    candidates.into_iter().next()
}

/// List all non-sidecar `.hfq` files directly under a models directory.
pub fn list_local_models_in(models_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            n.ends_with(".hfq") && !is_role_sidecar_name(&n)
        })
        .collect();
    out.sort();
    out
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelRegistryAsset {
    pub id: String,
    pub file: String,
    pub path: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelArtifactName {
    pub id: String,
    pub model: String,
    pub size: Option<String>,
    pub tags: Vec<String>,
    pub features: Vec<String>,
    pub quant: String,
    pub arch: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LlmModelRegistryEntry {
    pub id: String,
    pub file: String,
    pub path: String,
    pub bytes: u64,
    pub model: String,
    pub size: Option<String>,
    pub tags: Vec<String>,
    pub features: Vec<String>,
    pub quant: String,
    pub arch: Option<String>,
    pub triattn: Vec<ModelRegistryAsset>,
    pub drafts: Vec<ModelRegistryAsset>,
    pub chat_templates: Vec<ModelRegistryAsset>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LlmModelRegistry {
    pub models_dir: String,
    pub triattn_dir: String,
    pub drafts_dir: String,
    pub templates_dir: String,
    pub models: Vec<LlmModelRegistryEntry>,
    pub triattn: Vec<ModelRegistryAsset>,
    pub drafts: Vec<ModelRegistryAsset>,
    pub chat_templates: Vec<ModelRegistryAsset>,
}

impl LlmModelRegistry {
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    pub fn sidecar_count(&self) -> usize {
        self.triattn.len() + self.drafts.len() + self.chat_templates.len()
    }
}

pub fn build_local_llm_registry() -> LlmModelRegistry {
    let hipfire = hipfire_config::hipfire_dir();
    build_llm_registry_in(
        &hipfire_config::models_dir(),
        &hipfire.join("triattn"),
        &hipfire.join("drafts"),
        &hipfire.join("templates"),
    )
}

pub fn build_llm_registry_in(
    models_dir: &Path,
    triattn_dir: &Path,
    drafts_dir: &Path,
    templates_dir: &Path,
) -> LlmModelRegistry {
    let triattn = scan_registry_assets(triattn_dir, is_triattn_file_name);
    let drafts = scan_registry_assets(drafts_dir, is_draft_file_name);
    let chat_templates = scan_registry_assets(templates_dir, is_chat_template_file_name);
    let models = scan_canonical_primary_models(models_dir)
        .into_iter()
        .map(|model| {
            let file = model
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let parsed = parse_canonical_model_artifact_name(&file)
                .expect("scan_canonical_primary_models only returns parseable model artifacts");
            let mut model_triattn =
                matching_assets(&triattn, |asset| triattn_matches_model(&asset.file, &file));
            model_triattn.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.path.cmp(&b.path)));
            model_triattn.dedup_by(|a, b| a.path == b.path);

            let mut model_drafts =
                matching_assets(&drafts, |asset| draft_matches_model(&asset.file, &file));
            model_drafts.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.path.cmp(&b.path)));
            model_drafts.dedup_by(|a, b| a.path == b.path);

            let model_templates = matching_assets(&chat_templates, |asset| {
                template_matches_model(&asset.file, &file)
            });

            LlmModelRegistryEntry {
                id: model_display_name(&model),
                file,
                path: model.to_string_lossy().to_string(),
                bytes: file_len(&model),
                model: parsed.model,
                size: parsed.size,
                tags: parsed.tags,
                features: parsed.features,
                quant: parsed.quant,
                arch: parsed.arch,
                triattn: model_triattn,
                drafts: model_drafts,
                chat_templates: model_templates,
            }
        })
        .collect::<Vec<_>>();

    LlmModelRegistry {
        models_dir: models_dir.to_string_lossy().to_string(),
        triattn_dir: triattn_dir.to_string_lossy().to_string(),
        drafts_dir: drafts_dir.to_string_lossy().to_string(),
        templates_dir: templates_dir.to_string_lossy().to_string(),
        models,
        triattn,
        drafts,
        chat_templates,
    }
}

fn scan_canonical_primary_models(models_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|name| name.to_str())
                    .and_then(parse_canonical_model_artifact_name)
                    .is_some()
        })
        .collect();
    out.sort();
    out
}

fn scan_registry_assets(dir: &Path, predicate: impl Fn(&str) -> bool) -> Vec<ModelRegistryAsset> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let file = path.file_name()?.to_string_lossy().to_string();
            predicate(&file).then(|| registry_asset(path))
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.path.cmp(&b.path)));
    out
}

fn registry_asset(path: PathBuf) -> ModelRegistryAsset {
    let file = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    ModelRegistryAsset {
        id: file
            .trim_end_matches(".jinja")
            .trim_end_matches(".j2")
            .trim_end_matches(".hfq")
            .to_string(),
        file,
        bytes: file_len(&path),
        path: path.to_string_lossy().to_string(),
    }
}

fn matching_assets(
    assets: &[ModelRegistryAsset],
    predicate: impl Fn(&ModelRegistryAsset) -> bool,
) -> Vec<ModelRegistryAsset> {
    assets
        .iter()
        .filter(|asset| predicate(asset))
        .cloned()
        .collect()
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
}

fn is_triattn_file_name(file: &str) -> bool {
    file.to_ascii_lowercase().ends_with(".triattn.hfq")
}

fn is_draft_file_name(file: &str) -> bool {
    let file = file.to_ascii_lowercase();
    file.ends_with(".dflash.hfq")
}

fn is_chat_template_file_name(file: &str) -> bool {
    let file = file.to_ascii_lowercase();
    file.ends_with(".jinja")
}

fn triattn_matches_model(sidecar_file: &str, model_file: &str) -> bool {
    Path::new(model_file)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| sidecar_file == format!("{stem}.triattn.hfq"))
        .unwrap_or(false)
}

fn draft_matches_model(draft_file: &str, model_file: &str) -> bool {
    Path::new(model_file)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| draft_file == format!("{stem}.dflash.hfq"))
        .unwrap_or(false)
}

fn template_matches_model(template_file: &str, model_file: &str) -> bool {
    Path::new(model_file)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| template_file == format!("{stem}.jinja"))
        .unwrap_or(false)
}

pub fn parse_canonical_model_artifact_name(name: &str) -> Option<ModelArtifactName> {
    let id = name
        .strip_suffix(".hfq")
        .or_else(|| name.strip_suffix(".hfq.tmp"))?;
    if id.contains(':') || is_role_sidecar_name(name) {
        return None;
    }

    let mut groups = id.split('.').collect::<Vec<_>>();
    if groups.is_empty() {
        return None;
    }

    let arch = groups
        .last()
        .copied()
        .filter(|group| is_arch_group(group))
        .map(str::to_string);
    if arch.is_some() {
        groups.pop();
    }

    let (quant_start, quant) = find_canonical_quant_group(&groups)?;
    let mut prefix_groups = groups[..quant_start].to_vec();
    let mut features = Vec::new();
    while prefix_groups
        .last()
        .is_some_and(|group| is_feature_group(group))
    {
        features.push(prefix_groups.pop().unwrap().to_string());
    }
    features.reverse();

    let identity = prefix_groups.join(".");
    if identity.is_empty() {
        return None;
    }
    let (model, size, tags) = parse_model_identity(&identity)?;
    Some(ModelArtifactName {
        id: id.to_string(),
        model,
        size,
        tags,
        features,
        quant,
        arch,
    })
}

fn find_canonical_quant_group(groups: &[&str]) -> Option<(usize, String)> {
    if groups.len() >= 2 {
        let start = groups.len() - 2;
        let candidate = groups[start..].join(".");
        if is_canonical_mixed_quant_group(&candidate) {
            return Some((start, candidate.to_ascii_lowercase()));
        }
    }
    if let Some(group) = groups.last() {
        if is_canonical_quant_group(group) {
            return Some((groups.len() - 1, group.to_ascii_lowercase()));
        }
    }
    None
}

fn is_canonical_quant_group(group: &str) -> bool {
    let parts = group.split('-').collect::<Vec<_>>();
    let token = parts.last().copied().unwrap_or(group).to_ascii_lowercase();
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"^(?:m[q]|o[q])\d+(?:\.\d+)?\+{0,2}$|^(?:bf16|f16|fp16)$").unwrap()
    });
    re.is_match(&token)
        && parts[..parts.len().saturating_sub(1)]
            .iter()
            .all(|modifier| is_quant_modifier_segment(modifier))
}

fn is_quant_modifier_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase())
        && segment
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
}

fn is_canonical_mixed_quant_group(group: &str) -> bool {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"^(?:m[q]|o[q])\d+\.\d+\+{0,2}$").unwrap());
    re.is_match(&group.to_ascii_lowercase())
}

fn is_arch_group(group: &str) -> bool {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"^gfx\d{3,4}$").unwrap());
    re.is_match(group)
}

fn is_feature_group(group: &str) -> bool {
    let mut saw_token = false;
    for token in group.split('-') {
        saw_token = true;
        if !matches!(
            token,
            "mtp" | "vl" | "dflash" | "triattn" | "jinja" | "hessian"
        ) {
            return false;
        }
    }
    saw_token
}

fn parse_model_identity(identity: &str) -> Option<(String, Option<String>, Vec<String>)> {
    let parts = identity.split('-').collect::<Vec<_>>();
    let Some(size_idx) = parts.iter().position(|part| is_size_token(part)) else {
        return None;
    };
    let model = parts[..size_idx].join("-");
    if model.is_empty() {
        return None;
    }
    let mut size = parts[size_idx].to_string();
    let mut tag_start = size_idx + 1;
    if parts
        .get(tag_start)
        .is_some_and(|part| is_active_size_token(part))
    {
        size.push('-');
        size.push_str(parts[tag_start]);
        tag_start += 1;
    }
    let tags = parts[tag_start..]
        .iter()
        .filter(|tag| !tag.is_empty())
        .map(|tag| (*tag).to_string())
        .collect::<Vec<_>>();
    Some((model, Some(size), tags))
}

fn is_size_token(token: &str) -> bool {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"^\d+(?:\.\d+)?[kKmMbBtT]$").unwrap());
    re.is_match(token)
}

fn is_active_size_token(token: &str) -> bool {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"^[aAeE]\d+(?:\.\d+)?[kKmMbBtT]$").unwrap());
    re.is_match(token)
}

/// Discover a DFlash draft sidecar next to a target model artifact.
pub fn discover_dflash_draft_for_model(model: &Path) -> Option<PathBuf> {
    if !model.is_file() {
        return None;
    }
    let dir = model.parent().unwrap_or_else(|| Path::new("."));
    let filename = model.file_name().and_then(|name| name.to_str())?;
    dflash_draft_search_dirs(dir)
        .into_iter()
        .flat_map(|search_dir| {
            dflash_draft_candidates(filename)
                .into_iter()
                .map(move |candidate| search_dir.join(candidate))
        })
        .find(|candidate| candidate.is_file())
}

fn dflash_draft_search_dirs(model_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![model_dir.to_path_buf()];
    let hipfire = hipfire_config::hipfire_dir();
    dirs.push(hipfire.join("drafts"));
    dirs.push(hipfire.join("models"));
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("models"));
        dirs.push(cwd.join("../../models"));
    }
    dirs
}

/// Return candidate DFlash draft sidecar filenames for a target model filename.
pub fn dflash_draft_candidates(filename: &str) -> Vec<String> {
    let Some(target) = parse_dflash_target(filename) else {
        return Vec::new();
    };

    let mut quants = vec![target.quant.clone()];
    if target.family == "qwen3" && target.quant == "mq3" {
        quants.push("mq4".to_string());
    } else if target.family == "qwen3" && target.quant == "mq4" {
        quants.push("mq3".to_string());
    } else if target.family == "lfm2" {
        match target.quant.as_str() {
            "oq4" => {
                quants.push("op4".to_string());
                quants.push("mq4".to_string());
            }
            "op4" => {
                quants.push("oq4".to_string());
                quants.push("mq4".to_string());
            }
            "oq4+" => {
                quants.push("op4+".to_string());
                quants.push("mq4".to_string());
            }
            "op4+" => {
                quants.push("oq4+".to_string());
                quants.push("mq4".to_string());
            }
            "oq8" => quants.push("op8".to_string()),
            "op8" => quants.push("oq8".to_string()),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for q in quants {
        out.push(target.format_candidate(&q, "dflash"));
        out.push(target.format_candidate(&q, "draft"));
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DflashDraftTarget {
    family: &'static str,
    stem: String,
    quant: String,
    separator: char,
}

impl DflashDraftTarget {
    fn format_candidate(&self, quant: &str, role: &str) -> String {
        format!("{}{}{}.{}.hfq", self.stem, self.separator, quant, role)
    }
}

fn parse_dflash_target(filename: &str) -> Option<DflashDraftTarget> {
    let stem = filename.strip_suffix(".hfq").unwrap_or(filename);
    parse_qwen_dflash_target(stem).or_else(|| parse_lfm2_dflash_target(stem))
}

fn parse_qwen_dflash_target(stem: &str) -> Option<DflashDraftTarget> {
    let parts: Vec<_> = stem.split('-').collect();
    if parts.len() < 2 || !parts[0].starts_with("qwen3.") {
        return None;
    }
    let quant = parts
        .iter()
        .rev()
        .find(|part| matches!(**part, "mq3" | "mq4" | "mq6" | "mq8"))
        .map(|part| (*part).to_string())?;
    let quant_idx = parts
        .iter()
        .rposition(|part| *part == quant)
        .unwrap_or(parts.len());
    if quant_idx < 2 {
        return None;
    }
    Some(DflashDraftTarget {
        family: "qwen3",
        stem: parts[..quant_idx].join("-"),
        quant,
        separator: '-',
    })
}

fn parse_lfm2_dflash_target(stem: &str) -> Option<DflashDraftTarget> {
    let lower = stem.to_ascii_lowercase();
    if !(lower.starts_with("lfm2") || lower.starts_with("liquidai-lfm2")) {
        return None;
    }
    const QUANTS: &[&str] = &[
        "oq4+", "op4+", "q8f16", "mq3", "mq4", "mq6", "mq8", "oq4", "op4", "oq8", "op8", "q8",
    ];
    for quant in QUANTS {
        for separator in ['-', '.'] {
            let suffix = format!("{separator}{quant}");
            if lower.ends_with(&suffix) {
                let stem_end = stem.len().saturating_sub(suffix.len());
                if stem_end == 0 {
                    return None;
                }
                return Some(DflashDraftTarget {
                    family: "lfm2",
                    stem: stem[..stem_end].to_string(),
                    quant: (*quant).to_string(),
                    separator,
                });
            }
        }
    }
    None
}

fn scan_models_dir(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        maybe_push_model_candidate(&mut out, &path, stem);
        if path.is_dir() {
            if let Ok(sub) = std::fs::read_dir(&path) {
                for se in sub.flatten() {
                    maybe_push_model_candidate(&mut out, &se.path(), stem);
                }
            }
        }
    }
    out
}

fn maybe_push_model_candidate(out: &mut Vec<PathBuf>, path: &Path, stem: &str) {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    if name.ends_with(".hfq") && !is_role_sidecar_name(&name) && name.contains(stem) {
        out.push(path.to_path_buf());
    }
}

fn find_json_object_end(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

/// Shared typed contract for loading a model into a runtime worker.
#[derive(Debug, Deserialize, Serialize)]
pub struct ModelLoadRequest {
    pub model: String,
    pub params: ModelLoadParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Common load parameters shared by daemon protocol clients and future direct
/// library adapters. Daemon-only tuning fields remain in the daemon raw JSON
/// path until they have stable library ownership.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct ModelLoadParams {
    pub max_seq: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_cap: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_adaptive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dflash_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dflash_adaptive_b: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask_sidecar: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask_beta: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask_core_frac: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask_fold_m: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmq_screen: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmq_screen_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_compression: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_threshold: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_keep_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_alpha: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_min_keep: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_sink: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_recent: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_block: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_drafter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_drafter_device: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_profile: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_sparse_threshold: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_k: Option<u32>,
}

impl ModelLoadParams {
    pub fn from_hipfire_config(config: &hipfire_config::HipfireConfig) -> Self {
        let mut params = Self::from_common_config_values(
            config.max_seq,
            &config.kv_cache,
            &config.flash_mode,
            &config.dflash_mode,
            config.cask_sidecar.as_deref(),
        );
        params.kv_adaptive = non_off_value(&config.kv_adaptive);
        params.dflash_adaptive_b = Some(config.dflash_adaptive_b);
        params.mmq_screen = Some(config.mmq_screen != "off");
        params.mmq_screen_threshold = Some(config.mmq_screen_threshold);
        if params.cask_sidecar.is_some() {
            params.cask = Some(config.cask);
            params.cask_budget = Some(config.cask_budget);
            params.cask_beta = Some(config.cask_beta);
            params.cask_core_frac = Some(config.cask_core_frac);
            params.cask_fold_m = Some(config.cask_fold_m);
        }
        if config.prefill_compression != "off" {
            if let Some(drafter) = config.prefill_drafter.as_deref().and_then(non_empty_value) {
                params.prefill_compression = Some(config.prefill_compression.clone());
                params.prefill_threshold = Some(config.prefill_threshold);
                params.prefill_keep_ratio = Some(config.prefill_keep_ratio);
                params.prefill_alpha = Some(config.prefill_alpha);
                params.prefill_min_keep = Some(config.prefill_min_keep);
                params.prefill_sink = Some(config.prefill_sink);
                params.prefill_recent = Some(config.prefill_recent);
                params.prefill_block = Some(config.prefill_block);
                params.prefill_drafter = Some(drafter);
                params.prefill_drafter_device = Some(config.prefill_drafter_device);
                params.prefill_profile = Some(config.prefill_profile);
                params.prefill_sparse_threshold = Some(config.prefill_sparse_threshold);
            }
        }
        params.mtp_mode = non_empty_value(&config.mtp_mode);
        params.mtp_k = Some(config.mtp_k);
        params
    }

    pub fn from_common_config_values(
        max_seq: u32,
        kv_cache: &str,
        flash_mode: &str,
        dflash_mode: &str,
        cask_sidecar: Option<&str>,
    ) -> Self {
        Self {
            max_seq,
            kv_cache: non_auto_value(kv_cache),
            flash_mode: non_auto_value(flash_mode),
            dflash_mode: non_auto_value(dflash_mode),
            cask_sidecar: cask_sidecar.and_then(non_empty_value),
            ..Default::default()
        }
    }
}

fn non_auto_value(value: &str) -> Option<String> {
    (value != "auto").then(|| value.to_string())
}

fn non_off_value(value: &str) -> Option<String> {
    (value != "off").then(|| value.to_string())
}

fn non_empty_value(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}

/// Shared typed contract for the daemon's `loaded` response payload.
#[derive(Debug, Deserialize, Serialize)]
pub struct ModelLoadedResponse {
    pub worker_key_id: String,
    pub arch: Option<String>,
    #[serde(default)]
    pub cache_capable: Option<bool>,
    pub dim: Option<u32>,
    pub layers: Option<u32>,
    pub vocab: Option<u32>,
    pub model_worker: Option<serde_json::Value>,
    #[serde(default)]
    pub response_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{}-{stamp}", std::process::id()))
    }

    #[test]
    fn normalizes_tag_stems_for_fuzzy_lookup() {
        assert_eq!(normalize_tag_stem("qwen3.6:35b-a3b"), "qwen3.6-35b-a3b");
        assert_eq!(normalize_tag_stem("QWEN3.5:9B"), "qwen3.5-9b");
    }

    #[test]
    fn model_worker_key_id_normalizes_feature_flags_and_default_placement() {
        let base = ModelWorkerKey {
            artifact_path: "/models/qwen3.5-9b-mq4.hfq".to_string(),
            artifact_digest: Some("model-digest".to_string()),
            arch_id: "5".to_string(),
            quant_family: "mq4".to_string(),
            state_mode: "attention_kv+deltanet_recurrent".to_string(),
            max_seq_bucket: 8192,
            accelerator_kind: None,
            device_id: None,
            feature_flags: vec!["prefill_batch".to_string(), "qwen35".to_string()],
        };
        let shuffled = ModelWorkerKey {
            feature_flags: vec!["qwen35".to_string(), "prefill_batch".to_string()],
            accelerator_kind: Some("hip".to_string()),
            device_id: Some("0".to_string()),
            ..base.clone()
        };

        assert_eq!(
            normalize_feature_flags(&base.feature_flags),
            vec!["prefill_batch".to_string(), "qwen35".to_string()]
        );
        assert_eq!(model_worker_key_id(&base), model_worker_key_id(&shuffled));
        assert!(same_model_worker_key(&base, &shuffled));
    }

    #[test]
    fn accelerator_inventory_json_reports_empty_contract_state() {
        let json = accelerator_inventory_json(&AcceleratorInventory::not_probed());

        assert_eq!(json["source"], "not_probed");
        assert_eq!(json["device_count"], 0);
        assert_eq!(json["devices"], serde_json::json!([]));
    }

    #[test]
    fn accelerator_inventory_json_reports_hip_device_metadata() {
        let mut device = AcceleratorDeviceInfo::hip(
            "0",
            0,
            Some("gfx1201".to_string()),
            Some(24_000_000_000),
            Some(false),
            Some("HIP 6.4".to_string()),
        );
        device.selected = true;
        let inventory = AcceleratorInventory::from_devices("daemon", vec![device]);
        let json = accelerator_inventory_json(&inventory);

        assert_eq!(json["source"], "daemon");
        assert_eq!(json["device_count"], 1);
        assert_eq!(json["devices"][0]["kind"], "hip");
        assert_eq!(json["devices"][0]["device_id"], "0");
        assert_eq!(json["devices"][0]["ordinal"], 0);
        assert_eq!(json["devices"][0]["arch"], "gfx1201");
        assert_eq!(json["devices"][0]["total_memory_bytes"], 24_000_000_000u64);
        assert_eq!(json["devices"][0]["integrated"], false);
        assert_eq!(json["devices"][0]["device_class"], "discrete");
        assert_eq!(json["devices"][0]["runtime"], "HIP 6.4");
        assert_eq!(json["devices"][0]["available"], true);
        assert_eq!(json["devices"][0]["selected"], true);
    }

    #[test]
    fn arch_id_classification_identifies_qwen35_variants() {
        assert_eq!(
            model_arch_family(ARCH_ID_QWEN35_DENSE),
            ModelArchFamily::Qwen35Dense
        );
        assert_eq!(
            model_arch_family(ARCH_ID_QWEN35_MOE),
            ModelArchFamily::Qwen35Moe
        );
        assert!(is_qwen35_dense_arch_id(ARCH_ID_QWEN35_DENSE));
        assert!(is_qwen35_moe_arch_id(ARCH_ID_QWEN35_MOE));
        assert!(is_qwen35_family_arch_id(ARCH_ID_QWEN35_DENSE));
        assert!(is_qwen35_family_arch_id(ARCH_ID_QWEN35_MOE));
        assert!(!is_qwen35_family_arch_id(ARCH_ID_QWEN2));
        assert_eq!(
            model_arch_family(ARCH_ID_GEMMA3_TEXT),
            ModelArchFamily::Gemma3Text
        );
        assert_eq!(
            model_arch_family(ARCH_ID_GEMMA3_VL),
            ModelArchFamily::Gemma3Vl
        );
        assert_eq!(
            model_arch_family(ARCH_ID_NEMOTRON_H),
            ModelArchFamily::NemotronH
        );
        assert_eq!(model_arch_family(ARCH_ID_MAMBA2), ModelArchFamily::Mamba2);
        for &(arch_id, label) in KNOWN_RUNTIME_ARCH_IDS {
            assert_ne!(
                model_arch_family(arch_id),
                ModelArchFamily::Unknown,
                "{label}({arch_id}) should classify to a concrete ModelArchFamily"
            );
        }
        assert_eq!(model_arch_family(999), ModelArchFamily::Unknown);
    }

    #[test]
    fn role_sidecars_are_not_primary_models() {
        assert!(is_role_sidecar_name("qwen3.5-9b-mq4.mtp.hfq"));
        assert!(is_role_sidecar_name("qwen3.5-9b-mq4.triattn.hfq"));
        assert!(!is_role_sidecar_name("qwen3.5-9b-mq4.hfq"));
    }

    #[test]
    fn canonical_model_artifact_name_breaks_down_identity_features_quant_and_arch() {
        let parsed = parse_canonical_model_artifact_name(
            "Qwen3.5-122B-A10B-it.mtp-vl.lloyd-mq2.gfx1201.hfq",
        )
        .unwrap();

        assert_eq!(parsed.id, "Qwen3.5-122B-A10B-it.mtp-vl.lloyd-mq2.gfx1201");
        assert_eq!(parsed.model, "Qwen3.5");
        assert_eq!(parsed.size.as_deref(), Some("122B-A10B"));
        assert_eq!(parsed.tags, vec!["it"]);
        assert_eq!(parsed.features, vec!["mtp-vl"]);
        assert_eq!(parsed.quant, "lloyd-mq2");
        assert_eq!(parsed.arch.as_deref(), Some("gfx1201"));

        let mixed = parse_canonical_model_artifact_name("Gemma-4-8B.oq4.25++.hfq").unwrap();
        assert_eq!(mixed.model, "Gemma-4");
        assert_eq!(mixed.size.as_deref(), Some("8B"));
        assert_eq!(mixed.quant, "oq4.25++");

        let qwen = parse_canonical_model_artifact_name("Qwen3.5-9B.mq4.hfq").unwrap();
        assert_eq!(qwen.model, "Qwen3.5");
        assert_eq!(qwen.size.as_deref(), Some("9B"));
        assert_eq!(qwen.quant, "mq4");
    }

    #[test]
    fn canonical_model_artifact_name_rejects_old_quant_and_role_sidecar_names() {
        assert!(parse_canonical_model_artifact_name("qwen35-9b-hf4.hfq").is_none());
        assert!(parse_canonical_model_artifact_name("qwen3.5-9B-mq4.hfq").is_none());
        assert!(parse_canonical_model_artifact_name("qwen3.5-9B-op4.hfq").is_none());
        assert!(parse_canonical_model_artifact_name("qwen3.5-9B-q8f16.hfq").is_none());
        assert!(parse_canonical_model_artifact_name("qwen3.5-9B.mq4.mtp.hfq").is_none());
    }

    #[test]
    fn quant_rank_prefers_mq4_before_other_variants() {
        let mut names = [
            PathBuf::from("qwen3.5-9b-q8.hfq"),
            PathBuf::from("qwen3.5-9b-mq6.hfq"),
            PathBuf::from("qwen3.5-9b-mq4.hfq"),
        ];
        names.sort_by_key(|path| quant_preference_rank(path));
        assert_eq!(names[0], PathBuf::from("qwen3.5-9b-mq4.hfq"));
    }

    #[test]
    fn display_name_strips_hfq_and_tmp_suffixes() {
        assert_eq!(
            model_display_name(Path::new("qwen3.5-9b-mq4.hfq")),
            "qwen3.5-9b-mq4"
        );
        assert_eq!(
            model_display_name(Path::new("qwen3.5-9b-mq4.hfq.tmp")),
            "qwen3.5-9b-mq4.hfq"
        );
    }

    #[test]
    fn openai_model_list_json_matches_server_shape() {
        let models = [
            PathBuf::from("/models/qwen3.5-9b-mq4.hfq"),
            PathBuf::from("/models/qwen3.5-9b-q8.hfq"),
        ];

        assert_eq!(
            openai_model_list_json(models.iter()),
            serde_json::json!({
                "object": "list",
                "data": [
                    { "id": "qwen3.5-9b-mq4", "object": "model" },
                    { "id": "qwen3.5-9b-q8", "object": "model" }
                ]
            })
        );
    }

    #[test]
    fn llm_registry_scans_models_sidecars_drafts_and_templates() {
        let root = temp_dir("hipfire-model-registry");
        let models = root.join("models");
        let triattn = root.join("triattn");
        let drafts = root.join("drafts");
        let templates = root.join("templates");
        fs::create_dir_all(&models).unwrap();
        fs::create_dir_all(&triattn).unwrap();
        fs::create_dir_all(&drafts).unwrap();
        fs::create_dir_all(&templates).unwrap();
        fs::write(models.join("Qwen3.5-9B-it.mtp-vl.mq4.hfq"), "model").unwrap();
        fs::write(models.join("qwen35-9b-hf4.hfq"), "old model").unwrap();
        fs::write(models.join("Qwen3.5-9B-it.mtp-vl.mq4.mtp.hfq"), "mtp").unwrap();
        fs::write(
            models.join("Qwen3.5-9B-it.mtp-vl.mq4.hfq.triattn.hfq"),
            "old tri",
        )
        .unwrap();
        fs::write(triattn.join("Qwen3.5-9B-it.mtp-vl.mq4.triattn.hfq"), "tri").unwrap();
        fs::write(drafts.join("Qwen3.5-9B-it.mtp-vl.mq4.dflash.hfq"), "draft").unwrap();
        fs::write(
            drafts.join("Qwen3.5-9B-it.mtp-vl.mq4.draft.hfq"),
            "old draft",
        )
        .unwrap();
        fs::write(templates.join("Qwen3.5-9B-it.mtp-vl.mq4.jinja"), "template").unwrap();
        fs::write(
            templates.join("Qwen3.5-9B-it.mtp-vl.mq4.hfq.j2"),
            "old template",
        )
        .unwrap();

        let registry = build_llm_registry_in(&models, &triattn, &drafts, &templates);

        assert_eq!(registry.model_count(), 1);
        assert_eq!(registry.sidecar_count(), 3);
        let model = &registry.models[0];
        assert_eq!(model.id, "Qwen3.5-9B-it.mtp-vl.mq4");
        assert_eq!(model.model, "Qwen3.5");
        assert_eq!(model.size.as_deref(), Some("9B"));
        assert_eq!(model.tags, vec!["it"]);
        assert_eq!(model.features, vec!["mtp-vl"]);
        assert_eq!(model.quant, "mq4");
        assert_eq!(
            model.triattn[0].file,
            "Qwen3.5-9B-it.mtp-vl.mq4.triattn.hfq"
        );
        assert_eq!(model.drafts[0].file, "Qwen3.5-9B-it.mtp-vl.mq4.dflash.hfq");
        assert_eq!(
            model.chat_templates[0].file,
            "Qwen3.5-9B-it.mtp-vl.mq4.jinja"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn model_artifact_stem_sanitizes_paths_and_tags() {
        assert_eq!(
            model_artifact_stem("/tmp/qwen3.5-9b-awq-mq4.hfq"),
            "qwen3.5-9b-awq-mq4"
        );
        assert_eq!(model_artifact_stem("qwen3.5:9b"), "qwen3");
        assert_eq!(model_artifact_stem("***"), "");
    }

    #[test]
    fn detects_file_formats_by_extension() {
        assert_eq!(
            detect_model_artifact_format(Path::new("model.hfq")),
            ModelArtifactFormat::Hfq
        );
        assert_eq!(
            detect_model_artifact_format(Path::new("model.gguf")),
            ModelArtifactFormat::Gguf
        );
        assert_eq!(
            detect_model_artifact_format(Path::new("model.bin")),
            ModelArtifactFormat::Unknown
        );
    }

    #[test]
    fn open_model_source_policy_routes_files_to_hfq_loader() {
        let path = Path::new("model.hfq");
        let opened = open_model_source_with(
            path,
            |path| Ok::<_, String>(format!("hfq:{}", path.display())),
            |_| Err::<String, _>("unexpected safetensors loader".to_string()),
        )
        .unwrap();
        assert_eq!(opened, "hfq:model.hfq");
    }

    #[test]
    fn open_model_source_policy_routes_config_dirs_to_safetensors_loader() {
        let root = temp_dir("hipfire-model-open-policy");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();

        let opened = open_model_source_with(
            &root,
            |_| Err::<String, _>("unexpected hfq loader".to_string()),
            |path| Ok::<_, String>(format!("safetensors:{}", path.display())),
        )
        .unwrap();

        assert!(opened.starts_with("safetensors:"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn open_model_source_policy_rejects_non_model_dirs() {
        let root = temp_dir("hipfire-model-open-policy-no-config");
        fs::create_dir_all(&root).unwrap();

        let err = open_model_source_with(
            &root,
            |_| Ok::<_, String>("unexpected hfq loader".to_string()),
            |_| Ok::<_, String>("unexpected safetensors loader".to_string()),
        )
        .unwrap_err();

        assert!(err.ends_with("directory has no config.json"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn open_model_source_policy_preserves_loader_error_shape() {
        let file_err = open_model_source_with(
            Path::new("bad.hfq"),
            |_| Err::<String, _>("hfq failed".to_string()),
            |_| Ok::<_, String>("unexpected safetensors loader".to_string()),
        )
        .unwrap_err();
        assert_eq!(file_err, "hfq failed");

        let root = temp_dir("hipfire-model-open-policy-loader-error");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        let dir_err = open_model_source_with(
            &root,
            |_| Ok::<_, String>("unexpected hfq loader".to_string()),
            |_| Err::<String, _>("safetensors failed".to_string()),
        )
        .unwrap_err();
        assert_eq!(dir_err, "safetensors open failed: safetensors failed");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn tokenizer_signature_is_stable_for_vocab_specials_and_sentinel_ids() {
        let vocab = vec!["a".to_string(), "b".to_string(), "<eos>".to_string()];
        let specials = vec![("<eos>".to_string(), 2)];

        assert_eq!(
            tokenizer_signature(&vocab, &specials, 0, 2, None),
            tokenizer_signature(&vocab, &specials, 0, 2, None)
        );
        assert_ne!(
            tokenizer_signature(&vocab, &specials, 0, 2, None),
            tokenizer_signature(&vocab, &specials, 0, 2, Some(3))
        );
    }

    #[test]
    fn tokenizer_signature_preserves_vocab_and_special_token_order() {
        let vocab = vec!["a".to_string(), "b".to_string()];
        let reversed_vocab = vec!["b".to_string(), "a".to_string()];
        let specials = vec![("<a>".to_string(), 10), ("<aa>".to_string(), 11)];
        let reversed_specials = vec![("<aa>".to_string(), 11), ("<a>".to_string(), 10)];

        assert_ne!(
            tokenizer_signature(&vocab, &specials, 0, 1, None),
            tokenizer_signature(&reversed_vocab, &specials, 0, 1, None)
        );
        assert_ne!(
            tokenizer_signature(&vocab, &specials, 0, 1, None),
            tokenizer_signature(&vocab, &reversed_specials, 0, 1, None)
        );
    }

    #[test]
    fn model_worker_id_follows_runtime_shape() {
        assert_eq!(
            ModelWorkerId::from_runtime_parts(6, 1, Some("q8")).value,
            "worker:arch6:pp1:q8"
        );
        assert_eq!(
            ModelWorkerId::from_runtime_parts(5, 2, None).value,
            "worker:arch5:pp2:unknown"
        );
    }

    #[test]
    fn parse_model_worker_id_preserves_daemon_alias_priority() {
        let worker_id = parse_model_worker_id(
            &json!({
                "worker_id": "worker-a",
                "worker_key_id": "worker-b"
            }),
            "__default__",
        );
        assert_eq!(worker_id.value, "worker-a");

        let worker_key_id = parse_model_worker_id(
            &json!({
                "worker_key_id": "worker-b"
            }),
            "__default__",
        );
        assert_eq!(worker_key_id.value, "worker-b");
    }

    #[test]
    fn parse_model_worker_id_falls_back_to_default_worker() {
        let missing = parse_model_worker_id(&json!({}), "__default__");
        assert_eq!(missing.value, "__default__");

        let empty = parse_model_worker_id(
            &json!({
                "worker_id": "",
                "worker_key_id": ""
            }),
            "__default__",
        );
        assert_eq!(empty.value, "__default__");
    }

    #[test]
    fn worker_or_model_identity_requires_non_empty_worker_key_or_model() {
        assert!(has_worker_or_model_identity(&json!({
            "worker_key_id": "worker-a"
        })));
        assert!(has_worker_or_model_identity(&json!({
            "model": "qwen3.5-9b-mq4"
        })));
        assert!(!has_worker_or_model_identity(&json!({
            "worker_key_id": "",
            "model": ""
        })));
        assert!(!has_worker_or_model_identity(&json!({
            "worker_id": "legacy-worker"
        })));
    }

    #[test]
    fn hfq_tokenizer_metadata_prefers_hf_tokenizer_json() {
        let metadata = json!({
            "tokenizer": "{\"model\":{\"vocab\":{},\"merges\":[]}}",
            "gguf_meta": {
                "tokenizer.ggml.tokens": ["<s>", "</s>"]
            }
        });

        let extracted = hfq_tokenizer_metadata(&metadata.to_string()).unwrap();
        assert_eq!(
            extracted,
            Some(HfqTokenizerMetadata::HfJson(
                "{\"model\":{\"vocab\":{},\"merges\":[]}}".to_string()
            ))
        );
    }

    #[test]
    fn hfq_tokenizer_metadata_falls_back_to_gguf_metadata() {
        let gguf_meta = json!({
            "tokenizer.ggml.tokens": ["<s>", "</s>"],
            "tokenizer.ggml.model": "llama"
        });
        let metadata = json!({ "gguf_meta": gguf_meta.clone() });

        let extracted = hfq_tokenizer_metadata(&metadata.to_string()).unwrap();
        assert_eq!(extracted, Some(HfqTokenizerMetadata::GgufMeta(gguf_meta)));
    }

    #[test]
    fn hfq_tokenizer_metadata_reports_missing_payload() {
        let metadata = json!({ "architecture": "qwen3" });

        assert_eq!(hfq_tokenizer_metadata(&metadata.to_string()).unwrap(), None);
    }

    #[test]
    fn hfq_chat_template_reads_tokenizer_config_template() {
        let metadata = json!({
            "tokenizer_config": {
                "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"
            }
        });

        assert_eq!(
            hfq_chat_template(&metadata.to_string()).as_deref(),
            Some("{% for message in messages %}{{ message.content }}{% endfor %}")
        );
        assert_eq!(hfq_chat_template("{\"tokenizer_config\":{}}"), None);
    }

    #[test]
    fn read_optional_tokenizer_json_treats_missing_sidecar_as_absent() {
        let root = temp_dir("hipfire-model-tokenizer-json-missing");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tokenizer.json");

        assert_eq!(read_optional_tokenizer_json(&path), None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_optional_tokenizer_json_returns_existing_sidecar_content() {
        let root = temp_dir("hipfire-model-tokenizer-json-existing");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tokenizer.json");
        fs::write(&path, "{\"model\":{\"vocab\":{}}}").unwrap();

        assert_eq!(
            read_optional_tokenizer_json(&path).as_deref(),
            Some("{\"model\":{\"vocab\":{}}}")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn model_load_request_serializes_common_daemon_wire_shape() {
        let req = ModelLoadRequest {
            model: "model.hfq".to_string(),
            params: ModelLoadParams {
                max_seq: 4096,
                physical_cap: Some(2048),
                kv_cache: Some("asym3".to_string()),
                flash_mode: None,
                dflash_mode: Some("off".to_string()),
                draft: Some("draft.hfq".to_string()),
                cask_sidecar: Some("model.triattn.hfq".to_string()),
                mtp_mode: Some("auto".to_string()),
                mtp_k: Some(3),
                ..Default::default()
            },
            request_id: Some("load-1".to_string()),
        };

        let value = serde_json::to_value(req).unwrap();
        assert_eq!(value["model"], "model.hfq");
        assert_eq!(value["params"]["max_seq"], 4096);
        assert_eq!(value["params"]["physical_cap"], 2048);
        assert_eq!(value["params"]["kv_cache"], "asym3");
        assert_eq!(value["params"]["dflash_mode"], "off");
        assert_eq!(value["params"]["draft"], "draft.hfq");
        assert_eq!(value["params"]["cask_sidecar"], "model.triattn.hfq");
        assert_eq!(value["params"]["mtp_mode"], "auto");
        assert_eq!(value["params"]["mtp_k"], 3);
        assert!(value["params"].get("flash_mode").is_none());
        assert_eq!(value["request_id"], "load-1");
    }

    #[test]
    fn model_load_params_from_common_config_preserves_explicit_dflash_off() {
        let params = ModelLoadParams::from_common_config_values(
            8192,
            "auto",
            "auto",
            "off",
            Some("/models/qwen3.5-27b.triattn.hfq"),
        );

        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache, None);
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode.as_deref(), Some("off"));
        assert_eq!(
            params.cask_sidecar.as_deref(),
            Some("/models/qwen3.5-27b.triattn.hfq")
        );
    }

    #[test]
    fn model_load_params_from_hipfire_config_preserves_load_policy() {
        let config = hipfire_config::HipfireConfig {
            max_seq: 8192,
            kv_cache: "asym3".to_string(),
            kv_adaptive: "balanced".to_string(),
            flash_mode: "auto".to_string(),
            dflash_mode: "off".to_string(),
            dflash_adaptive_b: false,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            cask_sidecar: Some("/models/qwen3.5-27b.triattn.hfq".to_string()),
            cask: true,
            cask_budget: 1024,
            cask_beta: 256,
            cask_core_frac: 0.25,
            cask_fold_m: 4,
            mmq_screen: "off".to_string(),
            prefill_compression: "auto".to_string(),
            prefill_drafter: Some("/models/drafter.hfq".to_string()),
            prefill_drafter_device: 1,
            prefill_profile: true,
            ..Default::default()
        };

        let params = ModelLoadParams::from_hipfire_config(&config);

        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache.as_deref(), Some("asym3"));
        assert_eq!(params.kv_adaptive.as_deref(), Some("balanced"));
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode.as_deref(), Some("off"));
        assert_eq!(params.dflash_adaptive_b, Some(false));
        assert_eq!(params.mtp_mode.as_deref(), Some("auto"));
        assert_eq!(params.mtp_k, Some(3));
        assert_eq!(
            params.cask_sidecar.as_deref(),
            Some("/models/qwen3.5-27b.triattn.hfq")
        );
        assert_eq!(params.cask, Some(true));
        assert_eq!(params.cask_budget, Some(1024));
        assert_eq!(params.cask_beta, Some(256));
        assert_eq!(params.cask_core_frac, Some(0.25));
        assert_eq!(params.cask_fold_m, Some(4));
        assert_eq!(params.mmq_screen, Some(false));
        assert_eq!(params.prefill_compression.as_deref(), Some("auto"));
        assert_eq!(
            params.prefill_drafter.as_deref(),
            Some("/models/drafter.hfq")
        );
        assert_eq!(params.prefill_drafter_device, Some(1));
        assert_eq!(params.prefill_profile, Some(true));
    }

    #[test]
    fn model_load_params_from_common_config_omits_auto_and_empty_sidecar() {
        let params =
            ModelLoadParams::from_common_config_values(4096, "asym3", "auto", "auto", Some(""));

        assert_eq!(params.max_seq, 4096);
        assert_eq!(params.kv_cache.as_deref(), Some("asym3"));
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode, None);
        assert_eq!(params.cask_sidecar, None);
    }

    #[test]
    fn local_model_discovery_uses_direct_alias_and_fuzzy_lookup() {
        let root = temp_dir("hipfire-model-discovery");
        let models = root.join("models");
        let nested = models.join("nested");
        fs::create_dir_all(&nested).unwrap();
        let direct = root.join("direct.hfq");
        let alias_target = root.join("alias-target.hfq");
        let mq6 = models.join("qwen3.5-9b-mq6.hfq");
        let mq4 = nested.join("qwen3.5-9b-mq4.hfq");
        let sidecar = models.join("qwen3.5-9b-mq4.mtp.hfq");
        fs::write(&direct, "").unwrap();
        fs::write(&alias_target, "").unwrap();
        fs::write(&mq6, "").unwrap();
        fs::write(&mq4, "").unwrap();
        fs::write(&sidecar, "").unwrap();
        let aliases = root.join("models.json");
        fs::write(
            &aliases,
            serde_json::to_string(&json!({"alias": alias_target.display().to_string()})).unwrap(),
        )
        .unwrap();

        assert_eq!(
            find_model_in(direct.to_str().unwrap(), &models, Some(&aliases)),
            Some(direct.clone())
        );
        assert_eq!(
            find_model_in("alias", &models, Some(&aliases)),
            Some(alias_target)
        );
        assert_eq!(
            find_model_in("qwen3.5:9b", &models, Some(&aliases)),
            Some(mq4)
        );

        let listed = list_local_models_in(&models);
        assert_eq!(listed, vec![mq6]);
        assert!(!listed.contains(&sidecar));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dflash_draft_discovery_uses_adjacent_qwen_sidecar_names() {
        assert_eq!(
            dflash_draft_candidates("qwen3.5-27b-mq4.hfq"),
            vec![
                "qwen3.5-27b-mq4.dflash.hfq".to_string(),
                "qwen3.5-27b-mq4.draft.hfq".to_string(),
                "qwen3.5-27b-mq3.dflash.hfq".to_string(),
                "qwen3.5-27b-mq3.draft.hfq".to_string(),
            ]
        );
        assert!(dflash_draft_candidates("qwen3.5-35b-a3b-mq4.hfq")
            .contains(&"qwen3.5-35b-a3b-mq4.dflash.hfq".to_string()));
        assert!(dflash_draft_candidates("llama-8b-mq4.hfq").is_empty());

        let root = temp_dir("hipfire-dflash-draft-discovery");
        fs::create_dir_all(&root).unwrap();
        let target = root.join("qwen3.5-27b-mq4.hfq");
        let draft = root.join("qwen3.5-27b-mq4.dflash.hfq");
        fs::write(&target, "target").unwrap();
        fs::write(&draft, "draft").unwrap();

        assert_eq!(discover_dflash_draft_for_model(&target), Some(draft));
        assert_eq!(
            discover_dflash_draft_for_model(&root.join("missing.hfq")),
            None
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dflash_draft_discovery_uses_lfm2_sidecar_names() {
        let oq4 = dflash_draft_candidates("LFM2.5-350M-oq4.hfq");
        assert!(oq4.contains(&"LFM2.5-350M-oq4.dflash.hfq".to_string()));
        assert!(oq4.contains(&"LFM2.5-350M-op4.dflash.hfq".to_string()));
        assert!(oq4.contains(&"LFM2.5-350M-mq4.dflash.hfq".to_string()));

        let op4_plus = dflash_draft_candidates("LFM2.5-1.2B-Thinking.op4+.hfq");
        assert!(op4_plus.contains(&"LFM2.5-1.2B-Thinking.op4+.dflash.hfq".to_string()));
        assert!(op4_plus.contains(&"LFM2.5-1.2B-Thinking.oq4+.dflash.hfq".to_string()));

        let root = temp_dir("hipfire-lfm2-dflash-draft-discovery");
        fs::create_dir_all(&root).unwrap();
        let target = root.join("LFM2.5-350M-oq4.hfq");
        let draft = root.join("LFM2.5-350M-oq4.dflash.hfq");
        fs::write(&target, "target").unwrap();
        fs::write(&draft, "draft").unwrap();

        assert_eq!(discover_dflash_draft_for_model(&target), Some(draft));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn quant_token_recognizes_lfm2_opus_dotted_quant_names() {
        assert_eq!(quant_token("LFM2.5-350M-op4+"), "op4");
        assert_eq!(quant_token("LFM2.5-1.2B-Thinking.op4+"), "op4");
        assert_eq!(quant_token("LFM2.5-1.2B-Thinking.oq8"), "oq8");
    }

    #[test]
    fn model_manifest_entry_extracts_embedded_hfq_quantization_hash() {
        let root = temp_dir("hipfire-model-manifest-hfq");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("candidate.hfq");
        let metadata = json!({
            "architecture": "qwen3",
            "quantization_hash": {
                "algorithm": "xxh64",
                "seed": 0,
                "scope": "hfq_tensor_index_and_payload_v1",
                "value": "0123456789abcdef",
            }
        });
        write_minimal_hfq(&path, &metadata);

        let entry = model_manifest_entry("candidate", path.to_str().unwrap());
        assert_eq!(entry.role, "candidate");
        assert!(entry.path_exists);
        assert_eq!(entry.hfq_arch_id, Some(1));
        assert_eq!(entry.metadata_status, "pass");
        assert_eq!(
            entry
                .quantization_hash
                .as_ref()
                .and_then(|v| v.get("value"))
                .and_then(Value::as_str),
            Some("0123456789abcdef")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn model_manifest_entry_records_tags_without_hfq_metadata() {
        let entry = model_manifest_entry("candidate", "qwen3.5:9b");
        assert!(!entry.path_exists);
        assert!(entry.file_hash.is_none());
        assert!(entry.tag_hash.as_deref().unwrap_or("").starts_with("tag:"));
        assert_eq!(entry.metadata_status, "skip");
        assert!(entry.quantization_hash.is_none());
    }

    fn write_minimal_hfq(path: &Path, metadata: &serde_json::Value) {
        let metadata_bytes = serde_json::to_vec(metadata).unwrap();
        let metadata_offset = 32u64;
        let data_offset = metadata_offset + metadata_bytes.len() as u64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HFQM");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&metadata_offset.to_le_bytes());
        bytes.extend_from_slice(&data_offset.to_le_bytes());
        bytes.extend_from_slice(&metadata_bytes);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn model_loaded_response_deserializes_daemon_wire_shape() {
        let loaded: ModelLoadedResponse = serde_json::from_value(json!({
            "worker_key_id": "worker:arch5:pp1:mq4",
            "arch": "qwen35",
            "cache_capable": true,
            "dim": 4096,
            "layers": 32,
            "vocab": 248320,
            "model_worker": {
                "worker_id": {
                    "value": "worker:arch5:pp1:mq4",
                    "model": "qwen3.5-9b-mq4.hfq",
                    "arch_id": 5
                }
            },
            "response_id": "load-1"
        }))
        .unwrap();

        assert_eq!(loaded.worker_key_id, "worker:arch5:pp1:mq4");
        assert_eq!(loaded.arch.as_deref(), Some("qwen35"));
        assert_eq!(loaded.cache_capable, Some(true));
        assert_eq!(loaded.dim, Some(4096));
        assert_eq!(loaded.layers, Some(32));
        assert_eq!(loaded.vocab, Some(248320));
        assert_eq!(
            loaded.model_worker.unwrap()["worker_id"]["model"],
            "qwen3.5-9b-mq4.hfq"
        );
        assert_eq!(loaded.response_id.as_deref(), Some("load-1"));
    }
}
