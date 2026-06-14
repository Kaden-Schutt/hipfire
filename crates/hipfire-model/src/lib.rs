// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared model artifact identity helpers and model-source contracts.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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

/// Discover a DFlash draft sidecar next to a target model artifact.
pub fn discover_dflash_draft_for_model(model: &Path) -> Option<PathBuf> {
    if !model.is_file() {
        return None;
    }
    let dir = model.parent().unwrap_or_else(|| Path::new("."));
    let filename = model.file_name().and_then(|name| name.to_str())?;
    dflash_draft_candidates(filename)
        .into_iter()
        .map(|candidate| dir.join(candidate))
        .find(|candidate| candidate.is_file())
}

/// Return candidate DFlash draft sidecar filenames for a target model filename.
pub fn dflash_draft_candidates(filename: &str) -> Vec<String> {
    let Some((family, version, size, quant)) = parse_qwen_dflash_target(filename) else {
        return Vec::new();
    };
    let dotted_family = format!("{family}{version}");
    vec![
        format!("{dotted_family}-{size}-{quant}.dflash.hfq"),
        format!("{dotted_family}-{size}-{quant}.draft.hfq"),
    ]
}

fn parse_qwen_dflash_target(filename: &str) -> Option<(&'static str, String, String, String)> {
    let mut quant_from_ext = None;
    let stem = if let Some(stem) = filename.strip_suffix(".hfq") {
        stem
    } else if let Some(stem) = filename.strip_suffix("-mq4.hfq") {
        quant_from_ext = Some("mq4".to_string());
        stem
    } else if let Some(stem) = filename.strip_suffix("-mq3.hfq") {
        quant_from_ext = Some("mq3".to_string());
        stem
    } else if let Some(stem) = filename.strip_suffix("-mq6.hfq") {
        quant_from_ext = Some("mq6".to_string());
        stem
    } else {
        filename
    };
    let parts: Vec<_> = stem.split('-').collect();
    if parts.len() < 2 || !parts[0].starts_with("qwen3.") {
        return None;
    }
    let version = parts[0].trim_start_matches("qwen").to_string();
    let size = parts[1].to_string();
    let quant = quant_from_ext.or_else(|| {
        parts
            .iter()
            .rev()
            .find(|part| matches!(**part, "mq3" | "mq4" | "mq6" | "mq8"))
            .map(|part| (*part).to_string())
    })?;
    Some(("qwen", version, size, quant))
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
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct ModelLoadParams {
    pub max_seq: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_cap: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dflash_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask_sidecar: Option<String>,
}

/// Shared typed contract for the daemon's `loaded` response payload.
#[derive(Debug, Deserialize, Serialize)]
pub struct ModelLoadedResponse {
    pub worker_key_id: String,
    pub arch: Option<String>,
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
    fn role_sidecars_are_not_primary_models() {
        assert!(is_role_sidecar_name("qwen3.5-9b-mq4.mtp.hfq"));
        assert!(is_role_sidecar_name("qwen3.5-9b-mq4.triattn.hfq"));
        assert!(!is_role_sidecar_name("qwen3.5-9b-mq4.hfq"));
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
        assert!(value["params"].get("flash_mode").is_none());
        assert_eq!(value["request_id"], "load-1");
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
                "qwen3.5-27b-mq4.draft.hfq".to_string()
            ]
        );
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
    fn model_loaded_response_deserializes_daemon_wire_shape() {
        let loaded: ModelLoadedResponse = serde_json::from_value(json!({
            "worker_key_id": "worker:arch5:pp1:mq4",
            "arch": "qwen35",
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
