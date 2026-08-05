// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed model-registry loading and model identity resolution.
//!
//! The checked-in v1 registry is compiled into every binary as the offline
//! floor. Dynamic loading preserves the established fallback order: fresh
//! cache, network, stale cache, then bundled. A malformed registry is rejected
//! wholesale; model code never consumes a partially validated catalog.

use hipfire_config::{ConfigLayer, ConfigValue};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

pub const REGISTRY_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/warpfront/hipfire/master/registry/v1.json";
pub const REGISTRY_CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
pub const REGISTRY_FETCH_TIMEOUT: Duration = Duration::from_millis(3500);
const BUNDLED_REGISTRY: &str = include_str!("../../../registry/v1.json");

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse registry from {source_name}: {message}")]
    Parse {
        source_name: String,
        message: String,
    },
    #[error("invalid registry from {source_name}: {message}")]
    Invalid {
        source_name: String,
        message: String,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, RegistryError>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sidecar {
    pub file: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDefaults {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecommendedSettings {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

impl RecommendedSettings {
    pub fn config_layer(&self) -> std::result::Result<ConfigLayer, String> {
        let mut layer = ConfigLayer::default();
        if let Some(value) = self.temperature {
            layer
                .set("generation.temperature", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.top_p {
            layer
                .set("generation.top_p", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.top_k {
            let value = i64::try_from(value).map_err(|_| "top_k is too large".to_owned())?;
            layer
                .set("generation.top_k", ConfigValue::Integer(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.min_p {
            layer
                .set("generation.min_p", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.presence_penalty {
            layer
                .set("generation.presence_penalty", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.repeat_penalty {
            layer
                .set("generation.repeat_penalty", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.system_prompt {
            layer
                .set("prompt.system", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        Ok(layer)
    }
}

/// Per-mode sampling profiles for a model, mirroring the model card's
/// documented modes. Each is a full [`RecommendedSettings`] blob. `general`
/// is the thinking-mode default (equals the entry's `recommended_settings`);
/// `coding` is the precise thinking-coding profile; `instruct` is the
/// non-thinking profile. Profiles are entry-level metadata and are selected
/// client-side (e.g. serve_harness `--sampling registry:<profile>`), lowering
/// through the same `generation.*` config keys — no daemon request-JSON change.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingProfiles {
    #[serde(default)]
    pub general: Option<RecommendedSettings>,
    #[serde(default)]
    pub coding: Option<RecommendedSettings>,
    #[serde(default)]
    pub instruct: Option<RecommendedSettings>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub repo: String,
    pub file: String,
    pub size_gb: f64,
    pub min_vram_gb: f64,
    pub desc: String,
    #[serde(default)]
    pub triattn: Option<Sidecar>,
    #[serde(default)]
    pub mtp: Option<Sidecar>,
    #[serde(default)]
    pub dspark: Option<Sidecar>,
    /// Expert-prediction adapter for speculative expert paging (ds4).
    ///
    /// Listing one makes it fetchable alongside the model; it does NOT enable
    /// it. The runtime loads an adapter only when
    /// `HIPFIRE_DEEPSEEK4_EXPERT_ADAPTER` points at a file, so the default
    /// stays off however the registry is populated.
    ///
    /// Deliberately opt-in: as measured this is a REGRESSION (best 0.93x at a
    /// 8 GiB cache, 0.89x at 4 GiB, 0.86x at 32 GiB) because speculation more
    /// than doubles bytes read — 83.1 GiB to 174.4 GiB on the same workload.
    /// It is published for reproduction, not as a win. The lever that does
    /// work for paged ds4 is `HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=auto` (+28.8%).
    #[serde(default)]
    pub expert_adapter: Option<Sidecar>,
    #[serde(default)]
    pub default_tool_format: Option<String>,
    #[serde(default)]
    pub default_kv_mode: Option<String>,
    #[serde(default)]
    pub quant_recipe: Option<String>,
    #[serde(default)]
    pub sampling: Option<SamplingDefaults>,
    #[serde(default)]
    pub recommended_settings: Option<RecommendedSettings>,
    #[serde(default)]
    pub sampling_profiles: Option<SamplingProfiles>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub arch_id: Option<u32>,
    #[serde(default)]
    pub quant: Option<String>,
}

impl ModelEntry {
    /// Resolve a named sampling profile (`general` | `coding` | `instruct`).
    /// `general` falls back to `recommended_settings` (the default profile)
    /// when no explicit profile map is present. Unknown names return `None`.
    pub fn sampling_profile(&self, name: &str) -> Option<&RecommendedSettings> {
        let profiles = self.sampling_profiles.as_ref();
        match name {
            "general" => profiles
                .and_then(|p| p.general.as_ref())
                .or(self.recommended_settings.as_ref()),
            "coding" => profiles.and_then(|p| p.coding.as_ref()),
            "instruct" => profiles.and_then(|p| p.instruct.as_ref()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryV1 {
    pub schema_version: u32,
    pub generated_at: String,
    #[serde(default)]
    pub _comment: Option<String>,
    pub models: BTreeMap<String, ModelEntry>,
    pub aliases: BTreeMap<String, String>,
}

impl RegistryV1 {
    pub fn parse(raw: &str, source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        let mut registry: Self =
            serde_json::from_str(raw).map_err(|error| RegistryError::Parse {
                source_name: source_name.clone(),
                message: error.to_string(),
            })?;
        registry.validate(&source_name)?;
        // An alias is non-authoritative convenience data. Match v1 behavior by
        // dropping dangling redirects instead of rejecting an otherwise valid
        // registry.
        registry
            .aliases
            .retain(|_, target| registry.models.contains_key(target));
        Ok(registry)
    }

    pub fn validate(&self, source_name: &str) -> Result<()> {
        let fail = |message: String| RegistryError::Invalid {
            source_name: source_name.to_owned(),
            message,
        };
        if self.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err(fail(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        if self.generated_at.trim().is_empty() {
            return Err(fail("generated_at is empty".into()));
        }
        if self.models.is_empty() {
            return Err(fail("model catalog is empty".into()));
        }
        for (tag, entry) in &self.models {
            if tag.trim().is_empty() || entry.file.trim().is_empty() {
                return Err(fail(format!("model '{tag}' has an empty tag or file")));
            }
            if !entry.size_gb.is_finite()
                || entry.size_gb < 0.0
                || !entry.min_vram_gb.is_finite()
                || entry.min_vram_gb < 0.0
            {
                return Err(fail(format!("model '{tag}' has invalid size metadata")));
            }
            validate_digest(entry.sha256.as_deref(), tag).map_err(fail)?;
            for sidecar in [&entry.triattn, &entry.mtp, &entry.dspark]
                .into_iter()
                .flatten()
            {
                if sidecar.file.trim().is_empty() {
                    return Err(fail(format!("model '{tag}' has an empty sidecar file")));
                }
                validate_digest(sidecar.sha256.as_deref(), tag).map_err(fail)?;
            }
            if let Some(format) = entry.default_tool_format.as_deref() {
                if !matches!(format, "hermes" | "qwen_xml") {
                    return Err(fail(format!(
                        "model '{tag}' has invalid default_tool_format '{format}'"
                    )));
                }
            }
            if let Some(mode) = entry.default_kv_mode.as_deref() {
                let mut layer = ConfigLayer::default();
                layer
                    .set("memory.kv_cache", ConfigValue::String(mode.to_owned()))
                    .map_err(|error| fail(format!("model '{tag}': {error}")))?;
            }
            if let Some(settings) = &entry.recommended_settings {
                validate_recommendations(tag, settings).map_err(fail)?;
                settings
                    .config_layer()
                    .map_err(|error| fail(format!("model '{tag}': {error}")))?;
            }
            if let Some(profiles) = &entry.sampling_profiles {
                for (name, settings) in [
                    ("general", &profiles.general),
                    ("coding", &profiles.coding),
                    ("instruct", &profiles.instruct),
                ] {
                    if let Some(settings) = settings {
                        validate_recommendations(tag, settings).map_err(|error| {
                            fail(format!("model '{tag}' profile '{name}': {error}"))
                        })?;
                        settings.config_layer().map_err(|error| {
                            fail(format!("model '{tag}' profile '{name}': {error}"))
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn resolve_tag(&self, input: &str) -> String {
        let normalized = input
            .replace("-hfq4", "-hf4")
            .replace("-hfq6", "-hf6")
            .strip_suffix(".hfq")
            .map(|prefix| format!("{prefix}.hf4"))
            .unwrap_or_else(|| input.replace("-hfq4", "-hf4").replace("-hfq6", "-hf6"));
        if self.models.contains_key(&normalized) {
            return normalized;
        }
        if let Some(tag) = self.aliases.get(&normalized) {
            return tag.clone();
        }
        let qwen = format!("qwen3.5:{normalized}");
        if self.models.contains_key(&qwen) {
            return qwen;
        }
        self.models
            .iter()
            .find_map(|(tag, entry)| {
                (entry.file == normalized || entry.file == input).then(|| tag.clone())
            })
            .unwrap_or(normalized)
    }

    pub fn model(&self, input: &str) -> Option<(&str, &ModelEntry)> {
        let tag = self.resolve_tag(input);
        self.models
            .get_key_value(&tag)
            .map(|(tag, entry)| (tag.as_str(), entry))
    }
}

fn validate_digest(digest: Option<&str>, label: &str) -> std::result::Result<(), String> {
    if let Some(digest) = digest {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("'{label}' has an invalid SHA-256"));
        }
    }
    Ok(())
}

fn validate_recommendations(
    tag: &str,
    value: &RecommendedSettings,
) -> std::result::Result<(), String> {
    let ranged = |name: &str, value: Option<f64>, min: f64, max: f64| {
        if value.is_some_and(|value| !value.is_finite() || value < min || value > max) {
            Err(format!("model '{tag}' has invalid {name}"))
        } else {
            Ok(())
        }
    };
    ranged("temperature", value.temperature, 0.0, 2.0)?;
    ranged("top_p", value.top_p, 0.0, 1.0)?;
    ranged("min_p", value.min_p, 0.0, 1.0)?;
    ranged("presence_penalty", value.presence_penalty, 0.0, 2.0)?;
    ranged("repeat_penalty", value.repeat_penalty, 0.5, 2.0)?;
    if value
        .top_k
        .is_some_and(|value| value == 0 || value > 100_000)
    {
        return Err(format!("model '{tag}' has invalid top_k"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegistrySource {
    Cache,
    Network,
    StaleCache,
    Bundled,
}

#[derive(Clone, Debug)]
pub struct LoadedRegistry {
    pub registry: RegistryV1,
    pub source: RegistrySource,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct RegistryPaths {
    pub cache: PathBuf,
}

impl RegistryPaths {
    pub fn discover() -> Self {
        let root = env::var_os("HIPFIRE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".hipfire")))
            .unwrap_or_else(|| PathBuf::from(".hipfire"));
        Self {
            cache: root.join("registry.cache.json"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryCache {
    fetched_at: u64,
    url: String,
    registry: RegistryV1,
}

pub fn bundled() -> Result<RegistryV1> {
    RegistryV1::parse(BUNDLED_REGISTRY, "bundled registry/v1.json")
}

pub fn load(paths: &RegistryPaths) -> LoadedRegistry {
    let mut warnings = Vec::new();
    let bundled = bundled().expect("checked-in registry/v1.json must validate");
    if env::var_os("HIPFIRE_NO_REGISTRY_FETCH").as_deref() == Some("1".as_ref()) {
        return LoadedRegistry {
            registry: bundled,
            source: RegistrySource::Bundled,
            warnings,
        };
    }

    let url = env::var("HIPFIRE_REGISTRY_URL").unwrap_or_else(|_| DEFAULT_REGISTRY_URL.into());
    let now = epoch_millis();
    let cache = read_cache(&paths.cache, &url, &mut warnings);
    if cache
        .as_ref()
        .is_some_and(|cache| cache_is_fresh(cache, now, REGISTRY_CACHE_TTL))
    {
        return LoadedRegistry {
            registry: cache.expect("checked above").registry,
            source: RegistrySource::Cache,
            warnings,
        };
    }

    match fetch_registry(&url) {
        Ok(registry) => {
            let cache_file = RegistryCache {
                fetched_at: now,
                url,
                registry: registry.clone(),
            };
            if let Err(error) = write_cache(&paths.cache, &cache_file) {
                warnings.push(error.to_string());
            }
            LoadedRegistry {
                registry,
                source: RegistrySource::Network,
                warnings,
            }
        }
        Err(error) => {
            warnings.push(error);
            if let Some(cache) = cache {
                LoadedRegistry {
                    registry: cache.registry,
                    source: RegistrySource::StaleCache,
                    warnings,
                }
            } else {
                LoadedRegistry {
                    registry: bundled,
                    source: RegistrySource::Bundled,
                    warnings,
                }
            }
        }
    }
}

fn read_cache(path: &Path, url: &str, warnings: &mut Vec<String>) -> Option<RegistryCache> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            warnings.push(format!("registry cache read failed: {error}"));
            return None;
        }
    };
    let mut cache: RegistryCache = match serde_json::from_str(&raw) {
        Ok(cache) => cache,
        Err(error) => {
            warnings.push(format!("registry cache parse failed: {error}"));
            return None;
        }
    };
    if cache.url != url {
        return None;
    }
    if let Err(error) = cache.registry.validate("registry cache") {
        warnings.push(error.to_string());
        return None;
    }
    cache
        .registry
        .aliases
        .retain(|_, target| cache.registry.models.contains_key(target));
    Some(cache)
}

fn cache_is_fresh(cache: &RegistryCache, now_ms: u64, ttl: Duration) -> bool {
    cache.fetched_at <= now_ms && now_ms.saturating_sub(cache.fetched_at) < ttl.as_millis() as u64
}

fn fetch_registry(url: &str) -> std::result::Result<RegistryV1, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(REGISTRY_FETCH_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|error| format!("registry fetch failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "registry fetch returned HTTP {}",
            response.status()
        ));
    }
    let raw = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("registry response read failed: {error}"))?;
    RegistryV1::parse(&raw, url).map_err(|error| error.to_string())
}

fn write_cache(path: &Path, cache: &RegistryCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| RegistryError::Write {
            path: parent.to_owned(),
            source,
        })?;
    }
    let bytes = serde_json::to_vec(cache).map_err(|error| RegistryError::Parse {
        source_name: "registry cache serialization".into(),
        message: error.to_string(),
    })?;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&tmp, bytes).map_err(|source| RegistryError::Write {
        path: tmp.clone(),
        source,
    })?;
    fs::rename(&tmp, path).map_err(|source| RegistryError::Write {
        path: path.to_owned(),
        source,
    })
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_registry_is_strictly_valid() {
        let registry = bundled().unwrap();
        assert!(registry.models.len() > 20);
        let (tag, model) = registry.model("qwen3.6:35b-a3b-mq4r").unwrap();
        assert_eq!(tag, "qwen3.6:35b-a3b-mq4r");
        assert_eq!(model.quant.as_deref(), Some("mq4r"));
        assert_eq!(
            model.sha256.as_deref(),
            Some("4685c140c46b1a6f31a0fd9053bf09d5faf1d2529d715b84794249b66cde0428")
        );
    }

    #[test]
    fn aliases_and_filenames_resolve_to_canonical_tags() {
        let registry = bundled().unwrap();
        assert_eq!(registry.resolve_tag("qwen3.6"), "qwen3.6:35b-a3b");
        assert_eq!(
            registry.resolve_tag("qwen3.6-35b-a3b.mq4r"),
            "qwen3.6:35b-a3b-mq4r"
        );
    }

    #[test]
    fn recommended_settings_lower_the_full_sampling_contract_to_config() {
        let settings = RecommendedSettings {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(40),
            min_p: Some(0.05),
            presence_penalty: Some(1.5),
            repeat_penalty: Some(1.05),
            system_prompt: Some("You are MiniMax.".into()),
        };
        let layer = settings.config_layer().unwrap();
        assert_eq!(
            layer.get("generation.temperature"),
            Some(&ConfigValue::Float(1.0))
        );
        assert_eq!(
            layer.get("generation.top_p"),
            Some(&ConfigValue::Float(0.95))
        );
        assert_eq!(
            layer.get("generation.top_k"),
            Some(&ConfigValue::Integer(40))
        );
        assert_eq!(
            layer.get("generation.min_p"),
            Some(&ConfigValue::Float(0.05))
        );
        assert_eq!(
            layer.get("generation.presence_penalty"),
            Some(&ConfigValue::Float(1.5))
        );
        assert_eq!(
            layer.get("generation.repeat_penalty"),
            Some(&ConfigValue::Float(1.05))
        );
        assert_eq!(
            layer.get("prompt.system"),
            Some(&ConfigValue::String("You are MiniMax.".into()))
        );
    }

    #[test]
    fn malformed_entry_rejects_the_whole_registry() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"magic4"}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }

    #[test]
    fn dangling_aliases_are_dropped() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"ok":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{"good":"ok","bad":"missing"}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        assert_eq!(registry.aliases.len(), 1);
        assert_eq!(registry.resolve_tag("good"), "ok");
    }

    #[test]
    fn future_cache_timestamps_are_stale() {
        let cache = RegistryCache {
            fetched_at: 101,
            url: "x".into(),
            registry: bundled().unwrap(),
        };
        assert!(!cache_is_fresh(&cache, 100, REGISTRY_CACHE_TTL));
    }

    #[test]
    fn sampling_profiles_resolve_per_mode_with_general_fallback() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"m":{
                "repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x",
                "recommended_settings":{"temperature":1.0,"presence_penalty":1.5},
                "sampling_profiles":{
                    "coding":{"temperature":0.6,"presence_penalty":0.0},
                    "instruct":{"temperature":0.7,"top_p":0.8}
                }
            }},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("m").unwrap();
        assert_eq!(
            entry.sampling_profile("coding").unwrap().temperature,
            Some(0.6)
        );
        assert_eq!(entry.sampling_profile("instruct").unwrap().top_p, Some(0.8));
        // general has no explicit profile → falls back to recommended_settings.
        assert_eq!(
            entry.sampling_profile("general").unwrap().presence_penalty,
            Some(1.5)
        );
        assert!(entry.sampling_profile("nope").is_none());
    }

    #[test]
    fn out_of_range_sampling_profile_rejects_the_whole_registry() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"m":{
                "repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x",
                "sampling_profiles":{"coding":{"temperature":9.0}}
            }},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }
}
