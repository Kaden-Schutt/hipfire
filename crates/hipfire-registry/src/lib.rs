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
    /// Parent-model reasoning effort. This is prompt semantics, independent of
    /// any explicit token cap selected by `thinking_budget`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Optional named cap policy for reasoning tokens
    /// (`off|low|med|high|xhigh|max|uncapped`).
    #[serde(default)]
    pub thinking_budget: Option<String>,
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
        if let Some(value) = &self.reasoning_effort {
            layer
                .set("reasoning.effort", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.thinking_budget {
            layer
                .set("reasoning.budget", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        Ok(layer)
    }
}

/// Per-mode sampling profiles for a model, mirroring the model card's
/// documented modes. Each is a full [`RecommendedSettings`] blob, including
/// optional reasoning effort and explicit budget policy. `general` is the thinking-mode
/// default (equals the entry's `recommended_settings`); `coding` is the precise
/// thinking-coding profile; `instruct` is the non-thinking profile. Profiles
/// are entry-level metadata and are selected client-side (e.g. serve_harness
/// `--sampling registry:<profile>`), lowering through the typed config/request
/// path rather than architecture-specific environment variables.
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

    /// Lower this entry's load/sampling defaults into a sparse config layer.
    ///
    /// Starts from [`RecommendedSettings::config_layer`] when present (sampling
    /// and reasoning only), then overlays the entry-level load defaults onto
    /// their canonical keys. Callers merge this under `RegistryModel` precedence
    /// so global/model/one-shot user config still wins.
    pub fn config_layer(&self) -> std::result::Result<ConfigLayer, String> {
        let mut layer = match &self.recommended_settings {
            Some(settings) => settings.config_layer()?,
            None => ConfigLayer::default(),
        };
        if let Some(mode) = &self.default_kv_mode {
            layer
                .set("memory.kv_cache", ConfigValue::String(mode.clone()))
                .map_err(|error| error.to_string())?;
        }
        Ok(layer)
    }
}

/// Tag-aware registry config layer. Starts from [`ModelEntry::config_layer`]
/// then applies automatic load policy by canonical tag. This is the single
/// canonical place for tag policy so validation and CLI stay in sync.
///
/// Policy (static; no registry/v1 wire fields):
/// - exact family before ':' in {qwen3.5,qwen3.6,qwen3.8} and tag not containing
///   `draft`/`dflash` => `memory.kv_backend=vmm`, `memory.max_seq=262144`,
///   `generation.max_tokens=81920`
/// - exact family before ':' in {deepseek-v4-flash,deepseek-v4-flash-preview}
///   and tag not containing `draft`/`dflash` => `memory.kv_backend=vmm`,
///   `memory.max_seq=1048576`, `generation.max_tokens=393216`
/// - exact tag `muse-glimmer` or `muse-glimmer:fast` => `memory.kv_backend=vmm`,
///   `memory.max_seq=131072` (no invented `generation.max_tokens`)
/// - original `qwen3:*`, draft/dflash sidecars, and `muse-glimmer:draft` receive
///   no automatic policy
///
/// Explicit global/model/one-shot user config remains higher precedence than
/// this registry layer.
pub fn config_layer_for_tag(
    tag: &str,
    entry: &ModelEntry,
) -> std::result::Result<ConfigLayer, String> {
    let mut layer = entry.config_layer()?;
    let family = tag.split(':').next().unwrap_or(tag);
    let is_sidecar = tag.contains("draft") || tag.contains("dflash");
    let is_qwen_tag_policy = matches!(family, "qwen3.5" | "qwen3.6" | "qwen3.8") && !is_sidecar;
    if is_qwen_tag_policy {
        layer
            .set("memory.kv_backend", ConfigValue::String("vmm".into()))
            .map_err(|error| error.to_string())?;
        layer
            .set("memory.max_seq", ConfigValue::Integer(262144))
            .map_err(|error| error.to_string())?;
        layer
            .set("generation.max_tokens", ConfigValue::Integer(81920))
            .map_err(|error| error.to_string())?;
    }
    let is_deepseek_tag_policy =
        matches!(family, "deepseek-v4-flash" | "deepseek-v4-flash-preview") && !is_sidecar;
    if is_deepseek_tag_policy {
        layer
            .set("memory.kv_backend", ConfigValue::String("vmm".into()))
            .map_err(|error| error.to_string())?;
        layer
            .set("memory.max_seq", ConfigValue::Integer(1048576))
            .map_err(|error| error.to_string())?;
        layer
            .set("generation.max_tokens", ConfigValue::Integer(393216))
            .map_err(|error| error.to_string())?;
    }
    if matches!(tag, "muse-glimmer" | "muse-glimmer:fast") {
        layer
            .set("memory.kv_backend", ConfigValue::String("vmm".into()))
            .map_err(|error| error.to_string())?;
        layer
            .set("memory.max_seq", ConfigValue::Integer(131072))
            .map_err(|error| error.to_string())?;
    }
    Ok(layer)
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
            if let Some(settings) = &entry.recommended_settings {
                validate_recommendations(tag, settings).map_err(fail)?;
            }
            config_layer_for_tag(tag, entry)
                .map_err(|error| fail(format!("model '{tag}': {error}")))?;
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

    /// Pins Muse Glimmer's sampling contract to its model card's "Best Practices"
    /// section (`meta-models/Muse-Glimmer-30B`): temperature 1.0, top_p 0.95, top_k 64.
    ///
    /// `top_k` is the one that matters. Before this entry existed the tag resolved to
    /// nothing and callers silently fell back to `recipe(general)`, which uses the Qwen
    /// family's `top_k = 20` — a different model's sampling contract applied to Glimmer
    /// with only a warning on stderr.
    #[test]
    fn bundled_muse_glimmer_matches_the_model_card_sampling_contract() {
        let registry = bundled().unwrap();
        // Only one Glimmer size exists, so the canonical tag carries no size
        // suffix. `muse-glimmer:30b` stays a back-compat alias because
        // scripts/serve_harness.py infers it from the artifact filename.
        let (tag, model) = registry.model("muse-glimmer").unwrap();
        assert_eq!(tag, "muse-glimmer");
        assert_eq!(model.arch_id, Some(14));
        assert_eq!(model.file, "muse-glimmer-30b.mq4");
        for alias in [
            "muse-glimmer:30b",
            "muse-glimmer:latest",
            "muse-glimmer:quality",
        ] {
            let (resolved, _) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(resolved, "muse-glimmer", "{alias} resolves to the trunk");
        }

        // The speed SKU is a distinct entry on the MQ4-attention artifact, not an
        // alias of the trunk.
        let (fast_tag, fast) = registry.model("muse-glimmer:fast").unwrap();
        assert_eq!(fast_tag, "muse-glimmer:fast");
        assert_eq!(fast.file, "muse-glimmer-30b.mq4r");
        assert_eq!(fast.arch_id, Some(14));
        assert_ne!(
            fast.file, model.file,
            "the two SKUs are different artifacts"
        );

        // The drafter is arch 23 and pairs with either SKU.
        let (draft_tag, draft) = registry.model("muse-glimmer:draft").unwrap();
        assert_eq!(draft_tag, "muse-glimmer:draft");
        assert_eq!(draft.arch_id, Some(23));

        let settings = model
            .recommended_settings
            .as_ref()
            .expect("muse-glimmer must carry the card's recommended_settings");
        assert_eq!(settings.temperature, Some(1.0));
        assert_eq!(settings.top_p, Some(0.95));
        assert_eq!(settings.top_k, Some(64));

        // `general` must resolve to the same contract, so `--sampling registry:general`
        // and the bare default cannot diverge.
        let general = model
            .sampling_profile("general")
            .expect("general profile resolves");
        assert_eq!(general.top_k, Some(64));

        // The card specifies no separate coding/instruct sampling, so those are
        // deliberately absent rather than invented.
        assert!(model.sampling_profile("coding").is_none());
        assert!(model.sampling_profile("instruct").is_none());
    }

    /// Pins Qwen3.8-27B's release contract: aliases, artifact identity, arch/quant,
    /// default KV, and the upstream sampling card (including reasoning effort/budget).
    #[test]
    fn bundled_qwen38_matches_the_release_contract() {
        let registry = bundled().unwrap();
        let (tag, model) = registry.model("qwen3.8:27b").unwrap();
        assert_eq!(tag, "qwen3.8:27b");
        assert_eq!(model.file, "qwen3.8-27b.mq4");
        assert_eq!(model.arch_id, Some(5));
        assert_eq!(model.quant.as_deref(), Some("mq4"));
        assert_eq!(model.default_kv_mode.as_deref(), Some("q8"));

        for alias in ["qwen3.8", "qwen3.8:latest"] {
            let (resolved, _) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(
                resolved, "qwen3.8:27b",
                "{alias} resolves to the canonical tag"
            );
        }

        let settings = model
            .recommended_settings
            .as_ref()
            .expect("qwen3.8:27b must carry recommended_settings");
        assert_eq!(settings.temperature, Some(1.0));
        assert_eq!(settings.top_p, Some(0.95));
        assert_eq!(settings.top_k, Some(20));
        assert_eq!(settings.min_p, Some(0.0));
        assert_eq!(settings.presence_penalty, Some(0.0));
        assert_eq!(settings.repeat_penalty, Some(1.0));
        assert_eq!(settings.reasoning_effort.as_deref(), Some("xhigh"));
        assert_eq!(settings.thinking_budget.as_deref(), Some("uncapped"));

        // `general` must resolve to the same contract, so `--sampling registry:general`
        // and the bare default cannot diverge.
        let general = model
            .sampling_profile("general")
            .expect("general profile resolves");
        assert_eq!(general.temperature, Some(1.0));
        assert_eq!(general.top_p, Some(0.95));
        assert_eq!(general.top_k, Some(20));
        assert_eq!(general.min_p, Some(0.0));
        assert_eq!(general.presence_penalty, Some(0.0));
        assert_eq!(general.repeat_penalty, Some(1.0));
        assert_eq!(general.reasoning_effort.as_deref(), Some("xhigh"));
        assert_eq!(general.thinking_budget.as_deref(), Some("uncapped"));

        // Tag policy provides VMM + 262K + 81920 for Qwen3.8 canonical tags.
        let layer =
            config_layer_for_tag(tag, model).expect("qwen3.8:27b tag policy lowers cleanly");
        assert_eq!(
            layer.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert_eq!(
            layer.get("memory.kv_backend"),
            Some(&ConfigValue::String("vmm".into()))
        );
        assert_eq!(
            layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(262144))
        );
        assert_eq!(
            layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("xhigh".into()))
        );
        // Original Qwen3 family (without .5/.6/.8) receives no automatic policy.
        let qwen3_raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"}},
            "aliases":{}
        }"#;
        let qwen3_registry = RegistryV1::parse(qwen3_raw, "test").unwrap();
        let (q3_tag, q3_entry) = qwen3_registry.model("qwen3:8b").unwrap();
        let q3_layer = config_layer_for_tag(q3_tag, q3_entry).unwrap();
        assert_eq!(
            q3_layer.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert!(q3_layer.get("memory.kv_backend").is_none());
        assert!(q3_layer.get("memory.max_seq").is_none());
        assert!(q3_layer.get("generation.max_tokens").is_none());
    }

    #[test]
    fn bundled_0731_mq2r_is_default_and_mq2lloyd_stays_addressable() {
        let registry = bundled().unwrap();
        // MQ2R became the default on 2026-08-14 on the DeepSeek V4 contributor's
        // recommendation. The bare tag and the explicit `:mq2r` name must serve
        // the SAME artifact, while `:mq2lloyd` stays a distinct, still-pullable
        // identity — a silent collapse of the two would ship 86 GB under an 82 GB
        // name or vice versa.
        let (_, default_sku) = registry.model("deepseek-v4-flash").unwrap();
        let (_, mq2r) = registry.model("deepseek-v4-flash:mq2r").unwrap();
        let (_, mq2lloyd) = registry.model("deepseek-v4-flash:mq2lloyd").unwrap();

        assert_eq!(default_sku.file, "deepseek-v4-flash-0731.mq2r");
        assert_eq!(mq2r.file, "deepseek-v4-flash-0731.mq2r");
        assert_eq!(mq2lloyd.file, "deepseek-v4-flash-0731.mq2lloyd");
        assert_ne!(mq2r.sha256, mq2lloyd.sha256);

        assert_eq!(default_sku.default_kv_mode.as_deref(), Some("f32"));
        assert_eq!(mq2lloyd.default_kv_mode.as_deref(), Some("f32"));
        assert_eq!(
            mq2r.sha256.as_deref(),
            Some("cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce")
        );
        assert_eq!(
            mq2lloyd.sha256.as_deref(),
            Some("521c9687a3f5c12fb3d89bde4a3ed202698b95ae5a102d0b5ba7f3abb87982d0")
        );
        assert_eq!(
            mq2r.quant_recipe.as_deref(),
            Some("deepseek4-mq2r-e8-p3-v1")
        );

        assert_eq!(
            mq2r.dspark
                .as_ref()
                .and_then(|sidecar| sidecar.sha256.as_deref()),
            Some("bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e")
        );
        assert!(registry.model("deepseek4:preview-mq2r").is_none());
    }

    #[test]
    fn aliases_and_filenames_resolve_to_canonical_tags() {
        let registry = bundled().unwrap();
        assert_eq!(registry.resolve_tag("qwen3.6"), "qwen3.6:35b-a3b");
        assert_eq!(
            registry.resolve_tag("qwen3.6-35b-a3b.mq4r"),
            "qwen3.6:35b-a3b-mq4r"
        );
        assert_eq!(registry.resolve_tag("deepseek4"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("deepseek4:0731"), "deepseek-v4-flash");
        // `:mq2r` names now land on the default tag, since MQ2R *is* the default.
        assert_eq!(
            registry.resolve_tag("deepseek4:0731-mq2r"),
            "deepseek-v4-flash"
        );
        assert_eq!(
            registry.resolve_tag("deepseek-v4-flash-0731.mq2r"),
            "deepseek-v4-flash"
        );
        // The superseded artifact must still be reachable by its own filename.
        assert_eq!(
            registry.resolve_tag("deepseek-v4-flash-0731.mq2lloyd"),
            "deepseek-v4-flash:mq2lloyd"
        );
        assert_eq!(
            registry.resolve_tag("deepseek4:preview"),
            "deepseek-v4-flash-preview"
        );
        assert!(registry.model("deepseek4:preview-mq2r").is_none());

        // `ds4` is the short name used throughout the tree (ds4-adapter-r128.bin,
        // ds4_length_sweep.sh, hipfire-arch-deepseek4), so it resolves too, and
        // mirrors the full `deepseek4:*` surface.
        assert_eq!(registry.resolve_tag("ds4"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("ds4:0731"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("ds4:mq2r"), "deepseek-v4-flash");
        assert_eq!(
            registry.resolve_tag("ds4:preview"),
            "deepseek-v4-flash-preview"
        );
        // Every SKU reachable by a short `:mq2r` name must also be reachable by
        // the matching `:mq2lloyd` one — otherwise demoting MQ2-Lloyd from the
        // default silently strands it behind long-form names only.
        for name in [
            "deepseek4:mq2lloyd",
            "deepseek-v4:mq2lloyd",
            "deepseek4:0731-mq2lloyd",
            "ds4:mq2lloyd",
            "ds4:0731-mq2lloyd",
        ] {
            assert_eq!(
                registry.resolve_tag(name),
                "deepseek-v4-flash:mq2lloyd",
                "{name} must resolve to the MQ2-Lloyd tag"
            );
        }
        let (_, default_sku) = registry.model("deepseek4:0731").unwrap();
        let settings = default_sku.recommended_settings.as_ref().unwrap();
        assert_eq!(settings.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(settings.thinking_budget.as_deref(), Some("uncapped"));
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
            reasoning_effort: Some("high".into()),
            thinking_budget: Some("xhigh".into()),
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
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("high".into()))
        );
        assert_eq!(
            layer.get("reasoning.budget"),
            Some(&ConfigValue::String("xhigh".into()))
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
    fn tag_policy_pins_qwen_deepseek_and_glimmer_targets() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{
                "qwen3.5:4b":{"repo":"x","file":"qwen3.5-4b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.6:35b-a3b":{"repo":"x","file":"qwen3.6-35b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.5:27b-draft":{"repo":"x","file":"qwen35-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.5:27b-dflash":{"repo":"x","file":"qwen35-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash":{"repo":"x","file":"deepseek-v4-flash-0731.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"deepseek-v4-flash-0731.mq2lloyd","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash-preview":{"repo":"x","file":"deepseek-v4-flash-preview.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:draft":{"repo":"x","file":"deepseek-v4-flash-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:fast":{"repo":"x","file":"muse-glimmer-30b.mq4r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:draft":{"repo":"x","file":"muse-glimmer-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "other:model":{"repo":"x","file":"other.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}
            },
            "aliases":{
                "deepseek4":"deepseek-v4-flash",
                "ds4":"deepseek-v4-flash",
                "deepseek4:preview":"deepseek-v4-flash-preview",
                "muse-glimmer:quality":"muse-glimmer"
            }
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        // Exact Qwen families get VMM + 262144 + 81920
        for tag in ["qwen3.5:4b", "qwen3.6:35b-a3b", "qwen3.8:27b"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert_eq!(
                layer.get("memory.kv_backend"),
                Some(&ConfigValue::String("vmm".into())),
                "{tag} should get vmm"
            );
            assert_eq!(
                layer.get("memory.max_seq"),
                Some(&ConfigValue::Integer(262144)),
                "{tag} should get 262144"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(81920)),
                "{tag} should get 81920"
            );
        }

        // Original Qwen3 (no dot) receives none — stays contiguous
        let (_, entry) = registry.model("qwen3:8b").unwrap();
        let layer = config_layer_for_tag("qwen3:8b", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Draft/dflash sidecars do not get the Qwen policy
        for tag in ["qwen3.5:27b-draft", "qwen3.5:27b-dflash"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} sidecar must not get vmm"
            );
            assert!(layer.get("memory.max_seq").is_none());
            assert!(layer.get("generation.max_tokens").is_none());
        }

        // DeepSeek official / MQ2Lloyd / preview targets get VMM + 1M + 384Ki
        for tag in [
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
        ] {
            let (resolved, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(resolved, entry).unwrap();
            assert_eq!(
                layer.get("memory.kv_backend"),
                Some(&ConfigValue::String("vmm".into())),
                "{tag} should get vmm"
            );
            assert_eq!(
                layer.get("memory.max_seq"),
                Some(&ConfigValue::Integer(1048576)),
                "{tag} should get 1048576"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(393216)),
                "{tag} should get 393216"
            );
        }
        // Aliases resolve to canonical families that carry the same policy.
        for alias in ["deepseek4", "ds4", "deepseek4:preview"] {
            let (resolved, entry) = registry.model(alias).unwrap();
            let layer = config_layer_for_tag(resolved, entry).unwrap();
            assert_eq!(
                layer.get("memory.kv_backend"),
                Some(&ConfigValue::String("vmm".into())),
                "{alias}->{resolved} should get vmm"
            );
            assert_eq!(
                layer.get("memory.max_seq"),
                Some(&ConfigValue::Integer(1048576)),
                "{alias}->{resolved} should get 1048576"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(393216)),
                "{alias}->{resolved} should get 393216"
            );
        }
        // DeepSeek draft sidecar receives none
        let (_, entry) = registry.model("deepseek-v4-flash:draft").unwrap();
        let layer = config_layer_for_tag("deepseek-v4-flash:draft", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Muse Glimmer quality and fast targets get VMM + native 131072, no invented max_tokens
        for tag in ["muse-glimmer", "muse-glimmer:fast"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert_eq!(
                layer.get("memory.kv_backend"),
                Some(&ConfigValue::String("vmm".into())),
                "{tag} should get vmm"
            );
            assert_eq!(
                layer.get("memory.max_seq"),
                Some(&ConfigValue::Integer(131072)),
                "{tag} should get 131072"
            );
            assert!(
                layer.get("generation.max_tokens").is_none(),
                "{tag} must not get max_tokens"
            );
        }
        // quality alias lands on trunk and inherits the same policy.
        let (resolved, entry) = registry.model("muse-glimmer:quality").unwrap();
        assert_eq!(resolved, "muse-glimmer");
        let layer = config_layer_for_tag(resolved, entry).unwrap();
        assert_eq!(
            layer.get("memory.kv_backend"),
            Some(&ConfigValue::String("vmm".into()))
        );
        assert_eq!(
            layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(131072))
        );
        assert!(layer.get("generation.max_tokens").is_none());

        // Muse Glimmer draft receives none
        let (_, entry) = registry.model("muse-glimmer:draft").unwrap();
        let layer = config_layer_for_tag("muse-glimmer:draft", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Absent policy: unrelated models get no automatic policy
        let (_, entry) = registry.model("other:model").unwrap();
        let layer = config_layer_for_tag("other:model", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());
    }

    #[test]
    fn tag_policy_does_not_expand_wire_schema() {
        // Old v1 JSON without the invented fields must still parse (deny_unknown_fields).
        let raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"ok":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("ok").unwrap();
        let layer = entry.config_layer().unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());
        // Tag policy also absent for non-target.
        let tagged = config_layer_for_tag("ok", entry).unwrap();
        assert!(tagged.get("memory.kv_backend").is_none());

        // Invented wire fields must be rejected (no schema expansion).
        let bad = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_backend":"vmm"}},
            "aliases":{}
        }"#;
        assert!(
            RegistryV1::parse(bad, "test").is_err(),
            "default_kv_backend must be rejected as unknown field"
        );
        let bad2 = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_max_seq":262144}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(bad2, "test").is_err());
        let bad3 = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_max_tokens":81920}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(bad3, "test").is_err());
    }

    #[test]
    fn tag_policy_explicit_override_wins_over_registry() {
        // Registry tag policy is a low-precedence layer; global/model/one-shot user config wins.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let registry_layer = config_layer_for_tag(tag, entry).unwrap();
        assert_eq!(
            registry_layer.get("memory.kv_backend"),
            Some(&ConfigValue::String("vmm".into()))
        );

        // Simulate user global override to contiguous + different max_seq/max_tokens.
        let mut user_layer = ConfigLayer::default();
        user_layer
            .set(
                "memory.kv_backend",
                ConfigValue::String("contiguous".into()),
            )
            .unwrap();
        user_layer
            .set("memory.max_seq", ConfigValue::Integer(32768))
            .unwrap();
        user_layer
            .set("generation.max_tokens", ConfigValue::Integer(1024))
            .unwrap();

        let resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test.toml"),
                },
                layer: user_layer,
            },
        ])
        .unwrap();
        // User wins.
        assert_eq!(
            resolved.get("memory.kv_backend").map(|value| &value.value),
            Some(&ConfigValue::String("contiguous".into()))
        );
        assert_eq!(
            resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(32768))
        );
        assert_eq!(
            resolved
                .get("generation.max_tokens")
                .map(|value| &value.value),
            Some(&ConfigValue::Integer(1024))
        );

        // Glimmer target override likewise wins (backend + max_seq).
        let raw2 = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry2 = RegistryV1::parse(raw2, "test").unwrap();
        let (g_tag, g_entry) = registry2.model("muse-glimmer").unwrap();
        let g_registry_layer = config_layer_for_tag(g_tag, g_entry).unwrap();
        assert_eq!(
            g_registry_layer.get("memory.kv_backend"),
            Some(&ConfigValue::String("vmm".into()))
        );
        assert_eq!(
            g_registry_layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(131072))
        );
        assert!(g_registry_layer.get("generation.max_tokens").is_none());
        let mut g_user = ConfigLayer::default();
        g_user
            .set(
                "memory.kv_backend",
                ConfigValue::String("contiguous".into()),
            )
            .unwrap();
        g_user
            .set("memory.max_seq", ConfigValue::Integer(8192))
            .unwrap();
        let g_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: g_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: g_registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test2.toml"),
                },
                layer: g_user,
            },
        ])
        .unwrap();
        assert_eq!(
            g_resolved
                .get("memory.kv_backend")
                .map(|value| &value.value),
            Some(&ConfigValue::String("contiguous".into()))
        );
        assert_eq!(
            g_resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(8192))
        );

        // DeepSeek target override wins over 1M/384Ki policy.
        let raw3 = r#"{
            "schema_version":1,
            "generated_at":"now",
            "models":{"deepseek-v4-flash":{"repo":"x","file":"ds4.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry3 = RegistryV1::parse(raw3, "test").unwrap();
        let (d_tag, d_entry) = registry3.model("deepseek-v4-flash").unwrap();
        let d_registry_layer = config_layer_for_tag(d_tag, d_entry).unwrap();
        assert_eq!(
            d_registry_layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(1048576))
        );
        assert_eq!(
            d_registry_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(393216))
        );
        let mut d_user = ConfigLayer::default();
        d_user
            .set("memory.max_seq", ConfigValue::Integer(65536))
            .unwrap();
        d_user
            .set("generation.max_tokens", ConfigValue::Integer(2048))
            .unwrap();
        let d_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: d_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: d_registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test3.toml"),
                },
                layer: d_user,
            },
        ])
        .unwrap();
        assert_eq!(
            d_resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(65536))
        );
        assert_eq!(
            d_resolved
                .get("generation.max_tokens")
                .map(|value| &value.value),
            Some(&ConfigValue::Integer(2048))
        );
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
