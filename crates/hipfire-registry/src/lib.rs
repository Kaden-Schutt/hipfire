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
    /// (`off|low|med|high|xhigh|max|uncapped`). Absence means uncapped.
    /// Effort-native families (Qwen3.8, DeepSeek4, Muse Glimmer) omit this field.
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
/// optional reasoning effort and optional legacy budget policy. `general` is the thinking-mode
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
    /// Alternative `lm_head` carriers, keyed by short name (`q4k`, `bf16`).
    ///
    /// Each is a single-tensor `.hfq` from `hipfire-quantize --head-only` that
    /// shadows the base's `lm_head.weight` at load time. The BASE ships the
    /// recommended head and runs standalone; these only exist so a different
    /// carrier costs a 188-635 MB download instead of a near-identical 6.5 GB
    /// model. Three full variants would be 19.63 GB; base plus two overlays is
    /// 7.30 GB.
    #[serde(default)]
    pub heads: std::collections::BTreeMap<String, Sidecar>,
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
        validate_generated_at(&self.generated_at).map_err(fail)?;
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
                .chain(entry.heads.values())
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

fn validate_generated_at(ts: &str) -> std::result::Result<(), String> {
    // Strict normalized RFC3339 UTC: YYYY-MM-DDTHH:MM:SSZ (20 bytes).
    // Lexical order equals chronological order only in this normalized form,
    // which is what `prefer_bundled_if_newer` relies on. Reject any
    // non-normalized representation (offsets, fractional seconds, whitespace,
    // lowercase, etc.) and validate calendar ranges so malformed strings
    // cannot invert precedence via lexical comparison.
    if ts.len() != 20 {
        return Err(format!(
            "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
            ts
        ));
    }
    let b = ts.as_bytes();
    if b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return Err(format!(
            "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
            ts
        ));
    }
    for i in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        if !b[i].is_ascii_digit() {
            return Err(format!(
                "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
                ts
            ));
        }
    }
    let year = (b[0] - b'0') as u16 * 1000
        + (b[1] - b'0') as u16 * 100
        + (b[2] - b'0') as u16 * 10
        + (b[3] - b'0') as u16;
    let month = (b[5] - b'0') * 10 + (b[6] - b'0');
    let day = (b[8] - b'0') * 10 + (b[9] - b'0');
    let hour = (b[11] - b'0') * 10 + (b[12] - b'0');
    let minute = (b[14] - b'0') * 10 + (b[15] - b'0');
    let second = (b[17] - b'0') * 10 + (b[18] - b'0');
    if month == 0 || month > 12 {
        return Err(format!("generated_at '{}' has invalid month", ts));
    }
    if day == 0 || day > days_in_month(year, month) {
        return Err(format!("generated_at '{}' has invalid day", ts));
    }
    if hour > 23 {
        return Err(format!("generated_at '{}' has invalid hour", ts));
    }
    if minute > 59 {
        return Err(format!("generated_at '{}' has invalid minute", ts));
    }
    if second > 59 {
        return Err(format!("generated_at '{}' has invalid second", ts));
    }
    Ok(())
}

fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn is_leap_year(year: u16) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn is_effort_native_tag(tag: &str) -> bool {
    // Mirrors scripts/registry_gen.py:_effort_native_tag and the family/tag
    // conventions already used by config_layer_for_tag. Effort-native families
    // (Qwen3.8, DeepSeek V4 Flash/preview, Muse Glimmer product SKUs) have
    // no registry thinking_budget; absence means uncapped. Draft/dflash
    // sidecars are excluded.
    let base = tag.split(" sampling_profiles.").next().unwrap_or(tag);
    if base.contains("draft") || base.contains("dflash") {
        return false;
    }
    let family = base.split(':').next().unwrap_or(base);
    if family == "qwen3.8" {
        return true;
    }
    if matches!(family, "deepseek-v4-flash" | "deepseek-v4-flash-preview") {
        return true;
    }
    if base == "muse-glimmer" || base == "muse-glimmer:fast" {
        return true;
    }
    false
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
    const REASONING_EFFORTS: &[&str] = &["auto", "none", "low", "medium", "high", "xhigh", "max"];
    const THINKING_BUDGETS: &[&str] = &["off", "low", "med", "high", "xhigh", "max", "uncapped"];
    if let Some(effort) = &value.reasoning_effort {
        if !REASONING_EFFORTS.contains(&effort.as_str()) {
            return Err(format!(
                "model '{tag}' has invalid reasoning_effort '{effort}'"
            ));
        }
    }
    if let Some(budget) = &value.thinking_budget {
        if is_effort_native_tag(tag) {
            return Err(format!(
                "model '{tag}' has unsupported thinking_budget '{budget}' on effort-native model (omit the field; absence means uncapped)"
            ));
        }
        if !THINKING_BUDGETS.contains(&budget.as_str()) {
            return Err(format!(
                "model '{tag}' has invalid thinking_budget '{budget}'"
            ));
        }
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

/// Prefer the BUNDLED registry when it is newer than whatever was fetched.
///
/// `registry/v1.json` is compiled into the binary, so on a branch the bundled
/// copy IS that branch's registry — while the fetch targets master. Without
/// this, editing the registry on a branch changes nothing for a locally built
/// binary: the 24h cache or a master fetch silently wins, and nothing says so.
/// That cost a real debugging detour (a branch's `heads` map read as empty).
///
/// `generated_at` is the existing signal and needs no new configuration:
/// `scripts/registry_gen.py` stamps it on every regeneration, and its
/// `%Y-%m-%dT%H:%M:%SZ` form compares correctly as a plain string.
///
/// This does NOT break distribution. A released binary's bundled registry is
/// older than master's by construction, so the fetch keeps winning there and
/// users still get new models without upgrading. Only a freshly regenerated
/// local registry — i.e. someone editing it — takes precedence.
fn prefer_bundled_if_newer(
    loaded: LoadedRegistry,
    bundled: RegistryV1,
    warnings: &mut Vec<String>,
) -> LoadedRegistry {
    if loaded.source == RegistrySource::Bundled
        || bundled.generated_at <= loaded.registry.generated_at
    {
        return loaded;
    }
    warnings.push(format!(
        "using the bundled registry ({}), which is newer than the fetched one ({})",
        bundled.generated_at, loaded.registry.generated_at
    ));
    LoadedRegistry {
        registry: bundled,
        source: RegistrySource::Bundled,
        warnings: std::mem::take(warnings),
    }
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
        let loaded = LoadedRegistry {
            registry: cache.expect("checked above").registry,
            source: RegistrySource::Cache,
            warnings: std::mem::take(&mut warnings),
        };
        return prefer_bundled_if_newer(loaded, bundled, &mut warnings);
    }

    let loaded = match fetch_registry(&url) {
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
                warnings: std::mem::take(&mut warnings),
            }
        }
        Err(error) => {
            warnings.push(error);
            if let Some(cache) = cache {
                LoadedRegistry {
                    registry: cache.registry,
                    source: RegistrySource::StaleCache,
                    warnings: std::mem::take(&mut warnings),
                }
            } else {
                LoadedRegistry {
                    registry: bundled.clone(),
                    source: RegistrySource::Bundled,
                    warnings: std::mem::take(&mut warnings),
                }
            }
        }
    };
    prefer_bundled_if_newer(loaded, bundled, &mut warnings)
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

    /// The `heads` map must be VALIDATED, not merely parsed.
    ///
    /// Adding a field to the struct makes it round-trip; it does not make the
    /// validator look at it. This asserts the negative directly: a head with a
    /// malformed digest is REJECTED. Without the `.chain(entry.heads.values())`
    /// in `validate`, this test fails and the bundled-registry check would
    /// happily ship an unverifiable head overlay.
    #[test]
    fn heads_sidecars_are_digest_validated() {
        let with_bad_head = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{"repo":"r","file":"f.hfq","size_gb":1,"min_vram_gb":1,"desc":"d",
              "heads":{"q4k":{"file":"h.hfq","sha256":"not-a-sha"}}}},
            "aliases":{}
        }"#;
        let err = RegistryV1::parse(with_bad_head, "test")
            .expect_err("a malformed head digest must be rejected");
        assert!(
            format!("{err}").contains("invalid SHA-256"),
            "expected a digest complaint, got: {err}"
        );

        // Control: the SAME registry with a well-formed digest parses, so the
        // rejection above is about the digest and not about `heads` being
        // unparseable.
        let good = with_bad_head.replace("not-a-sha", &"a".repeat(64));
        let reg = RegistryV1::parse(&good, "test").expect("valid head must parse");
        let (_, entry) = reg.model("m").unwrap();
        assert_eq!(entry.heads.len(), 1);
        assert_eq!(entry.heads["q4k"].file, "h.hfq");
    }

    fn reg_at(stamp: &str) -> RegistryV1 {
        RegistryV1::parse(
            &format!(
                r#"{{"schema_version":1,"generated_at":"{stamp}",
                   "models":{{"m":{{"repo":"r","file":"f","size_gb":1,"min_vram_gb":1,"desc":"d"}}}},
                   "aliases":{{}}}}"#
            ),
            "test",
        )
        .unwrap()
    }

    /// A NEWER bundled registry wins — this is what makes a branch's registry
    /// edits visible to a locally built binary instead of being silently
    /// overridden by the 24h cache or a master fetch.
    #[test]
    fn newer_bundled_registry_beats_a_stale_fetch() {
        let mut w = Vec::new();
        let fetched = LoadedRegistry {
            registry: reg_at("2026-08-31T05:32:38Z"),
            source: RegistrySource::Cache,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at("2026-09-01T13:14:39Z"), &mut w);
        assert_eq!(out.source, RegistrySource::Bundled);
        assert_eq!(out.registry.generated_at, "2026-09-01T13:14:39Z");
        assert!(
            out.warnings.iter().any(|x| x.contains("newer")),
            "the override must be reported, not silent: {:?}",
            out.warnings
        );
    }

    /// The other direction is what keeps DISTRIBUTION working: a released
    /// binary's bundled registry is older than master's, so the fetch must
    /// still win and users get new models without upgrading. Without this the
    /// change above would freeze every client at its build-time registry.
    #[test]
    fn older_bundled_registry_defers_to_the_fetch() {
        let mut w = Vec::new();
        let fetched = LoadedRegistry {
            registry: reg_at("2026-09-01T13:14:39Z"),
            source: RegistrySource::Network,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at("2026-08-28T08:31:54Z"), &mut w);
        assert_eq!(out.source, RegistrySource::Network);
        assert_eq!(out.registry.generated_at, "2026-09-01T13:14:39Z");
        assert!(out.warnings.is_empty(), "no override, so nothing to report");
    }

    /// Equal stamps must not flap between sources.
    #[test]
    fn equal_timestamps_keep_the_fetched_registry() {
        let mut w = Vec::new();
        let same = "2026-09-01T13:14:39Z";
        let fetched = LoadedRegistry {
            registry: reg_at(same),
            source: RegistrySource::Cache,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at(same), &mut w);
        assert_eq!(out.source, RegistrySource::Cache);
    }

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
        assert_eq!(settings.reasoning_effort.as_deref(), Some("xhigh"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // `general` must resolve to the same contract, so `--sampling registry:general`
        // and the bare default cannot diverge.
        let general = model
            .sampling_profile("general")
            .expect("general profile resolves");
        assert_eq!(general.top_k, Some(64));
        assert_eq!(general.reasoning_effort.as_deref(), Some("xhigh"));
        assert!(
            general.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // The card specifies no separate coding/instruct sampling, so those are
        // deliberately absent rather than invented.
        assert!(model.sampling_profile("coding").is_none());
        assert!(model.sampling_profile("instruct").is_none());
    }

    /// Pins Qwen3.8-27B's release contract: aliases, artifact identity, arch/quant,
    /// default KV, and the upstream sampling card (effort-native; no thinking_budget).
    #[test]
    fn bundled_qwen38_matches_the_release_contract() {
        let registry = bundled().unwrap();
        let (tag, model) = registry.model("qwen3.8:27b").unwrap();
        assert_eq!(tag, "qwen3.8:27b");
        assert_eq!(model.file, "qwen3.8-27b.mq4");
        assert_eq!(model.arch_id, Some(5));
        assert_eq!(model.quant.as_deref(), Some("mq4"));
        assert_eq!(model.default_kv_mode.as_deref(), Some("q8"));
        // Pin the canonical MQ4V2 body by content as well as filename: MQ4V2
        // supersedes the former MQ4V1/MQ4R Qwen3.8 products.
        assert_eq!(
            model.sha256.as_deref(),
            Some("5bb556a6cc84035234995c017c9791aa3951ad1eae4cf8c8172b0eaef399e507")
        );
        assert_eq!(model.size_bytes, Some(15662615552));

        for alias in ["qwen3.8", "qwen3.8:latest", "qwen3.8:27b-mq4"] {
            let (resolved, _) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(
                resolved, "qwen3.8:27b",
                "{alias} resolves to the canonical tag"
            );
        }

        // The speed tier is MQ4V2 XT. Both former `fast` tags migrate to it;
        // the superseded `.mq4r` artifact is no longer in the registry.
        let (fast_tag, fast) = registry.model("qwen3.8:27b-fast").unwrap();
        assert_eq!(fast_tag, "qwen3.8:27b-mq4-xt");
        assert_eq!(fast.file, "qwen3.8-27b.mq4-xt");
        assert_eq!(fast.arch_id, Some(5));
        assert_eq!(fast.quant.as_deref(), Some("mq4"));
        assert_eq!(fast.default_kv_mode.as_deref(), Some("q8"));
        assert_eq!(
            fast.sha256.as_deref(),
            Some("9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7")
        );
        assert_eq!(fast.size_bytes, Some(14980361216));
        assert_ne!(
            fast.sha256, model.sha256,
            "the two tiers must not share a content digest"
        );

        for alias in ["qwen3.8:fast", "qwen3.8:27b-fast"] {
            let (resolved, _) = registry.model(alias).expect("fast alias must resolve");
            assert_eq!(resolved, "qwen3.8:27b-mq4-xt");
        }
        assert!(
            registry.model("qwen3.8-27b.mq4r").is_none(),
            "superseded MQ4R filename must not remain addressable"
        );

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
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

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
        assert!(
            general.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

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
        // Tag policy keys off the family before ':' and excludes only draft/dflash
        // tags, so the fast SKU receives the same VMM + 262K + 81920 lowers.
        let fast_layer = config_layer_for_tag(fast_tag, fast)
            .expect("qwen3.8:27b-fast tag policy lowers cleanly");
        assert_eq!(
            fast_layer.get("memory.kv_backend"),
            Some(&ConfigValue::String("vmm".into()))
        );
        assert_eq!(
            fast_layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(262144))
        );
        assert_eq!(
            fast_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("xhigh".into()))
        );
        // Original Qwen3 family (without .5/.6/.8) receives no automatic policy.
        let qwen3_raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
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

        // Effort-native DeepSeek defaults: main low / coding max / instruct none;
        // no registry thinking_budget (absence = uncapped). Preview defaults high.
        let settings = default_sku.recommended_settings.as_ref().unwrap();
        assert_eq!(settings.reasoning_effort.as_deref(), Some("low"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );
        let general = default_sku.sampling_profile("general").unwrap();
        assert_eq!(general.reasoning_effort.as_deref(), Some("low"));
        assert!(general.thinking_budget.is_none());
        let coding = default_sku.sampling_profile("coding").unwrap();
        assert_eq!(coding.reasoning_effort.as_deref(), Some("max"));
        assert!(coding.thinking_budget.is_none());
        let instruct = default_sku.sampling_profile("instruct").unwrap();
        assert_eq!(instruct.reasoning_effort.as_deref(), Some("none"));
        assert!(instruct.thinking_budget.is_none());
        let (_, preview) = registry.model("deepseek-v4-flash-preview").unwrap();
        let preview_rs = preview.recommended_settings.as_ref().unwrap();
        assert_eq!(preview_rs.reasoning_effort.as_deref(), Some("high"));
        assert!(
            preview_rs.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
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
        assert_eq!(registry.resolve_tag("qwen3.8-27b.mq4"), "qwen3.8:27b");
        assert_eq!(
            registry.resolve_tag("qwen3.8-27b.mq4-xt"),
            "qwen3.8:27b-mq4-xt"
        );
        assert_eq!(registry.resolve_tag("qwen3.8:fast"), "qwen3.8:27b-mq4-xt");

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
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
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
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"magic4"}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }

    #[test]
    fn tag_policy_pins_qwen_deepseek_and_glimmer_targets() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_backend":"vmm"}},
            "aliases":{}
        }"#;
        assert!(
            RegistryV1::parse(bad, "test").is_err(),
            "default_kv_backend must be rejected as unknown field"
        );
        let bad2 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_max_seq":262144}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(bad2, "test").is_err());
        let bad3 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
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
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{
                "repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x",
                "sampling_profiles":{"coding":{"temperature":9.0}}
            }},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }

    #[test]
    fn bundled_registry_passes_reasoning_validation() {
        // Bundled v1.json must validate under the mirrored generator invariants:
        // effort-native families have no thinking_budget (absence = uncapped)
        // and all present enums are recognized.
        let registry = bundled().unwrap();
        // Spot-check that effort-native bundled entries indeed omit budget
        for tag in [
            "qwen3.8:27b",
            "qwen3.8:27b-mq4-xt",
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
            "muse-glimmer",
            "muse-glimmer:fast",
        ] {
            let (_, entry) = registry
                .model(tag)
                .unwrap_or_else(|| panic!("{tag} must exist"));
            if let Some(rs) = &entry.recommended_settings {
                assert!(
                    rs.thinking_budget.is_none(),
                    "{tag} bundled thinking_budget must be absent"
                );
            }
            if let Some(profiles) = &entry.sampling_profiles {
                for (_, settings) in [
                    ("general", &profiles.general),
                    ("coding", &profiles.coding),
                    ("instruct", &profiles.instruct),
                ] {
                    if let Some(settings) = settings {
                        assert!(
                            settings.thinking_budget.is_none(),
                            "{tag} profile thinking_budget must be absent"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn effort_native_thinking_budget_rejected_for_fetched_or_cached_registry() {
        // Mirrors scripts/registry_gen.py:_effort_native_tag. A fetched/cache
        // registry that reintroduces a named thinking_budget on effort-native
        // families must fail validation wholesale so load() falls back to
        // bundled (network) or discards the cache entry (fresh/stale).
        let cases = [
            // Qwen3.8 product SKUs
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"high"}}},"aliases":{}}"#,
            // DeepSeek V4 Flash (also covers :mq2lloyd via family)
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"low","thinking_budget":"uncapped"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash-preview":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"high","thinking_budget":"med"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"low"}}},"aliases":{}}"#,
            // Muse Glimmer product SKUs
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"xhigh"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer:fast":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"max"}}},"aliases":{}}"#,
            // Effort-native sampling_profiles also rejected
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"coding":{"reasoning_effort":"xhigh","thinking_budget":"high"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"low"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"uncapped"}}}},"aliases":{}}"#,
        ];
        for raw in cases {
            let err = RegistryV1::parse(raw, "network/cache")
                .expect_err("must reject effort-native thinking_budget");
            let msg = err.to_string();
            assert!(
                msg.contains("thinking_budget") && msg.contains("effort-native"),
                "unexpected error for effort-native budget rejection: {msg}"
            );
        }
        // A cache entry that violates the invariant is also rejected via
        // validate(), causing read_cache to return None and load() to fall back.
        let stale_raw = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}}},"aliases":{}}"#;
        let stale: RegistryV1 = serde_json::from_str(stale_raw).unwrap();
        assert!(
            stale.validate("registry cache").is_err(),
            "stale cache with effort-native thinking_budget must fail validate()"
        );
    }

    #[test]
    fn invalid_reasoning_enums_rejected() {
        // Mirrors hipfire-config/registry_gen enum allowlists: recognizable
        // invalid values are rejected with a clear error; malformed types
        // already fail via surrounding validation and are not re-tested here.
        let invalid_effort = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"m":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"turbo"}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_effort, "test")
            .expect_err("invalid reasoning_effort must be rejected");
        assert!(err.to_string().contains("reasoning_effort"));

        let invalid_effort_profile = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"m":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"coding":{"reasoning_effort":"ultra"}}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_effort_profile, "test")
            .expect_err("invalid profile effort must be rejected");
        assert!(err.to_string().contains("reasoning_effort"));

        // thinking_budget invalid on legacy (non-effort-native) model
        let invalid_budget = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.5:9b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"yolo"}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_budget, "test")
            .expect_err("invalid thinking_budget must be rejected");
        assert!(err.to_string().contains("thinking_budget"));

        let invalid_budget_profile = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.6:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"superhigh"}}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_budget_profile, "test")
            .expect_err("invalid profile budget must be rejected");
        assert!(err.to_string().contains("thinking_budget"));
    }

    #[test]
    fn legacy_qwen35_qwen36_thinking_budget_accepted() {
        // Legacy Qwen3.5/3.6 families are not effort-native; named budgets
        // remain valid and pass validation for both top-level and profiles.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.5:9b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"high","thinking_budget":"high"}},
                "qwen3.5:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"med"},"coding":{"reasoning_effort":"max","thinking_budget":"max"}}},
                "qwen3.6:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"uncapped"}},
                "qwen3.6:35b-a3b-mq4r":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"instruct":{"thinking_budget":"off","reasoning_effort":"none"}}}
            },
            "aliases":{}
        }"#;
        RegistryV1::parse(raw, "test").expect("legacy thinking_budget must remain valid");

        // Draft/dflash sidecars are excluded from effort-native gating, so they
        // also accept thinking_budget even when family would otherwise be native.
        let sidecars = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.8:27b-draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}},
                "qwen3.8:27b-dflash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"low"}},
                "deepseek-v4-flash:draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"xhigh"}},
                "muse-glimmer:draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"med"}}
            },
            "aliases":{}
        }"#;
        RegistryV1::parse(sidecars, "test")
            .expect("draft/dflash sidecars must allow thinking_budget");
    }
}
