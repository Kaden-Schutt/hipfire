// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Shared CLI/server configuration and local filesystem paths.

pub mod resolve;
pub mod schema;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

pub use resolve::{
    config_layers_from_document, resolve_config_layers, ConfigLayer, ConfigLayerKind,
    ConfigResolution, ConfigValueSource, ResolvedConfigValue, UnknownConfigKey,
};
pub use schema::{
    config_schema, ConfigField, ConfigMutability, ConfigScope, ConfigType, Requirement,
    RestartImpact,
};

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    11435
}
fn default_max_seq() -> u32 {
    4096
}
fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f64 {
    0.3
}
fn default_top_p() -> f64 {
    0.8
}
fn default_repeat_penalty() -> f64 {
    1.05
}
fn default_idle_timeout() -> u32 {
    300
}
fn default_kv_cache() -> String {
    "auto".to_string()
}
fn default_kv_adaptive() -> String {
    "off".to_string()
}
fn default_flash_mode() -> String {
    "auto".to_string()
}
fn default_dflash_mode() -> String {
    "off".to_string()
}
fn default_dflash_adaptive_b() -> bool {
    true
}
fn default_dflash_ngram_block() -> serde_json::Value {
    serde_json::Value::String("auto".to_string())
}
fn default_mtp_mode() -> String {
    "auto".to_string()
}
fn default_mtp_k() -> u32 {
    3
}
fn default_thinking() -> String {
    "off".to_string()
}
fn default_gpu_slab_load() -> String {
    "auto".to_string()
}
fn default_prompt_normalize() -> bool {
    true
}
fn default_cask_auto_attach() -> bool {
    true
}
fn default_cask_budget() -> u32 {
    512
}
fn default_cask_beta() -> u32 {
    128
}
fn default_cask_core_frac() -> f64 {
    0.5
}
fn default_cask_fold_m() -> u32 {
    2
}
fn default_mmq_screen() -> String {
    "auto".to_string()
}
fn default_mmq_screen_threshold() -> f64 {
    0.10
}
fn default_prefill_compression() -> String {
    "off".to_string()
}
fn default_prefill_threshold() -> u32 {
    32768
}
fn default_prefill_keep_ratio() -> f64 {
    0.05
}
fn default_prefill_alpha() -> f64 {
    0.85
}
fn default_prefill_min_keep() -> u32 {
    2048
}
fn default_prefill_sink() -> u32 {
    256
}
fn default_prefill_recent() -> u32 {
    1024
}
fn default_prefill_block() -> u32 {
    128
}
fn default_prefill_drafter_device() -> i32 {
    -1
}
fn default_prefill_sparse_threshold() -> u32 {
    32768
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipfireConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default = "default_max_seq")]
    pub max_seq: u32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f64,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u32,
    #[serde(default = "default_kv_cache")]
    pub kv_cache: String,
    #[serde(default = "default_kv_adaptive")]
    pub kv_adaptive: String,
    #[serde(default = "default_flash_mode")]
    pub flash_mode: String,
    #[serde(default = "default_dflash_mode")]
    pub dflash_mode: String,
    #[serde(default = "default_dflash_adaptive_b")]
    pub dflash_adaptive_b: bool,
    #[serde(default = "default_dflash_ngram_block")]
    pub dflash_ngram_block: serde_json::Value,
    #[serde(default = "default_mtp_mode")]
    pub mtp_mode: String,
    #[serde(default = "default_mtp_k")]
    pub mtp_k: u32,
    #[serde(default = "default_thinking")]
    pub thinking: String,
    #[serde(default = "default_gpu_slab_load")]
    pub gpu_slab_load: String,
    #[serde(default = "default_prompt_normalize")]
    pub prompt_normalize: bool,
    #[serde(default = "default_cask_auto_attach")]
    pub cask_auto_attach: bool,
    #[serde(default)]
    pub cask_sidecar: Option<String>,
    #[serde(default)]
    pub cask: bool,
    #[serde(default = "default_cask_budget")]
    pub cask_budget: u32,
    #[serde(default = "default_cask_beta")]
    pub cask_beta: u32,
    #[serde(default = "default_cask_core_frac")]
    pub cask_core_frac: f64,
    #[serde(default = "default_cask_fold_m")]
    pub cask_fold_m: u32,
    #[serde(default = "default_mmq_screen")]
    pub mmq_screen: String,
    #[serde(default = "default_mmq_screen_threshold")]
    pub mmq_screen_threshold: f64,
    #[serde(default = "default_prefill_compression")]
    pub prefill_compression: String,
    #[serde(default = "default_prefill_threshold")]
    pub prefill_threshold: u32,
    #[serde(default = "default_prefill_keep_ratio")]
    pub prefill_keep_ratio: f64,
    #[serde(default = "default_prefill_alpha")]
    pub prefill_alpha: f64,
    #[serde(default = "default_prefill_min_keep")]
    pub prefill_min_keep: u32,
    #[serde(default = "default_prefill_sink")]
    pub prefill_sink: u32,
    #[serde(default = "default_prefill_recent")]
    pub prefill_recent: u32,
    #[serde(default = "default_prefill_block")]
    pub prefill_block: u32,
    #[serde(default)]
    pub prefill_drafter: Option<String>,
    #[serde(default = "default_prefill_drafter_device")]
    pub prefill_drafter_device: i32,
    #[serde(default)]
    pub prefill_profile: bool,
    #[serde(default = "default_prefill_sparse_threshold")]
    pub prefill_sparse_threshold: u32,
    #[serde(default)]
    pub model_overrides: HashMap<String, serde_json::Value>,
}

impl HipfireConfig {
    /// Merge per-model overrides for `tag` on top of global config.
    pub fn resolve_for_model(&self, tag: &str) -> Self {
        let raw = serde_json::to_value(self).unwrap_or_else(|_| Value::Object(Map::new()));
        resolve_typed_config_document(&raw, Some(tag)).config
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDiagnosticSeverity {
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigDiagnostic {
    pub severity: ConfigDiagnosticSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedTypedConfig {
    pub config: HipfireConfig,
    pub layers: Vec<ConfigLayer>,
    pub resolution: ConfigResolution,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadedConfig {
    pub config_path: PathBuf,
    pub raw_document: Value,
    pub read_error: Option<String>,
    pub additional_layers: Vec<ConfigLayer>,
    pub layers: Vec<ConfigLayer>,
    pub config: HipfireConfig,
    pub resolution: ConfigResolution,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

impl LoadedConfig {
    pub fn from_config(config: HipfireConfig) -> Self {
        let raw_document =
            serde_json::to_value(&config).unwrap_or_else(|_| Value::Object(Map::new()));
        loaded_config_from_document(config_path(), raw_document, None, Vec::new())
    }

    pub fn with_additional_layer(mut self, layer: ConfigLayer) -> Self {
        if !layer.values.is_empty() {
            self.additional_layers.push(layer);
            self.refresh();
        }
        self
    }

    pub fn resolve_for_model(&self, model_tag: &str) -> ResolvedTypedConfig {
        resolve_typed_config_document_with_layers(
            &self.raw_document,
            Some(model_tag),
            &self.additional_layers,
        )
    }

    fn refresh(&mut self) {
        let resolved = resolve_typed_config_document_with_layers(
            &self.raw_document,
            None,
            &self.additional_layers,
        );
        self.config = resolved.config;
        self.layers = resolved.layers;
        self.resolution = resolved.resolution;
        self.diagnostics = resolved.diagnostics;
    }
}

impl Default for HipfireConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            default_model: None,
            max_seq: default_max_seq(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            repeat_penalty: default_repeat_penalty(),
            idle_timeout: default_idle_timeout(),
            kv_cache: default_kv_cache(),
            kv_adaptive: default_kv_adaptive(),
            flash_mode: default_flash_mode(),
            dflash_mode: default_dflash_mode(),
            dflash_adaptive_b: default_dflash_adaptive_b(),
            dflash_ngram_block: default_dflash_ngram_block(),
            mtp_mode: default_mtp_mode(),
            mtp_k: default_mtp_k(),
            thinking: default_thinking(),
            gpu_slab_load: default_gpu_slab_load(),
            prompt_normalize: default_prompt_normalize(),
            cask_auto_attach: default_cask_auto_attach(),
            cask_sidecar: None,
            cask: false,
            cask_budget: default_cask_budget(),
            cask_beta: default_cask_beta(),
            cask_core_frac: default_cask_core_frac(),
            cask_fold_m: default_cask_fold_m(),
            mmq_screen: default_mmq_screen(),
            mmq_screen_threshold: default_mmq_screen_threshold(),
            prefill_compression: default_prefill_compression(),
            prefill_threshold: default_prefill_threshold(),
            prefill_keep_ratio: default_prefill_keep_ratio(),
            prefill_alpha: default_prefill_alpha(),
            prefill_min_keep: default_prefill_min_keep(),
            prefill_sink: default_prefill_sink(),
            prefill_recent: default_prefill_recent(),
            prefill_block: default_prefill_block(),
            prefill_drafter: None,
            prefill_drafter_device: default_prefill_drafter_device(),
            prefill_profile: false,
            prefill_sparse_threshold: default_prefill_sparse_threshold(),
            model_overrides: HashMap::new(),
        }
    }
}

pub fn hipfire_dir() -> PathBuf {
    dirs::home_dir()
        .expect("no home directory")
        .join(".hipfire")
}

pub fn config_path() -> PathBuf {
    hipfire_dir().join("config.json")
}

pub fn models_dir() -> PathBuf {
    hipfire_dir().join("models")
}

pub fn load_config_bundle() -> LoadedConfig {
    let path = config_path();
    match std::fs::read_to_string(&path) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(document) => loaded_config_from_document(path, document, None, Vec::new()),
            Err(err) => loaded_config_from_document(
                path,
                Value::Object(Map::new()),
                Some(format!("parse error: {err}")),
                Vec::new(),
            ),
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            loaded_config_from_document(path, Value::Object(Map::new()), None, Vec::new())
        }
        Err(err) => loaded_config_from_document(
            path,
            Value::Object(Map::new()),
            Some(format!("read error: {err}")),
            Vec::new(),
        ),
    }
}

pub fn load_config() -> HipfireConfig {
    load_config_bundle().config
}

pub fn resolve_typed_config_document(raw: &Value, model_tag: Option<&str>) -> ResolvedTypedConfig {
    resolve_typed_config_document_with_layers(raw, model_tag, &[])
}

pub fn resolve_typed_config_document_with_layers(
    raw: &Value,
    model_tag: Option<&str>,
    additional_layers: &[ConfigLayer],
) -> ResolvedTypedConfig {
    let mut layers = config_layers_from_document(raw, model_tag);
    layers.extend(additional_layers.iter().cloned());
    resolve_typed_config_layers(&layers, model_overrides_from_document(raw))
}

pub fn resolve_typed_config_layers(
    layers: &[ConfigLayer],
    model_overrides: HashMap<String, Value>,
) -> ResolvedTypedConfig {
    let resolution = resolve_config_layers(config_schema(), layers);
    let mut diagnostics = Vec::new();
    let config = materialize_config(&resolution, model_overrides, &mut diagnostics);
    ResolvedTypedConfig {
        config,
        layers: layers.to_vec(),
        resolution,
        diagnostics,
    }
}

pub fn loaded_config_from_document(
    config_path: PathBuf,
    raw_document: Value,
    read_error: Option<String>,
    additional_layers: Vec<ConfigLayer>,
) -> LoadedConfig {
    let resolved =
        resolve_typed_config_document_with_layers(&raw_document, None, &additional_layers);
    LoadedConfig {
        config_path,
        raw_document,
        read_error,
        additional_layers,
        layers: resolved.layers,
        config: resolved.config,
        resolution: resolved.resolution,
        diagnostics: resolved.diagnostics,
    }
}

fn materialize_config(
    resolution: &ConfigResolution,
    model_overrides: HashMap<String, Value>,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) -> HipfireConfig {
    let mut object = Map::new();
    for resolved in &resolution.values {
        if let Some(value) = &resolved.value {
            object.insert(resolved.key.clone(), value.clone());
        }
    }
    if !model_overrides.is_empty() {
        object.insert(
            "model_overrides".to_string(),
            Value::Object(model_overrides.into_iter().collect()),
        );
    }

    match serde_json::from_value::<HipfireConfig>(Value::Object(object)) {
        Ok(config) => config,
        Err(err) => {
            diagnostics.push(ConfigDiagnostic {
                severity: ConfigDiagnosticSeverity::Error,
                message: format!("failed to materialize typed config: {err}"),
            });
            HipfireConfig::default()
        }
    }
}

fn model_overrides_from_document(raw: &Value) -> HashMap<String, Value> {
    raw.get("model_overrides")
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

pub fn config_value_map(config: &HipfireConfig) -> BTreeMap<String, Value> {
    serde_json::to_value(config)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .map(|object| object.into_iter().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_server_config_values() {
        let cfg = HipfireConfig::default();

        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 11435);
        assert_eq!(cfg.max_seq, 4096);
        assert_eq!(cfg.max_tokens, 512);
        assert_eq!(cfg.temperature, 0.3);
        assert_eq!(cfg.top_p, 0.8);
        assert_eq!(cfg.repeat_penalty, 1.05);
        assert_eq!(cfg.idle_timeout, 300);
        assert_eq!(cfg.kv_cache, "auto");
        assert_eq!(cfg.kv_adaptive, "off");
        assert_eq!(cfg.flash_mode, "auto");
        assert_eq!(cfg.dflash_mode, "off");
        assert!(cfg.dflash_adaptive_b);
        assert_eq!(cfg.dflash_ngram_block, serde_json::json!("auto"));
        assert_eq!(cfg.mtp_mode, "auto");
        assert_eq!(cfg.mtp_k, 3);
        assert_eq!(cfg.thinking, "off");
        assert_eq!(cfg.gpu_slab_load, "auto");
        assert!(cfg.prompt_normalize);
        assert!(cfg.cask_auto_attach);
        assert!(!cfg.cask);
        assert_eq!(cfg.cask_budget, 512);
        assert_eq!(cfg.cask_beta, 128);
        assert_eq!(cfg.cask_core_frac, 0.5);
        assert_eq!(cfg.cask_fold_m, 2);
        assert_eq!(cfg.mmq_screen, "auto");
        assert_eq!(cfg.mmq_screen_threshold, 0.10);
        assert_eq!(cfg.prefill_compression, "off");
        assert_eq!(cfg.prefill_threshold, 32768);
        assert_eq!(cfg.prefill_keep_ratio, 0.05);
        assert_eq!(cfg.prefill_alpha, 0.85);
        assert_eq!(cfg.prefill_min_keep, 2048);
        assert_eq!(cfg.prefill_sink, 256);
        assert_eq!(cfg.prefill_recent, 1024);
        assert_eq!(cfg.prefill_block, 128);
        assert_eq!(cfg.prefill_drafter_device, -1);
        assert!(!cfg.prefill_profile);
        assert_eq!(cfg.prefill_sparse_threshold, 32768);
    }

    #[test]
    fn schema_defaults_materialize_to_typed_defaults() {
        let resolved = resolve_typed_config_document(&serde_json::json!({}), None);

        assert!(resolved.diagnostics.is_empty());
        assert_eq!(
            config_value_map(&resolved.config),
            config_value_map(&HipfireConfig::default())
        );
    }

    #[test]
    fn loaded_config_preserves_raw_model_overrides() {
        let loaded = loaded_config_from_document(
            PathBuf::from("/tmp/config.json"),
            serde_json::json!({
                "temperature": 0.4,
                "model_overrides": {
                    "qwen": {
                        "temperature": 0.1,
                        "max_tokens": 64
                    }
                }
            }),
            None,
            Vec::new(),
        );

        assert_eq!(loaded.config.temperature, 0.4);
        assert!(loaded.config.model_overrides.contains_key("qwen"));

        let resolved = loaded.resolve_for_model("qwen");
        assert_eq!(resolved.config.temperature, 0.1);
        assert_eq!(resolved.config.max_tokens, 64);
        assert!(resolved.config.model_overrides.contains_key("qwen"));
    }

    #[test]
    fn model_overrides_preserve_typed_merge_policy() {
        let mut cfg = HipfireConfig::default();
        cfg.temperature = 0.3;
        cfg.max_tokens = 512;
        cfg.model_overrides.insert(
            "qwen".to_string(),
            serde_json::json!({
                "temperature": 0.1,
                "top_p": 0.7,
                "max_tokens": 64,
                "kv_cache": "q8",
                "kv_adaptive": "balanced",
                "dflash_ngram_block": true,
                "cask": true,
                "cask_budget": 1024,
                "prefill_compression": "auto",
                "prefill_drafter_device": 1,
                "unknown": "ignored"
            }),
        );

        let resolved = cfg.resolve_for_model("qwen");
        assert_eq!(resolved.temperature, 0.1);
        assert_eq!(resolved.top_p, 0.7);
        assert_eq!(resolved.max_tokens, 64);
        assert_eq!(resolved.kv_cache, "q8");
        assert_eq!(resolved.kv_adaptive, "balanced");
        assert_eq!(resolved.dflash_ngram_block, serde_json::json!(true));
        assert!(resolved.cask);
        assert_eq!(resolved.cask_budget, 1024);
        assert_eq!(resolved.prefill_compression, "auto");
        assert_eq!(resolved.prefill_drafter_device, 1);
        assert_eq!(cfg.resolve_for_model("other").temperature, 0.3);
    }
}
