// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Shared CLI/server configuration and local filesystem paths.

pub mod schema;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

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
        let mut merged = self.clone();
        if let Some(overrides) = self.model_overrides.get(tag) {
            if let Some(obj) = overrides.as_object() {
                macro_rules! apply_str {
                    ($key:literal, $field:ident) => {
                        if let Some(v) = obj.get($key).and_then(|v| v.as_str()) {
                            merged.$field = v.to_string();
                        }
                    };
                }
                macro_rules! apply_f64 {
                    ($key:literal, $field:ident) => {
                        if let Some(v) = obj.get($key).and_then(|v| v.as_f64()) {
                            merged.$field = v;
                        }
                    };
                }
                macro_rules! apply_u32 {
                    ($key:literal, $field:ident) => {
                        if let Some(v) = obj.get($key).and_then(|v| v.as_u64()) {
                            merged.$field = v as u32;
                        }
                    };
                }
                macro_rules! apply_i32 {
                    ($key:literal, $field:ident) => {
                        if let Some(v) = obj.get($key).and_then(|v| v.as_i64()) {
                            merged.$field = v as i32;
                        }
                    };
                }
                macro_rules! apply_bool {
                    ($key:literal, $field:ident) => {
                        if let Some(v) = obj.get($key).and_then(|v| v.as_bool()) {
                            merged.$field = v;
                        }
                    };
                }
                apply_str!("kv_cache", kv_cache);
                apply_str!("kv_adaptive", kv_adaptive);
                apply_str!("flash_mode", flash_mode);
                apply_str!("dflash_mode", dflash_mode);
                apply_bool!("dflash_adaptive_b", dflash_adaptive_b);
                if let Some(v) = obj.get("dflash_ngram_block") {
                    if v.is_boolean() || v.as_str() == Some("auto") {
                        merged.dflash_ngram_block = v.clone();
                    }
                }
                apply_str!("mtp_mode", mtp_mode);
                apply_str!("thinking", thinking);
                apply_bool!("prompt_normalize", prompt_normalize);
                apply_str!("mmq_screen", mmq_screen);
                apply_str!("prefill_compression", prefill_compression);
                apply_f64!("temperature", temperature);
                apply_f64!("top_p", top_p);
                apply_f64!("repeat_penalty", repeat_penalty);
                apply_f64!("cask_core_frac", cask_core_frac);
                apply_f64!("mmq_screen_threshold", mmq_screen_threshold);
                apply_f64!("prefill_keep_ratio", prefill_keep_ratio);
                apply_f64!("prefill_alpha", prefill_alpha);
                apply_u32!("max_tokens", max_tokens);
                apply_u32!("max_seq", max_seq);
                apply_u32!("mtp_k", mtp_k);
                apply_bool!("cask_auto_attach", cask_auto_attach);
                apply_bool!("cask", cask);
                apply_u32!("cask_budget", cask_budget);
                apply_u32!("cask_beta", cask_beta);
                apply_u32!("cask_fold_m", cask_fold_m);
                apply_u32!("prefill_threshold", prefill_threshold);
                apply_u32!("prefill_min_keep", prefill_min_keep);
                apply_u32!("prefill_sink", prefill_sink);
                apply_u32!("prefill_recent", prefill_recent);
                apply_u32!("prefill_block", prefill_block);
                apply_i32!("prefill_drafter_device", prefill_drafter_device);
                apply_bool!("prefill_profile", prefill_profile);
                apply_u32!("prefill_sparse_threshold", prefill_sparse_threshold);
                if let Some(v) = obj.get("cask_sidecar").and_then(|v| v.as_str()) {
                    merged.cask_sidecar = Some(v.to_string());
                }
                if let Some(v) = obj.get("prefill_drafter").and_then(|v| v.as_str()) {
                    merged.prefill_drafter = Some(v.to_string());
                }
            }
        }
        merged
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

pub fn load_config() -> HipfireConfig {
    let path = config_path();
    if !path.exists() {
        return HipfireConfig {
            host: default_host(),
            port: default_port(),
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
            prefill_drafter_device: default_prefill_drafter_device(),
            prefill_sparse_threshold: default_prefill_sparse_threshold(),
            ..Default::default()
        };
    }
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => HipfireConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_server_config_values() {
        let cfg = HipfireConfig {
            host: default_host(),
            port: default_port(),
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
            prefill_drafter_device: default_prefill_drafter_device(),
            prefill_sparse_threshold: default_prefill_sparse_threshold(),
            ..Default::default()
        };

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
