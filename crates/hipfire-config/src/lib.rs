// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Shared CLI/server configuration and local filesystem paths.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

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
fn default_flash_mode() -> String {
    "auto".to_string()
}
fn default_dflash_mode() -> String {
    "off".to_string()
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    #[serde(default = "default_flash_mode")]
    pub flash_mode: String,
    #[serde(default = "default_dflash_mode")]
    pub dflash_mode: String,
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
    pub prefill_drafter: Option<String>,
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
                apply_str!("kv_cache", kv_cache);
                apply_str!("flash_mode", flash_mode);
                apply_str!("dflash_mode", dflash_mode);
                apply_str!("mtp_mode", mtp_mode);
                apply_str!("thinking", thinking);
                apply_f64!("temperature", temperature);
                apply_f64!("top_p", top_p);
                apply_f64!("repeat_penalty", repeat_penalty);
                apply_u32!("max_tokens", max_tokens);
                apply_u32!("max_seq", max_seq);
                apply_u32!("mtp_k", mtp_k);
            }
        }
        merged
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
            flash_mode: default_flash_mode(),
            dflash_mode: default_dflash_mode(),
            mtp_mode: default_mtp_mode(),
            mtp_k: default_mtp_k(),
            thinking: default_thinking(),
            gpu_slab_load: default_gpu_slab_load(),
            prompt_normalize: default_prompt_normalize(),
            cask_auto_attach: default_cask_auto_attach(),
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
            flash_mode: default_flash_mode(),
            dflash_mode: default_dflash_mode(),
            mtp_mode: default_mtp_mode(),
            mtp_k: default_mtp_k(),
            thinking: default_thinking(),
            gpu_slab_load: default_gpu_slab_load(),
            prompt_normalize: default_prompt_normalize(),
            cask_auto_attach: default_cask_auto_attach(),
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
        assert_eq!(cfg.flash_mode, "auto");
        assert_eq!(cfg.dflash_mode, "off");
        assert_eq!(cfg.mtp_mode, "auto");
        assert_eq!(cfg.mtp_k, 3);
        assert_eq!(cfg.thinking, "off");
        assert_eq!(cfg.gpu_slab_load, "auto");
        assert!(cfg.prompt_normalize);
        assert!(cfg.cask_auto_attach);
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
                "unknown": "ignored"
            }),
        );

        let resolved = cfg.resolve_for_model("qwen");
        assert_eq!(resolved.temperature, 0.1);
        assert_eq!(resolved.top_p, 0.7);
        assert_eq!(resolved.max_tokens, 64);
        assert_eq!(resolved.kv_cache, "q8");
        assert_eq!(cfg.resolve_for_model("other").temperature, 0.3);
    }
}
