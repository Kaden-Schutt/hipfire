// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{collections::BTreeMap, fs};

use serde_json::Value;

use super::HipfirePaths;

#[derive(Clone, Debug)]
pub struct ConfigState {
    pub host: String,
    pub port: u16,
    pub default_model: String,
    pub values: BTreeMap<String, String>,
    pub per_model_count: usize,
    pub loaded_from_disk: bool,
    pub warning: Option<String>,
}

impl ConfigState {
    pub fn load(paths: &HipfirePaths) -> Self {
        let mut values = defaults();
        let mut loaded_from_disk = false;
        let mut warning = None;

        if let Ok(raw) = fs::read_to_string(&paths.config) {
            match serde_json::from_str::<Value>(&raw) {
                Ok(Value::Object(map)) => {
                    loaded_from_disk = true;
                    for (k, v) in map {
                        values.insert(k, value_to_string(&v));
                    }
                }
                Ok(_) => warning = Some("config.json is not an object; using defaults".into()),
                Err(err) => warning = Some(format!("config parse error: {err}")),
            }
        }

        let per_model_count = fs::read_to_string(&paths.per_model_config)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| v.as_object().map(|m| m.len()))
            .unwrap_or(0);

        let host = values
            .get("host")
            .cloned()
            .unwrap_or_else(|| "0.0.0.0".into());
        let port = values
            .get("port")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11435);
        let default_model = values
            .get("default_model")
            .cloned()
            .unwrap_or_else(|| "qwen3.5:9b".into());

        Self {
            host,
            port,
            default_model,
            values,
            per_model_count,
            loaded_from_disk,
            warning,
        }
    }

    pub fn probe_host(&self) -> String {
        match self.host.as_str() {
            "0.0.0.0" | "" => "127.0.0.1".into(),
            "::" => "::1".into(),
            other => other.to_string(),
        }
    }

    pub fn easy_rows(&self) -> Vec<EasyRow> {
        let get = |key: &str, fallback: &str| -> String {
            self.values
                .get(key)
                .cloned()
                .unwrap_or_else(|| fallback.to_string())
        };
        vec![
            EasyRow {
                key: Some("default_model"),
                label: "Model",
                value: self.default_model.clone(),
                desc: "Default model pre-warmed by serve and used by chat.",
            },
            EasyRow {
                key: Some("max_seq"),
                label: "Context",
                value: get("max_seq", "32768"),
                desc: "KV cache capacity allocated at load.",
            },
            EasyRow {
                key: Some("dflash_mode"),
                label: "Spec decode",
                value: get("dflash_mode", "off"),
                desc: "DFlash mode. Keep off unless intentionally testing drafts.",
            },
            EasyRow {
                key: Some("kv_cache"),
                label: "KV cache",
                value: get("kv_cache", "auto"),
                desc: "Precision/memory tradeoff for attention cache.",
            },
            EasyRow {
                key: Some("thinking"),
                label: "Thinking",
                value: get("thinking", "on"),
                desc: "Whether reasoning models emit a hidden think block.",
            },
            EasyRow {
                key: None,
                label: "Serve",
                value: format!("{}:{}", self.host, self.port),
                desc: "Endpoint. Edit host/port as separate keys in advanced view.",
            },
        ]
    }
}

/// One row of the user-safe settings view. `key` is the underlying
/// `hipfire config set` key, or None when the row is informational only.
#[derive(Clone, Debug)]
pub struct EasyRow {
    pub key: Option<&'static str>,
    pub label: &'static str,
    pub value: String,
    pub desc: &'static str,
}

fn defaults() -> BTreeMap<String, String> {
    [
        ("kv_cache", "auto"),
        ("kv_adaptive", "off"),
        ("flash_mode", "auto"),
        ("default_model", "qwen3.5:9b"),
        ("temperature", "0.3"),
        ("top_p", "0.8"),
        ("repeat_penalty", "1.05"),
        ("max_tokens", "4096"),
        ("max_seq", "32768"),
        ("thinking", "on"),
        ("max_think_tokens", "2048"),
        ("max_total_think_tokens", "0"),
        ("host", "0.0.0.0"),
        ("port", "11435"),
        ("idle_timeout", "300"),
        ("dflash_mode", "off"),
        ("dflash_adaptive_b", "true"),
        ("dflash_ngram_block", "auto"),
        ("cask", "false"),
        ("cask_budget", "512"),
        ("cask_beta", "128"),
        ("cask_auto_attach", "true"),
        ("prompt_normalize", "true"),
        ("mmq_screen", "auto"),
        ("prefill_compression", "off"),
        ("mtp_mode", "auto"),
        ("mtp_k", "3"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        _ => v.to_string(),
    }
}
