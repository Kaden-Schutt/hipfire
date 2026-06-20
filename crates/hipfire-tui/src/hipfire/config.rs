// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{collections::BTreeMap, fs, time::Duration};

use hipfire_config::config_schema;
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
    pub schema_field_count: Option<usize>,
    pub schema_warning: Option<String>,
    pub resolved_from_daemon: bool,
}

impl ConfigState {
    pub fn load(paths: &HipfirePaths) -> Self {
        let mut values = defaults();
        let mut loaded_from_disk = false;
        let mut warning = None;

        match fs::read_to_string(&paths.config) {
            Ok(raw) => match serde_json::from_str::<Value>(&raw) {
                Ok(Value::Object(map)) => {
                    loaded_from_disk = true;
                    for (k, v) in map {
                        values.insert(k, value_to_string(&v));
                    }
                }
                Ok(_) => warning = Some("config.json is not an object; using defaults".into()),
                Err(err) => warning = Some(format!("config parse error: {err}")),
            },
            Err(_) => {}
        }

        let per_model_count = fs::read_to_string(&paths.per_model_config)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| v.as_object().map(|m| m.len()))
            .unwrap_or(0);

        let probe_host = values
            .get("host")
            .cloned()
            .unwrap_or_else(|| "0.0.0.0".into());
        let probe_port = values
            .get("port")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11435);
        let mut resolved_from_daemon = false;
        let (schema_field_count, schema_warning) =
            match load_remote_resolved(&probe_host_for(&probe_host), probe_port) {
                Ok(remote) => {
                    values = remote.values;
                    resolved_from_daemon = true;
                    (Some(remote.field_count), None)
                }
                Err(err) => {
                    let (count, schema_warning) =
                        load_remote_schema(&probe_host_for(&probe_host), probe_port);
                    (count, Some(schema_warning.unwrap_or(err)))
                }
            };

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
            .filter(|model| !model.is_empty())
            .cloned()
            .unwrap_or_else(|| "unset".into());

        Self {
            host,
            port,
            default_model,
            values,
            per_model_count,
            loaded_from_disk,
            warning,
            schema_field_count,
            schema_warning,
            resolved_from_daemon,
        }
    }

    pub fn probe_host(&self) -> String {
        probe_host_for(&self.host)
    }

    pub fn easy_rows(&self) -> Vec<(&'static str, String, String)> {
        vec![
            (
                "Model",
                self.default_model.clone(),
                "Default model pre-warmed by serve and used by chat.".into(),
            ),
            (
                "Context",
                self.values
                    .get("max_seq")
                    .cloned()
                    .unwrap_or_else(|| "32768".into()),
                "KV cache capacity allocated at load.".into(),
            ),
            (
                "Spec decode",
                self.values
                    .get("dflash_mode")
                    .cloned()
                    .unwrap_or_else(|| "off".into()),
                "DFlash mode. Keep off unless intentionally testing drafts.".into(),
            ),
            (
                "KV cache",
                self.values
                    .get("kv_cache")
                    .cloned()
                    .unwrap_or_else(|| "auto".into()),
                "Precision/memory tradeoff for attention cache.".into(),
            ),
            (
                "Thinking",
                self.values
                    .get("thinking")
                    .cloned()
                    .unwrap_or_else(|| "on".into()),
                "Whether reasoning models emit a hidden think block.".into(),
            ),
            (
                "Serve",
                format!("{}:{}", self.host, self.port),
                "OpenAI-compatible endpoint used by chat and API clients.".into(),
            ),
            (
                "Schema",
                self.schema_field_count
                    .map(|count| format!("{count} live fields"))
                    .unwrap_or_else(|| "offline".into()),
                self.schema_warning.clone().unwrap_or_else(|| {
                    if self.resolved_from_daemon {
                        "Loaded active config from daemon operator API.".into()
                    } else {
                        "Loaded schema from daemon operator API.".into()
                    }
                }),
            ),
        ]
    }
}

struct RemoteResolvedConfig {
    values: BTreeMap<String, String>,
    field_count: usize,
}

fn probe_host_for(host: &str) -> String {
    match host {
        "0.0.0.0" | "" => "127.0.0.1".into(),
        "::" => "::1".into(),
        other => other.to_string(),
    }
}

fn load_remote_schema(host: &str, port: u16) -> (Option<usize>, Option<String>) {
    let url = format!("http://{host}:{port}/operator/config/schema");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(450))
        .build();
    match agent.get(&url).call() {
        Ok(resp) => match resp
            .into_string()
            .ok()
            .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        {
            Some(Value::Array(fields)) => (Some(fields.len()), None),
            Some(_) => (
                None,
                Some("operator schema endpoint returned non-array JSON".into()),
            ),
            None => (None, Some("schema parse error".into())),
        },
        Err(ureq::Error::Status(code, resp)) => {
            let text = resp.into_string().unwrap_or_default();
            (
                None,
                Some(format!(
                    "schema HTTP {code}: {}",
                    text.chars().take(120).collect::<String>()
                )),
            )
        }
        Err(err) => (None, Some(format!("schema unavailable: {err}"))),
    }
}

fn load_remote_resolved(host: &str, port: u16) -> Result<RemoteResolvedConfig, String> {
    let url = format!("http://{host}:{port}/operator/config/resolved");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(650))
        .build();
    let body = agent
        .get(&url)
        .call()
        .map_err(|err| format!("resolved config unavailable: {err}"))?
        .into_string()
        .map_err(|err| format!("resolved config read error: {err}"))?;
    let payload = serde_json::from_str::<Value>(&body)
        .map_err(|err| format!("resolved config parse error: {err}"))?;
    let values = values_from_resolution(&payload)
        .ok_or_else(|| "resolved config endpoint returned unexpected JSON".to_string())?;
    let field_count = payload
        .get("resolution")
        .and_then(|resolution| resolution.get("values"))
        .and_then(Value::as_array)
        .map(|values| values.len())
        .unwrap_or(values.len());
    Ok(RemoteResolvedConfig {
        values,
        field_count,
    })
}

fn values_from_resolution(payload: &Value) -> Option<BTreeMap<String, String>> {
    let values = payload
        .get("resolution")?
        .get("values")?
        .as_array()?
        .iter()
        .filter_map(|entry| {
            let key = entry.get("key")?.as_str()?;
            let value = entry.get("value")?;
            if value.is_null() {
                return None;
            }
            Some((key.to_string(), value_to_string(value)))
        })
        .collect();
    Some(values)
}

fn defaults() -> BTreeMap<String, String> {
    config_schema()
        .iter()
        .filter_map(|field| {
            field
                .default
                .map(|default| (field.key.to_string(), default_to_string(default)))
        })
        .collect()
}

fn default_to_string(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .map(|value| value_to_string(&value))
        .unwrap_or_else(|_| raw.to_string())
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
