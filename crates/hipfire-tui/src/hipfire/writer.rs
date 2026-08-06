// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Writable TOML configuration for the TUI Settings tab.
//!
//! Validation and persistence are delegated to `hipfire-config`; this module
//! only retains the TUI's curated cycling order. The TUI and native CLI now
//! mutate the same sparse `~/.hipfire/config.toml` document.

use hipfire_config::{
    field, load_global, write_global_toml, ConfigLayer, ConfigPaths, ConfigValue,
};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    Enum(&'static [&'static str]),
    Bool,
    Float { min: f64, max: f64 },
    Int { min: i64, max: i64 },
    FreeStr { require_existing_file: bool },
}

#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    pub key: &'static str,
    pub kind: FieldKind,
}

const KV_CACHE: &[&str] = &[
    "auto", "q8", "asym4", "asym3", "asym2", "fwht4", "fwht3", "fwht2", "turbo", "turbo4",
    "turbo3", "turbo2",
];
const KV_ADAPTIVE: &[&str] = &[
    "off",
    "conservative",
    "balanced",
    "aggressive",
    "advanced:k=fwht4,v=lloyd4",
    "advanced:k=fwht4,v=lloyd3",
    "advanced:k=fwht4,v=lloyd2",
    "advanced:k=fwht3,v=lloyd4",
    "advanced:k=fwht3,v=lloyd3",
    "advanced:k=fwht3,v=lloyd2",
    "advanced:k=fwht2,v=lloyd4",
    "advanced:k=fwht2,v=lloyd3",
    "advanced:k=fwht2,v=lloyd2",
];
const AUTO_ON_OFF: &[&str] = &["off", "on", "auto"];
const THINKING: &[&str] = &["on", "off"];
const THINKING_BUDGET: &[&str] = &["low", "med", "high", "xhigh", "max", "uncapped"];
const FLASH_MODE: &[&str] = &["auto", "always", "never"];
const PREFILL_COMPRESSION: &[&str] = &["off", "auto", "always"];

/// Curated display order. Bounds are descriptive only; `hipfire-config` is the
/// sole validator.
pub const EDITABLE_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        key: "kv_cache",
        kind: FieldKind::Enum(KV_CACHE),
    },
    FieldSpec {
        key: "kv_adaptive",
        kind: FieldKind::Enum(KV_ADAPTIVE),
    },
    FieldSpec {
        key: "mtp_mode",
        kind: FieldKind::Enum(AUTO_ON_OFF),
    },
    FieldSpec {
        key: "dflash_mode",
        kind: FieldKind::Enum(AUTO_ON_OFF),
    },
    FieldSpec {
        key: "thinking",
        kind: FieldKind::Enum(THINKING),
    },
    FieldSpec {
        key: "thinking_budget",
        kind: FieldKind::Enum(THINKING_BUDGET),
    },
    FieldSpec {
        key: "chat_template",
        kind: FieldKind::FreeStr {
            require_existing_file: true,
        },
    },
    FieldSpec {
        key: "temperature",
        kind: FieldKind::Float { min: 0.0, max: 2.0 },
    },
    FieldSpec {
        key: "top_p",
        kind: FieldKind::Float { min: 0.0, max: 1.0 },
    },
    FieldSpec {
        key: "max_tokens",
        kind: FieldKind::Int {
            min: 1,
            max: 131072,
        },
    },
    FieldSpec {
        key: "max_seq",
        kind: FieldKind::Int {
            min: 512,
            max: 524288,
        },
    },
    FieldSpec {
        key: "flash_mode",
        kind: FieldKind::Enum(FLASH_MODE),
    },
    FieldSpec {
        key: "mmq_screen",
        kind: FieldKind::Enum(AUTO_ON_OFF),
    },
    FieldSpec {
        key: "prefill_compression",
        kind: FieldKind::Enum(PREFILL_COMPRESSION),
    },
    FieldSpec {
        key: "prefill_drafter",
        kind: FieldKind::FreeStr {
            require_existing_file: false,
        },
    },
    FieldSpec {
        key: "prefill_threshold",
        kind: FieldKind::Int {
            min: 0,
            max: 524288,
        },
    },
    FieldSpec {
        key: "mtp_k",
        kind: FieldKind::Int { min: 1, max: 10 },
    },
    FieldSpec {
        key: "dflash_adaptive_b",
        kind: FieldKind::Bool,
    },
    FieldSpec {
        key: "cask",
        kind: FieldKind::Bool,
    },
    FieldSpec {
        key: "prompt_normalize",
        kind: FieldKind::Bool,
    },
    FieldSpec {
        key: "default_chatml",
        kind: FieldKind::Bool,
    },
];

pub fn field_spec(key: &str) -> Option<&'static FieldSpec> {
    EDITABLE_FIELDS
        .iter()
        .find(|candidate| candidate.key == key)
}

pub fn write_value(path: &Path, key: &str, raw: &str) -> Result<Value, WriteError> {
    let schema = field(key).ok_or_else(|| WriteError::Invalid {
        key: key.to_owned(),
        value: raw.to_owned(),
    })?;
    let config_value = schema.parse_cli(raw).map_err(|_| WriteError::Invalid {
        key: key.to_owned(),
        value: raw.to_owned(),
    })?;
    let json = to_json(&config_value);
    write_config_value(path, schema.key, config_value)?;
    Ok(json)
}

pub fn write_raw_value(path: &Path, key: &str, value: Value) -> Result<(), WriteError> {
    let value = from_json(&value).ok_or_else(|| WriteError::Invalid {
        key: key.to_owned(),
        value: value.to_string(),
    })?;
    write_config_value(path, key, value)
}

fn write_config_value(path: &Path, key: &str, value: ConfigValue) -> Result<(), WriteError> {
    let paths = paths_for(path);
    let mut loaded = load_global(&paths).map_err(config_error)?;
    loaded.layer.set(key, value).map_err(config_error)?;
    write_global_toml(&paths, &loaded.layer).map_err(config_error)
}

pub fn delete_key(path: &Path, key: &str) -> Result<bool, WriteError> {
    let paths = paths_for(path);
    let mut loaded = load_global(&paths).map_err(config_error)?;
    let removed = loaded.layer.remove(key).map_err(config_error)?.is_some();
    if removed {
        write_global_toml(&paths, &loaded.layer).map_err(config_error)?;
    }
    Ok(removed)
}

pub fn reset_all(path: &Path) -> Result<(), WriteError> {
    write_global_toml(&paths_for(path), &ConfigLayer::default()).map_err(config_error)
}

pub fn cycle_enum(key: &str, current: &str, forward: bool) -> Option<String> {
    let spec = field_spec(key)?;
    if let FieldKind::Enum(allow) = spec.kind {
        let index = allow
            .iter()
            .position(|value| *value == current)
            .unwrap_or(0);
        let next = if forward {
            (index + 1) % allow.len()
        } else {
            (index + allow.len() - 1) % allow.len()
        };
        Some(allow[next].to_owned())
    } else {
        None
    }
}

fn paths_for(path: &Path) -> ConfigPaths {
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    ConfigPaths::under(root)
}

fn to_json(value: &ConfigValue) -> Value {
    match value {
        ConfigValue::Bool(value) => Value::Bool(*value),
        ConfigValue::Integer(value) => Value::Number((*value).into()),
        ConfigValue::Float(value) => serde_json::Number::from_f64(*value)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        ConfigValue::String(value) => Value::String(value.clone()),
        ConfigValue::Null => Value::Null,
    }
}

fn from_json(value: &Value) -> Option<ConfigValue> {
    match value {
        Value::Bool(value) => Some(ConfigValue::Bool(*value)),
        Value::Number(value) => value
            .as_i64()
            .map(ConfigValue::Integer)
            .or_else(|| value.as_f64().map(ConfigValue::Float)),
        Value::String(value) => Some(ConfigValue::String(value.clone())),
        Value::Null => Some(ConfigValue::Null),
        Value::Array(_) | Value::Object(_) => None,
    }
}

#[derive(Debug)]
pub enum WriteError {
    Invalid { key: String, value: String },
    Config(String),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid { key, value } => {
                write!(output, "rejected invalid value for {key}: {value:?}")
            }
            Self::Config(message) => output.write_str(message),
        }
    }
}

impl std::error::Error for WriteError {}

fn config_error(error: hipfire_config::ConfigError) -> WriteError {
    WriteError::Config(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_config() -> (std::path::PathBuf, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "hipfire-tui-writer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("config.toml");
        (root, path)
    }

    #[test]
    fn writes_sparse_toml_through_shared_schema() {
        let (root, path) = temp_config();
        write_value(&path, "kv_cache", "q8").unwrap();
        write_value(&path, "max_tokens", "8192").unwrap();
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert_eq!(
            loaded.layer.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert_eq!(
            loaded.layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(8192))
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn shared_validation_rejects_invalid_values() {
        let (root, path) = temp_config();
        assert!(write_value(&path, "top_p", "0").is_err());
        assert!(write_value(&path, "port", "70000").is_err());
        assert!(write_value(&path, "kv_cache", "magic4").is_err());
        assert!(!path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn reset_removes_only_the_selected_override() {
        let (root, path) = temp_config();
        write_value(&path, "kv_cache", "q8").unwrap();
        write_value(&path, "thinking", "off").unwrap();
        assert!(delete_key(&path, "kv_cache").unwrap());
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert!(loaded.layer.get("kv_cache").is_none());
        assert_eq!(
            loaded.layer.get("thinking"),
            Some(&ConfigValue::String("off".into()))
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cycling_order_remains_curated_for_the_tui() {
        assert_eq!(cycle_enum("kv_cache", "auto", true).as_deref(), Some("q8"));
        assert_eq!(
            cycle_enum("kv_adaptive", "advanced:k=fwht4,v=lloyd4", true).as_deref(),
            Some("advanced:k=fwht4,v=lloyd3")
        );
        assert!(cycle_enum("temperature", "0.3", true).is_none());
    }
}
