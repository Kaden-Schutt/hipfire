// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed configuration schema and deterministic layer resolution for hipfire.
//!
//! This crate is deliberately independent of GPU and model crates. It owns the
//! public configuration vocabulary, legacy-key compatibility, TOML/JSON
//! persistence, validation, and field-level provenance. Runtime crates consume
//! resolved values; they do not parse user configuration in hot paths.

pub mod rocm;

use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fmt, fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};
use thiserror::Error;

pub const CONFIG_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("unknown configuration key '{0}'")]
    UnknownKey(String),
    #[error("invalid value for {key}: {message}")]
    InvalidValue { key: String, message: String },
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, ConfigError>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Null,
}

impl ConfigValue {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Integer(_) => "integer",
            Self::Float(_) => "number",
            Self::String(_) => "string",
            Self::Null => "null",
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    fn into_toml(self) -> Option<toml::Value> {
        match self {
            Self::Bool(v) => Some(toml::Value::Boolean(v)),
            Self::Integer(v) => Some(toml::Value::Integer(v)),
            Self::Float(v) => Some(toml::Value::Float(v)),
            Self::String(v) => Some(toml::Value::String(v)),
            Self::Null => None,
        }
    }

    fn from_toml(value: &toml::Value) -> Option<Self> {
        match value {
            toml::Value::Boolean(v) => Some(Self::Bool(*v)),
            toml::Value::Integer(v) => Some(Self::Integer(*v)),
            toml::Value::Float(v) => Some(Self::Float(*v)),
            toml::Value::String(v) => Some(Self::String(v.clone())),
            _ => None,
        }
    }
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::Integer(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::String(v) => f.write_str(v),
            Self::Null => f.write_str("null"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConfigCategory {
    Generation,
    Reasoning,
    Hardware,
    Kernel,
    Memory,
    Attention,
    Speculation,
    Replay,
    Fusions,
    Prompt,
    Serve,
    Diagnostic,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConfigScope {
    Process,
    ModelLoad,
    Session,
    Request,
    Diagnostic,
}

#[derive(Clone, Copy, Debug)]
pub enum DefaultValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(&'static str),
    Null,
}

impl DefaultValue {
    fn to_value(self) -> ConfigValue {
        match self {
            Self::Bool(v) => ConfigValue::Bool(v),
            Self::Integer(v) => ConfigValue::Integer(v),
            Self::Float(v) => ConfigValue::Float(v),
            Self::String(v) => ConfigValue::String(v.to_owned()),
            Self::Null => ConfigValue::Null,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ValueRule {
    Bool,
    Integer {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
        min_inclusive: bool,
    },
    String,
    NonEmptyString,
    Host,
    PathOrEmpty,
    Enum(&'static [&'static str]),
    AutoBool,
    NullableString,
    NullableEnum(&'static [&'static str]),
    NullableInteger {
        min: i64,
        max: i64,
    },
    NullableFloat {
        min: f64,
        max: f64,
    },
    KvAdaptive,
}

#[derive(Clone, Copy, Debug)]
pub struct ConfigField {
    pub key: &'static str,
    pub legacy_key: &'static str,
    pub category: ConfigCategory,
    pub scope: ConfigScope,
    pub default: DefaultValue,
    pub rule: ValueRule,
    pub registry_allowed: bool,
    pub experimental: bool,
    pub env_compat: Option<&'static str>,
    pub include_builtin_in_process_config: bool,
    pub help: &'static str,
}

impl ConfigField {
    pub fn validate(&self, value: &ConfigValue) -> Result<()> {
        let valid = match self.rule {
            ValueRule::Bool => matches!(value, ConfigValue::Bool(_)),
            ValueRule::Integer { min, max } => {
                matches!(value, ConfigValue::Integer(v) if *v >= min && *v <= max)
            }
            ValueRule::Float {
                min,
                max,
                min_inclusive,
            } => value
                .as_f64()
                .is_some_and(|v| (if min_inclusive { v >= min } else { v > min }) && v <= max),
            ValueRule::String => matches!(value, ConfigValue::String(_)),
            ValueRule::NonEmptyString => {
                matches!(value, ConfigValue::String(v) if !v.trim().is_empty())
            }
            ValueRule::Host => matches!(value, ConfigValue::String(v)
                if !v.is_empty() && v.len() <= 255 && v.trim() == v && !v.chars().any(char::is_whitespace)),
            ValueRule::PathOrEmpty => matches!(value, ConfigValue::String(v) if {
                if v.is_empty() {
                    true
                } else {
                    expand_tilde(v).is_file()
                }
            }),
            ValueRule::Enum(values) => {
                matches!(value, ConfigValue::String(v) if values.contains(&v.as_str()))
            }
            ValueRule::AutoBool => {
                matches!(value, ConfigValue::Bool(_))
                    || matches!(value, ConfigValue::String(v) if v == "auto")
            }
            ValueRule::NullableString => {
                matches!(value, ConfigValue::Null | ConfigValue::String(_))
            }
            ValueRule::NullableEnum(values) => {
                matches!(value, ConfigValue::Null)
                    || matches!(value, ConfigValue::String(v) if values.contains(&v.as_str()))
            }
            ValueRule::NullableInteger { min, max } => {
                matches!(value, ConfigValue::Null)
                    || matches!(value, ConfigValue::Integer(v) if *v >= min && *v <= max)
            }
            ValueRule::NullableFloat { min, max } => {
                matches!(value, ConfigValue::Null)
                    || value.as_f64().is_some_and(|v| v >= min && v <= max)
            }
            ValueRule::KvAdaptive => matches!(value, ConfigValue::String(v) if {
                matches!(v.as_str(), "off" | "conservative" | "balanced" | "aggressive")
                    || valid_advanced_kv(v)
            }),
        };

        if valid {
            Ok(())
        } else {
            Err(ConfigError::InvalidValue {
                key: self.key.to_owned(),
                message: format!("{} does not satisfy {:?}", value.kind(), self.rule),
            })
        }
    }

    pub fn parse_cli(&self, raw: &str) -> Result<ConfigValue> {
        let value = match self.rule {
            ValueRule::Bool => parse_bool(raw).map(ConfigValue::Bool),
            ValueRule::Integer { .. } => raw.parse::<i64>().ok().map(ConfigValue::Integer),
            ValueRule::Float { .. } => raw.parse::<f64>().ok().map(ConfigValue::Float),
            ValueRule::AutoBool if raw == "auto" => Some(ConfigValue::String(raw.to_owned())),
            ValueRule::AutoBool => parse_bool(raw).map(ConfigValue::Bool),
            ValueRule::NullableString if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableEnum(_) if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableInteger { .. } if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableInteger { .. } => raw.parse::<i64>().ok().map(ConfigValue::Integer),
            ValueRule::NullableFloat { .. } if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableFloat { .. } => raw.parse::<f64>().ok().map(ConfigValue::Float),
            _ => Some(ConfigValue::String(raw.to_owned())),
        }
        .ok_or_else(|| ConfigError::InvalidValue {
            key: self.key.to_owned(),
            message: format!("could not parse '{raw}'"),
        })?;
        self.validate(&value)?;
        Ok(value)
    }
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn valid_advanced_kv(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("advanced:k=") else {
        return false;
    };
    let Some((k, v)) = rest.split_once(",v=") else {
        return false;
    };
    matches!(k, "fwht4" | "fwht3" | "fwht2") && matches!(v, "lloyd4" | "lloyd3" | "lloyd2")
}

fn expand_tilde(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/") {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(value)
}

const KV_MODES: &[&str] = &[
    "auto", "q8", "asym4", "asym3", "asym2", "fwht4", "fwht3", "fwht2", "turbo", "turbo4",
    "turbo3", "turbo2",
];
const AUTO_ON_OFF: &[&str] = &["auto", "on", "off"];
// `off` disables thinking outright. It resolves to a cap of 1, the engine's
// established "no thinking" sentinel (the daemon reads
// `enable_thinking: max_think_tokens != 1`) and the same value the OpenAI
// `enable_thinking=false` / `reasoning_effort="none"` paths already send. It is
// NOT 0 — 0 means `uncapped` (think until the model closes the block itself).
const THINKING_BUDGETS: &[&str] = &["off", "low", "med", "high", "xhigh", "max", "uncapped"];
const REASONING_EFFORTS: &[&str] = &["auto", "none", "low", "high", "max"];
const SPECULATION_MODES: &[&str] = &["off", "auto", "ngram", "dflash", "mtp", "dspark"];

macro_rules! field {
    ($key:literal, $legacy:literal, $category:ident, $scope:ident, $default:expr, $rule:expr, $registry:expr, $experimental:expr, $env:expr, $help:literal) => {
        ConfigField {
            key: $key,
            legacy_key: $legacy,
            category: ConfigCategory::$category,
            scope: ConfigScope::$scope,
            default: $default,
            rule: $rule,
            registry_allowed: $registry,
            experimental: $experimental,
            env_compat: $env,
            include_builtin_in_process_config: true,
            help: $help,
        }
    };
}

macro_rules! bridge_field {
    ($key:literal, $legacy:literal, $category:ident, $scope:ident, $default:expr, $rule:expr, $experimental:expr, $env:literal, $help:literal) => {
        ConfigField {
            key: $key,
            legacy_key: $legacy,
            category: ConfigCategory::$category,
            scope: ConfigScope::$scope,
            default: $default,
            rule: $rule,
            registry_allowed: false,
            experimental: $experimental,
            env_compat: Some($env),
            include_builtin_in_process_config: false,
            help: $help,
        }
    };
}

// These fields preserve legacy environment spellings while exposing stable
// TOML policy to the typed ProcessConfig handoff consumed by hipfire-runtime
// and rdna-compute. They are process-scoped because those crates snapshot the
// values once; advertising per-model overrides would be dishonest for a
// long-lived serve process.
macro_rules! process_bool_field {
    ($key:literal, $legacy:literal, $category:ident, $default:expr, $experimental:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            $category,
            Process,
            DefaultValue::Bool($default),
            ValueRule::Bool,
            $experimental,
            $env,
            $help
        )
    };
}

macro_rules! process_auto_bool_field {
    ($key:literal, $legacy:literal, $category:ident, $experimental:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            $category,
            Process,
            DefaultValue::String("auto"),
            ValueRule::AutoBool,
            $experimental,
            $env,
            $help
        )
    };
}

macro_rules! diagnostic_bool_field {
    ($key:literal, $legacy:literal, $default:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            Diagnostic,
            Diagnostic,
            DefaultValue::Bool($default),
            ValueRule::Bool,
            true,
            $env,
            $help
        )
    };
}

macro_rules! process_field {
    ($key:literal, $legacy:literal, $category:ident, $default:expr, $rule:expr, $experimental:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            $category,
            Process,
            $default,
            $rule,
            $experimental,
            $env,
            $help
        )
    };
}

macro_rules! diagnostic_field {
    ($key:literal, $legacy:literal, $default:expr, $rule:expr, $env:literal, $help:literal) => {
        bridge_field!($key, $legacy, Diagnostic, Diagnostic, $default, $rule, true, $env, $help)
    };
}

/// The public schema. Canonical dotted keys are the TOML surface; `legacy_key`
/// accepts the current flat JSON/CLI spelling during migration.
pub static FIELDS: &[ConfigField] = &[
    field!(
        "memory.kv_cache",
        "kv_cache",
        Memory,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(KV_MODES),
        true,
        false,
        Some("HIPFIRE_KV_MODE"),
        "KV cache format; auto inherits the registry recommendation, then q8."
    ),
    field!(
        "memory.kv_adaptive",
        "kv_adaptive",
        Memory,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::KvAdaptive,
        true,
        false,
        Some("HIPFIRE_KV_ADAPTIVE"),
        "Runtime VRAM-fit KV precision policy."
    ),
    field!(
        "model.deepseek4_experts_per_token",
        "deepseek4_experts_per_token",
        Experimental,
        ModelLoad,
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 1, max: 6 },
        true,
        false,
        None,
        "DeepSeek V4 routed experts per token; null preserves the checkpoint default."
    ),
    field!(
        "attention.flash",
        "flash_mode",
        Attention,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "always", "never"]),
        true,
        false,
        Some("HIPFIRE_ATTN_FLASH"),
        "Flash-attention admission preference."
    ),
    field!(
        "serve.default_model",
        "default_model",
        Serve,
        Process,
        DefaultValue::String("qwen3.5:9b"),
        ValueRule::NonEmptyString,
        false,
        false,
        Some("HIPFIRE_MODEL"),
        "Model pre-warmed by serve."
    ),
    process_bool_field!(
        "serve.local",
        "local",
        Serve,
        false,
        false,
        "HIPFIRE_LOCAL",
        "Force the current command to use a locally spawned daemon."
    ),
    field!(
        "generation.temperature",
        "temperature",
        Generation,
        Request,
        DefaultValue::Float(0.3),
        ValueRule::Float {
            min: 0.0,
            max: 2.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "Sampling temperature."
    ),
    field!(
        "generation.top_p",
        "top_p",
        Generation,
        Request,
        DefaultValue::Float(0.8),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: false
        },
        true,
        false,
        None,
        "Nucleus sampling probability."
    ),
    field!(
        "generation.top_k",
        "top_k",
        Generation,
        Request,
        DefaultValue::Integer(20),
        ValueRule::Integer {
            min: 1,
            max: 100000
        },
        true,
        false,
        None,
        "Maximum number of highest-probability tokens retained for sampling."
    ),
    field!(
        "generation.min_p",
        "min_p",
        Generation,
        Request,
        DefaultValue::Float(0.0),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "Minimum probability relative to the highest-probability token; zero disables."
    ),
    field!(
        "generation.presence_penalty",
        "presence_penalty",
        Generation,
        Request,
        DefaultValue::Float(0.0),
        ValueRule::Float {
            min: 0.0,
            max: 2.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "OpenAI-style flat penalty for tokens already present in the repeat window."
    ),
    field!(
        "generation.repeat_penalty",
        "repeat_penalty",
        Generation,
        Request,
        DefaultValue::Float(1.05),
        ValueRule::Float {
            min: 1.0,
            max: 3.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "Token repetition penalty."
    ),
    field!(
        "generation.max_tokens",
        "max_tokens",
        Generation,
        Request,
        DefaultValue::Integer(4096),
        ValueRule::Integer {
            min: 1,
            max: 393216
        },
        true,
        false,
        None,
        "Per-turn generation cap."
    ),
    field!(
        "memory.max_seq",
        "max_seq",
        Memory,
        ModelLoad,
        DefaultValue::Integer(32768),
        ValueRule::Integer {
            min: 512,
            max: 1048576
        },
        true,
        false,
        None,
        "Logical KV context capacity."
    ),
    field!(
        "reasoning.mode",
        "thinking",
        Reasoning,
        Session,
        DefaultValue::String("on"),
        ValueRule::Enum(&["on", "off"]),
        true,
        false,
        None,
        "Visible reasoning mode."
    ),
    field!(
        "reasoning.effort",
        "reasoning_effort",
        Reasoning,
        Request,
        DefaultValue::String("auto"),
        ValueRule::Enum(REASONING_EFFORTS),
        true,
        false,
        None,
        "Model-specific reasoning effort; auto preserves the carrier fallback."
    ),
    field!(
        "reasoning.budget",
        "thinking_budget",
        Reasoning,
        Request,
        DefaultValue::String("med"),
        ValueRule::Enum(THINKING_BUDGETS),
        true,
        false,
        None,
        "Named per-turn reasoning budget."
    ),
    field!(
        "reasoning.max_tokens",
        "max_think_tokens",
        Reasoning,
        Request,
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: 393216
        },
        true,
        false,
        None,
        "Raw reasoning-token override; absent means use the named budget."
    ),
    field!(
        "reasoning.max_total_tokens",
        "max_total_think_tokens",
        Reasoning,
        Request,
        DefaultValue::Integer(0),
        ValueRule::Integer {
            min: 0,
            max: 1000000
        },
        true,
        false,
        Some("HIPFIRE_MAX_TOTAL_THINK_TOKENS"),
        "Cross-reopen reasoning budget; zero disables it."
    ),
    field!(
        "serve.host",
        "host",
        Serve,
        Process,
        DefaultValue::String("0.0.0.0"),
        ValueRule::Host,
        false,
        false,
        Some("HIPFIRE_HOST"),
        "Serve bind address."
    ),
    field!(
        "serve.port",
        "port",
        Serve,
        Process,
        DefaultValue::Integer(11435),
        ValueRule::Integer { min: 1, max: 65535 },
        false,
        false,
        Some("HIPFIRE_PORT"),
        "Serve TCP port."
    ),
    field!(
        "serve.idle_timeout_seconds",
        "idle_timeout",
        Serve,
        Process,
        DefaultValue::Integer(300),
        ValueRule::Integer { min: 0, max: 86400 },
        false,
        false,
        Some("HIPFIRE_IDLE_TIMEOUT"),
        "Seconds before idle model unload; zero disables."
    ),
    field!(
        "serve.max_request_bytes",
        "max_request_bytes",
        Serve,
        Process,
        DefaultValue::Integer(67108864),
        ValueRule::Integer {
            min: 4096,
            max: 4294967296
        },
        false,
        false,
        Some("HIPFIRE_MAX_REQUEST_BYTES"),
        "Maximum request-body bytes."
    ),
    field!(
        "serve.max_queue",
        "serve_max_queue",
        Serve,
        Process,
        DefaultValue::Integer(64),
        ValueRule::Integer {
            min: 0,
            max: 100000
        },
        false,
        false,
        Some("HIPFIRE_SERVE_MAX_QUEUE"),
        "Maximum queued serve requests; zero is uncapped."
    ),
    field!(
        "serve.queue_timeout_ms",
        "serve_queue_timeout_ms",
        Serve,
        Process,
        DefaultValue::Integer(30000),
        ValueRule::Integer {
            min: 0,
            max: 3600000
        },
        false,
        false,
        Some("HIPFIRE_SERVE_QUEUE_TIMEOUT_MS"),
        "Maximum admission-queue wait."
    ),
    process_bool_field!(
        "serve.retry_enabled",
        "retry_enabled",
        Serve,
        false,
        true,
        "HIPFIRE_SERVE_RETRY_ENABLED",
        "Server-owned single retry on typed transient daemon failures; promoted only after merged-path GPU parity."
    ),
    process_field!(
        "serve.retry_backoff_ms",
        "retry_backoff_ms",
        Serve,
        DefaultValue::Integer(50),
        ValueRule::Integer {
            min: 0,
            max: 60000
        },
        true,
        "HIPFIRE_SERVE_RETRY_BACKOFF_MS",
        "Backoff before the single serve retry; slept outside runtime and admission locks."
    ),
    field!(
        "experimental.budget_alert",
        "experimental_budget_alert",
        Experimental,
        Request,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        false,
        true,
        Some("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT"),
        "Research-only in-band reasoning alert gate."
    ),
    field!(
        "speculation.dflash_adaptive_b",
        "dflash_adaptive_b",
        Speculation,
        ModelLoad,
        DefaultValue::Bool(true),
        ValueRule::Bool,
        true,
        false,
        None,
        "Adapt the DFlash block size to observed acceptance."
    ),
    field!(
        "speculation.dflash",
        "dflash_mode",
        Speculation,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::Enum(AUTO_ON_OFF),
        true,
        false,
        Some("HIPFIRE_DFLASH_MODE"),
        "DFlash eligibility policy."
    ),
    field!(
        "speculation.dflash_ngram_block",
        "dflash_ngram_block",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::AutoBool,
        true,
        false,
        Some("HIPFIRE_DFLASH_NGRAM_BLOCK"),
        "Verify-path n-gram defense."
    ),
    field!(
        "memory.cask.sidecar",
        "cask_sidecar",
        Memory,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::String,
        true,
        false,
        Some("HIPFIRE_CASK_SIDECAR"),
        "TriAttention sidecar path; empty disables eviction."
    ),
    field!(
        "memory.cask.enabled",
        "cask",
        Memory,
        ModelLoad,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        true,
        false,
        None,
        "Enable core-aware CASK folding."
    ),
    field!(
        "memory.cask.budget",
        "cask_budget",
        Memory,
        ModelLoad,
        DefaultValue::Integer(512),
        ValueRule::Integer {
            min: 64,
            max: 65536
        },
        true,
        false,
        None,
        "Active-token target after eviction."
    ),
    field!(
        "memory.cask.beta",
        "cask_beta",
        Memory,
        ModelLoad,
        DefaultValue::Integer(128),
        ValueRule::Integer { min: 0, max: 65536 },
        true,
        false,
        None,
        "Eviction hysteresis."
    ),
    field!(
        "memory.cask.core_fraction",
        "cask_core_frac",
        Memory,
        ModelLoad,
        DefaultValue::Float(0.5),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "Fraction of the CASK budget retained as core."
    ),
    field!(
        "memory.cask.fold",
        "cask_fold_m",
        Memory,
        ModelLoad,
        DefaultValue::Integer(2),
        ValueRule::Integer { min: 1, max: 16 },
        true,
        false,
        None,
        "CASK merge factor."
    ),
    field!(
        "memory.cask.auto_attach",
        "cask_auto_attach",
        Memory,
        ModelLoad,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        true,
        false,
        None,
        "Discover a matching TriAttention sidecar when explicitly enabled."
    ),
    field!(
        "prompt.normalize",
        "prompt_normalize",
        Prompt,
        Request,
        DefaultValue::Bool(true),
        ValueRule::Bool,
        true,
        false,
        Some("HIPFIRE_NORMALIZE_PROMPT"),
        "Collapse runs of three or more newlines before tokenization."
    ),
    field!(
        "prompt.system",
        "system_prompt",
        Prompt,
        Request,
        DefaultValue::String(""),
        ValueRule::String,
        true,
        false,
        None,
        "Default system prompt used only when a request supplies no system or developer message."
    ),
    field!(
        "memory.mmq.screen",
        "mmq_screen",
        Memory,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(AUTO_ON_OFF),
        true,
        false,
        Some("HIPFIRE_MMQ_SCREEN"),
        "Per-weight MMQ correctness screening policy."
    ),
    field!(
        "memory.mmq.screen_threshold",
        "mmq_screen_threshold",
        Memory,
        ModelLoad,
        DefaultValue::Float(0.1),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: false
        },
        true,
        false,
        Some("HIPFIRE_MMQ_SCREEN_THRESHOLD"),
        "MMQ screening absolute-error threshold."
    ),
    process_field!(
        "memory.prompt_cache_capacity",
        "prompt_cache_capacity",
        Memory,
        DefaultValue::Integer(32),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        false,
        "HIPFIRE_PROMPT_CACHE_CAP",
        "Maximum cached assistant-turn tokenizations; zero keeps no entries."
    ),
    process_bool_field!(
        "memory.prompt_cache_unbounded",
        "prompt_cache_unbounded",
        Memory,
        false,
        true,
        "HIPFIRE_PROMPT_CACHE_UNBOUNDED",
        "Remove the assistant-turn cache capacity bound."
    ),
    field!(
        "speculation.prefill.mode",
        "prefill_compression",
        Speculation,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::Enum(&["off", "auto", "always"]),
        true,
        true,
        None,
        "PFlash speculative-prefill policy."
    ),
    field!(
        "speculation.prefill.threshold",
        "prefill_threshold",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(32768),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        true,
        true,
        None,
        "PFlash auto-mode token threshold."
    ),
    field!(
        "speculation.prefill.keep_ratio",
        "prefill_keep_ratio",
        Speculation,
        ModelLoad,
        DefaultValue::Float(0.05),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: false
        },
        true,
        true,
        None,
        "PFlash retained-token ratio."
    ),
    field!(
        "speculation.prefill.alpha",
        "prefill_alpha",
        Speculation,
        ModelLoad,
        DefaultValue::Float(0.85),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: true
        },
        true,
        true,
        None,
        "PFlash block-selection strictness."
    ),
    field!(
        "speculation.prefill.min_keep",
        "prefill_min_keep",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(2048),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        true,
        true,
        None,
        "PFlash retained-token floor."
    ),
    field!(
        "speculation.prefill.sink",
        "prefill_sink",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(256),
        ValueRule::Integer { min: 0, max: 65536 },
        true,
        true,
        None,
        "Always-retained prompt prefix."
    ),
    field!(
        "speculation.prefill.recent",
        "prefill_recent",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(1024),
        ValueRule::Integer { min: 0, max: 65536 },
        true,
        true,
        None,
        "Always-retained prompt tail."
    ),
    field!(
        "speculation.prefill.block",
        "prefill_block",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(128),
        ValueRule::Integer { min: 1, max: 4096 },
        true,
        true,
        None,
        "PFlash scoring block size."
    ),
    field!(
        "speculation.prefill.drafter",
        "prefill_drafter",
        Speculation,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::String,
        true,
        true,
        None,
        "PFlash drafter path."
    ),
    field!(
        "speculation.prefill.drafter_device",
        "prefill_drafter_device",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(-1),
        ValueRule::Integer { min: -1, max: 15 },
        true,
        true,
        None,
        "PFlash drafter device; -1 uses the target device."
    ),
    field!(
        "speculation.prefill.profile",
        "prefill_profile",
        Speculation,
        Diagnostic,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        false,
        true,
        None,
        "Emit PFlash stage timings."
    ),
    field!(
        "speculation.prefill.sparse_threshold",
        "prefill_sparse_threshold",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(32768),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        true,
        true,
        None,
        "Sparse-attention threshold."
    ),
    field!(
        "speculation.prefill.drafter_kv",
        "prefill_drafter_kv",
        Speculation,
        ModelLoad,
        DefaultValue::String("q8"),
        ValueRule::Enum(&["q8", "fwht4", "fwht3", "fwht2"]),
        true,
        true,
        Some("HIPFIRE_PFLASH_DRAFTER_KV"),
        "KV quantization used by the PFlash drafter scorer."
    ),
    diagnostic_field!(
        "diagnostic.pflash.score_layer",
        "pflash_score_layer",
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 0, max: 65535 },
        "HIPFIRE_PFLASH_SCORE_LAYER",
        "Override the PFlash scoring layer; null uses model policy."
    ),
    field!(
        "speculation.mtp",
        "mtp_mode",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(AUTO_ON_OFF),
        true,
        false,
        Some("HIPFIRE_MTP_MODE"),
        "MTP eligibility policy."
    ),
    field!(
        "speculation.mtp_k",
        "mtp_k",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(3),
        ValueRule::Integer { min: 1, max: 10 },
        true,
        false,
        Some("HIPFIRE_MTP_K"),
        "MTP draft window."
    ),
    field!(
        "speculation.mode",
        "speculation",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(SPECULATION_MODES),
        true,
        false,
        Some("HIPFIRE_SPECULATION"),
        "Canonical speculative-decoding selector."
    ),
    field!(
        "speculation.dspark_confidence",
        "dspark_conf_threshold",
        Speculation,
        ModelLoad,
        DefaultValue::Null,
        ValueRule::NullableFloat { min: 0.0, max: 1.0 },
        true,
        true,
        None,
        "DSpark confidence truncation threshold."
    ),
    field!(
        "speculation.ngram",
        "ngram_mode",
        Speculation,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::Enum(&["auto", "on", "off", "1", "0"]),
        true,
        false,
        Some("HIPFIRE_NGRAM_DRAFT"),
        "Model-free n-gram speculation policy."
    ),
    field!(
        "speculation.ngram_k",
        "ngram_k",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(12),
        ValueRule::Integer { min: 2, max: 32 },
        true,
        false,
        Some("HIPFIRE_NGRAM_DRAFT_K"),
        "N-gram draft window."
    ),
    field!(
        "speculation.ngram_min_count",
        "ngram_min_count",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(2),
        ValueRule::Integer { min: 1, max: 10 },
        true,
        false,
        Some("HIPFIRE_NGRAM_MIN_COUNT"),
        "Minimum n-gram match count."
    ),
    field!(
        "speculation.ddtree_budget",
        "ddtree_budget",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(0),
        ValueRule::Integer { min: 0, max: 64 },
        true,
        true,
        Some("HIPFIRE_DDTREE_BUDGET"),
        "DFlash verify-tree node budget; zero selects chain mode."
    ),
    field!(
        "speculation.ddtree_topk",
        "ddtree_topk",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(4),
        ValueRule::Integer { min: 1, max: 8 },
        true,
        true,
        Some("HIPFIRE_DDTREE_TOPK"),
        "DFlash verify-tree fanout."
    ),
    field!(
        "prompt.chat_template",
        "chat_template",
        Prompt,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::PathOrEmpty,
        true,
        false,
        Some("HIPFIRE_CHAT_TEMPLATE_FILE"),
        "Optional Jinja chat-template path."
    ),
    field!(
        "prompt.default_chatml",
        "default_chatml",
        Prompt,
        ModelLoad,
        DefaultValue::Bool(true),
        ValueRule::Bool,
        true,
        false,
        Some("HIPFIRE_DEFAULT_CHATML"),
        "Allow the fallback ChatML frame when no template resolves."
    ),
    process_field!(
        "hardware.devices",
        "devices",
        Hardware,
        DefaultValue::Null,
        ValueRule::NullableString,
        false,
        "HIPFIRE_DEVICES",
        "Physical GPU list lowered to ROCr selectors and matching HIP logical selectors before GPU initialization."
    ),
    process_field!(
        "hardware.uniform_vram_tolerance_gb",
        "uniform_vram_tolerance_gb",
        Hardware,
        DefaultValue::Null,
        ValueRule::NullableFloat {
            min: f64::MIN,
            max: f64::MAX
        },
        false,
        "HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB",
        "Allowed free-VRAM spread across a uniform multi-GPU topology."
    ),
    process_field!(
        "generation.loop_guard_threshold",
        "ngram_loop_threshold",
        Generation,
        DefaultValue::Integer(0),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        false,
        "HIPFIRE_NGRAM_LOOP_THRESHOLD",
        "Repeated 4-gram count that forces EOS; zero disables the guard."
    ),
    process_field!(
        "generation.loop_guard_window",
        "ngram_window",
        Generation,
        DefaultValue::Integer(256),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        false,
        "HIPFIRE_NGRAM_WINDOW",
        "Token window inspected by the repeated 4-gram loop guard."
    ),
    process_field!(
        "kernel.flash_partials_batch",
        "flash_partials_batch",
        Kernel,
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 0, max: 65536 },
        true,
        "HIPFIRE_FLASH_PARTIALS_BATCH",
        "Override the prefill flash-attention partial-scratch batch multiplier."
    ),
    process_field!(
        "kernel.lm_head_f16",
        "lm_head_f16",
        Kernel,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "native", "f16", "1", "f32", "fp32", "legacy", "0"]),
        false,
        "HIPFIRE_LM_HEAD_F16",
        "Storage policy for native FP16 LM-head weights."
    ),
    diagnostic_field!(
        "diagnostic.prompt_heat_limit",
        "prompt_heat_limit",
        DefaultValue::Integer(64),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_PROMPT_HEAT_LIMIT",
        "Maximum rows emitted by prompt token-heat diagnostics."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gemv_rows",
        "gemv_rows",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "4", "8"]),
        "HIPFIRE_GEMV_ROWS",
        "Override the architecture-selected GEMV rows per workgroup."
    ),
    diagnostic_field!(
        "diagnostic.kernel.fp16_layer_min",
        "fp16_layer_min",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_FP16_LAYER_MIN",
        "First layer included in the FP16 route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.fp16_layer_max",
        "fp16_layer_max",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_FP16_LAYER_MAX",
        "Last layer included in the FP16 route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.hfq3_mmq_layer_min",
        "hfq3_mmq_layer_min",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_HFQ3_MMQ_LAYER_MIN",
        "First layer included in the HFQ3 MMQ route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.hfq3_mmq_layer_max",
        "hfq3_mmq_layer_max",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_HFQ3_MMQ_LAYER_MAX",
        "Last layer included in the HFQ3 MMQ route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.mmq_min_batch",
        "mmq_min_batch",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_MMQ_MIN_BATCH",
        "Minimum batch size for MMQ dispatch."
    ),
    diagnostic_field!(
        "diagnostic.kernel.rocblas_min_batch",
        "rocblas_min_batch",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_ROCBLAS_MIN_BATCH",
        "Minimum batch size for rocBLAS dispatch."
    ),
    diagnostic_field!(
        "diagnostic.kernel.ddtree_logw_cutoff",
        "ddtree_logw_cutoff",
        DefaultValue::Null,
        ValueRule::NullableFloat {
            min: 0.0,
            max: f64::MAX
        },
        "HIPFIRE_DDTREE_LOGW_CUTOFF",
        "Positive DDTree cumulative-log-weight expansion cutoff."
    ),
    diagnostic_field!(
        "diagnostic.kernel.lloyd_mb4",
        "lloyd_mb4",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "4"]),
        "HIPFIRE_LLOYD_MB4",
        "Lloyd kernel rows packed per MB4 work item."
    ),
    diagnostic_field!(
        "diagnostic.kernel.mq3_mb4",
        "mq3_mb4",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "4"]),
        "HIPFIRE_MQ3_MB4",
        "MQ3 kernel rows packed per MB4 work item."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gate_up_variant",
        "gate_up_variant",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["ldsx", "k4", "ldscoop", "2tile"]),
        "HIPFIRE_GATE_UP_VARIANT",
        "Select a gate/up WMMA experiment variant."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gfx11_weight_load_policy",
        "gfx11_weight_load_policy",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["buffer", "global", "flat-buffer"]),
        "HIPFIRE_GFX11_WEIGHT_LOAD_POLICY",
        "Select the gfx11 compiler weight-load policy."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gfx12_weight_load_policy",
        "gfx12_weight_load_policy",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["rt", "global", "ht", "nt-rt", "nt-ht"]),
        "HIPFIRE_GFX12_WEIGHT_LOAD_POLICY",
        "Select the gfx12 compiler weight-load policy."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gfx942_mfma_prefill",
        "gfx942_mfma_prefill",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "3", "4"]),
        "HIPFIRE_GFX942_MFMA_PREFILL",
        "Select the gfx942 direct-MFMA prefill experiment."
    ),
    diagnostic_field!(
        "diagnostic.kernel.rdna2_variant",
        "rdna2_variant",
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 1, max: 5 },
        "HIPFIRE_RDNA2_VARIANT",
        "Select the gfx1030/gfx1031 HFQ4-G256 GEMV variant."
    ),
    diagnostic_field!(
        "diagnostic.kernel.wo_wmma_variant",
        "wo_wmma_variant",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["ksplit", "ksplit_det", "k2", "k2x32", "k4", "wmma", "wmma2"]),
        "HIPFIRE_WO_WMMA_VARIANT",
        "Select a weight-only residual WMMA variant."
    ),
    diagnostic_field!(
        "diagnostic.compiler.hipcc_extra_flags",
        "hipcc_extra_flags",
        DefaultValue::String(""),
        ValueRule::String,
        "HIPFIRE_HIPCC_EXTRA_FLAGS",
        "Append advanced local flags to HIP kernel compilation."
    ),
    process_bool_field!(
        "hardware.allow_mixed_arch",
        "allow_mixed_arch",
        Hardware,
        false,
        false,
        "HIPFIRE_ALLOW_MIXED_ARCH",
        "Allow a multi-GPU topology containing different GPU architectures."
    ),
    process_bool_field!(
        "hardware.tp_use_rccl",
        "tp_use_rccl",
        Hardware,
        true,
        false,
        "HIPFIRE_TP_USE_RCCL",
        "Use RCCL for tensor-parallel all-reduce."
    ),
    process_bool_field!(
        "kernel.prefill_batched",
        "prefill_batched",
        Kernel,
        true,
        false,
        "HIPFIRE_PREFILL_BATCHED",
        "Use batched prefill kernels when eligible."
    ),
    process_bool_field!(
        "speculation.draft_f16",
        "draft_f16",
        Speculation,
        true,
        false,
        "HIPFIRE_DRAFT_F16",
        "Keep DFlash draft activations in FP16."
    ),
    diagnostic_bool_field!(
        "diagnostic.prompt_token_heat",
        "prompt_token_heat",
        false,
        "HIPFIRE_PROMPT_TOKEN_HEAT",
        "Emit prompt token-heat diagnostics."
    ),
    diagnostic_bool_field!(
        "diagnostic.prompt_heat_json",
        "prompt_heat_json",
        false,
        "HIPFIRE_PROMPT_HEAT_JSON",
        "Render prompt token-heat diagnostics as JSON."
    ),
    diagnostic_bool_field!(
        "diagnostic.draft_gemm_dump",
        "draft_gemm_dump",
        false,
        "HIPFIRE_DRAFT_GEMM_DUMP",
        "Dump DFlash draft GEMM diagnostics."
    ),
    diagnostic_bool_field!(
        "diagnostic.draft_subphase",
        "draft_subphase",
        false,
        "HIPFIRE_DRAFT_SUBPHASE",
        "Emit DFlash draft subphase timings."
    ),
    process_auto_bool_field!(
        "kernel.gemv_dp4a",
        "gemv_dp4a",
        Kernel,
        false,
        "HIPFIRE_GEMV_DP4A",
        "Override the architecture-selected DP4A GEMV route."
    ),
    process_auto_bool_field!(
        "kernel.gemv_prefetch",
        "gemv_prefetch",
        Kernel,
        false,
        "HIPFIRE_GEMV_PREFETCH",
        "Override the architecture-selected GEMV prefetch route."
    ),
    process_auto_bool_field!(
        "kernel.gfx942_lds_gemv",
        "gfx942_lds_gemv",
        Kernel,
        true,
        "HIPFIRE_GFX942_LDS_GEMV",
        "Override the gfx942 LDS GEMV experiment."
    ),
    process_auto_bool_field!(
        "kernel.hfq3_dp4a",
        "hfq3_dp4a",
        Kernel,
        true,
        "HIPFIRE_HFQ3_DP4A",
        "Override HFQ3 DP4A dispatch."
    ),
    process_auto_bool_field!(
        "kernel.hfq3_mmq",
        "hfq3_mmq",
        Kernel,
        true,
        "HIPFIRE_HFQ3_MMQ",
        "Override HFQ3 MMQ dispatch."
    ),
    process_auto_bool_field!(
        "kernel.hfq4_mmq_rdna2",
        "hfq4_mmq_rdna2",
        Kernel,
        true,
        "HIPFIRE_HFQ4_MMQ_RDNA2",
        "Override HFQ4 MMQ dispatch on RDNA2."
    ),
    process_auto_bool_field!(
        "kernel.gcn5_wave64_hybrid",
        "gcn5_wave64_hybrid",
        Kernel,
        true,
        "HIPFIRE_GCN5_WAVE64_HYBRID",
        "Override the GCN5 wave64 hybrid route."
    ),
    process_auto_bool_field!(
        "kernel.mmq",
        "mmq",
        Kernel,
        false,
        "HIPFIRE_MMQ",
        "Override architecture-selected MMQ dispatch."
    ),
    process_auto_bool_field!(
        "kernel.gfx942_gemv_v2",
        "gfx942_gemv_v2",
        Kernel,
        true,
        "HIPFIRE_GFX942_GEMV_V2",
        "Override gfx942 GEMV v2 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.moe_grouped_i8",
        "moe_grouped_i8",
        Kernel,
        true,
        "HIPFIRE_MOE_GROUPED_I8",
        "Override grouped MoE i8 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.moe_paro_i8",
        "moe_paro_i8",
        Kernel,
        true,
        "HIPFIRE_MOE_PARO_I8",
        "Override architecture-selected Paro grouped-GEMM i8 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.moe_paro_i8_k8",
        "moe_paro_i8_k8",
        Kernel,
        true,
        "HIPFIRE_MOE_PARO_I8_K8",
        "Override architecture-selected Paro grouped-GEMM i8 K8 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_hfq4_qkvza_k2048",
        "rdna3_hfq4_qkvza_k2048",
        Kernel,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_K2048",
        "Override the certified gfx1100 fixed-K QKVZA kernel."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_hfq4_residual_stage_x32",
        "rdna3_hfq4_residual_stage_x32",
        Kernel,
        true,
        "HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32",
        "Override the certified gfx1100 residual activation-staging kernel."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_hfq4_sigmoid_buffer",
        "rdna3_hfq4_sigmoid_buffer",
        Kernel,
        true,
        "HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER",
        "Override the certified gfx1100 sigmoid buffer-load kernel."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_rmsnorm_vecsum",
        "rdna3_rmsnorm_vecsum",
        Kernel,
        true,
        "HIPFIRE_RDNA3_RMSNORM_VECSUM",
        "Override the certified gfx1100 RMSNorm vecsum kernel."
    ),
    process_bool_field!(
        "kernel.fp8_wmma",
        "fp8_wmma",
        Kernel,
        false,
        true,
        "HIPFIRE_FP8_WMMA",
        "Enable the experimental FP8 WMMA route."
    ),
    process_bool_field!(
        "kernel.dot2_gemv",
        "dot2_gemv",
        Kernel,
        false,
        true,
        "HIPFIRE_DOT2_GEMV",
        "Enable dot2 GEMV dispatch."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkv_wave64",
        "rdna3_hfq4_qkv_wave64",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKV_WAVE64",
        "Enable the RDNA3 wave64 QKV experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_2wave",
        "rdna3_hfq4_qkvza_2wave",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_2WAVE",
        "Enable the RDNA3 two-wave QKVZA experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_wavepack4",
        "rdna3_hfq4_qkvza_wavepack4",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_WAVEPACK4",
        "Enable the RDNA3 four-wave QKVZA packing experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_ldsx8",
        "rdna3_hfq4_qkvza_ldsx8",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_LDSX8",
        "Enable the RDNA3 LDS-staged QKVZA experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_reduce_chain",
        "rdna3_hfq4_qkvza_reduce_chain",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_REDUCE_CHAIN",
        "Enable the RDNA3 explicit QKVZA reduction chain."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_hoist_x32",
        "rdna3_hfq4_qkvza_hoist_x32",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_HOIST_X32",
        "Enable the RDNA3 QKVZA activation-hoist experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_residual_k2048",
        "rdna3_hfq4_residual_k2048",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_RESIDUAL_K2048",
        "Enable the RDNA3 fixed-K residual experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_sigmoid_tight_grid",
        "rdna3_hfq4_sigmoid_tight_grid",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_SIGMOID_TIGHT_GRID",
        "Enable the RDNA3 tight-grid sigmoid kernel."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_sigmoid_rows4",
        "rdna3_hfq4_sigmoid_rows4",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_SIGMOID_ROWS4",
        "Enable the RDNA3 four-row sigmoid kernel."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_lm_head_k2048",
        "rdna3_hfq4_lm_head_k2048",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_LM_HEAD_K2048",
        "Enable the RDNA3 fixed-K LM-head experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_moe_gate_up_k2048",
        "rdna3_hfq4_moe_gate_up_k2048",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_MOE_GATE_UP_K2048",
        "Enable the RDNA3 fixed-K MoE gate/up experiment."
    ),
    process_bool_field!(
        "kernel.fp16",
        "fp16",
        Kernel,
        true,
        false,
        "HIPFIRE_FP16",
        "Allow FP16 kernel routes."
    ),
    process_bool_field!(
        "kernel.wo_mmq",
        "wo_mmq",
        Kernel,
        false,
        true,
        "HIPFIRE_WO_MMQ",
        "Enable weight-only MMQ dispatch."
    ),
    process_bool_field!(
        "kernel.lm_head_wmma",
        "lm_head_wmma",
        Kernel,
        true,
        false,
        "HIPFIRE_LM_HEAD_WMMA",
        "Allow WMMA LM-head dispatch."
    ),
    process_bool_field!(
        "kernel.lm_head_overwrite",
        "lm_head_overwrite",
        Kernel,
        false,
        true,
        "HIPFIRE_LM_HEAD_OVERWRITE",
        "Enable the LM-head overwrite experiment."
    ),
    diagnostic_bool_field!(
        "diagnostic.mmq_quantize_only",
        "mmq_diag_quantize_only",
        false,
        "HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY",
        "Stop MMQ screening after quantization diagnostics."
    ),
    process_bool_field!(
        "kernel.hfq4g128_mmq",
        "hfq4g128_mmq",
        Kernel,
        true,
        false,
        "HIPFIRE_HFQ4G128_MMQ",
        "Allow HFQ4G128 MMQ dispatch."
    ),
    process_bool_field!(
        "kernel.hfq4_mmq_gfx906_y64",
        "hfq4_mmq_gfx906_y64",
        Kernel,
        false,
        true,
        "HIPFIRE_HFQ4_MMQ_GFX906_Y64",
        "Enable the gfx906 HFQ4 MMQ Y64 experiment."
    ),
    process_bool_field!(
        "kernel.gate_up_nosync",
        "gate_up_nosync",
        Kernel,
        false,
        true,
        "HIPFIRE_GATE_UP_NOSYNC",
        "Enable the no-sync gate/up experiment."
    ),
    process_bool_field!(
        "kernel.qkvza_split_tail",
        "qkvza_split_tail",
        Kernel,
        false,
        true,
        "HIPFIRE_QKVZA_SPLIT_TAIL",
        "Enable the RDNA3 QKVZA split-tail prefill route."
    ),
    process_bool_field!(
        "kernel.gfx942_gemv_v3",
        "gfx942_gemv_v3",
        Kernel,
        false,
        true,
        "HIPFIRE_GFX942_GEMV_V3",
        "Enable gfx942 GEMV v3 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.gfx942_rmsnorm_split",
        "gfx942_rmsnorm_split",
        Kernel,
        true,
        "HIPFIRE_GFX942_RMSNORM_SPLIT",
        "Override the architecture-selected gfx942 split RMSNorm route."
    ),
    process_bool_field!(
        "kernel.rmsnorm_mq_tight_lds",
        "rmsnorm_mq_tight_lds",
        Kernel,
        false,
        true,
        "HIPFIRE_RMSNORM_MQ_TIGHT_LDS",
        "Enable the tight-LDS fused RMSNorm MQ route."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_wavegrid",
        "rdna3_rmsnorm_wavegrid",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_WAVEGRID",
        "Enable the RDNA3 wave-grid RMSNorm experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_split",
        "rdna3_rmsnorm_split",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_SPLIT",
        "Enable the RDNA3 split RMSNorm experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_sign_lds",
        "rdna3_rmsnorm_sign_lds",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_SIGN_LDS",
        "Enable LDS-staged MQ sign tables for RDNA3 RMSNorm."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_sign_const",
        "rdna3_rmsnorm_sign_const",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_SIGN_CONST",
        "Enable packed-constant MQ signs for RDNA3 RMSNorm."
    ),
    process_bool_field!(
        "kernel.moe_grouped_i8_k8",
        "moe_grouped_i8_k8",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_I8_K8",
        "Enable grouped MoE i8 K8 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_i8_k4",
        "moe_grouped_i8_k4",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_I8_K4",
        "Enable grouped MoE i8 K4 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_i8_k4_gfx12",
        "moe_grouped_i8_k4_gfx12",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_I8_K4_GFX12",
        "Enable grouped MoE i8 K4 dispatch on gfx12."
    ),
    process_bool_field!(
        "kernel.moe_grouped_m2",
        "moe_grouped_m2",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_M2",
        "Enable grouped two-row MoE dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_4w",
        "moe_grouped_4w",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_4W",
        "Enable grouped four-wave MoE dispatch."
    ),
    process_bool_field!(
        "kernel.moe_down_combine_vec4",
        "moe_down_combine_vec4",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_DOWN_COMBINE_VEC4",
        "Enable four-row MoE down-projection combining."
    ),
    process_bool_field!(
        "kernel.moe_hfq6_i8",
        "moe_hfq6_i8",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_HFQ6_I8",
        "Enable HFQ6 grouped MoE i8 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_hfq6_v2",
        "moe_hfq6_v2",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_HFQ6_V2",
        "Enable HFQ6 grouped MoE v2 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_gemm",
        "moe_grouped_gemm",
        Kernel,
        true,
        false,
        "HIPFIRE_MOE_GROUPED_GEMM",
        "Allow grouped-GEMM MoE prefill."
    ),
    diagnostic_bool_field!(
        "diagnostic.blob_force",
        "blob_force",
        false,
        "HIPFIRE_BLOB_FORCE",
        "Force the retained kernarg-blob launch path."
    ),
    diagnostic_bool_field!(
        "diagnostic.gemm_dump",
        "gemm_dump",
        false,
        "HIPFIRE_GEMM_DUMP",
        "Dump GEMM routing diagnostics."
    ),
    process_auto_bool_field!(
        "experimental.graph.forward",
        "graph_forward",
        Experimental,
        true,
        "HIPFIRE_GRAPH",
        "Enable hipGraph forward capture; auto follows architecture policy."
    ),
    process_bool_field!(
        "experimental.graph.ar",
        "graph_ar",
        Experimental,
        true,
        true,
        "HIPFIRE_AR_GRAPH",
        "Allow autoregressive forward graph capture when otherwise eligible."
    ),
    process_bool_field!(
        "experimental.graph.moe",
        "graph_moe",
        Experimental,
        true,
        true,
        "HIPFIRE_GRAPH_MOE",
        "Allow graph capture for supported MoE forward paths."
    ),
    process_bool_field!(
        "kernel.deterministic",
        "deterministic",
        Kernel,
        false,
        true,
        "HIPFIRE_DETERMINISTIC",
        "Select deterministic kernel variants where available."
    ),
    process_bool_field!(
        "kernel.mw16",
        "mw16",
        Kernel,
        false,
        true,
        "HIPFIRE_MW16",
        "Enable the MW16 kernel experiment."
    ),
    process_bool_field!(
        "kernel.q8_batched_legacy",
        "q8_batched_legacy",
        Kernel,
        false,
        true,
        "HIPFIRE_Q8_BATCHED_LEGACY",
        "Use the legacy batched Q8 route."
    ),
    process_bool_field!(
        "kernel.deepseek4_q8_wmma",
        "deepseek4_q8_wmma",
        Kernel,
        true,
        false,
        "HIPFIRE_DEEPSEEK4_Q8_WMMA",
        "Allow DeepSeek4 Q8 WMMA prefill."
    ),
    process_bool_field!(
        "kernel.deepseek4_q8_4w",
        "deepseek4_q8_4w",
        Kernel,
        true,
        false,
        "HIPFIRE_DEEPSEEK4_Q8_4W",
        "Allow the DeepSeek4 four-wave Q8 WMMA tile."
    ),
    process_bool_field!(
        "kernel.rope_interleaved_legacy",
        "rope_interleaved_legacy",
        Kernel,
        false,
        true,
        "HIPFIRE_ROPE_INTERLEAVED_LEGACY",
        "Use the legacy interleaved RoPE route."
    ),
    process_bool_field!(
        "kernel.rocblas_all_archs",
        "rocblas_all_archs",
        Kernel,
        false,
        true,
        "HIPFIRE_ROCBLAS_ALL_ARCHS",
        "Allow rocBLAS dispatch on all architectures."
    ),
    process_bool_field!(
        "kernel.rocblas_off",
        "rocblas_off",
        Kernel,
        false,
        false,
        "HIPFIRE_ROCBLAS_OFF",
        "Disable rocBLAS dispatch."
    ),
    process_bool_field!(
        "kernel.lloyd_force_baseline",
        "lloyd_force_baseline",
        Kernel,
        false,
        true,
        "HIPFIRE_LLOYD_FORCE_BASELINE",
        "Force the Lloyd baseline kernel route."
    ),
    process_bool_field!(
        "fusions.force_unfused",
        "force_unfused",
        Fusions,
        false,
        true,
        "HIPFIRE_FORCE_UNFUSED",
        "Force supported projection paths to remain unfused."
    ),
    process_bool_field!(
        "speculation.dflash_tree",
        "dflash_tree",
        Speculation,
        false,
        true,
        "HIPFIRE_DFLASH_TREE",
        "Enable DDTree tree-SWOR verification."
    ),
    process_bool_field!(
        "speculation.ddtree_tree_la",
        "ddtree_tree_la",
        Speculation,
        true,
        true,
        "HIPFIRE_DDTREE_TREE_LA",
        "Allow DDTree linearized-ancestor tape replay."
    ),
    process_bool_field!(
        "speculation.dflash_fast_sample",
        "dflash_fast_sample",
        Speculation,
        true,
        true,
        "HIPFIRE_DFLASH_FAST_SAMPLE",
        "Allow DFlash GPU sampling on sampled verification."
    ),
    process_bool_field!(
        "speculation.dflash_q8_lmhead_wmma",
        "dflash_q8_lmhead_wmma",
        Speculation,
        true,
        true,
        "HIPFIRE_DFLASH_Q8_LMHEAD_WMMA",
        "Allow Q8 WMMA LM-head dispatch during DFlash verification."
    ),
    process_bool_field!(
        "fusions.qkv_bias",
        "fuse_qkv_bias",
        Fusions,
        true,
        false,
        "HIPFIRE_FUSE_QKV_BIAS",
        "Fold supported QKV bias additions into fused QKV decode."
    ),
    diagnostic_bool_field!(
        "diagnostic.qkv_bias",
        "fuse_qkv_bias_debug",
        false,
        "HIPFIRE_FUSE_QKV_BIAS_DEBUG",
        "Log each QKV-bias fold."
    ),
    ConfigField {
        key: "replay.backend",
        legacy_key: "replay_backend",
        category: ConfigCategory::Replay,
        scope: ConfigScope::Process,
        default: DefaultValue::String("auto"),
        rule: ValueRule::Enum(&["auto", "hip", "redline", "shadow", "off"]),
        registry_allowed: false,
        experimental: false,
        env_compat: Some("HIPFIRE_REPLAY_BACKEND"),
        include_builtin_in_process_config: false,
        help: "Preferred launch backend; runtime default selection is distinct from certification/admission.",
    },
    process_field!(
        "replay.transport",
        "replay_transport",
        Replay,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "aql", "pm4", "pm4_ib", "ib"]),
        true,
        "HIPFIRE_REPLAY_TRANSPORT",
        "Retained replay transport; auto follows the runtime default route predicate, not certification/admission."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.route_proof_log",
        "replay_route_proof_log",
        false,
        "HIPFIRE_REPLAY_ROUTE_PROOF_LOG",
        "When enabled, the daemon emits one post-generate retained-route proof marker per successful request (fields: transport, position, request_id, replays) so product coherence can prove route identity without enabling capture."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.manual_capture",
        "replay_manual_capture",
        false,
        "HIPFIRE_REPLAY_MANUAL_CAPTURE",
        "Arm replay recording manually instead of using the model lifecycle."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pool_debug",
        "replay_pool_debug",
        false,
        "HIPFIRE_REDLINE_POOL_DEBUG",
        "Report which memory pool backs the retained indirect buffer."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.dispatch_profile",
        "replay_dispatch_profile",
        false,
        "HIPFIRE_REDLINE_DISPATCH_PROFILE",
        "Emit a GPU-clock write per dispatch and report the span distribution. \
         Changes the tape identity, so an instrumented run cannot satisfy a golden fixture."
    ),
    diagnostic_bool_field!(
        "diagnostic.compiler.no_device_compiler",
        "no_device_compiler",
        false,
        "HIPFIRE_NO_DEVICE_COMPILER",
        "Treat the device compiler as absent so pre-compiled code objects are used verbatim."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_min_parallel_width",
        "replay_pm4_min_parallel_width",
        DefaultValue::Integer(2),
        ValueRule::Integer { min: 2, max: 1024 },
        "HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH",
        "Minimum independent launches required to form a parallel PM4 phase."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_min_parallel_workgroups",
        "replay_pm4_min_parallel_workgroups",
        DefaultValue::Integer(0),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS",
        "Minimum aggregate workgroups required to form a parallel PM4 phase."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_max_parallel_phases",
        "replay_pm4_max_parallel_phases",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES",
        "Maximum parallel PM4 phases; null leaves the count unlimited."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pm4_native_phases",
        "replay_pm4_native_phases",
        false,
        "HIPFIRE_REPLAY_PM4_NATIVE_PHASES",
        "Use native cross-queue synchronization for parallel PM4 phases."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_queues",
        "replay_pm4_queues",
        DefaultValue::String("1"),
        ValueRule::Enum(&["1", "2", "4", "auto"]),
        "HIPFIRE_REPLAY_PM4_QUEUES",
        "Number of PM4 queues used for retained replay."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_register_policy",
        "replay_pm4_stateful",
        DefaultValue::String("static"),
        ValueRule::Enum(&["legacy", "static", "stateful"]),
        "HIPFIRE_REPLAY_PM4_STATEFUL",
        "PM4 register emission policy; static is the gfx12-safe product default and caches only queue-global invariants."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_wait_policy",
        "replay_pm4_wait_policy",
        DefaultValue::String("resource"),
        ValueRule::Enum(&["allowlist", "resource-audit", "resource"]),
        "HIPFIRE_REPLAY_PM4_WAIT_POLICY",
        "Dependency policy for waits between retained PM4 launches."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_acquire_policy",
        "replay_pm4_acquire_policy",
        DefaultValue::String("required-only"),
        ValueRule::Enum(&[
            "conservative",
            "entry-only",
            "required-only",
            "without-repeat-interleave",
            "without-fused-silu-rotate",
            "without-mq-rotate",
            "without-rope"
        ]),
        "HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY",
        "Cache-acquire policy between retained PM4 launches."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pm4_gcr_trim",
        "replay_pm4_gcr_trim",
        true,
        "HIPFIRE_REPLAY_PM4_GCR_TRIM",
        "Trim redundant GCR operations on supported retained PM4 routes."
    ),
    process_auto_bool_field!(
        "replay.pm4_gfx11_vmem_acquire",
        "replay_pm4_gfx11_vmem_acquire",
        Replay,
        true,
        "HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE",
        "Enable Radiowave-classified VMEM acquires on gfx11; auto selects gfx1151."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_initiator",
        "gfx1151_pm4_initiator",
        DefaultValue::String("legacy"),
        ValueRule::Enum(&["legacy", "order", "radv"]),
        "HIPFIRE_GFX1151_PM4_INITIATOR",
        "gfx1151 PM4 dispatch-initiator policy."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_interleave",
        "gfx1151_pm4_interleave",
        DefaultValue::String("inherit"),
        ValueRule::Enum(&["inherit", "off", "64", "128", "256", "512"]),
        "HIPFIRE_GFX1151_PM4_INTERLEAVE",
        "gfx1151 dispatch-interleave threads per shader engine."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_resource_limits",
        "gfx1151_pm4_resource_limits",
        DefaultValue::String("legacy"),
        ValueRule::Enum(&["legacy", "simd-always", "radv"]),
        "HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS",
        "gfx1151 compute-resource-limits packet policy."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_cu_count",
        "gfx1151_redline_cu_count",
        DefaultValue::String("all"),
        ValueRule::String,
        "HIPFIRE_GFX1151_REDLINE_CU_COUNT",
        "Even gfx1151 CU count below 40, or all."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_entry_acquire",
        "gfx1151_pm4_entry_acquire",
        DefaultValue::String("system"),
        ValueRule::Enum(&["system", "agent", "vmem", "none"]),
        "HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE",
        "gfx1151 PM4 entry cache-acquire scope."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pm4_dynamic_grid",
        "replay_pm4_dynamic_grid",
        false,
        "HIPFIRE_REPLAY_PM4_DYNAMIC_GRID",
        "Patch retained PM4 grid dimensions at replay time."
    ),
    field!(
        "fusions.policy",
        "fusion_policy",
        Fusions,
        ModelLoad,
        DefaultValue::String("safe"),
        ValueRule::Enum(&["safe", "off"]),
        true,
        false,
        None,
        "Certified fusion policy; individual kernel selection remains compiled."
    ),
];

pub fn fields() -> &'static [ConfigField] {
    FIELDS
}

pub fn field(key: &str) -> Option<&'static ConfigField> {
    FIELDS
        .iter()
        .find(|candidate| candidate.key == key || candidate.legacy_key == key)
}

const DEVELOPER_PREFIX: &str = "developer.";
const BOOTSTRAP_ENV: &[&str] = &[
    "HIPFIRE_HOME",
    "HIPFIRE_MODELS_DIR",
    "HIPFIRE_DAEMON_BIN",
    "HIPFIRE_TUI_BIN",
    "HIPFIRE_CLI_BIN",
    "HIPFIRE_HF_BASE",
    "HF_ENDPOINT",
    "HIPFIRE_REGISTRY_URL",
    "HIPFIRE_NO_REGISTRY_FETCH",
    "HIPFIRE_KERNEL_CACHE",
    "HIPFIRE_SPILL_DIR",
    "HIPFIRE_QUANT_DIAG_PATH",
];

pub fn is_developer_key(key: &str) -> bool {
    key.strip_prefix(DEVELOPER_PREFIX)
        .is_some_and(valid_developer_suffix)
}

pub fn developer_key_for_env(name: &str) -> Option<String> {
    let suffix = name.strip_prefix("HIPFIRE_")?.to_ascii_lowercase();
    valid_developer_suffix(&suffix).then(|| format!("{DEVELOPER_PREFIX}{suffix}"))
}

fn valid_developer_suffix(suffix: &str) -> bool {
    !suffix.is_empty()
        && suffix
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

pub fn canonical_config_key(key: &str) -> Option<String> {
    field(key)
        .map(|schema| schema.key.to_owned())
        .or_else(|| is_developer_key(key).then(|| key.to_owned()))
}

pub fn developer_env_for_key(key: &str) -> Option<String> {
    key.strip_prefix(DEVELOPER_PREFIX)
        .filter(|suffix| valid_developer_suffix(suffix))
        .map(|suffix| format!("HIPFIRE_{}", suffix.to_ascii_uppercase()))
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfigLayer {
    pub values: BTreeMap<String, ConfigValue>,
}

impl ConfigLayer {
    pub fn set(&mut self, key: &str, value: ConfigValue) -> Result<()> {
        if let Some(field) = field(key) {
            field.validate(&value)?;
            self.values.insert(field.key.to_owned(), value);
        } else if is_developer_key(key) {
            self.values.insert(key.to_owned(), value);
        } else {
            return Err(ConfigError::UnknownKey(key.to_owned()));
        }
        Ok(())
    }

    pub fn set_cli(&mut self, key: &str, raw: &str) -> Result<()> {
        if let Some(field) = field(key) {
            self.set(field.key, field.parse_cli(raw)?)
        } else if is_developer_key(key) {
            // Preserve the exact legacy spelling. Experimental consumers may
            // distinguish enum-like values such as `on` from boolean `1`.
            self.set(key, ConfigValue::String(raw.to_owned()))
        } else {
            Err(ConfigError::UnknownKey(key.to_owned()))
        }
    }

    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.values.get(&canonical_config_key(key)?)
    }

    pub fn remove(&mut self, key: &str) -> Result<Option<ConfigValue>> {
        let key =
            canonical_config_key(key).ok_or_else(|| ConfigError::UnknownKey(key.to_owned()))?;
        Ok(self.values.remove(&key))
    }

    pub fn validate(&self) -> Result<()> {
        for (key, value) in &self.values {
            if let Some(field) = field(key) {
                field.validate(value)?;
            } else if !is_developer_key(key) {
                return Err(ConfigError::UnknownKey(key.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ConfigSource {
    BuiltIn,
    RegistryModel {
        tag: String,
        revision: String,
    },
    RegistryTarget {
        tag: String,
        arch: String,
        revision: String,
    },
    GlobalUser {
        path: PathBuf,
    },
    ModelUser {
        model: String,
        path: PathBuf,
    },
    LegacyEnv {
        name: String,
    },
    OneShot {
        argument: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConfigCandidate {
    pub value: ConfigValue,
    pub source: ConfigSource,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResolvedValue {
    pub value: ConfigValue,
    pub source: ConfigSource,
    pub shadowed: Vec<ConfigCandidate>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ResolvedConfig {
    pub values: BTreeMap<String, ResolvedValue>,
}

impl ResolvedConfig {
    pub fn get(&self, key: &str) -> Option<&ResolvedValue> {
        self.values.get(&canonical_config_key(key)?)
    }

    pub fn legacy_values(&self) -> BTreeMap<String, ConfigValue> {
        FIELDS
            .iter()
            .filter_map(|field| {
                self.values
                    .get(field.key)
                    .map(|resolved| (field.legacy_key.to_owned(), resolved.value.clone()))
            })
            .collect()
    }
}

/// Versioned process-start policy sent from the native CLI to the engine
/// daemon. Values remain sparse where absence selects an architecture-specific
/// default, but every included value is validated against [`FIELDS`] before it
/// crosses the process boundary and again after deserialization.
///
/// The daemon lowers this envelope into compact runtime-specific structs; GPU
/// and model hot paths never parse TOML or inspect this generic value map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessConfig {
    pub schema_version: i64,
    pub values: ConfigLayer,
}

static ACTIVE_PROCESS_CONFIG: OnceLock<ProcessConfig> = OnceLock::new();

pub const HIP_VISIBLE_DEVICES: &str = "HIP_VISIBLE_DEVICES";
pub const ROCR_VISIBLE_DEVICES: &str = "ROCR_VISIBLE_DEVICES";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceVisibility {
    /// Physical selectors consumed by ROCr.
    pub rocr: String,
    /// Logical selectors inside the ROCr-filtered set consumed by HIP.
    pub hip: String,
}

impl ProcessConfig {
    pub fn from_resolved(resolved: &ResolvedConfig) -> Result<Self> {
        let mut values = ConfigLayer::default();
        for schema in FIELDS {
            if schema.env_compat.is_none() {
                continue;
            }
            let Some(resolved_value) = resolved.get(schema.key) else {
                continue;
            };
            // Architecture-sensitive defaults are represented by absence.
            // Once a user, registry, one-shot argument, or compatibility env
            // explicitly resolves the field, carry its concrete value.
            if matches!(resolved_value.source, ConfigSource::BuiltIn)
                && !schema.include_builtin_in_process_config
            {
                continue;
            }
            values.set(schema.key, resolved_value.value.clone())?;
        }
        for (key, resolved_value) in &resolved.values {
            if is_developer_key(key) {
                values.set(key, resolved_value.value.clone())?;
            }
        }
        let config = Self {
            schema_version: CONFIG_SCHEMA_VERSION,
            values,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(ConfigError::InvalidValue {
                key: "schema_version".into(),
                message: format!(
                    "unsupported process config schema {}; expected {}",
                    self.schema_version, CONFIG_SCHEMA_VERSION
                ),
            });
        }
        self.values.validate()?;
        for key in self.values.values.keys() {
            if is_developer_key(key) {
                continue;
            }
            let schema = field(key).expect("ConfigLayer::validate accepted stable key");
            if schema.env_compat.is_none() {
                return Err(ConfigError::InvalidValue {
                    key: key.clone(),
                    message: "field is not part of process-start policy".into(),
                });
            }
        }
        Ok(())
    }

    /// Render the compatibility spelling expected by an internal snapshot
    /// parser. This is an in-memory adapter only; it does not mutate or inspect
    /// the ambient process environment.
    pub fn legacy_value(&self, name: &str) -> Option<String> {
        if let Some(schema) = FIELDS.iter().find(|schema| schema.env_compat == Some(name)) {
            return self.values.get(schema.key).and_then(render_compat_value);
        }
        let key = developer_key_for_env(name)?;
        self.values.get(&key).and_then(render_compat_value)
    }
}

pub fn install_process_config(config: ProcessConfig) -> std::result::Result<(), ProcessConfig> {
    ACTIVE_PROCESS_CONFIG.set(config)
}

/// Resolve one physical GPU set for both ROCm frontends.
///
/// ROCr consumes the configured physical selectors. HIP consumes logical
/// `0..N-1` inside that already-filtered set; giving both runtimes the same
/// non-zero index can compound their filters into an empty device set.
/// Explicit `hardware.devices` policy wins. Compatible legacy pairs are
/// normalized, while ambiguous inherited pairs fail closed.
pub fn synchronized_device_visibility(
    config: &ProcessConfig,
    hip_visible: Option<&str>,
    rocr_visible: Option<&str>,
) -> Result<Option<DeviceVisibility>> {
    let configured = config.legacy_value("HIPFIRE_DEVICES");
    if let Some(configured) = configured.as_deref() {
        return visibility_from_physical(configured).map(Some);
    }

    let hip = hip_visible.map(normalize_device_visibility).transpose()?;
    let rocr = rocr_visible.map(normalize_device_visibility).transpose()?;
    match (hip, rocr) {
        (Some(hip), Some(rocr)) => {
            let expected_hip = logical_device_list(device_count(&rocr));
            if hip == expected_hip || hip == rocr {
                Ok(Some(DeviceVisibility {
                    rocr,
                    hip: expected_hip,
                }))
            } else {
                Err(ConfigError::InvalidValue {
                    key: "hardware.devices".into(),
                    message: format!(
                        "{ROCR_VISIBLE_DEVICES}={rocr:?} requires {HIP_VISIBLE_DEVICES}={expected_hip:?}, but inherited {hip:?}; set hardware.devices to one physical device list"
                    ),
                })
            }
        }
        (Some(hip), None) => visibility_from_physical(&hip).map(Some),
        (None, Some(rocr)) => visibility_from_physical(&rocr).map(Some),
        (None, None) => Ok(None),
    }
}

/// Install synchronized HIP/ROCr visibility before either GPU runtime is
/// initialized. Callers must invoke this during single-threaded startup.
pub fn apply_device_visibility(config: &ProcessConfig) -> Result<Option<DeviceVisibility>> {
    let hip = unicode_environment(HIP_VISIBLE_DEVICES)?;
    let rocr = unicode_environment(ROCR_VISIBLE_DEVICES)?;
    let visibility = synchronized_device_visibility(config, hip.as_deref(), rocr.as_deref())?;
    if let Some(visibility) = &visibility {
        std::env::set_var(HIP_VISIBLE_DEVICES, &visibility.hip);
        std::env::set_var(ROCR_VISIBLE_DEVICES, &visibility.rocr);
    }
    Ok(visibility)
}

fn unicode_environment(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(ConfigError::InvalidValue {
            key: "hardware.devices".into(),
            message: format!("{name} is not valid Unicode"),
        }),
    }
}

fn normalize_device_visibility(value: &str) -> Result<String> {
    let devices = value.split(',').map(str::trim).collect::<Vec<_>>();
    if devices.is_empty() || devices.iter().any(|device| device.is_empty()) {
        return Err(ConfigError::InvalidValue {
            key: "hardware.devices".into(),
            message: "expected a non-empty comma-separated physical device list".into(),
        });
    }
    Ok(devices.join(","))
}

fn visibility_from_physical(value: &str) -> Result<DeviceVisibility> {
    let rocr = normalize_device_visibility(value)?;
    Ok(DeviceVisibility {
        hip: logical_device_list(device_count(&rocr)),
        rocr,
    })
}

fn device_count(value: &str) -> usize {
    value.split(',').count()
}

fn logical_device_list(count: usize) -> String {
    (0..count)
        .map(|device| device.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn active_process_config() -> Option<&'static ProcessConfig> {
    ACTIVE_PROCESS_CONFIG.get()
}

/// Resolve global TOML plus the legacy compatibility layer for direct daemon
/// and developer-tool invocation. Native CLI launches install their already
/// resolved policy explicitly and never take this fallback.
pub fn load_local_process_config() -> Result<ProcessConfig> {
    let paths = ConfigPaths::discover();
    let loaded = load_global(&paths)?;
    let mut layers = vec![NamedLayer {
        source: ConfigSource::GlobalUser { path: loaded.path },
        layer: loaded.layer,
    }];
    let environment = load_env_layer()?;
    if !environment.values.is_empty() {
        layers.push(NamedLayer {
            source: ConfigSource::LegacyEnv {
                name: "HIPFIRE_*".into(),
            },
            layer: environment,
        });
    }
    ProcessConfig::from_resolved(&resolve(layers)?)
}

pub fn active_or_local_process_config() -> &'static ProcessConfig {
    ACTIVE_PROCESS_CONFIG.get_or_init(|| {
        load_local_process_config()
            .unwrap_or_else(|error| panic!("invalid hipfire process configuration: {error}"))
    })
}

/// Read one process-start value from the validated in-memory policy. The
/// argument is the temporary compatibility spelling used by compact runtime
/// parsers; this function never reads or mutates the ambient environment.
pub fn process_value(name: &str) -> Option<String> {
    active_or_local_process_config().legacy_value(name)
}

/// Compatibility-shaped access for experimental code while its public policy
/// is being consolidated. Values come exclusively from the process snapshot.
pub fn developer_var(name: &str) -> std::result::Result<String, std::env::VarError> {
    process_value(name).ok_or(std::env::VarError::NotPresent)
}

pub fn developer_var_os(name: &str) -> Option<std::ffi::OsString> {
    process_value(name).map(Into::into)
}

fn render_compat_value(value: &ConfigValue) -> Option<String> {
    Some(match value {
        ConfigValue::Bool(value) => {
            if *value {
                "1".into()
            } else {
                "0".into()
            }
        }
        ConfigValue::Integer(value) => value.to_string(),
        ConfigValue::Float(value) => value.to_string(),
        ConfigValue::String(value) => value.clone(),
        ConfigValue::Null => return None,
    })
}

#[derive(Clone, Debug)]
pub struct NamedLayer {
    pub source: ConfigSource,
    pub layer: ConfigLayer,
}

/// Resolve layers supplied from lowest to highest priority.
pub fn resolve(layers: impl IntoIterator<Item = NamedLayer>) -> Result<ResolvedConfig> {
    let mut out = ResolvedConfig::default();
    for field in FIELDS {
        out.values.insert(
            field.key.to_owned(),
            ResolvedValue {
                value: field.default.to_value(),
                source: ConfigSource::BuiltIn,
                shadowed: Vec::new(),
            },
        );
    }

    for named in layers {
        named.layer.validate()?;
        for (key, value) in named.layer.values {
            if let Some(current) = out.values.get_mut(&key) {
                current.shadowed.push(ConfigCandidate {
                    value: current.value.clone(),
                    source: current.source.clone(),
                });
                current.value = value;
                current.source = named.source.clone();
            } else if is_developer_key(&key) {
                out.values.insert(
                    key,
                    ResolvedValue {
                        value,
                        source: named.source.clone(),
                        shadowed: Vec::new(),
                    },
                );
            } else {
                return Err(ConfigError::UnknownKey(key));
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigFormat {
    Toml,
    LegacyJson,
    Empty,
}

#[derive(Clone, Debug)]
pub struct LoadedConfig {
    pub layer: ConfigLayer,
    pub path: PathBuf,
    pub format: ConfigFormat,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogFormat {
    Toml,
    LegacyJson,
    Empty,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalModelConfig {
    pub path: Option<PathBuf>,
    pub registry_tag: Option<String>,
    pub overrides: ConfigLayer,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelCatalog {
    /// Human-friendly name to canonical local model identity.
    pub aliases: BTreeMap<String, String>,
    /// Canonical local identity to path, registry identity, and sparse overrides.
    pub models: BTreeMap<String, LocalModelConfig>,
}

impl ModelCatalog {
    pub fn model_id(&self, name: &str) -> Option<&str> {
        if let Some((id, _)) = self.models.get_key_value(name) {
            return Some(id);
        }
        if let Some(id) = self.aliases.get(name) {
            return Some(id);
        }
        self.models.iter().find_map(|(id, model)| {
            let path_match = model.path.as_ref().is_some_and(|path| {
                path == Path::new(name)
                    || path.file_name().and_then(|file| file.to_str()) == Some(name)
            });
            (model.registry_tag.as_deref() == Some(name) || path_match).then_some(id.as_str())
        })
    }

    pub fn model(&self, name: &str) -> Option<(&str, &LocalModelConfig)> {
        let id = self.model_id(name)?;
        Some((id, self.models.get(id)?))
    }

    pub fn validate(&self) -> Result<()> {
        let mut paths = BTreeSet::new();
        for (id, model) in &self.models {
            if id.trim().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: "models".into(),
                    message: "model identity cannot be empty".into(),
                });
            }
            if let Some(path) = &model.path {
                if path.as_os_str().is_empty() {
                    return Err(ConfigError::InvalidValue {
                        key: format!("models.{id}.path"),
                        message: "path cannot be empty".into(),
                    });
                }
                if !paths.insert(path.clone()) {
                    return Err(ConfigError::InvalidValue {
                        key: format!("models.{id}.path"),
                        message: format!("duplicate catalog path {}", path.display()),
                    });
                }
            }
            validate_model_layer(&model.overrides)?;
        }
        for (alias, target) in &self.aliases {
            if alias.trim().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: "aliases".into(),
                    message: "alias cannot be empty".into(),
                });
            }
            if alias == target {
                return Err(ConfigError::InvalidValue {
                    key: format!("aliases.{alias}"),
                    message: "alias cannot point to itself".into(),
                });
            }
            if !self.models.contains_key(target) {
                return Err(ConfigError::InvalidValue {
                    key: format!("aliases.{alias}"),
                    message: format!("unknown local model identity {target}"),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LoadedCatalog {
    pub catalog: ModelCatalog,
    pub path: PathBuf,
    pub format: CatalogFormat,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ConfigPaths {
    pub root: PathBuf,
    pub models: PathBuf,
    pub profiles: PathBuf,
    pub config_toml: PathBuf,
    pub config_json: PathBuf,
    pub models_toml: PathBuf,
    pub models_json: PathBuf,
    pub legacy_per_model_json: PathBuf,
}

impl ConfigPaths {
    pub fn discover() -> Self {
        let root = env::var_os("HIPFIRE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".hipfire")))
            .unwrap_or_else(|| PathBuf::from(".hipfire"));
        let mut paths = Self::under(root);
        if let Some(models) = env::var_os("HIPFIRE_MODELS_DIR") {
            paths.models = PathBuf::from(models);
        }
        paths
    }

    pub fn under(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            models: root.join("models"),
            profiles: root.join("profiles"),
            config_toml: root.join("config.toml"),
            config_json: root.join("config.json"),
            models_toml: root.join("models.toml"),
            models_json: root.join("models.json"),
            legacy_per_model_json: root.join("per_model_config.json"),
            root,
        }
    }
}

pub fn load_global(paths: &ConfigPaths) -> Result<LoadedConfig> {
    if paths.config_toml.exists() {
        let layer = load_toml_layer(&paths.config_toml)?;
        return Ok(LoadedConfig {
            layer,
            path: paths.config_toml.clone(),
            format: ConfigFormat::Toml,
            warnings: Vec::new(),
        });
    }
    if paths.config_json.exists() {
        let (layer, warnings) = load_legacy_json_layer(&paths.config_json)?;
        return Ok(LoadedConfig {
            layer,
            path: paths.config_json.clone(),
            format: ConfigFormat::LegacyJson,
            warnings,
        });
    }
    Ok(LoadedConfig {
        layer: ConfigLayer::default(),
        path: paths.config_toml.clone(),
        format: ConfigFormat::Empty,
        warnings: Vec::new(),
    })
}

/// Load the local model catalog. TOML is authoritative once present. Legacy
/// JSON inputs are merged in memory so the first native write can preserve
/// aliases and per-model overrides without deleting the rollback files.
pub fn load_catalog(paths: &ConfigPaths) -> Result<LoadedCatalog> {
    if paths.models_toml.exists() {
        return Ok(LoadedCatalog {
            catalog: load_catalog_toml(&paths.models_toml)?,
            path: paths.models_toml.clone(),
            format: CatalogFormat::Toml,
            warnings: Vec::new(),
        });
    }

    if paths.models_json.exists() || paths.legacy_per_model_json.exists() {
        let (catalog, warnings) = load_legacy_catalog(paths)?;
        return Ok(LoadedCatalog {
            catalog,
            path: paths.models_json.clone(),
            format: CatalogFormat::LegacyJson,
            warnings,
        });
    }

    Ok(LoadedCatalog {
        catalog: ModelCatalog::default(),
        path: paths.models_toml.clone(),
        format: CatalogFormat::Empty,
        warnings: Vec::new(),
    })
}

pub fn load_catalog_toml(path: &Path) -> Result<ModelCatalog> {
    let raw = read_string(path)?;
    let mut root = toml::from_str::<toml::Table>(&raw).map_err(|source| ConfigError::Parse {
        path: path.to_owned(),
        message: source.to_string(),
    })?;
    let version = root
        .remove("schema_version")
        .and_then(|value| value.as_integer())
        .unwrap_or(CONFIG_SCHEMA_VERSION);
    if version != CONFIG_SCHEMA_VERSION {
        return Err(ConfigError::Parse {
            path: path.to_owned(),
            message: format!("unsupported schema_version {version}"),
        });
    }

    let aliases_value = root
        .remove("aliases")
        .unwrap_or_else(|| toml::Value::Table(toml::Table::new()));
    let aliases_table = aliases_value.as_table().ok_or_else(|| ConfigError::Parse {
        path: path.to_owned(),
        message: "aliases must be a table".into(),
    })?;
    let mut aliases = BTreeMap::new();
    for (alias, target) in aliases_table {
        let target = target.as_str().ok_or_else(|| ConfigError::Parse {
            path: path.to_owned(),
            message: format!("aliases.{alias} must be a string"),
        })?;
        aliases.insert(alias.clone(), target.to_owned());
    }

    let models_value = root
        .remove("models")
        .unwrap_or_else(|| toml::Value::Table(toml::Table::new()));
    let models_table = models_value.as_table().ok_or_else(|| ConfigError::Parse {
        path: path.to_owned(),
        message: "models must be a table".into(),
    })?;
    let mut models = BTreeMap::new();
    for (id, value) in models_table {
        let record = value.as_table().ok_or_else(|| ConfigError::Parse {
            path: path.to_owned(),
            message: format!("models.{id} must be a table"),
        })?;
        let unknown = record
            .keys()
            .filter(|key| !matches!(key.as_str(), "path" | "registry_tag" | "overrides"))
            .cloned()
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(ConfigError::Parse {
                path: path.to_owned(),
                message: format!("unknown fields in models.{id}: {}", unknown.join(", ")),
            });
        }
        let path_value = record
            .get("path")
            .map(|value| {
                value
                    .as_str()
                    .map(PathBuf::from)
                    .ok_or_else(|| ConfigError::Parse {
                        path: path.to_owned(),
                        message: format!("models.{id}.path must be a string"),
                    })
            })
            .transpose()?;
        let registry_tag = record
            .get("registry_tag")
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| ConfigError::Parse {
                        path: path.to_owned(),
                        message: format!("models.{id}.registry_tag must be a string"),
                    })
            })
            .transpose()?;
        let mut overrides = ConfigLayer::default();
        if let Some(value) = record.get("overrides") {
            let table = value.as_table().ok_or_else(|| ConfigError::Parse {
                path: path.to_owned(),
                message: format!("models.{id}.overrides must be a table"),
            })?;
            let mut flat = BTreeMap::new();
            flatten_toml("", table, &mut flat, path)?;
            for (key, value) in flat {
                overrides.set(&key, value)?;
            }
        }
        validate_model_layer(&overrides)?;
        models.insert(
            id.clone(),
            LocalModelConfig {
                path: path_value,
                registry_tag,
                overrides,
            },
        );
    }
    if !root.is_empty() {
        return Err(ConfigError::Parse {
            path: path.to_owned(),
            message: format!(
                "unknown top-level fields: {}",
                root.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        });
    }

    let catalog = ModelCatalog { aliases, models };
    catalog.validate()?;
    Ok(catalog)
}

pub fn write_catalog_toml(paths: &ConfigPaths, catalog: &ModelCatalog) -> Result<()> {
    catalog.validate()?;
    let mut root = toml::Table::new();
    root.insert(
        "schema_version".into(),
        toml::Value::Integer(CONFIG_SCHEMA_VERSION),
    );
    if !catalog.aliases.is_empty() {
        root.insert(
            "aliases".into(),
            toml::Value::Table(
                catalog
                    .aliases
                    .iter()
                    .map(|(alias, target)| (alias.clone(), toml::Value::String(target.clone())))
                    .collect(),
            ),
        );
    }
    let mut models = toml::Table::new();
    for (id, model) in &catalog.models {
        let mut record = toml::Table::new();
        if let Some(path) = &model.path {
            record.insert(
                "path".into(),
                toml::Value::String(path.display().to_string()),
            );
        }
        if let Some(tag) = &model.registry_tag {
            record.insert("registry_tag".into(), toml::Value::String(tag.clone()));
        }
        if !model.overrides.values.is_empty() {
            let mut overrides = toml::Table::new();
            for (key, value) in &model.overrides.values {
                let Some(value) = value.clone().into_toml() else {
                    continue;
                };
                insert_toml_path(&mut overrides, key, value)?;
            }
            record.insert("overrides".into(), toml::Value::Table(overrides));
        }
        models.insert(id.clone(), toml::Value::Table(record));
    }
    if !models.is_empty() {
        root.insert("models".into(), toml::Value::Table(models));
    }
    let rendered =
        toml::to_string_pretty(&toml::Value::Table(root)).map_err(|source| ConfigError::Parse {
            path: paths.models_toml.clone(),
            message: source.to_string(),
        })?;
    atomic_write(&paths.models_toml, rendered.as_bytes())
}

fn validate_model_layer(layer: &ConfigLayer) -> Result<()> {
    layer.validate()?;
    for key in layer.values.keys() {
        if is_developer_key(key) {
            return Err(ConfigError::InvalidValue {
                key: key.clone(),
                message: "developer fields are global process policy and not valid per-model"
                    .into(),
            });
        }
        let schema = field(key).expect("validated configuration field");
        if matches!(schema.scope, ConfigScope::Process | ConfigScope::Diagnostic) {
            return Err(ConfigError::InvalidValue {
                key: key.clone(),
                message: "field is not valid in a per-model override".into(),
            });
        }
    }
    Ok(())
}

fn load_legacy_catalog(paths: &ConfigPaths) -> Result<(ModelCatalog, Vec<String>)> {
    let mut catalog = ModelCatalog::default();
    let mut warnings = Vec::new();
    let raw = if paths.models_json.exists() {
        Some(read_json_object(&paths.models_json)?)
    } else {
        None
    };

    if let Some(raw) = raw.as_ref() {
        if raw
            .get("schema_version")
            .and_then(serde_json::Value::as_i64)
            == Some(2)
        {
            if let Some(models) = raw.get("models").and_then(serde_json::Value::as_object) {
                for (id, value) in models {
                    let Some(record) = value.as_object() else {
                        warnings.push(format!("ignored invalid legacy model {id}"));
                        continue;
                    };
                    let path_value = record
                        .get("path")
                        .and_then(serde_json::Value::as_str)
                        .map(PathBuf::from);
                    let registry_tag = record
                        .get("registry_tag")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned);
                    let overrides = legacy_override_layer(
                        record.get("config"),
                        &format!("models.{id}.config"),
                        &mut warnings,
                    );
                    catalog.models.insert(
                        id.clone(),
                        LocalModelConfig {
                            path: path_value,
                            registry_tag,
                            overrides,
                        },
                    );
                    if let Some(aliases) =
                        record.get("aliases").and_then(serde_json::Value::as_array)
                    {
                        for alias in aliases.iter().filter_map(serde_json::Value::as_str) {
                            catalog.aliases.insert(alias.to_owned(), id.clone());
                        }
                    }
                }
            }
            if let Some(configs) = raw.get("configs").and_then(serde_json::Value::as_object) {
                for (key, value) in configs {
                    merge_legacy_override(&mut catalog, key, value, &mut warnings);
                }
            }
            migrate_legacy_aliases(raw.get("aliases"), paths, &mut catalog, &mut warnings);
        } else {
            migrate_legacy_aliases(
                Some(&serde_json::Value::Object(raw.clone())),
                paths,
                &mut catalog,
                &mut warnings,
            );
        }
    }

    if paths.legacy_per_model_json.exists() {
        let legacy = read_json_object(&paths.legacy_per_model_json)?;
        for (key, value) in &legacy {
            merge_legacy_override(&mut catalog, key, value, &mut warnings);
        }
    }

    // Legacy catalogs can contain duplicate derived paths or dangling alias
    // records. Preserve every usable override and report the rest instead of
    // turning migration into a startup failure.
    for (alias, target) in catalog.aliases.clone() {
        if !catalog.models.contains_key(&target) {
            catalog.aliases.remove(&alias);
            warnings.push(format!("ignored dangling legacy alias {alias} -> {target}"));
        }
    }
    Ok((catalog, warnings))
}

fn read_json_object(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let raw = read_string(path)?;
    serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            message: source.to_string(),
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| ConfigError::Parse {
            path: path.to_owned(),
            message: "root must be a JSON object".into(),
        })
}

fn legacy_override_layer(
    value: Option<&serde_json::Value>,
    context: &str,
    warnings: &mut Vec<String>,
) -> ConfigLayer {
    let mut layer = ConfigLayer::default();
    let Some(object) = value.and_then(serde_json::Value::as_object) else {
        return layer;
    };
    for (key, value) in object {
        let Some(schema) = field(key) else {
            warnings.push(format!("ignored unknown legacy field {context}.{key}"));
            continue;
        };
        let Some(value) = from_json_value(value) else {
            warnings.push(format!("ignored unsupported legacy value {context}.{key}"));
            continue;
        };
        if matches!(schema.scope, ConfigScope::Process | ConfigScope::Diagnostic) {
            warnings.push(format!("ignored global-only legacy field {context}.{key}"));
            continue;
        }
        if let Err(error) = layer.set(schema.key, value) {
            warnings.push(format!(
                "ignored invalid legacy field {context}.{key}: {error}"
            ));
        }
    }
    layer
}

fn merge_legacy_override(
    catalog: &mut ModelCatalog,
    key: &str,
    value: &serde_json::Value,
    warnings: &mut Vec<String>,
) {
    let layer = legacy_override_layer(Some(value), key, warnings);
    if layer.values.is_empty() {
        return;
    }
    let model_id = catalog
        .model_id(key)
        .map(str::to_owned)
        .unwrap_or_else(|| key.to_owned());
    let model = catalog
        .models
        .entry(model_id)
        .or_insert_with(|| LocalModelConfig {
            registry_tag: Some(key.to_owned()),
            ..LocalModelConfig::default()
        });
    model.overrides.values.extend(layer.values);
}

fn migrate_legacy_aliases(
    value: Option<&serde_json::Value>,
    paths: &ConfigPaths,
    catalog: &mut ModelCatalog,
    warnings: &mut Vec<String>,
) {
    let Some(aliases) = value.and_then(serde_json::Value::as_object) else {
        return;
    };
    for (alias, value) in aliases {
        let Some(record) = value.as_object() else {
            warnings.push(format!("ignored invalid legacy alias {alias}"));
            continue;
        };
        let local_path = record
            .get("local_path")
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from);
        let file = record.get("file").and_then(serde_json::Value::as_str);
        let path = local_path.or_else(|| file.map(|file| paths.models.join(file)));
        let existing = catalog.models.iter().find_map(|(id, model)| {
            let same_path = path
                .as_ref()
                .is_some_and(|path| model.path.as_ref() == Some(path));
            let same_file = file.is_some_and(|file| {
                model
                    .path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|v| v.to_str())
                    == Some(file)
            });
            (same_path || same_file).then_some(id.clone())
        });
        let id = existing.unwrap_or_else(|| format!("local:{alias}"));
        catalog
            .models
            .entry(id.clone())
            .or_insert_with(|| LocalModelConfig {
                path,
                ..LocalModelConfig::default()
            });
        catalog.aliases.insert(alias.clone(), id);
    }
}

pub fn load_env_layer() -> Result<ConfigLayer> {
    let mut layer = ConfigLayer::default();
    let mut stable_names = BTreeSet::new();
    for field in FIELDS {
        let Some(name) = field.env_compat else {
            continue;
        };
        stable_names.insert(name);
        let Ok(raw) = env::var(name) else {
            continue;
        };
        let value = match (field.legacy_key, raw.as_str()) {
            ("default_chatml", "0") => ConfigValue::Bool(false),
            ("dflash_ngram_block", "1") => ConfigValue::Bool(true),
            ("prompt_normalize", "0" | "false" | "off" | "no") => ConfigValue::Bool(false),
            ("prompt_normalize", _) => ConfigValue::Bool(true),
            _ => field.parse_cli(&raw)?,
        };
        layer.set(field.key, value)?;
    }
    // Snapshot the experimental long tail once. These values are quarantined
    // under [developer], excluded from registry/model policy, and read by the
    // engine only through ProcessConfig after this point.
    for (name, raw) in env::vars_os() {
        let Some(name) = name.to_str() else {
            continue;
        };
        if stable_names.contains(name) || BOOTSTRAP_ENV.contains(&name) {
            continue;
        }
        let Some(key) = developer_key_for_env(name) else {
            continue;
        };
        let Ok(raw) = raw.into_string() else {
            continue;
        };
        layer.set(&key, ConfigValue::String(raw))?;
    }
    Ok(layer)
}

pub fn load_toml_layer(path: &Path) -> Result<ConfigLayer> {
    let raw = read_string(path)?;
    let table = toml::from_str::<toml::Table>(&raw).map_err(|source| ConfigError::Parse {
        path: path.to_owned(),
        message: source.to_string(),
    })?;
    if let Some(version) = table.get("schema_version") {
        if version.as_integer() != Some(CONFIG_SCHEMA_VERSION) {
            return Err(ConfigError::Parse {
                path: path.to_owned(),
                message: format!("unsupported schema_version {version}"),
            });
        }
    }
    let mut flat = BTreeMap::new();
    flatten_toml("", &table, &mut flat, path)?;
    let mut layer = ConfigLayer::default();
    for (key, value) in flat {
        layer.set(&key, value)?;
    }
    Ok(layer)
}

fn flatten_toml(
    prefix: &str,
    table: &toml::map::Map<String, toml::Value>,
    out: &mut BTreeMap<String, ConfigValue>,
    path: &Path,
) -> Result<()> {
    for (key, value) in table {
        if prefix.is_empty() && key == "schema_version" {
            continue;
        }
        let dotted = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        if let Some(nested) = value.as_table() {
            flatten_toml(&dotted, nested, out, path)?;
        } else if let Some(value) = ConfigValue::from_toml(value) {
            out.insert(dotted, value);
        } else {
            return Err(ConfigError::Parse {
                path: path.to_owned(),
                message: format!("unsupported value at {dotted}"),
            });
        }
    }
    Ok(())
}

pub fn load_legacy_json_layer(path: &Path) -> Result<(ConfigLayer, Vec<String>)> {
    let raw = read_string(path)?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            message: source.to_string(),
        })?;
    let object = parsed.as_object().ok_or_else(|| ConfigError::Parse {
        path: path.to_owned(),
        message: "root must be a JSON object".to_owned(),
    })?;
    let mut layer = ConfigLayer::default();
    let mut warnings = Vec::new();
    for (key, value) in object {
        let Some(field) = field(key) else {
            warnings.push(format!("ignored unknown legacy key {key}"));
            continue;
        };
        let Some(value) = from_json_value(value) else {
            warnings.push(format!("ignored unsupported legacy value for {key}"));
            continue;
        };
        if let Err(error) = layer.set(field.key, value) {
            warnings.push(format!("ignored invalid legacy {key}: {error}"));
        }
    }
    Ok((layer, warnings))
}

fn from_json_value(value: &serde_json::Value) -> Option<ConfigValue> {
    match value {
        serde_json::Value::Null => Some(ConfigValue::Null),
        serde_json::Value::Bool(v) => Some(ConfigValue::Bool(*v)),
        serde_json::Value::Number(v) => v
            .as_i64()
            .map(ConfigValue::Integer)
            .or_else(|| v.as_f64().map(ConfigValue::Float)),
        serde_json::Value::String(v) => Some(ConfigValue::String(v.clone())),
        _ => None,
    }
}

pub fn write_global_toml(paths: &ConfigPaths, layer: &ConfigLayer) -> Result<()> {
    write_layer_toml(&paths.config_toml, layer)
}

fn write_layer_toml(path: &Path, layer: &ConfigLayer) -> Result<()> {
    layer.validate()?;
    let mut root = toml::map::Map::new();
    root.insert(
        "schema_version".to_owned(),
        toml::Value::Integer(CONFIG_SCHEMA_VERSION),
    );
    for (key, value) in &layer.values {
        let Some(value) = value.clone().into_toml() else {
            continue;
        };
        insert_toml_path(&mut root, key, value)?;
    }
    let rendered =
        toml::to_string_pretty(&toml::Value::Table(root)).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            message: source.to_string(),
        })?;
    atomic_write(path, rendered.as_bytes())
}

fn insert_toml_path(
    table: &mut toml::map::Map<String, toml::Value>,
    dotted: &str,
    value: toml::Value,
) -> Result<()> {
    let mut parts = dotted.splitn(2, '.');
    let head = parts.next().expect("non-empty config key");
    let Some(tail) = parts.next() else {
        table.insert(head.to_owned(), value);
        return Ok(());
    };
    let entry = table
        .entry(head.to_owned())
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let nested = entry
        .as_table_mut()
        .ok_or_else(|| ConfigError::InvalidValue {
            key: dotted.to_owned(),
            message: format!("{head} is already a scalar"),
        })?;
    insert_toml_path(nested, tail, value)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| ConfigError::Write {
            path: parent.to_owned(),
            source,
        })?;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&tmp, bytes).map_err(|source| ConfigError::Write {
        path: tmp.clone(),
        source,
    })?;
    fs::rename(&tmp, path).map_err(|source| ConfigError::Write {
        path: path.to_owned(),
        source,
    })
}

fn read_string(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_owned(),
        source,
    })
}

/// Built-in configuration profile names selectable via `hipfire config profile set`.
///
/// Custom profiles live under `~/.hipfire/profiles/<name>.toml`. Profile names are
/// control-plane identifiers only; they are never persisted inside `config.toml`.
pub const CONFIG_PROFILE_NAMES: &[&str] = &["default", "dev", "hip", "redline"];

/// Origin of a selectable configuration profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigProfileKind {
    Builtin,
    Custom,
}

/// One built-in or on-disk custom profile entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigProfileEntry {
    pub name: String,
    pub kind: ConfigProfileKind,
    pub path: Option<PathBuf>,
}

/// Validate a custom profile name.
///
/// Accepts a single path segment of ASCII letters, digits, `.`, `_`, or `-`.
/// Rejects empty names, built-in names, path separators, and traversal markers.
pub fn validate_config_profile_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: "profile name must not be empty".to_owned(),
        });
    }
    if CONFIG_PROFILE_NAMES.contains(&name) {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!("'{name}' is a built-in profile name and cannot be overwritten"),
        });
    }
    if name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!("invalid profile name '{name}'"),
        });
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
    {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!(
                "invalid profile name '{name}'; use ASCII letters, digits, '.', '_', or '-'"
            ),
        });
    }
    Ok(())
}

/// Path for a custom profile TOML file under `paths.profiles`.
pub fn custom_config_profile_path(paths: &ConfigPaths, name: &str) -> Result<PathBuf> {
    validate_config_profile_name(name)?;
    Ok(paths.profiles.join(format!("{name}.toml")))
}

/// Load the sparse layer for a built-in or custom profile.
pub fn load_config_profile(paths: &ConfigPaths, name: &str) -> Result<ConfigLayer> {
    if let Some(layer) = builtin_config_profile_layer(name) {
        return Ok(layer);
    }
    let path = custom_config_profile_path(paths, name)?;
    if !path.exists() {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!(
                "unknown profile '{name}'; expected one of {} or a custom profile in {}",
                CONFIG_PROFILE_NAMES.join(", "),
                paths.profiles.display()
            ),
        });
    }
    load_toml_layer(&path)
}

/// Replace `layer` entirely with the selected profile contents.
///
/// Selection is a deterministic full cutover of the sparse global config: every
/// previous override is discarded and only the profile layer remains.
pub fn apply_config_profile(
    layer: &mut ConfigLayer,
    paths: &ConfigPaths,
    name: &str,
) -> Result<()> {
    *layer = load_config_profile(paths, name)?;
    Ok(())
}

/// Snapshot `layer` as a new custom profile. Fails when the name is a built-in,
/// invalid, or already present on disk.
pub fn create_config_profile(
    paths: &ConfigPaths,
    name: &str,
    layer: &ConfigLayer,
) -> Result<PathBuf> {
    let path = custom_config_profile_path(paths, name)?;
    if path.exists() {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!("profile '{name}' already exists at {}", path.display()),
        });
    }
    write_layer_toml(&path, layer)?;
    Ok(path)
}

/// List built-in profiles followed by custom on-disk profiles (sorted).
pub fn list_config_profiles(paths: &ConfigPaths) -> Result<Vec<ConfigProfileEntry>> {
    let mut entries: Vec<ConfigProfileEntry> = CONFIG_PROFILE_NAMES
        .iter()
        .map(|name| ConfigProfileEntry {
            name: (*name).to_owned(),
            kind: ConfigProfileKind::Builtin,
            path: None,
        })
        .collect();
    if paths.profiles.is_dir() {
        let mut custom = Vec::new();
        for entry in fs::read_dir(&paths.profiles).map_err(|source| ConfigError::Read {
            path: paths.profiles.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| ConfigError::Read {
                path: paths.profiles.clone(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if validate_config_profile_name(stem).is_err() {
                continue;
            }
            custom.push(ConfigProfileEntry {
                name: stem.to_owned(),
                kind: ConfigProfileKind::Custom,
                path: Some(path),
            });
        }
        custom.sort_by(|left, right| left.name.cmp(&right.name));
        entries.extend(custom);
    }
    Ok(entries)
}

/// Detect which named profile, if any, exactly matches `layer`.
///
/// Built-ins are checked first; custom profiles are matched by exact layer
/// equality against files under `paths.profiles`.
pub fn detect_config_profile(paths: &ConfigPaths, layer: &ConfigLayer) -> Option<String> {
    for name in CONFIG_PROFILE_NAMES {
        if let Some(bundle) = builtin_config_profile_layer(name) {
            if &bundle == layer {
                return Some((*name).to_owned());
            }
        }
    }
    let Ok(entries) = list_config_profiles(paths) else {
        return None;
    };
    for entry in entries {
        if entry.kind != ConfigProfileKind::Custom {
            continue;
        }
        let Ok(candidate) = load_config_profile(paths, &entry.name) else {
            continue;
        };
        if &candidate == layer {
            return Some(entry.name);
        }
    }
    None
}

fn builtin_config_profile_layer(name: &str) -> Option<ConfigLayer> {
    let pairs = config_profile_bundle(name)?;
    let mut layer = ConfigLayer::default();
    for (key, value) in pairs {
        layer
            .set(key, value)
            .expect("built-in profile values must validate");
    }
    Some(layer)
}

fn config_profile_bundle(name: &str) -> Option<Vec<(&'static str, ConfigValue)>> {
    Some(match name {
        "default" => vec![
            ("generation.max_tokens", ConfigValue::Integer(4096)),
            ("generation.loop_guard_threshold", ConfigValue::Integer(0)),
            ("generation.loop_guard_window", ConfigValue::Integer(256)),
            ("reasoning.mode", ConfigValue::String("on".to_owned())),
            ("reasoning.budget", ConfigValue::String("xhigh".to_owned())),
            ("reasoning.max_total_tokens", ConfigValue::Integer(0)),
            ("memory.kv_cache", ConfigValue::String("q8".to_owned())),
            ("memory.max_seq", ConfigValue::Integer(32768)),
            ("memory.prompt_cache_capacity", ConfigValue::Integer(32)),
            ("memory.prompt_cache_unbounded", ConfigValue::Bool(false)),
            ("attention.flash", ConfigValue::String("auto".to_owned())),
            ("prompt.normalize", ConfigValue::Bool(true)),
            ("prompt.default_chatml", ConfigValue::Bool(true)),
            ("speculation.mode", ConfigValue::String("auto".to_owned())),
            ("speculation.dflash", ConfigValue::String("off".to_owned())),
            ("speculation.mtp", ConfigValue::String("auto".to_owned())),
            ("speculation.mtp_k", ConfigValue::Integer(3)),
            ("speculation.ngram", ConfigValue::String("off".to_owned())),
            ("serve.host", ConfigValue::String("0.0.0.0".to_owned())),
            ("serve.port", ConfigValue::Integer(11435)),
            ("serve.idle_timeout_seconds", ConfigValue::Integer(300)),
            ("serve.local", ConfigValue::Bool(false)),
        ],
        "dev" => vec![
            ("hardware.devices", ConfigValue::String("0".to_owned())),
            ("hardware.allow_mixed_arch", ConfigValue::Bool(false)),
            ("hardware.tp_use_rccl", ConfigValue::Bool(true)),
            ("kernel.mmq", ConfigValue::String("auto".to_owned())),
            ("kernel.prefill_batched", ConfigValue::Bool(true)),
            ("kernel.lm_head_f16", ConfigValue::String("auto".to_owned())),
            (
                "experimental.graph.forward",
                ConfigValue::String("auto".to_owned()),
            ),
            ("experimental.graph.ar", ConfigValue::Bool(true)),
            ("experimental.graph.moe", ConfigValue::Bool(true)),
            ("diagnostic.prompt_token_heat", ConfigValue::Bool(false)),
            ("diagnostic.prompt_heat_json", ConfigValue::Bool(false)),
            ("diagnostic.prompt_heat_limit", ConfigValue::Integer(64)),
            (
                "diagnostic.kernel.gemv_rows",
                ConfigValue::String("4".to_owned()),
            ),
            (
                "diagnostic.kernel.gfx11_weight_load_policy",
                ConfigValue::String("buffer".to_owned()),
            ),
            ("developer.verify_graph", ConfigValue::Bool(false)),
            ("developer.dspark_profile", ConfigValue::Bool(true)),
        ],
        "hip" => vec![("replay.backend", ConfigValue::String("hip".to_owned()))],
        "redline" => vec![
            ("replay.backend", ConfigValue::String("redline".to_owned())),
            ("replay.transport", ConfigValue::String("pm4".to_owned())),
            (
                "replay.pm4_gfx11_vmem_acquire",
                ConfigValue::String("auto".to_owned()),
            ),
            ("diagnostic.replay.manual_capture", ConfigValue::Bool(false)),
            (
                "diagnostic.replay.pm4_min_parallel_width",
                ConfigValue::Integer(2),
            ),
            (
                "diagnostic.replay.pm4_min_parallel_workgroups",
                ConfigValue::Integer(0),
            ),
            (
                "diagnostic.replay.pm4_native_phases",
                ConfigValue::Bool(false),
            ),
            (
                "diagnostic.replay.pm4_queues",
                ConfigValue::String("1".to_owned()),
            ),
            (
                "diagnostic.replay.pm4_register_policy",
                ConfigValue::String("static".to_owned()),
            ),
            (
                "diagnostic.replay.pm4_wait_policy",
                ConfigValue::String("resource".to_owned()),
            ),
            (
                "diagnostic.replay.pm4_acquire_policy",
                ConfigValue::String("required-only".to_owned()),
            ),
            ("diagnostic.replay.pm4_gcr_trim", ConfigValue::Bool(true)),
            (
                "diagnostic.replay.pm4_dynamic_grid",
                ConfigValue::Bool(false),
            ),
            (
                "diagnostic.replay.gfx1151_initiator",
                ConfigValue::String("legacy".to_owned()),
            ),
            (
                "diagnostic.replay.gfx1151_interleave",
                ConfigValue::String("inherit".to_owned()),
            ),
            (
                "diagnostic.replay.gfx1151_resource_limits",
                ConfigValue::String("legacy".to_owned()),
            ),
            (
                "diagnostic.replay.gfx1151_cu_count",
                ConfigValue::String("all".to_owned()),
            ),
            (
                "diagnostic.replay.gfx1151_entry_acquire",
                ConfigValue::String("system".to_owned()),
            ),
        ],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        env::temp_dir().join(format!("hipfire-config-{name}-{}", std::process::id()))
    }

    #[test]
    fn schema_has_unique_keys_and_legacy_keys() {
        let mut canonical = std::collections::BTreeSet::new();
        let mut legacy = std::collections::BTreeSet::new();
        let mut environment = std::collections::BTreeSet::new();
        for field in FIELDS {
            assert!(canonical.insert(field.key), "duplicate {}", field.key);
            assert!(
                legacy.insert(field.legacy_key),
                "duplicate {}",
                field.legacy_key
            );
            if let Some(name) = field.env_compat {
                assert!(
                    environment.insert(name),
                    "duplicate environment alias {name}"
                );
            }
            if !field.include_builtin_in_process_config {
                assert!(field.env_compat.is_some(), "bridge without env alias");
                assert!(
                    matches!(field.scope, ConfigScope::Process | ConfigScope::Diagnostic),
                    "non-process bridge {}",
                    field.key
                );
                assert!(
                    !field.registry_allowed,
                    "registry-backed bridge {}",
                    field.key
                );
            }
            field
                .validate(&field.default.to_value())
                .unwrap_or_else(|error| panic!("invalid default {}: {error}", field.key));
        }
    }

    #[test]
    fn experimental_memory_and_prefill_features_default_off() {
        assert_eq!(
            field("memory.cask.enabled").unwrap().default.to_value(),
            ConfigValue::Bool(false)
        );
        assert_eq!(
            field("memory.cask.auto_attach").unwrap().default.to_value(),
            ConfigValue::Bool(false)
        );
        assert_eq!(
            field("memory.cask.sidecar").unwrap().default.to_value(),
            ConfigValue::String(String::new())
        );
        assert_eq!(
            field("speculation.prefill.mode")
                .unwrap()
                .default
                .to_value(),
            ConfigValue::String("off".into())
        );
    }

    #[test]
    fn deepseek4_expert_fanout_is_optional_and_bounded() {
        let field = field("model.deepseek4_experts_per_token").unwrap();
        assert_eq!(field.default.to_value(), ConfigValue::Null);
        assert_eq!(field.parse_cli("4").unwrap(), ConfigValue::Integer(4));
        assert!(field.parse_cli("0").is_err());
        assert!(field.parse_cli("7").is_err());
    }

    #[test]
    fn million_context_and_parent_output_limits_validate_without_coupling_effort() {
        let mut layer = ConfigLayer::default();
        layer
            .set_cli("memory.max_seq", "1048576")
            .expect("one-million-token context must validate");
        layer
            .set_cli("generation.max_tokens", "393216")
            .expect("384-Ki-token output must validate");
        layer
            .set_cli("reasoning.max_tokens", "393216")
            .expect("an explicit 384-Ki reasoning cap must validate");
        layer
            .set_cli("reasoning.effort", "max")
            .expect("parent max effort must validate independently");
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("max".into()))
        );
        assert_eq!(
            layer.get("reasoning.max_tokens"),
            Some(&ConfigValue::Integer(393216))
        );
        assert!(layer.set_cli("memory.max_seq", "1048577").is_err());
        assert!(layer.set_cli("generation.max_tokens", "393217").is_err());
    }

    #[test]
    fn legacy_json_maps_to_canonical_keys() {
        let root = temp_root("legacy");
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);
        fs::write(
            &paths.config_json,
            r#"{"kv_cache":"q8","max_tokens":8192,"prompt_normalize":false,"unknown":1}"#,
        )
        .unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.format, ConfigFormat::LegacyJson);
        assert_eq!(
            loaded.layer.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert_eq!(
            loaded.layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(8192))
        );
        assert_eq!(loaded.warnings.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sparse_toml_roundtrip() {
        let root = temp_root("roundtrip");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("kv_cache", "q8").unwrap();
        layer.set_cli("generation.max_tokens", "8192").unwrap();
        layer.set_cli("prompt_normalize", "false").unwrap();
        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.format, ConfigFormat::Toml);
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("[generation]"));
        assert!(rendered.contains("[memory]"));
        assert!(!rendered.contains("temperature"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn full_sampling_surface_roundtrips_through_toml() {
        let root = temp_root("sampling-roundtrip");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("generation.temperature", "1.0").unwrap();
        layer.set_cli("generation.top_p", "0.95").unwrap();
        layer.set_cli("generation.top_k", "40").unwrap();
        layer.set_cli("generation.min_p", "0.05").unwrap();
        layer.set_cli("generation.presence_penalty", "1.5").unwrap();
        layer.set_cli("generation.repeat_penalty", "1.05").unwrap();
        layer
            .set_cli("prompt.system", "Registry-compatible system prompt")
            .unwrap();

        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("top_k = 40"));
        assert!(rendered.contains("min_p = 0.05"));
        assert!(rendered.contains("presence_penalty = 1.5"));
        assert!(rendered.contains("system = \"Registry-compatible system prompt\""));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn developer_namespace_roundtrips_and_preserves_legacy_spelling() {
        let root = temp_root("developer-roundtrip");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer
            .set("developer.gfx1151_gate_up_wave64", ConfigValue::Bool(true))
            .unwrap();
        layer
            .set("developer.pm4_queue_count", ConfigValue::Integer(4))
            .unwrap();
        layer.set_cli("developer.experimental_mode", "on").unwrap();

        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("[developer]"));
        assert!(rendered.contains("gfx1151_gate_up_wave64 = true"));
        assert!(rendered.contains("experimental_mode = \"on\""));

        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: paths.config_toml.clone(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(
            process
                .legacy_value("HIPFIRE_GFX1151_GATE_UP_WAVE64")
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_EXPERIMENTAL_MODE").as_deref(),
            Some("on")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn developer_namespace_is_rejected_per_model() {
        let mut overrides = ConfigLayer::default();
        overrides
            .set("developer.gfx1151_gate_up_wave64", ConfigValue::Bool(true))
            .unwrap();
        assert!(validate_model_layer(&overrides).is_err());
    }

    #[test]
    fn typed_runtime_scalars_and_variants_roundtrip() {
        let root = temp_root("runtime-scalars");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("hardware.devices", "2,3").unwrap();
        layer
            .set_cli("hardware.uniform_vram_tolerance_gb", "1.5")
            .unwrap();
        layer
            .set_cli("diagnostic.kernel.gate_up_variant", "k4")
            .unwrap();
        layer
            .set_cli("diagnostic.kernel.rdna2_variant", "5")
            .unwrap();
        layer.set_cli("kernel.lm_head_f16", "f32").unwrap();
        assert!(layer
            .set_cli("diagnostic.kernel.gate_up_variant", "unknown")
            .is_err());
        assert!(layer
            .set_cli("diagnostic.kernel.rdna2_variant", "6")
            .is_err());

        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("[diagnostic.kernel]"));
        assert!(rendered.contains("gate_up_variant = \"k4\""));
        assert!(rendered.contains("rdna2_variant = 5"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_catalog_migrates_aliases_and_model_overrides_to_toml() {
        let root = temp_root("catalog-migration");
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);
        fs::write(
            &paths.models_json,
            r#"{
              "schema_version": 2,
              "aliases": {
                "my-qwen": {"file":"qwen.mq4r","local_path":"/models/qwen.mq4r"}
              },
              "configs": {"qwen:tag":{"thinking_budget":"xhigh"}},
              "models": {
                "qwen.mq4r": {
                  "path":"/models/qwen.mq4r",
                  "registry_tag":"qwen:tag",
                  "aliases":["q-local"],
                  "config":{"kv_cache":"q8"}
                }
              }
            }"#,
        )
        .unwrap();
        fs::write(
            &paths.legacy_per_model_json,
            r#"{"my-qwen":{"max_tokens":8192}}"#,
        )
        .unwrap();

        let loaded = load_catalog(&paths).unwrap();
        assert_eq!(loaded.format, CatalogFormat::LegacyJson);
        let (id, model) = loaded.catalog.model("my-qwen").unwrap();
        assert_eq!(id, "qwen.mq4r");
        assert_eq!(
            model.overrides.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert_eq!(
            model.overrides.get("reasoning.budget"),
            Some(&ConfigValue::String("xhigh".into()))
        );
        assert_eq!(
            model.overrides.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(8192))
        );

        write_catalog_toml(&paths, &loaded.catalog).unwrap();
        let roundtrip = load_catalog(&paths).unwrap();
        assert_eq!(roundtrip.format, CatalogFormat::Toml);
        assert_eq!(roundtrip.catalog, loaded.catalog);
        assert!(paths.models_json.exists());
        assert!(paths.legacy_per_model_json.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resolution_keeps_provenance_and_shadowed_values() {
        let mut registry = ConfigLayer::default();
        registry.set_cli("temperature", "1.0").unwrap();
        let mut global = ConfigLayer::default();
        global.set_cli("generation.temperature", "0.7").unwrap();
        let resolved = resolve([
            NamedLayer {
                source: ConfigSource::RegistryModel {
                    tag: "model".into(),
                    revision: "v1".into(),
                },
                layer: registry,
            },
            NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: PathBuf::from("config.toml"),
                },
                layer: global,
            },
        ])
        .unwrap();
        let temperature = resolved.get("temperature").unwrap();
        assert_eq!(temperature.value, ConfigValue::Float(0.7));
        assert!(matches!(
            temperature.source,
            ConfigSource::GlobalUser { .. }
        ));
        assert_eq!(temperature.shadowed.len(), 2);
    }

    #[test]
    fn process_config_is_sparse_versioned_and_revalidated() {
        let mut global = ConfigLayer::default();
        global.set_cli("kernel.mw16", "true").unwrap();
        global.set_cli("diagnostic.kernel.gemv_rows", "4").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();

        assert_eq!(process.legacy_value("HIPFIRE_MW16").as_deref(), Some("1"));
        assert_eq!(
            process.legacy_value("HIPFIRE_GEMV_ROWS").as_deref(),
            Some("4")
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_RDNA3_HFQ4_QKV_WAVE64"),
            None,
            "architecture-sensitive bridge defaults remain absent"
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_DEVICES"),
            None,
            "nullable TOML fields remain absent instead of becoming empty strings"
        );

        let encoded = serde_json::to_string(&process).unwrap();
        let decoded: ProcessConfig = serde_json::from_str(&encoded).unwrap();
        decoded.validate().unwrap();

        let wrong_version = encoded.replace("\"schema_version\":1", "\"schema_version\":2");
        let decoded: ProcessConfig = serde_json::from_str(&wrong_version).unwrap();
        assert!(decoded.validate().is_err());
        assert!(serde_json::from_str::<ProcessConfig>(
            r#"{"schema_version":1,"values":{"values":{}},"unknown":true}"#
        )
        .is_err());
    }

    #[test]
    fn diagnostic_replay_route_proof_log_lowers_from_toml() {
        let mut global = ConfigLayer::default();
        global
            .set_cli("diagnostic.replay.route_proof_log", "true")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(
            process
                .legacy_value("HIPFIRE_REPLAY_ROUTE_PROOF_LOG")
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            process.values.get("diagnostic.replay.route_proof_log"),
            Some(&ConfigValue::Bool(true))
        );
    }

    #[test]
    fn device_visibility_is_one_synchronized_physical_list() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("hardware.devices", " 3, 1 ").unwrap();
        let process = ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: PathBuf::from("config.toml"),
                },
                layer,
            }])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            synchronized_device_visibility(&process, Some("0"), Some("2")).unwrap(),
            Some(DeviceVisibility {
                rocr: "3,1".into(),
                hip: "0,1".into(),
            }),
            "explicit TOML overrides and synchronizes both inherited backends"
        );

        let defaults = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        assert_eq!(
            synchronized_device_visibility(&defaults, Some("2"), None).unwrap(),
            Some(DeviceVisibility {
                rocr: "2".into(),
                hip: "0".into(),
            }),
            "a lone physical filter is lowered to ROCr physical plus HIP logical"
        );
        assert_eq!(
            synchronized_device_visibility(&defaults, Some("0"), Some("2")).unwrap(),
            Some(DeviceVisibility {
                rocr: "2".into(),
                hip: "0".into(),
            })
        );
        assert!(synchronized_device_visibility(&defaults, Some("1"), Some("2")).is_err());
    }

    #[test]
    fn invalid_values_fail_closed() {
        let mut layer = ConfigLayer::default();
        assert!(layer.set_cli("top_p", "0").is_err());
        assert!(layer.set_cli("port", "70000").is_err());
        assert!(layer.set_cli("kv_cache", "magic4").is_err());
        assert!(layer.set_cli("dflash_ngram_block", "auto").is_ok());
        assert!(layer.set_cli("dflash_ngram_block", "false").is_ok());
    }

    #[test]
    fn documented_config_profiles_match_the_schema() {
        let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/configs");
        for name in ["user.toml", "developer.toml", "redline-pm4.toml"] {
            let path = docs.join(name);
            load_toml_layer(&path).unwrap_or_else(|error| {
                panic!("{} is not a valid config profile: {error}", path.display())
            });
        }
    }

    #[test]
    fn apply_config_profile_replaces_entire_sparse_layer() {
        let root = temp_root("profile-replace");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer
            .set("generation.temperature", ConfigValue::Float(0.42))
            .unwrap();
        layer
            .set(
                "developer.custom_experiment",
                ConfigValue::String("drop-me".into()),
            )
            .unwrap();

        apply_config_profile(&mut layer, &paths, "dev").unwrap();

        let expected = load_config_profile(&paths, "dev").unwrap();
        assert_eq!(layer, expected);
        assert!(layer.get("generation.temperature").is_none());
        assert!(layer.get("developer.custom_experiment").is_none());
        assert_eq!(
            detect_config_profile(&paths, &layer).as_deref(),
            Some("dev")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn apply_config_profile_materializes_builtin_bundles() {
        let root = temp_root("profile-builtins");
        let paths = ConfigPaths::under(&root);
        for name in CONFIG_PROFILE_NAMES {
            let mut layer = ConfigLayer::default();
            apply_config_profile(&mut layer, &paths, name).unwrap();
            let expected = builtin_config_profile_layer(name).unwrap();
            assert_eq!(layer, expected, "profile {name}");
            assert_eq!(
                detect_config_profile(&paths, &layer).as_deref(),
                Some(*name)
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn hip_profile_is_an_explicit_redline_opt_out() {
        let paths = ConfigPaths::under(temp_root("profile-hip"));
        let layer = load_config_profile(&paths, "hip").unwrap();
        assert_eq!(
            layer.get("replay.backend"),
            Some(&ConfigValue::String("hip".to_owned()))
        );
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn apply_config_profile_rejects_unknown_names() {
        let root = temp_root("profile-unknown");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set("serve.port", ConfigValue::Integer(9)).unwrap();
        let before = layer.clone();
        let err = apply_config_profile(&mut layer, &paths, "staging").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("unknown profile 'staging'"), "{message}");
        assert_eq!(layer, before, "failed apply must not mutate layer");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn create_config_profile_snapshots_and_rejects_duplicates() {
        let root = temp_root("profile-create");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("memory.kv_cache", "q8").unwrap();
        layer.set_cli("serve.port", "12000").unwrap();

        let path = create_config_profile(&paths, "lab", &layer).unwrap();
        assert_eq!(path, paths.profiles.join("lab.toml"));
        assert!(path.is_file());
        let loaded = load_config_profile(&paths, "lab").unwrap();
        assert_eq!(loaded, layer);

        let err = create_config_profile(&paths, "lab", &layer).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{}", err);
        let builtin = create_config_profile(&paths, "default", &layer).unwrap_err();
        assert!(builtin.to_string().contains("built-in"), "{}", builtin);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn validate_config_profile_name_rejects_traversal_and_invalid() {
        assert!(validate_config_profile_name("lab").is_ok());
        assert!(validate_config_profile_name("lab-1").is_ok());
        for bad in [
            "",
            "default",
            "dev",
            "redline",
            "hip",
            "..",
            "../x",
            "a/b",
            "a\\b",
            "has space",
            "ü",
        ] {
            assert!(
                validate_config_profile_name(bad).is_err(),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn list_config_profiles_includes_builtins_and_custom() {
        let root = temp_root("profile-list");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("serve.host", "127.0.0.1").unwrap();
        create_config_profile(&paths, "zeta", &layer).unwrap();
        create_config_profile(&paths, "alpha", &layer).unwrap();
        let names: Vec<_> = list_config_profiles(&paths)
            .unwrap()
            .into_iter()
            .map(|entry| (entry.name, entry.kind))
            .collect();
        assert_eq!(
            names,
            vec![
                ("default".into(), ConfigProfileKind::Builtin),
                ("dev".into(), ConfigProfileKind::Builtin),
                ("hip".into(), ConfigProfileKind::Builtin),
                ("redline".into(), ConfigProfileKind::Builtin),
                ("alpha".into(), ConfigProfileKind::Custom),
                ("zeta".into(), ConfigProfileKind::Custom),
            ]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn config_profile_bundles_match_documented_examples() {
        let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/configs");
        let paths = ConfigPaths::under(temp_root("profile-docs"));
        let pairs = [
            ("default", "user.toml"),
            ("dev", "developer.toml"),
            ("redline", "redline-pm4.toml"),
        ];
        for (name, file) in pairs {
            let path = docs.join(file);
            let example = load_toml_layer(&path)
                .unwrap_or_else(|error| panic!("{} failed to load: {error}", path.display()));
            let applied = load_config_profile(&paths, name).unwrap();
            assert_eq!(
                applied, example,
                "profile {name} must match docs/configs/{file}"
            );
        }
        let _ = fs::remove_dir_all(paths.root);
    }
}
