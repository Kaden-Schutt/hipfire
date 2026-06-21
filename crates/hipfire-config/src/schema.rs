// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Config schema metadata shared by CLI, daemon operator APIs, TUI/WebUI, and
//! generated docs.

use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigScope {
    Global,
    Host,
    Node,
    Pool,
    Model,
    Runtime,
    Eval,
    Training,
    Request,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigMutability {
    Static,
    LoadTime,
    RuntimeReloadable,
    RequestOnly,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RestartImpact {
    None,
    ReloadModel,
    RestartDaemon,
    RestartService,
    ReconnectClients,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigType {
    Bool,
    U8,
    U16,
    U32,
    I32,
    F64,
    String,
    Path,
    Enum { values: &'static [&'static str] },
    Json,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "kind", content = "condition", rename_all = "snake_case")]
pub enum Requirement {
    Optional,
    Required,
    RequiredWhen(&'static str),
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ConfigField {
    pub key: &'static str,
    #[serde(rename = "type")]
    pub ty: ConfigType,
    pub requirement: Requirement,
    pub default: Option<&'static str>,
    pub scopes: &'static [ConfigScope],
    pub mutability: ConfigMutability,
    pub owner: &'static str,
    pub description: &'static str,
    pub validation: Option<&'static str>,
    pub secret: bool,
    pub restart_impact: RestartImpact,
    pub env: &'static [&'static str],
}

const GLOBAL_RUNTIME: &[ConfigScope] = &[ConfigScope::Global, ConfigScope::Runtime];
const GLOBAL_MODEL_RUNTIME: &[ConfigScope] = &[
    ConfigScope::Global,
    ConfigScope::Model,
    ConfigScope::Runtime,
];
const GLOBAL_MODEL_REQUEST: &[ConfigScope] = &[
    ConfigScope::Global,
    ConfigScope::Model,
    ConfigScope::Request,
];

macro_rules! field {
    (
        $key:literal,
        $ty:expr,
        $requirement:expr,
        $default:expr,
        $scopes:expr,
        $mutability:expr,
        $description:literal
        $(, validation: $validation:literal)?
        $(, env: [$($env:literal),* $(,)?])?
    ) => {
        ConfigField {
            key: $key,
            ty: $ty,
            requirement: $requirement,
            default: $default,
            scopes: $scopes,
            mutability: $mutability,
            owner: "hipfire-config",
            description: $description,
            validation: field!(@validation $($validation)?),
            secret: false,
            restart_impact: RestartImpact::None,
            env: field!(@env $($($env),*)?),
        }
    };
    (@validation $validation:literal) => {
        Some($validation)
    };
    (@validation) => {
        None
    };
    (@env $($env:literal),*) => {
        &[$($env),*]
    };
    (@env) => {
        &[]
    };
}

pub static CONFIG_FIELDS: &[ConfigField] = &[
    field!(
        "host",
        ConfigType::String,
        Requirement::Optional,
        Some("127.0.0.1"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Bind host for the OpenAI-compatible HTTP server. Defaults to loopback; set to 0.0.0.0 to expose on all interfaces.",
        validation: "valid IP address or hostname"
    ),
    field!(
        "port",
        ConfigType::U16,
        Requirement::Optional,
        Some("11435"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Bind port for the OpenAI-compatible HTTP server."
    ),
    field!(
        "cors_allowed_origins",
        ConfigType::Json,
        Requirement::Optional,
        Some("[]"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Browser origins allowed to call the HTTP API cross-origin. Empty disables CORS (same-origin only); [\"*\"] allows any origin; otherwise an explicit allowlist such as [\"http://localhost:8080\"].",
        validation: "JSON array of origin strings, or [\"*\"]"
    ),
    field!(
        "admin_user",
        ConfigType::String,
        Requirement::Optional,
        Some("admin"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Username for the /admin console login. The password is set separately with `hipfire admin set-password` (argon2id hash stored in ~/.hipfire/admin.passwd, never in config)."
    ),
    field!(
        "default_model",
        ConfigType::String,
        Requirement::Optional,
        None,
        GLOBAL_RUNTIME,
        ConfigMutability::LoadTime,
        "Model tag, alias, or path to pre-load or use by default."
    ),
    field!(
        "max_seq",
        ConfigType::U32,
        Requirement::Optional,
        Some("4096"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Maximum context/KV-cache capacity allocated at model load."
    ),
    field!(
        "max_tokens",
        ConfigType::U32,
        Requirement::Optional,
        Some("512"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default maximum number of generated tokens per request."
    ),
    field!(
        "temperature",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.3"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default sampling temperature.",
        validation: "0.0.."
    ),
    field!(
        "top_p",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.8"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default nucleus sampling probability.",
        validation: "0.0..=1.0"
    ),
    field!(
        "repeat_penalty",
        ConfigType::F64,
        Requirement::Optional,
        Some("1.05"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default repeat penalty for generated text.",
        validation: "0.0.."
    ),
    field!(
        "idle_timeout",
        ConfigType::U32,
        Requirement::Optional,
        Some("300"),
        GLOBAL_RUNTIME,
        ConfigMutability::RuntimeReloadable,
        "Seconds of inactivity before the server may evict an idle model."
    ),
    field!(
        "kv_cache",
        ConfigType::Enum {
            values: &["auto", "q8", "asym2", "asym3", "asym4"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "KV-cache precision and memory policy."
    ),
    field!(
        "kv_adaptive",
        ConfigType::Enum {
            values: &["off", "auto"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Adaptive KV-cache policy."
    ),
    field!(
        "flash_mode",
        ConfigType::Enum {
            values: &["auto", "always", "never"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Flash-attention selection policy."
    ),
    field!(
        "dflash_mode",
        ConfigType::Enum {
            values: &["off", "auto", "on"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "DFlash speculative decode mode."
    ),
    field!(
        "dflash_adaptive_b",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Whether DFlash may adapt draft batch size."
    ),
    field!(
        "dflash_ngram_block",
        ConfigType::Json,
        Requirement::Optional,
        Some("\"auto\""),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "DFlash n-gram blocking policy; accepts boolean or auto."
    ),
    field!(
        "mtp_mode",
        ConfigType::Enum {
            values: &["auto", "off", "on"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Multi-token prediction sidecar mode."
    ),
    field!(
        "mtp_k",
        ConfigType::U32,
        Requirement::Optional,
        Some("3"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Number of MTP candidate tokens to consider."
    ),
    field!(
        "thinking",
        ConfigType::Enum {
            values: &["off", "on"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Reasoning/thinking display policy for compatible models."
    ),
    field!(
        "gpu_slab_load",
        ConfigType::Enum {
            values: &["auto", "off", "on"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "GPU slab loading policy for model weights."
    ),
    field!(
        "prompt_normalize",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Whether prompts are normalized before tokenization."
    ),
    field!(
        "cask_auto_attach",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Whether compatible CASK/TriAttention sidecars may auto-attach."
    ),
    field!(
        "cask_sidecar",
        ConfigType::Path,
        Requirement::RequiredWhen("cask == true && cask_auto_attach == false"),
        None,
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Explicit CASK/TriAttention sidecar path."
    ),
    field!(
        "cask",
        ConfigType::Bool,
        Requirement::Optional,
        Some("false"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Enable CASK/TriAttention behavior where supported."
    ),
    field!(
        "cask_budget",
        ConfigType::U32,
        Requirement::Optional,
        Some("512"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "CASK token or block budget."
    ),
    field!(
        "cask_beta",
        ConfigType::U32,
        Requirement::Optional,
        Some("128"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "CASK beta control value."
    ),
    field!(
        "cask_core_frac",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.5"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Fraction of CASK core candidates to keep.",
        validation: "0.0..=1.0"
    ),
    field!(
        "cask_fold_m",
        ConfigType::U32,
        Requirement::Optional,
        Some("2"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "CASK fold factor."
    ),
    field!(
        "mmq_screen",
        ConfigType::Enum {
            values: &["auto", "off", "on"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "MMQ safety screening mode."
    ),
    field!(
        "mmq_screen_threshold",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.10"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "MMQ screening rejection threshold.",
        validation: "0.0..=1.0"
    ),
    field!(
        "prefill_compression",
        ConfigType::Enum {
            values: &["off", "auto", "on"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Long-context prefill compression mode."
    ),
    field!(
        "prefill_threshold",
        ConfigType::U32,
        Requirement::Optional,
        Some("32768"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Context length threshold for prefill compression."
    ),
    field!(
        "prefill_keep_ratio",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.05"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Fraction of prefill blocks to keep under compression.",
        validation: "0.0..=1.0"
    ),
    field!(
        "prefill_alpha",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.85"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Prefill compression scoring alpha.",
        validation: "0.0..=1.0"
    ),
    field!(
        "prefill_min_keep",
        ConfigType::U32,
        Requirement::Optional,
        Some("2048"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Minimum tokens or blocks retained during prefill compression."
    ),
    field!(
        "prefill_sink",
        ConfigType::U32,
        Requirement::Optional,
        Some("256"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Prefix sink size retained during prefill compression."
    ),
    field!(
        "prefill_recent",
        ConfigType::U32,
        Requirement::Optional,
        Some("1024"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Recent context size retained during prefill compression."
    ),
    field!(
        "prefill_block",
        ConfigType::U32,
        Requirement::Optional,
        Some("128"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Block size used by prefill compression."
    ),
    field!(
        "prefill_drafter",
        ConfigType::Path,
        Requirement::RequiredWhen("prefill_compression != 'off' && prefill_drafter_device >= 0"),
        None,
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Optional drafter artifact for prefill compression."
    ),
    field!(
        "prefill_drafter_device",
        ConfigType::I32,
        Requirement::Optional,
        Some("-1"),
        &[
            ConfigScope::Global,
            ConfigScope::Host,
            ConfigScope::Node,
            ConfigScope::Model
        ],
        ConfigMutability::LoadTime,
        "Preferred accelerator device for the prefill drafter."
    ),
    field!(
        "prefill_profile",
        ConfigType::Bool,
        Requirement::Optional,
        Some("false"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Emit prefill compression profiling details."
    ),
    field!(
        "prefill_sparse_threshold",
        ConfigType::U32,
        Requirement::Optional,
        Some("32768"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Context threshold for sparse prefill behavior."
    ),
    ConfigField {
        key: "model_overrides",
        ty: ConfigType::Json,
        requirement: Requirement::Optional,
        default: Some("{}"),
        scopes: &[ConfigScope::Global, ConfigScope::Model],
        mutability: ConfigMutability::LoadTime,
        owner: "hipfire-config",
        description: "Sparse per-model override map layered on top of global config.",
        validation: Some("object keyed by model tag"),
        secret: false,
        restart_impact: RestartImpact::ReloadModel,
        env: &[],
    },
];

pub fn config_schema() -> &'static [ConfigField] {
    CONFIG_FIELDS
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{config_schema, Requirement};

    #[test]
    fn config_schema_keys_are_unique() {
        let fields = config_schema();
        let mut seen = BTreeSet::new();
        for field in fields {
            assert!(seen.insert(field.key), "duplicate config key {}", field.key);
        }
    }

    #[test]
    fn config_schema_has_conditional_required_fields() {
        assert!(config_schema()
            .iter()
            .any(|field| matches!(field.requirement, Requirement::RequiredWhen(_))));
    }
}
