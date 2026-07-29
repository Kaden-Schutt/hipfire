// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable env-var resolution for hipfire-runtime.
//!
//! All `HIPFIRE_*` env vars are read exactly once via the global
//! `RuntimeConfig::get()` accessor. Runtime hot paths access config
//! fields instead of hitting `std::env::var` on every call.

use std::sync::OnceLock;

/// Automatic Redline **runtime** default for single-GPU Qwen A3B MQ4R.
///
/// Matches exact GPU arch `gfx1100`/`gfx1151`/`gfx1201`, `arch_id == 6`,
/// `pp=tp=1`, and case-insensitive `.mq4r`. `gfx1200` and all other arches,
/// formats, model families, and multi-GPU layouts remain opt-in. This is a
/// runtime default only — not Redline certification or registry admission.
/// Built-in `hip` config profile or an explicit backend selection disables it.
/// MQ4R is purpose-built for retained PM4 replay.
pub fn a3b_mq4r_redline_default(
    gpu_arch: &str,
    model_path: &str,
    model_arch_id: u32,
    pp: usize,
    tp: usize,
) -> bool {
    matches!(gpu_arch, "gfx1100" | "gfx1151" | "gfx1201")
        && model_arch_id == 6
        && pp == 1
        && tp == 1
        && std::path::Path::new(model_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mq4r"))
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub normalize_prompt: bool,
    pub prompt_token_heat: bool,
    pub prompt_heat_json: bool,
    pub prompt_heat_limit: usize,
    pub attention_flash_mode: String,
    pub dflash_ngram_block: Option<bool>,
    pub dflash_fast_sample: bool,
    pub draft_f16: bool,
    pub draft_gemm_dump: bool,
    pub draft_subphase: bool,
    pub prefill_batched: bool,
    pub flash_partials_batch: Option<usize>,
    pub lm_head_f16: String,
    /// Tensor-parallel RCCL all-reduce toggle. `None` (unset) → RCCL is used
    /// (default). `Some(false)` (HIPFIRE_TP_USE_RCCL=0) → opt out of the RCCL
    /// path. `Some(true)` → force on. Read by `multi_gpu::Gpus::ensure_rccl`.
    pub tp_use_rccl: Option<bool>,
    pub ngram_loop_threshold: usize,
    pub ngram_window: usize,
    pub ngram_draft: bool,
    pub ngram_k: usize,
    pub ngram_min_count: u32,
    pub kv_mode: String,
    pub kv_adaptive: String,
    pub chat_template_file: Option<String>,
    pub prompt_cache_capacity: usize,
    pub prompt_cache_unbounded: bool,
    pub experimental_budget_alert: bool,
    pub max_total_think_tokens: usize,
    pub devices: Option<String>,
    pub allow_mixed_arch: bool,
    pub uniform_vram_tolerance_gb: Option<f32>,
    pub mtp_mode: String,
    pub mtp_k: usize,
}

static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

pub fn get() -> &'static RuntimeConfig {
    CONFIG.get_or_init(|| {
        RuntimeConfig::from_process_config(hipfire_config::active_or_local_process_config())
    })
}

pub fn init() {
    get();
}

pub fn init_with(config: RuntimeConfig) -> std::result::Result<(), RuntimeConfig> {
    CONFIG.set(config)
}

impl RuntimeConfig {
    pub fn from_process_config(config: &hipfire_config::ProcessConfig) -> Self {
        Self::from_lookup(|name| config.legacy_value(name))
    }

    fn from_lookup(mut value: impl FnMut(&str) -> Option<String>) -> Self {
        let normalize_prompt = match value("HIPFIRE_NORMALIZE_PROMPT").as_deref() {
            Some("0") | Some("false") | Some("off") | Some("no") => false,
            _ => true,
        };

        let prompt_heat_json = value("HIPFIRE_PROMPT_HEAT_JSON").as_deref() == Some("1");
        let prompt_heat_limit: usize = value("HIPFIRE_PROMPT_HEAT_LIMIT")
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);

        Self {
            normalize_prompt,
            prompt_token_heat: value("HIPFIRE_PROMPT_TOKEN_HEAT").as_deref() == Some("1"),
            prompt_heat_json,
            prompt_heat_limit,
            attention_flash_mode: value("HIPFIRE_ATTN_FLASH").unwrap_or_else(|| "auto".into()),
            dflash_ngram_block: match value("HIPFIRE_DFLASH_NGRAM_BLOCK").as_deref() {
                Some("1") | Some("true") | Some("on") => Some(true),
                Some("0") | Some("false") | Some("off") => Some(false),
                _ => None,
            },
            dflash_fast_sample: value("HIPFIRE_DFLASH_FAST_SAMPLE").as_deref() != Some("0"),
            draft_f16: value("HIPFIRE_DRAFT_F16").as_deref() != Some("0"),
            draft_gemm_dump: value("HIPFIRE_DRAFT_GEMM_DUMP").as_deref() == Some("1"),
            draft_subphase: value("HIPFIRE_DRAFT_SUBPHASE").as_deref() == Some("1"),
            prefill_batched: value("HIPFIRE_PREFILL_BATCHED").as_deref() != Some("0"),
            flash_partials_batch: value("HIPFIRE_FLASH_PARTIALS_BATCH")
                .and_then(|s| s.parse::<usize>().ok()),
            lm_head_f16: value("HIPFIRE_LM_HEAD_F16").unwrap_or_else(|| "auto".into()),
            tp_use_rccl: value("HIPFIRE_TP_USE_RCCL")
                .as_deref()
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false")),
            // Default OFF (0 = disabled). The content-blind 4-gram guard can
            // false-positive on repeated markdown/list scaffolding inside
            // sampled reasoning and force-EOS before a visible answer. The
            // think-budget force-close handles runaway reasoning instead;
            // operators can opt this guard back in with an explicit threshold.
            ngram_loop_threshold: value("HIPFIRE_NGRAM_LOOP_THRESHOLD")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            ngram_window: value("HIPFIRE_NGRAM_WINDOW")
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            ngram_draft: matches!(
                value("HIPFIRE_NGRAM_DRAFT").as_deref(),
                Some("1") | Some("on")
            ),
            ngram_k: value("HIPFIRE_NGRAM_DRAFT_K")
                .and_then(|v| v.parse().ok())
                .unwrap_or(12),
            ngram_min_count: value("HIPFIRE_NGRAM_MIN_COUNT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            kv_mode: value("HIPFIRE_KV_MODE").unwrap_or_else(|| "auto".into()),
            kv_adaptive: value("HIPFIRE_KV_ADAPTIVE").unwrap_or_else(|| "off".into()),
            chat_template_file: value("HIPFIRE_CHAT_TEMPLATE_FILE")
                .filter(|value| !value.is_empty()),
            prompt_cache_capacity: value("HIPFIRE_PROMPT_CACHE_CAP")
                .and_then(|value| value.parse().ok())
                .unwrap_or(32),
            prompt_cache_unbounded: value("HIPFIRE_PROMPT_CACHE_UNBOUNDED").as_deref()
                == Some("1"),
            experimental_budget_alert: value("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT").as_deref()
                == Some("1"),
            max_total_think_tokens: value("HIPFIRE_MAX_TOTAL_THINK_TOKENS")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            // `hardware.devices` is installed as physical ROCr selectors and
            // matching logical HIP selectors before GPU initialization. The
            // engine therefore addresses the filtered set as logical 0..N-1.
            devices: value("HIPFIRE_DEVICES")
                .filter(|value| !value.is_empty())
                .map(|value| {
                    (0..value.split(',').count())
                        .map(|device| device.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                }),
            allow_mixed_arch: value("HIPFIRE_ALLOW_MIXED_ARCH").as_deref() == Some("1"),
            uniform_vram_tolerance_gb: value("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB")
                .and_then(|s| s.parse().ok()),
            mtp_mode: value("HIPFIRE_MTP_MODE").unwrap_or_else(|| "auto".into()),
            mtp_k: value("HIPFIRE_MTP_K")
                .and_then(|value| value.parse().ok())
                .unwrap_or(3),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{a3b_mq4r_redline_default, RuntimeConfig};
    use hipfire_config::{resolve, ConfigLayer, ConfigSource, NamedLayer, ProcessConfig};

    #[test]
    fn ngram_loop_guard_is_off_by_default() {
        let process = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        let cfg = RuntimeConfig::from_process_config(&process);
        assert_eq!(cfg.ngram_loop_threshold, 0);
    }

    #[test]
    fn normalize_prompt_accepts_no_as_false() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("prompt.normalize", "no").unwrap();
        let process = ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: "config.toml".into(),
                },
                layer,
            }])
            .unwrap(),
        )
        .unwrap();
        let cfg = RuntimeConfig::from_process_config(&process);
        assert!(!cfg.normalize_prompt);
    }

    #[test]
    fn process_config_populates_runtime_without_ambient_env() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("prompt.normalize", "false").unwrap();
        layer
            .set_cli("generation.loop_guard_threshold", "12")
            .unwrap();
        layer.set_cli("hardware.devices", "2,3").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let config = RuntimeConfig::from_process_config(&process);

        assert!(!config.normalize_prompt);
        assert_eq!(config.ngram_loop_threshold, 12);
        assert_eq!(config.devices.as_deref(), Some("0,1"));
        assert!(config.prefill_batched, "sparse arch defaults remain intact");
    }

    #[test]
    fn redline_default_is_limited_to_exact_runtime_arches_and_single_gpu_a3b_mq4r() {
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(a3b_mq4r_redline_default(
                gpu_arch,
                "/models/qwen3.6-35b-a3b.mq4r",
                6,
                1,
                1,
            ));
        }

        assert!(!a3b_mq4r_redline_default(
            "gfx1200",
            "/models/QWEN3.6-35B-A3B.MQ4R",
            6,
            1,
            1,
        ));
        assert!(!a3b_mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4",
            6,
            1,
            1,
        ));
        assert!(!a3b_mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4r",
            5,
            1,
            1,
        ));
        assert!(!a3b_mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4r",
            6,
            2,
            1,
        ));
        assert!(!a3b_mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4r",
            6,
            1,
            2,
        ));
    }
}
