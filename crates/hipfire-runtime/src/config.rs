// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable env-var resolution for hipfire-runtime.
//!
//! All `HIPFIRE_*` env vars are read exactly once via the global
//! `RuntimeConfig::get()` accessor. Runtime hot paths access config
//! fields instead of hitting `std::env::var` on every call.

use std::sync::OnceLock;

/// Automatic Redline runtime default for single-GPU MQ4R models.
pub fn mq4r_redline_default(gpu_arch: &str, model_path: &str, pp: usize, tp: usize) -> bool {
    matches!(gpu_arch, "gfx1100" | "gfx1151" | "gfx1201")
        && pp == 1
        && tp == 1
        && std::path::Path::new(model_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mq4r"))
}

/// Automatic retained-Redline admission for a loaded model.
///
/// MQ4R keeps its existing cross-model policy with one temporary carve-out:
/// Muse Glimmer (arch 14) IS an MQ4R Redline SKU, but it is not lowered to
/// Redline PM4 yet, so automatic admission is withheld until that lowering
/// lands — the extension-only policy would otherwise route Glimmer onto a
/// dispatch path that cannot yet execute it. Delete this carve-out together
/// with the lowering. DeepSeek4 MQ2R is narrower still: only the certified
/// gfx1151 single-GPU AR route is admitted, and an installed drafter keeps the
/// model on its speculative execution path.
pub fn retained_redline_default(
    gpu_arch: &str,
    model_arch: &str,
    model_path: &str,
    pp: usize,
    tp: usize,
    has_drafter: bool,
) -> bool {
    if model_arch.eq_ignore_ascii_case("muse_glimmer") {
        return false;
    }
    if mq4r_redline_default(gpu_arch, model_path, pp, tp) {
        return true;
    }
    gpu_arch.eq_ignore_ascii_case("gfx1151")
        && model_arch.eq_ignore_ascii_case("deepseek4")
        && pp == 1
        && tp == 1
        && !has_drafter
        && std::path::Path::new(model_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mq2r"))
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
    /// Proof that device visibility was successfully applied. When `Some`,
    /// the hardware resolver lowers physical selectors to logical `0..N-1`;
    /// when `None`, physical IDs are preserved. Direct callers without
    /// visibility preserve physical IDs.
    pub visibility: Option<hipfire_config::DeviceVisibility>,
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
        Self::from_process_config_with_visibility(config, None)
    }

    /// Construct with explicit visibility proof. Physical selectors are
    /// preserved in `devices`; lowering to logical happens only in
    /// `hipfire_hardware::resolve_device_ids` when the proof matches the
    /// current process environment. Without proof, physical IDs are preserved.
    pub fn from_process_config_with_visibility(
        config: &hipfire_config::ProcessConfig,
        visibility: Option<hipfire_config::DeviceVisibility>,
    ) -> Self {
        Self::from_lookup(|name| config.legacy_value(name), visibility)
    }

    fn from_lookup(
        mut value: impl FnMut(&str) -> Option<String>,
        visibility: Option<hipfire_config::DeviceVisibility>,
    ) -> Self {
        let normalize_prompt = match value("HIPFIRE_NORMALIZE_PROMPT").as_deref() {
            Some("0") | Some("false") | Some("off") | Some("no") => false,
            _ => true,
        };

        let prompt_heat_json = value("HIPFIRE_PROMPT_HEAT_JSON").as_deref() == Some("1");
        let prompt_heat_limit: usize = value("HIPFIRE_PROMPT_HEAT_LIMIT")
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);

        // Resolve devices: preserve physical list verbatim (validated) in all
        // cases. The lowering to logical 0..N-1 happens only in
        // `hipfire_hardware::resolve_device_ids` when `visibility` proof is
        // present and matches the current process environment. This ensures
        // direct callers without proof preserve physical IDs, while daemon
        // paths with proof get logical IDs after hardware verification.
        // Empty/malformed lists are preserved for hardware to fail closed.
        let raw_devices = value("HIPFIRE_DEVICES");
        let devices = match &raw_devices {
            Some(raw) => {
                if raw.trim().is_empty() {
                    Some(raw.clone())
                } else {
                    let parts: Vec<&str> = raw.split(',').collect();
                    if parts.iter().any(|p| p.trim().is_empty()) {
                        Some(raw.clone())
                    } else {
                        Some(parts.iter().map(|p| p.trim()).collect::<Vec<_>>().join(","))
                    }
                }
            }
            None => None,
        };
        // Keep visibility for device_resolve_opts threading.
        let _ = &visibility;

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
            prompt_cache_unbounded: value("HIPFIRE_PROMPT_CACHE_UNBOUNDED").as_deref() == Some("1"),
            experimental_budget_alert: value("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT").as_deref()
                == Some("1"),
            max_total_think_tokens: value("HIPFIRE_MAX_TOTAL_THINK_TOKENS")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            devices,
            visibility,
            allow_mixed_arch: value("HIPFIRE_ALLOW_MIXED_ARCH").as_deref() == Some("1"),
            uniform_vram_tolerance_gb: value("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB")
                .and_then(|s| s.parse().ok()),
            mtp_mode: value("HIPFIRE_MTP_MODE").unwrap_or_else(|| "auto".into()),
            mtp_k: value("HIPFIRE_MTP_K")
                .and_then(|value| value.parse().ok())
                .unwrap_or(3),
        }
    }

    /// Lower the already-resolved hardware settings into the hardware leaf's
    /// dependency-free construction value. `devices` is preserved physical;
    /// `visibility` proof is threaded to `hipfire_hardware` so it can lower
    /// to logical `0..N-1` only when the proof matches the current process
    /// environment.
    pub fn device_resolve_opts(&self) -> hipfire_hardware::DeviceResolveOpts {
        hipfire_hardware::DeviceResolveOpts {
            tp_use_rccl: self.tp_use_rccl,
            devices: self.devices.clone(),
            allow_mixed_arch: self.allow_mixed_arch,
            uniform_vram_tolerance_gb: self.uniform_vram_tolerance_gb,
            visibility: self.visibility.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{mq4r_redline_default, retained_redline_default, RuntimeConfig};
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
        // Without visibility proof, physical IDs are preserved, not lowered to 0,1.
        assert_eq!(config.devices.as_deref(), Some("2,3"));
        assert!(config.visibility.is_none());
        let opts = config.device_resolve_opts();
        assert_eq!(opts.devices.as_deref(), Some("2,3"));
        assert!(opts.visibility.is_none());
        assert!(config.prefill_batched, "sparse arch defaults remain intact");

        // With visibility proof, hardware will lower physical 2,3 to logical 0,1
        // (runtime keeps physical, but opts carries proof).
        let vis = hipfire_config::DeviceVisibility {
            rocr: "2,3".into(),
            hip: "0,1".into(),
        };
        let config_with_vis =
            RuntimeConfig::from_process_config_with_visibility(&process, Some(vis.clone()));
        assert_eq!(config_with_vis.devices.as_deref(), Some("2,3"));
        assert_eq!(config_with_vis.visibility, Some(vis.clone()));
        let opts_with_vis = config_with_vis.device_resolve_opts();
        assert_eq!(opts_with_vis.devices.as_deref(), Some("2,3"));
        assert_eq!(opts_with_vis.visibility, Some(vis));
    }

    #[test]
    fn redline_default_is_limited_to_exact_runtime_arches_and_single_gpu_mq4r() {
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(mq4r_redline_default(
                gpu_arch,
                "/models/qwen3.6-35b-a3b.mq4r",
                1,
                1,
            ));
        }

        assert!(mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-27b.mq4r",
            1,
            1,
        ));

        assert!(!mq4r_redline_default(
            "gfx1200",
            "/models/QWEN3.6-35B-A3B.MQ4R",
            1,
            1,
        ));
        assert!(!mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4",
            1,
            1,
        ));
        assert!(!mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4r",
            2,
            1,
        ));
        assert!(!mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4r",
            1,
            2,
        ));
    }

    #[test]
    fn muse_glimmer_mq4r_defers_redline_until_pm4_lowering() {
        // Glimmer's trunk is a genuine MQ4R Redline SKU, but it is not lowered
        // to Redline PM4 yet, so the extension-only policy must not admit it.
        // Retire this test with the carve-out when the lowering lands.
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(
                mq4r_redline_default(gpu_arch, "/models/muse-glimmer-30b.mq4r", 1, 1),
                "extension-only policy still matches the Glimmer trunk"
            );
            assert!(
                !retained_redline_default(
                    gpu_arch,
                    "muse_glimmer",
                    "/models/muse-glimmer-30b.mq4r",
                    1,
                    1,
                    false,
                ),
                "glimmer must stay on HIP by default on {gpu_arch}"
            );
            assert!(
                !retained_redline_default(
                    gpu_arch,
                    "muse_glimmer",
                    "/models/muse-glimmer-30b.mq4r",
                    1,
                    1,
                    true,
                ),
                "a loaded drafter does not re-admit glimmer on {gpu_arch}"
            );
            // The carve-out is glimmer-scoped, not a global MQ4R retreat.
            assert!(
                retained_redline_default(
                    gpu_arch,
                    "qwen3_6_moe",
                    "/models/qwen3.6-35b-a3b.mq4r",
                    1,
                    1,
                    false,
                ),
                "qwen MQ4R keeps auto-Redline on {gpu_arch}"
            );
        }
    }

    #[test]
    fn deepseek4_mq2r_redline_default_requires_gfx1151_ar() {
        assert!(retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            true,
        ));
        assert!(!retained_redline_default(
            "gfx1100",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "qwen3_5_moe",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            2,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2",
            1,
            1,
            false,
        ));
    }
}
