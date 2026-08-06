// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable env-var resolution for hipfire-runtime.
//!
//! All `HIPFIRE_*` env vars are read exactly once via the global
//! `RuntimeConfig::get()` accessor. Runtime hot paths access config
//! fields instead of hitting `std::env::var` on every call.

use hipfire_hardware::{DeviceMesh, DimKind};
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
    /// Debug: `HIPFIRE_EMULATE_GPUS=N` makes the engine treat the single
    /// physical GPU as N logical devices (all aliased to device 0), so the
    /// multi-GPU (PP / EP) paths run on one card. Values < 2 = off. See
    /// docs/superpowers/specs/2026-07-03-emulate-gpus-dual-gpu-design.md.
    pub emulate_gpus: Option<usize>,
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

    /// Environment-only construction (legacy test/CLI path). Reads the same
    /// `HIPFIRE_*` variables `from_process_config` resolves, so single-axis
    /// behavior is unchanged.
    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
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
            prompt_cache_unbounded: value("HIPFIRE_PROMPT_CACHE_UNBOUNDED").as_deref() == Some("1"),
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
            // Debug: `HIPFIRE_EMULATE_GPUS=N` aliases the single physical GPU
            // as N logical devices so the multi-GPU (PP / EP) paths run on one
            // card. Values < 2 = off.
            emulate_gpus: std::env::var("HIPFIRE_EMULATE_GPUS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 2),
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

pub fn validate_parallel_axes(pp: usize, tp: usize, ep: usize) -> Result<(), String> {
    let _ = pp;
    if tp > 1 && ep > 1 {
        return Err(
            "tensor-parallel (tp) and expert-parallel (ep) are mutually exclusive (COMP-001)"
                .to_owned(),
        );
    }
    Ok(())
}

/// Decide the effective `(pp, tp, ep)` parallelism degrees for a load. An
/// explicitly-requested `pp`/`tp`/`ep > 1` (from the load message) always wins.
/// Only when NONE is requested does `HIPFIRE_EMULATE_GPUS` default the mode — to
/// EP (the primary working multi-GPU path, expert-parallel); PP and real
/// tensor-parallel (`tp`) stay opt-in via an explicit degree. The
/// mutual-exclusion check remains with the caller.
pub fn resolve_parallelism(
    pp: usize,
    tp: usize,
    ep: usize,
    emulate_gpus: Option<usize>,
) -> (usize, usize, usize) {
    if pp == 1 && tp == 1 && ep == 1 {
        if let Some(n) = emulate_gpus {
            if n >= 2 {
                // Emulate defaults to EP (the Ep axis) — unchanged from the
                // pre-disentanglement behavior where `tp` aliased EP.
                return (pp, tp, n);
            }
        }
    }
    (pp, tp, ep)
}

/// Resolve the `pp`/`tp`/`ep` load knobs (with `HIPFIRE_EMULATE_GPUS` defaulting)
/// to a [`DeviceMesh`] — the mesh producer as the daemon adopts mesh-driven
/// load/dispatch.
///
/// **EP↔TP disentangled (PB-TP5 prep):** each degree maps to its OWN axis —
/// `pp`→`Pp` (pipeline), `ep`→`Ep` (expert-parallel, MoE routed experts),
/// `tp`→`Tp` (real tensor-parallel, dense row/col sharding). `tp` no longer
/// aliases the `Ep` axis. Single-axis only in this phase (precedence
/// `pp > ep > tp`); composed 2×N meshes are a later phase. Degenerate: none set →
/// single-device (1×1) mesh.
pub fn resolve_mesh(pp: usize, tp: usize, ep: usize, emulate_gpus: Option<usize>) -> DeviceMesh {
    let (pp, tp, ep) = resolve_parallelism(pp, tp, ep, emulate_gpus);
    if pp > 1 {
        DeviceMesh::rect(&[(DimKind::Pp, pp)])
    } else if ep > 1 {
        DeviceMesh::rect(&[(DimKind::Ep, ep)])
    } else if tp > 1 {
        DeviceMesh::rect(&[(DimKind::Tp, tp)])
    } else {
        DeviceMesh::single()
    }
}

/// Parse + length-validate the `HIPFIRE_PP_LAYERS` ragged-band spec at the load
/// edge. `raw` is the raw env value (`std::env::var(..).ok()`). Empty / absent →
/// `Ok(None)` (uniform banding). A comma list of `pp` counts → `Ok(Some(..))`.
/// The sum-vs-`n_layers` check is deferred to the carrier (which knows the
/// model config); this only enforces the shape (`len == pp`).
pub fn parse_pp_layers(raw: Option<String>, pp: usize) -> Result<Option<Vec<usize>>, String> {
    let Some(spec) = raw.filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let counts: Vec<usize> = spec
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<usize>, _>>()
        .map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
    if counts.len() != pp {
        return Err(format!(
            "HIPFIRE_PP_LAYERS has {} entries, expected pp={}",
            counts.len(),
            pp
        ));
    }
    Ok(Some(counts))
}

#[cfg(test)]
mod tests {
    use super::{mq4r_redline_default, resolve_mesh, RuntimeConfig};
    use hipfire_config::{resolve, ConfigLayer, ConfigSource, NamedLayer, ProcessConfig};
    use hipfire_hardware::DimKind;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn resolve_mesh_maps_knobs_to_axes() {
        // single-GPU: no axes, one device.
        assert_eq!(resolve_mesh(1, 1, 1, None).n_devices(), 1);
        // pp>1 → Pp axis.
        let pp = resolve_mesh(2, 1, 1, None);
        assert_eq!(pp.n_devices(), 2);
        assert!(pp.has_axis(DimKind::Pp));
        // ep>1 → Ep axis (expert-parallel).
        let ep = resolve_mesh(1, 1, 4, None);
        assert_eq!(ep.n_devices(), 4);
        assert!(ep.has_axis(DimKind::Ep));
        // tp>1 → Tp axis (real tensor-parallel — the disentanglement).
        let tp = resolve_mesh(1, 4, 1, None);
        assert_eq!(tp.n_devices(), 4);
        assert!(tp.has_axis(DimKind::Tp));
        // emulate defaults to Ep when nothing else is set.
        let em = resolve_mesh(1, 1, 1, Some(2));
        assert_eq!(em.n_devices(), 2);
        assert!(em.has_axis(DimKind::Ep));
        // explicit pp wins over emulate; ep wins over tp.
        assert!(resolve_mesh(2, 1, 1, Some(4)).has_axis(DimKind::Pp));
        assert!(resolve_mesh(1, 2, 2, None).has_axis(DimKind::Ep));
    }

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
    fn emulate_gpus_parses_and_filters() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("HIPFIRE_EMULATE_GPUS").ok();

        std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, Some(2));

        std::env::set_var("HIPFIRE_EMULATE_GPUS", "1"); // < 2 = off
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, None);

        std::env::set_var("HIPFIRE_EMULATE_GPUS", "abc"); // unparseable = off
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, None);

        std::env::remove_var("HIPFIRE_EMULATE_GPUS");
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, None);

        match prev {
            Some(v) => std::env::set_var("HIPFIRE_EMULATE_GPUS", v),
            None => std::env::remove_var("HIPFIRE_EMULATE_GPUS"),
        }
    }

    #[test]
    fn resolve_parallelism_defaults_ep_only_when_unset() {
        // Nothing requested + emulate -> default EP (the tp slot carries the degree).
        assert_eq!(super::resolve_parallelism(1, 1, 1, Some(2)), (1, 1, 2));
        // Explicit pp wins; no default.
        assert_eq!(super::resolve_parallelism(2, 1, 1, Some(2)), (2, 1, 1));
        // Explicit tp (real tensor-parallel) wins.
        assert_eq!(super::resolve_parallelism(1, 4, 1, Some(2)), (1, 4, 1));
        // Explicit ep (expert-parallel) wins.
        assert_eq!(super::resolve_parallelism(1, 1, 4, Some(2)), (1, 1, 4));
        // No emulate -> unchanged.
        assert_eq!(super::resolve_parallelism(1, 1, 1, None), (1, 1, 1));
    }

    #[test]
    fn validate_parallel_axes_rejects_tp_and_ep() {
        assert_eq!(
            super::validate_parallel_axes(1, 2, 2),
            Err(
                "tensor-parallel (tp) and expert-parallel (ep) are mutually exclusive (COMP-001)"
                    .to_owned()
            )
        );
    }

    #[test]
    fn validate_parallel_axes_accepts_other_axis_combinations() {
        assert_eq!(super::validate_parallel_axes(1, 2, 1), Ok(()));
        assert_eq!(super::validate_parallel_axes(1, 1, 2), Ok(()));
        assert_eq!(super::validate_parallel_axes(2, 1, 1), Ok(()));
    }

    #[test]
    fn parse_pp_layers_none_and_empty_are_uniform() {
        assert_eq!(super::parse_pp_layers(None, 2), Ok(None));
        assert_eq!(super::parse_pp_layers(Some(String::new()), 2), Ok(None));
    }

    #[test]
    fn parse_pp_layers_valid_split() {
        assert_eq!(
            super::parse_pp_layers(Some("2, 5, 3".to_string()), 3),
            Ok(Some(vec![2, 5, 3]))
        );
    }

    #[test]
    fn parse_pp_layers_len_mismatch_errors() {
        let err = super::parse_pp_layers(Some("2,5".to_string()), 3).unwrap_err();
        assert!(err.contains("has 2 entries"), "got: {err}");
        assert!(err.contains("expected pp=3"), "got: {err}");
    }

    #[test]
    fn parse_pp_layers_unparseable_errors() {
        let err = super::parse_pp_layers(Some("2,x,3".to_string()), 3).unwrap_err();
        assert!(err.contains("HIPFIRE_PP_LAYERS parse"), "got: {err}");
    }
}
