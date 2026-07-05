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

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub normalize_prompt: bool,
    pub prompt_token_heat: bool,
    pub prompt_heat_json: bool,
    pub prompt_heat_limit: usize,
    pub dflash_draft: Option<String>,
    pub dflash_mode: String,
    pub draft_f16: bool,
    pub draft_gemm_dump: bool,
    pub draft_subphase: bool,
    pub ddtree_budget: usize,
    pub ddtree_topk: usize,
    pub prefill_batched: bool,
    pub flash_partials_batch: Option<usize>,
    /// Tensor-parallel RCCL all-reduce toggle. `None` (unset) → RCCL is used
    /// (default). `Some(false)` (HIPFIRE_TP_USE_RCCL=0) → opt out of the RCCL
    /// path. `Some(true)` → force on. Read by `multi_gpu::Gpus::ensure_rccl`.
    pub tp_use_rccl: Option<bool>,
    pub ngram_loop_threshold: usize,
    pub ngram_window: usize,
    pub devices: Option<String>,
    /// Debug: `HIPFIRE_EMULATE_GPUS=N` makes the engine treat the single
    /// physical GPU as N logical devices (all aliased to device 0), so the
    /// multi-GPU (PP / EP) paths run on one card. Values < 2 = off. See
    /// docs/superpowers/specs/2026-07-03-emulate-gpus-dual-gpu-design.md.
    pub emulate_gpus: Option<usize>,
    pub allow_mixed_arch: bool,
    pub uniform_vram_tolerance_gb: Option<f32>,
    pub lm_head_f16: String,
    pub mtp_mode: String,
    pub mtp_k: usize,
}

static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

pub fn get() -> &'static RuntimeConfig {
    CONFIG.get_or_init(RuntimeConfig::from_env)
}

pub fn init() {
    get();
}

impl RuntimeConfig {
    pub fn from_env() -> Self {
        let normalize_prompt = match std::env::var("HIPFIRE_NORMALIZE_PROMPT").ok().as_deref() {
            Some("0") | Some("false") | Some("off") | Some("no") => false,
            _ => true,
        };

        let prompt_token_heat =
            std::env::var("HIPFIRE_PROMPT_TOKEN_HEAT").ok().as_deref() == Some("1");
        let prompt_heat_json =
            std::env::var("HIPFIRE_PROMPT_HEAT_JSON").ok().as_deref() == Some("1");
        let prompt_heat_limit: usize = std::env::var("HIPFIRE_PROMPT_HEAT_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);

        Self {
            normalize_prompt,
            prompt_token_heat,
            prompt_heat_json,
            prompt_heat_limit,
            dflash_draft: std::env::var("HIPFIRE_DFLASH_DRAFT").ok(),
            dflash_mode: std::env::var("HIPFIRE_DFLASH_MODE").unwrap_or_else(|_| "off".to_string()),
            draft_f16: std::env::var("HIPFIRE_DRAFT_F16").ok().as_deref() != Some("0"),
            draft_gemm_dump: std::env::var("HIPFIRE_DRAFT_GEMM_DUMP").ok().as_deref() == Some("1"),
            draft_subphase: std::env::var("HIPFIRE_DRAFT_SUBPHASE").ok().as_deref() == Some("1"),
            ddtree_budget: std::env::var("HIPFIRE_DDTREE_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            ddtree_topk: std::env::var("HIPFIRE_DDTREE_TOPK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            prefill_batched: std::env::var("HIPFIRE_PREFILL_BATCHED").ok().as_deref() != Some("0"),
            flash_partials_batch: std::env::var("HIPFIRE_FLASH_PARTIALS_BATCH")
                .ok()
                .and_then(|s| s.parse::<usize>().ok()),
            tp_use_rccl: std::env::var("HIPFIRE_TP_USE_RCCL")
                .ok()
                .as_deref()
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false")),
            ngram_loop_threshold: std::env::var("HIPFIRE_NGRAM_LOOP_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            ngram_window: std::env::var("HIPFIRE_NGRAM_WINDOW")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            devices: std::env::var("HIPFIRE_DEVICES").ok(),
            emulate_gpus: std::env::var("HIPFIRE_EMULATE_GPUS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 2),
            allow_mixed_arch: std::env::var("HIPFIRE_ALLOW_MIXED_ARCH").ok().as_deref()
                == Some("1"),
            uniform_vram_tolerance_gb: std::env::var("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB")
                .ok()
                .and_then(|s| s.parse().ok()),
            lm_head_f16: std::env::var("HIPFIRE_LM_HEAD_F16")
                .unwrap_or_else(|_| "auto".to_string()),
            mtp_mode: std::env::var("HIPFIRE_MTP_MODE").unwrap_or_else(|_| "auto".to_string()),
            mtp_k: std::env::var("HIPFIRE_MTP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
        }
    }
}

/// Decide the effective `(pp, tp)` parallelism degrees for a load. An
/// explicitly-requested `pp > 1` or `tp > 1` (from the load message) always
/// wins. Only when neither is requested does `HIPFIRE_EMULATE_GPUS` default
/// the mode — to TP (EP), the primary multi-GPU path; PP stays opt-in via an
/// explicit `pp`. The pp/tp mutual-exclusion check remains with the caller.
pub fn resolve_parallelism(pp: usize, tp: usize, emulate_gpus: Option<usize>) -> (usize, usize) {
    if pp == 1 && tp == 1 {
        if let Some(n) = emulate_gpus {
            if n >= 2 {
                return (pp, n);
            }
        }
    }
    (pp, tp)
}

/// Resolve the `pp`/`tp` load knobs (with `HIPFIRE_EMULATE_GPUS` defaulting) to
/// a [`DeviceMesh`] — the mesh producer that replaces the flat
/// `resolve_parallelism` pair as the daemon adopts mesh-driven load/dispatch.
/// Today `tp` == expert-parallel (`Ep` axis); real row/col TP and composed 2×N
/// meshes are later phases. Degenerate: neither set → single-device (1×1) mesh.
pub fn resolve_mesh(pp: usize, tp: usize, emulate_gpus: Option<usize>) -> DeviceMesh {
    let (pp, tp) = resolve_parallelism(pp, tp, emulate_gpus);
    if pp > 1 {
        DeviceMesh::rect(&[(DimKind::Pp, pp)])
    } else if tp > 1 {
        DeviceMesh::rect(&[(DimKind::Ep, tp)])
    } else {
        DeviceMesh::single()
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_mesh, RuntimeConfig};
    use hipfire_hardware::DimKind;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn resolve_mesh_maps_knobs_to_axes() {
        // single-GPU: no axes, one device.
        assert_eq!(resolve_mesh(1, 1, None).n_devices(), 1);
        // pp>1 → Pp axis.
        let pp = resolve_mesh(2, 1, None);
        assert_eq!(pp.n_devices(), 2);
        assert!(pp.has_axis(DimKind::Pp));
        // tp>1 → Ep axis (tp == EP today).
        let tp = resolve_mesh(1, 4, None);
        assert_eq!(tp.n_devices(), 4);
        assert!(tp.has_axis(DimKind::Ep));
        // emulate defaults to Ep when neither pp nor tp set.
        let em = resolve_mesh(1, 1, Some(2));
        assert_eq!(em.n_devices(), 2);
        assert!(em.has_axis(DimKind::Ep));
        // explicit pp wins over emulate.
        assert!(resolve_mesh(2, 1, Some(4)).has_axis(DimKind::Pp));
    }

    #[test]
    fn normalize_prompt_accepts_no_as_false() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("HIPFIRE_NORMALIZE_PROMPT").ok();
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "no");

        let cfg = RuntimeConfig::from_env();
        assert!(!cfg.normalize_prompt);

        match prev {
            Some(value) => std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", value),
            None => std::env::remove_var("HIPFIRE_NORMALIZE_PROMPT"),
        }
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
    fn resolve_parallelism_defaults_tp_only_when_unset() {
        // Neither requested + emulate -> default TP.
        assert_eq!(super::resolve_parallelism(1, 1, Some(2)), (1, 2));
        // Explicit pp wins; no tp default.
        assert_eq!(super::resolve_parallelism(2, 1, Some(2)), (2, 1));
        // Explicit tp wins.
        assert_eq!(super::resolve_parallelism(1, 4, Some(2)), (1, 4));
        // No emulate -> unchanged.
        assert_eq!(super::resolve_parallelism(1, 1, None), (1, 1));
    }
}
