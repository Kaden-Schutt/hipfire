// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Consolidated runtime configuration from environment variables.
//!
//! All `HIPFIRE_*` env reads in the hipfire-runtime crate should go
//! through [`RuntimeConfig`] instead of calling `std::env::var` directly.
//! This gives us a single place to document, validate, and default
//! every env knob.

/// Runtime configuration snapshot, populated from `HIPFIRE_*` env vars.
///
/// Create via [`RuntimeConfig::from_env()`] or the convenience accessor
/// [`RuntimeConfig::get()`]. All fields default to safe values when the
/// corresponding env var is unset or unparseable.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    // ── Tokenizer ──────────────────────────────────────────────────────
    /// Collapse `\n{3,}` runs before tokenize (default: on since 2026-04-26).
    /// `HIPFIRE_NORMALIZE_PROMPT=0` to opt out.
    pub normalize_prompt: bool,
    /// `HIPFIRE_PROMPT_TOKEN_HEAT=1` — dump per-position merge-rank table.
    pub prompt_token_heat: bool,
    /// `HIPFIRE_PROMPT_HEAT_JSON=1` — emit heat dump as JSON on stdout.
    pub prompt_heat_json: bool,
    /// `HIPFIRE_PROMPT_HEAT_LIMIT=N` — max rows in heat dump (default 64).
    pub prompt_heat_limit: usize,

    // ── DFlash (speculative decode) ────────────────────────────────────
    /// `HIPFIRE_DRAFT_F16=0` to fall back to legacy F16→F32 lift.
    pub draft_f16: bool,
    /// `HIPFIRE_DRAFT_GEMM_DUMP=1` — per-call GEMM timing dump.
    pub draft_gemm_dump: bool,
    /// `HIPFIRE_DRAFT_SUBPHASE=1` — per-layer timing inside `draft_forward`.
    pub draft_subphase: bool,

    // ── LLM / Paro ─────────────────────────────────────────────────────
    /// `HIPFIRE_PARO_SMALL_DIRECT` — direct GEMV threshold for small
    /// PARO4G128T weights. `"1"` or empty defaults to 64; parsed usize for
    /// custom values; absent = disabled.
    pub paro_small_direct: Option<String>,
    /// `HIPFIRE_PARO_PREROTATE` — set to any non-empty value to enable
    /// FWHT-prerotated GEMV for PARO4G128 weights.
    pub paro_prerotate: bool,
    /// `HIPFIRE_PARO_FUSE_RMSNORM` — opt-in for fused rmsnorm + rotation.
    /// `"1"` or `"true"` to enable; absent or other = off.
    pub paro_fuse_rmsnorm: bool,
    /// `HIPFIRE_PARO_SWIGLU_FUSED` — set to any non-empty value to use
    /// fused SiLU-mul + residual GEMV for PARO4G128.
    pub paro_swiglu_fused: bool,
    /// `HIPFIRE_FLASH_PARTIALS_BATCH=N` — flash-attention partials batch
    /// multiplier (clamped to `[1, PREFILL_MAX_BATCH]`; default 16).
    pub flash_partials_batch: Option<usize>,
    /// `HIPFIRE_PREFILL_BATCHED=0` — force fallback to single-token prefill.
    pub prefill_batched: bool,

    // ── Loop guard ─────────────────────────────────────────────────────
    /// `HIPFIRE_NGRAM_LOOP_THRESHOLD` — 4-gram count that triggers guard
    /// (default 8; 0 = disabled).
    pub ngram_loop_threshold: usize,
    /// `HIPFIRE_NGRAM_WINDOW` — trailing-token window size (default 256).
    pub ngram_window: usize,

    // ── Multi-GPU ──────────────────────────────────────────────────────
    /// `HIPFIRE_DEVICES=0,1` — comma-separated HIP device IDs.
    pub devices: Option<String>,
    /// `HIPFIRE_ALLOW_MIXED_ARCH=1` — tolerate mixed GPU architectures.
    pub allow_mixed_arch: bool,
    /// `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=N` — VRAM delta tolerance (GiB).
    pub uniform_vram_tolerance_gb: Option<f64>,

    // ── Daemon / DFlash / Bench ────────────────────────────────────────
    /// `HIPFIRE_EMIT_TOKEN_IDS=1` — emit `committed` events for every token.
    pub emit_token_ids: bool,
    /// `HIPFIRE_PP_DFLASH=1` — opt into experimental pp>1 DFlash path.
    pub pp_dflash: bool,
    /// `HIPFIRE_PP_PFLASH=1` — opt into experimental pp>1 PFlash path.
    pub pp_pflash: bool,
    /// `HIPFIRE_DPM_WARMUP_SECS=N` — pin GPU DPM for N seconds before bench.
    pub dpm_warmup_secs: Option<f64>,
    /// `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT=1` — enable budget-alert nudge.
    pub experimental_budget_alert: bool,
    /// `HIPFIRE_CHAT_TEMPLATE_FILE=<path>` — override chat template file.
    pub chat_template_file: Option<String>,
    /// `HIPFIRE_KV_MODE=<mode>` — KV cache quantization mode override.
    pub kv_mode: Option<String>,
    /// `HIPFIRE_KV_PHYSICAL_CAP=N` — explicit physical KV cache capacity.
    pub kv_physical_cap: Option<usize>,
    /// `HIPFIRE_PP_LAYERS=a,b,c` — per-GPU layer counts for pipeline parallel.
    pub pp_layers: Option<String>,
    /// `HIPFIRE_DDTREE_BUDGET=N` — DDTree node budget (0 = disabled).
    pub ddtree_budget: Option<usize>,
    /// `HIPFIRE_DDTREE_TOPK=N` — DDTree per-position top-K.
    pub ddtree_topk: Option<usize>,
    /// `HIPFIRE_JINJA_CHAT=1` — enable Jinja chat template rendering.
    pub jinja_chat: bool,
    /// `HIPFIRE_DDTREE_PATH_C=<phase>` — DDTree Path-C mode (phase1/phase2).
    pub ddtree_path_c: Option<String>,
    /// `HIPFIRE_ADAPTIVE_B_UNSAFE=1` — allow adaptive-B past draft block_size.
    pub adaptive_b_unsafe: bool,
    /// `HIPFIRE_DFLASH_LOOP_BREAK=<mode>` — DFlash loop break mode.
    pub dflash_loop_break: Option<String>,
    /// `HIPFIRE_DFLASH_LOOP_BREAK_TEMP=N` — loop break temperature bump.
    pub dflash_loop_break_temp: f32,
    /// `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER=N` — consecutive hits to stop.
    pub dflash_loop_break_stop_after: usize,
    /// `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP=N` — repeat penalty step per hit.
    pub dflash_loop_break_rp_step: f32,
    /// `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX=N` — max repeat penalty.
    pub dflash_loop_break_rp_max: f32,
    /// `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY=N` — clean cycles to decay RP.
    pub dflash_loop_break_recovery: usize,
    /// `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS=N` — max RP escalations.
    pub dflash_loop_break_max_escalations: usize,
    /// `HIPFIRE_PROFILE=1` — enable per-kernel profiling.
    pub profile: bool,
    /// `HIPFIRE_PROFILE_CYCLES=N` — profiling cycle count.
    pub profile_cycles: usize,
    /// `HIPFIRE_HOST_TIMING=1` — dump per-cycle host timing breakdown.
    pub host_timing: bool,
    /// `HIPFIRE_ADAPTIVE_B_UP=N` — adaptive-B up multiplier.
    pub adaptive_b_up: Option<f64>,
    /// `HIPFIRE_ADAPTIVE_B_DOWN=N` — adaptive-B down multiplier.
    pub adaptive_b_down: Option<f64>,
    /// `HIPFIRE_DDTREE_LOGW_CUTOFF=<val>` — DDTree log-weight cutoff.
    pub ddtree_logw_cutoff: Option<String>,

    // ── Smoke-test / small example env vars ────────────────────────────
    /// `HIPFIRE_SMOKE_STEPS=N` — generation steps for smoke tests.
    pub smoke_steps: usize,
    /// `HIPFIRE_SMOKE_KV_SEQ=N` — KV cache sequence length for smoke tests.
    pub smoke_kv_seq: usize,
    /// `HIPFIRE_SMOKE_KV=<mode>` — KV cache mode for smoke tests.
    pub smoke_kv: Option<String>,
    /// `HIPFIRE_SMOKE_MODE=<mode>` — prompt mode (raw/chat) for smoke tests.
    pub smoke_mode: Option<String>,
    /// `HIPFIRE_SMOKE_PROMPT=<text>` — prompt text for smoke tests.
    pub smoke_prompt: Option<String>,
    /// `HIPFIRE_CHATML=1` — enable ChatML framing.
    pub chatml: bool,
    /// `HIPFIRE_CALIB_PROFILE=1` — calibration profiling.
    pub calib_profile: bool,
    /// `HIPFIRE_GPU_TOPK=1` — GPU-assisted top-K sampler.
    pub gpu_topk: bool,
    /// `HIPFIRE_SAMPLE_COMPARE=1` — compare GPU/CPU sampling divergence.
    pub sample_compare: bool,
    /// `HIPFIRE_GEN=N` — generation token count override.
    pub gen: Option<usize>,
    /// `HIPFIRE_VL_DUMP_DIR=<path>` — vision-language tensor dump directory.
    pub vl_dump_dir: Option<String>,
    /// `HIPFIRE_BASELINE_ARCH=<arch>` — baseline architecture for coherence.
    pub baseline_arch: Option<String>,
    /// `HIPFIRE_GRAPH_PREFILL=N` — prefill graph-capture mode.
    pub graph_prefill: Option<String>,
    /// `HIPFIRE_ROCPROF_CSV=<path>` — rocprof CSV output path.
    pub rocprof_csv: Option<String>,
    /// `HIPFIRE_PROFILE_DECODE=1` — profile decode phase separately.
    pub profile_decode: bool,

    // ── rdna-compute example env vars ──────────────────────────────────
    /// `HIPFIRE_LLOYD_FORCE_BASELINE=<val>` — force Lloyd baseline kernel.
    pub lloyd_force_baseline: Option<String>,
    /// `HIPFIRE_MOE_HFQ6_V2=1` — enable MoE HFQ6 v2 path.
    pub moe_hfq6_v2: bool,
}

impl RuntimeConfig {
    /// Build a [`RuntimeConfig`] from the current process environment.
    pub fn from_env() -> Self {
        let v = |name: &str| -> Option<String> { std::env::var(name).ok() };
        let v_parse = |name: &str| -> Option<f64> {
            std::env::var(name).ok().and_then(|s| s.parse().ok())
        };
        let v_opt_on = |name: &str| -> bool {
            std::env::var(name).ok().as_deref() == Some("1")
        };

        Self {
            normalize_prompt: match std::env::var("HIPFIRE_NORMALIZE_PROMPT") {
                Ok(s) => {
                    let s = s.to_ascii_lowercase();
                    !(s == "0" || s == "false" || s == "off" || s == "no")
                }
                Err(_) => true,
            },

            prompt_token_heat: v_opt_on("HIPFIRE_PROMPT_TOKEN_HEAT"),
            prompt_heat_json: v_opt_on("HIPFIRE_PROMPT_HEAT_JSON"),
            prompt_heat_limit: v("HIPFIRE_PROMPT_HEAT_LIMIT")
                .and_then(|s| s.parse().ok())
                .unwrap_or(64),

            draft_f16: v("HIPFIRE_DRAFT_F16")
                .map_or(true, |v| v != "0"),
            draft_gemm_dump: v_opt_on("HIPFIRE_DRAFT_GEMM_DUMP"),
            draft_subphase: v_opt_on("HIPFIRE_DRAFT_SUBPHASE"),

            paro_small_direct: std::env::var_os("HIPFIRE_PARO_SMALL_DIRECT")
                .and_then(|s| s.into_string().ok()),
            paro_prerotate: std::env::var_os("HIPFIRE_PARO_PREROTATE").is_some(),
            paro_fuse_rmsnorm: v("HIPFIRE_PARO_FUSE_RMSNORM")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            paro_swiglu_fused: std::env::var_os("HIPFIRE_PARO_SWIGLU_FUSED").is_some(),
            flash_partials_batch: v("HIPFIRE_FLASH_PARTIALS_BATCH")
                .and_then(|s| s.parse().ok()),
            prefill_batched: v("HIPFIRE_PREFILL_BATCHED")
                .map_or(true, |s| s != "0"),

            ngram_loop_threshold: v("HIPFIRE_NGRAM_LOOP_THRESHOLD")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8),
            ngram_window: v("HIPFIRE_NGRAM_WINDOW")
                .and_then(|s| s.parse().ok())
                .unwrap_or(256),

            devices: v("HIPFIRE_DEVICES").filter(|s| !s.is_empty()),
            allow_mixed_arch: v("HIPFIRE_ALLOW_MIXED_ARCH")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            uniform_vram_tolerance_gb: v("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB")
                .and_then(|s| s.parse().ok()),

            // ── Daemon / DFlash / Bench ────────────────────────────────
            emit_token_ids: v_opt_on("HIPFIRE_EMIT_TOKEN_IDS"),
            pp_dflash: v_opt_on("HIPFIRE_PP_DFLASH"),
            pp_pflash: v_opt_on("HIPFIRE_PP_PFLASH"),
            dpm_warmup_secs: v_parse("HIPFIRE_DPM_WARMUP_SECS"),
            experimental_budget_alert: v_opt_on("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT"),
            chat_template_file: v("HIPFIRE_CHAT_TEMPLATE_FILE"),
            kv_mode: v("HIPFIRE_KV_MODE").filter(|s| !s.is_empty()),
            kv_physical_cap: v("HIPFIRE_KV_PHYSICAL_CAP")
                .and_then(|s| s.parse().ok()),
            pp_layers: v("HIPFIRE_PP_LAYERS").filter(|s| !s.is_empty()),
            ddtree_budget: v("HIPFIRE_DDTREE_BUDGET")
                .and_then(|s| {
                    if s.is_empty() { None }
                    else { s.parse::<usize>().ok() }
                }),
            ddtree_topk: v("HIPFIRE_DDTREE_TOPK")
                .and_then(|s| {
                    if s.is_empty() { None }
                    else { s.parse::<usize>().ok() }
                }),
            jinja_chat: v_opt_on("HIPFIRE_JINJA_CHAT"),
            ddtree_path_c: v("HIPFIRE_DDTREE_PATH_C")
                .filter(|s| !s.is_empty()),
            adaptive_b_unsafe: v_opt_on("HIPFIRE_ADAPTIVE_B_UNSAFE"),

            dflash_loop_break: v("HIPFIRE_DFLASH_LOOP_BREAK")
                .filter(|s| !s.is_empty()),
            dflash_loop_break_temp: v("HIPFIRE_DFLASH_LOOP_BREAK_TEMP")
                .and_then(|s| s.parse().ok()).unwrap_or(1.0),
            dflash_loop_break_stop_after: v("HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER")
                .and_then(|s| s.parse().ok()).unwrap_or(3),
            dflash_loop_break_rp_step: v("HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP")
                .and_then(|s| s.parse().ok()).unwrap_or(0.10),
            dflash_loop_break_rp_max: v("HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX")
                .and_then(|s| s.parse().ok()).unwrap_or(1.30),
            dflash_loop_break_recovery: v("HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY")
                .and_then(|s| s.parse().ok()).unwrap_or(32),
            dflash_loop_break_max_escalations: v("HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS")
                .and_then(|s| s.parse().ok()).unwrap_or(4),

            profile: v_opt_on("HIPFIRE_PROFILE"),
            profile_cycles: v("HIPFIRE_PROFILE_CYCLES")
                .and_then(|s| s.parse().ok()).unwrap_or(5),
            host_timing: v_opt_on("HIPFIRE_HOST_TIMING"),
            adaptive_b_up: v("HIPFIRE_ADAPTIVE_B_UP")
                .and_then(|s| s.parse::<f64>().ok()),
            adaptive_b_down: v("HIPFIRE_ADAPTIVE_B_DOWN")
                .and_then(|s| s.parse::<f64>().ok()),
            ddtree_logw_cutoff: v("HIPFIRE_DDTREE_LOGW_CUTOFF")
                .filter(|s| !s.is_empty()),

            // ── Smoke-test / small example env vars ────────────────────
            smoke_steps: v("HIPFIRE_SMOKE_STEPS")
                .and_then(|s| s.parse().ok()).unwrap_or(1),
            smoke_kv_seq: v("HIPFIRE_SMOKE_KV_SEQ")
                .and_then(|s| s.parse().ok()).unwrap_or(256),
            smoke_kv: v("HIPFIRE_SMOKE_KV").filter(|s| !s.is_empty()),
            smoke_mode: v("HIPFIRE_SMOKE_MODE").filter(|s| !s.is_empty()),
            smoke_prompt: v("HIPFIRE_SMOKE_PROMPT").filter(|s| !s.is_empty()),
            chatml: v_opt_on("HIPFIRE_CHATML"),
            calib_profile: v("HIPFIRE_CALIB_PROFILE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            gpu_topk: v_opt_on("HIPFIRE_GPU_TOPK"),
            sample_compare: v_opt_on("HIPFIRE_SAMPLE_COMPARE"),
            gen: v("HIPFIRE_GEN").and_then(|s| s.parse().ok()),
            vl_dump_dir: v("HIPFIRE_VL_DUMP_DIR").filter(|s| !s.is_empty()),
            baseline_arch: v("HIPFIRE_BASELINE_ARCH").filter(|s| !s.is_empty()),
            graph_prefill: v("HIPFIRE_GRAPH_PREFILL").filter(|s| !s.is_empty()),
            rocprof_csv: v("HIPFIRE_ROCPROF_CSV").filter(|s| !s.is_empty()),
            profile_decode: v_opt_on("HIPFIRE_PROFILE_DECODE"),

            // ── rdna-compute example env vars ──────────────────────────
            lloyd_force_baseline: v("HIPFIRE_LLOYD_FORCE_BASELINE")
                .filter(|s| !s.is_empty()),
            moe_hfq6_v2: std::env::var_os("HIPFIRE_MOE_HFQ6_V2").is_some(),
        }
    }

    /// Convenience accessor — reads from the current process environment.
    ///
    /// Returns a fresh snapshot on every call. Cache if called in a hot path.
    pub fn get() -> Self {
        Self::from_env()
    }
}
