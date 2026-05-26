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
}

impl RuntimeConfig {
    /// Build a [`RuntimeConfig`] from the current process environment.
    pub fn from_env() -> Self {
        let v = |name: &str| -> Option<String> { std::env::var(name).ok() };

        Self {
            normalize_prompt: match std::env::var("HIPFIRE_NORMALIZE_PROMPT") {
                Ok(s) => {
                    let s = s.to_ascii_lowercase();
                    !(s == "0" || s == "false" || s == "off" || s == "no")
                }
                Err(_) => true,
            },

            prompt_token_heat: v("HIPFIRE_PROMPT_TOKEN_HEAT")
                .map_or(false, |v| v == "1"),
            prompt_heat_json: v("HIPFIRE_PROMPT_HEAT_JSON")
                .map_or(false, |v| v == "1"),
            prompt_heat_limit: v("HIPFIRE_PROMPT_HEAT_LIMIT")
                .and_then(|s| s.parse().ok())
                .unwrap_or(64),

            draft_f16: v("HIPFIRE_DRAFT_F16")
                .map_or(true, |v| v != "0"),
            draft_gemm_dump: v("HIPFIRE_DRAFT_GEMM_DUMP").as_deref() == Some("1"),
            draft_subphase: v("HIPFIRE_DRAFT_SUBPHASE").as_deref() == Some("1"),

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
        }
    }

    /// Convenience accessor — reads from the current process environment.
    ///
    /// Returns a fresh snapshot on every call. Cache if called in a hot path.
    pub fn get() -> Self {
        Self::from_env()
    }
}
