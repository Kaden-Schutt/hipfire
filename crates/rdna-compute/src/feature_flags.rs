// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable env-var resolution for rdna-compute.
//!
//! All `HIPFIRE_*` env vars are read exactly once at `Gpu::init()` time via
//! `FeatureFlags::from_env()`. Dispatching hot paths access `self.flags.*`
//! instead of hitting `std::env::var`'s global lock on every call.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mb4Mode {
    Pack1,
    Pack2,
    Pack4,
}

#[derive(Debug, Clone)]
pub struct FeatureFlags {
    // ── Arch identity ──────────────────────────────────────────────
    pub arch: String,

    // ── GEMV tuning ────────────────────────────────────────────────
    pub gemv_rows: Option<u32>,
    pub gemv_dp4a_default_on: bool,
    pub gemv_prefetch: Option<bool>,
    pub gemv_prefetch_default_on: bool,
    pub gfx942_lds_gemv: Option<bool>,
    pub gfx942_lds_gemv_default_on: bool,
    pub gemv_rows_default: u32,

    // ── Quant / format toggles ────────────────────────────────────
    pub hfq3_dp4a: Option<bool>,
    pub hfq3_mmq: Option<bool>,
    pub hfq4_mmq_rdna2: Option<bool>,
    pub fp8_wmma: bool,
    pub dot2_gemv: bool,
    pub gcn5_wave64_hybrid: Option<bool>,
    pub mmq_override: Option<bool>,
    pub mmq_min_batch: Option<usize>,
    pub fp16_disabled: bool,
    pub fp16_layer_min: Option<usize>,
    pub fp16_layer_max: Option<usize>,
    pub wo_mmq: bool,
    pub lm_head_wmma_disabled: bool,

    // ── MMQ screening ─────────────────────────────────────────────
    pub mmq_screen: bool,
    pub mmq_screen_threshold: f32,
    pub mmq_diag_quantize_only: bool,

    // ── Kernel variant overrides ─────────────────────────────────
    pub lloyd_mb4: Option<Mb4Mode>,
    pub mq3_mb4: Option<Mb4Mode>,
    pub hfq4g128_mmq: bool,
    pub hfq3_mmq_layer_min: Option<usize>,
    pub hfq3_mmq_layer_max: Option<usize>,
    pub hfq4_mmq_gfx906_y64: bool,
    pub gate_up_variant: Option<String>,
    pub gfx942_gemv_v2: Option<bool>,
    pub gfx942_gemv_v3: bool,
    pub gfx942_rmsnorm_split: bool,
    pub gfx942_mfma_prefill: Option<String>,
    pub moe_grouped_i8: Option<bool>,
    pub moe_grouped_i8_k8: bool,
    pub moe_grouped_i8_k4: bool,
    pub moe_grouped_i8_k4_gfx12: bool,
    pub moe_grouped_m2: bool,
    pub moe_hfq6_v2: bool,

    // ── Graph / capture / deterministic ─────────────────────────────
    pub force_blob_path: bool,
    pub gemm_dump: bool,
    pub deterministic: bool,
    pub mw16: bool,
    pub q8_batched_legacy: bool,
    pub rope_interleaved_legacy: bool,
    pub wo_wmma_variant: Option<String>,

    // ── rocBLAS ────────────────────────────────────────────────────
    pub rocblas_all_archs: bool,
    pub rocblas_off: bool,
    pub rocblas_min_batch: Option<usize>,

    // ── Kernels.rs env reads ───────────────────────────────────────
    pub lloyd_force_baseline: bool,
    pub rdna2_variant: Option<u32>,

    // ── Compiler.rs env reads ──────────────────────────────────────
    pub hipcc_extra_flags: String,
}

impl FeatureFlags {
    pub fn from_env(arch: &str) -> Self {
        let parse_bool = |name: &str| -> Option<bool> {
            match std::env::var(name).ok().as_deref() {
                Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => Some(true),
                Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => Some(false),
                _ => None,
            }
        };

        let parse_usize = |name: &str| -> Option<usize> {
            std::env::var(name).ok().and_then(|s| s.parse().ok())
        };

        let parse_mb4 = |name: &str| -> Option<Mb4Mode> {
            match std::env::var(name).ok().as_deref() {
                Some("1") => Some(Mb4Mode::Pack1),
                Some("2") => Some(Mb4Mode::Pack2),
                Some("4") => Some(Mb4Mode::Pack4),
                _ => None,
            }
        };

        let is_gfx906 = arch == "gfx906";

        let mmq_screen_default: bool = false;
        let mmq_screen_threshold_default: f32 = if is_gfx906 { 0.50 } else { 0.10 };

        let gemv_rows_default: u32 = match arch {
            "gfx1100" | "gfx1101" | "gfx1102" => 1,
            "gfx1030" | "gfx1031" => 1,
            "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
            _ => 2,
        };

        Self {
            arch: arch.to_string(),

            // GEMV tuning
            gemv_rows: std::env::var("HIPFIRE_GEMV_ROWS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|r| match r { 1 | 2 | 4 | 8 => r, _ => 1 }),
            gemv_dp4a_default_on: is_gfx906,
            gemv_prefetch: parse_bool("HIPFIRE_GEMV_PREFETCH"),
            gemv_prefetch_default_on: is_gfx906,
            gfx942_lds_gemv: parse_bool("HIPFIRE_GFX942_LDS_GEMV"),
            gfx942_lds_gemv_default_on: false,
            gemv_rows_default,

            // Quant/format toggles
            hfq3_dp4a: parse_bool("HIPFIRE_HFQ3_DP4A"),
            hfq3_mmq: parse_bool("HIPFIRE_HFQ3_MMQ"),
            hfq4_mmq_rdna2: parse_bool("HIPFIRE_HFQ4_MMQ_RDNA2"),
            fp8_wmma: std::env::var("HIPFIRE_FP8_WMMA").map_or(false, |v| v == "1"),
            dot2_gemv: std::env::var("HIPFIRE_DOT2_GEMV").map_or(false, |v| v == "1"),
            gcn5_wave64_hybrid: parse_bool("HIPFIRE_GCN5_WAVE64_HYBRID"),
            mmq_override: match std::env::var("HIPFIRE_MMQ").ok().as_deref() {
                Some("0") | Some("off") => Some(false),
                Some("1") | Some("on") => Some(true),
                _ => None,
            },
            mmq_min_batch: parse_usize("HIPFIRE_MMQ_MIN_BATCH"),
            fp16_disabled: std::env::var("HIPFIRE_FP16").map_or(false, |v| v == "0"),
            fp16_layer_min: parse_usize("HIPFIRE_FP16_LAYER_MIN"),
            fp16_layer_max: parse_usize("HIPFIRE_FP16_LAYER_MAX"),
            wo_mmq: std::env::var("HIPFIRE_WO_MMQ").ok().as_deref() == Some("1"),
            lm_head_wmma_disabled: std::env::var("HIPFIRE_LM_HEAD_WMMA").map_or(false, |v| v == "0"),

            // MMQ screening
            mmq_screen: std::env::var("HIPFIRE_MMQ_SCREEN")
                .ok()
                .map(|v| v == "1")
                .unwrap_or(mmq_screen_default),
            mmq_screen_threshold: std::env::var("HIPFIRE_MMQ_SCREEN_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(mmq_screen_threshold_default),
            mmq_diag_quantize_only: std::env::var("HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY")
                .ok()
                .as_deref() == Some("1"),

            // Kernel variant overrides
            lloyd_mb4: parse_mb4("HIPFIRE_LLOYD_MB4"),
            mq3_mb4: parse_mb4("HIPFIRE_MQ3_MB4"),
            hfq4g128_mmq: std::env::var("HIPFIRE_HFQ4G128_MMQ").as_deref() != Ok("0"),
            hfq3_mmq_layer_min: parse_usize("HIPFIRE_HFQ3_MMQ_LAYER_MIN"),
            hfq3_mmq_layer_max: parse_usize("HIPFIRE_HFQ3_MMQ_LAYER_MAX"),
            hfq4_mmq_gfx906_y64: std::env::var("HIPFIRE_HFQ4_MMQ_GFX906_Y64")
                .map_or(false, |v| v == "1"),
            gate_up_variant: std::env::var("HIPFIRE_GATE_UP_VARIANT").ok(),
            gfx942_gemv_v2: parse_bool("HIPFIRE_GFX942_GEMV_V2"),
            gfx942_gemv_v3: std::env::var("HIPFIRE_GFX942_GEMV_V3").map_or(false, |v| v == "1"),
            gfx942_rmsnorm_split: matches!(arch, "gfx940" | "gfx941" | "gfx942")
                && std::env::var("HIPFIRE_GFX942_RMSNORM_SPLIT").as_deref() != Ok("0"),
            gfx942_mfma_prefill: std::env::var("HIPFIRE_GFX942_MFMA_PREFILL").ok(),
            moe_grouped_i8: match std::env::var("HIPFIRE_MOE_GROUPED_I8").ok().as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            },
            moe_grouped_i8_k8: std::env::var("HIPFIRE_MOE_GROUPED_I8_K8").as_deref() == Ok("1"),
            moe_grouped_i8_k4: std::env::var("HIPFIRE_MOE_GROUPED_I8_K4").as_deref() == Ok("1"),
            moe_grouped_i8_k4_gfx12: std::env::var("HIPFIRE_MOE_GROUPED_I8_K4_GFX12")
                .as_deref() == Ok("1"),
            moe_grouped_m2: std::env::var("HIPFIRE_MOE_GROUPED_M2").as_deref() == Ok("1"),
            moe_hfq6_v2: std::env::var("HIPFIRE_MOE_HFQ6_V2").as_deref() == Ok("1"),

            // Graph / capture / deterministic
            force_blob_path: std::env::var("HIPFIRE_BLOB_FORCE").ok().as_deref() == Some("1"),
            gemm_dump: std::env::var("HIPFIRE_GEMM_DUMP").ok().as_deref() == Some("1"),
            deterministic: std::env::var("HIPFIRE_DETERMINISTIC").ok().as_deref() == Some("1"),
            mw16: std::env::var("HIPFIRE_MW16").map_or(false, |v| v == "1"),
            q8_batched_legacy: std::env::var("HIPFIRE_Q8_BATCHED_LEGACY").as_deref() == Ok("1"),
            rope_interleaved_legacy: std::env::var("HIPFIRE_ROPE_INTERLEAVED_LEGACY")
                .ok()
                .as_deref() == Some("1"),
            wo_wmma_variant: std::env::var("HIPFIRE_WO_WMMA_VARIANT").ok(),

            // rocBLAS
            rocblas_all_archs: std::env::var("HIPFIRE_ROCBLAS_ALL_ARCHS")
                .ok()
                .as_deref() == Some("1"),
            rocblas_off: std::env::var("HIPFIRE_ROCBLAS_OFF").ok().as_deref() == Some("1"),
            rocblas_min_batch: parse_usize("HIPFIRE_ROCBLAS_MIN_BATCH"),

            // Kernels.rs
            lloyd_force_baseline: std::env::var("HIPFIRE_LLOYD_FORCE_BASELINE")
                .ok()
                .as_deref() == Some("1"),
            rdna2_variant: std::env::var("HIPFIRE_RDNA2_VARIANT")
                .ok()
                .and_then(|s| s.parse::<u32>().ok()),

            // Compiler.rs
            hipcc_extra_flags: std::env::var("HIPFIRE_HIPCC_EXTRA_FLAGS").unwrap_or_default(),
        }
    }

    // ── Methods replacing free functions ─────────────────────────────

    pub fn gemv_dp4a_enabled(&self) -> bool {
        self.hfq3_dp4a.unwrap_or(self.gemv_dp4a_default_on)
    }

    pub fn gemv_prefetch_enabled(&self) -> bool {
        self.gemv_prefetch.unwrap_or(self.gemv_prefetch_default_on)
    }

    pub fn gfx942_lds_gemv_enabled(&self) -> bool {
        self.gfx942_lds_gemv.unwrap_or(self.gfx942_lds_gemv_default_on)
    }

    pub fn has_wmma_f16(&self) -> bool {
        self.arch.starts_with("gfx11")
    }

    pub fn has_wmma_f16_gfx12(&self) -> bool {
        self.arch.starts_with("gfx12")
    }

    pub fn has_wmma_fp8_gfx12(&self) -> bool {
        self.arch.starts_with("gfx12")
    }

    pub fn has_dot2_f32_f16(&self) -> bool {
        matches!(self.arch.as_str(),
            "gfx1011" | "gfx1012"
            | "gfx1030" | "gfx1031" | "gfx1032"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152"
            | "gfx1200" | "gfx1201")
    }

    pub fn hfq3_sdot4_gfx10_enabled(&self) -> bool {
        matches!(self.arch.as_str(),
            "gfx1011" | "gfx1012" | "gfx1030" | "gfx1031" | "gfx1032")
    }

    pub fn hfq3_dp4a_enabled(&self) -> bool {
        self.hfq3_dp4a.unwrap_or(false) && self.hfq3_sdot4_gfx10_enabled()
    }

    pub fn hfq3_mmq_rdna2_enabled(&self) -> bool {
        self.hfq3_mmq.unwrap_or(false) && self.hfq3_sdot4_gfx10_enabled()
    }

    pub fn hfq4_mmq_rdna2_enabled(&self) -> bool {
        self.hfq4_mmq_rdna2.unwrap_or(false)
            && self.has_dot2_f32_f16()
    }

    pub fn has_mmq_dp4a_or_wmma(&self) -> bool {
        matches!(self.arch.as_str(),
            "gfx906"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152")
    }

    pub fn has_wave64_native(&self) -> bool {
        matches!(self.arch.as_str(),
            "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942")
    }

    pub fn is_gcn5_wave64(&self) -> bool {
        if self.arch == "gfx906" {
            return true;
        }
        self.arch == "gfx908" && self.gcn5_wave64_hybrid.unwrap_or(false)
    }

    pub fn should_use_mmq(&self, batch_size: usize) -> bool {
        if !self.has_mmq_dp4a_or_wmma() {
            return false;
        }
        match self.mmq_override {
            Some(false) => false,
            Some(true) => true,
            None => {
                let arch_min_batch: usize = if self.arch == "gfx906" { 8 } else { 256 };
                let min_batch = self.mmq_min_batch.unwrap_or(arch_min_batch);
                batch_size >= min_batch
            }
        }
    }

    pub fn hfq3_mmq_layer_gate_pass(&self) -> bool {
        let lo = self.hfq3_mmq_layer_min;
        let hi = self.hfq3_mmq_layer_max;
        if lo.is_none() && hi.is_none() {
            return true;
        }
        let layer = super::dispatch::MMQ_CURRENT_LAYER.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(lo) = lo { if layer < lo { return false; } }
        if let Some(hi) = hi { if layer > hi { return false; } }
        true
    }

    pub fn fp16_disabled_for_current_layer(&self) -> bool {
        if self.fp16_disabled { return true; }
        let lo = self.fp16_layer_min;
        let hi = self.fp16_layer_max;
        if lo.is_none() && hi.is_none() {
            return false;
        }
        let layer = super::dispatch::MMQ_CURRENT_LAYER.load(std::sync::atomic::Ordering::Relaxed);
        let above_min = lo.map(|m| layer >= m).unwrap_or(true);
        let below_max = hi.map(|m| layer <= m).unwrap_or(true);
        above_min && below_max
    }

    pub fn hfq4_mmq_gfx906_y64_enabled(&self) -> bool {
        self.hfq4_mmq_gfx906_y64
    }
}
