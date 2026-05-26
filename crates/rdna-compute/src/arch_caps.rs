// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-architecture capability defaults.
//!
//! Single source of truth for "is feature X default-on for arch Y" questions.
//! Adding a new arch = one-line edit per feature.

use crate::feature_flags::FeatureFlags;

/// Single source of truth for GPU architecture family membership and
/// hardware capability predicates. Computed once at `Gpu::init()` time;
/// immutable thereafter.
///
/// Named predicates replace the 13+ repeated arch-set literals that were
/// scattered across `dispatch.rs` and `kernels.rs`. Adding a new arch
/// (e.g. gfx1300) requires touching this struct's `new()` constructor and
/// exactly zero match arms elsewhere.
pub struct ArchCaps {
    arch: String,

    // ── Family membership ────────────────────────────────────────
    is_rdna3_dgpu: bool,
    is_rdna3_dgpu_1151: bool,
    is_rdna3_full: bool,
    is_strix_halo: bool,
    is_strix_halo_igpu: bool,
    is_rdna4: bool,
    is_cdna3: bool,
    is_rdna2: bool,
    is_gfx906: bool,
    is_gfx908: bool,
    is_wave64_native: bool,

    // ── Capability predicates (derived from family + FeatureFlags) ──
    has_wmma_f16: bool,
    has_wmma_f16_gfx12: bool,
    has_wmma_fp8_gfx12: bool,
    has_dot2_f32_f16: bool,
    has_mmq_dp4a_or_wmma: bool,
    is_gcn5_wave64: bool,
    should_use_mmq_cache: bool,
    gemv_dp4a: bool,
    gemv_prefetch: bool,
    gemv_rows_default: u32,
    hfq3_sdot4_gfx10: bool,
    hfq3_dp4a: bool,
    hfq3_mmq_rdna2: bool,
    hfq4_mmq_rdna2: bool,
    is_rdna_wave32: bool,
    gfx942_lds_gemv: bool,

    // ── Reference to FeatureFlags for env-var overrides ────────
    flags: std::sync::Arc<FeatureFlags>,
}

impl ArchCaps {
    pub fn new(arch: &str, flags: std::sync::Arc<FeatureFlags>) -> Self {
        let is_rdna3_dgpu = matches!(arch, "gfx1100" | "gfx1101" | "gfx1102");
        let is_rdna3_dgpu_1151 = is_rdna3_dgpu || arch == "gfx1151";
        let is_rdna3_full = is_rdna3_dgpu_1151 || arch == "gfx1150";
        let is_strix_halo = arch == "gfx1151";
        let is_strix_halo_igpu = matches!(arch, "gfx1150" | "gfx1151" | "gfx1152");
        let is_rdna4 = matches!(arch, "gfx1200" | "gfx1201");
        let is_cdna3 = matches!(arch, "gfx940" | "gfx941" | "gfx942");
        let is_rdna2 = matches!(arch, "gfx1030" | "gfx1031");
        let is_gfx906 = arch == "gfx906";
        let is_gfx908 = arch == "gfx908";
        let is_wave64_native = matches!(arch, "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942");

        let has_wmma_f16 = arch.starts_with("gfx11");
        let has_wmma_f16_gfx12 = arch.starts_with("gfx12");
        let has_wmma_fp8_gfx12 = arch.starts_with("gfx12");
        let has_dot2_f32_f16 = matches!(arch,
            "gfx1011" | "gfx1012"
            | "gfx1030" | "gfx1031" | "gfx1032"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152"
            | "gfx1200" | "gfx1201"
        );
        let has_mmq_dp4a_or_wmma = matches!(arch,
            "gfx906"
            | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103"
            | "gfx1150" | "gfx1151" | "gfx1152"
        );

        // Derived from family + FeatureFlags env overrides
        let is_gcn5_wave64 = is_gfx906
            || (is_gfx908 && flags.gcn5_wave64_hybrid.unwrap_or(false));
        let dp4a_default = is_gfx906;
        let gemv_dp4a = flags.gemv_dp4a.unwrap_or(dp4a_default);
        let gemv_prefetch = flags.gemv_prefetch.unwrap_or(is_gfx906);
        let gemv_rows_default = flags.gemv_rows.unwrap_or_else(|| {
            match arch {
                "gfx1100" | "gfx1101" | "gfx1102"
                | "gfx1030" | "gfx1031"
                | "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
                _ => 2,
            }
        });
        let hfq3_sdot4_gfx10 = matches!(arch,
            "gfx1011" | "gfx1012" | "gfx1030" | "gfx1031" | "gfx1032"
        );
        let hfq3_dp4a = flags.hfq3_dp4a.unwrap_or(false) && hfq3_sdot4_gfx10;
        let hfq3_mmq_rdna2 = flags.hfq3_mmq.unwrap_or(false) && hfq3_sdot4_gfx10;
        let hfq4_mmq_rdna2 = flags.hfq4_mmq_rdna2.unwrap_or(false) && has_dot2_f32_f16;
        let is_rdna_wave32 = arch.starts_with("gfx10")
            || arch.starts_with("gfx11")
            || arch.starts_with("gfx12");
        let gfx942_lds_gemv = flags.gfx942_lds_gemv.unwrap_or(false);

        Self {
            arch: arch.to_string(),
            is_rdna3_dgpu,
            is_rdna3_dgpu_1151,
            is_rdna3_full,
            is_strix_halo,
            is_strix_halo_igpu,
            is_rdna4,
            is_cdna3,
            is_rdna2,
            is_gfx906,
            is_gfx908,
            is_wave64_native,
            has_wmma_f16,
            has_wmma_f16_gfx12,
            has_wmma_fp8_gfx12,
            has_dot2_f32_f16,
            has_mmq_dp4a_or_wmma,
            is_gcn5_wave64,
            should_use_mmq_cache: false,
            gemv_dp4a,
            gemv_prefetch,
            gemv_rows_default,
            hfq3_sdot4_gfx10,
            hfq3_dp4a,
            hfq3_mmq_rdna2,
            hfq4_mmq_rdna2,
            is_rdna_wave32,
            gfx942_lds_gemv,
            flags,
        }
    }

    pub fn should_use_mmq(&self, batch_size: usize) -> bool {
        if !self.has_mmq_dp4a_or_wmma { return false; }
        match self.flags.mmq_override {
            Some(false) => false,
            Some(true) => true,
            None => {
                let arch_min_batch: usize = if self.is_gfx906 { 8 } else { 256 };
                let min_batch = self.flags.mmq_min_batch.unwrap_or(arch_min_batch);
                batch_size >= min_batch
            }
        }
    }

    // ── Family membership ────────────────────────────────────────
    pub fn is_rdna3_dgpu(&self) -> bool { self.is_rdna3_dgpu }
    pub fn is_rdna3_dgpu_1151(&self) -> bool { self.is_rdna3_dgpu_1151 }
    pub fn is_rdna3_full(&self) -> bool { self.is_rdna3_full }
    pub fn is_strix_halo(&self) -> bool { self.is_strix_halo }
    pub fn is_strix_halo_igpu(&self) -> bool { self.is_strix_halo_igpu }
    pub fn is_rdna4(&self) -> bool { self.is_rdna4 }
    pub fn is_cdna3(&self) -> bool { self.is_cdna3 }
    pub fn is_rdna2(&self) -> bool { self.is_rdna2 }
    pub fn is_gfx906(&self) -> bool { self.is_gfx906 }
    pub fn is_gfx908(&self) -> bool { self.is_gfx908 }
    pub fn is_wave64_native(&self) -> bool { self.is_wave64_native }

    // ── Capability predicates ────────────────────────────────────
    pub fn has_wmma_f16(&self) -> bool { self.has_wmma_f16 }
    pub fn has_wmma_f16_gfx12(&self) -> bool { self.has_wmma_f16_gfx12 }
    pub fn has_wmma_fp8_gfx12(&self) -> bool { self.has_wmma_fp8_gfx12 }
    pub fn has_dot2_f32_f16(&self) -> bool { self.has_dot2_f32_f16 }
    pub fn has_mmq_dp4a_or_wmma(&self) -> bool { self.has_mmq_dp4a_or_wmma }
    pub fn is_gcn5_wave64(&self) -> bool { self.is_gcn5_wave64 }
    pub fn gemv_dp4a_enabled(&self) -> bool { self.gemv_dp4a }
    pub fn gemv_prefetch_enabled(&self) -> bool { self.gemv_prefetch }
    pub fn gemv_rows_default(&self) -> u32 { self.gemv_rows_default }
    pub fn hfq3_sdot4_gfx10_enabled(&self) -> bool { self.hfq3_sdot4_gfx10 }
    pub fn hfq3_dp4a_enabled(&self) -> bool { self.hfq3_dp4a }
    pub fn hfq3_mmq_rdna2_enabled(&self) -> bool { self.hfq3_mmq_rdna2 }
    pub fn hfq4_mmq_rdna2_enabled(&self) -> bool { self.hfq4_mmq_rdna2 }
    pub fn is_rdna_wave32(&self) -> bool { self.is_rdna_wave32 }
    pub fn gfx942_lds_gemv_enabled(&self) -> bool { self.gfx942_lds_gemv }
    pub fn arch(&self) -> &str { &self.arch }
}

/// MQ4G128 in-engine encoding of LinearAttention in_proj_a / in_proj_b weights
/// at model load time.
///
/// **Default-off pending fused rotation-GEMV kernel.** Bench on 2026-05-22
/// (`.scratch/rocprof-2026-05-22-mq4g128-on/`) measured −0.5% decode tok/s vs
/// baseline on shisa-Qwen3.6-35B-A3B-PARO / gfx1151. Root cause: alpha/beta
/// have M=16 shape; the existing dispatch chain (`mq_rotate_x_128` +
/// `gemv_mq4g128_prerotated` → `gemv_hfq4g128`) doubles launch count vs the
/// single F32 GEMV, and at small M the GPU kernel time (~4 µs) is dwarfed
/// by launch overhead. K-split variant was tried and also regressed.
///
/// The lever can only be made net-positive by FUSING the FWHT-128 rotation
/// into the GEMV in a single kernel launch. See
/// `docs/superpowers/specs/2026-05-22-lever1-fused-mq4g128-design.md` for
/// the design.
///
/// The infrastructure (DType variant, kernel, dispatch wrappers, codec) is
/// kept in place since it's correct end-to-end (round-trip + smoke + KLD
/// equivalent argmax verified). Opt-in via `HIPFIRE_PARO_LA_GATES_MQ4G128=1`.
// NOTE: `const fn` with `&str` pattern matching is not yet stable (PartialEq not const).
// Using plain `pub fn` instead — semantically identical at runtime.
pub fn paro_la_gates_mq4g128_default(_arch: &str) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn default_flags() -> Arc<FeatureFlags> {
        Arc::new(FeatureFlags::from_env_for_test("gfx1100"))
    }

    fn make_caps(arch: &str) -> ArchCaps {
        ArchCaps::new(arch, default_flags())
    }

    #[test]
    fn rdna3_dgpu() {
        let caps = make_caps("gfx1100");
        assert!(caps.is_rdna3_dgpu());
        assert!(caps.is_rdna3_dgpu_1151());
        assert!(caps.is_rdna3_full());
        assert!(!caps.is_strix_halo());
        assert!(!caps.is_rdna4());
        assert!(!caps.is_cdna3());
    }

    #[test]
    fn strix_halo_1151() {
        let caps = make_caps("gfx1151");
        assert!(!caps.is_rdna3_dgpu());
        assert!(caps.is_rdna3_dgpu_1151());
        assert!(caps.is_rdna3_full());
        assert!(caps.is_strix_halo());
        assert!(caps.is_strix_halo_igpu());
    }

    #[test]
    fn strix_point_1150() {
        let caps = make_caps("gfx1150");
        assert!(!caps.is_rdna3_dgpu());
        assert!(!caps.is_rdna3_dgpu_1151());
        assert!(caps.is_rdna3_full());
        assert!(!caps.is_strix_halo());
        assert!(caps.is_strix_halo_igpu());
    }

    #[test]
    fn rdna4() {
        let caps = make_caps("gfx1200");
        assert!(caps.is_rdna4());
        assert!(caps.has_wmma_f16_gfx12());
        assert!(caps.has_wmma_fp8_gfx12());
        assert!(!caps.is_rdna3_dgpu());
    }

    #[test]
    fn cdna3_942() {
        let caps = make_caps("gfx942");
        assert!(caps.is_cdna3());
        assert!(caps.is_wave64_native());
        assert!(!caps.is_rdna3_dgpu());
    }

    #[test]
    fn gfx906_wave64() {
        let caps = make_caps("gfx906");
        assert!(caps.is_gfx906());
        assert!(caps.is_gcn5_wave64());
        assert!(caps.is_wave64_native());
        assert!(caps.has_mmq_dp4a_or_wmma());
    }

    #[test]
    fn rdna2() {
        let caps = make_caps("gfx1030");
        assert!(caps.is_rdna2());
        assert!(caps.has_dot2_f32_f16());
    }

    #[test]
    fn gfx1152_strix_point_igpu() {
        let caps = make_caps("gfx1152");
        assert!(caps.is_strix_halo_igpu());
        assert!(!caps.is_strix_halo());
        assert!(!caps.is_rdna3_dgpu_1151());
        assert!(caps.has_dot2_f32_f16());
    }

    #[test]
    fn dot2_coverage() {
        for arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201"] {
            assert!(make_caps(arch).has_dot2_f32_f16(), "dot2 missing for {arch}");
        }
        assert!(!make_caps("gfx906").has_dot2_f32_f16());
    }

    #[test]
    fn default_table() {
        assert!(!paro_la_gates_mq4g128_default("gfx1151"));
        assert!(!paro_la_gates_mq4g128_default("gfx1100"));
        assert!(!paro_la_gates_mq4g128_default("gfx1010"));
        assert!(!paro_la_gates_mq4g128_default(""));
    }
}