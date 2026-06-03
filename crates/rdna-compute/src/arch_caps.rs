// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-architecture capability defaults.
//!
//! Three-layer hierarchy:
//!   Atoms (`is_gfx*`)       — individual gfx-architecture codes
//!   Molecules (`is_rdna*`)  — architecture families composed of atoms
//!   Capabilities (`has_*`)  — hardware features composed of molecules/atoms
//!
//! Adding a new arch (e.g. gfx1300) requires:
//!   1. Add `is_gfx1300` atom
//!   2. Add it to any architecture molecules that cover it
//!   3. Capabilities propagate automatically

use crate::feature_flags::FeatureFlags;

/// Three-layer architecture descriptor computed once at `Gpu::init()` time.
pub struct ArchCaps {
    arch: String,

    // ── Atoms: per-gfx-code predicates ──────────────────────────────
    is_gfx906: bool,
    is_gfx908: bool,
    is_gfx1010: bool,
    is_gfx1011: bool,
    is_gfx1012: bool,
    is_gfx1030: bool,
    is_gfx1031: bool,
    is_gfx1032: bool,
    is_gfx1100: bool,
    is_gfx1101: bool,
    is_gfx1102: bool,
    is_gfx1103: bool,
    is_gfx1150: bool,
    is_gfx1151: bool,
    is_gfx1152: bool,
    is_gfx1200: bool,
    is_gfx1201: bool,
    is_gfx940: bool,
    is_gfx941: bool,
    is_gfx942: bool,

    // ── Molecules: architecture families composed from atoms ────────
    is_gcn5: bool,
    is_cdna1: bool,
    is_rdna1: bool,
    is_rdna1p1: bool,
    is_rdna2: bool,
    is_rdna3: bool,
    is_rdna3_dgpu: bool,
    is_rdna3p5: bool,
    is_rdna4: bool,
    is_cdna3: bool,

    // ── Capabilities: hardware features ─────────────────────────────
    has_wmma: bool,
    has_wmma_w32: bool,
    has_wmma_w32_gfx12: bool,
    has_dot2_f32_f16: bool,
    has_mmq: bool,
    is_gcn5_wave64: bool,
    is_wave32: bool,
    is_wave64_native: bool,
    has_hfq3_sdot4: bool,
    has_hfq3_dp4a: bool,
    has_hfq3_mmq: bool,
    has_hfq4_mmq: bool,
    has_cdna3_lds_gemv: bool,

    // ── Env-overridable tuning parameters ─────────────────────────
    gemv_dp4a: bool,
    gemv_prefetch: bool,
    gemv_rows_default: u32,

    // ── Reference to FeatureFlags for env-var overrides ───────────
    flags: std::sync::Arc<FeatureFlags>,
}

impl ArchCaps {
    pub fn new(arch: &str, flags: std::sync::Arc<FeatureFlags>) -> Self {
        // Atoms
        let is_gfx906 = arch == "gfx906";
        let is_gfx908 = arch == "gfx908";
        let is_gfx1010 = arch == "gfx1010";
        let is_gfx1011 = arch == "gfx1011";
        let is_gfx1012 = arch == "gfx1012";
        let is_gfx1030 = arch == "gfx1030";
        let is_gfx1031 = arch == "gfx1031";
        let is_gfx1032 = arch == "gfx1032";
        let is_gfx1100 = arch == "gfx1100";
        let is_gfx1101 = arch == "gfx1101";
        let is_gfx1102 = arch == "gfx1102";
        let is_gfx1103 = arch == "gfx1103";
        let is_gfx1150 = arch == "gfx1150";
        let is_gfx1151 = arch == "gfx1151";
        let is_gfx1152 = arch == "gfx1152";
        let is_gfx1200 = arch == "gfx1200";
        let is_gfx1201 = arch == "gfx1201";
        let is_gfx940 = arch == "gfx940";
        let is_gfx941 = arch == "gfx941";
        let is_gfx942 = arch == "gfx942";

        // Architecture molecules
        let is_gcn5 = is_gfx906;
        let is_cdna1 = is_gfx908;
        let is_rdna1 = is_gfx1010;
        let is_rdna1p1 = is_gfx1011 || is_gfx1012;
        let is_rdna2 = is_gfx1030 || is_gfx1031 || is_gfx1032;
        let is_rdna3 = is_gfx1100
            || is_gfx1101
            || is_gfx1102
            || is_gfx1103
            || is_gfx1150
            || is_gfx1151
            || is_gfx1152;
        let is_rdna3_dgpu = is_gfx1100 || is_gfx1101 || is_gfx1102;
        let is_rdna3p5 = is_gfx1150 || is_gfx1151 || is_gfx1152;
        let is_rdna4 = is_gfx1200 || is_gfx1201;
        let is_cdna3 = is_gfx940 || is_gfx941 || is_gfx942;

        // Capabilities
        let has_wmma = is_rdna3 || is_rdna4;
        let has_wmma_w32 = is_rdna3;
        let has_wmma_w32_gfx12 = is_rdna4;
        let has_dot2_f32_f16 = is_rdna1p1 || is_rdna2 || is_rdna3 || is_rdna4;
        let has_mmq = is_gfx906 || is_rdna3;
        let is_gcn5_wave64 = is_gfx906 || (is_gfx908 && flags.gcn5_wave64_hybrid.unwrap_or(false));
        let is_wave32 = is_rdna1 || is_rdna1p1 || is_rdna2 || is_rdna3 || is_rdna4;
        let is_wave64_native = is_gfx906 || is_gfx908 || is_cdna3;
        let has_hfq3_sdot4 = is_rdna1p1 || is_rdna2;

        // Env-gated capabilities
        let has_hfq3_dp4a = flags.hfq3_dp4a.unwrap_or(false) && has_hfq3_sdot4;
        // Issue #300: production HFQ3/HFQ4 MMQ prefill is default-on for
        // the supported sdot4 gfx10 allowlist; env flags remain escape hatches.
        let has_hfq3_mmq = flags.hfq3_mmq.unwrap_or(true) && has_hfq3_sdot4;
        let has_hfq4_mmq = flags.hfq4_mmq_rdna2.unwrap_or(true) && has_hfq3_sdot4;
        let has_cdna3_lds_gemv = flags.gfx942_lds_gemv.unwrap_or(false);

        // Tuning parameters
        let gemv_dp4a = flags.gemv_dp4a.unwrap_or(is_gfx906);
        let gemv_prefetch = flags.gemv_prefetch.unwrap_or(is_gfx906);
        let gemv_rows_default = flags.gemv_rows.unwrap_or_else(|| {
            if is_wave64_native || is_rdna2 || is_rdna3_dgpu {
                1
            } else {
                2
            }
        });

        Self {
            arch: arch.to_string(),
            is_gfx906,
            is_gfx908,
            is_gfx1010,
            is_gfx1011,
            is_gfx1012,
            is_gfx1030,
            is_gfx1031,
            is_gfx1032,
            is_gfx1100,
            is_gfx1101,
            is_gfx1102,
            is_gfx1103,
            is_gfx1150,
            is_gfx1151,
            is_gfx1152,
            is_gfx1200,
            is_gfx1201,
            is_gfx940,
            is_gfx941,
            is_gfx942,
            is_gcn5,
            is_cdna1,
            is_rdna1,
            is_rdna1p1,
            is_rdna2,
            is_rdna3,
            is_rdna3_dgpu,
            is_rdna3p5,
            is_rdna4,
            is_cdna3,
            has_wmma,
            has_wmma_w32,
            has_wmma_w32_gfx12,
            has_dot2_f32_f16,
            has_mmq,
            is_gcn5_wave64,
            is_wave32,
            is_wave64_native,
            has_hfq3_sdot4,
            has_hfq3_dp4a,
            has_hfq3_mmq,
            has_hfq4_mmq,
            has_cdna3_lds_gemv,
            gemv_dp4a,
            gemv_prefetch,
            gemv_rows_default,
            flags,
        }
    }

    pub fn should_use_mmq(&self, batch_size: usize) -> bool {
        if !self.has_mmq {
            return false;
        }
        match self.flags.mmq_override {
            Some(false) => false,
            Some(true) => true,
            None => {
                // gfx906 hits MMQ at very small batches (its dp4a MMQ kernel
                // beats fp16 wave64 from B≥8). RDNA3+ has a higher fixed cost
                // for the WMMA-INT8 dispatch (Q8_1 quantize prelude + MMQ tile
                // setup) but still beats single-warp WMMA from B=128 up —
                // measured +118% prefill on qwen3.6-27b.mq4 / gfx1151 by
                // dropping the cutoff from 256 → 128, with byte-identical
                // greedy decode parity (2026-05-29). RDNA2 keeps 256 for now
                // (untested at lower cutoffs in this session).
                let arch_min_batch: usize = if self.is_gfx906 {
                    8
                } else if self.is_rdna3 || self.is_rdna4 {
                    128
                } else {
                    256
                };
                let min_batch = self.flags.mmq_min_batch.unwrap_or(arch_min_batch);
                batch_size >= min_batch
            }
        }
    }

    // ── Atoms ─────────────────────────────────────────────────────
    pub fn is_gfx906(&self) -> bool {
        self.is_gfx906
    }
    pub fn is_gfx908(&self) -> bool {
        self.is_gfx908
    }
    pub fn is_gfx1010(&self) -> bool {
        self.is_gfx1010
    }
    pub fn is_gfx1011(&self) -> bool {
        self.is_gfx1011
    }
    pub fn is_gfx1012(&self) -> bool {
        self.is_gfx1012
    }
    pub fn is_gfx1030(&self) -> bool {
        self.is_gfx1030
    }
    pub fn is_gfx1031(&self) -> bool {
        self.is_gfx1031
    }
    pub fn is_gfx1032(&self) -> bool {
        self.is_gfx1032
    }
    pub fn is_gfx1100(&self) -> bool {
        self.is_gfx1100
    }
    pub fn is_gfx1101(&self) -> bool {
        self.is_gfx1101
    }
    pub fn is_gfx1102(&self) -> bool {
        self.is_gfx1102
    }
    pub fn is_gfx1103(&self) -> bool {
        self.is_gfx1103
    }
    pub fn is_gfx1150(&self) -> bool {
        self.is_gfx1150
    }
    pub fn is_gfx1151(&self) -> bool {
        self.is_gfx1151
    }
    pub fn is_gfx1152(&self) -> bool {
        self.is_gfx1152
    }
    pub fn is_gfx1200(&self) -> bool {
        self.is_gfx1200
    }
    pub fn is_gfx1201(&self) -> bool {
        self.is_gfx1201
    }
    pub fn is_gfx940(&self) -> bool {
        self.is_gfx940
    }
    pub fn is_gfx941(&self) -> bool {
        self.is_gfx941
    }
    pub fn is_gfx942(&self) -> bool {
        self.is_gfx942
    }

    // ── Molecules ─────────────────────────────────────────────────
    pub fn is_gcn5(&self) -> bool {
        self.is_gcn5
    }
    pub fn is_cdna1(&self) -> bool {
        self.is_cdna1
    }
    pub fn is_rdna1(&self) -> bool {
        self.is_rdna1
    }
    pub fn is_rdna1p1(&self) -> bool {
        self.is_rdna1p1
    }
    pub fn is_rdna2(&self) -> bool {
        self.is_rdna2
    }
    pub fn is_rdna3(&self) -> bool {
        self.is_rdna3
    }
    pub fn is_rdna3_dgpu(&self) -> bool {
        self.is_rdna3_dgpu
    }
    pub fn is_rdna3p5(&self) -> bool {
        self.is_rdna3p5
    }
    pub fn is_rdna4(&self) -> bool {
        self.is_rdna4
    }
    pub fn is_cdna3(&self) -> bool {
        self.is_cdna3
    }

    // ── Capabilities ──────────────────────────────────────────────
    pub fn has_wmma(&self) -> bool {
        self.has_wmma
    }
    pub fn has_wmma_w32(&self) -> bool {
        self.has_wmma_w32
    }
    pub fn has_wmma_w32_gfx12(&self) -> bool {
        self.has_wmma_w32_gfx12
    }
    pub fn has_dot2_f32_f16(&self) -> bool {
        self.has_dot2_f32_f16
    }
    pub fn has_mmq(&self) -> bool {
        self.has_mmq
    }
    pub fn is_gcn5_wave64(&self) -> bool {
        self.is_gcn5_wave64
    }
    pub fn is_wave32(&self) -> bool {
        self.is_wave32
    }
    pub fn is_wave64_native(&self) -> bool {
        self.is_wave64_native
    }
    pub fn has_hfq3_sdot4(&self) -> bool {
        self.has_hfq3_sdot4
    }
    pub fn has_hfq3_dp4a(&self) -> bool {
        self.has_hfq3_dp4a
    }
    pub fn has_hfq3_mmq(&self) -> bool {
        self.has_hfq3_mmq
    }
    pub fn has_hfq4_mmq(&self) -> bool {
        self.has_hfq4_mmq
    }
    pub fn has_cdna3_lds_gemv(&self) -> bool {
        self.has_cdna3_lds_gemv
    }
    pub fn gemv_dp4a_enabled(&self) -> bool {
        self.gemv_dp4a
    }
    pub fn gemv_prefetch_enabled(&self) -> bool {
        self.gemv_prefetch
    }
    pub fn gemv_rows_default(&self) -> u32 {
        self.gemv_rows_default
    }
    pub fn arch(&self) -> &str {
        &self.arch
    }
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

    // ── Atom tests ────────────────────────────────────────────────

    #[test]
    fn gfx_atoms_exclusive() {
        let all_gfxs = [
            "gfx906", "gfx908", "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032",
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx1201", "gfx940", "gfx941", "gfx942",
        ];
        for &arch in &all_gfxs {
            let caps = make_caps(arch);
            // Exactly one gfx atom should be true
            let count = [
                caps.is_gfx906(),
                caps.is_gfx908(),
                caps.is_gfx1010(),
                caps.is_gfx1011(),
                caps.is_gfx1012(),
                caps.is_gfx1030(),
                caps.is_gfx1031(),
                caps.is_gfx1032(),
                caps.is_gfx1100(),
                caps.is_gfx1101(),
                caps.is_gfx1102(),
                caps.is_gfx1103(),
                caps.is_gfx1150(),
                caps.is_gfx1151(),
                caps.is_gfx1152(),
                caps.is_gfx1200(),
                caps.is_gfx1201(),
                caps.is_gfx940(),
                caps.is_gfx941(),
                caps.is_gfx942(),
            ]
            .iter()
            .filter(|&&v| v)
            .count();
            assert_eq!(
                count, 1,
                "expected exactly 1 atom true for {arch}, got {count}"
            );
        }
    }

    // ── RDNA3 tests ───────────────────────────────────────────────

    #[test]
    fn rdna3_dgpu_covers_dgpus() {
        let caps = make_caps("gfx1100");
        assert!(caps.is_rdna3());
        assert!(caps.is_rdna3_dgpu());
        assert!(!caps.is_rdna3p5());
    }

    #[test]
    fn gfx1151_is_rdna3_and_rdna3p5() {
        let caps = make_caps("gfx1151");
        assert!(caps.is_rdna3());
        assert!(!caps.is_rdna3_dgpu());
        assert!(caps.is_rdna3p5());
        assert!(caps.is_gfx1151());
    }

    #[test]
    fn gfx1150_is_rdna3_and_rdna3p5() {
        let caps = make_caps("gfx1150");
        assert!(caps.is_rdna3());
        assert!(!caps.is_rdna3_dgpu());
        assert!(caps.is_rdna3p5());
        assert!(caps.is_gfx1150());
    }

    #[test]
    fn gfx1152_is_rdna3_and_rdna3p5() {
        let caps = make_caps("gfx1152");
        assert!(caps.is_rdna3());
        assert!(!caps.is_rdna3_dgpu());
        assert!(caps.is_rdna3p5());
    }

    #[test]
    fn gfx1103_is_rdna3() {
        let caps = make_caps("gfx1103");
        assert!(caps.is_rdna3());
        assert!(!caps.is_rdna3_dgpu());
        assert!(!caps.is_rdna3p5());
    }

    // ── RDNA4 / CDNA3 / GCN5 tests ────────────────────────────────

    #[test]
    fn rdna4() {
        let caps = make_caps("gfx1200");
        assert!(caps.is_rdna4());
        assert!(caps.has_wmma_w32_gfx12());
        assert!(caps.has_wmma());
        assert!(!caps.has_wmma_w32());
        assert!(!caps.is_rdna3());
    }

    #[test]
    fn cdna3_942() {
        let caps = make_caps("gfx942");
        assert!(caps.is_cdna3());
        assert!(caps.is_wave64_native());
        assert!(!caps.is_rdna3());
        assert!(!caps.has_wmma());
    }

    #[test]
    fn gcn5() {
        let caps = make_caps("gfx906");
        assert!(caps.is_gcn5());
        assert!(caps.is_gcn5_wave64());
        assert!(caps.is_wave64_native());
        assert!(caps.has_mmq());
        assert!(caps.is_gfx906());
    }

    #[test]
    fn rdna2() {
        let caps = make_caps("gfx1030");
        assert!(caps.is_rdna2());
        assert!(caps.has_dot2_f32_f16());
        assert!(!caps.has_wmma());
    }

    // ── Capability tests ──────────────────────────────────────────

    #[test]
    fn has_mmq_coverage() {
        assert!(make_caps("gfx906").has_mmq());
        assert!(make_caps("gfx1100").has_mmq());
        assert!(make_caps("gfx1150").has_mmq());
        assert!(!make_caps("gfx1030").has_mmq());
        assert!(!make_caps("gfx1200").has_mmq());
    }

    #[test]
    fn dot2_coverage() {
        for arch in &[
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201",
            "gfx1030", "gfx1031", "gfx1032", "gfx1011", "gfx1012",
        ] {
            assert!(
                make_caps(arch).has_dot2_f32_f16(),
                "dot2 missing for {arch}"
            );
        }
        assert!(!make_caps("gfx906").has_dot2_f32_f16());
        assert!(!make_caps("gfx1010").has_dot2_f32_f16());
    }

    #[test]
    fn hfq4_mmq_coverage_stays_on_sdot4_archs() {
        for arch in &["gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032"] {
            assert!(
                make_caps(arch).has_hfq4_mmq(),
                "hfq4 mmq missing for {arch}"
            );
        }
        for arch in &[
            "gfx906", "gfx1010", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
            "gfx1152", "gfx1200", "gfx1201",
        ] {
            assert!(
                !make_caps(arch).has_hfq4_mmq(),
                "hfq4 mmq should not compile on {arch}"
            );
        }
    }

    #[test]
    fn wmma_convenience() {
        assert!(make_caps("gfx1100").has_wmma());
        assert!(make_caps("gfx1151").has_wmma());
        assert!(make_caps("gfx1200").has_wmma());
        assert!(!make_caps("gfx1030").has_wmma());
        assert!(!make_caps("gfx906").has_wmma());
    }

    #[test]
    fn is_wave32_detection() {
        assert!(make_caps("gfx1010").is_wave32());
        assert!(make_caps("gfx1030").is_wave32());
        assert!(make_caps("gfx1100").is_wave32());
        assert!(make_caps("gfx1200").is_wave32());
        assert!(!make_caps("gfx906").is_wave32());
        assert!(!make_caps("gfx908").is_wave32());
    }

    #[test]
    fn wave64_native() {
        assert!(make_caps("gfx906").is_wave64_native());
        assert!(make_caps("gfx908").is_wave64_native());
        assert!(make_caps("gfx942").is_wave64_native());
        assert!(!make_caps("gfx1100").is_wave64_native());
    }
}
