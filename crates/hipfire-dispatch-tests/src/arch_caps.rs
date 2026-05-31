use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::feature_flags::FeatureFlags;
use std::sync::Arc;

const ALL_ARCHS: &[&str] = &[
    "gfx906", "gfx908",
    "gfx1010", "gfx1011", "gfx1012",
    "gfx1030", "gfx1031", "gfx1032",
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152",
    "gfx1200", "gfx1201",
    "gfx940", "gfx941", "gfx942",
];

fn make_caps(arch: &str) -> ArchCaps {
    ArchCaps::new(arch, Arc::new(FeatureFlags::from_env_for_test(arch)))
}

// ── Atom exclusivity ───────────────────────────────────────────

#[test]
fn atoms_are_exclusive() {
    for &arch in ALL_ARCHS {
        let caps = make_caps(arch);
        let mut count = 0;
        if caps.is_gfx906() { count += 1; }
        if caps.is_gfx908() { count += 1; }
        if caps.is_gfx1010() { count += 1; }
        if caps.is_gfx1011() { count += 1; }
        if caps.is_gfx1012() { count += 1; }
        if caps.is_gfx1030() { count += 1; }
        if caps.is_gfx1031() { count += 1; }
        if caps.is_gfx1032() { count += 1; }
        if caps.is_gfx1100() { count += 1; }
        if caps.is_gfx1101() { count += 1; }
        if caps.is_gfx1102() { count += 1; }
        if caps.is_gfx1103() { count += 1; }
        if caps.is_gfx1150() { count += 1; }
        if caps.is_gfx1151() { count += 1; }
        if caps.is_gfx1152() { count += 1; }
        if caps.is_gfx1200() { count += 1; }
        if caps.is_gfx1201() { count += 1; }
        if caps.is_gfx940() { count += 1; }
        if caps.is_gfx941() { count += 1; }
        if caps.is_gfx942() { count += 1; }
        assert_eq!(count, 1, "{arch}: expected exactly 1 atom true, got {count}");
    }
}

// ── Molecule membership ────────────────────────────────────────

#[test]
fn gcn5_is_only_gfx906() {
    assert!(make_caps("gfx906").is_gcn5());
    for &arch in &ALL_ARCHS[1..] {
        assert!(!make_caps(arch).is_gcn5(), "{arch} should not be GCN5");
    }
}

#[test]
fn rdna1_is_gfx1010() {
    assert!(make_caps("gfx1010").is_rdna1());
    assert!(!make_caps("gfx1011").is_rdna1());
    assert!(!make_caps("gfx1030").is_rdna1());
}

#[test]
fn rdna1p1_is_gfx1011_or_gfx1012() {
    assert!(make_caps("gfx1011").is_rdna1p1());
    assert!(make_caps("gfx1012").is_rdna1p1());
    assert!(!make_caps("gfx1010").is_rdna1p1());
}

#[test]
fn rdna2_coverage() {
    for &arch in &["gfx1030", "gfx1031", "gfx1032"] {
        assert!(make_caps(arch).is_rdna2(), "{arch} should be RDNA2");
    }
    assert!(!make_caps("gfx1010").is_rdna2());
    assert!(!make_caps("gfx1100").is_rdna2());
}

#[test]
fn rdna3_coverage() {
    for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1103",
                    "gfx1150", "gfx1151", "gfx1152"] {
        assert!(make_caps(arch).is_rdna3(), "{arch} should be RDNA3");
    }
    assert!(!make_caps("gfx1030").is_rdna3());
    assert!(!make_caps("gfx1200").is_rdna3());
}

#[test]
fn rdna3_dgpu_is_gfx1100_1101_1102() {
    for &arch in &["gfx1100", "gfx1101", "gfx1102"] {
        assert!(make_caps(arch).is_rdna3_dgpu(), "{arch} should be RDNA3 dGPU");
    }
    assert!(!make_caps("gfx1103").is_rdna3_dgpu());
    assert!(!make_caps("gfx1150").is_rdna3_dgpu());
    assert!(!make_caps("gfx1151").is_rdna3_dgpu());
}

#[test]
fn rdna3p5_is_strix_halo() {
    for &arch in &["gfx1150", "gfx1151", "gfx1152"] {
        assert!(make_caps(arch).is_rdna3p5(), "{arch} should be RDNA3.5");
    }
    assert!(!make_caps("gfx1100").is_rdna3p5());
}

#[test]
fn rdna4_coverage() {
    for &arch in &["gfx1200", "gfx1201"] {
        assert!(make_caps(arch).is_rdna4(), "{arch} should be RDNA4");
    }
    assert!(!make_caps("gfx1100").is_rdna4());
}

#[test]
fn cdna3_coverage() {
    for &arch in &["gfx940", "gfx941", "gfx942"] {
        assert!(make_caps(arch).is_cdna3(), "{arch} should be CDNA3");
    }
    assert!(!make_caps("gfx906").is_cdna3());
}

// ── Capability matrix ──────────────────────────────────────────

#[test]
fn wmma_is_rdna3_or_rdna4() {
    for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1103",
                    "gfx1150", "gfx1151", "gfx1152",
                    "gfx1200", "gfx1201"] {
        assert!(make_caps(arch).has_wmma(), "{arch} should have WMMA");
    }
    for &arch in &["gfx906", "gfx908", "gfx1010", "gfx1030", "gfx940"] {
        assert!(!make_caps(arch).has_wmma(), "{arch} should NOT have WMMA");
    }
}

#[test]
fn wmma_w32_is_rdna3_only() {
    for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1103",
                    "gfx1150", "gfx1151", "gfx1152"] {
        assert!(make_caps(arch).has_wmma_w32(), "{arch} should have WMMA wave32");
    }
    assert!(!make_caps("gfx1200").has_wmma_w32(), "gfx1200 should NOT have WMMA wave32");
    assert!(!make_caps("gfx1030").has_wmma_w32(), "gfx1030 should NOT have WMMA wave32");
}

#[test]
fn wmma_w32_gfx12_is_rdna4_only() {
    for &arch in &["gfx1200", "gfx1201"] {
        assert!(make_caps(arch).has_wmma_w32_gfx12(), "{arch} should have gfx12 WMMA");
    }
    assert!(!make_caps("gfx1100").has_wmma_w32_gfx12());
}

#[test]
fn dot2_f32_f16_coverage() {
    for &arch in &["gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032",
                    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
                    "gfx1150", "gfx1151", "gfx1152",
                    "gfx1200", "gfx1201"] {
        assert!(make_caps(arch).has_dot2_f32_f16(), "{arch} should have dot2");
    }
    assert!(!make_caps("gfx906").has_dot2_f32_f16());
    assert!(!make_caps("gfx1010").has_dot2_f32_f16());
}

#[test]
fn mmq_is_gcn5_or_rdna3() {
    assert!(make_caps("gfx906").has_mmq());
    for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1103",
                    "gfx1150", "gfx1151", "gfx1152"] {
        assert!(make_caps(arch).has_mmq(), "{arch} should have MMQ");
    }
    assert!(!make_caps("gfx1030").has_mmq());
    assert!(!make_caps("gfx1010").has_mmq());
    assert!(!make_caps("gfx1200").has_mmq());
    assert!(!make_caps("gfx940").has_mmq());
}

#[test]
fn hfq3_sdot4_is_rdna1p1_or_rdna2() {
    for &arch in &["gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032"] {
        assert!(make_caps(arch).has_hfq3_sdot4(), "{arch} should have sdot4");
    }
    assert!(!make_caps("gfx1010").has_hfq3_sdot4());
    assert!(!make_caps("gfx1100").has_hfq3_sdot4());
    assert!(!make_caps("gfx1200").has_hfq3_sdot4());
}

#[test]
fn wave32_on_rdna() {
    for &arch in &["gfx1010", "gfx1030", "gfx1100", "gfx1200"] {
        assert!(make_caps(arch).is_wave32(), "{arch} should be wave32");
    }
    assert!(!make_caps("gfx906").is_wave32());
    assert!(!make_caps("gfx942").is_wave32());
}

#[test]
fn wave64_native_on_gcn5_cdna1_cdna3() {
    for &arch in &["gfx906", "gfx908", "gfx940", "gfx941", "gfx942"] {
        assert!(make_caps(arch).is_wave64_native(), "{arch} should be wave64 native");
    }
    assert!(!make_caps("gfx1100").is_wave64_native());
}

#[test]
fn gemv_rows_default_is_1_on_wave64_rdna2_rdna3dgpu() {
    for &arch in &["gfx906", "gfx908", "gfx940", "gfx941", "gfx942",
                    "gfx1030", "gfx1031",
                    "gfx1100", "gfx1101", "gfx1102"] {
        assert_eq!(make_caps(arch).gemv_rows_default(), 1,
                    "{arch}: expected gemv_rows_default = 1");
    }
    // RDNA1, RDNA1p1, RDNA3p5, RDNA4 default to 2
    for &arch in &["gfx1010", "gfx1011", "gfx1012",
                    "gfx1150", "gfx1151", "gfx1152",
                    "gfx1200", "gfx1201"] {
        assert_eq!(make_caps(arch).gemv_rows_default(), 2,
                    "{arch}: expected gemv_rows_default = 2");
    }
}

// ── should_use_mmq ─────────────────────────────────────────────

#[test]
fn mmq_disabled_on_non_mmq_archs() {
    for &arch in &["gfx1010", "gfx1030", "gfx1200"] {
        assert!(!make_caps(arch).should_use_mmq(1024), "{arch}: MMQ should be disabled");
    }
}

#[test]
fn mmq_on_gfx906_at_small_batch() {
    let caps = make_caps("gfx906");
    assert!(caps.should_use_mmq(8));
    assert!(caps.should_use_mmq(16));
    assert!(!caps.should_use_mmq(4));
}

#[test]
fn mmq_on_rdna3_at_128() {
    for &arch in &["gfx1100", "gfx1150"] {
        let caps = make_caps(arch);
        assert!(caps.should_use_mmq(128), "{arch}: MMQ should engage at batch 128");
        assert!(caps.should_use_mmq(256), "{arch}: MMQ should engage at batch 256");
        assert!(!caps.should_use_mmq(64), "{arch}: MMQ should NOT engage at batch 64");
    }
}
