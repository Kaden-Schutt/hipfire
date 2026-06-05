use rdna_compute::DType;

/// Every DType variant that represents a quantized format (byte-level).
const QUANTIZED_DTYPES: &[DType] = &[
    DType::Q4K, DType::Q6K, DType::Q8_0,
    DType::Q4F16G64, DType::Q4F16G32, DType::Q8HFQ,
    DType::HFQ4G256, DType::HFQ4G128,
    DType::HFQ3G256, DType::HFQ3G128,
    DType::MQ4G256, DType::MQ4G128,
    DType::MQ8G256, DType::MQ6G256,
    DType::MQ3G256, DType::MQ2G256,
    DType::MQ2G256Lloyd, DType::MQ3G256Lloyd, DType::MQ4G256Lloyd,
    DType::HFP4G32, DType::MFP4G32,
    DType::HFQ2G256, DType::HFQ2G128, DType::HFQ6G256,
    DType::ParoQ4G128, DType::Raw,
];

/// DTypes that are MQ-family (FWHT-rotated MagnumQuant).
const MAGNUMQUANT_DTYPES: &[DType] = &[
    DType::MQ4G256, DType::MQ4G128,
    DType::MQ8G256, DType::MQ6G256,
    DType::MQ3G256, DType::MQ2G256,
    DType::MQ2G256Lloyd, DType::MQ3G256Lloyd, DType::MQ4G256Lloyd,
    DType::MFP4G32,
];

/// DTypes that are HFQ-family (flat quant with inline f32 scale+zero).
const HFQ_DTYPES: &[DType] = &[
    DType::HFQ4G256, DType::HFQ4G128,
    DType::HFQ3G256, DType::HFQ3G128,
    DType::HFQ2G256, DType::HFQ2G128,
    DType::HFQ6G256,
];

// ── DType::size() ──────────────────────────────────────────────

#[test]
fn f32_size_is_correct() {
    assert_eq!(DType::F32.size(), 4);
}

#[test]
fn f16_size_is_correct() {
    assert_eq!(DType::F16.size(), 2);
}

#[test]
fn quantized_dtypes_have_size_1() {
    for dt in QUANTIZED_DTYPES {
        assert_eq!(dt.size(), 1, "DType::{dt:?} expected size 1");
    }
}

// ── DType::supports_awq_sidecar() ──────────────────────────────

#[test]
fn awq_sidecar_on_mq4_mq3_mq2_and_lloyd() {
    // #415 broadened AWQ-sidecar to the sub-4-bit + Lloyd arms (AWQ×Lloyd).
    assert!(DType::MQ4G256.supports_awq_sidecar());
    assert!(DType::MQ3G256.supports_awq_sidecar());
    assert!(DType::MQ2G256.supports_awq_sidecar());
    assert!(DType::MQ3G256Lloyd.supports_awq_sidecar());
    assert!(DType::MQ2G256Lloyd.supports_awq_sidecar());
}

#[test]
fn awq_sidecar_not_on_non_awq_dtypes() {
    // AWQ-eligible set after #415: MQ4/MQ3/MQ2/MQ3-Lloyd/MQ2-Lloyd. Everything else off.
    for dt in MAGNUMQUANT_DTYPES {
        if matches!(
            *dt,
            DType::MQ4G256 | DType::MQ3G256 | DType::MQ2G256 | DType::MQ3G256Lloyd | DType::MQ2G256Lloyd
        ) { continue; }
        assert!(!dt.supports_awq_sidecar(), "DType::{dt:?} should NOT support AWQ");
    }
    for dt in HFQ_DTYPES {
        assert!(!dt.supports_awq_sidecar(), "DType::{dt:?} should NOT support AWQ");
    }
    assert!(!DType::F32.supports_awq_sidecar());
    assert!(!DType::F16.supports_awq_sidecar());
    assert!(!DType::Q8_0.supports_awq_sidecar());
    assert!(!DType::HFP4G32.supports_awq_sidecar());
    assert!(!DType::ParoQ4G128.supports_awq_sidecar());
}

// ─── Quant family dispatch dimensions ──────────────────────────

#[test]
fn mq_dtypes_are_magnum_quant_formats() {
    for dt in MAGNUMQUANT_DTYPES {
        assert_eq!(dt.size(), 1, "DType::{dt:?} is MQ-family");
    }
}

#[test]
fn rotation_plan_covers_every_dtype() {
    use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
    use rdna_compute::DType;
    assert_eq!(dtype_rotation_plan(DType::F32), RotationPlan::None);
    assert_eq!(dtype_rotation_plan(DType::HFQ4G256), RotationPlan::None);
    assert_eq!(dtype_rotation_plan(DType::Q8HFQ), RotationPlan::None);
    assert_eq!(dtype_rotation_plan(DType::MQ4G256), RotationPlan::FwhtG256);
    assert_eq!(dtype_rotation_plan(DType::MQ6G256), RotationPlan::FwhtG256);
    assert_eq!(dtype_rotation_plan(DType::MFP4G32), RotationPlan::FwhtG256);
    assert_eq!(dtype_rotation_plan(DType::MQ4G128), RotationPlan::FwhtG128);
    assert_eq!(dtype_rotation_plan(DType::MQ8G256), RotationPlan::Mq8Internal);
    assert_eq!(dtype_rotation_plan(DType::ParoQ4G128), RotationPlan::Givens);
}

#[test]
fn rotation_plan_matches_legacy_needs_fwht() {
    use hipfire_dispatch::types::{dtype_rotation_plan, dtype_needs_rotation, RotationPlan};
    for d in QUANTIZED_DTYPES {
        assert_eq!(
            dtype_rotation_plan(*d) != RotationPlan::None,
            dtype_needs_rotation(*d),
            "rotation_plan/needs_fwht disagree for {:?}", d
        );
    }
    for d in [DType::F32, DType::F16, DType::Q8_0] {
        assert_eq!(
            dtype_rotation_plan(d) != RotationPlan::None,
            dtype_needs_rotation(d),
            "rotation_plan/needs_fwht disagree for {:?}", d
        );
    }
}

#[test]
fn post_rotation_variant_paro_is_plain_mq_is_prerotated() {
    use hipfire_dispatch::types::{dtype_post_rotation_variant, GemvVariant};
    use rdna_compute::DType;
    assert_eq!(dtype_post_rotation_variant(DType::ParoQ4G128), GemvVariant::Plain);
    assert_eq!(dtype_post_rotation_variant(DType::MQ4G256), GemvVariant::Prerotated);
    assert_eq!(dtype_post_rotation_variant(DType::MQ8G256), GemvVariant::Prerotated);
    assert_eq!(dtype_post_rotation_variant(DType::MQ4G128), GemvVariant::Prerotated);
    assert_eq!(dtype_post_rotation_variant(DType::HFQ4G256), GemvVariant::Plain);
}

#[test]
fn q8hfq_resolves_to_plain_gemv_key() {
    use hipfire_dispatch::types::{KernelKey, GemvVariant};
    use rdna_compute::DType;
    let key = KernelKey::for_gemv(DType::Q8HFQ, GemvVariant::Plain, false)
        .expect("Q8HFQ Plain must resolve");
    assert_eq!(key, KernelKey::GemvQ8HFQ);
}

#[test]
fn rotation_tag_distinguishes_awq_and_batched() {
    use hipfire_dispatch::families::gemv::RotationTag;
    use hipfire_dispatch::types::RotationPlan;
    let base = RotationTag { plan: RotationPlan::FwhtG256, awq: false, batched: false };
    let awq  = RotationTag { plan: RotationPlan::FwhtG256, awq: true,  batched: false };
    let bat  = RotationTag { plan: RotationPlan::FwhtG256, awq: false, batched: true };
    assert_ne!(base, awq, "AWQ vs non-AWQ must not compare equal");
    assert_ne!(base, bat, "batched vs non-batched must not compare equal");
    assert_eq!(base, RotationTag { plan: RotationPlan::FwhtG256, awq: false, batched: false });
}

#[test]
fn run_rejects_tag_plan_mismatch() {
    use hipfire_dispatch::families::gemv::{check_rotation_tag, RotationTag};
    use hipfire_dispatch::types::RotationPlan;
    let want = RotationTag { plan: RotationPlan::FwhtG256, awq: false, batched: false };
    let givens = RotationTag { plan: RotationPlan::Givens, awq: false, batched: false };
    let awq = RotationTag { plan: RotationPlan::FwhtG256, awq: true, batched: false };
    assert!(check_rotation_tag(want, want).is_ok());
    assert!(check_rotation_tag(want, givens).is_err(), "plan mismatch must reject");
    assert!(check_rotation_tag(want, awq).is_err(), "awq mismatch must reject");
}
#[test]
fn rotate_variant_selection() {
    use hipfire_dispatch::families::gemv::select_rotation_variant;
    use hipfire_dispatch::types::{RotationPlan, RotationVariant};
    assert_eq!(select_rotation_variant(RotationPlan::FwhtG256, false, false), RotationVariant::Plain);
    assert_eq!(select_rotation_variant(RotationPlan::FwhtG256, true,  false), RotationVariant::WithRmsnorm);
    assert_eq!(select_rotation_variant(RotationPlan::FwhtG256, false, true),  RotationVariant::WithSwiGLU);
    assert_eq!(select_rotation_variant(RotationPlan::FwhtG128, false, false), RotationVariant::PlainG128);
    assert_eq!(select_rotation_variant(RotationPlan::Givens,   false, false), RotationVariant::Givens);
}
