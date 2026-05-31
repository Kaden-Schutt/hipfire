use rdna_compute::DType;

/// Every DType variant that represents a quantized format (byte-level).
const QUANTIZED_DTYPES: &[DType] = &[
    DType::Q4K, DType::Q6K, DType::Q8_0,
    DType::Q4F16G64, DType::Q4F16G32, DType::Q8HFQ,
    DType::HFQ4G256, DType::HFQ4G128,
    DType::PARO4G128, DType::PARO4G128T,
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
fn awq_sidecar_only_on_mq4_and_mq3() {
    assert!(DType::MQ4G256.supports_awq_sidecar());
    assert!(DType::MQ3G256.supports_awq_sidecar());
}

#[test]
fn awq_sidecar_not_on_mq_lloyd_or_hfq() {
    for dt in MAGNUMQUANT_DTYPES {
        if *dt == DType::MQ4G256 || *dt == DType::MQ3G256 { continue; }
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
