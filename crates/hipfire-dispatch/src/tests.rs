// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unit tests for the hipfire-dispatch layer.
//!
//! Tests cover:
//! - `ShapePredicate::eval` — all three variants, boundary values
//! - `ArchPredicate::eval_arch` — key arch identities (RDNA1/2/3)
//! - `KernelRegistry` — register, resolve, arch gating, shape gating, fallback
//! - `KernelKey::for_gemv*` — dtype/variant → key mapping
//! - `dtype_needs_fwht` — MQ family true, HFQ/F32 false
//! - `GemvFamily::resolve` — arch predicate filtering via a real registry
//! - `Pipeline::can_satisfy` — prefix-match semantics

use crate::context::DispatchCtx;
use crate::families::gemv::GemvFamily;
use crate::pipeline::Pipeline;
use crate::tables::KernelRegistry;
use crate::types::*;
use rdna_compute::DType;

// ── helpers ───────────────────────────────────────────────────────────────────

/// gfx1010 = RDNA1: no dp4a, no WMMA, no MMQ.
fn ctx_rdna1() -> DispatchCtx {
    DispatchCtx::for_test("gfx1010")
}

/// gfx1030 = RDNA2: has dp4a, no WMMA w32, no MMQ.
fn ctx_rdna2() -> DispatchCtx {
    DispatchCtx::for_test("gfx1030")
}

/// gfx1100 = RDNA3: has dp4a, WMMA w32, MMQ.
fn ctx_rdna3() -> DispatchCtx {
    DispatchCtx::for_test("gfx1100")
}

/// gfx1200 = RDNA4: has dp4a, WMMA w32 gfx12, no MMQ via gfx11 path.
fn ctx_rdna4() -> DispatchCtx {
    DispatchCtx::for_test("gfx1200")
}

fn always_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[],
        has_awq: false,
    }
}

fn wmma_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[],
        has_awq: false,
    }
}

fn dp4a_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::HasDp4a,
        shape_gate: None,
        steps: &[],
        has_awq: false,
    }
}

// ── ShapePredicate::eval ──────────────────────────────────────────────────────

#[test]
fn shape_batch_gt_passes_when_strictly_greater() {
    let s = ShapeInfo { batch_size: 2, ..Default::default() };
    assert!(ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_batch_gt_fails_when_equal() {
    let s = ShapeInfo { batch_size: 1, ..Default::default() };
    assert!(!ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_batch_gt_fails_when_less() {
    let s = ShapeInfo { batch_size: 0, ..Default::default() };
    assert!(!ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_head_dim_eq_passes_on_match() {
    let s = ShapeInfo { head_dim: 128, ..Default::default() };
    assert!(ShapePredicate::HeadDimEq(128).eval(&s));
}

#[test]
fn shape_head_dim_eq_fails_on_mismatch() {
    let s = ShapeInfo { head_dim: 64, ..Default::default() };
    assert!(!ShapePredicate::HeadDimEq(128).eval(&s));
}

#[test]
fn shape_m_lt_passes_when_strictly_less() {
    let s = ShapeInfo { m: 7, ..Default::default() };
    assert!(ShapePredicate::MLt(8).eval(&s));
}

#[test]
fn shape_m_lt_fails_when_equal() {
    let s = ShapeInfo { m: 8, ..Default::default() };
    assert!(!ShapePredicate::MLt(8).eval(&s));
}

#[test]
fn shape_m_lt_fails_when_greater() {
    let s = ShapeInfo { m: 9, ..Default::default() };
    assert!(!ShapePredicate::MLt(8).eval(&s));
}

// ── ArchPredicate::eval_arch ──────────────────────────────────────────────────

#[test]
fn arch_always_passes_on_all_archs() {
    assert!(ArchPredicate::Always.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::Always.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::Always.eval_arch(&ctx_rdna3()));
}

#[test]
fn arch_has_wmma_w32_requires_rdna3() {
    assert!(!ArchPredicate::HasWmmaW32.eval_arch(&ctx_rdna1()));
    assert!(!ArchPredicate::HasWmmaW32.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasWmmaW32.eval_arch(&ctx_rdna3()));
    assert!(!ArchPredicate::HasWmmaW32.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_wmma_w32_gfx12_requires_rdna4() {
    assert!(!ArchPredicate::HasWmmaW32Gfx12.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasWmmaW32Gfx12.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_dp4a_requires_rdna1p1_or_newer() {
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_mmq_on_rdna3_only() {
    assert!(!ArchPredicate::HasMmq.eval_arch(&ctx_rdna1()));
    assert!(!ArchPredicate::HasMmq.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasMmq.eval_arch(&ctx_rdna3()));
    assert!(!ArchPredicate::HasMmq.eval_arch(&ctx_rdna4()));
}

// ── KernelRegistry ────────────────────────────────────────────────────────────

#[test]
fn registry_resolve_happy_path() {
    let mut reg = KernelRegistry::new();
    reg.register(always_variant(KernelKey::GemvF32));
    let ctx = ctx_rdna1();
    assert_eq!(reg.resolve(KernelKey::GemvF32, &ctx, None).unwrap().key, KernelKey::GemvF32);
}

#[test]
fn registry_resolve_unregistered_key_returns_not_found() {
    let mut reg = KernelRegistry::new();
    let ctx = ctx_rdna1();
    let err = reg.resolve(KernelKey::GemvF32, &ctx, None).unwrap_err();
    assert!(matches!(err, DispatchError::NotFound { .. }));
}

#[test]
fn registry_resolve_arch_gate_fails_returns_missing_impl() {
    let mut reg = KernelRegistry::new();
    reg.register(wmma_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna1(); // no WMMA
    let err = reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None).unwrap_err();
    assert!(matches!(err, DispatchError::MissingImpl { .. }));
}

#[test]
fn registry_resolve_arch_gate_passes_on_capable_arch() {
    let mut reg = KernelRegistry::new();
    reg.register(wmma_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna3(); // has WMMA w32
    assert_eq!(
        reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None).unwrap().key,
        KernelKey::GemmHfq4G256Wmma,
    );
}

#[test]
fn registry_resolve_falls_through_to_second_variant() {
    // Register WMMA variant first, then fallback Always variant for same key.
    // On RDNA1 (no WMMA), the WMMA entry is skipped and fallback is selected.
    let mut reg = KernelRegistry::new();
    reg.register(wmma_variant(KernelKey::GemmHfq4G256Wmma));
    reg.register(always_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna1();
    assert_eq!(
        reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None).unwrap().key,
        KernelKey::GemmHfq4G256Wmma,
    );
}

#[test]
fn registry_resolve_shape_gate_passes_when_shape_matches() {
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
    });
    let ctx = ctx_rdna1();
    let shape = ShapeInfo { head_dim: 128, ..Default::default() };
    assert_eq!(reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape)).unwrap().key, KernelKey::AttnF32);
}

#[test]
fn registry_resolve_shape_gate_skips_when_shape_mismatches() {
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
    });
    let ctx = ctx_rdna1();
    let shape = ShapeInfo { head_dim: 64, ..Default::default() };
    let err = reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape)).unwrap_err();
    assert!(matches!(err, DispatchError::MissingImpl { .. }));
}

#[test]
fn registry_resolve_shape_none_bypasses_shape_gate() {
    // With shape=None, even a shape-gated variant should be selected.
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
    });
    let ctx = ctx_rdna1();
    assert_eq!(reg.resolve(KernelKey::AttnF32, &ctx, None).unwrap().key, KernelKey::AttnF32);
}

#[test]
fn registry_resolve_shape_gate_fallback_to_ungated_variant() {
    // Shape-gated fast path (head_dim=128) followed by ungated fallback.
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
    });
    reg.register(always_variant(KernelKey::AttnF32)); // ungated fallback
    let ctx = ctx_rdna1();
    let shape = ShapeInfo { head_dim: 64, ..Default::default() }; // doesn't match gated
    assert_eq!(reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape)).unwrap().key, KernelKey::AttnF32);
}

#[test]
fn registry_validate_succeeds_on_populated_registry() {
    let mut reg = KernelRegistry::new();
    reg.register(always_variant(KernelKey::GemvF32));
    assert!(reg.validate().is_ok());
}

#[test]
fn registry_all_keys_returns_registered_keys() {
    let mut reg = KernelRegistry::new();
    reg.register(always_variant(KernelKey::GemvF32));
    reg.register(always_variant(KernelKey::GemvF16));
    let keys = reg.all_keys();
    assert!(keys.contains(&KernelKey::GemvF32));
    assert!(keys.contains(&KernelKey::GemvF16));
    assert_eq!(keys.len(), 2);
}

// ── KernelKey::for_gemv* ──────────────────────────────────────────────────────

#[test]
fn for_gemv_plain_maps_all_scalar_dtypes() {
    let cases = [
        (DType::F32,       KernelKey::GemvF32),
        (DType::F16,       KernelKey::GemvF16),
        (DType::Q8_0,      KernelKey::GemvQ8_0),
        (DType::HFQ4G256,  KernelKey::GemvHfq4G256),
        (DType::MQ4G256,   KernelKey::GemvMq4G256),
        (DType::MQ3G256,   KernelKey::GemvMq3G256),
        (DType::MFP4G32,   KernelKey::GemvMfp4G32),
    ];
    for (dtype, expected) in cases {
        assert_eq!(
            KernelKey::for_gemv(dtype, GemvVariant::Plain, false).unwrap(),
            expected,
            "dtype {dtype:?}",
        );
    }
}

#[test]
fn for_gemv_prerotated_maps_mq_family() {
    let cases = [
        (DType::MQ4G256, KernelKey::GemvMq4G256Prerotated),
        (DType::MQ3G256, KernelKey::GemvMq3G256Prerotated),
        (DType::MQ2G256, KernelKey::GemvMq2G256Prerotated),
        (DType::MQ6G256, KernelKey::GemvMq6G256Prerotated),
        (DType::MQ8G256, KernelKey::GemvMq8G256Prerotated),
        (DType::MFP4G32, KernelKey::GemvMfp4G32Prerotated),
    ];
    for (dtype, expected) in cases {
        assert_eq!(
            KernelKey::for_gemv_prerotated(dtype).unwrap(),
            expected,
            "dtype {dtype:?}",
        );
    }
}

#[test]
fn for_gemv_prerotated_rejects_non_mq_dtypes() {
    for dtype in [DType::F32, DType::HFQ4G256, DType::Q8_0] {
        assert!(
            KernelKey::for_gemv_prerotated(dtype).is_err(),
            "expected error for {dtype:?}",
        );
    }
}

#[test]
fn for_gemv_residual_maps_hfq_and_mq() {
    let cases = [
        (DType::HFQ4G256,     KernelKey::GemvHfq4G256Residual),
        (DType::HFQ3G256,     KernelKey::GemvHfq3G256Residual),
        (DType::HFQ6G256,     KernelKey::GemvHfq6G256Residual),
        (DType::MQ4G256,      KernelKey::GemvMq4G256Residual),
        (DType::MQ3G256Lloyd, KernelKey::GemvMq3G256LloydResidual),
    ];
    for (dtype, expected) in cases {
        assert_eq!(
            KernelKey::for_gemv_residual(dtype).unwrap(),
            expected,
            "dtype {dtype:?}",
        );
    }
}

#[test]
fn for_gemv_swiglu_residual_maps_hfq_and_mq() {
    assert_eq!(
        KernelKey::for_gemv_swiglu_residual(DType::HFQ4G256).unwrap(),
        KernelKey::GemvHfq4G256SwiGLUResidual,
    );
    assert_eq!(
        KernelKey::for_gemv_swiglu_residual(DType::MQ4G256Lloyd).unwrap(),
        KernelKey::GemvMq4G256LloydSwiGLUResidual,
    );
}

#[test]
fn for_gemv_rejects_unsupported_variant_combo() {
    // Prerotated for F32 has no kernel.
    assert!(KernelKey::for_gemv_prerotated(DType::F32).is_err());
    // Residual for F32 has no kernel.
    assert!(KernelKey::for_gemv_residual(DType::F32).is_err());
}

// ── dtype_needs_fwht ──────────────────────────────────────────────────────────

#[test]
fn dtype_needs_fwht_true_for_mq_family() {
    for dtype in [
        DType::MQ4G256, DType::MQ3G256, DType::MQ2G256, DType::MQ6G256,
        DType::MQ8G256, DType::MQ4G256Lloyd, DType::MFP4G32,
    ] {
        assert!(dtype_needs_fwht(dtype), "{dtype:?} should need FWHT");
    }
}

#[test]
fn dtype_needs_fwht_false_for_hfq_and_scalar() {
    for dtype in [DType::F32, DType::F16, DType::HFQ4G256, DType::Q8_0, DType::HFP4G32] {
        assert!(!dtype_needs_fwht(dtype), "{dtype:?} should NOT need FWHT");
    }
}

#[test]
fn gemv_steps_rotation_matches_plan() {
    for dtype in [DType::MQ4G256, DType::MFP4G32, DType::ParoQ4G128, DType::HFQ4G256] {
        let steps = KernelKey::gemv_steps(dtype, GemvVariant::Plain);
        let plan = dtype_rotation_plan(dtype);
        let has_fwht = steps.contains(&PipelineOp::RotateFwht);
        let has_givens = steps.contains(&PipelineOp::GivensRotate);
        match plan {
            RotationPlan::Givens => { assert!(has_givens && !has_fwht, "{dtype:?}: Givens plan must emit GivensRotate, not FWHT"); }
            RotationPlan::FwhtG256 | RotationPlan::FwhtG128 => { assert!(has_fwht && !has_givens, "{dtype:?}: FWHT plan must emit RotateFwht"); }
            RotationPlan::None => { assert!(!has_fwht && !has_givens, "{dtype:?}: no rotation"); }
            RotationPlan::Mq8Internal => {}
        }
    }
}

// ── GemvFamily::resolve via populated table ───────────────────────────────────

#[test]
fn gemv_family_resolves_f32_on_all_archs() {
    let fam = GemvFamily::new();
    assert!(fam.resolve(DType::F32, GemvVariant::Plain, false, &ctx_rdna1(), None).is_ok());
    assert!(fam.resolve(DType::F32, GemvVariant::Plain, false, &ctx_rdna3(), None).is_ok());
}

#[test]
fn gemv_family_resolves_hfq4_only_on_dp4a_arch() {
    let fam = GemvFamily::new();
    // RDNA1 has no dp4a → HFQ4G256 plain should fail
    assert!(fam.resolve(DType::HFQ4G256, GemvVariant::Plain, false, &ctx_rdna1(), None).is_err());
    assert!(fam.resolve(DType::HFQ4G256, GemvVariant::Plain, false, &ctx_rdna2(), None).is_ok());
    assert!(fam.resolve(DType::HFQ4G256, GemvVariant::Plain, false, &ctx_rdna3(), None).is_ok());
}

#[test]
fn gemv_family_resolves_mq3_prerotated_only_on_wmma_arch() {
    let fam = GemvFamily::new();
    assert!(fam.resolve(DType::MQ3G256, GemvVariant::Prerotated, false, &ctx_rdna2(), None).is_err());
    assert!(fam.resolve(DType::MQ3G256, GemvVariant::Prerotated, false, &ctx_rdna3(), None).is_ok());
    assert!(fam.resolve(DType::MQ4G256, GemvVariant::Prerotated, false, &ctx_rdna2(), None).is_ok());
    assert!(fam.resolve(DType::F32, GemvVariant::Prerotated, false, &ctx_rdna3(), None).is_err());
}

// ── Pipeline::can_satisfy ─────────────────────────────────────────────────────

#[test]
fn pipeline_exact_match_satisfies() {
    let p = Pipeline::new(&[PipelineOp::RotateFwht, PipelineOp::Gemv]);
    assert!(p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
}

#[test]
fn pipeline_prefix_satisfies_longer_request() {
    let p = Pipeline::new(&[PipelineOp::RotateFwht]);
    assert!(p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
}

#[test]
fn pipeline_empty_satisfies_any_request() {
    let p = Pipeline::new(&[]);
    assert!(p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
    assert!(p.can_satisfy(&[]));
}

#[test]
fn pipeline_longer_than_request_fails() {
    let p = Pipeline::new(&[PipelineOp::RotateFwht, PipelineOp::Gemv]);
    assert!(!p.can_satisfy(&[PipelineOp::RotateFwht]));
}

#[test]
fn pipeline_prefix_mismatch_fails() {
    let p = Pipeline::new(&[PipelineOp::Gemv]);
    assert!(!p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
}

#[test]
fn pipeline_single_op_self_satisfies() {
    let p = Pipeline::new(&[PipelineOp::Gemv]);
    assert!(p.can_satisfy(&[PipelineOp::Gemv]));
    assert!(!p.can_satisfy(&[PipelineOp::RotateFwht]));
}
