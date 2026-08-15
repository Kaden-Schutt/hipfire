// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Unit tests for the hipfire-dispatch layer.
//!
//! Tests cover:
//! - `ShapePredicate::eval` — all three variants, boundary values
//! - `ArchPredicate::eval_arch` — key arch identities (RDNA1/2/3)
//! - `KernelRegistry` — register, resolve, arch gating, shape gating, fallback
//! - `KernelKey::for_gemv*` — dtype/variant → key mapping
//! - `dtype_needs_rotation` — MQ family true, HFQ/F32 false
//! - `GemvFamily::resolve` — arch predicate filtering via a real registry
//! - `Pipeline::can_satisfy` — prefix-match semantics

use crate::context::DispatchCtx;
use crate::families::fused_qkv::FusedQkvFamily;
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

/// gfx1100 = RDNA3: has dp4a, WMMA, MMQ.
fn ctx_rdna3() -> DispatchCtx {
    DispatchCtx::for_test("gfx1100")
}

/// gfx1200 = RDNA4: has dp4a, WMMA, no MMQ via gfx11 path.
fn ctx_rdna4() -> DispatchCtx {
    DispatchCtx::for_test("gfx1200")
}

/// gfx906 = Vega 20: wave64, sdot4/dp4a, gemv_dp4a_enabled by default.
fn ctx_gfx906() -> DispatchCtx {
    DispatchCtx::for_test("gfx906")
}

fn always_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    }
}

fn has_wmma_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    }
}

fn dp4a_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::HasDot2F32F16,
        shape_gate: None,
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    }
}

// ── ShapePredicate::eval ──────────────────────────────────────────────────────

#[test]
fn shape_batch_gt_passes_when_strictly_greater() {
    let s = ShapeInfo {
        batch_size: 2,
        ..Default::default()
    };
    assert!(ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_batch_gt_fails_when_equal() {
    let s = ShapeInfo {
        batch_size: 1,
        ..Default::default()
    };
    assert!(!ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_batch_gt_fails_when_less() {
    let s = ShapeInfo {
        batch_size: 0,
        ..Default::default()
    };
    assert!(!ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_head_dim_eq_passes_on_match() {
    let s = ShapeInfo {
        head_dim: 128,
        ..Default::default()
    };
    assert!(ShapePredicate::HeadDimEq(128).eval(&s));
}

#[test]
fn shape_head_dim_eq_fails_on_mismatch() {
    let s = ShapeInfo {
        head_dim: 64,
        ..Default::default()
    };
    assert!(!ShapePredicate::HeadDimEq(128).eval(&s));
}

#[test]
fn shape_m_lt_passes_when_strictly_less() {
    let s = ShapeInfo {
        m: 7,
        ..Default::default()
    };
    assert!(ShapePredicate::MLt(8).eval(&s));
}

#[test]
fn shape_m_lt_fails_when_equal() {
    let s = ShapeInfo {
        m: 8,
        ..Default::default()
    };
    assert!(!ShapePredicate::MLt(8).eval(&s));
}

#[test]
fn shape_m_lt_fails_when_greater() {
    let s = ShapeInfo {
        m: 9,
        ..Default::default()
    };
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
fn arch_has_wmma_requires_rdna3_or_rdna4() {
    assert!(!ArchPredicate::HasWmma.eval_arch(&ctx_rdna1()));
    assert!(!ArchPredicate::HasWmma.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasWmma.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasWmma.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_wave32_admits_all_rdna_excludes_cdna() {
    // W4: HasWave32 = is_wave32() — every RDNA gen, NOT CDNA wave64. Strict
    // superset of HasWmma on RDNA3/4 (so a HasWmma→HasWave32 flip is byte-
    // identical there) and newly admits RDNA1/2.
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna4()));
    assert!(!ArchPredicate::HasWave32.eval_arch(&ctx_gfx906()));
    assert!(!ArchPredicate::HasWave32.eval_arch(&DispatchCtx::for_test("gfx942")));
}

#[test]
fn arch_has_dp4a_requires_rdna1p1_or_newer() {
    assert!(!ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_mmq_on_rdna3_or_rdna4() {
    assert!(!ArchPredicate::HasMmq.eval_arch(&ctx_rdna1()));
    assert!(!ArchPredicate::HasMmq.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasMmq.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasMmq.eval_arch(&ctx_rdna4())); // RDNA4 MQ6/HFQ6
}

#[test]
fn arch_gemv_dp4a_gfx906_only() {
    // HasDp4a (=v_dot4_i32_i8, gfx906-only)
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_gfx906()));
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna2()));
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna3()));
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna4()));
}

#[test]
fn fused_qkv_hfq6_resolves_cross_arch() {
    // Regression: the HFQ6 fused keys (qkv / gate_up / qkvza) used to be gated
    // `HasDp4a` (gfx906-only), which dead-gated the AWQ A3B trunk's HFQ6-promoted
    // layers on RDNA3/RDNA4 → batched-prefill panic "no implementation for
    // FusedQkvzaHfq6G256". Their batched run-arms call gemm_{qkv,gate_up,qkvza}_
    // hfq6g256, which carry the full cross-arch ladder, so they are now `Always`.
    let fam = FusedQkvFamily::new();
    for ctx in [&ctx_gfx906(), &ctx_rdna3()] {
        assert!(fam
            .resolve(KernelKey::FusedQkvzaHfq6G256, ctx, None)
            .is_ok());
        assert!(fam.resolve(KernelKey::FusedQkvHfq6G256, ctx, None).is_ok());
        assert!(fam
            .resolve(KernelKey::FusedGateUpHfq6G256, ctx, None)
            .is_ok());
    }
}

#[test]
fn fused_qkv_variant_for_key_classifies_by_family() {
    use KernelKey::*;
    // 3-way QKV family (incl. Q4K + Paro QKV synthesis).
    for k in [
        FusedQkvHfq4G256,
        FusedQkvMq3G256Lloyd,
        FusedQkvMq4G256Lloyd,
        FusedQkvHfq6G256,
        FusedQkvQ4K,
        FusedQkvParo4G128T,
    ] {
        assert_eq!(
            fused_qkv_variant_for_key(k),
            Some(FusedQkvVariant::Qkv),
            "{k:?}"
        );
    }
    // 4-way QKVZA family (incl. Paro).
    for k in [
        FusedQkvzaHfq4G256,
        FusedQkvzaMq3G256Lloyd,
        FusedQkvzaMq4G256Lloyd,
        FusedQkvzaHfq6G256,
        FusedQkvzaParo4G128T,
    ] {
        assert_eq!(
            fused_qkv_variant_for_key(k),
            Some(FusedQkvVariant::Qkvza),
            "{k:?}"
        );
    }
    // 2-way Gate+Up family (incl. Q8_0 + Q4K + Paro).
    for k in [
        FusedGateUpHfq4G256,
        FusedGateUpMq3G256Lloyd,
        FusedGateUpMq4G256Lloyd,
        FusedGateUpHfq6G256,
        FusedGateUpQ4K,
        FusedGateUpQ8_0,
        FusedGateUpParo4G128T,
    ] {
        assert_eq!(
            fused_qkv_variant_for_key(k),
            Some(FusedQkvVariant::GateUp),
            "{k:?}"
        );
    }
    // Non-fused keys → None.
    assert_eq!(fused_qkv_variant_for_key(KernelKey::GemvF32), None);
    assert_eq!(fused_qkv_variant_for_key(KernelKey::GemmHfq4G256), None);
}

// ── KernelRegistry ────────────────────────────────────────────────────────────

#[test]
fn registry_resolve_happy_path() {
    let mut reg = KernelRegistry::new();
    reg.register(always_variant(KernelKey::GemvF32));
    let ctx = ctx_rdna1();
    assert_eq!(
        reg.resolve(KernelKey::GemvF32, &ctx, None).unwrap().key,
        KernelKey::GemvF32
    );
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
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna1(); // no WMMA
    let err = reg
        .resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None)
        .unwrap_err();
    assert!(matches!(err, DispatchError::MissingImpl { .. }));
}

#[test]
fn registry_resolve_arch_gate_passes_on_capable_arch() {
    let mut reg = KernelRegistry::new();
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna3(); // has WMMA w32
    assert_eq!(
        reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None)
            .unwrap()
            .key,
        KernelKey::GemmHfq4G256Wmma,
    );
}

#[test]
fn registry_resolve_falls_through_to_second_variant() {
    // Register WMMA variant first, then fallback Always variant for same key.
    // On RDNA1 (no WMMA), the WMMA entry is skipped and fallback is selected.
    let mut reg = KernelRegistry::new();
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
    reg.register(always_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna1();
    assert_eq!(
        reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None)
            .unwrap()
            .key,
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
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    let shape = ShapeInfo {
        head_dim: 128,
        ..Default::default()
    };
    assert_eq!(
        reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape))
            .unwrap()
            .key,
        KernelKey::AttnF32
    );
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
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    let shape = ShapeInfo {
        head_dim: 64,
        ..Default::default()
    };
    let err = reg
        .resolve(KernelKey::AttnF32, &ctx, Some(&shape))
        .unwrap_err();
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
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    assert_eq!(
        reg.resolve(KernelKey::AttnF32, &ctx, None).unwrap().key,
        KernelKey::AttnF32
    );
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
        tile: TileImpl::None,
    });
    reg.register(always_variant(KernelKey::AttnF32)); // ungated fallback
    let ctx = ctx_rdna1();
    let shape = ShapeInfo {
        head_dim: 64,
        ..Default::default()
    }; // doesn't match gated
    assert_eq!(
        reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape))
            .unwrap()
            .key,
        KernelKey::AttnF32
    );
}

#[test]
fn resolve_honors_shape_gate() {
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::GemvF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::BatchGt(1)),
        steps: &[PipelineOp::Gemv],
        has_awq: true,
        tile: TileImpl::None,
    });
    reg.register(KernelVariant {
        key: KernelKey::GemvF32,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    let batched = ShapeInfo {
        batch_size: 8,
        head_dim: 0,
        m: 4096,
        is_tree: false,
    };
    let scalar = ShapeInfo {
        batch_size: 1,
        head_dim: 0,
        m: 4096,
        is_tree: false,
    };
    assert!(
        reg.resolve(KernelKey::GemvF32, &ctx, Some(&batched))
            .unwrap()
            .has_awq
    );
    assert!(
        !reg.resolve(KernelKey::GemvF32, &ctx, Some(&scalar))
            .unwrap()
            .has_awq
    );
    assert!(reg.resolve(KernelKey::GemvF32, &ctx, None).unwrap().has_awq);
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
        (DType::F32, KernelKey::GemvF32),
        (DType::F16, KernelKey::GemvF16),
        (DType::Q8_0, KernelKey::GemvQ8_0),
        (DType::HFQ4G256, KernelKey::GemvHfq4G256),
        (DType::MQ4G256, KernelKey::GemvMq4G256),
        (DType::MQ3G256, KernelKey::GemvMq3G256),
        (DType::MFP4G32, KernelKey::GemvMfp4G32),
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
        (DType::MQ5G256, KernelKey::GemvMq5G256Prerotated),
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
fn for_gemv_prerotated_falls_back_to_plain_for_non_rotated() {
    // Rotation-free dtypes (RotationPlan::None) have no separate prerotated kernel —
    // their prerotated input is their plain input, so for_gemv_prerotated falls through
    // to for_gemv(Plain). This was changed to support the interpreter's unified
    // Prerotated input path (Ship 2.1) — previously these returned Err, which
    // caused a MissingImpl panic at runtime when the interpreter tried to dispatch
    // a non-rotated dtype through the Prerotated variant.
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::F32).unwrap(),
        KernelKey::GemvF32
    );
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::Q8_0).unwrap(),
        KernelKey::GemvQ8_0
    );
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::HFQ4G256).unwrap(),
        KernelKey::GemvHfq4G256
    );
    // Rotation-needing dtypes still resolve to dedicated prerotated keys.
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::MQ4G256).unwrap(),
        KernelKey::GemvMq4G256Prerotated
    );
}

#[test]
fn for_gemv_residual_maps_hfq_and_mq() {
    let cases = [
        (DType::HFQ4G256, KernelKey::GemvHfq4G256Residual),
        (DType::HFQ3G256, KernelKey::GemvHfq3G256Residual),
        (DType::HFQ6G256, KernelKey::GemvHfq6G256Residual),
        (DType::MQ4G256, KernelKey::GemvMq4G256Residual),
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
    // Residual for F32 has no kernel.
    assert!(KernelKey::for_gemv_residual(DType::F32).is_err());
    // Prerotated for F32 now falls back to the plain key (rotation-free dtype).
    // See for_gemv_prerotated_falls_back_to_plain_for_non_rotated.
    assert!(KernelKey::for_gemv_prerotated(DType::F32).is_ok());
}

// ── dtype_needs_rotation ──────────────────────────────────────────────────────────

#[test]
fn dtype_needs_rotation_true_for_mq_family() {
    for dtype in [
        DType::MQ4G256,
        DType::MQ3G256,
        DType::MQ2G256,
        DType::MQ5G256,
        DType::MQ6G256,
        DType::MQ8G256,
        DType::MQ4G256Lloyd,
        DType::MFP4G32,
    ] {
        assert!(dtype_needs_rotation(dtype), "{dtype:?} should need FWHT");
    }
}

#[test]
fn dtype_needs_rotation_false_for_hfq_and_scalar() {
    for dtype in [
        DType::F32,
        DType::F16,
        DType::HFQ4G256,
        DType::Q8_0,
        DType::HFP4G32,
    ] {
        assert!(
            !dtype_needs_rotation(dtype),
            "{dtype:?} should NOT need FWHT"
        );
    }
}

#[test]
fn gemv_steps_rotation_matches_plan() {
    for dtype in [
        DType::MQ4G256,
        DType::MFP4G32,
        DType::ParoQ4G128,
        DType::HFQ4G256,
    ] {
        let steps = KernelKey::gemv_steps(dtype, GemvVariant::Plain);
        let plan = dtype_rotation_plan(dtype);
        let has_fwht = steps.contains(&PipelineOp::RotateFwht);
        let has_givens = steps.contains(&PipelineOp::GivensRotate);
        match plan {
            RotationPlan::Givens => {
                assert!(
                    has_givens && !has_fwht,
                    "{dtype:?}: Givens plan must emit GivensRotate, not FWHT"
                );
            }
            RotationPlan::FwhtG256 | RotationPlan::FwhtG128 => {
                assert!(
                    has_fwht && !has_givens,
                    "{dtype:?}: FWHT plan must emit RotateFwht"
                );
            }
            RotationPlan::None => {
                assert!(!has_fwht && !has_givens, "{dtype:?}: no rotation");
            }
            RotationPlan::Mq8Internal => {}
        }
    }
}

// ── GemvFamily::resolve via populated table ───────────────────────────────────

#[test]
fn gemv_family_resolves_f32_on_all_archs() {
    let fam = GemvFamily::new();
    assert!(fam
        .resolve(DType::F32, GemvVariant::Plain, false, &ctx_rdna1(), None)
        .is_ok());
    assert!(fam
        .resolve(DType::F32, GemvVariant::Plain, false, &ctx_rdna3(), None)
        .is_ok());
}

#[test]
fn gemv_family_resolves_hfq4_on_all_archs() {
    let fam = GemvFamily::new();
    // HFQ4G256 uses generic wave32/wave64 kernels with a fallback for every arch
    // (gfx906 via dp4a/sdot4, gfx1010 via generic). Previously gated on HasDp4a
    // (has_dot2_f32_f16 = RDNA1.1+) which excluded gfx906/gfx1010.
    assert!(fam
        .resolve(
            DType::HFQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna1(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::HFQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::HFQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna3(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna1(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
}

#[test]
fn gemv_family_resolves_mq3_prerotated_on_all_wave32_archs_not_cdna() {
    // W4: MQ3G256's GEMV is a WMMA-free [32,1,1] wave32 scalar kernel. Its arch
    // gate is now HasWave32 (was HasWmma), so it resolves on every RDNA gen
    // (RDNA1/2/3/4) but still NOT on CDNA wave64 (a [32,1,1] kernel needs wave32).
    let fam = GemvFamily::new();
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna1(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna3(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna4(),
            None
        )
        .is_ok());
    // CDNA wave64 (gfx906) still excluded by HasWave32.
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_gfx906(),
            None
        )
        .is_err());
    assert!(fam
        .resolve(
            DType::MQ4G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
    // F32 Prerotated now falls back to GemvF32 (rotation-free dtype → plain key).
    // It resolves on any arch because GemvF32 has no arch gate.
    assert!(fam
        .resolve(
            DType::F32,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna3(),
            None
        )
        .is_ok());
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

// ── MoeResolution eligibility lattice (mirrors qwen35.rs:4598-4671) ──
use crate::families::moe::{MoeDtypes, MoeResolution};

fn dtypes_all_mq4() -> MoeDtypes {
    MoeDtypes {
        router: DType::MQ4G256,
        shared_gate: DType::MQ4G256,
        shared_expert_gate: DType::MQ4G256,
        shared_expert_up: DType::MQ4G256,
        shared_expert_down: DType::MQ4G256,
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ4G256,
        routed_down: DType::MQ4G256,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    }
}

#[test]
fn moe_res_all_mq4_k8_uses_gpu_topk_and_xrot() {
    let r = MoeResolution::resolve(&dtypes_all_mq4(), 8);
    assert!(r.gate_side_mq4);
    assert!(r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local);
}

#[test]
fn moe_res_q8_router_still_gpu_topk() {
    // The non-obvious coupling: a Q8 router disqualifies the 4-way fused
    // gate-side GEMV (gate_side_mq4=false) but the routed experts are still
    // MQ4, so the device-side top-K + indexed path stays on (use_gpu_topk=true).
    let mut d = dtypes_all_mq4();
    d.router = DType::Q8_0;
    d.experts_all_gate_up_mq4 = true; // experts unchanged
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.gate_side_mq4);
    assert!(r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local); // routed_gate_up_mq4 alone fires x_rot
}

#[test]
fn moe_res_k6_disables_gpu_topk_even_when_indexable() {
    // deepseek-shaped: indexable routed dtype but k != 8 => no GPU fast path
    let r = MoeResolution::resolve(&dtypes_all_mq4(), 6);
    assert!(r.routed_indexable_mq4);
    assert!(!r.use_gpu_topk);
}

#[test]
fn moe_res_mq5_routed_indexable() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ5G256;
    d.routed_down = DType::MQ5G256;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_mq5);
    assert!(!r.routed_indexable_mq4);
    assert!(!r.routed_indexable_mq6);
    assert!(r.use_gpu_topk);
}

/// Per-projection Lloyd mix (gate_up MQ2-Lloyd, down MQ3-Lloyd) must reach the
/// GPU-top-K path. Before this arm existed the pair fell through every
/// `routed_indexable_*` test, so `use_gpu_topk` was false and the layer silently
/// took the CPU-top-K fallback — correct output, so the only symptom was lost
/// throughput. Guard both directions.
#[test]
fn moe_res_mixed_lloyd_routed_indexable() {
    for (gu, dn) in [
        (DType::MQ2G256Lloyd, DType::MQ3G256Lloyd),
        (DType::MQ3G256Lloyd, DType::MQ2G256Lloyd),
    ] {
        let mut d = dtypes_all_mq4();
        d.routed_gate_up = gu;
        d.routed_down = dn;
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.routed_indexable_mixed_lloyd, "{gu:?}/{dn:?}");
        assert!(r.use_gpu_topk, "{gu:?}/{dn:?}");
        assert!(r.routed_indexable(), "{gu:?}/{dn:?}");
        // The uniform arms must NOT claim a mixed pair.
        assert!(!r.routed_indexable_mq2lloyd, "{gu:?}/{dn:?}");
        assert!(!r.routed_indexable_mq3lloyd, "{gu:?}/{dn:?}");
        // gate_up is Lloyd either way -> x must be rotated into the local buffer.
        assert!(r.needs_x_rot_local, "{gu:?}/{dn:?}");
    }
}

/// A Lloyd gate_up paired with a NON-Lloyd down must stay unindexable — there is
/// no MQ4/MQ6 down kernel that self-combines, and `routed_down_self_combines`
/// keys on `routed_down`, so admitting this would skip the shared down-combine
/// and silently drop every expert's contribution.
#[test]
fn moe_res_lloyd_gate_up_with_nonlloyd_down_not_indexable() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ2G256Lloyd;
    d.routed_down = DType::MQ4G256;
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.routed_indexable_mixed_lloyd);
    assert!(!r.routed_indexable());
    assert!(!r.use_gpu_topk);
}

/// MQ2/MQ3-G256-GL routed experts, in every per-projection combination with each
/// other AND with the Lloyd pair (all four are FwhtG256 codebook formats whose
/// down kernels self-combine, so any cross is executable).
///
/// The target SKU is gate_up = MQ2-GL + down = MQ3-GL. If the GL dtypes were
/// missing from `routed_indexable_mixed_lloyd`, `use_gpu_topk` would be false
/// and the layer would silently take the CPU-top-K fallback — CORRECT output,
/// so the only symptom is lost throughput. That is exactly the failure this
/// pins.
#[test]
fn moe_res_gl_routed_indexable() {
    use DType::{MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL};
    let codebook = [MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL];
    for gu in codebook {
        for dn in codebook {
            let mut d = dtypes_all_mq4();
            d.routed_gate_up = gu;
            d.routed_down = dn;
            let r = MoeResolution::resolve(&d, 8);
            assert!(r.routed_indexable_mixed_lloyd, "{gu:?}/{dn:?}");
            assert!(r.routed_indexable(), "{gu:?}/{dn:?}");
            assert!(r.use_gpu_topk, "{gu:?}/{dn:?}");
            // use_gpu_topk implies x_rot_local must exist — run_moe_decode
            // `.expect()`s it, and every codebook gate_up kernel reads x_rot.
            assert!(r.needs_x_rot_local, "{gu:?}/{dn:?}");
        }
    }
}

/// The GL dtypes must not leak into the MQ4/MQ5/MQ6 uniform arms, and a GL
/// gate_up with a non-self-combining down must stay unindexable (same
/// double-count / drop-out hazard as the Lloyd case above).
#[test]
fn moe_res_gl_gate_up_with_nonlloyd_down_not_indexable() {
    for dn in [DType::MQ4G256, DType::MQ6G256, DType::Q8_0] {
        let mut d = dtypes_all_mq4();
        d.routed_gate_up = DType::MQ2G256GL;
        d.routed_down = dn;
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.routed_indexable_mixed_lloyd, "{dn:?}");
        assert!(!r.routed_indexable_mq4, "{dn:?}");
        assert!(!r.routed_indexable_mq6, "{dn:?}");
        assert!(!r.routed_indexable(), "{dn:?}");
        assert!(!r.use_gpu_topk, "{dn:?}");
    }
}

/// The GL formats are FWHT-256 rotated (the encoder runs `cpu_fwht_256` per
/// 256-block before quantizing), so dispatch must classify them as FwhtG256 —
/// an un-rotated activation into a rotated weight is silent garbage.
#[test]
fn gl_dtypes_are_fwht_g256() {
    use crate::types::{dtype_needs_rotation, dtype_rotation_plan, RotationPlan};
    for dt in [DType::MQ2G256GL, DType::MQ3G256GL] {
        assert_eq!(dtype_rotation_plan(dt), RotationPlan::FwhtG256, "{dt:?}");
        assert!(dtype_needs_rotation(dt), "{dt:?}");
    }
}

/// GL has MoE-indexed kernels ONLY — no dense plain/prerotated GEMV exists. Both
/// key lookups must return a clean `UnsupportedVariant` rather than resolving to
/// some other dtype's kernel. (Note `for_gemv_prerotated`'s rotation-free
/// fallthrough must NOT catch these: they are FwhtG256, not None.)
#[test]
fn gl_dtypes_have_no_dense_gemv_key() {
    use crate::types::{GemvVariant, KernelKey};
    for dt in [DType::MQ2G256GL, DType::MQ3G256GL] {
        assert!(
            KernelKey::for_gemv(dt, GemvVariant::Plain, false).is_err(),
            "{dt:?} must not resolve a plain GEMV key"
        );
        assert!(
            KernelKey::for_gemv_prerotated(dt).is_err(),
            "{dt:?} must not resolve a prerotated GEMV key"
        );
    }
}

#[test]
fn moe_res_mq6_routed_indexable() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ6G256;
    d.routed_down = DType::MQ6G256;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_mq6);
    assert!(!r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
}

#[test]
fn moe_decode_oplist_prefix_matches_gate_side() {
    // The 4-way fused gate-side projection is capturable as a length-1 prefix.
    let oplist = [
        PipelineOp::MoeGateSideProj,
        PipelineOp::Softmax,
        PipelineOp::TopKRenorm,
        PipelineOp::SharedExpertDown,
        PipelineOp::IndexedGateUp,
        PipelineOp::SiluMulRotate,
        PipelineOp::IndexedDownExpanded,
        PipelineOp::MoeCombine,
    ];
    let fused = Pipeline::new(&[PipelineOp::MoeGateSideProj]);
    assert!(fused.can_satisfy(&oplist));
    let too_long = Pipeline::new(&[PipelineOp::MoeGateSideProj, PipelineOp::TopKRenorm]);
    assert!(!too_long.can_satisfy(&oplist)); // second op mismatches Softmax
}

#[test]
fn moe_res_paro_needs_sidecar() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::ParoQ4G128;
    d.routed_down = DType::ParoQ4G128;
    d.has_paro_shared = false;
    assert!(!MoeResolution::resolve(&d, 8).routed_indexable_paro);
    d.has_paro_shared = true;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_paro);
    assert!(r.use_gpu_topk);
}

// ── op-list interpreter: match_prefix (pure logic) ──────────────────────────

use crate::families::gemv::WeightRef;
use crate::pipeline::steps::{match_prefix, GemvInput};
use crate::pipeline::{FusedPattern, Step};

fn dummy_wr<'a>(t: &'a rdna_compute::GpuTensor) -> WeightRef<'a> {
    WeightRef {
        buf: t,
        dtype: rdna_compute::DType::F32,
        m: 1,
        k: 1,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    }
}

fn gemv_step<'a>(t: &'a rdna_compute::GpuTensor, wr: &'a WeightRef<'a>) -> Step<'a> {
    Step::Gemv {
        w: wr,
        input: GemvInput::Raw(t),
        out: t,
    }
}

#[test]
fn match_prefix_empty_table_never_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    assert_eq!(match_prefix(&[], &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_picks_longest() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    let table = [
        FusedPattern {
            ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
            key: KernelKey::GemvF32,
            guard: |_, _| true,
        },
        FusedPattern {
            ops: &[PipelineOp::Gemv, PipelineOp::Gemv, PipelineOp::Gemv],
            key: KernelKey::GemvF16,
            guard: |_, _| true,
        },
    ];
    assert_eq!(
        match_prefix(&table, &steps, &ctx_rdna3()),
        Some((KernelKey::GemvF16, 3))
    );
}

#[test]
fn match_prefix_no_pattern_longer_than_steps() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |_, _| true,
    }];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_single_op_consumes_one() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |_, _| true,
    }];
    // a len-1 pattern matches the first step, consuming exactly 1
    assert_eq!(
        match_prefix(&table, &steps, &ctx_rdna3()),
        Some((KernelKey::GemvF32, 1))
    );
}

#[test]
fn match_prefix_guard_false_blocks_match() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr), gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |_, _| false, // always reject
    }];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_guard_receives_correct_window() {
    // Guard inspects window length — verifies it gets exactly ops.len() steps.
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |window, _| window.len() == 2, // must see exactly 2 steps
    }];
    assert_eq!(
        match_prefix(&table, &steps, &ctx_rdna3()),
        Some((KernelKey::GemvF32, 2))
    );
}

// ── FUSED_TABLE guard tests ──────────────────────────────────────────────────

use crate::pipeline::steps::{
    guard_gate_up_hfq4g256, guard_gate_up_hfq6g256, guard_gate_up_mq3g256lloyd,
    guard_gate_up_mq4g256lloyd, guard_qkv_hfq4g256, guard_qkv_hfq6g256, guard_qkv_mq3g256lloyd,
    guard_qkv_mq4g256lloyd,
};

fn make_qkv3_steps<'a>(
    dummy: &'a rdna_compute::GpuTensor,
    wr: &'a WeightRef<'a>,
    rotation: RotationPlan,
) -> Vec<Step<'a>> {
    vec![
        Step::RmsnormAutomatic {
            x: dummy,
            norm_weight: dummy,
            x_plain: dummy,
            out: dummy,
            awq_scale: None,
            k: 4096,
            eps: 1e-6,
            rotation,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
    ]
}

fn make_gate_up2_steps<'a>(
    dummy: &'a rdna_compute::GpuTensor,
    wr: &'a WeightRef<'a>,
    rotation: RotationPlan,
) -> Vec<Step<'a>> {
    vec![
        Step::RmsnormAutomatic {
            x: dummy,
            norm_weight: dummy,
            x_plain: dummy,
            out: dummy,
            awq_scale: None,
            k: 4096,
            eps: 1e-6,
            rotation,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
    ]
}

#[test]
fn guard_qkv_mq4g256lloyd_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_wrong_dtype() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::HFQ4G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::None);
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_awq_scale() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: Some(&dummy),
    }; // AWQ present → reject
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_force_unfused() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    let mut ctx = ctx_rdna3();
    std::sync::Arc::make_mut(&mut ctx.flags).force_unfused = true;
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx));
}

#[test]
fn guard_qkv_hfq4g256_covers_mq4g256() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_qkv_hfq4g256(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_hfq4g256_covers_hfq4g256() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::HFQ4G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::None);
    assert!(guard_qkv_hfq4g256(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_hfq6g256_dp4a_decoupled() {
    // Post-merge: 458's commit f478d9b6 ("MQ6 dp4a-decouple") removed the
    // `dp4a_eligible` gate from `guard_qkv_hfq6g256` — HFQ6/MQ6 QKV fusion is
    // safe + beneficial on RDNA3+ even without dp4a (the None arm falls back to
    // gemm n=1). So RDNA3 (gfx1100) now FIRES the guard where master had it
    // blocked. Only `force_unfused` or a non-uniform / wrong-length step window
    // suppresses it now.
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr_hfq6 = WeightRef {
        buf: &dummy,
        dtype: DType::HFQ6G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let wr_mq6 = WeightRef {
        buf: &dummy,
        dtype: DType::MQ6G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps_hfq6 = make_qkv3_steps(&dummy, &wr_hfq6, RotationPlan::FwhtG256);
    let steps_mq6 = make_qkv3_steps(&dummy, &wr_mq6, RotationPlan::FwhtG256);

    // gfx906 (dp4a) still fires.
    assert!(guard_qkv_hfq6g256(&steps_hfq6, &ctx_gfx906()));
    assert!(guard_qkv_hfq6g256(&steps_mq6, &ctx_gfx906()));
    // RDNA3 (gfx1100) now ALSO fires (dp4a decoupled).
    assert!(guard_qkv_hfq6g256(&steps_hfq6, &ctx_rdna3()));
    assert!(guard_qkv_hfq6g256(&steps_mq6, &ctx_rdna3()));
    // RDNA1 (gfx1010) fires too — the guard no longer gates on dp4a; the
    // fused_qkv None arm safely lowers to gemm n=1 on any arch.
    assert!(guard_qkv_hfq6g256(&steps_hfq6, &ctx_rdna1()));
}

#[test]
fn guard_qkv_rejects_mixed_gemv_input() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps = vec![
        Step::RmsnormAutomatic {
            x: &dummy,
            norm_weight: &dummy,
            x_plain: &dummy,
            out: &dummy,
            awq_scale: None,
            k: 4096,
            eps: 1e-6,
            rotation: RotationPlan::FwhtG256,
        },
        Step::Gemv {
            w: &wr,
            input: GemvInput::Prerotated(&dummy),
            out: &dummy,
        },
        Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&dummy),
            out: &dummy,
        }, // mixed!
        Step::Gemv {
            w: &wr,
            input: GemvInput::Prerotated(&dummy),
            out: &dummy,
        },
    ];
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_gate_up_mq4g256lloyd_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    let steps = make_gate_up2_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_gate_up_mq4g256lloyd(&steps, &ctx_rdna3()));
}

// ── MoePrefillResolution cells (Ship 4.2) ─────────────────────────

use crate::families::moe::MoePrefillResolution;

/// Helper: default MoeDtypes for MQ4 routed experts (the common A3B case).
fn moe_dtypes_mq4() -> MoeDtypes {
    MoeDtypes {
        router: DType::Q8_0,
        shared_gate: DType::Q8_0,
        shared_expert_gate: DType::MQ4G256,
        shared_expert_up: DType::MQ4G256,
        shared_expert_down: DType::MQ4G256,
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ4G256,
        routed_down: DType::MQ4G256,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    }
}

fn moe_dtypes_mq6() -> MoeDtypes {
    let mut d = moe_dtypes_mq4();
    d.routed_gate_up = DType::MQ6G256;
    d.routed_down = DType::MQ6G256;
    d.experts_all_gate_up_mq4 = false;
    d
}

fn moe_dtypes_paro() -> MoeDtypes {
    let mut d = moe_dtypes_mq4();
    d.routed_gate_up = DType::ParoQ4G128;
    d.routed_down = DType::ParoQ4G128;
    d.experts_all_gate_up_mq4 = false;
    d.has_paro_shared = true;
    d
}

fn flags_default() -> rdna_compute::feature_flags::FeatureFlags {
    rdna_compute::feature_flags::FeatureFlags::for_test("gfx1100")
}

#[test]
fn moe_prefill_resolution_path2_gfx11_mq4() {
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx11 should have Path 2 (WMMA)");
    assert!(!r.down_path0, "gfx11 should not be Path 0");
    assert!(!r.paro_mode);
    assert!(!r.use_paro_i8);
    assert!(!r.use_paro_i8_k8);
}

#[test]
fn moe_prefill_resolution_path2_gfx12_mq4() {
    let arch = crate::context::DispatchCtx::for_test("gfx1200");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx12 should have Path 2 (WMMA)");
    assert!(!r.down_path0);
}

#[test]
fn moe_prefill_resolution_path2_gfx12_mq6() {
    let arch = crate::context::DispatchCtx::for_test("gfx1200");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx12 should have Path 2 for MQ6");
    assert!(!r.paro_mode);
}

#[test]
fn moe_prefill_resolution_gfx1151_mixed_mq6_fences_mq4_i8() {
    let arch = crate::context::DispatchCtx::for_test("gfx1151");

    let pure_mq4 = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(pure_mq4.use_path2);
    assert!(
        !pure_mq4.force_mq4_grouped_fp16,
        "pure MQ4 layers should keep gfx1151's existing grouped-i8 default"
    );

    let mixed_mq6 = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(mixed_mq6.use_path2);
    assert!(
        mixed_mq6.force_mq4_grouped_fp16,
        "MQ6-promoted/mixed A3B layers must not run remaining MQ4 projections through grouped-i8 by default"
    );
}

#[test]
fn moe_prefill_resolution_path2_gfx11_paro() {
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_paro(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx11 should have Path 2 for Paro");
    assert!(r.paro_mode);
    assert!(!r.use_paro_i8, "gfx1100 is not gfx1151 — no i8");
}

#[test]
fn moe_prefill_resolution_path2_gfx1151_paro_i8() {
    let arch = crate::context::DispatchCtx::for_test("gfx1151");
    let r = MoePrefillResolution::resolve(&moe_dtypes_paro(), &arch.arch, &arch.flags);
    assert!(r.use_path2);
    assert!(r.paro_mode);
    assert!(r.use_paro_i8, "gfx1151 should default to i8 for Paro");
    assert!(r.use_paro_i8_k8, "gfx1151 should default to i8 k8 for Paro");
}

#[test]
fn moe_prefill_resolution_path1_gfx1030() {
    let arch = crate::context::DispatchCtx::for_test("gfx1030");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(!r.use_path2, "gfx1030 has no WMMA — no Path 2");
    assert!(!r.down_path0, "gfx1030 is not gfx9 — no Path 0");
}

#[test]
fn moe_prefill_resolution_path0_gfx906() {
    let arch = crate::context::DispatchCtx::for_test("gfx906");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(!r.use_path2, "gfx906 has no WMMA — no Path 2");
    assert!(r.down_path0, "gfx906 should be Path 0 (atomic GEMV)");
}

#[test]
fn moe_prefill_resolution_path0_gfx942() {
    let arch = crate::context::DispatchCtx::for_test("gfx942");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(!r.use_path2, "gfx942 has no WMMA — no Path 2");
    assert!(r.down_path0, "gfx942 (CDNA3) should be Path 0");
}

#[test]
fn moe_prefill_resolution_grouped_gemm_opt_out() {
    let mut flags = flags_default();
    flags.moe_grouped_gemm = false;
    let flags = std::sync::Arc::new(flags);
    let caps = rdna_compute::arch_caps::ArchCaps::new("gfx1100", flags.clone());
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &caps, &flags);
    assert!(!r.use_path2, "moe_grouped_gemm=0 should disable Path 2");
}

#[test]
fn moe_prefill_resolution_paro_i8_opt_out() {
    let mut flags = flags_default();
    flags.moe_paro_i8 = Some(false);
    let flags = std::sync::Arc::new(flags);
    let caps = rdna_compute::arch_caps::ArchCaps::new("gfx1151", flags.clone());
    let r = MoePrefillResolution::resolve(&moe_dtypes_paro(), &caps, &flags);
    assert!(r.use_path2);
    assert!(r.paro_mode);
    assert!(!r.use_paro_i8, "moe_paro_i8=0 should disable i8");
    assert!(!r.use_paro_i8_k8);
}

#[test]
fn moe_prefill_resolution_mq6_gfx11_uses_path2() {
    // Post-merge: the gfx11 MQ6 grouped-WMMA `_k2` kernel now exists, so the
    // 458 widen (8d555fc6) admits Path 2 on any wmma_w32 gfx11 arch (gfx1100
    // included), superseding master's earlier gfx12-only fallback assertion.
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(
        r.use_path2,
        "MQ6 on gfx11 should use Path 2 (gfx11 `_k2` grouped WMMA exists)"
    );
    assert!(!r.down_path0, "gfx11 is not Path 0");
}

#[test]
fn moe_prefill_resolution_mq6_gfx1151_uses_path2() {
    let arch = crate::context::DispatchCtx::for_test("gfx1151");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(
        r.use_path2,
        "MQ6 on gfx1151 should use Path 2 (grouped WMMA available)"
    );
    assert!(!r.down_path0);
}

#[test]
fn moe_prefill_resolution_mq6_gfx12_uses_path2() {
    let arch = crate::context::DispatchCtx::for_test("gfx1200");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(
        r.use_path2,
        "MQ6 on gfx12 should use Path 2 (grouped WMMA available)"
    );
    assert!(!r.down_path0);
}

#[test]
fn moe_prefill_resolution_mq4_gfx11_still_path2() {
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "MQ4 on gfx11 should still use Path 2");
}

// ── Uniform codebook routed pair (MQ2-Lloyd gate_up / MQ3-Lloyd down) ────────
//
// The shape every current low-bit a3b SKU ships (`--format mq4-mqlloyd-antirez`,
// no k-map ⇒ `expert_dtype_tags == None`). Batched prefill for it is Path 2 only:
// `run_moe_prefill`'s Path 0/1 indexed-GEMV matches have no MQ2/MQ3-Lloyd arm and
// return `UnsupportedVariant`, so these tests pin BOTH that Path 2 is selected on
// every WMMA arch AND that each grouped-GEMM leg names a real entry point.

/// The antirez asymmetric routed pair: gate_up MQ2-Lloyd, down MQ3-Lloyd,
/// MQ4 shared expert, no tag table.
fn moe_dtypes_codebook_pair() -> MoeDtypes {
    let mut d = moe_dtypes_mq4();
    d.routed_gate_up = DType::MQ2G256Lloyd;
    d.routed_down = DType::MQ3G256Lloyd;
    d.experts_all_gate_up_mq4 = false;
    d
}

#[test]
fn moe_prefill_resolution_codebook_pair_uses_path2_on_wmma() {
    for arch_name in [
        "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
    ] {
        let arch = crate::context::DispatchCtx::for_test(arch_name);
        let r = MoePrefillResolution::resolve(&moe_dtypes_codebook_pair(), &arch.arch, &arch.flags);
        assert!(
            r.use_path2,
            "{arch_name}: MQ2L/MQ3L routed pair must take Path 2 (grouped WMMA) — \
             Path 0/1 have no Lloyd indexed-GEMV arm"
        );
        assert!(!r.down_path0, "{arch_name} is not a Path 0 arch");
        assert!(!r.paro_mode);
    }
}

/// Non-WMMA archs fall out of Path 2. There is no Lloyd Path 0/1 arm, which is
/// exactly why `codebook_batched_admit_enabled_from_env` (hipfire-arch-qwen35)
/// refuses to admit these layers on such archs — they must never get here.
#[test]
fn moe_prefill_resolution_codebook_pair_no_path2_without_wmma() {
    for arch_name in ["gfx1010", "gfx1030", "gfx906", "gfx942"] {
        let arch = crate::context::DispatchCtx::for_test(arch_name);
        let r = MoePrefillResolution::resolve(&moe_dtypes_codebook_pair(), &arch.arch, &arch.flags);
        assert!(
            !r.use_path2,
            "{arch_name} has no grouped-WMMA kernel — must not resolve to Path 2"
        );
    }
}

/// Both grouped-GEMM legs of the pair must name a REAL `extern "C" __global__`
/// entry point that exists in the source string the launcher will hand to the
/// JIT — on both arch legs. This is the no-GPU proof that the new dispatch arms
/// resolve to actual kernels rather than to a fallback, a stale name, or a
/// gfx11 kernel wrongly reused on RDNA4.
#[test]
fn codebook_grouped_gemm_legs_name_real_kernels() {
    let legs = [
        (
            "MQ2-Lloyd",
            false,
            rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(false),
        ),
        (
            "MQ2-Lloyd",
            true,
            rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(true),
        ),
        (
            "MQ3-Lloyd",
            false,
            rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(false),
        ),
        (
            "MQ3-Lloyd",
            true,
            rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(true),
        ),
    ];
    for (label, is_gfx12, (name, src)) in legs {
        let entry = format!("__global__ void {name}(");
        assert!(
            src.contains(&entry),
            "{label} is_gfx12={is_gfx12}: source does not define `{name}`"
        );
        // The gfx11 leg must use the gfx11 WMMA builtin and the gfx12 leg the
        // `_gfx12` one — swapping them fails the JIT at first launch, which on
        // a prefill path means a serving outage, not a fallback.
        if is_gfx12 {
            assert!(
                src.contains("wmma_f32_16x16x16_f16_w32_gfx12"),
                "{label} gfx12 leg must use the _gfx12 WMMA builtin"
            );
        } else {
            assert!(
                src.contains("wmma_f32_16x16x16_f16_w32")
                    && !src.contains("wmma_f32_16x16x16_f16_w32_gfx12"),
                "{label} gfx11 leg must use the plain _w32 WMMA builtin"
            );
        }
    }
    // Distinct module names per arch — the JIT cache is keyed by module name
    // only, so a collision makes one arch's source dead code.
    assert_ne!(
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(false).0,
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(true).0
    );
    assert_ne!(
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(false).0,
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(true).0
    );
}

/// The MQ2-Lloyd and MQ3-Lloyd grouped kernels must agree with the uniform
/// grouped kernarg contract the scatter pipeline emits: 5 pointers then
/// `M, K, x_row_div, m_total`. A tenth arg (as in the merged dtype-tag kernel,
/// which inserts `dtype_tags` second) would shift every kernarg by 8 bytes.
#[test]
fn codebook_grouped_gemm_kernarg_contract_is_the_uniform_nine() {
    for (name, src) in [
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(false),
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(true),
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(false),
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(true),
    ] {
        let start = src
            .find(&format!("__global__ void {name}("))
            .unwrap_or_else(|| panic!("{name}: entry point not found"));
        let sig_end = src[start..].find(')').expect("unterminated signature") + start;
        let sig = &src[start..sig_end];
        assert!(
            !sig.contains("dtype_tags"),
            "{name}: uniform grouped kernels must NOT take a dtype_tags table"
        );
        for arg in [
            "expert_weight_ptrs",
            "expert_tile_ids",
            "sorted_slot_index",
            "X_src",
            "Y_grouped",
            "int M",
            "int K",
            "int x_row_div",
            "int m_total",
        ] {
            assert!(sig.contains(arg), "{name}: signature is missing `{arg}`");
        }
    }
}

