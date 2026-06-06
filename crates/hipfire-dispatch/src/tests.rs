// SPDX-License-Identifier: MIT OR Apache-2.0
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
        arch_required: ArchPredicate::HasDp4a,
        shape_gate: None,
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
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
fn arch_has_wmma_requires_rdna3_or_rdna4() {
    assert!(!ArchPredicate::HasWmma.eval_arch(&ctx_rdna1()));
    assert!(!ArchPredicate::HasWmma.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasWmma.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasWmma.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_dp4a_requires_rdna1p1_or_newer() {
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_rdna4()));
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
    assert!(ArchPredicate::GemvDp4a.eval_arch(&ctx_gfx906()));
    assert!(!ArchPredicate::GemvDp4a.eval_arch(&ctx_rdna2()));
    assert!(!ArchPredicate::GemvDp4a.eval_arch(&ctx_rdna3()));
    assert!(!ArchPredicate::GemvDp4a.eval_arch(&ctx_rdna4()));
}

#[test]
fn fused_qkv_hfq6_resolved_only_on_gemv_dp4a_arch() {
    let fam = FusedQkvFamily::new();
    // gfx906 (gemv_dp4a enabled) → HFQ6 variants resolve
    assert!(fam.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx_gfx906(), None).is_ok());
    assert!(fam.resolve(KernelKey::FusedQkvHfq6G256, &ctx_gfx906(), None).is_ok());
    assert!(fam.resolve(KernelKey::FusedGateUpHfq6G256, &ctx_gfx906(), None).is_ok());
    // RDNA3 (gemv_dp4a disabled by default) → HFQ6 variants rejected
    assert!(fam.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx_rdna3(), None).is_err());
    assert!(fam.resolve(KernelKey::FusedQkvHfq6G256, &ctx_rdna3(), None).is_err());
    assert!(fam.resolve(KernelKey::FusedGateUpHfq6G256, &ctx_rdna3(), None).is_err());
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
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna1(); // no WMMA
    let err = reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None).unwrap_err();
    assert!(matches!(err, DispatchError::MissingImpl { .. }));
}

#[test]
fn registry_resolve_arch_gate_passes_on_capable_arch() {
    let mut reg = KernelRegistry::new();
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
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
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
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
        tile: TileImpl::None,
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
        tile: TileImpl::None,
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
        tile: TileImpl::None,
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
        tile: TileImpl::None,
    });
    reg.register(always_variant(KernelKey::AttnF32)); // ungated fallback
    let ctx = ctx_rdna1();
    let shape = ShapeInfo { head_dim: 64, ..Default::default() }; // doesn't match gated
    assert_eq!(reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape)).unwrap().key, KernelKey::AttnF32);
}

#[test]
fn resolve_honors_shape_gate() {
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::GemvF32, arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::BatchGt(1)), steps: &[PipelineOp::Gemv], has_awq: true, tile: TileImpl::None,
    });
    reg.register(KernelVariant {
        key: KernelKey::GemvF32, arch_required: ArchPredicate::Always,
        shape_gate: None, steps: &[PipelineOp::Gemv], has_awq: false,
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    let batched = ShapeInfo { batch_size: 8, head_dim: 0, m: 4096, is_tree: false };
    let scalar  = ShapeInfo { batch_size: 1, head_dim: 0, m: 4096, is_tree: false };
    assert!(reg.resolve(KernelKey::GemvF32, &ctx, Some(&batched)).unwrap().has_awq);
    assert!(!reg.resolve(KernelKey::GemvF32, &ctx, Some(&scalar)).unwrap().has_awq);
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
fn for_gemv_prerotated_falls_back_to_plain_for_non_rotated() {
    // Rotation-free dtypes (RotationPlan::None) have no separate prerotated kernel —
    // their prerotated input is their plain input, so for_gemv_prerotated falls through
    // to for_gemv(Plain). This was changed to support the interpreter's unified
    // Prerotated input path (Ship 2.1) — previously these returned Err, which
    // caused a MissingImpl panic at runtime when the interpreter tried to dispatch
    // a non-rotated dtype through the Prerotated variant.
    assert_eq!(KernelKey::for_gemv_prerotated(DType::F32).unwrap(), KernelKey::GemvF32);
    assert_eq!(KernelKey::for_gemv_prerotated(DType::Q8_0).unwrap(), KernelKey::GemvQ8_0);
    assert_eq!(KernelKey::for_gemv_prerotated(DType::HFQ4G256).unwrap(), KernelKey::GemvHfq4G256);
    // Rotation-needing dtypes still resolve to dedicated prerotated keys.
    assert_eq!(KernelKey::for_gemv_prerotated(DType::MQ4G256).unwrap(), KernelKey::GemvMq4G256Prerotated);
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
        DType::MQ4G256, DType::MQ3G256, DType::MQ2G256, DType::MQ6G256,
        DType::MQ8G256, DType::MQ4G256Lloyd, DType::MFP4G32,
    ] {
        assert!(dtype_needs_rotation(dtype), "{dtype:?} should need FWHT");
    }
}

#[test]
fn dtype_needs_rotation_false_for_hfq_and_scalar() {
    for dtype in [DType::F32, DType::F16, DType::HFQ4G256, DType::Q8_0, DType::HFP4G32] {
        assert!(!dtype_needs_rotation(dtype), "{dtype:?} should NOT need FWHT");
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
    // F32 Prerotated now falls back to GemvF32 (rotation-free dtype → plain key).
    // It resolves on any arch because GemvF32 has no arch gate.
    assert!(fam.resolve(DType::F32, GemvVariant::Prerotated, false, &ctx_rdna3(), None).is_ok());
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
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ4G256,
        routed_down: DType::MQ4G256,
        has_paro_shared: false,
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
        PipelineOp::MoeGateSideProj, PipelineOp::Softmax, PipelineOp::TopKRenorm,
        PipelineOp::SharedExpertDown, PipelineOp::IndexedGateUp,
        PipelineOp::SiluMulRotate, PipelineOp::IndexedDownExpanded, PipelineOp::MoeCombine,
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
    WeightRef { buf: t, dtype: rdna_compute::DType::F32, m: 1, k: 1,
                row_stride: 0, rotation: None, awq_scale: None }
}

fn gemv_step<'a>(t: &'a rdna_compute::GpuTensor, wr: &'a WeightRef<'a>) -> Step<'a> {
    Step::Gemv { w: wr, input: GemvInput::Raw(t), out: t }
}

#[test]
fn match_prefix_empty_table_never_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr), gemv_step(&dummy, &wr), gemv_step(&dummy, &wr)];
    assert_eq!(match_prefix(&[], &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_picks_longest() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr), gemv_step(&dummy, &wr), gemv_step(&dummy, &wr)];
    let table = [
        FusedPattern { ops: &[PipelineOp::Gemv, PipelineOp::Gemv], key: KernelKey::GemvF32, guard: |_, _| true },
        FusedPattern { ops: &[PipelineOp::Gemv, PipelineOp::Gemv, PipelineOp::Gemv], key: KernelKey::GemvF16, guard: |_, _| true },
    ];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), Some((KernelKey::GemvF16, 3)));
}

#[test]
fn match_prefix_no_pattern_longer_than_steps() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv], key: KernelKey::GemvF32, guard: |_, _| true,
    }];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_single_op_consumes_one() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr), gemv_step(&dummy, &wr), gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv], key: KernelKey::GemvF32, guard: |_, _| true,
    }];
    // a len-1 pattern matches the first step, consuming exactly 1
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), Some((KernelKey::GemvF32, 1)));
}

#[test]
fn match_prefix_guard_false_blocks_match() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr), gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |_, _| false,  // always reject
    }];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_guard_receives_correct_window() {
    // Guard inspects window length — verifies it gets exactly ops.len() steps.
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr), gemv_step(&dummy, &wr), gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |window, _| window.len() == 2,  // must see exactly 2 steps
    }];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), Some((KernelKey::GemvF32, 2)));
}

// ── FUSED_TABLE guard tests ──────────────────────────────────────────────────

use crate::pipeline::steps::{
    guard_qkv_mq4g256lloyd, guard_qkv_mq3g256lloyd,
    guard_qkv_hfq4g256, guard_qkv_hfq6g256,
    guard_gate_up_mq4g256lloyd, guard_gate_up_mq3g256lloyd,
    guard_gate_up_hfq4g256, guard_gate_up_hfq6g256,
};

fn make_qkv3_steps<'a>(
    dummy: &'a rdna_compute::GpuTensor,
    wr: &'a WeightRef<'a>,
    rotation: RotationPlan,
) -> Vec<Step<'a>> {
    vec![
        Step::RmsnormAutomatic {
            x: dummy, norm_weight: dummy, x_plain: dummy, out: dummy,
            awq_scale: None, k: 4096, eps: 1e-6, rotation,
        },
        Step::Gemv { w: wr, input: GemvInput::Prerotated(dummy), out: dummy },
        Step::Gemv { w: wr, input: GemvInput::Prerotated(dummy), out: dummy },
        Step::Gemv { w: wr, input: GemvInput::Prerotated(dummy), out: dummy },
    ]
}

fn make_gate_up2_steps<'a>(
    dummy: &'a rdna_compute::GpuTensor,
    wr: &'a WeightRef<'a>,
    rotation: RotationPlan,
) -> Vec<Step<'a>> {
    vec![
        Step::RmsnormAutomatic {
            x: dummy, norm_weight: dummy, x_plain: dummy, out: dummy,
            awq_scale: None, k: 4096, eps: 1e-6, rotation,
        },
        Step::Gemv { w: wr, input: GemvInput::Prerotated(dummy), out: dummy },
        Step::Gemv { w: wr, input: GemvInput::Prerotated(dummy), out: dummy },
    ]
}

#[test]
fn guard_qkv_mq4g256lloyd_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::MQ4G256Lloyd,
                         m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_wrong_dtype() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::HFQ4G256,
                         m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::None);
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_awq_scale() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::MQ4G256Lloyd,
                         m: 4096, k: 4096, row_stride: 0, rotation: None,
                         awq_scale: Some(&dummy) }; // AWQ present → reject
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_force_unfused() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::MQ4G256Lloyd,
                         m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    let mut ctx = ctx_rdna3();
    std::sync::Arc::make_mut(&mut ctx.flags).force_unfused = true;
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx));
}

#[test]
fn guard_qkv_hfq4g256_covers_mq4g256() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::MQ4G256,
                         m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_qkv_hfq4g256(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_hfq4g256_covers_hfq4g256() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::HFQ4G256,
                         m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::None);
    assert!(guard_qkv_hfq4g256(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_hfq6g256_dp4a_gated() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr_hfq6 = WeightRef { buf: &dummy, dtype: DType::HFQ6G256,
                               m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let wr_mq6  = WeightRef { buf: &dummy, dtype: DType::MQ6G256,
                               m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps_hfq6 = make_qkv3_steps(&dummy, &wr_hfq6, RotationPlan::FwhtG256);
    let steps_mq6  = make_qkv3_steps(&dummy, &wr_mq6,  RotationPlan::FwhtG256);

    // gfx906 has gemv_dp4a enabled → fires
    assert!(guard_qkv_hfq6g256(&steps_hfq6, &ctx_gfx906()));
    assert!(guard_qkv_hfq6g256(&steps_mq6,  &ctx_gfx906()));
    // RDNA1 (gfx1010) has no dp4a → blocked
    assert!(!guard_qkv_hfq6g256(&steps_hfq6, &ctx_rdna1()));
    // RDNA3 (gfx1100) has no gemv_dp4a → blocked
    assert!(!guard_qkv_hfq6g256(&steps_hfq6, &ctx_rdna3()));
}

#[test]
fn guard_qkv_rejects_mixed_gemv_input() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::MQ4G256Lloyd,
                         m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps = vec![
        Step::RmsnormAutomatic {
            x: &dummy, norm_weight: &dummy, x_plain: &dummy, out: &dummy,
            awq_scale: None, k: 4096, eps: 1e-6, rotation: RotationPlan::FwhtG256,
        },
        Step::Gemv { w: &wr, input: GemvInput::Prerotated(&dummy), out: &dummy },
        Step::Gemv { w: &wr, input: GemvInput::Raw(&dummy), out: &dummy }, // mixed!
        Step::Gemv { w: &wr, input: GemvInput::Prerotated(&dummy), out: &dummy },
    ];
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_gate_up_mq4g256lloyd_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef { buf: &dummy, dtype: DType::MQ4G256Lloyd,
                         m: 4096, k: 4096, row_stride: 0, rotation: None, awq_scale: None };
    let steps = make_gate_up2_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_gate_up_mq4g256lloyd(&steps, &ctx_rdna3()));
}
