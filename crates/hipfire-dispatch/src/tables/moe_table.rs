// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::tables::KernelRegistry;
use crate::types::*;

/// Register all MoE kernel variants into the registry.
///
/// Covers 4 kernel keys: MoeIndexedGateUpLloyd, MoeIndexedDownLloyd,
/// MoeGroupedGemm, and MoeGroupedI8.
///
/// # Prefill coverage (Ship 4.2)
///
/// The batched MoE prefill path (`MoeFamily::run_prefill`) dispatches by
/// **dtype/env internally**, not through `KernelRegistry::resolve()`. The
/// grouped keys below carry `shape_gate: BatchGt(1)` as **documentation**
/// (the prefill executor never calls `resolve()`). Per-dtype grouped-GEMM
/// coverage — MQ4 → HFQ4-layout WMMA, MQ6 → HFQ6-layout WMMA, ParoQ4G128 →
/// HFQ4G128 WMMA (with gfx1151 i8/k8 MMQ levers) — is exercised by the
/// `MoePrefillResolution` GPU-free tests in `tests.rs`.
pub fn populate(registry: &mut KernelRegistry) {
    registry.register(KernelVariant {
        key: KernelKey::MoeIndexedGateUpLloyd,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });

    registry.register(KernelVariant {
        key: KernelKey::MoeIndexedDownLloyd,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });

    registry.register(KernelVariant {
        key: KernelKey::MoeGroupedGemm,
        arch_required: ArchPredicate::HasWmma,
        // Ship 4.2: BatchGt(1) is documentation-only. The prefill executor
        // (`run_moe_prefill`) dispatches by dtype internally and never calls
        // resolve() on this key.
        shape_gate: Some(ShapePredicate::BatchGt(1)),
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });

    registry.register(KernelVariant {
        key: KernelKey::MoeGroupedI8,
        arch_required: ArchPredicate::HasWmma,
        // Ship 4.2: same documentation-only pattern as MoeGroupedGemm above.
        shape_gate: Some(ShapePredicate::BatchGt(1)),
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
}
