// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::tables::KernelRegistry;
use crate::types::*;

/// Register all GEMM kernel variants into the registry.
///
/// Covers plain batched GEMM for all supported quant formats.
/// Each entry pairs a KernelKey with the arch predicate that must
/// be satisfied.
pub fn populate(registry: &mut KernelRegistry) {
    registry.register(KernelVariant {
        key: KernelKey::GemmF32RegisterTiled,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0BatchedChunked,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0Wmma,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0Wmma4W,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256,
        // HFQ4G256 batched GEMM: cross-arch (dp4a for gfx906, wave64 for CDNA,
        // generic for RDNA). Previously gated on HasDp4a (=has_dot2_f32_f16=RDNA1.1+)
        // which excluded gfx906 where the kernel works via v_dot4_i32_i8.
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G128,
        // HFQ4G128 batched GEMM: same cross-arch rationale as HFQ4G256 above.
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256Wmma,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmF16XF16Wmma,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });

    // ── #397 Ship 5.1: plain-GEMM catalog ────────────────
    // All entries below take the canonical plain signature
    // `(a, x, y, m, k, batch_size)` (verified against rdna-compute/src/gemm.rs)
    // and dispatch through GemmFamily::run. Predicates are the narrowest
    // correct ArchPredicate for each kernel's ISA requirements.

    // F16 generic (scalar/tiled) — no WMMA, runs on every arch.
    registry.register(KernelVariant {
        key: KernelKey::GemmF16,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmF16Tiled,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // F16 WMMA MB4/MB8 fused-transpose, both wave32-WMMA. MB4 has NO gfx12
    // source sibling (gemm_f16_wmma_mb4.hip only) → gfx11-family gate
    // (HasWmmaW32 = RDNA3 + RDNA3.5); RDNA4 falls through to a non-WMMA entry.
    // MB8 DOES have gemm_f16_wmma_mb8.gfx12.hip → HasWmma (admits RDNA4).
    registry.register(KernelVariant {
        key: KernelKey::GemmF16WmmaMb4,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmF16WmmaMb8,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // F32 batched — generic, runs on every arch.
    registry.register(KernelVariant {
        key: KernelKey::GemmF32Batched,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // Q8_0 WMMA x64 (N%64 layout) — gfx11-family wave32 WMMA only
    // (gemm_q8_0_wmma.hip, no gfx12 sibling) → HasWmmaW32, NOT HasWmma.
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0WmmaX64,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // Q8_0 residual WMMA. The base method auto-routes to the gfx12 sibling on
    // RDNA4, so HasWmma is correct (admits RDNA4). The *_gfx12 key is the
    // direct RDNA4-only entry → HasWmmaGfx12.
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0ResidualWmma,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0ResidualWmmaGfx12,
        arch_required: ArchPredicate::HasWmmaGfx12,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // HFQ4G256 wave64 dp4a (v_dot4_i32_i8). This is AMD "dp4a" proper
    // (gfx906/gfx908), gated by gemv_dp4a_enabled() → HasDp4a.
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256Dp4a,
        arch_required: ArchPredicate::HasDp4a,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // HFQ4G256 MMQ set (q8_1_mmq quantize + MMQ tile kernel). HasMmq widens to
    // gfx906 | RDNA3 | RDNA4 in eval_arch. MMQ (+ mmqscreen) is a valuable path
    // — keep it. TODO(#397 Ship 5.2): empirically validate the gfx12/RDNA4 MMQ
    // codepath on hiptrx (and RDNA3 on k9lin, RDNA3.5 on hipx) once wired;
    // narrow the predicate ONLY if a box empirically fails. Do not disable on
    // uncertainty.
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256MmqSet,
        arch_required: ArchPredicate::HasMmq,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
}
