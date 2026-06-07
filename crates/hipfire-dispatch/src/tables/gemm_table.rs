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
    // HFQ4G256 MMQ set (q8_1 activation-quant + integer tile matmul; uses the
    // i8-WMMA intrinsic). EMPIRICALLY VERIFIED correct on the full WMMA fleet
    // (#397 Ship 5, 2026-06-06):
    //   - gfx1100/RDNA3 (k9lin):   NRMSE 0.024% vs WMMA, ~2.1x FASTER at prefill.
    //   - gfx1151/RDNA3.5 (hipx):  correct (i8-WMMA gated to __gfx1151__).
    //   - gfx1201/RDNA4 (hiptrx):  correct via the NEW gemm_hfq4g256_residual_mmq.gfx12.hip
    //     (i8-WMMA gfx12 intrinsic), loaded under a distinct module name to dodge the
    //     cache collision; NRMSE 0.18-0.24% vs CPU oracle, ~2.5x faster than fp16_wave64.
    // Predicate = HasWmma (has_wmma() = RDNA3 + RDNA3.5 + RDNA4) — exactly the WMMA-
    // capable arches where this i8-WMMA kernel has a real body. NOT HasMmq: gfx906
    // has no WMMA (it uses a separate dp4a MMQ kernel), so it must not select this one.
    // (Still dormant until 5.2 wires GemmFamily; the live should_use_mmq path is
    // independent. 5.2 follow-up: MMQ is 2-2.5x WMMA at prefill — promote candidate.)
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256MmqSet,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
}
