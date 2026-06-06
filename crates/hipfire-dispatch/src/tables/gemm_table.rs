// SPDX-License-Identifier: MIT OR Apache-2.0
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
}
