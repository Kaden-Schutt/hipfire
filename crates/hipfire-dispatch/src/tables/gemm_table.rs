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
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0BatchedChunked,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0Wmma,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0Wmma4W,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256,
        arch_required: ArchPredicate::HasDp4a,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G128,
        arch_required: ArchPredicate::HasDp4a,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256Wmma,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    // Batched lm_head wrappers — ArchPredicate::Always: the rdna-compute
    // `gemm_hfqXg256_batched_lmhead` methods self-select WMMA-residual on
    // gfx11/gfx12, dp4a-residual on gfx906 (HFQ6), and a per-row GEMV
    // fallback on every other arch. Gating these on HasWmmaW32 would make
    // resolve() error on the legitimate scalar-fallback archs and break the
    // DFlash lm_head call. The kernel is callable everywhere, so Always.
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256BatchedLmhead,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq3G256BatchedLmhead,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq6G256BatchedLmhead,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmF16XF16Wmma,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
}
