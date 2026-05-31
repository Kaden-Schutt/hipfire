// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::tables::KernelRegistry;
use crate::types::*;

/// Register all MoE kernel variants into the registry.
///
/// Covers all 4 kernel keys: MoeIndexedGateUpLloyd, MoeIndexedDownLloyd,
/// MoeGroupedGemm, and MoeGroupedI8.
pub fn populate(registry: &KernelRegistry) {
    registry.register(KernelVariant {
        key: KernelKey::MoeIndexedGateUpLloyd,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });

    registry.register(KernelVariant {
        key: KernelKey::MoeIndexedDownLloyd,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });

    registry.register(KernelVariant {
        key: KernelKey::MoeGroupedGemm,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });

    registry.register(KernelVariant {
        key: KernelKey::MoeGroupedI8,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
    });
}
