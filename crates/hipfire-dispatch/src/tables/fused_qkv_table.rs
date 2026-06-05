// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::tables::KernelRegistry;
use crate::types::*;

pub fn populate(registry: &mut KernelRegistry) {
    // ── Fused QKV (Q, K, V in one launch) ───────────────────────
    let qkv_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::FusedQkvHfq4G256,     ArchPredicate::HasWmmaW32),
        (KernelKey::FusedQkvMq3G256Lloyd, ArchPredicate::HasWmmaW32),
        (KernelKey::FusedQkvMq4G256Lloyd, ArchPredicate::HasWmmaW32),
        (KernelKey::FusedQkvHfq6G256,     ArchPredicate::GemvDp4a),
        (KernelKey::FusedQkvParo4G128T,   ArchPredicate::Always),
        (KernelKey::FusedQkvQ4K,          ArchPredicate::Always),
    ];
    for &(key, arch) in qkv_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
        });
    }

    // ── Fused QKVZA (Q, K, V + linear attention Z in one launch) ─
    let qkvza_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::FusedQkvzaHfq4G256,     ArchPredicate::HasWmmaW32),
        (KernelKey::FusedQkvzaMq3G256Lloyd, ArchPredicate::HasWmmaW32),
        (KernelKey::FusedQkvzaMq4G256Lloyd, ArchPredicate::HasWmmaW32),
        (KernelKey::FusedQkvzaHfq6G256,     ArchPredicate::GemvDp4a),
        (KernelKey::FusedQkvzaParo4G128T,   ArchPredicate::Always),
    ];
    for &(key, arch) in qkvza_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
        });
    }

    // ── Fused Gate+Up (FFN gate & up projections in one launch) ──
    let gate_up_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::FusedGateUpHfq4G256,     ArchPredicate::HasWmmaW32),
        (KernelKey::FusedGateUpMq3G256Lloyd, ArchPredicate::HasWmmaW32),
        (KernelKey::FusedGateUpMq4G256Lloyd, ArchPredicate::HasWmmaW32),
        (KernelKey::FusedGateUpHfq6G256,     ArchPredicate::GemvDp4a),
        (KernelKey::FusedGateUpParo4G128T,   ArchPredicate::Always),
        (KernelKey::FusedGateUpQ4K,          ArchPredicate::Always),
        // Q8_0 gate+up: plain wave32 kernel (`gpu.fused_gate_up_q8_0`),
        // no arch gate — mirrors the Q4K row. Used by qwen2 FFN.
        (KernelKey::FusedGateUpQ8_0,         ArchPredicate::Always),
    ];
    for &(key, arch) in gate_up_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
        });
    }
}

