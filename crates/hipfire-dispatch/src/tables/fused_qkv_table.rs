// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::tables::KernelRegistry;
use crate::types::*;

pub fn populate(registry: &mut KernelRegistry) {
    // ── Fused QKV (Q, K, V in one launch) ───────────────────────
    let qkv_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::FusedQkvHfq4G256,     ArchPredicate::HasWmma),
        (KernelKey::FusedQkvMq3G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvMq4G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvHfq6G256,     ArchPredicate::GemvDp4a),
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
        (KernelKey::FusedQkvzaHfq4G256,     ArchPredicate::HasWmma),
        (KernelKey::FusedQkvzaMq3G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvzaMq4G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvzaHfq6G256,     ArchPredicate::GemvDp4a),
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
        (KernelKey::FusedGateUpHfq4G256,     ArchPredicate::HasWmma),
        (KernelKey::FusedGateUpMq3G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedGateUpMq4G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedGateUpHfq6G256,     ArchPredicate::GemvDp4a),
        (KernelKey::FusedGateUpQ4K,          ArchPredicate::Always),
        (KernelKey::FusedGateUpParo4G128T,   ArchPredicate::HasDp4a),
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

    // ── Fused QKVZA Paro4G128T (dp4a) ────────────────────────────
    let qkvza_paro_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::FusedQkvzaParo4G128T, ArchPredicate::HasDp4a),
    ];
    for &(key, arch) in qkvza_paro_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
        });
    }

    // ── Fused QKV Paro4G128T (3-way FullAttn, dp4a) ─────────────
    let qkv_paro_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::FusedQkvParo4G128T, ArchPredicate::HasDp4a),
    ];
    for &(key, arch) in qkv_paro_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
        });
    }
}