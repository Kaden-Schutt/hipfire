// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::tables::KernelRegistry;
use crate::types::*;

pub fn populate(registry: &mut KernelRegistry) {
    // ── Fused QKV (Q, K, V in one launch) ───────────────────────
    let qkv_variants: &[(KernelKey, ArchPredicate)] = &[
        // HFQ4G256 fused QKV: `gpu.fused_qkv_hfq4g256` is precompiled on every
        // arch that uses the HFQ4 weight path (dispatch.rs `"hfq4"`/`"mq4"`
        // branches — generic wave32 + CDNA wave64 siblings), so the prior
        // `HasWmma` gate was a dead-gate that rejected RDNA1/RDNA2/CDNA even
        // though the kernel runs there. `Always` matches the kernel's true
        // cross-arch availability (mirrors the FusedQkvQ4K row).
        (KernelKey::FusedQkvHfq4G256,     ArchPredicate::Always),
        (KernelKey::FusedQkvMq3G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvMq4G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvHfq6G256,     ArchPredicate::HasDp4a),
        (KernelKey::FusedQkvQ4K,          ArchPredicate::Always),
    ];
    for &(key, arch) in qkv_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
            tile: TileImpl::None,
        });
    }

    // ── Fused QKVZA (Q, K, V + linear attention Z in one launch) ─
    let qkvza_variants: &[(KernelKey, ArchPredicate)] = &[
        // HFQ4G256 fused QKVZA: `gpu.fused_qkvza_hfq4g256` is cross-arch
        // precompiled (dp4a for gfx906, wave64 for CDNA, wave32 generic for
        // RDNA1/2/3/4). The prior `HasWmma` gate was a dead-gate that rejected
        // gfx906/gfx1030/gfx1031 even though the kernel runs there. `Always`
        // matches the true cross-arch availability (mirrors FusedQkvHfq4G256
        // and FusedGateUpHfq4G256 rows above).
        (KernelKey::FusedQkvzaHfq4G256,     ArchPredicate::Always),
        (KernelKey::FusedQkvzaMq3G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvzaMq4G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedQkvzaHfq6G256,     ArchPredicate::HasDp4a),
    ];
    for &(key, arch) in qkvza_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
            tile: TileImpl::None,
        });
    }

    // ── Fused Gate+Up (FFN gate & up projections in one launch) ──
    let gate_up_variants: &[(KernelKey, ArchPredicate)] = &[
        // HFQ4G256 fused gate+up: `gpu.fused_gate_up_hfq4g256` (+ _dp4a sibling)
        // is cross-arch precompiled (generic wave32 + CDNA wave64), mirroring the
        // QKV kernel. The prior `HasWmma` gate was a dead-gate that rejected
        // RDNA1/RDNA2/CDNA even though the kernel runs there. `Always` matches
        // the kernel's true cross-arch availability (mirrors FusedQkvHfq4G256
        // and FusedGateUpQ4K rows).
        (KernelKey::FusedGateUpHfq4G256,     ArchPredicate::Always),
        (KernelKey::FusedGateUpMq3G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedGateUpMq4G256Lloyd, ArchPredicate::HasWmma),
        (KernelKey::FusedGateUpHfq6G256,     ArchPredicate::HasDp4a),
        (KernelKey::FusedGateUpQ4K,          ArchPredicate::Always),
        // Paro4G128T fused: generic wave32 kernels, no ISA-specific intrinsics.
        // Previously gated on HasDp4a (has_dot2_f32_f16) which excluded gfx906/gfx1010.
        (KernelKey::FusedGateUpParo4G128T,   ArchPredicate::Always),
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
            tile: TileImpl::None,
        });
    }

    // ── Fused QKVZA Paro4G128T (dp4a) ────────────────────────────
    let qkvza_paro_variants: &[(KernelKey, ArchPredicate)] = &[
        // Paro4G128T QKVZA: generic wave32 kernels.
        (KernelKey::FusedQkvzaParo4G128T, ArchPredicate::Always),
    ];
    for &(key, arch) in qkvza_paro_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
            tile: TileImpl::None,
        });
    }

    // ── Fused QKV Paro4G128T (3-way FullAttn, dp4a) ─────────────
    let qkv_paro_variants: &[(KernelKey, ArchPredicate)] = &[
        // Paro4G128T QKV (3-way): generic wave32 kernels.
        (KernelKey::FusedQkvParo4G128T, ArchPredicate::Always),
    ];
    for &(key, arch) in qkv_paro_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
            tile: TileImpl::None,
        });
    }
}
