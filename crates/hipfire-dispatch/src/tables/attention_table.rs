// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::tables::KernelRegistry;
use crate::types::*;

pub fn populate(registry: &mut KernelRegistry) {
    // ── KV Cache Write — single-token (decode + per-token fallback) ──
    let kv_write_variants: &[(KernelKey, ArchPredicate, Option<ShapePredicate>)] = &[
        (KernelKey::KvWriteAsym4,     ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteAsym4Fwht, ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteAsym3,     ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteAsym3Fwht, ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteAsym2,     ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteAsym2Fwht, ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteQ8_0,      ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteF32,       ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
    ];
    for (key, arch, shape) in kv_write_variants {
        registry.register(KernelVariant {
            key: *key,
            arch_required: *arch,
            shape_gate: shape.clone(),
            steps: &[PipelineOp::Attend],
            has_awq: false,
        });
    }

    // ── KV Cache Write — batched prefill ───────────────────────
    let kv_write_batched: &[(KernelKey, ArchPredicate, Option<ShapePredicate>)] = &[
        (KernelKey::KvWriteAsym4Batched,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::KvWriteAsym4FwhtBatched, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::KvWriteAsym3Batched,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::KvWriteAsym3FwhtBatched, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::KvWriteAsym2Batched,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::KvWriteAsym2FwhtBatched, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::KvWriteQ8_0Batched,      ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
    ];
    for (key, arch, shape) in kv_write_batched {
        registry.register(KernelVariant {
            key: *key,
            arch_required: *arch,
            shape_gate: shape.clone(),
            steps: &[PipelineOp::Attend],
            has_awq: false,
        });
    }

    // ── Flash Attention — single-token (decode + per-token fallback) ──
    let attn_variants: &[(KernelKey, ArchPredicate, Option<ShapePredicate>)] = &[
        (KernelKey::AttnFlashAsym4,     ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnFlashAsym4Fwht, ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnFlashAsym3,     ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnFlashAsym3Fwht, ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnFlashAsym2,     ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnFlashAsym2Fwht, ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnFlashQ8_0,      ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnQ8_0Kv,         ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnGqaFused,       ArchPredicate::HasWmma, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnF32,            ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
    ];
    for (key, arch, shape) in attn_variants {
        registry.register(KernelVariant {
            key: *key,
            arch_required: *arch,
            shape_gate: shape.clone(),
            steps: &[PipelineOp::Attend],
            has_awq: false,
        });
    }

    // ── Flash Attention — batched prefill / tree-verify ────────
    let attn_batched: &[(KernelKey, ArchPredicate, Option<ShapePredicate>)] = &[
        (KernelKey::AttnFlashAsym4BatchedMasked,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym4FwhtBatchedMasked, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym3BatchedMasked,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym3FwhtBatchedMasked, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        // 2-bit tiers: _batched only (no _masked — tree-verify gap, 3.3)
        (KernelKey::AttnFlashAsym2Batched,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym2FwhtBatched, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        // Q8_0: P-1 no-LDS-cap tiled kernel (replaces old per-position fallback >15k)
        (KernelKey::AttnQ8_0KvBatchedMasked,   ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
    ];
    for (key, arch, shape) in attn_batched {
        registry.register(KernelVariant {
            key: *key,
            arch_required: *arch,
            shape_gate: shape.clone(),
            steps: &[PipelineOp::Attend],
            has_awq: false,
        });
    }
}
