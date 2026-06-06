// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::tables::KernelRegistry;
use crate::types::*;

pub fn populate(registry: &mut KernelRegistry) {
    // ── KV Cache Write ─────────────────────────────────────────
    let kv_write_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::KvWriteAsym4,    ArchPredicate::Always),
        (KernelKey::KvWriteAsym4Fwht,ArchPredicate::Always),
        (KernelKey::KvWriteAsym3,    ArchPredicate::Always),
        (KernelKey::KvWriteAsym3Fwht,ArchPredicate::Always),
        (KernelKey::KvWriteAsym2,    ArchPredicate::Always),
        (KernelKey::KvWriteAsym2Fwht,ArchPredicate::Always),
        (KernelKey::KvWriteQ8_0,     ArchPredicate::Always),
        (KernelKey::KvWriteF32,      ArchPredicate::Always),
    ];
    for &(key, arch) in kv_write_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
        });
    }

    // ── Flash Attention ────────────────────────────────────────
    let attn_variants: &[(KernelKey, ArchPredicate)] = &[
        (KernelKey::AttnFlashAsym4,     ArchPredicate::Always),
        (KernelKey::AttnFlashAsym4Fwht, ArchPredicate::Always),
        (KernelKey::AttnFlashAsym3,     ArchPredicate::Always),
        (KernelKey::AttnFlashAsym3Fwht, ArchPredicate::Always),
        (KernelKey::AttnFlashAsym2,     ArchPredicate::Always),
        (KernelKey::AttnFlashAsym2Fwht, ArchPredicate::Always),
        (KernelKey::AttnFlashQ8_0,      ArchPredicate::Always),
        (KernelKey::AttnQ8_0Kv,          ArchPredicate::Always),
        (KernelKey::AttnGqaFused,       ArchPredicate::HasWmma),
        (KernelKey::AttnF32,            ArchPredicate::Always),
    ];
    for &(key, arch) in attn_variants {
        registry.register(KernelVariant {
            key,
            arch_required: arch,
            shape_gate: None,
            steps: &[PipelineOp::Gemv],
            has_awq: false,
        });
    }
}

