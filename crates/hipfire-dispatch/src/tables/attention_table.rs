// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
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
        // Llama legacy quant KV write (decode only)
        (KernelKey::KvWriteHfq4,      ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::KvWriteQ4,        ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
    ];
    for (key, arch, shape) in kv_write_variants {
        registry.register(KernelVariant {
            key: *key,
            arch_required: *arch,
            shape_gate: shape.clone(),
            steps: &[PipelineOp::Attend],
            has_awq: false,
            tile: TileImpl::None,
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
            tile: TileImpl::None,
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
        // Llama legacy quant KV (decode only)
        (KernelKey::AttnHfq4Kv,        ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
        (KernelKey::AttnQ4Kv,          ArchPredicate::Always, Some(ShapePredicate::BatchEq(1))),
    ];
    for (key, arch, shape) in attn_variants {
        registry.register(KernelVariant {
            key: *key,
            arch_required: *arch,
            shape_gate: shape.clone(),
            steps: &[PipelineOp::Attend],
            has_awq: false,
            tile: TileImpl::None,
        });
    }

    // ── Flash Attention — batched prefill / tree-verify ────────
    // PRIORITY ORDER within each key: gfx12 → gfx11 → scalar — DO NOT REORDER.
    // resolve() returns the first passing (arch × shape) variant.

    // WMMA-FA: asym4 + Q8-V only. head_dim ∈ {128,256}, tree-verify excluded,
    // batch >= WMMA_BLOCK_M (16).
    use crate::types::TileImpl;
    const WMMA_BLOCK_M: usize = 16;
    registry.register(KernelVariant {
        key: KernelKey::AttnFlashAsym4BatchedMasked,
        arch_required: ArchPredicate::HasWmmaGfx12,
        shape_gate: Some(ShapePredicate::And(&[
            ShapePredicate::HeadDimIn(&[128, 256]),
            ShapePredicate::BatchGe(WMMA_BLOCK_M),
            ShapePredicate::IsTree(false),
        ])),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::Asym4WmmaTileGfx12,
    });
    registry.register(KernelVariant {
        key: KernelKey::AttnFlashAsym4BatchedMasked,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: Some(ShapePredicate::And(&[
            ShapePredicate::HeadDimIn(&[128, 256]),
            ShapePredicate::BatchGe(WMMA_BLOCK_M),
            ShapePredicate::IsTree(false),
        ])),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::Asym4WmmaTile,
    });

    // Scalar batched (fallback when WMMA doesn't apply)
    let attn_batched: &[(KernelKey, ArchPredicate, Option<ShapePredicate>)] = &[
        (KernelKey::AttnFlashAsym4BatchedMasked,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym4FwhtBatchedMasked, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym3BatchedMasked,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym3FwhtBatchedMasked, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        // 2-bit tiers: _batched only (no _masked — tree-verify gap, 3.3)
        (KernelKey::AttnFlashAsym2Batched,     ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        (KernelKey::AttnFlashAsym2FwhtBatched, ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
        // Q8_0: P-1 no-LDS-cap tiled kernel.
        // NOTE(F2): This dispatches to the P-1 tiled kernel for ALL Q8 batched
        // prefill, replacing the old single-pass LDS-staged softmax that was used
        // for max_ctx_len ≤ 15000. Different reduction order → not byte-identical
        // to master's ≤15k path. Numerically verified (NIAH 32k passed). Any
        // regression investigation should compare against master's two-path Q8,
        // not against this single-path kernel.
        (KernelKey::AttnQ8_0KvBatchedMasked,   ArchPredicate::Always, Some(ShapePredicate::BatchGt(1))),
    ];
    for (key, arch, shape) in attn_batched {
        registry.register(KernelVariant {
            key: *key,
            arch_required: *arch,
            shape_gate: shape.clone(),
            steps: &[PipelineOp::Attend],
            has_awq: false,
            tile: TileImpl::None,
        });
    }

    // ── Full attention (no KV cache — vision / DFlash) ─────────
    // PRIORITY ORDER within each key: gfx12 → gfx11 → scalar — DO NOT REORDER.
    // resolve() returns the first passing (arch × shape) variant.

    // AttnFullF16: non-causal, F16 K/V
    // v5 F16-K/V rung (gfx12). BatchGe(32) (not 64 like the gfx11 v5 rung
    // below) so it ALSO covers the 32..63 window that on gfx11 falls to the
    // DflashN128 tile — whose kernel (attention_dflash_wmma_n128_f16kv_f32)
    // is gfx11-only (`_w32` intrinsic, no gfx12 lowering) and would JIT-fail
    // on gfx1201. The v5 gfx12 kernel handles batch<64 correctly (host grid
    // = ceil(b/64), kernel masks gq>=B), so this shadows DflashN128 on gfx12
    // and keeps RDNA4 F16 full-attention on WMMA. gfx12-only predicate →
    // gfx11 behaviour (v5 for >=64, n128 for 32..63) is unchanged.
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF16,
        arch_required: ArchPredicate::HasWmmaGfx12,
        shape_gate: Some(ShapePredicate::And(&[
            ShapePredicate::HeadDimEq(128),
            ShapePredicate::BatchGe(32),
        ])),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashV5Gfx12,
    });
    // v5 F16-K/V rung (gfx11)
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF16,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: Some(ShapePredicate::And(&[
            ShapePredicate::HeadDimEq(128),
            ShapePredicate::BatchGe(64),
        ])),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashV5,
    });
    // n128 F16-K/V rung. gfx11 only — kernel has no gfx12 lowering. On gfx12
    // this is shadowed by the DflashV5Gfx12 rung above (BatchGe(32)), so it
    // is never selected on RDNA4 despite the HasWmma predicate (a gfx12 v5
    // entry at >=32 takes priority). Keep HasWmma: on gfx11 it is the live
    // 32..63 rung; do NOT narrow to HasWmmaW32 (AttnFullF16 has no scalar
    // floor, so an over-narrow gate would dead-gate the shape on gfx12).
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF16,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: Some(ShapePredicate::And(&[
            ShapePredicate::HeadDimEq(128),
            ShapePredicate::BatchGe(32),
        ])),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashN128,
    });
    // No scalar floor for F16 — fall to AttnFullF32 at caller level.

    // AttnFullF32: non-causal, F32 K/V
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF32,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: Some(ShapePredicate::And(&[
            ShapePredicate::HeadDimMultipleOf(16),
            ShapePredicate::HeadDimLe(128),
            ShapePredicate::BatchGe(32),
        ])),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashM32,
    });
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF32,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: Some(ShapePredicate::HeadDimMultipleOf(16)),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashWmmaF32,
    });
    // Scalar floor — Always
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF32,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashScalar,
    });

    // AttnFullF16Causal: causal, F16 K/V
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF16Causal,
        arch_required: ArchPredicate::HasWmmaGfx12,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashV3CausalGfx12,
    });
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF16Causal,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::DflashV3Causal,
    });
    // No scalar floor for F16 causal — fall to AttnFullF32Causal.

    // AttnFullF32Causal: causal, F32 K/V
    registry.register(KernelVariant {
        key: KernelKey::AttnFullF32Causal,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Attend],
        has_awq: false,
        tile: TileImpl::CausalScalar,
    });
}
