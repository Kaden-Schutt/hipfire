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
        // gfx11-only: the kernel source uses the gfx11 half16-layout
        // __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 and has no gfx12
        // sibling — admitting RDNA4 here would fail loud at JIT. gfx12
        // F16 GEMMs route through gemm_f16_wmma_mb8 (has a _gfx12 source).
        arch_required: ArchPredicate::HasWmmaW32,
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
    // HFQ4G256 MMQ set (q8_1 activation-quant + integer tile matmul; i8-WMMA).
    // Kernel correct on ALL WMMA arches, but PER-ARCH PERF differs (#397 Ship 5,
    // 2026-06-06, fleet-measured):
    //   - gfx1100/RDNA3 (k9lin):   correct (NRMSE 0.024%) AND ~2.1x FASTER than WMMA.
    //   - gfx1151/RDNA3.5 (hipx):  correct; mixed (WMMA wins small tiles, MMQ wins large).
    //   - gfx1201/RDNA4 (hiptrx):  correct (NRMSE 0.18-0.24%) BUT a measured PREFILL
    //     REGRESSION vs the live gemm_hfq4g256_residual_wmma_gfx12 (K2-unroll):
    //     -11% (0.8B) / -34.5% (27B) e2e, AND it tips a marginal 0.8B+code prompt into
    //     a 4-gram attractor that WMMA does not. (The earlier "2.5x" was vs fp16_wave64,
    //     the WRONG baseline — vs the real gfx12 WMMA kernel MMQ LOSES.)
    // => Predicate = HasWmmaW32 (RDNA3 + RDNA3.5): the arches where MMQ can WIN. RDNA4
    // stays on WMMA — MMQ is correct there but never worth selecting. NOT HasMmq (gfx906
    // has no WMMA; uses a separate dp4a MMQ kernel). The gfx12 MMQ kernel is kept in-tree
    // (validated, dormant) but excluded here. Do NOT enable has_mmq()+rdna4 / promote MMQ
    // on RDNA4 (measured loss; feat/rdna4-mmq-live left unmerged).
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256MmqSet,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });

    // ── #397 Ship 5.2 FINAL: residual-fused GEMM catalog ────────────
    // All entries below take the residual signature `(a, x, y, m, k, batch_size)`
    // and compute `y += a·x` (the residual add is INTERNAL to each kernel — the
    // caller passes the residual stream as `y` and the kernel never reuses `y` as
    // GEMV scratch, so the migration cannot reintroduce the a9e8dfda aliasing
    // bug). Dispatched through GemmFamily::run_key against the explicit key, so
    // each method's own internal arch routing is preserved byte-for-byte.
    //
    // HFQ6 residual: `gpu.gemm_hfq6g256_residual` is the dispatcher entry — it
    // auto-routes WMMA(gfx12/gfx11) → dp4a(gfx906) → FP16 → scalar fallback, so
    // it runs on EVERY arch. `Always` mirrors the plain HFQ6 availability.
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq6G256Residual,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // HFQ4 residual: `gpu.gemm_hfq4g256_residual` is the dispatcher entry — full
    // cross-arch ladder (CDNA MFMA/rocBLAS → MMQ → dp4a → FP16 → WMMA → scalar).
    // Runs on every arch → `Always` (mirrors the plain GemmHfq4G256 row).
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256Residual,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // HFQ3 residual: the qwen35 site picks `gpu.gemm_hfq3g256_residual_wmma` on
    // has_wmma() archs (incl. gfx12, routed inside) else the base
    // `gpu.gemm_hfq3g256_residual`, whose own ladder is MMQ → dot2 → FP16 →
    // scalar. So the dtype runs on EVERY arch; the run-arm replicates the
    // call-site WMMA-vs-base arch split internally (mirrors FusedQkvHfq3G256).
    // `Always`, NOT HasWmma (the base has a non-WMMA scalar body).
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq3G256Residual,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // HFP4 residual: `gpu.gemm_hfp4g32_residual` dispatches ONLY to WMMA kernels
    // (`_wmma_gfx12` on has_wmma_w32_gfx12() RDNA4 else `_wmma` gfx11 RDNA3) —
    // NO scalar/dp4a fallback. Real support is WMMA-only. HasWmma (= has_wmma(),
    // includes gfx12) is correct; the gfx12 sibling is reached inside the method,
    // so HasWmmaW32 (gfx12-excluding) would be wrong. Mirrors FusedGateUpHfp4G32.
    registry.register(KernelVariant {
        key: KernelKey::GemmHfp4G32Residual,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // MQ3-Lloyd residual: `gpu.gemm_mq3g256_lloyd_residual_wmma` is WMMA-only —
    // `_for_arch` routes the gfx12 (RDNA4) sibling else the gfx11 (RDNA3) kernel
    // and PANICS on any other arch (the is_batchable_la upstream gate guarantees
    // only RDNA3/RDNA4 reach it). Real support is WMMA-only → HasWmma (includes
    // gfx12). Mirrors FusedQkvMq3G256Lloyd / FusedGateUpMq3G256Lloyd.
    registry.register(KernelVariant {
        key: KernelKey::GemmMq3G256LloydResidual,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });

    // ── #397 Ship 5.3: spec-decode (DFlash) batched lm_head catalog ────────
    // All entries below take the canonical signature `(a, x, y, m, k, batch_size)`
    // and are the dispatcher entries for the spec-decode draft/verify batched
    // lm_head GEMMs in hipfire-arch-qwen35/src/speculative.rs. Dispatched through
    // GemmFamily::run_key against the explicit key, so each method's own internal
    // arch routing is preserved byte-for-byte.
    //
    // `gpu.gemm_q8_0_batched` is a scalar generic kernel (32-thread blocks, no
    // ISA-specific intrinsics) — runs on EVERY arch → `Always`.
    registry.register(KernelVariant {
        key: KernelKey::GemmQ8_0Batched,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    // The three `gemm_*_batched_lmhead` methods are full dispatcher entries: each
    // auto-routes WMMA (gfx11 `_w32` / gfx12 `_w32_gfx12`) for batch>1, dp4a
    // (gfx906) for HFQ6, and an fp16/scalar fallback otherwise — so the dtype
    // runs on EVERY arch. `Always` mirrors the plain GemmHfq4G256 availability.
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq4G256BatchedLmhead,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq3G256BatchedLmhead,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    registry.register(KernelVariant {
        key: KernelKey::GemmHfq6G256BatchedLmhead,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
}
