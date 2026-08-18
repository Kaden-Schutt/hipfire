// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
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
        (KernelKey::FusedQkvMq4G256V2,    ArchPredicate::Always),
        (KernelKey::FusedQkvMq4CG256,     ArchPredicate::Always),
        // MQ3/MQ4-Lloyd fused QKV: the merged Lloyd kernels are WMMA-free [32,1,1]
        // wave32 scalar (direct-LUT decode), so they run on every wave32 arch incl.
        // RDNA1/2 — NOT just WMMA archs. The old HasWmma gate was a dead-gate (it
        // was harmless on qwen35, which direct-dispatches these, but it lied to any
        // resolve()-routed caller). HasWave32 matches true kernel availability and
        // is byte-identical on RDNA3/4 (RDNA3/4 ⊂ wave32). CDNA stays excluded.
        (KernelKey::FusedQkvMq3G256Lloyd, ArchPredicate::HasWave32),
        (KernelKey::FusedQkvMq4G256Lloyd, ArchPredicate::HasWave32),
        // HFQ6G256 fused QKV: batched run-arm `gpu.gemm_qkv_hfq6g256` carries the
        // full cross-arch ladder (same as the qkvza sibling below). `Always`, not
        // gfx906-only `HasDp4a` — the old gate dead-gated HFQ6-promoted qkv layers
        // on RDNA3/4 (AWQ A3B trunk batched-prefill panic). Run-arm keeps the
        // gfx906 dp4a decode fast-path, cross-arch gemm (n=1) decode elsewhere.
        (KernelKey::FusedQkvHfq6G256, ArchPredicate::Always),
        (KernelKey::FusedQkvQ4K, ArchPredicate::Always),
        // Q8_0 fused QKV. Decode (None) → scalar `fused_qkv_q8_0` (cross-arch,
        // added 2026-06-14 for Qwen3.5-A3B .mq4p). Prefill (Some(n)) →
        // WMMA-only `gemm_qkv_q8_0_wmma` (gfx12 sibling on RDNA4 else gfx11);
        // the non-WMMA prefill case stays as three plain GemmQ8_0BatchedChunked
        // GEMMs at the call site (slice 1). `Always`: scalar decode covers all
        // non-WMMA archs; WMMA prefill is guarded inside the dispatch arm by
        // the call site's own `q8_wmma_arch = has_wmma()` check.
        (KernelKey::FusedQkvQ8_0, ArchPredicate::Always),
        // HFQ3G256 fused QKV (#397 Ship 5.2 slice 3). Always — the qwen35 prefill
        // site picks `gpu.gemm_qkv_hfq3g256_wmma` on has_wmma() archs and the base
        // `gpu.gemm_qkv_hfq3g256` otherwise, and the base method itself carries a
        // full cross-arch internal ladder (MMQ → dp4a → dot2 → fp16 → scalar
        // gfx1010 fallback). So the dtype runs on every arch (the run-arm picks
        // WMMA vs base by arch, mirroring the call site). Mirrors the
        // FusedGateUpHfq3G256 row; NOT HasWmma (the base has a non-WMMA body).
        (KernelKey::FusedQkvHfq3G256, ArchPredicate::Always),
        // HFP4G32 fused QKV (#397 Ship 5.2 FINAL). WMMA-only: the run-arm calls
        // `gpu.gemm_qkv_hfp4g32`, which dispatches ONLY to WMMA kernels (gfx12 FP8/
        // WMMA siblings on has_wmma_w32_gfx12() RDNA4 else the gfx11 `_wmma`
        // kernel) — there is NO scalar/dp4a fallback. HasWmma (= has_wmma(),
        // includes gfx12) is correct; the gfx12 sibling is reached INSIDE the
        // method, so HasWmmaW32 (gfx12-excluding) would be wrong. Mirrors the
        // FusedGateUpHfp4G32 row.
        (KernelKey::FusedQkvHfp4G32, ArchPredicate::HasWmma),
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
        (KernelKey::FusedQkvzaMq4G256V2,    ArchPredicate::Always),
        (KernelKey::FusedQkvzaMq4CG256,    ArchPredicate::Always),
        // MQ3/MQ4-Lloyd fused QKVZA: WMMA-free [32,1,1] wave32 scalar Lloyd kernels,
        // run on every wave32 arch incl. RDNA1/2. HasWmma was a dead-gate (qwen35
        // direct-dispatches); HasWave32 matches true availability, byte-identical
        // on RDNA3/4, CDNA excluded. Mirrors the FusedQkv*Lloyd rows above.
        (KernelKey::FusedQkvzaMq3G256Lloyd, ArchPredicate::HasWave32),
        (KernelKey::FusedQkvzaMq4G256Lloyd, ArchPredicate::HasWave32),
        // HFQ6G256 fused QKVZA. The batched run-arm calls `gpu.gemm_qkvza_hfq6g256`,
        // which carries the full cross-arch ladder internally (wmma_gfx12 → wmma →
        // wave64_dp4a → dot2 → fp16 → scalar base), so the key is available on
        // EVERY arch — not just gfx906. The prior `HasDp4a` gate was the decode
        // (dp4a) method's reach leaking onto the key; it dead-gated the AWQ A3B
        // trunk's HFQ6-promoted qkvza layers on RDNA3/RDNA4 (batched-prefill panic
        // "no implementation for FusedQkvzaHfq6G256"). `Always` matches true
        // cross-arch availability (mirrors FusedQkvzaHfq4G256 / FusedQkvzaHfq3G256);
        // the run-arm keeps the gfx906 dp4a decode fast-path and falls to the
        // cross-arch gemm (n=1) for decode on other archs.
        (KernelKey::FusedQkvzaHfq6G256, ArchPredicate::Always),
        // Q8_0 fused QKVZA. Decode (None) → scalar `fused_qkvza_q8_0` (cross-arch,
        // added 2026-06-14 for Qwen3.5-A3B .mq4p DeltaNet layers). Prefill
        // (Some(n)) → WMMA-only `gemm_qkvza_q8_0_wmma`. `Always` — scalar decode
        // covers all non-WMMA archs; WMMA prefill guard is inside the dispatch arm.
        (KernelKey::FusedQkvzaQ8_0, ArchPredicate::Always),
        // HFQ3G256 fused QKVZA (#397 Ship 5.2 slice 3). Always — base
        // `gpu.gemm_qkvza_hfq3g256` carries the full cross-arch ladder
        // (MMQ → dot2 → fp16 → scalar); the run-arm picks `_wmma` vs base by arch.
        // Mirrors FusedQkvHfq3G256 / FusedGateUpHfq3G256.
        (KernelKey::FusedQkvzaHfq3G256, ArchPredicate::Always),
        // HFP4G32 fused QKVZA (#397 Ship 5.2 FINAL). WMMA-only — the run-arm calls
        // `gpu.gemm_qkvza_hfp4g32` (gfx12 WMMA sibling on RDNA4 else gfx11 `_wmma`),
        // NO scalar/dp4a fallback. Mirrors the FusedQkvHfp4G32 row above.
        // HasWmma (includes gfx12).
        (KernelKey::FusedQkvzaHfp4G32, ArchPredicate::HasWmma),
        // MFP4G32E8 fused QKVZA — decode-only, gfx1151 (Strix Halo) launch-fusion.
        // `Always` here because the gfx1151 firewall lives in the producing GUARD
        // (`guard_qkvza_mfp4g32e8` in pipeline/steps.rs checks
        // `ctx.arch.is_gfx1151()`); this key is ONLY ever produced by that decode
        // guard — no prefill site emits it — so resolve() never sees it on another
        // arch. The run-arm (`gpu.fused_qkvza_mfp4g32_e8`) launches the
        // gfx1151-only kernel. No bleed to gfx1100/942/1201.
        (KernelKey::FusedQkvzaMfp4G32E8, ArchPredicate::Always),
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
        (KernelKey::FusedGateUpMq4G256V2,    ArchPredicate::Always),
        (KernelKey::FusedGateUpMq4CG256,    ArchPredicate::Always),
        // MQ3/MQ4-Lloyd fused gate+up: WMMA-free [32,1,1] wave32 scalar Lloyd
        // kernels, run on every wave32 arch incl. RDNA1/2. HasWmma was a dead-gate
        // (qwen35 direct-dispatches); HasWave32 matches true availability,
        // byte-identical on RDNA3/4, CDNA excluded. Mirrors the rows above.
        (KernelKey::FusedGateUpMq3G256Lloyd, ArchPredicate::HasWave32),
        (KernelKey::FusedGateUpMq4G256Lloyd, ArchPredicate::HasWave32),
        // HFQ6G256 fused gate+up: batched run-arm `gpu.gemm_gate_up_hfq6g256` is
        // cross-arch (same ladder as the qkv/qkvza siblings). `Always`, not
        // gfx906-only `HasDp4a` — the old gate dead-gated HFQ6-promoted gate_up
        // layers on RDNA3/4 (AWQ A3B trunk batched-prefill panic). Run-arm keeps
        // the gfx906 dp4a decode fast-path, cross-arch gemm (n=1) decode elsewhere.
        (KernelKey::FusedGateUpHfq6G256, ArchPredicate::Always),
        (KernelKey::FusedGateUpQ4K, ArchPredicate::Always),
        // HFQ3G256 fused gate+up (#397 Ship 5.2 slice 2). The qwen35 prefill
        // site routes this dtype to `gpu.gemm_gate_up_hfq3g256_wmma` on
        // has_wmma() archs and to the base `gpu.gemm_gate_up_hfq3g256` otherwise
        // — and the base method itself carries a full cross-arch internal ladder
        // (MMQ → dp4a → dot2 → fp16 → scalar gfx1010 fallback). So the *dtype*
        // runs on every arch (the run-arm picks WMMA vs base by arch, mirroring
        // the call site). `Always` matches that true cross-arch availability —
        // NOT HasWmma: unlike the GEMV-side MQ3 key (which has no scalar GEMM
        // fallback), the gate_up prefill kernel has a non-WMMA body.
        (KernelKey::FusedGateUpHfq3G256, ArchPredicate::Always),
        // HFP4G32 fused gate+up (#397 Ship 5.2 slice 2). FLAGGED: differs from the
        // sibling FusedGateUpHfq4G256 (`Always`) row. `gpu.gemm_gate_up_hfp4g32`
        // dispatches ONLY to WMMA kernels — `gemm_gate_up_hfp4g32_wmma_gfx12` on
        // has_wmma_w32_gfx12() (RDNA4) else `gemm_gate_up_hfp4g32_wmma` (gfx11
        // RDNA3) — with NO scalar/dp4a fallback body. So the real arch support is
        // WMMA-only (RDNA3 + RDNA4). HasWmma (= has_wmma(), which includes gfx12)
        // is the correct gate; the gfx12 sibling is reached inside the method, so
        // HasWmmaW32 (gfx12-excluding) would be wrong.
        (KernelKey::FusedGateUpHfp4G32, ArchPredicate::HasWmma),
        // Paro4G128T fused: generic wave32 kernels, no ISA-specific intrinsics.
        // Previously gated on HasDp4a (has_dot2_f32_f16) which excluded gfx906/gfx1010.
        (KernelKey::FusedGateUpParo4G128T, ArchPredicate::Always),
        // Q8_0 gate+up: plain wave32 kernel (`gpu.fused_gate_up_q8_0`),
        // no arch gate — mirrors the Q4K row. Used by qwen2 FFN.
        (KernelKey::FusedGateUpQ8_0, ArchPredicate::Always),
        // MFP4G32E8 gate+up — decode-only, gfx1151 (Strix Halo) launch-fusion.
        // `Always` for the same reason as the FusedQkvzaMfp4G32E8 row above: the
        // gfx1151 firewall is in the producing GUARD (`guard_gate_up_mfp4g32e8`),
        // the sole emitter of this key, so resolve() never sees it elsewhere.
        (KernelKey::FusedGateUpMfp4G32E8, ArchPredicate::Always),
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
