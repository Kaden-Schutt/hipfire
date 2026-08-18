// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Built-in HIP kernel sources for inference operations.

use crate::arch_caps::ArchCaps;

/// GEMV F32: y = alpha * A * x + beta * y
/// Uses shared memory reduction across wavefronts.
pub const GEMV_SRC: &str = include_str!("../../../kernels/src/gemv.hip");

/// GEMV Q4_K: matrix-vector multiply with on-the-fly Q4_K dequantization.
/// A is stored as Q4_K blocks (144 bytes per 256 elements).
/// x is F32, y is F32. y = A_dequant * x.
///
/// Q4_K block layout (144 bytes for 256 elements):
///   [0:2]   f16 d (super-block scale)
///   [2:4]   f16 dmin (super-block min)
///   [4:16]  scales[12] (packed 6-bit scales/mins for 8 sub-blocks)
///   [16:144] qs[128] (4-bit quantized values, paired sub-blocks share 32 bytes)
///
/// Data layout: 4 groups of 64 elements. Each group has 2 sub-blocks sharing 32 bytes.
///   Group g (elements g*64..g*64+63):
///     sub-block 2g:   lower nibbles of qs[g*32..g*32+32] → elements g*64+0..g*64+31
///     sub-block 2g+1: upper nibbles of qs[g*32..g*32+32] → elements g*64+32..g*64+63
pub const GEMV_Q4K_SRC: &str = include_str!("../../../kernels/src/gemv_q4k.hip");

/// HFQ4-G128: flat 4-bit with 128-weight groups.
/// Block: [f32 scale][f32 zero][64B nibbles] = 72 bytes per 128 weights.
/// Minimal metadata → minimal VGPRs. Hypothesis: ≤32 VGPRs → max occupancy.
pub const GEMV_HFQ4G128_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g128.hip");

/// HFQ4-G128 batched GEMV with fused per-token sigmoid-scaled residual.
/// HFQ4-G256 sister: `GEMV_HFQ4G256_RESIDUAL_SCALED_SRC`. Used by the
/// PARO shared-expert down dispatch (Phase 2 — moe_ffn_batched_admissible
/// under HIPFIRE_PARO_BATCHED=1).
pub const GEMV_HFQ4G128_RESIDUAL_SIGMOID_SCALED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g128_residual_sigmoid_scaled.hip");

/// PARO4-G128: ParoQuant-compatible rotated activation + W4 GEMV.
/// Block: [f32 scale][f32 zero][64B nibbles] = 72 bytes per 128 weights,
/// followed by shared pair-rotation metadata and channel scales.
pub const GEMV_PARO4G128_SRC: &str = include_str!("../../../kernels/src/gemv_paro4g128.hip");

/// HFQ4-G128 batched GEMM: same tiled approach as G256 but 72 bytes/group, 4 weights/thread.
pub const GEMM_HFQ4G128_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g128.hip");

/// HFQ4-G128 i8 WMMA MMQ (non-grouped) for gfx1151. Mirror of the routed
/// grouped k8 kernel minus expert scatter — same i8 WMMA pattern, same
/// 16×16 tile, same 8-WMMA-per-group K-pipeline. Activation must be
/// pre-quantized to block_q8_1_mmq via `quantize_q8_1_mmq_ds4` (handled
/// transparently by the dispatcher). Closes the perf gap left when only
/// the routed-expert path got the MMQ port — see the
/// `paroquant-real-bottleneck-gemm-hfq4g128` rocprof finding (66% of
/// pp256 prefill time was the non-grouped baseline).
pub const GEMM_HFQ4G128_MMQ_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g128_mmq.gfx1151.hip");

/// HFQ2-G256: flat 2-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][64B data] = 72 bytes per 256 weights (0.28 B/w).
pub const GEMV_HFQ2G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq2g256.hip");

/// MQ2G256Lloyd: 2-bit + per-block 4-entry fp16 codebook (72 B/group).
pub const GEMV_MQ2G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd.hip");

/// MQ3G256Lloyd: 3-bit + per-block 8-entry fp16 codebook (112 B/group).
pub const GEMV_MQ3G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd.hip");
/// MQ4G256Lloyd: 4-bit + per-block 16-entry fp16 codebook (160 B/group).
pub const GEMV_MQ4G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq4g256_lloyd.hip");
/// gfx1100 (RDNA3) variant: K4 unroll + 64-slot LDS-codebook (two-phase
/// cooperative load) + SINGLE linear accumulator. 71 VGPR / 18 SGPR /
/// 256 B LDS / 0 spills. See kernel header for why single-acc (multi-acc K4
/// produced 1.7% PPL drift on Qwen3.5-9B vs slow generic).
pub const GEMV_MQ4G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq4g256_lloyd.gfx1100.hip");

/// MQ4G256Lloyd residual GEMV: y[row] += A[row] · x. Eliminates the
/// alloc + gemv + add_inplace_f32 + free fallback chain on residual paths.
pub const GEMV_MQ4G256_LLOYD_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq4g256_lloyd_residual.hip");
pub const GEMV_MQ4G256_LLOYD_RESIDUAL_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq4g256_lloyd_residual.gfx1100.hip");
/// MQ4G256Lloyd fused gate+up GEMV: 2 GEMVs in one launch (FFN preamble).
pub const FUSED_GATE_UP_MQ4G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_mq4g256_lloyd.hip");
pub const FUSED_GATE_UP_MQ4G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_mq4g256_lloyd.gfx1100.hip");
/// MQ4G256Lloyd fused QKVZA GEMV: 4 GEMVs in one launch (LinearAttention
/// preamble — qkv + z + beta + alpha).
pub const FUSED_QKVZA_MQ4G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_mq4g256_lloyd.hip");
pub const FUSED_QKVZA_MQ4G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_mq4g256_lloyd.gfx1100.hip");
/// MQ4G256Lloyd fused QKV GEMV: 3 GEMVs in one launch (FullAttention preamble).
pub const FUSED_QKV_MQ4G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_mq4g256_lloyd.hip");
pub const FUSED_QKV_MQ4G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_mq4g256_lloyd.gfx1100.hip");
pub const FUSED_QKV_MQ4G256_LLOYD_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq4g256_lloyd_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_mq4g256_lloyd.hip")
);
pub const FUSED_QKV_MQ4G256_LLOYD_GFX1100_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq4g256_lloyd_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_mq4g256_lloyd.gfx1100.hip")
);
/// DIAGNOSTIC ONLY — broken K4 multi-accumulator MQ4-Lloyd kernel kept for
/// the open-question investigation of why MQ3-Lloyd's multi-acc works but
/// MQ4-Lloyd's doesn't. NOT used in the production dispatch path; reachable
/// only via the explicit `Gpu::gemv_mq4g256_lloyd_multiacc_diag` method that
/// `examples/diag_mq4_lloyd_multiacc.rs` calls.
pub const GEMV_MQ4G256_LLOYD_MULTIACC_DIAG_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq4g256_lloyd_multiacc_diag.gfx1100.hip");

/// MQ4-Lloyd WMMA prefill kernels (Phase 5b — see
/// docs/plans/mq4-lloyd-wmma-prefill.md). 16-row × 16-batch tile, per-row
/// LDS-staged fp16 codebook (512 B/workgroup, no cvt at decode — fp16
/// inherited from MQ3 Phase A's 7.15% bench win).
pub const GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq4g256_lloyd_residual_wmma.hip");
/// gfx12 (RDNA4) sibling — code-complete but runtime-unvalidated locally per Phase B1 plan.
pub const GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq4g256_lloyd_residual_wmma.gfx12.hip");
/// Phase D-A: 16×64 output tile per WG (4 batch sub-tiles share A_reg decode).
/// Shipped to close the batch ≥ 128 GiB/s gap diagnosed in
/// `benchmarks/results/devlog_20260509_mq4_lloyd_gfx1151_bench.md`. Same 160 B
/// stride and codebook decode as `_wmma`; only the batch-fanout and grid shape change.
pub const GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq4g256_lloyd_residual_wmma_mb4.hip");
/// Phase D experiment: 16×32 output tile (2 batch sub-tiles per WG). Lower
/// VGPR pressure (~85 vs mb4's 106) at the cost of half the per-WG weight
/// reuse — better at small-M residual where mb4 is occupancy-bound.
pub const GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_MB2_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq4g256_lloyd_residual_wmma_mb2.hip");
/// MQ4G256Lloyd WMMA fused QKVZA (LA preamble: qkv + z + beta + alpha, 4-way).
pub const GEMM_QKVZA_MQ4G256_LLOYD_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq4g256_lloyd_wmma.hip");
pub const GEMM_QKVZA_MQ4G256_LLOYD_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq4g256_lloyd_wmma.gfx12.hip");
pub const GEMM_QKVZA_MQ4G256_LLOYD_WMMA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq4g256_lloyd_wmma.gfx1151.hip");
/// MQ4G256Lloyd WMMA fused QKV (FA preamble: q + k + v, 3-way).
pub const GEMM_QKV_MQ4G256_LLOYD_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq4g256_lloyd_wmma.hip");
pub const GEMM_QKV_MQ4G256_LLOYD_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq4g256_lloyd_wmma.gfx12.hip");
pub const GEMM_QKV_MQ4G256_LLOYD_WMMA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq4g256_lloyd_wmma.gfx1151.hip");
/// MQ4G256Lloyd WMMA fused gate+up (FFN, 2-way).
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.hip");
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.gfx12.hip");
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma.gfx1151.hip");
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_NOSYNC_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_nosync.hip");
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_NOSYNC_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_nosync.gfx1151.hip");

/// Phase D-B: 16×64 output tile per WG (4 batch sub-tiles share A_reg decode).
/// Same shape as `_wmma`; only the per-WG output fanout and grid differ.
/// gfx11 only (gfx12 sibling deferred per Phase D plan).
pub const GEMM_QKVZA_MQ4G256_LLOYD_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq4g256_lloyd_wmma_mb4.hip");
pub const GEMM_QKV_MQ4G256_LLOYD_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq4g256_lloyd_wmma_mb4.hip");
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_mb4.hip");

/// gfx1151 (Strix Halo) K4 variants of the Phase D-B mb4 family.
/// K4 unroll front-loads 8 nibble-pack reads per inner iteration (vs K2's 4)
/// to better hide LPDDR5x unified-memory latency on the APU.
pub const GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_MB4_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq4g256_lloyd_residual_wmma_mb4.gfx1151.hip");
pub const GEMM_QKVZA_MQ4G256_LLOYD_WMMA_MB4_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq4g256_lloyd_wmma_mb4.gfx1151.hip");
pub const GEMM_QKV_MQ4G256_LLOYD_WMMA_MB4_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq4g256_lloyd_wmma_mb4.gfx1151.hip");
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_mb4.gfx1151.hip");

/// Barrier-free nosync variant of the MQ4-Lloyd fuse gate+up wmma mb4.
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_NOSYNC_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync.hip");
/// gfx1151 K4 barrier-free variant.
pub const GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_NOSYNC_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync.gfx1151.hip");

/// Returns the MQ4G256Lloyd WMMA residual GEMM kernel source AND module name for
/// the given arch.
///
/// **Arch matrix** (matched by all 4 MQ4-Lloyd `*_for_arch` selectors):
/// - `gfx1100/1101/1102/1151` → rdna3 module (gfx11 source)
/// - `gfx1200/1201` → rdna4 module (gfx12 source, `_w32_gfx12` builtin + K4-unroll
///   half8_t lane-split — see `*.gfx12.hip` headers for why K4 vs gfx11's K2)
/// - everything else → default arm (gfx11 source, generic module name)
///
/// **gfx1150 is intentionally excluded** to keep symmetric arch coverage with the
/// MQ4-Lloyd GEMV/fused decode path (which only validates gfx1100/1101/1102/1151);
/// admitting gfx1150 to one and not the other is the asymmetry called out in the
/// PR #195 review (GLM-5 L1 / Claude M3). gfx1150 hardware can be enabled here
/// after a parity + bench round on Strix-Halo-class hardware.
///
/// The C symbol is unsuffixed on both gfx11 and gfx12 (the gfx12 `.hip` files drop
/// the `_gfx12` suffix from the C symbol so the unsuffixed dispatch lookup
/// resolves under both per-arch hsaco caches).
pub fn gemm_mq4g256_lloyd_residual_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_mq4g256_lloyd_residual_wmma_rdna4",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_SRC,
            "gemm_mq4g256_lloyd_residual_wmma_rdna3",
        ),
        _ => panic!(
            "MQ4-Lloyd WMMA residual: unsupported arch {arch}. The is_batchable_la upstream \
             gate should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_mq4g256_lloyd_residual_wmma_for_arch (160 B Lloyd stride would mismatch \
             any default kernel)."
        ),
    }
}
/// Phase D-A selector — same arch matrix as `_wmma_for_arch` (gfx1100/1101/1102/1151
/// only; gfx12 sibling deferred per the Phase D plan). Single-arch source since
/// the kernel's tile shape is gfx11/wave32 specific.
pub fn gemm_mq4g256_lloyd_residual_wmma_mb4_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1151" => (
            GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_MB4_GFX1151_SRC,
            "gemm_mq4g256_lloyd_residual_wmma_mb4_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_MB4_SRC,
            "gemm_mq4g256_lloyd_residual_wmma_mb4_rdna3",
        ),
        _ => panic!(
            "MQ4-Lloyd WMMA mb4 residual: unsupported arch {arch}. Phase D-A is gfx11-only; \
             gfx12 sibling deferred. is_batchable_la must not admit gfx12 to the mb4 path \
             (160 B Lloyd stride would mismatch any default kernel)."
        ),
    }
}
pub fn gemm_qkvza_mq4g256_lloyd_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_QKVZA_MQ4G256_LLOYD_WMMA_GFX12_SRC,
            "gemm_qkvza_mq4g256_lloyd_wmma_rdna4",
        ),
        "gfx1151" => (
            GEMM_QKVZA_MQ4G256_LLOYD_WMMA_GFX1151_SRC,
            "gemm_qkvza_mq4g256_lloyd_wmma_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_QKVZA_MQ4G256_LLOYD_WMMA_SRC,
            "gemm_qkvza_mq4g256_lloyd_wmma_rdna3",
        ),
        _ => panic!(
            "MQ4-Lloyd WMMA qkvza: unsupported arch {arch}. The is_batchable_la upstream gate \
             should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_qkvza_mq4g256_lloyd_wmma_for_arch."
        ),
    }
}
pub fn gemm_qkv_mq4g256_lloyd_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_QKV_MQ4G256_LLOYD_WMMA_GFX12_SRC,
            "gemm_qkv_mq4g256_lloyd_wmma_rdna4",
        ),
        "gfx1151" => (
            GEMM_QKV_MQ4G256_LLOYD_WMMA_GFX1151_SRC,
            "gemm_qkv_mq4g256_lloyd_wmma_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_QKV_MQ4G256_LLOYD_WMMA_SRC,
            "gemm_qkv_mq4g256_lloyd_wmma_rdna3",
        ),
        _ => panic!(
            "MQ4-Lloyd WMMA qkv: unsupported arch {arch}. The is_batchable_la upstream gate \
             should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_qkv_mq4g256_lloyd_wmma_for_arch."
        ),
    }
}
pub fn gemm_gate_up_mq4g256_lloyd_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_GFX12_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_rdna4",
        ),
        "gfx1151" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_GFX1151_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_rdna3",
        ),
        _ => panic!(
            "MQ4-Lloyd WMMA gate_up: unsupported arch {arch}. The is_batchable_la upstream gate \
             should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_gate_up_mq4g256_lloyd_wmma_for_arch."
        ),
    }
}

pub fn gemm_gate_up_mq4g256_lloyd_wmma_nosync_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1151" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_NOSYNC_GFX1151_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_nosync_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_NOSYNC_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_nosync_rdna3",
        ),
        _ => panic!("MQ4-Lloyd WMMA gate_up nosync: unsupported arch {arch}."),
    }
}

/// Phase D experiment selector for residual mb2 (16×32 output tile).
pub fn gemm_mq4g256_lloyd_residual_wmma_mb2_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_MB2_SRC,
            "gemm_mq4g256_lloyd_residual_wmma_mb2_rdna3",
        ),
        _ => panic!("MQ4-Lloyd WMMA mb2 residual: unsupported arch {arch}. gfx11-only."),
    }
}

/// Phase D-B selectors for fused siblings. Same arch matrix as the residual
/// `_mb4` selector (gfx11 only — gfx12 sibling deferred per Phase D plan).
pub fn gemm_qkvza_mq4g256_lloyd_wmma_mb4_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1151" => (
            GEMM_QKVZA_MQ4G256_LLOYD_WMMA_MB4_GFX1151_SRC,
            "gemm_qkvza_mq4g256_lloyd_wmma_mb4_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_QKVZA_MQ4G256_LLOYD_WMMA_MB4_SRC,
            "gemm_qkvza_mq4g256_lloyd_wmma_mb4_rdna3",
        ),
        _ => panic!("MQ4-Lloyd WMMA mb4 qkvza: unsupported arch {arch}. Phase D-B is gfx11-only."),
    }
}
pub fn gemm_qkv_mq4g256_lloyd_wmma_mb4_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1151" => (
            GEMM_QKV_MQ4G256_LLOYD_WMMA_MB4_GFX1151_SRC,
            "gemm_qkv_mq4g256_lloyd_wmma_mb4_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_QKV_MQ4G256_LLOYD_WMMA_MB4_SRC,
            "gemm_qkv_mq4g256_lloyd_wmma_mb4_rdna3",
        ),
        _ => panic!("MQ4-Lloyd WMMA mb4 qkv: unsupported arch {arch}. Phase D-B is gfx11-only."),
    }
}
pub fn gemm_gate_up_mq4g256_lloyd_wmma_mb4_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1151" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_GFX1151_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_mb4_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_mb4_rdna3",
        ),
        _ => {
            panic!("MQ4-Lloyd WMMA mb4 gate_up: unsupported arch {arch}. Phase D-B is gfx11-only.")
        }
    }
}

pub fn gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1151" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_NOSYNC_GFX1151_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync_k4_gfx1151",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMM_GATE_UP_MQ4G256_LLOYD_WMMA_MB4_NOSYNC_SRC,
            "gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync_rdna3",
        ),
        _ => panic!("MQ4-Lloyd WMMA mb4 gate_up nosync: unsupported arch {arch}."),
    }
}

/// Returns the MQ4G256-Lloyd GEMV kernel source AND module name for the given
/// arch. gfx1100/1101/1102 (RDNA3) and gfx1151 (RDNA3.5 Strix Halo APU) get the
/// K4-unrolled + LDS-codebook fast variant; other archs fall back to the
/// chip-agnostic baseline switch-dispatch path. gfx1151 is included for
/// on-host conformance testing — definitive MQ4-Lloyd perf comparisons happen
/// on gfx1100 (the format's calibrated target arch).
pub fn gemv_mq4g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    // Same HIPFIRE_LLOYD_FORCE_BASELINE escape hatch as MQ3-Lloyd, so the fast
    // variant can be A/B'd against the baseline on the same model file.
    if force_baseline {
        return (GEMV_MQ4G256_LLOYD_SRC, "gemv_mq4g256_lloyd");
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
            (GEMV_MQ4G256_LLOYD_GFX1100_SRC, "gemv_mq4g256_lloyd_rdna3")
        }
        _ => (GEMV_MQ4G256_LLOYD_SRC, "gemv_mq4g256_lloyd"),
    }
}

/// Same arch dispatch as `gemv_mq4g256_lloyd_for_arch` but returns the residual
/// variant (y[row] += A[row] · x).
pub fn gemv_mq4g256_lloyd_residual_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (
            GEMV_MQ4G256_LLOYD_RESIDUAL_SRC,
            "gemv_mq4g256_lloyd_residual",
        );
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            GEMV_MQ4G256_LLOYD_RESIDUAL_GFX1100_SRC,
            "gemv_mq4g256_lloyd_residual_rdna3",
        ),
        _ => (
            GEMV_MQ4G256_LLOYD_RESIDUAL_SRC,
            "gemv_mq4g256_lloyd_residual",
        ),
    }
}

/// Arch dispatch for fused gate+up MQ4-Lloyd. Mirrors MQ3-Lloyd's selector.
pub fn fused_gate_up_mq4g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (
            FUSED_GATE_UP_MQ4G256_LLOYD_SRC,
            "fused_gate_up_mq4g256_lloyd",
        );
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_GATE_UP_MQ4G256_LLOYD_GFX1100_SRC,
            "fused_gate_up_mq4g256_lloyd_rdna3",
        ),
        _ => (
            FUSED_GATE_UP_MQ4G256_LLOYD_SRC,
            "fused_gate_up_mq4g256_lloyd",
        ),
    }
}

/// Arch dispatch for fused QKVZA MQ4-Lloyd (4-way demux: qkv/z/beta/alpha).
pub fn fused_qkvza_mq4g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (FUSED_QKVZA_MQ4G256_LLOYD_SRC, "fused_qkvza_mq4g256_lloyd");
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_QKVZA_MQ4G256_LLOYD_GFX1100_SRC,
            "fused_qkvza_mq4g256_lloyd_rdna3",
        ),
        _ => (FUSED_QKVZA_MQ4G256_LLOYD_SRC, "fused_qkvza_mq4g256_lloyd"),
    }
}

/// Arch dispatch for fused QKV MQ4-Lloyd (3-way demux: q/k/v).
pub fn fused_qkv_mq4g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (FUSED_QKV_MQ4G256_LLOYD_SRC, "fused_qkv_mq4g256_lloyd");
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_QKV_MQ4G256_LLOYD_GFX1100_SRC,
            "fused_qkv_mq4g256_lloyd_rdna3",
        ),
        _ => (FUSED_QKV_MQ4G256_LLOYD_SRC, "fused_qkv_mq4g256_lloyd"),
    }
}

pub fn fused_qkv_mq4g256_lloyd_qwen2_bias_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    if force_baseline {
        return (
            FUSED_QKV_MQ4G256_LLOYD_QWEN2_BIAS_SRC,
            "fused_qkv_mq4g256_lloyd_qwen2_bias",
        );
    }
    match caps.arch() {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_QKV_MQ4G256_LLOYD_GFX1100_QWEN2_BIAS_SRC,
            "fused_qkv_mq4g256_lloyd_rdna3_qwen2_bias",
        ),
        _ => (
            FUSED_QKV_MQ4G256_LLOYD_QWEN2_BIAS_SRC,
            "fused_qkv_mq4g256_lloyd_qwen2_bias",
        ),
    }
}
/// gfx1100 (RDNA3) variant: K4 unroll + LDS-resident codebook lookup.
pub const GEMV_MQ3G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd.gfx1100.hip");
/// MQ3G256Lloyd residual GEMV: y[row] += A[row] dot x. Eliminates the
/// add_inplace_f32 launch on the residual path (~4.4% of decode time).
pub const GEMV_MQ3G256_LLOYD_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_residual.hip");
pub const GEMV_MQ3G256_LLOYD_RESIDUAL_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_residual.gfx1100.hip");
/// MQ3G256Lloyd WMMA residual GEMM (Phase 5 / issue #116 — batched-prefill kernel).
/// gfx1100+ wave32 WMMA. 16-row × 16-batch tile, per-row LDS-staged fp16 codebook
/// (256 B/workgroup, no cvt at decode — fp16 won the Phase A bench by 7.15%).
pub const GEMM_MQ3G256_LLOYD_RESIDUAL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq3g256_lloyd_residual_wmma.hip");
/// gfx12 (RDNA4) sibling — code-complete but runtime-unvalidated locally per Phase B1 plan.
pub const GEMM_MQ3G256_LLOYD_RESIDUAL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq3g256_lloyd_residual_wmma.gfx12.hip");
/// MQ3-Lloyd batch-fanout (mb4) family — 16×64 output tile per WG, 4 batch
/// sub-tiles share A_reg decode. Same multi-batch-tile pattern as the
/// MQ4-Lloyd mb4 family. gfx11 only (gfx12 sibling deferred).
pub const GEMM_MQ3G256_LLOYD_RESIDUAL_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq3g256_lloyd_residual_wmma_mb4.hip");

/// MQ3G256Lloyd WMMA fused QKVZA (LA preamble: qkv + z + beta + alpha, 4-way).
pub const GEMM_QKVZA_MQ3G256_LLOYD_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq3g256_lloyd_wmma.hip");
pub const GEMM_QKVZA_MQ3G256_LLOYD_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq3g256_lloyd_wmma.gfx12.hip");
pub const GEMM_QKVZA_MQ3G256_LLOYD_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_mq3g256_lloyd_wmma_mb4.hip");
/// MQ3G256Lloyd WMMA fused QKV (FA preamble: q + k + v, 3-way).
pub const GEMM_QKV_MQ3G256_LLOYD_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq3g256_lloyd_wmma.hip");
pub const GEMM_QKV_MQ3G256_LLOYD_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq3g256_lloyd_wmma.gfx12.hip");
pub const GEMM_QKV_MQ3G256_LLOYD_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_mq3g256_lloyd_wmma_mb4.hip");
/// MQ3G256Lloyd WMMA fused gate+up (FFN, 2-way).
pub const GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq3g256_lloyd_wmma.hip");
pub const GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq3g256_lloyd_wmma.gfx12.hip");
pub const GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq3g256_lloyd_wmma_mb4.hip");
pub const GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_NOSYNC_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq3g256_lloyd_wmma_nosync.hip");
pub const GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_MB4_NOSYNC_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_mq3g256_lloyd_wmma_mb4_nosync.hip");

/// Returns the MQ3G256Lloyd WMMA residual GEMM kernel source AND module name for
/// the given arch. Mirrors `gemm_hfq3g256_residual_wmma_for_arch`'s arch matrix.
pub fn gemm_mq3g256_lloyd_residual_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_MQ3G256_LLOYD_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_mq3g256_lloyd_residual_wmma_rdna4",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_MQ3G256_LLOYD_RESIDUAL_WMMA_SRC,
            "gemm_mq3g256_lloyd_residual_wmma_rdna3",
        ),
        _ => panic!(
            "MQ3-Lloyd WMMA residual: unsupported arch {arch}. The is_batchable_la upstream \
             gate should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_mq3g256_lloyd_residual_wmma_for_arch (112 B Lloyd stride would mismatch \
             any default kernel)."
        ),
    }
}
/// MQ3-Lloyd mb4 residual selector. gfx11 only — gfx12 sibling deferred.
pub fn gemm_mq3g256_lloyd_residual_wmma_mb4_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_MQ3G256_LLOYD_RESIDUAL_WMMA_MB4_SRC,
            "gemm_mq3g256_lloyd_residual_wmma_mb4_rdna3",
        ),
        _ => panic!("MQ3-Lloyd WMMA mb4 residual: unsupported arch {arch}. gfx11-only."),
    }
}

pub fn gemm_qkvza_mq3g256_lloyd_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_QKVZA_MQ3G256_LLOYD_WMMA_GFX12_SRC,
            "gemm_qkvza_mq3g256_lloyd_wmma_rdna4",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_QKVZA_MQ3G256_LLOYD_WMMA_SRC,
            "gemm_qkvza_mq3g256_lloyd_wmma_rdna3",
        ),
        _ => panic!(
            "MQ3-Lloyd WMMA qkvza: unsupported arch {arch}. The is_batchable_la upstream gate \
             should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_qkvza_mq3g256_lloyd_wmma_for_arch."
        ),
    }
}
pub fn gemm_qkv_mq3g256_lloyd_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_QKV_MQ3G256_LLOYD_WMMA_GFX12_SRC,
            "gemm_qkv_mq3g256_lloyd_wmma_rdna4",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_QKV_MQ3G256_LLOYD_WMMA_SRC,
            "gemm_qkv_mq3g256_lloyd_wmma_rdna3",
        ),
        _ => panic!(
            "MQ3-Lloyd WMMA qkv: unsupported arch {arch}. The is_batchable_la upstream gate \
             should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_qkv_mq3g256_lloyd_wmma_for_arch."
        ),
    }
}
pub fn gemm_gate_up_mq3g256_lloyd_wmma_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1200" | "gfx1201" => (
            GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_GFX12_SRC,
            "gemm_gate_up_mq3g256_lloyd_wmma_rdna4",
        ),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_SRC,
            "gemm_gate_up_mq3g256_lloyd_wmma_rdna3",
        ),
        _ => panic!(
            "MQ3-Lloyd WMMA gate_up: unsupported arch {arch}. The is_batchable_la upstream gate \
             should reject this; if you reached here, is_batchable_la was extended without \
             updating gemm_gate_up_mq3g256_lloyd_wmma_for_arch."
        ),
    }
}

pub fn gemm_gate_up_mq3g256_lloyd_wmma_nosync_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_NOSYNC_SRC,
            "gemm_gate_up_mq3g256_lloyd_wmma_nosync_rdna3",
        ),
        _ => panic!("MQ3-Lloyd WMMA gate_up nosync: unsupported arch {arch}."),
    }
}

/// MQ3-Lloyd fused mb4 selectors (gfx11 only).
pub fn gemm_qkvza_mq3g256_lloyd_wmma_mb4_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_QKVZA_MQ3G256_LLOYD_WMMA_MB4_SRC,
            "gemm_qkvza_mq3g256_lloyd_wmma_mb4_rdna3",
        ),
        _ => panic!("MQ3-Lloyd WMMA mb4 qkvza: unsupported arch {arch}. gfx11-only."),
    }
}
pub fn gemm_qkv_mq3g256_lloyd_wmma_mb4_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_QKV_MQ3G256_LLOYD_WMMA_MB4_SRC,
            "gemm_qkv_mq3g256_lloyd_wmma_mb4_rdna3",
        ),
        _ => panic!("MQ3-Lloyd WMMA mb4 qkv: unsupported arch {arch}. gfx11-only."),
    }
}
pub fn gemm_gate_up_mq3g256_lloyd_wmma_mb4_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_MB4_SRC,
            "gemm_gate_up_mq3g256_lloyd_wmma_mb4_rdna3",
        ),
        _ => panic!("MQ3-Lloyd WMMA mb4 gate_up: unsupported arch {arch}. gfx11-only."),
    }
}

pub fn gemm_gate_up_mq3g256_lloyd_wmma_mb4_nosync_for_arch(
    caps: &ArchCaps,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => (
            GEMM_GATE_UP_MQ3G256_LLOYD_WMMA_MB4_NOSYNC_SRC,
            "gemm_gate_up_mq3g256_lloyd_wmma_mb4_nosync_rdna3",
        ),
        _ => panic!("MQ3-Lloyd WMMA mb4 gate_up nosync: unsupported arch {arch}."),
    }
}
/// MQ3G256Lloyd fused gate+up GEMV: two GEMVs in one launch (saves 1 launch
/// per FFN). Mirrors fused_gate_up_hfq4g256.{,gfx1100.}hip.
pub const FUSED_GATE_UP_MQ3G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_mq3g256_lloyd.hip");
pub const FUSED_GATE_UP_MQ3G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_mq3g256_lloyd.gfx1100.hip");
/// MQ3G256Lloyd fused QKVZA GEMV: four LA-preamble GEMVs (wqkv + wz + w_beta
/// + w_alpha) in one launch. Saves 3 launches per LA layer per token + lets
/// the 16-row beta/alpha tails co-schedule with the 6144-row qkv body.
/// Mirrors fused_qkvza_hfq4g256.hip.
pub const FUSED_QKVZA_MQ3G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_mq3g256_lloyd.hip");
pub const FUSED_QKVZA_MQ3G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_mq3g256_lloyd.gfx1100.hip");
/// MQ3G256Lloyd fused QKV GEMV: three FA-preamble GEMVs (wq + wk + wv) in
/// one launch. Saves 2 launches per FA layer per token. Mirrors
/// fused_qkv_hfq4g256.hip — sibling of fused_qkvza for FullAttention.
pub const FUSED_QKV_MQ3G256_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_mq3g256_lloyd.hip");
pub const FUSED_QKV_MQ3G256_LLOYD_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_mq3g256_lloyd.gfx1100.hip");
pub const FUSED_QKV_MQ3G256_LLOYD_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq3g256_lloyd_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_mq3g256_lloyd.hip")
);
pub const FUSED_QKV_MQ3G256_LLOYD_GFX1100_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq3g256_lloyd_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_mq3g256_lloyd.gfx1100.hip")
);

/// Returns the MQ3G256-Lloyd GEMV kernel source AND module name for the given
/// arch. gfx1100/1101/1102 (RDNA3) gets the K4-unrolled + LDS-codebook variant
/// that closes the per-launch perf gap from the divergent-execution switch.
/// Other archs use the baseline (slower but correct switch-dispatch path).
pub fn gemv_mq3g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    // Debug escape hatch: HIPFIRE_LLOYD_FORCE_BASELINE=1 forces the slow generic
    // switch-dispatch kernel even on RDNA3, so the K4+LDS variant can be
    // logits-Δ'd against the baseline on the same model file. No perf cost when
    // unset (one missed-getenv per dispatch arm), and ensure_kernel short-
    // circuits after the first call regardless.
    if force_baseline {
        return (GEMV_MQ3G256_LLOYD_SRC, "gemv_mq3g256_lloyd");
    }
    let arch = caps.arch();
    match arch {
        // gfx1151 (Strix Halo APU, RDNA3.5) added 2026-05-07 after empirical
        // validation: K4 + LDS-codebook GEMV produces byte-equal PPL on
        // Qwen3.5-9B vs the slow generic kernel (NLL/tok 3.2110607378
        // byte-match at 10 decimal precision). Residual + fused (gate+up,
        // QKV, QKVZA) variants are NOT enabled on gfx1151 — extending the
        // matcher to all 5 produces ~0.9% PPL drift (multi-acc fp32-reorder
        // noise compounding under full coverage). gfx1100 remains the
        // calibrated perf target.
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
            (GEMV_MQ3G256_LLOYD_GFX1100_SRC, "gemv_mq3g256_lloyd_rdna3")
        }
        _ => (GEMV_MQ3G256_LLOYD_SRC, "gemv_mq3g256_lloyd"),
    }
}

/// Same arch dispatch as `gemv_mq3g256_lloyd_for_arch` but returns the residual
/// variant (y[row] += A[row] · x). HIPFIRE_LLOYD_FORCE_BASELINE=1 also routes
/// here to the baseline (for parity-test purposes).
pub fn gemv_mq3g256_lloyd_residual_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (
            GEMV_MQ3G256_LLOYD_RESIDUAL_SRC,
            "gemv_mq3g256_lloyd_residual",
        );
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            GEMV_MQ3G256_LLOYD_RESIDUAL_GFX1100_SRC,
            "gemv_mq3g256_lloyd_residual_rdna3",
        ),
        _ => (
            GEMV_MQ3G256_LLOYD_RESIDUAL_SRC,
            "gemv_mq3g256_lloyd_residual",
        ),
    }
}

/// Arch dispatch for fused gate+up MQ3-Lloyd. Same arch matrix as the GEMV
/// variants. Used by `qwen35.rs` FFN forward when both `w_gate` and `w_up`
/// are MQ3G256Lloyd to collapse 2 GEMV launches into 1.
pub fn fused_gate_up_mq3g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (
            FUSED_GATE_UP_MQ3G256_LLOYD_SRC,
            "fused_gate_up_mq3g256_lloyd",
        );
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_GATE_UP_MQ3G256_LLOYD_GFX1100_SRC,
            "fused_gate_up_mq3g256_lloyd_rdna3",
        ),
        _ => (
            FUSED_GATE_UP_MQ3G256_LLOYD_SRC,
            "fused_gate_up_mq3g256_lloyd",
        ),
    }
}

/// Arch dispatch for fused QKVZA MQ3-Lloyd. Used by `qwen35.rs` LA decode
/// when all four projections (wqkv, wz, w_beta, w_alpha) are MQ3G256Lloyd
/// to collapse 4 GEMV launches into 1.
pub fn fused_qkvza_mq3g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (FUSED_QKVZA_MQ3G256_LLOYD_SRC, "fused_qkvza_mq3g256_lloyd");
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_QKVZA_MQ3G256_LLOYD_GFX1100_SRC,
            "fused_qkvza_mq3g256_lloyd_rdna3",
        ),
        _ => (FUSED_QKVZA_MQ3G256_LLOYD_SRC, "fused_qkvza_mq3g256_lloyd"),
    }
}

/// Arch dispatch for fused QKV MQ3-Lloyd. Used by `qwen35.rs` FA decode
/// when wq, wk, wv are all MQ3G256Lloyd to collapse 3 GEMV launches into 1.
pub fn fused_qkv_mq3g256_lloyd_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    if force_baseline {
        return (FUSED_QKV_MQ3G256_LLOYD_SRC, "fused_qkv_mq3g256_lloyd");
    }
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_QKV_MQ3G256_LLOYD_GFX1100_SRC,
            "fused_qkv_mq3g256_lloyd_rdna3",
        ),
        _ => (FUSED_QKV_MQ3G256_LLOYD_SRC, "fused_qkv_mq3g256_lloyd"),
    }
}

pub fn fused_qkv_mq3g256_lloyd_qwen2_bias_for_arch(
    caps: &ArchCaps,
    force_baseline: bool,
) -> (&'static str, &'static str) {
    if force_baseline {
        return (
            FUSED_QKV_MQ3G256_LLOYD_QWEN2_BIAS_SRC,
            "fused_qkv_mq3g256_lloyd_qwen2_bias",
        );
    }
    match caps.arch() {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => (
            FUSED_QKV_MQ3G256_LLOYD_GFX1100_QWEN2_BIAS_SRC,
            "fused_qkv_mq3g256_lloyd_rdna3_qwen2_bias",
        ),
        _ => (
            FUSED_QKV_MQ3G256_LLOYD_QWEN2_BIAS_SRC,
            "fused_qkv_mq3g256_lloyd_qwen2_bias",
        ),
    }
}

/// HFQ8-G256: flat 8-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][256B data] = 264 bytes per 256 weights (1.03 B/w).
pub const GEMV_HFQ8G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq8g256.hip");

/// HFQ6-G256: flat 6-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][192B data] = 200 bytes per 256 weights (0.78 B/w).
/// Packing: 4 weights per 3 bytes (24 bits = 4×6 bits).
pub const GEMV_HFQ6G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq6g256.hip");
pub const GEMV_HFQ6G256_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq6g256_residual.hip");

/// Wave64-native HFQ6-G256 residual GEMV. Mirror of the HFQ4 sibling
/// (`gemv_hfq4g256_residual_wave64.hip`) with 6-bit unpack from
/// `gemv_hfq6g256_residual.hip`. Used for HFQ6/MQ6 `wo` and `w_down`
/// projections on wave64-native arches (gfx906/908/94x). Plan §3.1.1
/// item 2 (gfx906-mq6-mq8-port.md v3.2.1).
pub const GEMV_HFQ6G256_RESIDUAL_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq6g256_residual_wave64.hip");

/// Wave64-native HFQ6-G256 residual GEMV with software-pipelined
/// across-quad weight prefetch. Mirror of `gemv_hfq4g256_residual_wave64_prefetch.hip`.
/// Plan §3.1.1 item 2 / v3.2.2 §5.1 item 1b (the ILP-prefetch lever).
/// Default-on for gfx906 via `gemv_prefetch_enabled(arch)`.
pub const GEMV_HFQ6G256_RESIDUAL_WAVE64_PREFETCH_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq6g256_residual_wave64_prefetch.hip");

/// HFQ5-G256: flat 5-bit with 256-weight groups (168 B/group, 5.25 bpw).
/// Block: [f32 scale][f32 min][160B data] = 168 bytes per 256 weights.
/// Packing: 8 weights per 5 bytes (40 bits = 8x5 bits), cross-byte.
pub const GEMV_HFQ5G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq5g256.hip");
pub const GEMV_HFQ5G256_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq5g256_residual.hip");

/// Wave64-native HFQ5-G256 residual GEMV. Mirror of the HFQ6 sibling with
/// 5-bit unpack from `gemv_hfq5g256_residual.hip`. Used for HFQ5/MQ5 `wo`
/// and `w_down` projections on wave64-native arches (gfx906/908/94x).
pub const GEMV_HFQ5G256_RESIDUAL_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq5g256_residual_wave64.hip");

/// Wave64-native HFQ5-G256 residual GEMV with software-pipelined
/// across-quad weight prefetch. Mirror of the HFQ6 prefetch sibling.
pub const GEMV_HFQ5G256_RESIDUAL_WAVE64_PREFETCH_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq5g256_residual_wave64_prefetch.hip");

/// gfx906 wave64+dp4a fused single-token GEMVs for HFQ6/MQ6 — the
/// Phase A.1c headline lever. Mirror of HFQ4 fused-dp4a family; uses
/// sdot4 with HFQ6's 6-bit unsigned weights (no zp shift correction).
/// Plan §3.1.1 item 3 / v3.2.2 §5.1 item 1c.
pub const FUSED_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_hfq6g256_wave64_dp4a.hip");
pub const FUSED_QKV_HFQ6G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_hfq6g256_wave64_dp4a.hip");
pub const FUSED_QKVZA_HFQ6G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_hfq6g256_wave64_dp4a.hip");

/// Phase A.2 (plan v3.2.3 §5.1 item 2): wave64+dp4a batched residual
/// GEMM for HFQ6/MQ6 prefill. Mirror of `gemm_hfq4g256_wave64_dp4a.hip`
/// with HFQ6 6-bit unpack and `+=` residual write semantic. Used for
/// per-layer wo + w_down at B>1.
pub const GEMM_HFQ6G256_RESIDUAL_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_residual_wave64_dp4a.hip");

/// HFQ4 sibling of `GEMM_HFQ6G256_RESIDUAL_WAVE64_DP4A_SRC` (issue #276,
/// Gap 2 from the HFQ4/HFQ6 dp4a parity audit). Same structural pattern
/// as the HFQ4 lm-head `gemm_hfq4g256_wave64_dp4a` with `+=` residual
/// write semantic. Closes the dispatch gap where MQ4 at gfx906 B>1 below
/// the MMQ cutover (B ∈ [2, 7]) falls to FP16 wave64; the dp4a path wins
/// on per-call ALU and reuses the Q8_1 scratch. Ships BATCH_TILE=16 from
/// the start per the HFQ6 Phase B.1.1 measurement (commit ff9e2105).
pub const GEMM_HFQ4G256_RESIDUAL_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wave64_dp4a.hip");

/// Phase A.3 (plan v3.2.3 §5.1 item 3): wave64+dp4a batched fused
/// GEMMs for HFQ6/MQ6 prefill. Sibling of A.2 with multi-output row
/// routing (qkvza 4-way, qkv 3-way, gate_up 2-way). Overwrite output
/// semantics — caller fuses residual at the wo + w_down sites via
/// gemm_hfq6g256_residual_wave64_dp4a (Phase A.2).
pub const GEMM_QKVZA_HFQ6G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_wave64_dp4a.hip");
pub const GEMM_QKV_HFQ6G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq6g256_wave64_dp4a.hip");
pub const GEMM_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_wave64_dp4a.hip");

/// HFQ4 siblings of the HFQ6 Phase A.3 fused dp4a kernels (issue #276
/// Gap 2 part 2 of 4). Wave64+dp4a batched fused GEMMs at gfx906 for
/// MQ4 prefill at B>1. Close the dispatch fallthroughs where MQ4 today
/// drops to `gemm_*_hfq4g256_fp16_wave64` for the multi-output paths.
/// Ship `BATCH_TILE=16` from the start per HFQ6 Phase B.1.1 (commits
/// 2bee6e6b / ff9e2105). Math identity: signed 4-bit nibble unpack +
/// `zp_eff = zp + 8*sc` matching the HFQ4 lm-head dp4a sibling
/// `gemm_hfq4g256_wave64_dp4a.hip` and the HFQ4 residual dp4a
/// `gemm_hfq4g256_residual_wave64_dp4a.hip`.
pub const GEMM_QKVZA_HFQ4G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_wave64_dp4a.hip");
pub const GEMM_QKV_HFQ4G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_wave64_dp4a.hip");
pub const GEMM_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wave64_dp4a.hip");

/// HFQ3-G256: flat 3-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][96B data] = 104 bytes per 256 weights (0.41 B/w).
/// Packing: 8 weights per 3 bytes (24 bits = 8×3 bits).
pub const GEMV_HFQ3G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq3g256.hip");
pub const GEMV_HFQ3G256_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq3g256.gfx1100.hip");
pub const GEMV_HFQ3G256_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq3g256_residual.hip");
pub const GEMV_HFQ3G256_RESIDUAL_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq3g256_residual.gfx1100.hip");
pub const GEMV_HFQ3G128_SRC: &str = include_str!("../../../kernels/src/gemv_hfq3g128.hip");
pub const GEMV_MQ4G256_SRC: &str = include_str!("../../../kernels/src/gemv_mq4g256.hip");
pub const MQ_ROTATE_X_GFX942_SRC: &str =
    include_str!("../../../kernels/src/mq_rotate_x.gfx942.hip");
pub const GEMV_MQ4G128_SRC: &str = include_str!("../../../kernels/src/gemv_mq4g128.hip");
pub const GEMV_MQ8G256_SRC: &str = include_str!("../../../kernels/src/gemv_mq8g256.hip");
/// MQ6-G256 GEMV: FWHT-rotated HFQ6 (6-bit, 200 B/group). Uses pre-rotated x.
pub const GEMV_MQ6G256_SRC: &str = include_str!("../../../kernels/src/gemv_mq6g256.hip");
/// MQ6-G256 batched GEMM for pre-rotated x. Scalar baseline mirroring
/// gemm_hfq4g256.hip (BATCH_TILE=8, 32 threads/row, __shfl_down reduction).
/// Used by gemma4 `proj_gemm_batched` for Promote6 tensors (v_proj, down_proj).
pub const GEMM_MQ6G256_SRC: &str = include_str!("../../../kernels/src/gemm_mq6g256.hip");
/// MQ5-G256 GEMV: FWHT-rotated HFQ5 (5-bit, 168 B/group). Uses pre-rotated x.
pub const GEMV_MQ5G256_SRC: &str = include_str!("../../../kernels/src/gemv_mq5g256.hip");
pub const FUSED_RMSNORM_MQ_ROTATE_SRC: &str =
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate.hip");
pub const FUSED_RMSNORM_MQ_ROTATE_WAVEGRID_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate_wavegrid.gfx1100.hip");
pub const RMSNORM_REDUCE_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/rmsnorm_reduce.gfx1100.hip");
pub const ROTATE_WITH_RMS_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/rotate_with_rms.gfx1100.hip");
pub const FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate_vecsum.gfx1100.hip");
pub const FUSED_RMSNORM_MQ_ROTATE_VECSUM_SIGN_LDS_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_RMSNORM_SIGN_LDS 1\n",
    "#define HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL fused_rmsnorm_mq_rotate_vecsum_sign_lds\n",
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate_vecsum.gfx1100.hip")
);
pub const FUSED_RMSNORM_MQ_ROTATE_VECSUM_SIGN_CONST_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_RMSNORM_SIGN_CONST 1\n",
    "#define HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL fused_rmsnorm_mq_rotate_vecsum_sign_const\n",
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate_vecsum.gfx1100.hip")
);
pub const FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC: &str =
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate_awq.hip");

pub const RMSNORM_REDUCE_GFX942_SRC: &str =
    include_str!("../../../kernels/src/rmsnorm_reduce.gfx942.hip");
pub const ROTATE_WITH_RMS_GFX942_SRC: &str =
    include_str!("../../../kernels/src/rotate_with_rms.gfx942.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MFMA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mfma.gfx942.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MFMA_V2_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mfma_v2.gfx942.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MFMA_V3_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mfma_v3.gfx942.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MFMA_V4_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mfma_v4.gfx942.hip");
pub const FUSED_SILU_MUL_MQ_ROTATE_SRC: &str =
    include_str!("../../../kernels/src/fused_silu_mul_mq_rotate.hip");
pub const GATED_NORM_MQ_ROTATE_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gated_norm_mq_rotate.gfx1100.hip");
pub const GATED_NORM_MQ_ROTATE_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL gated_norm_mq_rotate_gfx1151\n",
    include_str!("../../../kernels/src/gated_norm_mq_rotate.gfx1100.hip")
);
/// Phase A Stage A — F2: AWQ-aware variant of `mq_rotate_x` for the
/// post-projection input-rotate path (o_proj / out_proj inputs). Dispatched
/// when the upcoming linear carries an `awq_scale` sidecar. Math:
/// (W·s) · (x/s) = W·x — divide before FWHT mirrors the offline pre-scaling.
pub const ROTATE_X_MQ_AWQ_SRC: &str = include_str!("../../../kernels/src/rotate_x_mq_awq.hip");
/// Phase A Stage A — F2: AWQ-aware variant of `fused_silu_mul_mq_rotate`
/// for the down_proj / w_down input stage. Dispatched when down_proj
/// carries an `awq_scale`. Divide happens AFTER silu*up reduction, BEFORE
/// signs1 gather and FWHT.
pub const FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC: &str =
    include_str!("../../../kernels/src/fused_silu_mul_mq_rotate_awq.hip");
pub const FUSED_SILU_MUL_MQ_ROTATE_AWQ_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/fused_silu_mul_mq_rotate_awq_indexed.hip");

/// HFP4-G32 GEMV — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale).
/// v1 correctness anchor: no WMMA, no FP8, no rotation. See docs/quant-formats/hfp4.md.
/// Block: per-row 16 B header (row_scale_a:f16, row_scale_b:f16, block_count, flags),
/// then (K/32) blocks × 17 B (UE8M0:u8 + 16 B nibbles).
pub const GEMV_HFP4G32_SRC: &str = include_str!("../../../kernels/src/gemv_hfp4g32.hip");
pub const GEMV_MFP4G32_LLOYD_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_lloyd.hip");
pub const GEMV_MFP4G32_P_SRC: &str = include_str!("../../../kernels/src/gemv_mfp4g32_p.hip");
pub const GEMV_MFP4G32_E8_SRC: &str = include_str!("../../../kernels/src/gemv_mfp4g32_e8.hip");
/// mfp3-E8 dense GEMV — 3-bit E8 lattice, 13-byte blocks, 104-byte groups.
pub const GEMV_MFP3G32_E8_SRC: &str = include_str!("../../../kernels/src/gemv_mfp3g32_e8.hip");
/// gfx1151-specific mfp4-E8 GEMV with coalesced LDS-staged group loads.
/// ONLY dispatched on gfx1151 (Strix Halo); all other archs use GEMV_MFP4G32_E8_SRC.
pub const GEMV_MFP4G32_E8_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8.gfx1151.hip");
/// mfp4-E8 SoA GEMV — generic fallback for non-gfx1151 arches.
/// Reads the SoA layout (flag=0x06); bit-exact output vs AoS.
pub const GEMV_MFP4G32_E8_SOA_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.hip");
/// Exact-gfx1201 E8 SoA GEMV with value-identical branchless lattice decode.
/// No adjacent architecture selects this source.
pub const GEMV_MFP4G32_E8_SOA_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.gfx1201.hip");
/// gfx1151-specific mfp4-E8 SoA GEMV — fully-coalesced 128B codeword reads.
/// ONLY dispatched on gfx1151 (Strix Halo); other archs use GEMV_MFP4G32_E8_SOA_SRC.
pub const GEMV_MFP4G32_E8_SOA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.gfx1151.hip");
/// gfx942 wave64 mfp4-E8 SoA GEMV. One wave computes two rows as isolated
/// 32-lane segments; dispatch is chip-strict so no RDNA/Qwen route sees it.
pub const GEMV_MFP4G32_E8_SOA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.gfx942.hip");
/// Experimental gfx942 two-wave workgroup variant of the E8-SoA GEMV.
/// Micro-screen only: no product dispatch selects this source.
pub const GEMV_MFP4G32_E8_SOA_W128_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_w128.gfx942.hip");
/// Experimental fixed two-job gfx942 E8-SoA GEMV. Each job has a separate
/// weight/output allocation but shares one activation. Micro-screen only.
pub const GEMV_MFP4G32_E8_SOA_PAIR_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_pair.gfx942.hip");
/// Experimental gfx942 FP8-MFMA lowering of the MFP4-E8 SoA decode GEMV.
/// Default-off: used for channel/roofline screening before any product route.
pub const GEMV_MFP4G32_E8_SOA_FP8_MFMA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_fp8_mfma.gfx942.hip");
/// gfx1151 grouped E8-SoA GEMV for DeepSeek4's eight O-LoRA blocks.
pub const GEMV_MFP4G32_E8_SOA_GROUPED_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped.gfx1151.hip");
/// Exact-gfx1201 grouped E8-SoA GEMV for DeepSeek4's eight O-LoRA blocks.
pub const GEMV_MFP4G32_E8_SOA_GROUPED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped.gfx1201.hip");
pub const GEMV_MFP4G32_E8_SOA_MIXED_JOBS_GFX1201_SRC: &str = concat!(
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.gfx1201.hip"),
    "\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_mixed_jobs.gfx1201.hip")
);
/// Experimental exact-gfx1201 two-projection GEMV that shares activation
/// loads while preserving each incumbent projection's accumulation tree.
pub const GEMV_MFP4G32_E8_SOA_SHARED_PAIR_GFX1201_SRC: &str = concat!(
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.gfx1201.hip"),
    "\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_shared_pair.gfx1201.hip")
);
/// Experimental exact-gfx1201 four-group prefetch variant. The two incumbent
/// accumulation chains and reduction tree are unchanged.
pub const GEMV_MFP4G32_E8_SOA_PREFETCH4_GFX1201_SRC: &str = concat!(
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.gfx1201.hip"),
    "\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_prefetch4.gfx1201.hip")
);
/// Experimental exact-gfx1201 late-scale E8 variant. Mathematically
/// equivalent, but changes FP32 grouping and therefore owes product parity.
pub const GEMV_MFP4G32_E8_SOA_LATE_SCALE_GFX1201_SRC: &str = concat!(
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa.gfx1201.hip"),
    "\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_late_scale.gfx1201.hip")
);
/// Exact-gfx1100 build of the arithmetic-identical grouped E8 O-LoRA kernel.
pub const GEMV_MFP4G32_E8_SOA_GROUPED_GFX1100_SRC: &str = concat!(
    "#define gemv_mfp4g32_e8_soa_grouped_gfx1151 gemv_mfp4g32_e8_soa_grouped_gfx1100\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_SHARED_JOBS_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_shared_jobs.gfx1100.hip");
pub const GEMM_MFP4G32_E8_SOA_WMMA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma.gfx1151.hip");
pub const GEMM_MFP4G32_E8_SOA_WMMA_B2_GFX1151_SRC: &str = concat!(
    "#define E8_PREFILL_BATCH_TILES 2\n",
    "#define E8_PREFILL_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b2_gfx1151\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma.gfx1151.hip")
);
pub const GEMM_MFP4G32_E8_SOA_WMMA_B4_GFX1151_SRC: &str = concat!(
    "#define E8_PREFILL_BATCH_TILES 4\n",
    "#define E8_PREFILL_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b4_gfx1151\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma.gfx1151.hip")
);
pub const GEMM_MFP4G32_E8_SOA_WMMA_GFX1201_SRC: &str = concat!(
    "#define E8_PREFILL_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_gfx1201\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma.gfx12.hip")
);
pub const GEMM_MFP4G32_E8_SOA_WMMA_B2_GFX1201_SRC: &str = concat!(
    "#define E8_PREFILL_BATCH_TILES 2\n",
    "#define E8_PREFILL_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b2_gfx1201\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma.gfx12.hip")
);
pub const GEMM_MFP4G32_E8_SOA_WMMA_B4_GFX1201_SRC: &str = concat!(
    "#define E8_PREFILL_BATCH_TILES 4\n",
    "#define E8_PREFILL_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b4_gfx1201\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma.gfx12.hip")
);
pub const GEMM_MFP4G32_E8_SOA_WMMA_B8_GFX1201_SRC: &str = concat!(
    "#define E8_PREFILL_WIDE_TILES 8\n",
    "#define E8_PREFILL_WIDE_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b8_gfx1201\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma_wide.gfx12.hip")
);
pub const GEMM_MFP4G32_E8_SOA_WMMA_B16_GFX1201_SRC: &str = concat!(
    "#define E8_PREFILL_WIDE_TILES 16\n",
    "#define E8_PREFILL_WIDE_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b16_gfx1201\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma_wide.gfx12.hip")
);

pub const GEMM_MFP4G32_E8_SOA_WMMA_COOP4_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma_coop.gfx1151.hip");
pub const GEMM_MFP4G32_E8_SOA_WMMA_COOP16_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma_coop16.gfx1151.hip");
pub const GEMM_MFP4G32_E8_SOA_WMMA_B8_GFX1151_SRC: &str = concat!(
    "#define E8_PREFILL_WIDE_TILES 8\n",
    "#define E8_PREFILL_WIDE_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b8_gfx1151\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma_wide.gfx1151.hip")
);
pub const GEMM_MFP4G32_E8_SOA_WMMA_B16_GFX1151_SRC: &str = concat!(
    "#define E8_PREFILL_WIDE_TILES 16\n",
    "#define E8_PREFILL_WIDE_KERNEL_NAME gemm_mfp4g32_e8_soa_wmma_b16_gfx1151\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_wmma_wide.gfx1151.hip")
);
pub const GEMM_MFP4G32_E8_SOA_GROUPED_WMMA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_grouped_wmma.gfx1151.hip");
pub const GEMM_MFP4G32_E8_SOA_GROUPED_WMMA_B2_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_PREFILL_BATCH_TILES 2\n",
    "#define E8_GROUPED_PREFILL_KERNEL_NAME gemm_mfp4g32_e8_soa_grouped_wmma_b2_gfx1151\n",
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_grouped_wmma.gfx1151.hip")
);
pub const GEMM_MFP4G32_E8_SOA_GROUPED_WMMA_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_soa_grouped_wmma.gfx12.hip");
/// SoA E8 GEMV 4-way / 8-way unroll variants — bench experiments for the gfx1100
/// cache-roofline MLP sweep (more codewords in flight per wave).
pub const GEMV_MFP4G32_E8_SOA_U4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_u4.hip");
/// Accepted gfx1151 temporal-buffer route. The cpol20/cpol22 variants remain
/// diagnostic controls after regressing the large DS4 tensor shapes.
pub const GEMV_MFP4G32_E8_SOA_U4_BUFFER_CPOL0_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define E8_SOA_U4_BUFFER_KERNEL gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_u4_buffer.hip")
);
/// gfx1151 E8-SoA U4 buffer GEMV with `sqrt(softplus(.))` fused into the store.
///
/// The DeepSeek V4 MoE router is `gate.weight @ x` (M = 256 experts, K = hidden)
/// followed by `sqrt_softplus_f32` over the 256 logits. In the retained ds4
/// route the GEMV carries a NEGATIVE marginal — it is already fully hidden —
/// while the standalone activation costs 1.43 ms/token on a 1x1x1 grid that is
/// almost entirely launch and drain. The reduced row sum is in a register at
/// the store, so folding the activation there moves exposed work into a kernel
/// that already runs for free, and deletes a launch plus its dependent boundary.
///
/// Fusing into the CONSUMER was tried (G8.1, `deepseek4_moe_topk_bias_aware`)
/// and lost 0.48% — that merged two 8-wave kernels into one 8-wave kernel.
/// Folding into the producer is the opposite transform.
/// gfx1151 grouped E8 GEMV with buffer-SRD weight loads and the gfx12 weight
/// cache policy — the treatment the main E8 decode GEMV already has.
///
/// The shipped grouped kernel fetches weights with plain global loads: its code
/// object reports `buffer_loads = 0` against the main kernel's 11, and it
/// achieves 143 GB/s at 8192 waves (6.4 fills) where the wave sweep says ~192
/// is available at that occupancy. That is the accepted B2 cache-policy lever
/// (+2.94% on the rest of the dense route) never having been applied here.
///
/// Bit-exact against the plain variant: only the fetch path changes, not the
/// arithmetic or its order.
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define E8_GROUPED_BUFFER_LOADS 1\n",
    "#define GEMV_MFP4G32_E8_SOA_GROUPED_KERNEL gemv_mfp4g32_e8_soa_grouped_buffer_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped.gfx1151.hip")
);
/// Batched E8-SoA decode GEMV for small-batch (speculative verify) forwards.
/// One wave per output row, weight row read once, applied to B activation
/// vectors. See `kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip` for why
/// the WMMA prefill GEMM is the wrong tool below B=16.
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B1_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 1\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b1_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B2_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 2\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b2_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B3_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 3\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b3_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_PAIR_B3_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 3\n",
    "#define E8_BATCHED_PAIR 1\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_pair_b3_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_PACK_B3_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 3\n",
    "#define E8_BATCHED_PACK 1\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_pack_b3_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B4_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 4\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b4_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B5_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 5\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b5_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B6_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 6\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b6_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B7_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 7\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b7_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B8_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 8\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b8_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_BATCHED_B16_GFX1151_SRC: &str = concat!(
    "#define E8_BATCHED_B 16\n",
    "#define E8_BATCHED_KERNEL gemv_mfp4g32_e8_soa_batched_b16_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_batched.gfx1151.hip")
);
/// Batched grouped E8-SoA decode GEMV: `A[G,M,K] @ X[B,G,K] -> Y[B,G,M]`.
/// One wave per (group, row), weight row read once, applied to B activation
/// vectors. Replaces `gemm_mfp4g32_e8_soa_grouped_wmma` below B=16, where the
/// grouped WMMA computes a full 16-token tile for however many tokens are
/// real. See `kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip`.
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B1_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 1\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b1_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B2_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 2\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b2_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B3_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 3\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b3_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B4_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 4\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b4_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B5_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 5\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b5_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B6_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 6\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b6_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B7_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 7\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b7_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B8_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 8\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b8_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_GROUPED_BATCHED_B16_GFX1151_SRC: &str = concat!(
    "#define E8_GROUPED_BATCHED_B 16\n",
    "#define E8_GROUPED_BATCHED_KERNEL gemv_mfp4g32_e8_soa_grouped_batched_b16_gfx1151\n",
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_batched.gfx1151.hip")
);
pub const GEMV_MFP4G32_E8_SOA_U4_BUFFER_SQRT_SOFTPLUS_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define E8_SOA_U4_BUFFER_KERNEL gemv_mfp4g32_e8_soa_u4_buffer_sqrt_softplus_gfx1151\n",
    "#define E8_SOA_U4_BUFFER_EPILOGUE(v) e8_soa_u4_sqrt_softplus(v)\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_u4_buffer.hip")
);
pub const GEMV_MFP4G32_E8_SOA_U4_BUFFER_CPOL20_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 20\n",
    "#define E8_SOA_U4_BUFFER_KERNEL gemv_mfp4g32_e8_soa_u4_buffer_cpol20_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_u4_buffer.hip")
);
pub const GEMV_MFP4G32_E8_SOA_U4_BUFFER_CPOL22_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 22\n",
    "#define E8_SOA_U4_BUFFER_KERNEL gemv_mfp4g32_e8_soa_u4_buffer_cpol22_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_u4_buffer.hip")
);
pub const GEMV_MFP4G32_E8_SOA_U8_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_u8.hip");
/// SoA E8 GEMV decode probes: STRIP (lattice decode removed → loads+scale+reduce
/// ceiling) and LUT (2D constant-LUT coordinate conversion). Cache-regime decode-
/// wall diagnostics for the gfx1100 roofline chase.
pub const GEMV_MFP4G32_E8_SOA_STRIP_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_strip.hip");
pub const GEMV_MFP4G32_E8_SOA_LUT_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_lut.hip");
/// gfx1151-specific fused gate+up mfp4-E8 decode GEMV (2-way launch-fusion).
/// ONLY dispatched on gfx1151 (Strix Halo) via the steps.rs guard; embeds the
/// byte-identical gemv_mfp4g32_e8_gfx1151 per-row body. No bleed to other archs.
pub const FUSED_GATE_UP_MFP4G32_E8_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_mfp4g32_e8.gfx1151.hip");
/// gfx1151-specific fused QKVZA mfp4-E8 decode GEMV (4-way launch-fusion) for
/// the Qwen3.5 DeltaNet LA preamble. ONLY dispatched on gfx1151.
pub const FUSED_QKVZA_MFP4G32_E8_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_mfp4g32_e8.gfx1151.hip");
/// DIAGNOSTIC ONLY (env HIPFIRE_E8_STRIP=1): compute-stripped E8 GEMV that keeps
/// the exact memory access but guts the decode — measures the compute ceiling.
/// Output is garbage; for tok/s probing the memory-vs-compute bound only.
pub const GEMV_MFP4G32_E8_STRIP_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_strip.gfx1151.hip");
/// EXPERIMENT (env HIPFIRE_E8_LDSX=1): E8 GEMV with LDS-staged x + 4 rows/block
/// (memory-level-parallelism lever). Bit-exact with gemv_mfp4g32_e8_gfx1151.
pub const GEMV_MFP4G32_E8_LDSX_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_ldsx.gfx1151.hip");
/// gfx1151 mfp4-E8 grouped MoE gate_up (k8 indexed) — mq4-parity expert kernel.
pub const GEMV_MFP4G32_E8_MOE_GATE_UP_K8_INDEXED_BATCHED_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched.gfx1151.hip");
/// gfx1151 mfp4-E8 grouped MoE down (k8 indexed, atomic-free expanded).
pub const GEMV_MFP4G32_E8_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_GFX1151_SRC: &str = include_str!(
    "../../../kernels/src/gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded.gfx1151.hip"
);
/// gfx11_dgpu (RDNA3 discrete, e.g. gfx1100 RX 7900 XTX) twin of the gfx1151 E8 MoE
/// gate_up kernel. Uses a 4-way group unroll with 4 independent accumulators +
/// __launch_bounds__(32,4) to raise VGPR budget to ~94 (10 waves/SIMD) and expose
/// 4× independent load streams per iteration, hiding GDDR6 latency.
/// Guard: arch_caps.is_rdna3_dgpu(); kill-switch: HIPFIRE_E8_DGPU_TWIN=0.
pub const GEMV_MFP4G32_E8_MOE_GATE_UP_K8_INDEXED_BATCHED_GFX11_DGPU_SRC: &str = include_str!(
    "../../../kernels/src/gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched.gfx11_dgpu.hip"
);
/// SoA-coalesced E8 MoE gate_up (k8 indexed). Reads SoA-laid-out expert weights
/// (scales contiguous, codewords contiguous 16B-aligned) — +38-73% on the GEMV
/// bench by killing the AoS 17B-block cache-line over-fetch on the gfx1100 dGPU.
/// Dispatched iff experts were transposed AoS->SoA at load (HIPFIRE_E8_SOA_EXPERTS=1)
/// on an RDNA3 dGPU. 2-way unroll (max wave32 occupancy).
pub const GEMV_MFP4G32_E8_SOA_MOE_GATE_UP_K8_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mfp4g32_e8_soa_moe_gate_up_k8_indexed_batched.hip");
/// gfx11_dgpu twin of the gfx1151 E8 MoE down (k8 indexed, atomic-free expanded) kernel.
/// Same 4-way unroll strategy as the gate_up twin. Guard + kill-switch: see gate_up twin.
pub const GEMV_MFP4G32_E8_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_GFX11_DGPU_SRC: &str = include_str!(
    "../../../kernels/src/gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded.gfx11_dgpu.hip"
);
/// Batched sigmoid-scaled residual add (generic f32): y[t,:] += sigmoid(scalars[t]) * x[t,:].
/// Folds the Q8 shared-expert down output into the residual in batched MoE prefill.
pub const SIGMOID_SCALED_RESIDUAL_ADD_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/sigmoid_scaled_residual_add_batched.hip");
/// gfx1151 mfp4-E8 grouped-WMMA-GEMM (Path 2 prefill). Amortizes expert-weight
/// reads via the SGLang scatter pipeline + wave32 F16 WMMA with E8 lattice dequant.
pub const GEMM_MFP4G32_E8_MOE_GROUPED_WMMA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_moe_grouped_wmma.gfx1151.hip");
/// gfx12 (RDNA4) sister of GEMM_MFP4G32_E8_MOE_GROUPED_WMMA_GFX1151_SRC. Same
/// dispatch contract and kernarg layout; differs in WMMA intrinsic
/// (`wmma_f32_16x16x16_f16` → `_w32_gfx12` variant) required on gfx1200/gfx1201.
pub const GEMM_MFP4G32_E8_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mfp4g32_e8_moe_grouped_wmma.gfx12.hip");
pub const GEMV_HFP4G32_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfp4g32.gfx1100.hip");
// gfx11 (RDNA3) v_dot2_f32_f16-accelerated decode-path variant.
// Inner loop uses 4 fdot2 ops per K-block (8 K-elts), replacing the
// fallback's 8 F32 mul + 8 F32 fma chain. Activation X consumed as
// FP16 via ensure_fp16_x. Wins biggest on ALU-bound shapes (FFN
// M=11008 measured 40% peak BW on 7900 XTX with fallback — headroom
// to ~2×). Reaches gfx11/RDNA3.5 archs (gfx1100/1101/1102/1150/1151).
pub const GEMV_HFP4G32_DOT2_GFX11_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfp4g32_dot2.gfx11.hip");
// gfx12 (RDNA4) FP8-dot4 decode-path variant. dot4_f32_fp8_fp8 cuts inner-loop
// ALU ~2-2.4× vs the fallback dequant/FMA chain; biggest win on ALU-bound
// small-M attention shapes (k_proj/v_proj at ~16-20% peak BW on R9700).
// Activation X consumed as FP8 (E4M3), pre-packed by `ensure_fp8_x`.
pub const GEMV_HFP4G32_FP8_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfp4g32_fp8.gfx12.hip");

/// HFQ4-G512: flat 4-bit with 512-weight groups.
/// Block: [f32 scale][f32 zero][256B nibbles] = 264 bytes per 512 weights (0.516 B/w).
/// 264B ≈ 1 PCIe TLP, 2 L2 cache lines.
pub const GEMV_HFQ4G512_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g512.hip");

/// HFQ4-G1024: flat 4-bit with 1024-weight groups.
/// Block: [f32 scale][f32 zero][512B nibbles] = 520 bytes per 1024 weights (0.508 B/w).
pub const GEMV_HFQ4G1024_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g1024.hip");

/// HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights.
/// Same coalesced width as Q4_K, 14 VGPRs instead of 39.
pub const GEMV_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.hip");
/// gfx1151 LM-head one-row candidate. Keep wave-uniform HFQ headers on scalar
/// loads, lower lane-divergent packed weights to temporal VMEM, and specialize
/// the hot 248320x2048 shape so the compiler removes dynamic tail control.
/// x-batched HFQ4-G256 GEMV: one weight pass, B activation vectors. Used by
/// the multi-slot decode path for the lm_head, where every slot dots the same
/// 270 MB weight matrix against its own hidden state. `K2048` is set because
/// that is the lm_head shape this is built for; the kernel still reads K at
/// runtime for the non-specialised case.
///
/// Deliberately does NOT set the weight-buffer-load / cache-policy defines the
/// single-vector gfx1151 lm_head variant uses: those tune a pass that reads
/// weights once per x, and this kernel's premise is that the weight pass is
/// amortised over B.
/// General-K companion of `GEMV_HFQ4G256_XBATCH_SRC`. The lm_head build hard-
/// codes `groups_per_row = 8` via `HIPFIRE_HFQ4G256_K2048`; the MoE expert
/// shapes are K=2048 (gate_up) and K=512 (down), so the down side needs the
/// runtime `K / 256`.
pub const GEMV_HFQ4G256_XBATCH_GEN_SRC: &str = concat!(
    "#define HIPFIRE_HFQ4G256_XBATCH 1\n",
    "#define HIPFIRE_HFQ4G256_XBATCH_MAX 4\n",
    "#define HIPFIRE_HFQ4G256_XBATCH_KERNEL gemv_hfq4g256_xbatch_gen\n",
    include_str!("../../../kernels/src/gemv_hfq4g256.hip")
);

pub const GEMV_HFQ4G256_XBATCH_SRC: &str = concat!(
    "#define HIPFIRE_HFQ4G256_XBATCH 1\n",
    "#define HIPFIRE_HFQ4G256_XBATCH_MAX 4\n",
    "#define HIPFIRE_HFQ4G256_K2048 1\n",
    "#define HIPFIRE_HFQ4G256_XBATCH_KERNEL gemv_hfq4g256_xbatch\n",
    include_str!("../../../kernels/src/gemv_hfq4g256.hip")
);

pub const GEMV_HFQ4G256_LM_HEAD_R1_HYBRID_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_HFQ4G256_K2048 1\n",
    "#define HIPFIRE_HFQ4G256_KERNEL gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256.hip")
);
/// gfx1151 LM-head structural experiment: inline F32->FP16 activation packing,
/// packed FP16 HFQ dequant, and four independent v_dot2_f32_f16 operations per
/// group. The distinct symbol/code object is required to keep gfx1100 and
/// gfx12 retained tapes isolated from the gfx1151 numerical contract.
pub const GEMV_HFQ4G256_LM_HEAD_DOT2_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_dot2.gfx1151.hip")
);

// ── RDNA2 (gfx1030) HFQ4-G256 variants ──
// 5 kernel variants exploring the occupancy/unroll/cache tradeoff space.
// Select via HIPFIRE_RDNA2_VARIANT=N env var (default: 1).
// v1: baseline-rdna2 — launch_bounds(32,16), 2x unroll, ~64 VGPRs
// v2: high-occupancy — launch_bounds(32,20), 2x unroll, ~51 VGPRs (scoped vars)
// v3: wide-unroll    — launch_bounds(32,12), 4x unroll, ~85 VGPRs
// v4: dp4a-packed    — launch_bounds(32,16), dp4a intrinsics, factored scale/zero
// v5: cache-aggressive — launch_bounds(32,16), 2x unroll, packed loads, factored math
pub const GEMV_HFQ4G256_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256.gfx1100.hip")
);
pub const GEMV_HFQ4G256_K2048_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    "#define HIPFIRE_HFQ4G256_KERNEL gemv_hfq4g256_k2048\n",
    "#define HIPFIRE_HFQ4G256_K2048 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256.gfx1100.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.hip")
);
/// Radiowave gfx1151 probe: preserve the generic dual-row arithmetic while
/// lowering the weight stream to temporal raw-buffer loads. gfx1151 is not in
/// the gfx1100 dGPU admission set, so this remains inert unless the exact-arch
/// runtime selector chooses it.
pub const GEMV_HFQ4G256_RESIDUAL_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.hip")
);
/// gfx1151 hybrid residual cache lowering. Keep wave-uniform scale and
/// zero-point headers on scalar/global loads; lower only per-lane packed
/// weights to cache-policy-zero buffer VMEM.
pub const GEMV_HFQ4G256_RESIDUAL_HYBRID_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_hybrid_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.hip")
);
/// Exact gfx1151 A3B residual specialization. Preserve the admitted dual-row
/// global-load arithmetic while exposing M=2048/K=4096 to LLVM so all four
/// group quads can be unrolled and scheduled as one gfx1151-only kernel.
pub const GEMV_HFQ4G256_RESIDUAL_K4096_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX1151_RESIDUAL_K4096 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_k4096_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_STAGE_X32_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_RESIDUAL_STAGE_X32 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
/// Radiowave gfx1100 residual probe: retain the certified K=2048 activation
/// schedule while using ordinary temporal raw-buffer loads for weights. This
/// isolates addressing/codegen from the separately rejected SLC policy.
pub const GEMV_HFQ4G256_RESIDUAL_CPOL_RT_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_STAGE_X32 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_cpol_rt\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
/// RT addressing with a matrix-wide SRD and immediate load-consume schedule.
/// The source shape is intended to recover the global kernel's VGPR occupancy
/// while retaining buffer-address generation and temporal cache semantics.
pub const GEMV_HFQ4G256_RESIDUAL_CPOL_RT_LOW_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_STAGE_X32 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_MATRIX_RSRC 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_BUFFER_CONSUME 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_cpol_rt_low\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
/// gfx1151 temporal-buffer residual candidate. Reuse the one-row immediate-
/// consume source shape, but keep an exact-arch artifact and cache-policy-zero
/// VMEM contract instead of inheriting gfx1100 admission.
pub const GEMV_HFQ4G256_RESIDUAL_RT_LOW_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_STAGE_X32 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_MATRIX_RSRC 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_BUFFER_CONSUME 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_rt_low_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
/// Global-load control for the gfx1151 one-row immediate-consume schedule.
/// Kept distinct from the temporal-buffer artifact so scheduling and cache
/// lowering can be admitted independently.
pub const GEMV_HFQ4G256_RESIDUAL_ROW1_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_RESIDUAL_STAGE_X32 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_MATRIX_RSRC 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_BUFFER_CONSUME 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_row1_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
/// Radiowave gfx1100 residual probe: preserve the certified activation staging
/// and reduction order while replacing flat weight loads with SLC raw-buffer
/// loads. Kept as a distinct replay artifact until model-level certification.
pub const GEMV_HFQ4G256_RESIDUAL_CPOL_SLC_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_STAGE_X32 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_cpol_slc\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_K2048_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_RESIDUAL_STAGE_X32 1\n",
    "#define HIPFIRE_RDNA3_RESIDUAL_K2048 1\n",
    "#define HIPFIRE_RESIDUAL_KERNEL gemv_hfq4g256_residual_k2048\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_WEIGHT_CACHE_FLAT_GEMV 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_wave64.hip");
pub const GEMV_HFQ4G256_RESIDUAL_WAVE64_GFX1151_SRC: &str = concat!(
    "// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n",
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_wave64.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_WAVE64_PREFETCH_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_wave64_prefetch.hip");
pub const GEMV_HFQ4G256_RESIDUAL_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx942.hip");
pub const GEMV_HFQ4G256_RESIDUAL_V2_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_v2.gfx942.hip");
pub const GEMV_HFQ4G256_RESIDUAL_V3_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_v3.gfx942.hip");
pub const FUSED_GATE_UP_HFQ4G256_V2_GFX942_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_hfq4g256_v2.gfx942.hip");
pub const FUSED_QKV_HFQ4G256_V2_GFX942_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_hfq4g256_v2.gfx942.hip");
pub const FUSED_QKV_HFQ4G256_V2_GFX942_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_hfq4g256_v2_gfx942_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_hfq4g256_v2.gfx942.hip")
);
pub const FUSED_QKVZA_HFQ4G256_V2_GFX942_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_v2.gfx942.hip");

/// HFQ4-G256 GEMV with fused SCALED residual: y[row] += scale * (A[row] · x).
/// Two flavors in one file: `_cpu` takes `scale` by kernarg, `_gpu` reads it
/// from a 1-element device buffer. Used by the MoE FFN accumulator — the
/// routed-expert variant scales by a CPU top-K weight, and the shared-expert
/// variant scales by an on-device sigmoid gate (no D2H sync).
pub const GEMV_HFQ4G256_RESIDUAL_SCALED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip");
pub const GEMV_HFQ4G256_RESIDUAL_SIGMOID_BUFFER_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip")
);
/// Radiowave gfx1100 shared-down specialization. The hot A3B decode shape is
/// K=512 (two HFQ4 groups), so the compiler can remove the empty quad loop and
/// all dynamic tail predicates while retaining the public kernarg ABI.
pub const GEMV_HFQ4G256_RESIDUAL_SIGMOID_K512_BUFFER_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_SIGMOID_K512 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip")
);
/// Exact gfx1151 shared-down candidate. Fixing K=512 removes the empty quad
/// loop and dynamic tails while temporal raw-buffer loads preserve the K4
/// target's native cache policy instead of inheriting gfx1100's SLC tuning.
pub const GEMV_HFQ4G256_RESIDUAL_SIGMOID_K512_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_RDNA3_SIGMOID_K512 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip")
);
/// Exact M=2048/K=512 sibling with both activation groups hoisted ahead of
/// the two row-weight streams. This probes gfx1100 VMEM/VALU overlap without
/// changing the reduction tree, output mapping, or replay ABI.
pub const GEMV_HFQ4G256_RESIDUAL_SIGMOID_K512_HOIST_X16_BUFFER_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_SIGMOID_K512 1\n",
    "#define HIPFIRE_RDNA3_SIGMOID_M2048 1\n",
    "#define HIPFIRE_RDNA3_SIGMOID_HOIST_X16 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_SIGMOID_ROWS4_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_SIGMOID_ROWS4 1\n",
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_SIGMOID_ROWS4_BUFFER_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    "#define HIPFIRE_RDNA3_SIGMOID_ROWS4 1\n",
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip")
);

/// HFQ6/MQ6-G256 batched GEMV with fused sigmoid-scaled residual:
///   y_batch[bid,row] += sigmoid(c_batch[bid]) * (A[row] · x_batch[bid]).
/// HFQ6 analogue of `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` —
/// unlocks the batched MoE-FFN shared-expert `down` projection for the
/// AWQ-style mixed-precision path where shared.down is MQ6 (storage-
/// compatible with HFQ6G256, 200 B / group of 256).
pub const GEMV_HFQ6G256_RESIDUAL_SIGMOID_SCALED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq6g256_residual_sigmoid_scaled.hip");

/// HFQ5G256 sibling of the sigmoid-scaled batched residual GEMV. Same
/// shape; reads the 168 B/group 5-bit layout. Used for the batched MoE-FFN
/// shared-expert `down` projection where shared.down is MQ5.
pub const GEMV_HFQ5G256_RESIDUAL_SIGMOID_SCALED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq5g256_residual_sigmoid_scaled.hip");

/// MoE fused gate_up GEMV: runs 8 top-K experts' HFQ4-G256 GEMV in one
/// launch. Grid.y is the expert rank (0..7); each block selects its
/// expert's weight base from the W0..W7 kernarg array and runs the
/// standard HFQ4G256 body. Saves 7 launches per MoE layer.
pub const GEMV_HFQ4G256_MOE_GATE_UP_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up.hip");

/// MoE fused down GEMV with scaled residual: 8 experts' weighted
/// contributions accumulate into a single residual buffer via atomicAdd
/// in one kernel launch. Grid.y selects the expert. Saves 7 launches
/// per MoE layer.
pub const GEMV_HFQ4G256_MOE_DOWN_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down.hip");

/// GPU softmax + top-K + (optional) renormalize for the MoE router.
/// Reads [n_exp] logits, writes [k] indices and [k] weights to device
/// buffers. Eliminates the per-layer D2H sync the CPU-side top-K used
/// to need — required for hipGraph capture of MoE decode.
pub const MOE_SOFTMAX_TOPK_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_softmax_topk_k8.hip");
pub const MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/moe_router_softmax_topk_k8_wave64.hip");
pub const MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SRC: &str =
    include_str!("../../../kernels/src/moe_router_softmax_topk_k8_wave64_exact.hip");
pub const MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SHARED_SILU_MQ_ROTATE_SRC: &str = concat!(
    "#define HIPFIRE_ROUTER_SHARED_SILU_MQ_ROTATE 1\n",
    "#define HIPFIRE_ROUTER_EXACT_KERNEL moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate\n",
    include_str!("../../../kernels/src/moe_router_softmax_topk_k8_wave64_exact.hip")
);

/// MoE top-K + renorm only, given pre-softmaxed probs. Companion to
/// the regular softmax_f32 kernel; the dispatch site runs softmax_f32
/// first, then this kernel for top-K + renorm. Avoids the 1-ULP
/// precision divergence that the fused softmax+topk variant exhibits
/// on MQ4 MoE: in-kernel softmax order + mul-by-reciprocal renorm
/// produced weights that differed from gpu.softmax_f32 + manual
/// division by 1 LSB per element, which compounds to a structural
/// attractor on Qwen3.5-A3B / 122B-A10B.
pub const MOE_TOPK_RENORM_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_topk_renorm_k8.hip");

/// Batched companion of MOE_TOPK_RENORM_K8_SRC for the prefill path.
/// Same per-block algorithm; one workgroup per token row.
pub const MOE_TOPK_RENORM_K8_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/moe_topk_renorm_k8_batched.hip");

/// Index-aware MoE gate_up GEMV — reads expert IDs from a device-side
/// topk_indices buffer and the per-expert weight base from an
/// expert-pointers table. hipGraph-capture-safe replacement for the
/// kernarg-pointer variant.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC: &str = concat!(
    // gfx1151 product: current-toolchain private0-capable occupancy dial.
    "#if defined(__gfx1151__)\n",
    "#define HIPFIRE_MOE_GATE_UP_MIN_BLOCKS 12\n",
    "#endif\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// Exact gfx1100 A3B decode specialization: K=2048 fixes the group count at
/// eight while preserving the base gate/up load, FMA, and reduction order.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);
/// Exact gfx1151 A3B specialization: expose the fixed eight-group K=2048
/// contract to LLVM while retaining ordinary global loads and the generic
/// wave32 schedule. This isolates code shape from raw-buffer cache policy.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1151 structural gate producer for MQ4R A3B decode. Gate and up are
/// intentionally compiled as separate fixed-K=2048 kernels so Redline can
/// overlap their independent weight streams on retained PM4 queues.
pub const GEMV_HFQ4G256_MOE_GATE_INDEXED_K2048_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_MOE_PROJECTION_KERNEL gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151\n",
    "#define HIPFIRE_MOE_PROJECTION 0\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_projection_indexed_gfx1151.hip")
);

/// gfx1151 structural up producer. See the gate producer above; the only
/// source-level distinction is the projection row selected from the packed
/// gate_up expert allocation.
pub const GEMV_HFQ4G256_MOE_UP_INDEXED_K2048_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_MOE_PROJECTION_KERNEL gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151\n",
    "#define HIPFIRE_MOE_PROJECTION 1\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_projection_indexed_gfx1151.hip")
);

/// gfx1151 paired-wave producer: one 64-thread workgroup owns one output row,
/// with wave 0 streaming gate weights and wave 1 streaming up weights. This
/// preserves the low-register per-projection schedule without adding replay
/// nodes or a cross-queue join.
pub const GEMV_HFQ4G256_MOE_GATE_UP_PAIRED_WAVES_K2048_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_MOE_PROJECTION_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151\n",
    "#define HIPFIRE_MOE_PAIRED_WAVES 1\n",
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_projection_indexed_gfx1151.hip")
);

/// gfx1151 persistent rank tile: eight waves share one LDS activation, retain
/// one expert route each, and walk multiple rows with fused gate/up ownership.
pub const GEMV_HFQ4G256_MOE_GATE_UP_PERSISTENT_RANK8_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_persistent_rank8.gfx1151.hip")
);

/// gfx1151 occupancy probe: one-group-at-a-time activation/header live range,
/// fixed K=2048, and ordinary global loads. Kept separate from gfx1100's
/// admitted buffer/cache policy so one ISA cannot silently tune the other.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_LOW_VGPR_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_MOE_GATE_UP_LOW_VGPR 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_low_vgpr_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);
/// gfx1151 middle-live-range probe: retain two groups of activation/header
/// state at once, fixed K=2048, with ordinary global loads.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_PAIR_VGPR_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_MOE_GATE_UP_PAIR_VGPR 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_vgpr_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);
/// gfx1151 Radiowave candidate: the two-group live range paired with temporal
/// raw-buffer weight loads. The pair schedule aims to retain enough MLP while
/// avoiding the register growth of the four-group buffer lowering.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_PAIR_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_MOE_GATE_UP_PAIR_VGPR 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1151 Radiowave pair schedule with all decode inputs on temporal VMEM.
/// Two groups stay live to preserve useful memory-level parallelism while the
/// activation vectors form a four-request B128 clause before both weight
/// streams.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_PAIR_ALL_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_MOE_GATE_UP_PAIR_VGPR 1\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_all_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);
/// Exact gfx1151 A3B candidate: expose K=2048 to LLVM and use temporal
/// raw-buffer weight loads without importing gfx1100 cache or scheduling
/// policy.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// Task-1 gfx1151 gate/up candidate: one wave32 row per workgroup, exact
/// K=2048, typed plain B64 scale/zero headers, and temporal raw-buffer loads
/// only for lane-divergent packed weights. The hot A3B shape owns exactly
/// 512x8 workgroups without inactive rows.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_HYBRID_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_GFX1151_MOE_GATE_UP_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1151 Radiowave all-input cache candidate: retain the fixed-K gate/up
/// schedule while routing both weights and aligned activation vectors through
/// temporal VMEM. This mirrors the proven geometry/reduction lowering without
/// changing arithmetic or grid policy.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_ALL_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1151 bandwidth candidate: keep weights, activations, the freshly
/// produced top-k index, and the expert-pointer lookup on temporal VMEM. The
/// two uniform routing loads are scalarized with readfirstlane, allowing the
/// replay boundary to invalidate vector caches without flushing scalar data.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_K2048_ROUTE_ALL_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_K2048 1\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER 1\n",
    "#define HIPFIRE_RDNA3_MOE_GATE_UP_ROUTE_BUFFER 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_route_all_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1100 K=8 scheduling experiment: linearize the two-dimensional grid with
/// expert rank as the fastest-moving coordinate. Arithmetic stays byte-for-byte
/// identical while adjacent workgroups share the activation working set and
/// expose eight independent expert-weight streams to the scheduler.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_RANK_INTERLEAVE_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1100 low-register schedule: process one HFQ group at a time while
/// retaining the base kernel's four accumulator chains and temporal buffer
/// loads. This shortens activation/header live ranges without changing math.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_LOW_VGPR_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_MOE_GATE_UP_LOW_VGPR 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1100 two-group schedule with the product-certified SLC weight policy.
/// This sits between the 96-VGPR four-group default and the rejected
/// one-group low-VGPR arm, retaining two independent memory streams.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_PAIR_SLC_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_MOE_GATE_UP_PAIR_VGPR 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// Radiowave gfx1100 cache-policy probes. LLVM's raw-buffer auxiliary bits are
/// GLC=1, SLC=2, and DLC=4 on gfx10/gfx11. Keep them as distinct artifacts so
/// a model-level exact shadow can select policy without runtime ambiguity.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_GLC_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 1\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_SLC_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_DLC_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 4\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// gfx1100 scheduler-packing experiment: two independent wave32 rows per
/// workgroup. Each wave preserves the base kernel's gate/up accumulator and
/// shuffle order; there is no cross-wave reduction or extra row accumulator.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WG2_SRC: &str = concat!(
    "#define HIPFIRE_MOE_GATE_UP_WG_WAVES 2\n",
    "#define HIPFIRE_MOE_GATE_UP_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_wg2\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip")
);

/// NUM_ROWS register row-tile of the indexed MoE gate_up GEMV (opt-in,
/// HIPFIRE_MOE_GATE_UP_FUSED=1). Each block owns NUM_ROWS output rows/expert,
/// reusing x across them; grid.x = ceil(M/NUM_ROWS). Token-id exact vs the base.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_ROWTILE_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed_rowtile.hip");

/// HFQ4G128 (ParoQuant) variant of the indexed MoE gate_up GEMV. Same
/// device-side expert-pointer table + topk_indices contract as the
/// HFQ4G256 sibling; closes the residual hipGraph-capture gap for
/// ParoQuant routed experts (Qwen3.6-A3B-PARO etc.) left open by
/// PR #317.
pub const GEMV_PARO_Q4G128_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_paro_q4g128_moe_gate_up_indexed.hip");

/// N-batched indexed MoE gate_up GEMV for HFQ4G128 (ParoQuant routed
/// experts). Sister of `GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_SRC` with
/// 72 B/group stride. Used by Path 1 fallback in
/// `prefill_moe_ffn_body_batched` when ParoQ4G128 expert weights are admitted
/// (HIPFIRE_PARO_BATCHED=1) on non-WMMA archs (CDNA/gfx10). gfx11/gfx12 takes
/// Path 2's grouped-WMMA kernel instead (Phase 4).
pub const GEMV_PARO_Q4G128_MOE_GATE_UP_K8_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_paro_q4g128_moe_gate_up_k8_indexed_batched.hip");

/// CDNA3 (MI300X / gfx94x) wave64-native counterpart to the indexed
/// gate_up GEMV. Block=[64,1,1] with 2 rows per block (one per warp) —
/// halves the grid count vs the wave32 variant, which otherwise wastes
/// half a wave64 per workgroup. Byte-exact math.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed_wave64.hip");
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_GFX1151_SRC: &str = concat!(
    "// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n",
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed_wave64.hip")
);

/// Index-aware MoE down GEMV — same pattern as the indexed gate_up,
/// also reads scales from a device topk_weights buffer. Pairs with the
/// GPU top-K kernel to make MoE decode hipGraph-capturable end-to-end.
pub const GEMV_HFQ4G256_MOE_DOWN_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_indexed.hip");

/// CDNA3 (MI300X / gfx94x) wave64-native counterpart to the indexed
/// down-residual GEMV. Same 2-rows-per-block packing as the gate_up
/// wave64 variant; atomicAdd semantics preserved per (row, krank).
pub const GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_indexed_wave64.hip");

/// N-batched MoE router softmax + top-8 + renorm. Drop-in replacement
/// for the single-token kernel when prefilling N tokens through an MoE
/// layer; one workgroup per token. Enables batched MoE prefill.
pub const MOE_SOFTMAX_TOPK_K8_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/moe_softmax_topk_k8_batched.hip");

/// N-batched indexed MoE gate_up GEMV. Extends the single-token indexed
/// variant with a batch dimension (grid.z = N). Each (token, k-slot)
/// block picks its own expert via topk_indices[token×K_TOP + slot] and
/// reads the token's x row from x[token×K..].
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed_batched.hip");

/// CDNA3 wave64-native batched indexed MoE gate_up. 2 rows per block.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed_batched_wave64.hip");

/// N-batched indexed MoE down + scaled residual. Mirrors the batched
/// gate_up: grid.z = N, per-token routing + scaling, atomicAdd into
/// x_residual[token×M..].
pub const GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_indexed_batched.hip");

/// CDNA3 wave64-native batched indexed MoE down. 2 rows per block.
pub const GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_indexed_batched_wave64.hip");

/// Atomic-free batched indexed MoE down — writes per-(token, krank) row into
/// an expanded [N × K_TOP × M] output buffer instead of atomicAdd'ing into
/// a shared residual row. Pairs with `MOE_DOWN_COMBINE_K8_BATCHED_SRC`.
/// Observed lift: 387 → ~900 GiB/s on R9700/gfx1201 (no K_TOP-way atomic
/// contention per output cell).
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip")
);

/// Nine-path fused MoE gate_up (routed k=8, decode T=1): one CTA stages the
/// activation into LDS once and all 8 routed-expert warps share it, replacing
/// the per-(row,krank) x restaging of `gemv_hfq4g256_moe_gate_up_k8_indexed`.
/// Byte-exact per-row math with the indexed kernel. See kernel header.
pub const GEMV_HFQ4G256_MOE_NINEPATH_D3_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_ninepath_d3.hip");

/// Nine-path fused MoE down + weighted combine (routed k=8, decode T=1):
/// folds the expanded [8 × down_m] intermediate into LDS (never written to
/// global) and applies the weighted k-ordered fold in the same kernel,
/// replacing the expanded down GEMV + `moe_down_combine_k8_batched` pair.
/// Byte-exact with that pair at down_k=512. See kernel header.
pub const GEMV_HFQ4G256_MOE_NINEPATH_D4_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_ninepath_d4.hip");

/// MQ3-Lloyd codebook port of the nine-path fused MoE down + weighted combine.
/// Stages the rotated activation ONCE for all 8 routed experts (the incumbent
/// re-reads it in each of 16,384 single-wave workgroups) and folds the 8
/// partials in LDS in ascending krank order — single owner per row, no atomics.
pub const GEMV_MQ3G256_LLOYD_MOE_NINEPATH_D4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_ninepath_d4.hip");
/// Exact gfx1151 temporal-buffer candidate for the K=512 routed down stream.
/// The kernel ABI, four-row mapping, reduction tree, and launch geometry stay
/// unchanged; only weight addressing is lowered differently.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_MOE_DOWN_KERNEL gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip")
);
/// gfx1151 hybrid routed-down candidate. Preserve wave-uniform scale and
/// zero-point headers on scalar/global loads while lowering only the
/// lane-divergent packed weights to temporal buffer VMEM.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_HYBRID_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX1151_MOE_DOWN_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_MOE_DOWN_KERNEL gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_hybrid_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip")
);
/// gfx1151 one-row routed-down schedule with temporal buffer VMEM for routing
/// tables, expert pointers, and weights.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW1_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!(
        "../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer.gfx1151.hip"
    )
);
/// gfx1151 two-row routed-down schedule: temporal VMEM with an occupancy/MLP
/// point between the row1 and row4 implementations.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW2_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!(
        "../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer.gfx1151.hip"
    )
);
/// Same two-row gfx1151 kernel with an explicit weight-VMEM clause before the
/// dependent dequant/FMA phase.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW2_CLUSTERED_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_MOE_DOWN_ROW2_CLUSTERED 1\n",
    "#define HIPFIRE_MOE_DOWN_ROW2_KERNEL gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_clustered_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!(
        "../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer.gfx1151.hip"
    )
);

/// gfx1151 K=512 eight-row schedule. One wave amortizes each activation load
/// across eight output rows, targeting the APU's bandwidth ceiling without
/// changing the routed-down ABI or any other architecture's code object.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_ROW8_GFX1151_SRC: &str = include_str!(
    "../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8.gfx1151.hip"
);

/// Radiowave gfx1100 cache-policy probe for the atomic-free MoE down stream.
/// Keep this replay-visible and opt-in until exact-shadow and product-level
/// certification establish whether SLC helps this access pattern.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_CPOL_SLC_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_MOE_DOWN_KERNEL gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip")
);

/// gfx1100 expert-wave-preserving down+combine experiment. Every workgroup
/// still computes one expert rank; the last arriving rank for each four-row
/// tile performs the deterministic weighted fold and resets its scratch-tail
/// counter, eliminating the separate combine launch without serializing the
/// eight expert GEMVs inside one workgroup.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_LAST_COMBINE_SRC: &str = concat!(
    "#define HIPFIRE_MOE_DOWN_LAST_COMBINE 1\n",
    "#define HIPFIRE_MOE_DOWN_KERNEL gemv_hfq4g256_moe_down_k8_indexed_last_combine\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip")
);

/// Fused atomic-free MoE down: GEMV + K_TOP weighted-accumulate + residual
/// add in a single kernel. Replaces the two-launch
/// `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` (writes an
/// [N × K_TOP × M] expanded buffer) FOLLOWED BY `moe_down_combine_k8_batched`
/// (weighted-sums the K_TOP slots into x_residual). Each block owns one
/// (token, row), loops all K_TOP experts internally, and does a single
/// race-free `x_residual[token][row] += Σ_k weight[k]·down_k(row)` — no
/// expanded intermediate materialized, no atomicAdd. Wave32-only (RDNA);
/// opt-in via `HIPFIRE_MOE_DOWN_FUSED=1`.
pub const GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_FUSED_ACC_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_down_k8_indexed_fused_acc.hip");

/// HFQ4G128 (ParoQuant) variant of the atomic-free batched indexed MoE
/// down. Same expanded-output contract as the HFQ4G256 sibling; pairs
/// with `MOE_DOWN_COMBINE_K8_BATCHED_SRC` for the K_TOP fold. Closes the
/// hipGraph-capture gap for ParoQuant routed experts.
pub const GEMV_PARO_Q4G128_MOE_DOWN_K8_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_paro_q4g128_moe_down_k8_indexed_batched.hip");

/// SGLang-style grouped-WMMA-GEMM for HFQ4G128 (ParoQuant) routed-expert
/// gate_up + down dispatch on RDNA3/4. Port of
/// `GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC` with 72 B/group stride and 128
/// elements per group. Same scatter-pipeline contract (expert_tile_ids +
/// sorted_slot_index) and same WMMA 16×16×16 F16 layout. Caller pre-
/// rotates X by the layer's shared Givens sidecar (gate_up or down) and
/// the kernel reads HFQ4G128 nibbles; F32→F16 X conversion is handled by
/// the Rust dispatch wrapper via `ensure_fp16_x`.
pub const GEMM_PARO_Q4G128_MOE_GROUPED_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_paro_q4g128_moe_grouped_wmma_k2.hip");

/// i8 WMMA MMQ MoE grouped GEMM for HFQ4G128 (ParoQuant) on gfx1151
/// (Strix Halo iGPU). Port of GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX1151_SRC
/// with the 72 B/group HFQ4G128 stride. Targets the same +2× FP16 WMMA
/// throughput lift on routed-expert grouped GEMM. Compute-bound regime
/// per Phase 4 rocprof attribution (gemm_paro_q4g128_moe_grouped_wmma_k2
/// = 68.5% of GPU time, 25.8 GiB/s effective — far from the ~256 GB/s BW
/// roof, so compute-throughput doubling has full upside).
pub const GEMM_PARO_Q4G128_MOE_GROUPED_MMQ_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_paro_q4g128_moe_grouped_mmq.gfx1151.hip");

/// k8 (deepest pipeline) sibling of GEMM_PARO_Q4G128_MOE_GROUPED_MMQ_GFX1151_SRC.
/// Processes all 4 Q8_1 sub-blocks of the (single) mmq block per HFQ4G128 group
/// in one inner iteration — 8 WMMAs into 4 independent int32 accumulators
/// before the per-sub-block scale FMA resolves. Same kernarg + grid as k2.
/// Opt-IN via HIPFIRE_MOE_PARO_I8_K8=1 (default OFF pending bench validation).
pub const GEMM_PARO_Q4G128_MOE_GROUPED_MMQ_K8_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_paro_q4g128_moe_grouped_mmq_k8.gfx1151.hip");

/// Fused silu(gate)*up + per-channel scale + krot rounds of Givens
/// rotation. Replaces the silu_mul_f32 + givens_rotate two-launch
/// composition the ParoQuant MoE decode path used for the gate→down
/// hop. Structural mirror of MQ4's `fused_silu_mul_rotate_mq` — fuses
/// the gate→down activations to eliminate inter-launch state and
/// match the single-kernel reduction pattern that holds up under
/// hipGraph capture.
pub const FUSED_SILU_MUL_GIVENS_ROTATE_SRC: &str =
    include_str!("../../../kernels/src/fused_silu_mul_givens_rotate.hip");

/// Out-of-place variant of `givens_rotate_f32`. Reads `x_in`, writes
/// rotated activations to `x_out`. Used by `rotate_x_paro_for` to
/// eliminate the preceding `copy_d2d` (one fewer graph node + one
/// fewer inter-node dependency for the hipGraph dependency analyzer).
pub const GIVENS_ROTATE_TO_SRC: &str = include_str!("../../../kernels/src/givens_rotate_to.hip");

/// Index-aware MoE gate_up GEMV for HFQ6G256-layout routed experts. Used
/// by mixed-kmap MoE models where some layers are promoted from MQ4 → MQ6
/// (post-PR-199 alternating-kmap default). Without this kernel, those
/// layers fall to the CPU-topK D2H path which crashes under hipGraph capture.
pub const GEMV_HFQ6G256_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq6g256_moe_gate_up_indexed.hip");

/// Index-aware MoE gate_up GEMV for HFQ5G256-layout routed experts. Mirror
/// of the HFQ6 sibling; reads the 168 B/group 5-bit layout.
pub const GEMV_HFQ5G256_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq5g256_moe_gate_up_indexed.hip");

/// HFQ6G256 counterpart to the atomic-free expanded batched MoE down kernel.
/// Same expand-then-combine pattern; pairs with `MOE_DOWN_COMBINE_K8_BATCHED_SRC`
/// (combine is dtype-independent — operates on the f32 expanded buffer).
pub const GEMV_HFQ6G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq6g256_moe_down_k8_indexed_batched_expanded.hip");

/// Merged per-expert MIXED-PRECISION expanded MoE down GEMV. Per-block branch
/// on `dtype_tags[expert_id]` (0=MQ6 200B/group affine, 1=MQ2-Lloyd 72B/group
/// codebook). Block-uniform branch (grid block-per-(row,krank,token)) -> no
/// warp divergence. Single shared accumulator + expanded write; pairs with
/// `MOE_DOWN_COMBINE_K8_BATCHED_SRC` (dtype-independent f32 combine).
pub const GEMV_MIXED_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mixed_moe_down_k8_indexed_batched_expanded.hip");

/// HFQ5G256 counterpart to the atomic-free expanded batched MoE down kernel.
/// Same expand-then-combine pattern; pairs with `MOE_DOWN_COMBINE_K8_BATCHED_SRC`.
pub const GEMV_HFQ5G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq5g256_moe_down_k8_indexed_batched_expanded.hip");

/// HFQ6G256 batched gate_up: same kernarg signature + grid (M, K_TOP, N) +
/// gate/up output split as the HFQ4 batched gate_up kernel, only the per-group
/// dequant differs (200 B/group, 6-bit). Pairs with the HFQ6 expanded down
/// kernel for the batched LFM2.5-MoE decode path (MQ6-promoted expert layers).
pub const GEMV_HFQ6G256_MOE_GATE_UP_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq6g256_moe_gate_up_k8_indexed_batched.hip");

/// Merged per-expert MIXED-PRECISION MoE gate_up GEMV. Per-block branch on
/// `dtype_tags[expert_id]` (0=MQ6 200B/group affine, 1=MQ2-Lloyd 72B/group
/// codebook). Block-uniform branch -> no divergence. Single shared accumulator
/// + the gate/up split write (M = 2*MI). x indexed [bid*K] (per-token).
pub const GEMV_MIXED_MOE_GATE_UP_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mixed_moe_gate_up_k8_indexed_batched.hip");

/// HFQ5G256 batched gate_up: same kernarg signature + grid (M, K_TOP, N) +
/// gate/up output split as the HFQ6 batched gate_up kernel; only the
/// per-group dequant differs (168 B/group, 5-bit).
pub const GEMV_HFQ5G256_MOE_GATE_UP_INDEXED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq5g256_moe_gate_up_k8_indexed_batched.hip");

/// Combine kernel for the atomic-free MoE down path. Sums K_TOP expert
/// slots per (token, m) with topk_weights applied; accumulates into the
/// per-token residual row. No cross-token contention.
pub const MOE_DOWN_COMBINE_K8_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/moe_down_combine_k8_batched.hip");

/// Cross-layer gfx1100 experiment: combine one K=8 routed output row and
/// consume it immediately for the following K=2048 RMSNorm + MQ rotation.
pub const MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/moe_down_combine_rmsnorm_mq_rotate_vecsum.gfx1100.hip");
pub const MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151\n",
    include_str!("../../../kernels/src/moe_down_combine_rmsnorm_mq_rotate_vecsum.gfx1100.hip")
);

/// Four adjacent output columns per thread with four independent accumulator
/// chains. Preserves the scalar K=8 reduction order per output.
pub const MOE_DOWN_COMBINE_K8_BATCHED_VEC4_SRC: &str =
    include_str!("../../../kernels/src/moe_down_combine_k8_batched_vec4.hip");

/// SGLang-style MoE scatter pipeline — Phase 1: per-expert histogram
/// over flattened topk_indices. Single workgroup, LDS atomics.
pub const MOE_SCATTER_HISTOGRAM_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_scatter_histogram_k8.hip");

/// SGLang-style MoE scatter pipeline — Phase 2: pad raw histogram up to
/// BLOCK_M and exclusive prefix-sum into expert_offsets[E+1]. Rewrites
/// expert_token_counts in place from raw → padded. expert_offsets[E] is
/// M_total (total padded slot count).
pub const MOE_SCATTER_OFFSETS_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_scatter_offsets_k8.hip");

/// SGLang-style MoE scatter pipeline — Phase 3: scatter each flat slot
/// (n*K_TOP + k) into sorted_slot_index at offsets[e] + bucket[e]. Pads
/// with -1 sentinel. Emits expert_tile_ids[M_total/BLOCK_M] for the
/// grouped-GEMM dispatch loop (Stage 2).
pub const MOE_SCATTER_PERMUTE_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_scatter_permute_k8.hip");

/// Path 2 grouped-GEMM kernel for MoE prefill gate_up (and reusable
/// for moe down). WMMA 16×16×16 F16, 2× K-tile pipeline. Per-tile
/// expert lookup + sorted_slot_index X gather; -1 padding lanes
/// substitute zero. Writes Y_grouped[m_total × M] directly (no
/// residual; combine kernel handles the fanout into per-token
/// gate_batch/up_batch). **gfx11 (RDNA3) only** — the gfx12 sister
/// kernel below uses the _gfx12 WMMA intrinsic.
pub const GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_wmma_k2.hip");

/// gfx12 (RDNA4) sister of GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC. Same
/// dispatch contract; differs in WMMA intrinsic (_gfx12), operand
/// width (half8_t vs half16_t), and K-lane split (K split across 2
/// lane groups). K4 unroll like the gfx12 residual variant.
pub const GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_wmma.gfx12.hip");

/// 2×1 M-direction reg-blocked sister of GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC.
/// Each warp covers a 32-row × 16-slot output tile; B-load shared across both
/// M-blocks halves X-gather BW per output. Same kernarg layout. Lever from
/// glovepost/wmma_ops (`wmma_kernels_optimized.hpp`). Gated on
/// `HIPFIRE_MOE_GROUPED_M2=1`.
pub const GEMM_HFQ4G256_MOE_GROUPED_WMMA_M2_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_wmma_m2.gfx12.hip");

/// gfx1151 (Strix Halo iGPU) i8 MMQ variant of MOE_GROUPED_WMMA_K2. Uses
/// i8 WMMA (`wmma_i32_16x16x16_iu8_w32`) at 2× the FLOP rate of FP16 WMMA
/// on gfx11. X must be pre-quantized to Q8_1 via `ensure_q8_1_mmq_x`
/// before dispatch. Same 16×16 output tile + grouped scatter contract as
/// the FP16 sister, but the kernel reads Q8_1 packed activations and
/// applies HFQ4 scale/zero through the Q8_1 (d, sum) correction at the
/// FMA step. Lifts the grouped-MoE-GEMM compute ceiling from ~71 to
/// ~140 TFLOPS on Strix Halo, matching the analog "MMQ for attention
/// beats FP16 WMMA at batch≥256" empirical win seen on gfx1100.
pub const GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_mmq.gfx1151.hip");

/// k4 (deeper K-tile pipeline) sibling of GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX1151_SRC.
/// Pairs adjacent Q8_1 sub-blocks (sb=0,1 then sb=2,3) so each inner
/// iteration issues 4 WMMAs into 2 independent int32 accumulators before
/// the per-sub-block scale FMA resolves. Same kernarg signature + grid +
/// block geometry as the k2 sibling; the only difference is unroll depth.
/// Rationale: i8 WMMA has ~2× FP16 WMMA throughput on gfx11 so per-tile
/// latency is shorter — a deeper pipeline hides more dequant + (d, sum)
/// load latency than the k2 baseline could. Opt-IN via
/// `HIPFIRE_MOE_GROUPED_I8_K4=1` (default off pending bench validation).
pub const GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_mmq_k4.gfx1151.hip");

/// k8 (deepest K-tile pipeline) sibling of GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX1151_SRC.
/// Processes all 4 sub-blocks of one Q8_1 block per inner iteration —
/// 8 WMMAs into 4 independent int32 accumulators before the per-sub-block
/// scale FMA resolves. Same kernarg signature + grid + block geometry as
/// the k2/k4 siblings; the only difference is unroll depth. Natural next
/// experiment after k4 hit +4.6% on gfx1151 with zero register spills;
/// expected diminishing return (+1-3% upside, possible 0% or regression
/// if registers spill). Opt-IN via `HIPFIRE_MOE_GROUPED_I8_K8=1` (default
/// OFF pending bench validation).
pub const GEMM_HFQ4G256_MOE_GROUPED_MMQ_K8_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_mmq_k8.gfx1151.hip");

/// gfx12 (RDNA4 — R9700/gfx1201, gfx1200) i8 MMQ variant of MOE_GROUPED_WMMA.
/// Uses the gfx12-specific i8 WMMA intrinsic
/// (`wmma_i32_16x16x16_iu8_w32_gfx12`). X pre-quantized to Q8_1 via
/// `ensure_q8_1_mmq_x` before dispatch. Same 16×16 output tile + grouped
/// scatter contract as the FP16 sister; kernel reads Q8_1 packed activations
/// and applies HFQ4 scale/zero through the Q8_1 (d, sum) correction post-WMMA.
///
/// Key gfx12 differences vs gfx11/gfx1151:
///   - Intrinsic has `_gfx12` suffix.
///   - Per-lane operand width is int32x2 (8 int8) instead of int32x4 (16 int8).
///   - K=16 per WMMA is split across 2 lane groups (lanes 0-15 carry K[0..7],
///     lanes 16-31 carry K[8..15]).
///   - C-output mapping: `acc[j] = C[8*k_grp + j][m_lane]` (8 contiguous M-rows
///     per lane half), unlike gfx11/gfx1151's `acc[j] = C[2*j + k_grp][m_lane]`.
///   - Per-row HFQ4 sc/zp shuffle uses `src_lane = 8*k_grp + j`.
///
/// **EMPIRICAL RESULT (2026-05-19, R9700/gfx1201, A3B uniform.mq4, prefill=256):
/// 2960 → 2607 tok/s = -11.6% REGRESSION** vs FP16 grouped WMMA. Kernel-level:
/// 279µs/call (FP16) → 408µs/call (i8) = +46% slowdown despite correctness
/// PASS (NRMSE ~0.4% on a3b-slice shape). Root cause: i8 WMMA's theoretical
/// 2× FLOP rate is offset by per-sub-block scale FMA serial dependency chain
/// (each Q8_1 sub-block: 2 WMMAs → 8 INT32→F32 conversions → 16 FMAs). Same
/// synth-win → prod-falsify pattern as docs/memory items
/// `project_fp8_wmma_hfp4g32_2026_05_10` and
/// `project_gfx11_dot2_trickle_down_falsified_2026_05_11`. Shipped as opt-in
/// research artifact; default OFF for gfx12. Opt-in via
/// HIPFIRE_MOE_GROUPED_I8=1.
pub const GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_mmq.gfx12.hip");

/// k4 (deeper K-tile pipeline) variant of GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX12_SRC.
/// Processes all 4 sub-blocks of one Q8_1 block per inner iteration —
/// 8 WMMAs into 4 independent int32 accumulators before the per-sub-block
/// scale FMA chain resolves. Same kernarg layout + grid + block geometry as
/// the k2 sibling; the only difference is unroll depth.
///
/// Rationale: the k2 gfx12 path runs at +46% kernel time vs FP16 (R9700/
/// gfx1201 A3B prefill, 2026-05-19). Structural diagnosis was that the
/// per-sub-block scale FMA serial dependency chain dominates the 16×16
/// output tile. k4 amortizes that chain over more WMMA dispatches per
/// scale-FMA boundary — same lever that gave +4.6% on gfx1151 and +2.8%
/// on gfx11_dgpu. R9700's fewer CUs (~40-48 vs 96 on 7900 XTX) should
/// make this proportionally more effective IF the diagnosis is right.
///
/// Opt-IN via `HIPFIRE_MOE_GROUPED_I8=1 HIPFIRE_MOE_GROUPED_I8_K4_GFX12=1`
/// (both default OFF — this is an experiment).
pub const GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_mmq_k4.gfx12.hip");

/// gfx1151 (RDNA3.5) HFQ6/MQ6 sister of GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC
/// for AWQ MoE experts. Same gfx11 WMMA tile geometry + grouped dispatch
/// contract; differs only in the 200 B/group HFQ6 dequant inner loop.
pub const GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_moe_grouped_wmma.gfx1151.hip");

/// gfx1151 i8-WMMA MMQ sister of GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX1151_SRC.
/// Uses Q8_1 activation prequant plus the HFQ6/MQ6 correction term:
/// sc * d_x * dot_i8(q, qx) + zp * sum_x.
pub const GEMM_HFQ6G256_MOE_GROUPED_MMQ_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_moe_grouped_mmq.gfx1151.hip");

/// HFQ6/MQ6 sister of GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC for AWQ MoE
/// experts. Same WMMA tile geometry + grouped dispatch contract; differs
/// only in the 200 B/group HFQ6 dequant inner loop (4 B scale + 4 B zero
/// + 192 B packed 6-bit). The kernel is dtype-agnostic between HFQ6 and
/// MQ6 -- MQ6G256 uses the identical 200 B layout and the caller applies
/// the FWHT rotation to X before dispatch, same convention as MQ4/HFQ4.
/// **gfx12 (RDNA4) only.** Unblocks AWQ A3B prefill (~50% of experts MQ6).
pub const GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_moe_grouped_wmma.gfx12.hip");

/// M-direction 2×1 reg-blocked sister of GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX12_SRC.
/// Each warp covers a 32-row × 16-slot output tile (vs 16×16 in v1); per
/// K-substep 2 A-dequants + 1 B-load → 2 WMMAs (vs 1+1+1 in v1). Halves X-gather
/// BW per output element. Same kernarg layout + BLOCK_M=16 slot stride
/// (expert-boundary contract unchanged). Lever from `wmma_kernels_optimized.hpp`
/// extending the HFQ4 m2 trick to BW-bound HFQ6 (where the dequant
/// serialization that falsified HFQ4-m2 is hidden behind memory waits).
/// Gated on `HIPFIRE_MOE_HFQ6_V2=1` (default off).
pub const GEMM_HFQ6G256_MOE_GROUPED_WMMA_V2_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_moe_grouped_wmma_v2.gfx12.hip");

/// gfx11 (RDNA3/3.5) sister of GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX12_SRC.
/// Same kernarg + grouped contract; differs in WMMA intrinsic
/// (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32, half16_t), output mapping
/// (acc[j]=C[2*j+(tid>>4)][tid&15]), and the 6-bit unpack (200 B/group).
/// Enables uniform MQ6 expert batched prefill on gfx1100+gfx1151.
pub const GEMM_HFQ6G256_MOE_GROUPED_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_moe_grouped_wmma_k2.hip");

/// gfx11 (RDNA3/3.5) MQ3-Lloyd grouped-WMMA GEMM for MoE prefill. 112 B/group
/// = 16 B (8 × fp16 codebook) + 96 B (256 × 3-bit indices). Codebook into
/// __shared__ _Float16[8], broadcast to all 32 lanes. T3-3L cold-tier block.
pub const GEMM_MQ3G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq3g256_lloyd_moe_grouped_wmma_k2.hip");

/// gfx12 (RDNA4) MQ3-Lloyd sister. Same 112 B/group codebook layout;
/// differs in WMMA (_gfx12), half8_t operands, k_grp=tid>>4 split, and
/// acc[j]=C[8*k_grp+j][m_lane] output. Distinct from
/// GEMM_HFQ3G256_MOE_GROUPED_WMMA_GFX12_SRC (that one is affine 104 B/group).
pub const GEMM_MQ3G256_LLOYD_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq3g256_lloyd_moe_grouped_wmma.gfx12.hip");

/// gfx11 (RDNA3/3.5) merged dtype-tag MoE grouped-WMMA prefill kernel.
/// Handles all four per-expert precision tiers in one kernel via a
/// block-uniform `dtype_tags[expert_id]` branch (zero warp divergence):
///   tag 0 = MQ6 (200 B/group), tag 1 = MQ2-Lloyd (72 B/group),
///   tag 2 = MQ4 (136 B/group),  tag 3 = MQ3-Lloyd (112 B/group).
/// Enables T3-3L graded-precision MoE batched prefill on gfx1100/gfx1151.
/// Kernarg layout: expert_weight_ptrs, dtype_tags, expert_tile_ids,
/// sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total.
pub const GEMM_MIXED_MOE_GROUPED_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_mixed_moe_grouped_wmma_k2.hip");

/// 4-warp LDS-tiled variant of the mixed grouped kernel (gfx11/RDNA3 only).
/// Stages the 16-slot X tile in LDS and shares it across 4 warps → ~4× less X
/// re-read (the load-bound prefill lever). Env-gated via HIPFIRE_MOE_GROUPED_4W;
/// default OFF (single-warp `_k2` remains production). Same kernarg layout.
pub const GEMM_MIXED_MOE_GROUPED_WMMA_4W_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_mixed_moe_grouped_wmma_4w_k2.hip");

/// gfx12 (RDNA4) merged dtype-tag MoE grouped-WMMA prefill kernel.
/// Same four-tag dispatch as `GEMM_MIXED_MOE_GROUPED_WMMA_K2_SRC` but
/// uses the gfx12 _w32_gfx12 WMMA intrinsic, half8_t operands, K4-unroll,
/// and k_grp=tid>>4 output mapping. Covers all four dtype tiers including
/// an inline gfx12 MQ2-Lloyd port (no separate source file exists for that
/// one). Same kernarg layout as the gfx11 sibling.
pub const GEMM_MIXED_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mixed_moe_grouped_wmma.gfx12.hip");

/// gfx12 (RDNA4) HFQ3/MQ3 sister of GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC.
/// Same WMMA tile geometry + expert_tile_ids sentinel pattern + kernarg
/// layout; differs in dequant (HFQ3-G256 = 104 B/group, 8 × 3-bit chunks
/// packed across 24 bits per 3-byte slice). Same FWHT-rotated 3-bit
/// storage applies to MQ3G256 — the kernel handles both buffers (rotation
/// is applied by the caller, matching the MQ4/HFQ4 dispatch convention).
pub const GEMM_HFQ3G256_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_moe_grouped_wmma.gfx12.hip");

/// i8 MMQ sister of GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC for gfx11 dGPUs
/// (gfx1100/1101/1102/1103 — 7900 XTX, 7800/7700, 7600, Phoenix mobile).
/// Same 9-tuple kernarg layout + 16×16 output tile + expert_tile_ids /
/// sorted_slot_index dispatch contract; differs in that X is pre-quantized
/// to `block_q8_1_mmq` (via `ensure_q8_1_mmq_x`) and the per-K-tile WMMA
/// uses the int8 `iu8` builtin instead of the FP16 `f16` builtin. Roughly
/// 2× the FLOP rate on gfx11 dGPUs (compute-bound on this path); BW
/// reduced via i8 X (vs FP16 X). HFQ4 sc/zp correction applied in float
/// at the 32-K Q8_1 sub-block boundary. Gated on
/// `HIPFIRE_MOE_GROUPED_I8` (default on for gfx1100/1101/1102/1103).
pub const GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX11_DGPU_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_mmq.gfx11_dgpu.hip");

/// k4 (deeper K-tile pipeline) variant of GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX11_DGPU_SRC.
/// Same kernarg layout; processes 4 K-tiles per inner iteration instead of 2.
/// Matches the gfx1151 k4 design (validated +4.6% over k2 there with zero spills).
/// Opt-in on gfx11 dGPUs via `HIPFIRE_MOE_GROUPED_I8_K4=1` (default off pending
/// A/B confirmation).
pub const GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX11_DGPU_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_moe_grouped_mmq_k4.gfx11_dgpu.hip");

/// Non-residual WMMA Q8_0 GEMM (gfx12). Direct write to Y[N, M] without
/// reading prior values (= rather than +=). Drop-in replacement for the
/// scalar `gemm_q8_0_batched` kernel; rocprof 2026-05-19 showed that
/// scalar version was 65% of A3B prefill GPU time (invisible to
/// HIPFIRE_PROFILE, no begin_timer wired).
pub const GEMM_Q8_0_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_q8_0_wmma.gfx12.hip");

/// Non-residual WMMA Q8_0 GEMM (RDNA3+ / gfx1100+). Generic variant
/// using the cross-RDNA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
/// intrinsic — works on gfx1100/gfx115x/gfx1200, distinct from the
/// gfx12-specific intrinsic used by `GEMM_Q8_0_WMMA_GFX12_SRC`. Same
/// shape contract (Y[N, M] = X[N, K] @ A_q8[M, K]^T, K % 32 == 0).
/// Selected by `gemm_q8_0_wmma` dispatch when the runtime arch is
/// not gfx12.
pub const GEMM_Q8_0_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_q8_0_wmma.hip");

/// Path 2 unscatter combine for gate_up: fans Y_grouped[m_total × 2*mi]
/// back into per-token gate_batch[N × K_TOP × mi] + up_batch[N × K_TOP
/// × mi] via the inverse permutation in sorted_slot_index.
pub const MOE_GATE_UP_UNSCATTER_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_gate_up_unscatter_k8.hip");

/// Phase D1 (2026-05-26): fused unscatter + SwiGLU + asymmetric clamp.
/// Replaces `MOE_GATE_UP_UNSCATTER_K8_SRC` followed by
/// `DEEPSEEK4_SILU_MUL_CLAMP_F32_SRC` (batched) — eliminates the
/// `moe_up_batch` intermediate buffer and saves 1 launch per layer.
pub const MOE_UNSCATTER_SILU_CLAMP_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_unscatter_silu_clamp_k8.hip");

/// 4-warp 64×64 Q8 WMMA GEMM for gfx1151 (RDNA3.5). LDS-staged X.
/// Follows the llama.cpp MMQ pattern (pedapudi #21284) for 4× weight
/// reuse per block vs the single-warp 16×16 kernel.
pub const GEMM_Q8_0_WMMA_4W_SRC: &str = include_str!("../../../kernels/src/gemm_q8_0_wmma_4w.hip");

/// Dense i8-WMMA MMQ GEMM for Q8_0 weights — 4-warp 64x64 tile (gfx1151). i8
/// port of gemm_q8_0_wmma_4w: int8 weights direct + Q8_1 activations + i8 WMMA
/// at ~2x. Y[N,M] = X[N,K] @ W[M,K]^T. Requires M%64==0, N%64==0, K%128==0.
pub const GEMM_Q8_0_MMQ_4W_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_q8_0_mmq_4w.gfx1151.hip");

/// Exact-gfx1030 sdot4 MMQ probe for Q8_0 weights. This source is exposed to
/// the dedicated channel test only; production dispatch remains unchanged.
pub const GEMM_Q8_0_MMQ_GFX1030_SRC: &str =
    include_str!("../../../kernels/src/gemm_q8_0_mmq.gfx1030.hip");

/// Path 2 combine for down: per (token, m) iterates K_TOP slots via
/// `inverse_perm[token*K_TOP + k]`, applies topk_weights, and += into
/// x_residual. No atomic contention (each token's m column is owned by
/// a unique thread).
pub const MOE_DOWN_COMBINE_GROUPED_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_down_combine_grouped_k8.hip");

/// Fused single-CTA SGLang-style MoE scatter pipeline: combines
/// histogram + padded prefix-sum + permutation in one launch. Saves
/// 2 kernel launches per MoE layer (~75µs each).
pub const MOE_SCATTER_FUSED_K8_SRC: &str =
    include_str!("../../../kernels/src/moe_scatter_fused_k8.hip");

/// LA-layer fusion: fused L2-norm(Q) + scale(Q) + L2-norm(K) +
/// repeat-interleave(Q,K). Replaces fused_qk_l2_norm_scale_f32_batched
/// + repeat_interleave_qk_f32_batched. Saves 1 launch per LA layer.
pub const FUSED_QK_L2_NORM_SCALE_INTERLEAVE_F32_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/fused_qk_l2_norm_scale_interleave_f32_batched.hip");

// Batched HFQ4-G256 GEMM with fused residual add. Processes N batch elements
// per launch with the same 4-accumulator interleave as the single-row GEMV, so
// output is bitwise identical to calling gemv_hfq4g256_residual N times. Used
// for batched prefill (FFN down + wo projection) where N prompt tokens share
// the same weight matrix.
pub const GEMM_HFQ4G256_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual.hip");

// Batched HFQ3-G256 GEMM with fused residual add. HFQ3 sibling of
// GEMM_HFQ4G256_RESIDUAL_SRC — same dispatch shape and batching tile,
// 104 B group stride and 3-bit unpack instead of 136 B / 4-bit. Used
// for batched prefill of MQ3-quantized down + wo projections when
// `is_batchable_la(MQ3G256, arch)` returns true (non-WMMA archs only;
// gfx11+ uses the WMMA family).
pub const GEMM_HFQ3G256_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual.hip");
// v_dot2_f32_f16 variant — HFQ3 sibling of GEMM_HFQ4G256_RESIDUAL_FP16_SRC, upgraded
// from v_pk_fma_f16 to v_dot2_f32_f16 (4 amd_mixed_dot calls per group, FP32 accum).
pub const GEMM_HFQ3G256_RESIDUAL_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_dot2.hip");
// FP16-packed (v_pk_fma_f16) variant — fallback for gfx1010/1013. Same inner loop
// as gemm_hfq4g256_residual_fp16 (1 __hmul2 + 3 __hfma2 + extract + add).
pub const GEMM_HFQ3G256_RESIDUAL_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_fp16.hip");

// CDNA3 wave64-native batched HFQ4-G256 residual GEMM. 2 rows per block
// (one per warp), halves grid.x. Byte-exact with the wave32 kernel.
pub const GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wave64.hip");

// GCN5/CDNA1 wave64 FP16 hybrid residual GEMM. Same __hfma2 inner loop
// as the FP16 variant, but block=[64,1,1] with 2 rows/block via warp_id.
// Scoped to gfx906/gfx908 — CDNA3 uses the rocBLAS MFMA path instead.
pub const GEMM_HFQ4G256_RESIDUAL_FP16_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_fp16_wave64.hip");

// FP16-packed variant: dequant to __half, v_pk_fma_f16 inner loop, FP32 accumulation.
// 2× throughput over FP32 on all RDNA. Same grid/block layout.
pub const GEMM_HFQ4G256_RESIDUAL_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_fp16.hip");

// WMMA variant: gfx1100+ only. Uses __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
// for 16×16 tiled matrix multiply. Same FP16 X input, FP32 Y output.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA2_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma2.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_k2.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_K2X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_k2x32.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_K4_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_k4.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_ksplit.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_DET_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_ksplit_det.hip");
pub const GEMM_KSPLIT_DET_FINALIZE_SRC: &str =
    include_str!("../../../kernels/src/gemm_ksplit_det_finalize.hip");
// gfx12 (RDNA4) sister of GEMM_HFQ4G256_RESIDUAL_WMMA_K2_SRC. Same recipe
// as the qkv / qkvza / gate_up gfx12 ports (PR #56): `_w32_gfx12` builtin,
// half8_t operands, K-split via tid>>4, contiguous C-row mapping. Closes
// the residual-GEMM gap on 9B prefill (42% of decode-batch GEMM time was
// stuck on the dot2 fp16 fallback before this).
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma.gfx12.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx12_bt.hip");
/// Muse Glimmer-owned batch-tiled gfx12 residual GEMM. Sibling of
/// `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC`; same math, register-lean
/// full-tile variants only. Used exclusively by hipfire-arch-muse-glimmer.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_MUSE_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx12_muse.hip");
/// Muse Glimmer-owned batch-tiled gfx1100 residual GEMM (K2). Sibling of
/// `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_MUSE_SRC`; full-tile widths 4/6/8/12/16
/// for gate/up shape measurement. Used exclusively by hipfire-arch-muse-glimmer.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_bt.hip");
/// Muse Glimmer-owned batch-tiled gfx1100 residual GEMM with LDS HFQ affine
/// codebook precompute (16 values per output-row/group). Sibling of
/// `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_BT_SRC`; full-tile widths 4/6/12
/// only. Used exclusively by hipfire-arch-muse-glimmer.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_CB_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_cb_bt.hip");
/// Muse Glimmer-owned multi-wave batch-tiled gfx1100 residual GEMM (K2, BT4).
/// Sibling of `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_BT_SRC`; groups
/// NWAVES identical BT4 row waves into one block to expose activation-cache
/// reuse/scheduling for exact gate/up. Used exclusively by
/// hipfire-arch-muse-glimmer. Full-tile only (`batch % 64 == 0`).
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_MW_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_mw_bt.hip");
/// Muse Glimmer-owned batch-tiled gfx1100 residual GEMM, deterministic
/// K-split phase 1 (partials only). Sibling of
/// `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_BT_SRC`; same K2 arithmetic and
/// batch-tile structure but partitioned over K_SPLITS=4. Writes plain
/// partials to the existing `[K_SPLITS][batch][M]` scratch layout — no
/// atomicAdd, no residual. Finalize remains `gemm_ksplit_det_finalize`.
/// Used exclusively by hipfire-arch-muse-glimmer. Full-tile only
/// (`batch % (16*bt) == 0`).
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_KS_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_ks_bt.hip");
/// Muse Glimmer-owned multi-wave LDS-staged-X gfx1100 residual GEMM
/// (K2, MW8, BT6). Sibling of `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_MW_BT_SRC`
/// and `…_MUSE_BT_SRC`; stages one 96×256 FP16 X group into static LDS and
/// reuses it across 8 row waves. Exact gate/up only (M=19968, K=6656,
/// batch % 96 == 0). Used exclusively by hipfire-arch-muse-glimmer.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_LDS_G256_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_lds_g256.hip");
/// Muse Glimmer-owned row-reuse batch-tiled gfx1100 residual GEMM (K2).
/// Sibling of `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_BT_SRC`; each wave owns
/// RM row tiles × BV batch tiles with RM*BV=12, loading each B half16 once per
/// batch tile and reusing it across RM WMMA calls. Exact gate/up only
/// (M=19968, K=6656, rm in {2,3,4,6}, batch % (16*BV) == 0). RM2/BV6 is
/// production for gfx1100 B=192 gate/up via `gemm_hfq4g256_batched_lmhead`;
/// other RM/BV instantiations remain measurement-only.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt.hip");
/// Muse Glimmer-owned row-reuse gfx1100 residual GEMM with half-broadcast A
/// dequant (K2). Sibling of `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_BT_SRC`;
/// only lanes 0..15 load/dequant each A fragment and bit-broadcast the raw
/// half16 u32s to the paired lane via wave32 shuffle. Exact gate/up only
/// (M=19968, K=6656, rm in {2,4}, batch % (16*BV) == 0). Measurement-only;
/// used exclusively by hipfire-arch-muse-glimmer.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_HB_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_hb.hip");
/// Muse Glimmer-owned row-reuse gfx1100 residual GEMM with packed-half2 A
/// dequant (K2). Sibling of `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_BT_SRC`;
/// dequants each 16-nibble A fragment via eight `__hfma2` half2 ops and
/// reinterprets as the half16 WMMA operand. Exact gate/up only
/// (M=19968, K=6656, rm in {1,2}, batch % (16*BV) == 0). Measurement-only;
/// used exclusively by hipfire-arch-muse-glimmer. Packed path is not assumed
/// bit-exact vs scalar; full-output bit oracle is authoritative.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_PK_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_pk.hip");
/// Muse Glimmer-owned row-reuse gfx1100 residual GEMM with two-slot X-fragment
/// software pipeline (K2). Sibling of `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_BT_SRC`;
/// RM2/BV6 only. For each g/kt fragment issues B loads ahead of A dequant and
/// keeps at most two live B fragments. Scalar and packed-half2 dequant symbols
/// share one module. Exact gate/up only (M=19968, K=6656, batch=192).
/// Measurement-only; used exclusively by hipfire-arch-muse-glimmer.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_PIPE_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_pipe.hip");







pub const GEMM_HFQ4G256_LMHEAD_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_lmhead_wmma.gfx12.hip");
// Q8_1 MMQ prefill variant — opt-in via HIPFIRE_MMQ=1, gated to RDNA3/3.5.
// Pre-quantizes activations to Q8_1 + uses i8 WMMA over 128×128 tiles. Targets
// the Strix Halo prefill gap vs llama.cpp (#60); also wins ~+20% on gfx1100
// at pp≥256.
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq.hip");
// gfx12 (RDNA4) i8-WMMA MMQ port (single-wave 16-row tile, [32,1,1], LDS 0).
// RDNA3's #if guard excludes gfx12, so RDNA4 needs this separate source.
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq.gfx12.hip");
// gfx906 MMQ kernel (see docs/plans/gfx906-mmq-prd.md and
// docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md).
// Topology: nwarps=4, runtime-dispatched mmq_x ∈ {8,16,24,32,40,48,56,64},
// per-mmq_x X_STRIDE (33 or 40) for ds_read_b128 alignment vs
// bank-conflict tradeoff.
// Shared body + per-mmq_x wrapper files.
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X8_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x8.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x16.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X24_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x24.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x32.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X40_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x40.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X48_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x48.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X56_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x56.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X64_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_gfx906_x64.hip");
pub const GEMM_QKV_HFQ4G256_MMQ_GFX906_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_gfx906_body.cuh");
pub const GEMM_QKV_HFQ4G256_MMQ_GFX906_X8_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_gfx906_x8.hip");
pub const GEMM_QKV_HFQ4G256_MMQ_GFX906_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_gfx906_x16.hip");
pub const GEMM_QKV_HFQ4G256_MMQ_GFX906_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_gfx906_x32.hip");
pub const GEMM_QKV_HFQ4G256_MMQ_GFX906_X64_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_gfx906_x64.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X8_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_x8.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_x16.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_x32.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X64_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_x64.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X16_Y64_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_x16_y64.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X32_Y64_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_x32_y64.hip");
pub const GEMM_MW16_RESIDUAL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_mw16_residual_wmma.hip");
pub const DEQUANT_HFQ4G256_TO_F16_SRC: &str =
    include_str!("../../../kernels/src/dequant_hfq4g256_to_f16.hip");
pub const DEQUANTIZE_MFP4G32_LLOYD_TO_F16_SRC: &str =
    include_str!("../../../kernels/src/dequantize_mfp4g32_lloyd_to_f16.hip");
pub const DEQUANTIZE_MFP4G32_P_TO_F16_SRC: &str =
    include_str!("../../../kernels/src/dequantize_mfp4g32_p_to_f16.hip");
pub const DEQUANTIZE_MFP4G32_E8_TO_F16_SRC: &str =
    include_str!("../../../kernels/src/dequantize_mfp4g32_e8_to_f16.hip");
pub const DEQUANTIZE_MFP4G32_E8_SOA_TO_F16_GFX942_SRC: &str =
    include_str!("../../../kernels/src/dequantize_mfp4g32_e8_soa_to_f16.gfx942.hip");
/// Dense parent FP8 (E4M3) + UE8M0 128×128 block-scale → BF16 (gfx942).
pub const DEQUANT_FP8_E4M3_UE8M0_BLK128_TO_BF16_GFX942_SRC: &str =
    include_str!("../../../kernels/src/dequant_fp8_e4m3_ue8m0_blk128_to_bf16.gfx942.hip");
/// Routed-expert parent FP4 (packed E2M1) + UE8M0 g32 scale → BF16 (gfx942).
pub const DEQUANT_FP4_E2M1_UE8M0_G32_TO_BF16_GFX942_SRC: &str =
    include_str!("../../../kernels/src/dequant_fp4_e2m1_ue8m0_g32_to_bf16.gfx942.hip");
/// Parent FP8 (E4M3 OCP) + UE8M0 dynamic act quant, fused quant+dequant in-place BF16 (gfx942).
pub const ACT_QUANT_FP8_UE8M0_INPLACE_GFX942_SRC: &str =
    include_str!("../../../kernels/src/act_quant_fp8_ue8m0_inplace.gfx942.hip");
/// Parent FP4 (E2M1) + UE8M0 g32 dynamic act quant, fused quant+dequant in-place BF16 (gfx942).
pub const ACT_QUANT_FP4_UE8M0_G32_INPLACE_GFX942_SRC: &str =
    include_str!("../../../kernels/src/act_quant_fp4_ue8m0_g32_inplace.gfx942.hip");
/// BF16×BF16→FP32 MFMA GEMM for parent-checkpoint linears (gfx942).
/// Layout: A[M,K] BF16 weights, B[batch,K] BF16 acts, D[batch,M] FP32.
pub const GEMM_BF16_MFMA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemm_bf16_mfma.gfx942.hip");

pub const GEMM_GATE_UP_HFQ4G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma.hip");
// LDS-staged X variant. Opt-in via HIPFIRE_GATE_UP_VARIANT=ldsx for
// Gate 1 microbench measurement. See
// docs/perf-checkpoints/2026-05-01-gate-up-lds-x-share-plan.md.
pub const GEMM_GATE_UP_HFQ4G256_WMMA_LDSX_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma_ldsx.hip");
// K4 4-tile pipeline variant. Opt-in via HIPFIRE_GATE_UP_VARIANT=k4 to
// test deeper memory-load pipelining (3-4 in-flight B loads vs k2's 2).
// Target: lift gate_up_wmma from 305 GB/s (32% peak on gfx1100) toward
// 60-70% peak. See feedback_v3_gate_up_k4_2026_05_21.md.
pub const GEMM_GATE_UP_HFQ4G256_WMMA_K4_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma_k4.hip");
// LDSCOOP variant: cooperative LDS weight staging. 32 threads load
// each row's 136 bytes in 1-2 coalesced cache lines (128B per warp
// instruction), vs the base k2 kernel's scattered 16-thread loads
// (16 different cache lines per warp). Targets the 32% peak BW
// observed in the base kernel — coalesced DRAM loads should get
// closer to 60-70%. Opt-in via HIPFIRE_GATE_UP_VARIANT=ldscoop.
pub const GEMM_GATE_UP_HFQ4G256_WMMA_LDSCOOP_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma_ldscoop.hip");
/// Barrier-free variant of the LDSCOOP kernel. Each warp loads its own
/// weights and X from global directly, eliminating __syncthreads().
pub const GEMM_GATE_UP_HFQ4G256_WMMA_LDSCOOP_NOSYNC_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma_ldscoop_nosync.hip");
// 2tile variant: 32 rows × 16 cols output tile per block, 64 threads
// (2 wave32). Halves grid in M-dim (1728 → 864 blocks at M=27648),
// amortizing per-block X-tile (FP16 batch matrix) loads across both
// waves via L0/L1 cache. Targets the 32% peak BW seen in base kernel.
// Opt-in via HIPFIRE_GATE_UP_VARIANT=2tile.
pub const GEMM_GATE_UP_HFQ4G256_WMMA_2TILE_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma_2tile.hip");
// gfx12 (RDNA4) sister of GEMM_GATE_UP_HFQ4G256_WMMA_SRC. Same recipe as
// the QKV gfx12 scaffold (validated on R9700): _w32_gfx12 builtin,
// half8_t operands, K-split via tid>>4, contiguous C-row mapping.
pub const GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma.gfx12.hip");
pub const GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma_gfx12_bt.hip");
pub const GEMM_QKVZA_HFQ4G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_wmma.hip");
// gfx12 (RDNA4) sister: gfx12 hfq4 recipe + 4-output qkv/z/beta/alpha
// routing for the DeltaNet LinearAttention preamble.
pub const GEMM_QKVZA_HFQ4G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_wmma.gfx12.hip");
pub const GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_wmma_gfx12_bt.hip");
// HFQ3-G256 sister of GEMM_QKVZA_HFQ4G256_WMMA_SRC. Same WMMA shape +
// lane decomposition; only the inner K-tile unpack differs (3-bit
// cross-byte vs 4-bit nibble). Used for MQ3 prefill via dispatch
// wrapper that pre-rotates X. gfx11 K2 unroll variant — gfx12 K4 to
// follow once K2 is validated.
pub const GEMM_QKVZA_HFQ3G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_wmma.hip");
pub const GEMM_GATE_UP_HFQ3G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_wmma.hip");
pub const GEMM_HFQ3G256_RESIDUAL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_wmma.hip");
pub const GEMM_QKV_HFQ3G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_wmma.hip");
/// HFQ3 mb4 sources: 16×64 output tile per WG, 4 batch sub-tiles share
/// A_reg decode. gfx11 only. No LDS, no syncs (HFQ3 has no codebook).
pub const GEMM_HFQ3G256_RESIDUAL_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_wmma_mb4.hip");
pub const GEMM_QKVZA_HFQ3G256_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_wmma_mb4.hip");
pub const GEMM_QKV_HFQ3G256_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_wmma_mb4.hip");
pub const GEMM_GATE_UP_HFQ3G256_WMMA_MB4_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_wmma_mb4.hip");
pub const GEMM_QKVZA_HFQ3G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_wmma.gfx12.hip");
pub const GEMM_QKV_HFQ3G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_wmma.gfx12.hip");
pub const GEMM_GATE_UP_HFQ3G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_wmma.gfx12.hip");
pub const GEMM_HFQ3G256_RESIDUAL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_wmma.gfx12.hip");
pub const GEMM_QKV_HFQ4G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_wmma.hip");
// gfx12 (RDNA4) sister of GEMM_QKV_HFQ4G256_WMMA_SRC. Uses
// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (vs the gfx11 `_w32`)
// and half8_t operands (vs half16_t). C-output mapping
// `acc[j] = C[8*(tid>>4) + j][tid & 15]` (lane group 0 → rows 0..7, group
// 1 → rows 8..15) — derived from the CK trait kCM0/kCM1PerLane swap and
// validated on R9700 in PR #56's channel-tests.
pub const GEMM_QKV_HFQ4G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_wmma.gfx12.hip");

// Batched 3-way fused HFP4-G32 GEMM (FA preamble: Q + K + V). Sister of
// GEMM_QKV_HFQ4G256_WMMA_SRC for the FP4 (E2M1 + UE8M0 g32 + FP16 row
// scale) family. Same WMMA shape (16x16x16 f16) and lane decomposition;
// only the per-row layout (16-B header + 17-B blocks) and per-tile
// dequant arithmetic (row_scale * 2^(block_e-127) * E2M1_LUT[nibble])
// differ from the HFQ4G256 anchor.
pub const GEMM_QKV_HFP4G32_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfp4g32_wmma.hip");
// gfx12 (RDNA4) sister of GEMM_QKV_HFP4G32_WMMA_SRC. half8_t lane-split
// + K4 unroll (each iter consumes 2 HFP4 blocks). Same C-output mapping
// as gemm_qkv_hfq4g256_wmma.gfx12 (validated on R9700).
pub const GEMM_QKV_HFP4G32_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfp4g32_wmma.gfx12.hip");
// gfx12 FP8-WMMA variant of GEMM_QKV_HFP4G32_WMMA_GFX12_SRC. Uses
// wmma_f32_16x16x16_fp8_fp8 (~1.87x raw issue throughput vs fp16 WMMA
// on gfx1201, microbenched). Weight LUT pre-converts E2M1->E4M3 bytes
// (no scale baked); per-output-row row_scale * UE8M0_block is applied
// to the F32 accumulator after each WMMA-pair via lane-shuffle.
// Activation X is consumed in pre-packed FP8 (E4M3) layout, produced
// by PACK_F32_TO_FP8_GFX12_SRC + ensure_fp8_x.
pub const GEMM_QKV_HFP4G32_WMMA_FP8_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfp4g32_wmma_fp8.gfx12.hip");
// Activation pre-pass for FP8 WMMA kernels: F32 -> E4M3 elementwise,
// no scaling. Memory-BW-bound; lifts the FP8 GEMM kernels above FP16
// parity by moving the cvt out of the WMMA inner loop.
pub const PACK_F32_TO_FP8_GFX12_SRC: &str =
    include_str!("../../../kernels/src/pack_f32_to_fp8.gfx12.hip");
// Fused MagnumQuant FWHT rotation + FP8 (E4M3) pack — gfx12 only.
// Writes both F32 (for legacy consumers) and FP8 outputs in one launch.
// Replaces the standalone mq_rotate_x + pack_f32_to_fp8 sequence on the
// FP8 decode path so the pack launch is no longer on the critical path
// of every weight_gemv(MFP4G32) call.
pub const MQ_ROTATE_X_DUAL_FP8_GFX12_SRC: &str =
    include_str!("../../../kernels/src/mq_rotate_x_dual.gfx12.hip");

// HFP4-G32 residual GEMM (used for wo + w_down). Mirrors the K2 HFQ4
// variant — canonical wave32 WMMA C-output mapping `acc[j] = C[2*j +
// (tid>>4)][tid & 15]`.
pub const GEMM_HFP4G32_RESIDUAL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfp4g32_residual_wmma.hip");
pub const GEMM_HFP4G32_RESIDUAL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfp4g32_residual_wmma.gfx12.hip");

// HFP4-G32 batched 2-way fused GEMM (gate + up). Sister of
// GEMM_QKV_HFP4G32_WMMA_SRC for the FFN preamble.
pub const GEMM_GATE_UP_HFP4G32_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfp4g32_wmma.hip");
pub const GEMM_GATE_UP_HFP4G32_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfp4g32_wmma.gfx12.hip");

// HFP4-G32 batched 4-way fused GEMM (qkv + z + beta + alpha) for the
// Qwen3.5 DeltaNet LA preamble.
pub const GEMM_QKVZA_HFP4G32_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfp4g32_wmma.hip");
pub const GEMM_QKVZA_HFP4G32_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfp4g32_wmma.gfx12.hip");

// HFP4-G32 grouped-WMMA-GEMM for MoE prefill on gfx12. Sister of the
// HFQ4 grouped variant — same tile geometry / expert_tile_ids sentinel
// pattern, with HFP4G32's 18 B/group dequant (1 B UE8M0 + 16 B packed
// FP4 + LUT) and the G32 inner loop. MFP4G32 shares storage with HFP4G32.
pub const GEMM_HFP4G32_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfp4g32_moe_grouped_wmma.gfx12.hip");

// Batched 4-way fused HFQ4-G256 GEMM (LA preamble: wqkv + wz + w_beta + w_alpha).
// Batched counterpart of fused_qkvza_hfq4g256 — byte-exact vs running that kernel
// N times on the same x[b]. Used for batched prefill of the LA layer projection.
pub const GEMM_QKVZA_HFQ4G256_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256.hip");
// CDNA3 wave64-native batched 4-way fused LA GEMM. 2 rows per block via
// warp_id, halves grid.x. Byte-exact with wave32 base. Hottest DFlash
// verify kernel on MI300X — targeted first for this port.
pub const GEMM_QKVZA_HFQ4G256_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_wave64.hip");
// GCN5/CDNA1 wave64 FP16 hybrid 4-way fused LA GEMM. Same __hfma2
// inner loop as the FP16 variant, but block=[64,1,1] with 2 rows/block.
// Scoped to gfx906/gfx908 — CDNA3 uses the rocBLAS MFMA path instead.
pub const GEMM_QKVZA_HFQ4G256_FP16_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_fp16_wave64.hip");
// FP16 packed variant — RDNA1/2 fast path (no WMMA available).
pub const GEMM_QKVZA_HFQ4G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_fp16.hip");
// v_dot2_f32_f16 variant — emits v_dot2_f32_f16 on gfx1011/1012/1030-1032 and gfx11/12.
pub const GEMM_QKVZA_HFQ4G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_dot2.hip");

// Batched 4-way fused HFQ3-G256 GEMM. HFQ3 sibling of GEMM_QKVZA_HFQ4G256_SRC;
// same dispatch shape, 104 B group stride, 3-bit unpack. Wired in alongside
// GEMM_QKV_HFQ3G256_SRC / GEMM_GATE_UP_HFQ3G256_SRC for the gfx10 MQ3
// prefill path on dense Qwen3.5 LA layers.
pub const GEMM_QKVZA_HFQ3G256_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256.hip");
// v_dot2_f32_f16 variant — HFQ3 sibling of GEMM_QKVZA_HFQ4G256_DOT2_SRC.
pub const GEMM_QKVZA_HFQ3G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_dot2.hip");
// FP16-packed (v_pk_fma_f16) variant — fallback for gfx1010/1013.
pub const GEMM_QKVZA_HFQ3G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_fp16.hip");

// Batched 3-way fused HFQ4-G256 GEMM (FA preamble: wq + wk + wv).
// Batched counterpart of fused_qkv_hfq4g256 — byte-exact vs running that kernel
// N times on the same x[b]. Used for batched prefill of the FA layer projection.
pub const GEMM_QKV_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq4g256.hip");
// CDNA3 wave64-native batched 3-way fused FA preamble. 2 rows per block.
pub const GEMM_QKV_HFQ4G256_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_wave64.hip");
// GCN5/CDNA1 wave64 FP16 hybrid 3-way fused FA GEMM. Same __hfma2
// inner loop as the FP16 variant, but block=[64,1,1] with 2 rows/block.
// Scoped to gfx906/gfx908 — CDNA3 uses the rocBLAS MFMA path instead.
pub const GEMM_QKV_HFQ4G256_FP16_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_fp16_wave64.hip");
// FP16 packed variant — RDNA1/2 fast path (no WMMA available).
pub const GEMM_QKV_HFQ4G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_fp16.hip");
// v_dot2_f32_f16 variant — emits v_dot2_f32_f16 on gfx1011/1012/1030-1032 and gfx11/12.
pub const GEMM_QKV_HFQ4G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_dot2.hip");

// Batched 3-way fused HFQ3-G256 GEMM. HFQ3 sibling of GEMM_QKV_HFQ4G256_SRC
// with 104 B group stride and 3-bit unpack via uint24 byte-combine.
// Wired into the gfx10 MQ3 prefill path when `is_batchable_la(MQ3G256, arch)`
// admits the format (non-WMMA archs only).
pub const GEMM_QKV_HFQ3G256_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq3g256.hip");
// v_dot2_f32_f16 variant — HFQ3 sibling of GEMM_QKV_HFQ4G256_DOT2_SRC.
// Emits v_dot2_f32_f16 on gfx1011/1012/1030-1032 and gfx11/12 — NOT on gfx1010
// (5700 XT, Navi 10) or gfx1013 (BC-250 APU), which lack the dot extension.
pub const GEMM_QKV_HFQ3G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_dot2.hip");
// FP16-packed (v_pk_fma_f16) variant — fallback for gfx1010/1013 which lack dot2.
pub const GEMM_QKV_HFQ3G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_fp16.hip");
// Wave32+dp4a (v_dot4_i32_i8) variant — gfx1030+ experimental path (Phase 2).
// Port of gemm_qkv_hfq4g256_wave64_dp4a from gfx906 to wave32 + HFQ3 unpack.
pub const GEMM_QKV_HFQ3G256_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_dp4a.gfx1030.hip");
// Wave32 MMQ residual family — gfx1030+ Phase 3 probe. Tile-size templates
// over the shared body in `gemm_hfq3g256_residual_mmq_body.cuh`. The body
// defines MMQ_Y=128 rows × MMQ_X cols per workgroup with LDS-tiled X reuse;
// each instantiation specializes MMQ_X for a batch-size regime.
//   x8  → batch ≤ ~12 (short prefill, tile granularity dominates)
//   x16 → batch 12-24 (mid-prefill)
//   x32 → batch ≥ 24  (long prefill, b128 LDS path is profitable)
//
// The body is included via the shared `_BODY_CUH` const at dispatch time
// (string-replace into each tile wrapper) — the runtime hipcc compile
// happens in cache_dir which doesn't have kernels/src on its -I path.
// Same pattern as the gfx906 MMQ family.
pub const GEMM_HFQ3G256_RESIDUAL_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_mmq_body.cuh");
pub const GEMM_HFQ3G256_RESIDUAL_MMQ_X8_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_mmq_x8.gfx1030.hip");
pub const GEMM_HFQ3G256_RESIDUAL_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_mmq_x16.gfx1030.hip");
pub const GEMM_HFQ3G256_RESIDUAL_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_mmq_x32.gfx1030.hip");
pub const GEMM_HFQ3G256_RESIDUAL_MMQ_X32_Y64_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_mmq_x32_y64.gfx1030.hip");
pub const GEMM_HFQ3G256_RESIDUAL_MMQ_X32_Y32_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq3g256_residual_mmq_x32_y32.gfx1030.hip");

// HFQ3 qkv (3-way fused: Q + K + V) MMQ family. Same body template,
// different output routing.
pub const GEMM_QKV_HFQ3G256_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_mmq_body.cuh");
pub const GEMM_QKV_HFQ3G256_MMQ_X8_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_mmq_x8.gfx1030.hip");
pub const GEMM_QKV_HFQ3G256_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_mmq_x16.gfx1030.hip");
pub const GEMM_QKV_HFQ3G256_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq3g256_mmq_x32.gfx1030.hip");

// HFQ3 gate_up (2-way fused: gate + up) MMQ family.
pub const GEMM_GATE_UP_HFQ3G256_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_mmq_body.cuh");
pub const GEMM_GATE_UP_HFQ3G256_MMQ_X8_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_mmq_x8.gfx1030.hip");
pub const GEMM_GATE_UP_HFQ3G256_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_mmq_x16.gfx1030.hip");
pub const GEMM_GATE_UP_HFQ3G256_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_mmq_x32.gfx1030.hip");
pub const GEMM_GATE_UP_HFQ3G256_MMQ_X32_Y64_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_mmq_x32_y64.gfx1030.hip");
pub const GEMM_GATE_UP_HFQ3G256_MMQ_X32_Y96_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_mmq_x32_y96.gfx1030.hip");

// HFQ3 qkvza (4-way fused: wqkv + wz + w_beta + w_alpha — LinearAttention
// preamble) MMQ family.
pub const GEMM_QKVZA_HFQ3G256_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_mmq_body.cuh");
pub const GEMM_QKVZA_HFQ3G256_MMQ_X8_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_mmq_x8.gfx1030.hip");
pub const GEMM_QKVZA_HFQ3G256_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_mmq_x16.gfx1030.hip");
pub const GEMM_QKVZA_HFQ3G256_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq3g256_mmq_x32.gfx1030.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_RDNA2_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq.gfx1030.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_body.cuh");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_x16.gfx1030.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_x32.gfx1030.hip");
pub const GEMM_HFQ4G256_RESIDUAL_MMQ_X32_Y64_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq_x32_y64.gfx1030.hip");
pub const GEMM_QKV_HFQ4G256_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_body.cuh");
pub const GEMM_QKV_HFQ4G256_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_x16.gfx1030.hip");
pub const GEMM_QKV_HFQ4G256_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq4g256_mmq_x32.gfx1030.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_body.cuh");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_x16.gfx1030.hip");
pub const GEMM_GATE_UP_HFQ4G256_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_mmq_x32.gfx1030.hip");
pub const GEMM_QKVZA_HFQ4G256_MMQ_BODY_CUH: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_mmq_body.cuh");
pub const GEMM_QKVZA_HFQ4G256_MMQ_X16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_mmq_x16.gfx1030.hip");
pub const GEMM_QKVZA_HFQ4G256_MMQ_X32_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_mmq_x32.gfx1030.hip");

// Batched 2-way fused HFQ4-G256 GEMM (FFN preamble: w_gate + w_up).
// Batched counterpart of fused_gate_up_hfq4g256 — byte-exact vs running that kernel
// N times on the same x[b]. Used for batched prefill of the FFN gate/up projections.
pub const GEMM_GATE_UP_HFQ4G256_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256.hip");
// GCN5/CDNA1 wave64 FP16 hybrid 2-way fused FFN GEMM. Same __hfma2
// inner loop as the FP16 variant, but block=[64,1,1] with 2 rows/block.
// Scoped to gfx906/gfx908 — CDNA3 uses the rocBLAS MFMA path instead.
pub const GEMM_GATE_UP_HFQ4G256_FP16_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_fp16_wave64.hip");
// FP16 packed variant — RDNA1/2 fast path (no WMMA available).
pub const GEMM_GATE_UP_HFQ4G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_fp16.hip");
// v_dot2_f32_f16 variant — emits v_dot2_f32_f16 on gfx1011/1012/1030-1032 and gfx11/12.
// Does NOT work on gfx1010 (5700 XT) or gfx1013 (BC-250 APU) — lack dot instructions.
pub const GEMM_GATE_UP_HFQ4G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_dot2.hip");

// Batched 2-way fused HFQ3-G256 GEMM. HFQ3 sibling of GEMM_GATE_UP_HFQ4G256_SRC;
// same dispatch shape, 104 B group stride, 3-bit unpack. Wired in alongside
// GEMM_QKV_HFQ3G256_SRC for the gfx10 MQ3 prefill path.
pub const GEMM_GATE_UP_HFQ3G256_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256.hip");
// v_dot2_f32_f16 variant — HFQ3 sibling of GEMM_GATE_UP_HFQ4G256_DOT2_SRC.
pub const GEMM_GATE_UP_HFQ3G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_dot2.hip");
// FP16-packed (v_pk_fma_f16) variant — fallback for gfx1010/1013.
pub const GEMM_GATE_UP_HFQ3G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_fp16.hip");
// Wave32+dp4a (v_dot4_i32_i8) variant — gfx1030+ experimental path (Phase 2).
pub const GEMM_GATE_UP_HFQ3G256_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq3g256_dp4a.gfx1030.hip");

// ── HFQ6-G256 batched GEMM (for MQ6 prefill) ──
pub const GEMM_HFQ6G256_RESIDUAL_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_residual.hip");
pub const GEMM_HFQ6G256_RESIDUAL_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_residual_fp16.hip");
pub const GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_residual_wmma_k2.hip");
// gfx12 (RDNA4) sister of GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC. Pure composition
// of validated patterns — hfq6 dequant (gemm_qkv_hfq6g256_wmma.gfx12.hip) +
// fused residual `+=` (gemm_q8_0_residual_wmma.gfx12.hip).
pub const GEMM_HFQ6G256_RESIDUAL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq6g256_residual_wmma.gfx12.hip");
pub const GEMM_QKVZA_HFQ6G256_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq6g256.hip");
pub const GEMM_QKVZA_HFQ6G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_fp16.hip");
pub const GEMM_QKVZA_HFQ6G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_dot2.hip");
pub const GEMM_QKVZA_HFQ6G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_wmma.hip");
// gfx12 (RDNA4) sister: pure composition of validated patterns —
// hfq6 dequant + 4-output qkv/z/beta/alpha routing.
pub const GEMM_QKVZA_HFQ6G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_wmma.gfx12.hip");
pub const GEMM_QKV_HFQ6G256_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq6g256.hip");
pub const GEMM_QKV_HFQ6G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq6g256_fp16.hip");
pub const GEMM_QKV_HFQ6G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq6g256_dot2.hip");
pub const GEMM_QKV_HFQ6G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq6g256_wmma.hip");
// gfx12 (RDNA4) sister of GEMM_QKV_HFQ6G256_WMMA_SRC. Same gfx12 recipe
// as the hfq4 scaffolds, with the hfq6 dequant inner loop carried over
// (200B groups, 4-byte unaligned reads at byte-offsets {0, 3} per K
// half-tile to extract 8 6-bit values per lane).
pub const GEMM_QKV_HFQ6G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_hfq6g256_wmma.gfx12.hip");
pub const GEMM_GATE_UP_HFQ6G256_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq6g256.hip");
pub const GEMM_GATE_UP_HFQ6G256_FP16_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_fp16.hip");
pub const GEMM_GATE_UP_HFQ6G256_DOT2_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_dot2.hip");
pub const GEMM_GATE_UP_HFQ6G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_wmma.hip");
// gfx12 (RDNA4) sister: combines the hfq6 dequant inner loop (validated
// in gemm_qkv_hfq6g256_wmma.gfx12.hip) with the 2-output gate/up
// routing (validated in gemm_gate_up_hfq4g256_wmma.gfx12.hip).
pub const GEMM_GATE_UP_HFQ6G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_wmma.gfx12.hip");

// Multi-row GEMV variants: one warp computes R output rows at a time, sharing
// x register state across rows. Exposes R=2, R=4, R=8 extern "C" entry points
// from one source file. See kernel header for VGPR budget details.
pub const GEMV_HFQ4G256_MULTIROW_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.gfx1100.hip");
pub const GEMV_HFQ4G256_MULTIROW_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
/// Exact gfx1151 LM-head shape specialization with the established global
/// load path unchanged. This isolates compile-time K=2048 loop/tail/address
/// simplification from the independently rejected buffer-load experiments.
pub const GEMV_HFQ4G256_MULTIROW_K2048_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX1151_LM_HEAD_K2048 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
/// gfx1151 LM-head R2 candidate: retain the existing exact dual-row schedule
/// and lower its dominant matrix stream through temporal raw-buffer VMEM.
/// The selector constrains this artifact to M=248320, K=2048 on gfx1151.
pub const GEMV_HFQ4G256_MULTIROW_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
/// Hybrid gfx1151 LM-head lowering: scalar/global uniform headers plus
/// temporal buffer VMEM for the lane-divergent packed word.
pub const GEMV_HFQ4G256_MULTIROW_HYBRID_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
/// gfx1151 streaming LM-head probes. The 270 MB output matrix is consumed
/// once per token, so keep its lane-divergent packed weights on raw-buffer
/// VMEM while testing whether GLC/SLC/DLC prevents the stream from displacing
/// higher-reuse decode state. Uniform headers remain on the scalar path.
pub const GEMV_HFQ4G256_MULTIROW_HYBRID_GLC_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 1\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
pub const GEMV_HFQ4G256_MULTIROW_HYBRID_SLC_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
pub const GEMV_HFQ4G256_MULTIROW_HYBRID_DLC_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 4\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
/// Full gfx1151 LM-head specialization: fixed K removes tail machinery,
/// uniform headers remain scalar, packed words use temporal buffer VMEM, and
/// aligned activation vectors form an explicit B128 clause.
pub const GEMV_HFQ4G256_MULTIROW_ALL_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_X_BUFFER 1\n",
    "#define HIPFIRE_GFX1151_LM_HEAD_K2048 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip")
);
pub const GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_multirow.gfx1100.hip");
/// Exact gfx1151 residual R2 candidate. The body is shared source, but the
/// exported symbols and JIT module are gfx1151-specific so neither the code
/// object cache nor a retained replay tape can alias the gfx1100 artifact.
pub const GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_RESIDUAL_MULTIROW_R2_KERNEL gemv_hfq4g256_residual_multirow_r2_gfx1151\n",
    "#define HIPFIRE_RESIDUAL_MULTIROW_R4_KERNEL gemv_hfq4g256_residual_multirow_r4_gfx1151\n",
    "#define HIPFIRE_RESIDUAL_MULTIROW_R8_KERNEL gemv_hfq4g256_residual_multirow_r8_gfx1151\n",
    include_str!("../../../kernels/src/gemv_hfq4g256_residual_multirow.gfx1100.hip")
);

// 4-way fused HFQ4-G256 projection for Qwen3.5 DeltaNet LA preamble:
// wqkv + wz + w_beta + w_alpha in a single launch. Same 4x-unroll inner
// loop as gemv_hfq4g256.hip; grid = sum of the four projections' output
// row counts. Works on every RDNA generation — see the kernel header.
pub const FUSED_QKVZA_HFQ4G256_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// Exact gfx1151 A3B candidate: preserve the generic one-row schedule while
/// making K=2048 constant and lowering weights through temporal raw-buffer
/// loads. This deliberately does not inherit gfx1100's R2 or SLC policy.
pub const FUSED_QKVZA_HFQ4G256_K2048_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_buffer_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// Faithful gfx1151 Radiowave cache candidate: temporal raw-buffer weights and
/// aligned B128 temporal activation loads. The arithmetic, launch geometry,
/// accumulator ownership, and reduction tree remain identical to the fixed-K
/// one-row control.
pub const FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_X_BUFFER 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// gfx1151 small-QKVZA hybrid cache path: immutable wave-uniform scale/zero
/// headers use scalar B64 loads, while packed weights and the shared activation
/// retain the admitted temporal buffer lowering. This removes redundant
/// per-lane header VMEM without changing row ownership or the arithmetic tree.
pub const FUSED_QKVZA_HFQ4G256_K2048_HYBRID_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_X_BUFFER 1\n",
    "#define HIPFIRE_GFX1151_QKVZA_HYBRID_BUFFER 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// gfx1151 selective QKVZA lowering for the large projection geometry. Keep
/// HFQ weights on global loads and issue only the shared K=2048 activation as
/// aligned temporal B128 buffer loads.
pub const FUSED_QKVZA_HFQ4G256_K2048_X_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_X_BUFFER 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// Cache-policy sweep for the gfx1151 all-buffer QKVZA lowering. The
/// activation stream remains temporal; only the much larger weight stream
/// changes policy so replay-level cache contamination can be measured without
/// changing arithmetic, geometry, or instruction addressing.
pub const FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_GLC_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_X_BUFFER 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

pub const FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_SLC_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_X_BUFFER 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

pub const FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_DLC_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 4\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_X_BUFFER 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// gfx1151 temporal-buffer candidate with pairwise load/consume scheduling.
/// It retains two groups of VMEM overlap while avoiding the four-group
/// artifact's higher VGPR band.
pub const FUSED_QKVZA_HFQ4G256_K2048_PAIR_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_PAIR_BUFFER 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// Exact gfx1151 A3B candidate: expose the fixed K=2048 contract and hoist
/// the complete 32-value activation quad before the weight/FMA stream. Keep
/// ordinary global weight loads: temporal raw-buffer addressing was neutral
/// in isolation on gfx1151 and must not be coupled to this codegen probe.
pub const FUSED_QKVZA_HFQ4G256_K2048_HOIST_X32_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_HOIST_X32 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_hoist_x32_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

pub const FUSED_QKVZA_HFQ4G256_HOIST_X32_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_HOIST_X32 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

// Exact gfx1100 A3B decode specialization. The runtime signature remains
// unchanged for replay compatibility; host routing enforces K=2048 while the
// source-level constant lets LLVM delete the general loop and tail machinery.
pub const FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

// Muse Glimmer gfx1100 AR decode: one fused Q+K+V+attention_gate HFQ4/MQ4
// projection at exact shape q=4096, k=256, v=256, attention_gate=4096, K=6656
// (26 groups). Host enforces dimensions and selects this identity; reuses the
// generic fused_qkvza_hfq4g256 arithmetic template with fixed groups_per_row.
// No buffer-load or experimental schedule opts — weight-cache policy only.
pub const FUSED_GLIMMER_QKVG_HFQ4G256_K6656_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_GLIMMER_QKVG_K6656 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_glimmer_qkvg_hfq4g256_k6656_gfx1100\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// Radiowave gfx1100 dual-row QKVZA probe. Two adjacent rows share each
/// K=2048 activation load while preserving independent weight streams and the
/// shipping four-chain reduction order.
pub const FUSED_QKVZA_HFQ4G256_K2048_R2_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_r2.gfx1100.hip")
);

/// gfx1151 Radiowave dual-row candidate: retain the exact K=2048 two-row
/// activation reuse schedule while lowering both independent weight streams
/// through ordinary temporal buffer VMEM. The exact-architecture runtime gate
/// keeps this from changing gfx1100's admitted R2 artifact or gfx12 codegen.
pub const FUSED_QKVZA_HFQ4G256_K2048_R2_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_r2.gfx1100.hip")
);

/// gfx1151 large-QKVZA four-row stream tile. One wave reuses each eight-value
/// activation stripe across four adjacent weight rows while retaining the
/// shipping four-chain arithmetic independently for every output row.
pub const FUSED_QKVZA_HFQ4G256_K2048_R4_STREAM_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_r4_stream.gfx1151.hip")
);

/// gfx1151-only wave64 activation-sharing experiment. Matching lanes in the
/// two physical wave32 halves consume one K=2048 activation load stream via
/// V_PERMLANE64 while retaining independent row weights and reductions.
pub const FUSED_QKVZA_HFQ4G256_WAVE64_SHARE_X_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_wave64_share_x.gfx1151.hip");

/// gfx1100 decode experiment: keep the certified K=2048 projection schedule
/// and fold the beta sigmoid plus alpha softplus/decay preparation into the
/// two scalar projection tails. This removes one launch per DeltaNet layer.
pub const FUSED_QKVZA_HFQ4G256_K2048_SCALAR_PREP_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_QKVZA_SCALAR_PREP 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_scalar_prep\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// Radiowave gfx1100 QKVZA probe: retain the certified fixed-K arithmetic and
/// apply the SLC raw-buffer policy that won on the analogous MoE gate-up weight
/// stream. Kept as a distinct replay-visible artifact until model certification.
pub const FUSED_QKVZA_HFQ4G256_K2048_CPOL_SLC_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_k2048_cpol_slc\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

pub const FUSED_QKVZA_HFQ4G256_2WAVE_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_2wave.gfx1100.hip")
);

// Four independent wave32 rows per workgroup. Unlike the 2wave experiment,
// this preserves one full wave and the base reduction tree for every row;
// only the workgroup packaging changes.
pub const FUSED_QKVZA_HFQ4G256_WAVEPACK4_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_QKVZA_WAVES_PER_BLOCK 4\n",
    "#define HIPFIRE_QKVZA_BLOCK_SIZE 128\n",
    "#define HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU 4\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_wavepack4\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

// Eight exact row waves share one bank-conflict-free LDS activation tile.
// Exact gfx1100/K=2048 experiment; host routing enforces the fixed-K contract.
pub const FUSED_QKVZA_HFQ4G256_LDSX8_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_LDSX8 1\n",
    "#define HIPFIRE_QKVZA_WAVES_PER_BLOCK 8\n",
    "#define HIPFIRE_QKVZA_BLOCK_SIZE 256\n",
    "#define HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU 2\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_ldsx8\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

/// gfx1151 producer/consumer QKVZA candidate. Eight exact row waves share one
/// bank-conflict-free LDS activation tile, while the independent weight streams
/// retain Radiowave temporal buffer addressing. Fixed K=2048 keeps the loop and
/// resource geometry architecture-local and replay-stable.
pub const FUSED_QKVZA_HFQ4G256_LDSX8_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKVZA_LDSX8 1\n",
    "#define HIPFIRE_QKVZA_WAVES_PER_BLOCK 8\n",
    "#define HIPFIRE_QKVZA_BLOCK_SIZE 256\n",
    "#define HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU 2\n",
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_ldsx8_buffer_gfx1151\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

pub const FUSED_QKVZA_HFQ4G256_REDUCE_CHAIN_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256_reduce_chain\n",
    "#define HIPFIRE_RDNA3_QKVZA_REDUCE_CHAIN 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip")
);

// CDNA3 (MI300X / gfx94x) wave64-native counterpart: block=[64,1,1] with
// two fused-qkvza rows per block (one per warp). Grid halves from total_m
// to (total_m+1)/2. Byte-exact vs the wave32 base kernel.
pub const FUSED_QKVZA_HFQ4G256_WAVE64_SRC: &str = concat!(
    "// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n",
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_wave64.hip")
);
// gfx906 dp4a-port — see fused_gate_up_hfq4g256_wave64_dp4a.hip for the
// math derivation and lane-mapping invariants.
pub const FUSED_QKVZA_HFQ4G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/fused_qkvza_hfq4g256_wave64_dp4a.hip");

// 3-way fused HFQ4-G256 projection for Qwen3.5 FullAttention preamble:
// wq + wk + wv in a single launch. Same 4x-unroll inner loop as the LA
// variant; grid = q_m + k_m + v_m. Cross-arch.
pub const FUSED_QKV_HFQ4G256_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkv_hfq4g256.hip")
);
pub const FUSED_QKV_HFQ4G256_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_hfq4g256_qwen2_bias\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkv_hfq4g256.hip")
);

/// gfx1151 fixed-K FullAttention projection with Radiowave temporal buffer
/// lowering for weights and aligned activation vectors.
pub const FUSED_QKV_HFQ4G256_K2048_ALL_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 0\n",
    "#define HIPFIRE_RDNA3_QKV_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKV_X_BUFFER 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_hfq4g256_k2048_all_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkv_hfq4g256.hip")
);

/// gfx1151 selective Radiowave lowering: retain the established global weight
/// stream and use temporal B128 buffer loads only for the small, repeatedly
/// consumed K=2048 activation vector.
pub const FUSED_QKV_HFQ4G256_K2048_X_BUFFER_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_RDNA3_QKV_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKV_X_BUFFER 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_hfq4g256_k2048_x_buffer_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkv_hfq4g256.hip")
);

/// Same gfx1151 addressing and activation schedule, with SLC on the large
/// weight stream to reduce pollution of replay-resident data.
pub const FUSED_QKV_HFQ4G256_K2048_ALL_BUFFER_SLC_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN 1\n",
    "#define HIPFIRE_WEIGHT_CPOL_AUX 2\n",
    "#define HIPFIRE_RDNA3_QKV_K2048 1\n",
    "#define HIPFIRE_RDNA3_QKV_X_BUFFER 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_hfq4g256_k2048_all_buffer_slc_gfx1151\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_qkv_hfq4g256.hip")
);

// Per-row 3-way fused HFQ6/MQ6-G256 preamble (qwen2-family decode). Authored
// from gemv_hfq6g256's row body + qkv routing + optional bias epilogue.
pub const FUSED_QKV_HFQ6G256_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_hfq6g256.hip");

// CDNA3 (MI300X / gfx94x) wave64-native 3-way fused preamble — 2 rows per
// block via warp_id, halved grid. Byte-exact with the wave32 base kernel.
pub const FUSED_QKV_HFQ4G256_WAVE64_SRC: &str = concat!(
    "// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n",
    include_str!("../../../kernels/src/fused_qkv_hfq4g256_wave64.hip")
);
pub const FUSED_QKV_HFQ4G256_WAVE64_QWEN2_BIAS_SRC: &str = concat!(
    "// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n",
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_hfq4g256_wave64_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_hfq4g256_wave64.hip")
);
// gfx906 dp4a-port — see fused_gate_up_hfq4g256_wave64_dp4a.hip for the
// math derivation and lane-mapping invariants.
pub const FUSED_QKV_HFQ4G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/fused_qkv_hfq4g256_wave64_dp4a.hip");
// Note: 2-way fused gate+up uses the existing FUSED_GATE_UP_HFQ4G256_SRC
// constant declared further down (kernels/src/fused_gate_up_hfq4g256.hip).
pub const GEMV_HFQ4G256_GFX1030_V1_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v1.hip");
pub const GEMV_HFQ4G256_GFX1030_V2_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v2.hip");
pub const GEMV_HFQ4G256_GFX1030_V3_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v3.hip");
pub const GEMV_HFQ4G256_GFX1030_V4_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v4.hip");
pub const GEMV_HFQ4G256_GFX1030_V5_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v5.hip");

/// Returns the HFQ4-G256 GEMV kernel source AND module name for the given arch.
/// On gfx1030/gfx1031 (RDNA2), selects variant via HIPFIRE_RDNA2_VARIANT env var.
/// Module name is variant-specific so each variant gets its own precompiled .hsaco blob.
/// The function name inside the .hsaco is always "gemv_hfq4g256" (the extern "C" symbol).
pub fn gemv_hfq4g256_for_arch(
    caps: &ArchCaps,
    rdna2_variant: Option<u32>,
) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1030" | "gfx1031" => {
            let variant: u32 = rdna2_variant.unwrap_or(1);
            match variant {
                2 => (GEMV_HFQ4G256_GFX1030_V2_SRC, "gemv_hfq4g256_rdna2v2"),
                3 => (GEMV_HFQ4G256_GFX1030_V3_SRC, "gemv_hfq4g256_rdna2v3"),
                4 => (GEMV_HFQ4G256_GFX1030_V4_SRC, "gemv_hfq4g256_rdna2v4"),
                5 => (GEMV_HFQ4G256_GFX1030_V5_SRC, "gemv_hfq4g256_rdna2v5"),
                _ => (GEMV_HFQ4G256_GFX1030_V1_SRC, "gemv_hfq4g256_rdna2v1"),
            }
        }
        "gfx1100" | "gfx1101" | "gfx1102" => (GEMV_HFQ4G256_GFX1100_SRC, "gemv_hfq4g256_rdna3"),
        // RDNA4 variants (existing)
        // "gfx1200" | "gfx1201" => ...,
        _ => (GEMV_HFQ4G256_SRC, "gemv_hfq4g256"), // gfx1010 baseline
    }
}

/// HFP4-G32 GEMV arch dispatch.
///
/// v1: gfx1100 variant is the byte-exact baseline (currently bit-identical to the
/// default source; v2 adds VOPD + V_PERMLANE16 + SGPR-LUT here). All other archs
/// route to the default source — same FP add ordering and accumulator structure
/// guarantees byte-exact output across gfx1010, gfx1030, gfx1151, gfx1201, gfx906.
/// gfx1201 WMMA-FP8 hero kernel ships in v2. See `docs/quant-formats/hfp4.md`.
pub fn gemv_hfp4g32_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => {
            (GEMV_HFP4G32_GFX1100_SRC, "gemv_hfp4g32_rdna3")
        }
        _ => (GEMV_HFP4G32_SRC, "gemv_hfp4g32"),
    }
}

/// Same arch dispatch as `gemv_hfq4g256_for_arch` but returns the residual
/// variant (y[row] += A[row] · x instead of y[row] = ...). RDNA2 variants
/// fall back to the baseline residual kernel for now.
pub fn gemv_hfq4g256_residual_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMV_HFQ4G256_RESIDUAL_GFX1100_SRC,
            "gemv_hfq4g256_residual_rdna3",
        ),
        _ => (GEMV_HFQ4G256_RESIDUAL_SRC, "gemv_hfq4g256_residual"),
    }
}

/// Returns the HFQ3-G256 GEMV kernel source AND module name for the given arch.
/// gfx1100/1101/1102 (RDNA3) gets the K4-unrolled 4-accumulator variant that
/// closes the per-launch perf gap with MQ4. Other archs use the baseline.
pub fn gemv_hfq3g256_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => (GEMV_HFQ3G256_GFX1100_SRC, "gemv_hfq3g256_rdna3"),
        _ => (GEMV_HFQ3G256_SRC, "gemv_hfq3g256"),
    }
}

/// Same arch dispatch as `gemv_hfq3g256_for_arch` but returns the residual
/// variant (y[row] += A[row] · x). Used by `weight_gemv_residual` MQ3 arm
/// to eliminate the alloc+gemv+add+free fallback chain.
pub fn gemv_hfq3g256_residual_for_arch(caps: &ArchCaps) -> (&'static str, &'static str) {
    let arch = caps.arch();
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => (
            GEMV_HFQ3G256_RESIDUAL_GFX1100_SRC,
            "gemv_hfq3g256_residual_rdna3",
        ),
        _ => (GEMV_HFQ3G256_RESIDUAL_SRC, "gemv_hfq3g256_residual"),
    }
}

/// HFQ2-G128: flat 2-bit with 128-weight groups. Finer granularity than G256.
/// [f32 scale (4B)][f32 zero (4B)][2-bit × 128 (32B)] = 40 bytes per 128 weights (0.3125 B/w).
/// 32 threads × 4 elements = 128 per group. Each thread reads 1 byte.
pub const GEMV_HFQ2G128_SRC: &str = include_str!("../../../kernels/src/gemv_hfq2g128.hip");

/// HFQ4-G256 wide GEMV: 2 rows per block (64 threads = 2 warps).
/// Each warp processes one row independently. Halves grid size.
pub const GEMV_HFQ4G256_WIDE_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_wide.hip");

/// HFQ4-G256 batched GEMM: y[batch][row] = sum_k(A[row][k] * x[batch][k])
/// Loads weight data ONCE per group, multiplies against BATCH_TILE input vectors.
/// Grid: [M, ceil(batch_size/BATCH_TILE), 1]. Each block handles one row × BATCH_TILE batch elements.
/// x layout: [batch_size × K] row-major. y layout: [batch_size × M] row-major.
/// BATCH_TILE=8 keeps register pressure at ~26 VGPRs for good occupancy on RDNA.
pub const GEMM_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256.hip");
// CDNA3 wave64-native batched HFQ4-G256 GEMM (overwrite). 2 rows per block.
pub const GEMM_HFQ4G256_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_wave64.hip");
// gfx906 dp4a-port — see kernels/src/gemm_hfq4g256_wave64_dp4a.hip for the
// math + lane-mapping invariants. Targets the LM-head batched GEMM that
// PMC at 2026-05-06 showed was 17.0 % of DFlash 27B steady-state decode
// time on the FP wave64 path.
pub const GEMM_HFQ4G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_wave64_dp4a.hip");

/// One-shot dequantize HFQ4-G256 matrix → FP16 row-major. Used when the
/// downstream prefill GEMM path uses rocBLAS MFMA kernels (CDNA3 only —
/// the FP16 shadow is 4× the MQ4 size, so the engine only allocates it on
/// large-VRAM GPUs). Launch grid = (M, K/256, 1), block = (128, 1, 1).
pub const HFQ4G256_DEQUANTIZE_TO_F16_SRC: &str =
    include_str!("../../../kernels/src/hfq4g256_dequantize_to_f16.hip");

/// Fused QKV Q4_K: three GEMVs in one kernel launch.
/// Grid = (q_m + k_m + v_m) blocks. Each block determines which matrix by blockIdx range.
/// All three projections read the same input x (cached). Saves 2 kernel launches per layer.
pub const FUSED_QKV_Q4K_SRC: &str = include_str!("../../../kernels/src/fused_qkv_q4k.hip");
pub const FUSED_QKV_Q4K_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_q4k_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_q4k.hip")
);

/// Fused Gate+Up Q4_K: two GEMVs in one kernel launch for FFN gate and up projections.
/// Grid = (gate_m + up_m) blocks. Saves 1 kernel launch per layer.
pub const FUSED_GATE_UP_Q4K_SRC: &str = include_str!("../../../kernels/src/fused_gate_up_q4k.hip");

/// GEMV Q8_0: matrix-vector multiply with on-the-fly Q8_0 dequantization.
/// Q8_0 block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements.
/// v3: Processes 8 blocks (256 elements) per outer iteration to match Q4_K's loop count.
/// Byte loads → no nibble extraction → 16 VGPRs → F32-class occupancy.
/// Q8_0 GEMV wide: 256 threads with shared memory reduction for small matrices.
/// Each thread processes K/256 elements strided, then tree-reduce via shared memory.
/// Better for dim=1024 where 32-thread kernel underutilizes the GPU.
pub const GEMV_Q8_0_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_q8_0_wide.hip");
pub const GEMM_Q8_0_BATCHED_WIDE_EXACT_SRC: &str =
    include_str!("../../../kernels/src/gemm_q8_0_batched_wide_exact.hip");

pub const GEMV_Q8_0_SRC: &str = include_str!("../../../kernels/src/gemv_q8_0.hip");

/// Batched Q8_0 GEMM. Same per-row math as gemv_q8_0 but holds MAX_BATCH
/// per-row accumulators in registers, broadcasting each weight load across
/// all batch elements. Saves the (batch_size - 1)× weight re-reads of the
/// serial-GEMV loop for DFlash lm_heads.
pub const GEMM_Q8_0_BATCHED_SRC: &str = include_str!("../../../kernels/src/gemm_q8_0_batched.hip");

/// WMMA-accelerated 3-way fused QKV GEMM for Q8_0 weights. gfx1100+ wave32.
/// Recipe-selected per docs/plans/q8-fused-prefill-kernels.md T3-1a microbench
/// (FP16-WMMA, register-redundant dequant, no LDS).
pub const GEMM_QKV_Q8_0_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_q8_0_wmma.hip");

/// WMMA 4-way fused qkv+z+beta+alpha GEMM for Q8_0 (DeltaNet LA preamble).
pub const GEMM_QKVZA_Q8_0_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_q8_0_wmma.hip");

/// WMMA 2-way fused gate+up GEMM for Q8_0 (FFN preamble).
pub const GEMM_GATE_UP_Q8_0_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_q8_0_wmma.hip");

/// WMMA Q8_0 GEMM with fused residual add (wo, w_down post-projection).
pub const GEMM_Q8_0_RESIDUAL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_q8_0_residual_wmma.hip");

// gfx12 (RDNA4) sister of GEMM_QKV_Q8_0_WMMA_SRC. Uses
// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (vs the gfx11 `_w32`)
// and half8_t operands (vs half16_t). Lane-grp K split (tid>>4 selects
// K-half) and `acc[j] = C[8*(tid>>4) + j][tid & 15]` C-output mapping —
// pattern mirrors gemm_qkv_hfq4g256_wmma.gfx12 and is silicon-validated
// on R9700 (test_gemm_q8_qkv_wmma 22/22 PASS, 2026-05-14).
pub const GEMM_QKV_Q8_0_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkv_q8_0_wmma.gfx12.hip");

/// gfx12 sister of GEMM_QKVZA_Q8_0_WMMA_SRC. Same lane-grp + half8_t
/// pattern as the QKV gfx12 sibling.
pub const GEMM_QKVZA_Q8_0_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_qkvza_q8_0_wmma.gfx12.hip");

/// gfx12 sister of GEMM_GATE_UP_Q8_0_WMMA_SRC. Same lane-grp + half8_t
/// pattern as the QKV gfx12 sibling.
pub const GEMM_GATE_UP_Q8_0_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_gate_up_q8_0_wmma.gfx12.hip");

/// gfx12 sister of GEMM_Q8_0_RESIDUAL_WMMA_SRC. Same lane-grp + half8_t
/// pattern; preserves the non-overlapping-write invariant under the new
/// lane-group row partition (lane group 0 → rows 0..7, group 1 → rows 8..15).
pub const GEMM_Q8_0_RESIDUAL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_q8_0_residual_wmma.gfx12.hip");

/// GEMV Q6_K: matrix-vector multiply with on-the-fly Q6_K dequantization.
/// Q6_K block: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per 256 elements.
pub const GEMV_Q6K_SRC: &str = include_str!("../../../kernels/src/gemv_q6k.hip");

/// RMSNorm: y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
pub const RMSNORM_SRC: &str = include_str!("../../../kernels/src/rmsnorm.hip");

/// Fused sandwich post-norm + residual-add: out = residual + rmsnorm(x, weight).
/// Collapses (rmsnorm + memcpy + add_inplace) 3 launches into 1 (gemma4 L4).
pub const RMSNORM_RESIDUAL_ADD_SRC: &str =
    include_str!("../../../kernels/src/rmsnorm_residual_add.hip");

/// gemma4 L3: fused per-head weighted q/k RMSNorm + q prescale + dual RoPE.
/// Collapses (q_norm + k_norm + scale_f32(q) + rope) 4 launches into 1.
pub const FUSED_GEMMA4_QK_NORM_ROPE_SRC: &str =
    include_str!("../../../kernels/src/fused_gemma4_qk_norm_rope.hip");

/// TriAttention sidecar calibration: GPU band-statistics accumulator.
/// Replaces the CPU BandAccumulator loop (99% of sidecar cal wall time).
pub const TRIATTN_ACCUMULATE_SRC: &str =
    include_str!("../../../kernels/src/triattn_accumulate.hip");

/// Element-wise add
pub const ADD_SRC: &str = include_str!("../../../kernels/src/add.hip");

/// Element-wise in-place add: a[i] += b[i]
pub const ADD_INPLACE_SRC: &str = include_str!("../../../kernels/src/add_inplace.hip");

/// Scaled in-place add: y[i] += c * x[i] — one kernel for both
/// CPU-scalar (c via kernarg) and GPU-scalar (c via device buffer)
/// variants. Used in the MoE FFN accumulator to fuse the old
/// (scale_f32 + add_inplace_f32) pair.
pub const SCALED_ADD_INPLACE_SRC: &str =
    include_str!("../../../kernels/src/scaled_add_inplace.hip");

/// Element-wise multiply
pub const MUL_SRC: &str = include_str!("../../../kernels/src/mul.hip");

/// SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x)
pub const SILU_SRC: &str = include_str!("../../../kernels/src/silu.hip");

/// Fused SiLU(gate) * up: out[i] = silu(gate[i]) * up[i]
/// Saves one kernel launch + one intermediate buffer.
pub const SILU_MUL_SRC: &str = include_str!("../../../kernels/src/silu_mul.hip");

/// Softmax over last dimension (one block per row)
pub const SOFTMAX_SRC: &str = include_str!("../../../kernels/src/softmax.hip");

/// RoPE (Rotary Positional Embedding)
pub const ROPE_SRC: &str = include_str!("../../../kernels/src/rope.hip");

/// Batched RoPE: apply RoPE to [batch_size] positions at once.
/// q: [batch_size × n_heads_q × head_dim], k: [batch_size × n_heads_k × head_dim]
/// positions: [batch_size] int array of position indices.
/// Grid: [half, batch_size, 1]. Each thread handles one (position, freq_index) pair.
pub const ROPE_BATCHED_SRC: &str = include_str!("../../../kernels/src/rope_batched.hip");

/// Single-head causal attention on GPU.
/// One thread block per query head. Handles GQA (kv_group heads share same KV).
/// q: [n_heads * head_dim], k_cache: [seq_len * n_kv_heads * head_dim],
/// v_cache: same layout, out: [n_heads * head_dim].
pub const ATTENTION_SRC: &str = include_str!("../../../kernels/src/attention.hip");

/// Flash-Decoding attention: split KV scan across multiple blocks per head.
/// Phase 1: each block processes a chunk of KV positions, writes partial (max, sum, output).
/// Phase 2: reduction across chunks using online softmax correction.
/// Grid: [n_heads, n_chunks, 1]. Each block handles one (head, chunk) pair.
/// Partial results stored in partials buffer: [n_heads × n_chunks × (1 + 1 + head_dim)] floats.
pub const ATTENTION_FLASH_SRC: &str = include_str!("../../../kernels/src/attention_flash.hip");

/// GQA-aware flash decode partial: grid [n_kv_heads, n_chunks]; each block
/// reuses one K/V load across its query-head group. Reduce phase reuses
/// `attention_flash_reduce` (ATTENTION_FLASH_SRC).
pub const ATTENTION_FLASH_GQA_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_gqa.hip");
pub const ATTENTION_FLASH_GQA_FUSED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_gqa_fused.hip");

/// Warp-cooperative GQA decode attention. One warp per head in the kv-group,
/// chunked KV processing with partials. Grid=[n_kv_heads, n_chunks], block=[kv_group*32].
/// 3.5× faster than scalar attention_flash on decode (271→77 µs).
pub const ATTENTION_GQA_WARP_SRC: &str =
    include_str!("../../../kernels/src/attention_gqa_warp.hip");
/// Device-side seq_len variant for hipGraph capture: seq_len read from device
/// pointer (baked into graph), only content changes between replays.
pub const ATTENTION_GQA_WARP_DV_SRC: &str =
    include_str!("../../../kernels/src/attention_gqa_warp_dv.hip");

/// Fused Gate+Up HFQ4-G256: two GEMVs in one launch (saves 1 launch per layer).
/// Grid: [gate_m + up_m, 1, 1]. Each block processes one row from gate or up weight.
pub const FUSED_GATE_UP_HFQ4G256_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_gate_up_hfq4g256.hip")
);
/// gfx1201 dense decode specialization. The host admits this artifact only
/// for K=1024, allowing LLVM to remove the generic quad loop and dead tail
/// bodies while retaining the base kernel's exact load/FMA/reduction order.
pub const FUSED_GATE_UP_HFQ4G256_K1024_GFX1201_SRC: &str = concat!(
    "#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_gate_up_hfq4g256_k1024_gfx1201\n",
    "#define HIPFIRE_FUSED_GATE_UP_K1024 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_gate_up_hfq4g256.hip")
);
// Muse Glimmer gfx1100 AR decode: one fused gate+up HFQ4/MQ4 projection at
// exact shape gate_m=up_m=19968, K=6656 (26 groups). Host enforces dimensions
// and selects this identity; reuses the generic fused_gate_up_hfq4g256
// arithmetic template with fixed groups_per_row. No buffer-load or
// experimental schedule opts — weight-cache policy only.
pub const FUSED_GLIMMER_GATE_UP_HFQ4G256_K6656_GFX1100_SRC: &str = concat!(
    "#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_glimmer_gate_up_hfq4g256_k6656_gfx1100\n",
    "#define HIPFIRE_GLIMMER_GATE_UP_K6656 1\n",
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/fused_gate_up_hfq4g256.hip")
);
/// Fused gate+up for Q8_0 weights. Two Q8 GEMVs in one launch.
/// Grid=[gate_m + up_m], block=[32]. +5.8 tok/s decode on dots.ocr.
pub const FUSED_GATE_UP_Q8_0_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_q8_0.hip");

/// 4-way fused QKVZA for Q8_0 weights (DECODE, n=1). Mirrors
/// fused_qkvza_hfq4g256 but with Q8_0 dequant. Grid=[total_m], block=[32].
/// Bit-exact with four sequential gemv_q8_0 calls.
pub const FUSED_QKVZA_Q8_0_SRC: &str = include_str!("../../../kernels/src/fused_qkvza_q8_0.hip");

/// 3-way fused QKV for Q8_0 weights (DECODE, n=1). Mirrors
/// fused_qkv_hfq4g256 but with Q8_0 dequant. Grid=[q_m+k_m+v_m], block=[32].
/// Bit-exact with three sequential gemv_q8_0 calls.
pub const FUSED_QKV_Q8_0_SRC: &str = include_str!("../../../kernels/src/fused_qkv_q8_0.hip");
pub const FUSED_QKV_Q8_0_QWEN2_BIAS_SRC: &str = concat!(
    "#define HIPFIRE_QKV_WITH_BIAS 1\n",
    "#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_q8_0_qwen2_bias\n",
    include_str!("../../../kernels/src/fused_qkv_q8_0.hip")
);

/// Wave64-native counterpart to FUSED_GATE_UP_HFQ4G256_SRC for CDNA1/3.
/// block=[64,1,1] with 2 rows per block (one per warp); grid halves from
/// gate_m + up_m to (total + 1) / 2. Byte-exact with the wave32 base.
pub const FUSED_GATE_UP_HFQ4G256_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_hfq4g256_wave64.hip");
// gfx906 dp4a-port — see kernels/src/fused_gate_up_hfq4g256_wave64_dp4a.hip
// for the math + lane-mapping invariants. Per-kernel PMC at 2026-05-05
// showed this kernel was memory-bound (3.86 % MemUnitStalled, 41 %
// VALUBusy) so dp4a's 75 % x-traffic reduction lands on the right
// bottleneck. Activations must be pre-quantized to block_q8_1_mmq
// (use ensure_q8_1_mmq_x). Skip on gemv_residual — it was ILP-bound
// and got its win from the prefetch variant instead.
pub const FUSED_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC: &str =
    include_str!("../../../kernels/src/fused_gate_up_hfq4g256_wave64_dp4a.hip");

/// INT8 co-located KV v2: [f16 scale (2B)][padding (2B)][int8 × head_dim] = 132 bytes per head.
/// f16 scale matches Q8_0 but with one block per head. Padding for 4-byte alignment.
pub const KV_CACHE_WRITE_INT8C_F16_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_int8c_f16.hip");

/// Attention with INT8 co-located f16 scale KV.
pub const ATTENTION_INT8C_F16_KV_SRC: &str =
    include_str!("../../../kernels/src/attention_int8c_f16_kv.hip");

/// INT8 co-located KV: [f32 scale][int8 × head_dim] = 132 bytes per head.
/// Symmetric quantization, no zero point. Dequant: scale * (float)val.
/// Minimized VGPRs: no zero register, no nibble math.
pub const KV_CACHE_WRITE_INT8C_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_int8c.hip");

/// Attention with INT8 co-located KV. Deferred scale multiply, 4×32 unrolled inner loop.
/// Q preloaded into shared memory. Scale applied ONCE per position, not per element.
pub const ATTENTION_INT8C_KV_SRC: &str =
    include_str!("../../../kernels/src/attention_int8c_kv.hip");

/// HFQ8 KV: FP32 scale+zero per head, contiguous uint8 data. Asymmetric quantization.
/// Scales: [max_seq × n_kv_heads × 2] f32 (scale, zero pairs).
/// Data: [max_seq × n_kv_heads × head_dim] uint8.
pub const KV_CACHE_WRITE_HFQ8_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_hfq8.hip");

/// Attention with HFQ8 KV cache. Flat layout, FP32 scale+zero, contiguous uint8 data.
pub const ATTENTION_HFQ8_KV_SRC: &str = include_str!("../../../kernels/src/attention_hfq8_kv.hip");

/// INT8 KV with separate scale array. Contiguous int8 values, one f32 scale per head.
/// Keys: [max_seq × n_kv_heads × head_dim] int8, Scales: [max_seq × n_kv_heads] f32.
/// Write: one warp per head, find amax via shuffle, quantize 4 elements per thread.
pub const KV_CACHE_WRITE_INT8_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_int8.hip");

/// Attention with INT8 KV (separate scale array). Clean indexed access, no block math.
pub const ATTENTION_INT8_KV_SRC: &str = include_str!("../../../kernels/src/attention_int8_kv.hip");

/// Batched causal attention: all query positions attend to their causal context.
/// Grid: [n_heads, seq_len, 1]. Each block handles one (head, query_position) pair.
/// Q/K/V are FP32: [seq_len × n_heads × head_dim] or [seq_len × n_kv_heads × head_dim].
/// Output: [seq_len × n_heads × head_dim].
/// For prefill: Q/K/V come from batched projections. KV also written to cache.
pub const ATTENTION_CAUSAL_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_causal_batched.hip");

/// Batched decode-with-history attention over an F32 KV cache. One block per
/// (query head, block row); each row at absolute position `base+row` attends to
/// the full history [0..base+row] read from the persistent cache by absolute
/// position. Drives block-parallel qwen2 spec-decode verify (arch_id=7).
/// Grid: [n_heads, batch, 1]. Q/out: [batch × n_heads × head_dim].
/// k_cache/v_cache: [max_seq × n_kv_heads × head_dim].
pub const ATTENTION_DECODE_BATCHED_HISTORY_SRC: &str =
    include_str!("../../../kernels/src/attention_decode_batched_history.hip");

/// Batched Q8_0 KV cache write: quantize multiple positions at once.
/// src: [batch_size × kv_dim] FP32. positions: [batch_size] int32.
/// Grid: [total_blocks × batch_size]. Each block handles one Q8_0 group for one position.
pub const KV_CACHE_WRITE_Q8_0_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_q8_0_batched.hip");

/// Quantize KV vector to Q8_0 format (same as GGML Q8_0 / existing GEMV kernels).
/// Block: [f16 scale (2B)][int8 × 32 (32B)] = 34 bytes per 32 elements.
/// head_dim=128 → 4 blocks × 34 = 136 bytes per head.
/// Layout: [max_seq × n_kv_heads × blocks_per_head × 34].
pub const KV_CACHE_WRITE_Q8_0_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_q8_0.hip");

/// gfx1100-only paired K/V Q8_0 cache writer. Kept in a separate translation
/// unit so its dormant body cannot perturb portable/gfx12 writer codegen.
pub const KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_q8_0_pair.gfx1100.hip");

/// Attention with Q8_0 quantized KV cache — same format as GGML Q8_0.
/// K and V caches stored as [max_seq × n_kv_heads × blocks_per_head × 34].
pub const ATTENTION_Q8_0_KV_SRC: &str = include_str!("../../../kernels/src/attention_q8_0_kv.hip");

/// Multi-slot KV descriptor header — see `kernels/src/kv_slot_desc.h`. The
/// runtime hipcc compile happens in a cache dir that doesn't have
/// `kernels/src` on its `-I` path, so kernels that `#include
/// "kv_slot_desc.h"` have that directive stripped and this body prepended
/// before compilation (same pattern as `TURBO_COMMON_H` / `GIVENS_COMMON_SRC`).
pub const KV_SLOT_DESC_H: &str = include_str!("../../../kernels/src/kv_slot_desc.h");

/// Sliding-window variant of ATTENTION_Q8_0_KV_SRC. Adds a `window`
/// parameter: 0 = full causal (identical to the baseline), >0 = attend only
/// to the last `window` keys ([max(0, seq_len - window), seq_len)). Used by
/// Gemma 4 sliding layers (window=1024); full layers call with window=0.
pub const ATTENTION_Q8_0_KV_SWA_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_kv_swa.hip");

/// Batched counterpart of ATTENTION_Q8_0_KV_SRC. Processes N queries in
/// one launch with per-row causal windows from a positions[] array.
pub const ATTENTION_Q8_0_KV_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_kv_batched.hip");

/// Query-tiled Q8_0 flash prefill attention. LDS depends only on BR/BC,
/// never on context length, so one kernel serves every sequence length.
pub const ATTENTION_Q8_0_FLASH_PREFILL_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill.hip");

/// WMMA (matrix-core) variant of ATTENTION_Q8_0_FLASH_PREFILL_SRC. Same
/// algorithm; QK^T and P·V run on RDNA3 wave32 matrix cores, and the Q8
/// dequant is amortised 16x by the 16x16x16 fragment layout.
pub const ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill_wmma.hip");

/// RDNA4 sister of ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SRC. gfx12 splits
/// each 16-wide contraction into two 8-element half-wave operands and uses
/// the gfx12-specific WMMA builtin/output mapping.
pub const ATTENTION_Q8_0_FLASH_PREFILL_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill_wmma.gfx12.hip");

/// Benchmark-only gfx1201 LongSpec partition WMMA flash: same arithmetic as
/// `ATTENTION_Q8_0_FLASH_PREFILL_WMMA_GFX12_SRC` but writes retained online-
/// softmax state `[m,l,O_unnormalized]` per query/head instead of normalized O.
/// JIT-only via `attention_q8_0_flash_prefill_wmma_partial_gfx1201_bench`.
pub const ATTENTION_Q8_0_FLASH_PREFILL_WMMA_PARTIAL_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill_wmma_partial.gfx1201.hip");

/// Benchmark-only stable LSE merge of two `[m,l,O_unnormalized]` partials into
/// normalized O. JIT-only via `attention_partition_merge_f32_bench`.
pub const ATTENTION_PARTITION_MERGE_F32_SRC: &str =
    include_str!("../../../kernels/src/attention_partition_merge_f32.hip");

/// Muse Glimmer-owned sliding-window variant of
/// `ATTENTION_Q8_0_FLASH_PREFILL_WMMA_GFX12_SRC`. Same WMMA tiling and Q8
/// dequant; adds a `window` argument, a tile-aligned lower bound so fully
/// out-of-window key tiles are skipped, and the matching mask term.
/// `window <= 0` reproduces the parent exactly.
pub const ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SWA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill_wmma_swa.gfx12.hip");

/// RDNA3 counterpart of `ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SWA_GFX12_SRC`.
/// Same algorithm and `window` contract; WMMA tiling uses the gfx11
/// interleaved accumulator mapping and 16-element K-lane split.
/// `window <= 0` reproduces the parent `ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SRC` exactly.
pub const ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SWA_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill_wmma_swa.hip");

/// Gfx1100 (discrete RDNA3) arch-isolated variant of `ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SWA_SRC`.
/// Byte-identical to the plain `..._swa.hip` at introduction; the split exists
/// solely so per-arch tuning (prefetch, LDS pressure, FIXED_HEAD_DIM defaults)
/// can diverge without arch-bleed. See `autoresearch/ar/certify/cross_arch.py:66`
/// `check_cross_arch` — arch-suffixed files (`*.gfxNNNN.hip`) are dispatch-
/// isolated and skipped by the gate, while an un-suffixed shared file would
/// require `#if`-gating and would be flagged as bleed.
pub const ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SWA_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill_wmma_swa.gfx1100.hip");

/// Gfx1151 (RDNA3.5 APU, Strix Halo) arch-isolated variant. Same contract as
/// the gfx1100 sister; byte-identical at introduction.
pub const ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SWA_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill_wmma_swa.gfx1151.hip");

/// Phase-timed variant of ATTENTION_Q8_0_KV_SRC. Functionally equivalent
/// to the baseline kernel but instrumented with wall_clock64() around each
/// of the 3 internal phases (QK^T, softmax, V-weighted-sum). Writes per-head
/// cycle counts into an extra output buffer of length [n_heads * 3]. For
/// profiling/diagnostic use only.
pub const ATTENTION_Q8_0_KV_TIMED_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_kv_timed.hip");

/// Flash attention tile kernel — zero LDS, online softmax, 32-thread WAVE32.
/// Grid: [n_heads, n_tiles]. Each block fuses QK-dot + softmax + V-accumulate
/// for its tile of positions, writing partials to global memory.
pub const ATTENTION_FLASH_Q8_0_TILE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_q8_0_tile.hip");

/// gfx1151-only ISA experiment: preserve the flash tile's reduction tree but
/// lower cross-lane exchanges to ds_swizzle + DPP8/quad-perm operations.
pub const ATTENTION_FLASH_Q8_0_TILE_DPP_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE 1\n",
    include_str!("../../../kernels/src/attention_flash_q8_0_tile.hip")
);

/// Flash attention reduce kernel — combines tile partials via online softmax
/// correction. Grid: [n_heads]. Reads per-tile {max, sum, out[head_dim]},
/// combines across tiles, normalizes, writes final output.
pub const ATTENTION_FLASH_Q8_0_REDUCE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_q8_0_reduce.hip");

/// gfx1100-only Qwen attention reduce/gate/MQ epilogue. Kept in a separate
/// translation unit so its dormant body cannot perturb portable/gfx12 reducer
/// codegen.
pub const ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_q8_0_reduce_gated_mq_rotate.gfx1100.hip");
pub const ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151\n",
    include_str!("../../../kernels/src/attention_flash_q8_0_reduce_gated_mq_rotate.gfx1100.hip")
);

/// Turbo common header: shared definitions for turbo/givens kernels.
pub const TURBO_COMMON_H: &str = include_str!("../../../kernels/src/turbo_common.h");

/// Givens rotation common header: 2x2 block-diagonal rotation primitives.
pub const GIVENS_COMMON_SRC: &str = include_str!("../../../kernels/src/givens_common.h");

// ── asym4 / asym3 / asym2: K at rotated-quantized + V at Q8_0 (RotorQuant planar/Q8 style) ──
//
// K is rotated and stored 4-bit (asym4) or 2-bit (asym2) — same byte layout
// as givens4 / givens2 K. V is stored at Q8_0 in NORMAL (un-rotated) space.
// Attention reads K in rotated space, V in normal space; accumulation thus
// ends in normal space — the plain Q8_0 flash reduce works as-is.
pub const KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens4.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens3.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens2.hip");
pub const ATTENTION_FLASH_ASYM4_TILE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym4_tile.hip");
pub const ATTENTION_FLASH_ASYM3_TILE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym3_tile.hip");
pub const ATTENTION_FLASH_ASYM2_TILE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym2_tile.hip");

// asym batched prefill variants: K rotated + V Q8 in one launch for N positions.
pub const KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens4_batched.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens3_batched.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens2_batched.hip");
pub const ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym4_tile_batched.hip");
pub const ATTENTION_FLASH_ASYM4_WMMA_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym4_wmma_tile_batched.hip");
pub const ATTENTION_FLASH_ASYM4_WMMA_TILE_BATCHED_GFX12_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym4_wmma_tile_batched.gfx12.hip");
pub const ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym3_tile_batched.hip");
pub const ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym2_tile_batched.hip");
pub const ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_q8_0_tile_batched.hip");
pub const ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym_reduce_batched.hip");

// lloyd-V (FWHT-rotated centroid) dedicated reduce kernels. Used ONLY when
// v_mode != 8 — the tile kernels now write rotated V partials and these
// reduces apply the inverse FWHT once after the cross-tile combine. The
// Q8/asym paths keep using the untouched q8_0_reduce / asym_reduce_batched.
// Both require turbo_common.h (for fwht_shfl_inverse_256), so they MUST be
// loaded via ensure_givens4_kernel.
pub const ATTENTION_FLASH_LLOYD_REDUCE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_lloyd_reduce.hip");
pub const ATTENTION_FLASH_LLOYD_REDUCE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_lloyd_reduce_batched.hip");

// Signed-FWHT K-write + FA tile variants — same byte layout as asym family,
// rotation primitive swapped from Givens (per-quad cos/sin) to signed-FWHT
// (128-wide butterfly via ds_swizzle_b32). Q is forward-rotated by the same
// signed-FWHT in the FA path; K cache is byte-identical to asym4.
pub const KV_CACHE_WRITE_ASYM_K_FWHT4_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_fwht4.hip");
pub const KV_CACHE_WRITE_ASYM_K_FWHT4_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_fwht4_batched.hip");
pub const ATTENTION_FLASH_FWHT4_TILE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht4_tile.hip");
pub const ATTENTION_FLASH_FWHT4_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht4_tile_batched.hip");
pub const KV_CACHE_WRITE_ASYM_K_FWHT3_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_fwht3.hip");
pub const KV_CACHE_WRITE_ASYM_K_FWHT3_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_fwht3_batched.hip");
pub const KV_CACHE_WRITE_FWHT256_2BIT_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_fwht256_2bit.hip");
pub const KV_CACHE_WRITE_FWHT256_2BIT_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_fwht256_2bit_batched.hip");
pub const KV_CACHE_WRITE_FWHT256_4BIT_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_fwht256_4bit.hip");
pub const KV_CACHE_WRITE_FWHT256_4BIT_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_fwht256_4bit_batched.hip");
// Adaptive-KV V transcode kernels (re-quantize an existing V cache in place,
// all positions of one FA layer, higher tier → lower tier).
pub const KV_TRANSCODE_V_Q8_TO_LLOYD4_SRC: &str =
    include_str!("../../../kernels/src/kv_transcode_v_q8_to_lloyd4.hip");
pub const KV_TRANSCODE_V_LLOYD_DOWN_SRC: &str =
    include_str!("../../../kernels/src/kv_transcode_v_lloyd_down.hip");
// Adaptive-KV K transcode (fwht4 → fwht2, same-width 128-LUT remap, no FWHT).
pub const KV_TRANSCODE_K_FWHT4_TO_FWHT2_SRC: &str =
    include_str!("../../../kernels/src/kv_transcode_k_fwht4_to_fwht2.hip");
// Adaptive-KV K transcode (fwht4 → fwht3, RE-ROTATION 128→256, advanced selector).
pub const KV_TRANSCODE_K_FWHT4_TO_FWHT3_SRC: &str =
    include_str!("../../../kernels/src/kv_transcode_k_fwht4_to_fwht3.hip");
pub const ATTENTION_FLASH_FWHT3_TILE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht3_tile.hip");
pub const ATTENTION_FLASH_FWHT3_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht3_tile_batched.hip");
pub const KV_CACHE_WRITE_ASYM_K_FWHT2_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_fwht2.hip");
pub const KV_CACHE_WRITE_ASYM_K_FWHT2_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_fwht2_batched.hip");
pub const ATTENTION_FLASH_FWHT2_TILE_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht2_tile.hip");
pub const ATTENTION_FLASH_FWHT2_TILE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht2_tile_batched.hip");

/// TriAttention scoring on Q8 post-RoPE K cache (arXiv:2604.04921).
pub const TRIATTN_SCORE_Q8_SRC: &str = include_str!("../../../kernels/src/triattn_score_q8.hip");

/// TriAttention scoring on asym3 (Givens-rotated 3-bit) K cache.
pub const TRIATTN_SCORE_ASYM3_SRC: &str =
    include_str!("../../../kernels/src/triattn_score_asym3.hip");

/// TriAttention scoring on asym4 (Givens-rotated 4-bit) K cache.
pub const TRIATTN_SCORE_ASYM4_SRC: &str =
    include_str!("../../../kernels/src/triattn_score_asym4.hip");

/// TriAttention scoring on asym2 (Givens-rotated 2-bit) K cache.
pub const TRIATTN_SCORE_ASYM2_SRC: &str =
    include_str!("../../../kernels/src/triattn_score_asym2.hip");
/// TriAttention scoring for adaptive signed-FWHT K caches.
pub const TRIATTN_SCORE_FWHT_SRC: &str =
    include_str!("../../../kernels/src/triattn_score_fwht.hip");

/// Gather-based compaction for KV eviction: copy `budget` src rows to dst.
pub const KV_COMPACT_GATHER_SRC: &str = include_str!("../../../kernels/src/kv_compact_gather.hip");

/// CASK m-folding merge: weighted-average m Q8_0 rows into 1 per slot (arXiv:2604.10900).
pub const KV_FOLD_Q8_SRC: &str = include_str!("../../../kernels/src/kv_fold_q8.hip");

/// CASK m-folding merge for asym3 K (givens-rotated 3-bit).
pub const KV_FOLD_ASYM3_SRC: &str = include_str!("../../../kernels/src/kv_fold_asym3.hip");

/// CASK m-folding merge for asym4 K (givens-rotated 4-bit).
pub const KV_FOLD_ASYM4_SRC: &str = include_str!("../../../kernels/src/kv_fold_asym4.hip");

/// CASK m-folding merge for asym2 K (givens-rotated 2-bit).
pub const KV_FOLD_ASYM2_SRC: &str = include_str!("../../../kernels/src/kv_fold_asym2.hip");

/// Quantize KV vector to Q8 (int8 symmetric) and write to quantized KV cache.
/// Per head: [4B f32 scale][head_dim × int8 values] = head_dim + 4 bytes.
/// For head_dim=128: 132 bytes vs 512 bytes FP32 = 3.88x compression.
pub const KV_CACHE_WRITE_Q8_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q8.hip");

/// Attention with Q8 quantized KV cache — symmetric int8, dequant on read.
pub const ATTENTION_Q8KV_SRC: &str = include_str!("../../../kernels/src/attention_q8kv.hip");

/// HFQ4 KV block: co-located FP32 scale+zero + packed nibbles. 72 bytes per head.
/// Layout per position: [n_kv_heads × 72] bytes. One cache line per head.
pub const KV_CACHE_WRITE_HFQ4_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_hfq4.hip");

/// Attention with HFQ4 KV blocks v2. Tight single-block pattern.
/// 72 bytes per head = one HFQ4-G128 block (scale+zero+64 nibble bytes).
/// Q preloaded into shared memory. One scale+zero load per position.
pub const ATTENTION_HFQ4_KV_SRC: &str = include_str!("../../../kernels/src/attention_hfq4_kv.hip");

/// Quantize KV vector to HFQ4-G128 and write to quantized KV cache.
/// Input: kv_dim floats at kv_src. Output: packed HFQ4 at dst[pos * bytes_per_pos].
/// Each group of 128 floats → 72 bytes (4B scale + 4B zero + 64B nibbles).
/// For head_dim=128, one head = exactly one group = 72 bytes.
pub const KV_CACHE_WRITE_Q4_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q4.hip");

/// Attention with quantized HFQ4 KV cache.
/// Same structure as attention_f32 but dequantizes K and V on the fly.
pub const ATTENTION_Q4KV_SRC: &str = include_str!("../../../kernels/src/attention_q4kv.hip");

// ═══════════════════════════════════════════════════════════════════════
// DeltaNet ops (Qwen3.5 linear attention)
// ═══════════════════════════════════════════════════════════════════════

/// Sigmoid: σ(x) = 1 / (1 + exp(-x)). Element-wise, in-place.
#[cfg(feature = "deltanet")]
pub const SIGMOID_SRC: &str = include_str!("../../../kernels/src/sigmoid.hip");

/// Softplus: log(1 + exp(x)), numerically stable. Element-wise, in-place.
#[cfg(feature = "deltanet")]
pub const SOFTPLUS_SRC: &str = include_str!("../../../kernels/src/softplus.hip");

/// L2 normalization per head: out[i] = x[i] / sqrt(sum(x²) + eps).
/// Grid: [n_heads]. Block: [32]. Each warp normalizes one head of head_dim elements.
#[cfg(feature = "deltanet")]
pub const L2_NORM_SRC: &str = include_str!("../../../kernels/src/l2_norm.hip");

/// Fused L2-norm(Q) + L2-norm(K) + scale(Q). Replaces three back-to-back
/// launches in the DeltaNet attention path with one. See kernel header for
/// details.
#[cfg(feature = "deltanet")]
pub const FUSED_QK_L2_NORM_SCALE_SRC: &str =
    include_str!("../../../kernels/src/fused_qk_l2_norm_scale.hip");

/// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Two back-to-back
/// scalar ops in the DeltaNet preamble merged into one launch.
#[cfg(feature = "deltanet")]
pub const FUSED_SIGMOID_ALPHA_GATE_SRC: &str =
    include_str!("../../../kernels/src/fused_sigmoid_alpha_gate.hip");

/// Fused sigmoid(gate) * x — the FA attention epilogue that used to be
/// `sigmoid_f32(gate)` + `mul_f32(attn_out, gate, attn_out)`.
pub const SIGMOID_MUL_SRC: &str = include_str!("../../../kernels/src/sigmoid_mul.hip");

/// Top-K=128 extraction over a logits vector. Lets the host sampler work
/// on a 1 KB GPU-side candidate set instead of DtoH'ing the full 600 KB
/// logits array. See kernel header for bit-exactness reasoning.
pub const TOPK_LOGITS_SRC: &str = include_str!("../../../kernels/src/topk_logits.hip");
pub const TOPK_LOGSUMEXP_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/topk_logsumexp_batched.hip");

/// Partial interleaved RoPE: rotate only first n_rot dims, pairs are adjacent (d0,d1),(d2,d3),...
/// Dims >= n_rot pass through unchanged.
/// Grid: [n_rot/2]. Block: [1]. Each thread handles one rotation pair.
#[cfg(feature = "deltanet")]
pub const ROPE_PARTIAL_INTERLEAVED_SRC: &str =
    include_str!("../../../kernels/src/rope_partial_interleaved.hip");

/// Half-split-pair partial RoPE, matching HF `rotate_half` convention used by
/// Qwen2 / Qwen3 / Qwen3.5 `apply_rotary_pos_emb`. Wired in via env-gate
/// `HIPFIRE_ROPE_HALFSPLIT=1` from `rope_partial_interleaved_f32`. See
/// docs/plans/qwen35-mq4-quality-gap.md §"RoPE convention probe".
#[cfg(feature = "deltanet")]
pub const ROPE_PARTIAL_HALFSPLIT_SRC: &str =
    include_str!("../../../kernels/src/rope_partial_halfsplit.hip");

/// Gemma 4 full-attention partial RoPE (HF `rotate_half` pairing). head_dim=512
/// but only the first `n_rot_pairs` pairs (i, i+head_dim/2) rotate; the rest are
/// NoPE pass-through. Pairs differ from the interleaved/halfsplit variants — see
/// `kernels/src/rope_partial_halved.hip` for the proportional-RoPE math.
pub const ROPE_PARTIAL_HALVED_SRC: &str =
    include_str!("../../../kernels/src/rope_partial_halved.hip");

/// Batched (N-token) counterpart to `ROPE_PARTIAL_HALVED_SRC`, used by Gemma 4
/// batched prefill. Reads one i32 position per token from a device array.
pub const ROPE_PARTIAL_HALVED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/rope_partial_halved_batched.hip");

/// Exact gfx1100 Qwen3.6 full-attention decode preparation: deinterleave Q and
/// gate, RMS-normalize Q/K, then apply partial half-split RoPE head-locally.
#[cfg(feature = "deltanet")]
pub const QWEN35_FA_PREP_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/qwen35_fa_prep.gfx1100.hip");
#[cfg(feature = "deltanet")]
pub const QWEN35_FA_PREP_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_QWEN35_FA_PREP_KERNEL qwen35_fa_prep_gfx1151\n",
    include_str!("../../../kernels/src/qwen35_fa_prep.gfx1100.hip")
);

/// 2-D spatial RoPE with precomputed per-patch cos/sin tables. Used by
/// the dots.ocr (Qwen2-VL family) `DotsVisionTransformer` for vision
/// attention. See `kernels/src/rope_2d_halfsplit.hip` for the layout
/// + algorithm and `crates/hipfire-arch-dots-ocr/src/rope.rs` for the
/// host-side cos/sin table builder.
pub const ROPE_2D_HALFSPLIT_SRC: &str = include_str!("../../../kernels/src/rope_2d_halfsplit.hip");

/// 2-D spatial RoPE applied IN-PLACE to the Q and K slices of a fused
/// interleaved `[N, 3*hidden]` QKV buffer. Companion to the separate-
/// buffer variant above. Initially intended for the dots.ocr vision
/// encoder's single-GEMM → attention path, but `vit_attention_opt`
/// turned out to overflow RDNA3 LDS at the smoke image's N≈19520; the
/// dots.ocr forward pass therefore splits QKV into separate Q/K/V
/// buffers (see `QKV_SPLIT_INTERLEAVED_SRC`) and routes through
/// `attention_dflash_f32` instead. Kernel kept for future fast-path
/// when a non-overflowing fused vision attention exists.
pub const ROPE_2D_HALFSPLIT_QKV_INTERLEAVED_SRC: &str =
    include_str!("../../../kernels/src/rope_2d_halfsplit_qkv_interleaved.hip");

/// Split a fused interleaved `[N, 3*hidden]` QKV buffer into three
/// separate `[N, hidden]` Q, K, V buffers. Used by the dots.ocr vision
/// encoder to feed `attention_dflash_f32` (FlashAttention-style with
/// online softmax — supports L > 16128 without SLM overflow, unlike
/// `vit_attention_opt` which materialises a `scores[N]` LDS buffer).
/// See `kernels/src/qkv_split_interleaved.hip`.
pub const QKV_SPLIT_INTERLEAVED_SRC: &str =
    include_str!("../../../kernels/src/qkv_split_interleaved.hip");

/// WMMA-accelerated FlashAttention-style non-causal attention (gfx1100+).
/// Companion to `ATTENTION_DFLASH_SRC` for the large-B / large-L case
/// where one block tiles 16 queries via WMMA. Grid `[n_heads, ceil(B/16)]`,
/// block `[32]`. See `kernels/src/attention_dflash_wmma.hip`.
pub const ATTENTION_DFLASH_WMMA_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma.hip");

/// gfx12/RDNA4 sister of `ATTENTION_DFLASH_WMMA_SRC`. Same algorithm; the
/// WMMA fragments use `half8_t` operands + the `_w32_gfx12` intrinsic (the
/// gfx11 `_w32` builtin does not lower on gfx12 — "Cannot select intrinsic").
/// Routed via `has_wmma_w32_gfx12()` in `attention_dflash_wmma_f32`. See
/// `kernels/src/attention_dflash_wmma.gfx12.hip`.
pub const ATTENTION_DFLASH_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma.gfx12.hip");

/// M=32 variant of `ATTENTION_DFLASH_WMMA_SRC` — two-wave block (64
/// threads), processes 32 queries per block instead of 16. Halves the
/// number of query-tile blocks at large B, which halves global K-tile
/// fetches and gives ~2× wall-time speedup at vision-encoder shapes
/// where the M=16 kernel is memory-bound. LDS-capped at head_dim ≤ 128.
/// See `kernels/src/attention_dflash_wmma_m32.hip`.
pub const ATTENTION_DFLASH_WMMA_M32_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m32.hip");

/// gfx12/RDNA4 sister of `ATTENTION_DFLASH_WMMA_M32_SRC` (`_w32_gfx12`
/// WMMA, K split across wave-halves). Routed via `has_wmma_w32_gfx12()`
/// in `attention_dflash_wmma_m32_f32`. See
/// `kernels/src/attention_dflash_wmma_m32.gfx12.hip`.
pub const ATTENTION_DFLASH_WMMA_M32_GFX12_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m32.gfx12.hip");

/// N=64 K-tile variant — M=32 queries per block, **64 keys per outer
/// loop iteration** (vs 16 in M32_SRC). Q lives in registers across all
/// K-tiles; phase C fuses the alpha-scale and SV epilogue. Designed to
/// amortise per-K-tile fixed costs (syncs, softmax, O-scaling) over 4×
/// more keys per visit. LDS at hd=128 ≈ 57.7 KB (under 64 KB cap).
/// See `kernels/src/attention_dflash_wmma_n64.hip` and the rocprof
/// investigation in `docs/plans/dots-ocr.perf-investigation.md`.
pub const ATTENTION_DFLASH_WMMA_N64_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_n64.hip");

/// N=64 variant that consumes K and V already stored as **f16 in DRAM**
/// (Q and output stay f32). Halves the attention DRAM traffic for K and
/// V — the dominant cost on memory-bound vision-encoder shapes per the
/// rocprof analysis. Caller is responsible for casting K and V from f32
/// to f16 once (see `cast_f32_to_f16`) before calling this kernel; the
/// cast cost (~120 MB) is trivial against the ~73 GB K+V DRAM traffic
/// per attention call at vision shape.
/// See `kernels/src/attention_dflash_wmma_n64_f16kv.hip`.
pub const ATTENTION_DFLASH_WMMA_N64_F16KV_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_n64_f16kv.hip");

/// N=128 variant — K-tile 128, K/V f16 in DRAM, V_lds and S_lds in
/// f16. Same shape as the N=64 f16-K/V sibling but with twice the
/// K-tile width, halving the outer-loop trip count (and therefore
/// __syncthreads / softmax / alpha-scale overhead per attention call).
/// Only feasible because moving V_lds and S_lds to f16 reclaimed
/// enough LDS budget to fit a 128-row V_lds.
/// See `kernels/src/attention_dflash_wmma_n128_f16kv.hip`.
pub const ATTENTION_DFLASH_WMMA_N128_F16KV_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_n128_f16kv.hip");

/// M=64 N=128 variant — 4-wave block, 64 queries per block (vs 32 in
/// the N128 sibling). Halves the query-block count B/M from 610 to
/// 305 at vision shape, which halves K and V DRAM traffic per
/// attention call (~73 GB → ~36.5 GB at f16). O moves from O_lds to
/// per-lane O_frags register arrays (8 float8_t in WMMA frag_c
/// layout = 64 VGPRs/lane) to free the LDS budget that the doubled
/// query rows would have eaten.
/// See `kernels/src/attention_dflash_wmma_m64_n128_f16kv.hip`.
pub const ATTENTION_DFLASH_WMMA_M64_N128_F16KV_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n128_f16kv.hip");

/// V2 of M=64 N=128 — adds (a) S_lds row padding 128 → 130 to break
/// a 16-way LDS bank conflict in phase C's S_lds reads, and (b)
/// cooperative wave-32 softmax (each row uses all 32 lanes via
/// __shfl_xor butterfly, vs 1 lane sequential over 128 vals).
/// See `kernels/src/attention_dflash_wmma_m64_n128_f16kv_v2.hip`.
pub const ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V2_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n128_f16kv_v2.hip");

/// V3 of M=64 N=128 — keeps v2's S_lds padding + cooperative softmax
/// and adds hoisted S_lds reads in phase C (outer c, inner dc) so
/// each a_reg_sm row chunk is read once per c instead of once per
/// (dc, c). Reduces phase C S_lds reads from 1024/lane/iter to
/// 128/lane/iter. O alpha-folded at start of phase C so SV
/// accumulates directly into the running output.
/// See `kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3.hip`.
pub const ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3.hip");

/// Vision attention winner (M=64, V_tile=32, f16 K/V, 2 WG/CU).
/// ~40% faster than v3 at B=L=19520, hd=128. V_tile=32 stages V in 4
/// v_chunks per K-tile, keeping LDS at 25.6 KB (2 WG/CU occupancy).
/// Grid=[n_heads, ceil(B/64)], block=[128] (4 waves).
pub const ATTENTION_DFLASH_WMMA_M64_N32_F16KV_V5_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n32_f16kv_v5_f32.hip");
/// gfx12/RDNA4 sibling of the dots.ocr v5 vision attention kernel.
pub const ATTENTION_DFLASH_WMMA_M64_N32_F16KV_V5_GFX12_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n32_f16kv_v5_f32.gfx12.hip");

/// Causal variant of v3 (M=64, N=128, f16 K/V). Adds causal mask:
/// S[q, k] = -inf when k > q. Skips entirely-masked tiles. Grid
/// `[n_heads, ceil(B/64)]`, block `[128]`.
/// See `kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3_causal.hip`.
pub const ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_CAUSAL_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3_causal.hip");
/// gfx12/RDNA4 sibling of the causal v3 kernel. Same C symbol, separate module
/// because gfx12 WMMA uses half8 operands and the `_w32_gfx12` builtin.
pub const ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_CAUSAL_GFX12_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3_causal.gfx12.hip");

/// V_lds transpose variant of v3 (M=64, N=128, f16 K/V). V_lds transposed
/// from [n_tile][head_dim] to [head_dim][V_T_STRIDE] so Phase C b_reg reads
/// are 16 consecutive f16 values (compiler-vectorizable) instead of
/// stride-128 scattered. V_T_STRIDE=130 (padded) eliminates bank conflicts.
/// LDS: V_lds_T[128][130] + S_lds[64][130] + m/l/alpha = 49.5 KB, 1 WG/CU.
pub const ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V4_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n128_f16kv_v4.hip");

/// V_lds transpose variant of v5 (M=64, V_tile=32, f16 K/V). Same
/// transpose optimization: Phase C b_reg reads become contiguous,
/// bank-conflict-free. V_T_STRIDE=34 (V_tile+2). LDS: 25.5 KB, 2 WG/CU.
/// Negative result on vision shape (-6.5% vs v5). Kept for bench.
pub const ATTENTION_DFLASH_WMMA_M64_N32_F16KV_V6_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n32_f16kv_v6_f32.hip");

/// v7: M=128 two-pass sub-tiling (§14.4C). K-shared sub-tiles.
/// Negative result (-10.9% vs v5). Kept for bench.
pub const ATTENTION_DFLASH_WMMA_M128_N32_F16KV_V7_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m128_n32_f16kv_v7_f32.hip");

/// v7b: M=128 sequential sub-tiling, no K-sharing. Tests L2 warmth only.
/// Negative result (-2.4% vs v5). Kept for bench.
pub const ATTENTION_DFLASH_WMMA_M128_N32_F16KV_V7B_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m128_n32_f16kv_v7b_f32.hip");

/// V_lds transpose variant of v3-causal. Same as v4 but with causal mask
/// and tile skip. V_T_STRIDE_CAUSAL=130 for bank-conflict-free reads.
pub const ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V4_CAUSAL_SRC: &str =
    include_str!("../../../kernels/src/attention_dflash_wmma_m64_n128_f16kv_v4_causal.hip");

/// Standalone f32 → f16 elementwise cast kernel. Block [256], grid
/// `ceil(n / 256)`. See `kernels/src/cast_f32_to_f16.hip`.
pub const CAST_F32_TO_F16_SRC: &str = include_str!("../../../kernels/src/cast_f32_to_f16.hip");

/// In-place F32 → bf16 → F32 round-trip. Truncates each F32 to bf16's
/// 7-bit mantissa with round-to-nearest-even. Used by the dots.ocr
/// vision encoder to match HF's bf16 forward path at residual-stream
/// points. See `kernels/src/bf16_round_trip.hip`.
pub const BF16_ROUND_TRIP_SRC: &str = include_str!("../../../kernels/src/bf16_round_trip.hip");

/// Batched partial-interleaved RoPE — per-row positions read from a
/// positions[] array. Used by the batched prefill FA path.
#[cfg(feature = "deltanet")]
pub const ROPE_PARTIAL_INTERLEAVED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/rope_partial_interleaved_batched.hip");

/// Batched half-split partial RoPE — twin of the interleaved batched kernel
/// with HF `rotate_half` convention. Default for Qwen3.5 since 2026-05-12.
/// See docs/plans/qwen35-mq4-quality-gap.md §"RoPE convention probe / halfsplit
/// fix" for the rationale.
#[cfg(feature = "deltanet")]
pub const ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/rope_partial_halfsplit_batched.hip");

/// 3D mrope, half-split convention. Twin of ROPE_PARTIAL_HALFSPLIT_SRC,
/// differing only in which position each frequency index reads: band mapping
/// is H if (d%3==1 && d < 3*sec_h), W if (d%3==2 && d < 3*sec_w), else T
/// (the interleaved [THWTHW...TT] layout HF's apply_interleaved_mrope
/// produces). See `hipfire_arch_qwen35_vl::mrope::mrope_axis_for_freq` for
/// the CPU-side spec this arithmetic must match exactly.
#[cfg(feature = "deltanet")]
pub const ROPE_MROPE_HALFSPLIT_SRC: &str =
    include_str!("../../../kernels/src/rope_mrope_halfsplit.hip");

/// Batched twin of ROPE_MROPE_HALFSPLIT_SRC — see that const for the
/// band-mapping rationale.
#[cfg(feature = "deltanet")]
pub const ROPE_MROPE_HALFSPLIT_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/rope_mrope_halfsplit_batched.hip");

/// 1D causal depthwise convolution (kernel_size=4) with persistent ring buffer state.
/// For decode: one token at a time. conv_state: [n_channels × 3] ring buffer.
/// out[c] = w[0]*x[c] + w[1]*state[c][0] + w[2]*state[c][1] + w[3]*state[c][2]
/// Then shift state: state = [x, state[0], state[1]].
#[cfg(feature = "deltanet")]
pub const CONV1D_DECODE_SRC: &str = include_str!("../../../kernels/src/conv1d_decode.hip");

/// LFM2 LIV double-gated short-conv, single-token decode (runtime kernel_size).
/// Fuses the B*x pre-gate, depthwise causal conv, C*conv_out post-gate, and the
/// rolling conv-state ring-buffer advance into one launch. conv_bias is always
/// false for LFM2. See kernels/src/conv1d_gated_decode.hip.
pub const CONV1D_GATED_DECODE_SRC: &str =
    include_str!("../../../kernels/src/conv1d_gated_decode.hip");

/// Gated output norm: rmsnorm(x) * silu(z). Fused single kernel.
/// x and z are [n_heads × head_dim]. weight is [head_dim] (shared across heads).
#[cfg(feature = "deltanet")]
pub const GATED_NORM_SRC: &str = include_str!("../../../kernels/src/gated_norm.hip");

/// Gated Delta Net — tiled LDS + warp-shuffle.
/// S[128×128] tiled into TILE_ROWS=8 row chunks. Each tile = 8×128×4 = 4KB LDS.
/// 64KB/4KB = 16 blocks/CU → 4 waves/SIMD. Rows are independent → perfect tiling.
/// 32 threads per block (one warp), each handles 4 columns.
/// Grid: [n_heads, HD/TILE_ROWS]. Block: [32].
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_SRC: &str = include_str!("../../../kernels/src/gated_delta_net.hip");

/// GDN Q8 — tiled LDS + warp-shuffle. Dequant tile into LDS, recurrence, requant back.
/// Tile = TILE_ROWS × 128 × 4B = 4KB. Same tiling as FP32 variant.
/// Grid: [n_heads, HD/TILE_ROWS]. Block: [32].
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_Q8_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_q8.hip");

/// Fast variant for the default MQ4/HFQ4 path: no per-token requant,
/// requant outside the loop. Supports EF residual. Lower VGPR pressure.
pub const GATED_DELTA_NET_Q8_FAST_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip");

/// Decode-only compact-QK variants for Qwen3.5 DeltaNet GQA (16 Q/K heads,
/// 32 value/state heads). Each pair of state heads reads one normalized Q/K
/// head directly, eliminating the materializing repeat-interleave launch.
pub const GATED_DELTA_NET_Q8_COMPACT2_B2_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_MIN_BLOCKS 2\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_b2\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);
pub const GATED_DELTA_NET_Q8_COMPACT2_B4_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_MIN_BLOCKS 4\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_b4\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);
pub const GATED_DELTA_NET_Q8_COMPACT2_B8_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_MIN_BLOCKS 8\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_b8\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);
pub const GATED_DELTA_NET_Q8_COMPACT2_B12_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_MIN_BLOCKS 12\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_b12\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);
pub const GATED_DELTA_NET_Q8_COMPACT2_B16_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_MIN_BLOCKS 16\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_b16\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);

/// gfx1151 decode experiment: one wave owns eight state rows instead of four.
/// This halves the workgroup count and reuses Q/K plus gate/beta across twice
/// as many rows without changing the per-row recurrence or replay ABI.
pub const GATED_DELTA_NET_Q8_COMPACT2_R8_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_TILE_ROWS 8\n#define HIPFIRE_GDN_MIN_BLOCKS 2\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_r8_gfx1151\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);

/// gfx1151 decode experiment: pair two four-row waves in one workgroup and
/// stage Q/K once in LDS. This preserves the baseline wave count while halving
/// Q/K traffic and workgroup setup for the recurrent state update.
pub const GATED_DELTA_NET_Q8_COMPACT2_R4X2_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_BLOCK_SIZE 64\n#define HIPFIRE_GDN_WAVES_PER_BLOCK 2\n#define HIPFIRE_GDN_MIN_BLOCKS 2\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_r4x2_gfx1151\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);

/// gfx1151 reduction experiment: retain the certified four-row wave geometry
/// but replace shuffle-backed LDS bpermutes with DPP/permlane VALU operations.
pub const GATED_DELTA_NET_Q8_COMPACT2_DPP_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GDN_QK_HEAD_DIV 2\n#define HIPFIRE_GDN_MIN_BLOCKS 2\n#define HIPFIRE_GFX1151_GDN_DPP_REDUCE 1\n#define HIPFIRE_GDN_KERNEL gated_delta_net_q8_compact2_dpp_gfx1151\n",
    include_str!("../../../kernels/src/gated_delta_net_q8_fast.hip")
);

/// Tree-aware variant of gated_delta_net_q8. Per-token S-tile persist-write
/// to a caller-owned tape buffer, so sibling tokens read the parent's
/// post-update state rather than the previous sibling's. Required for
/// correctness when processing a DDTree-linearized token block.
///
/// s_q8_init / s_scales_init are the pre-block snapshot (READ-ONLY). The
/// kernel never advances persistent dn_state.s_matrices — caller runs
/// linear replay on the accepted spine post-acceptance to commit the
/// trajectory (same pattern as conv1d_silu_split_tree).
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_Q8_TREE_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_q8_tree.hip");

/// FP32 tree-aware GDN recurrence — full-precision counterpart of
/// GATED_DELTA_NET_Q8_TREE_SRC (no per-token dequant/requant or scales).
/// Used by the FP32 `StateQuant` spec-decode tree-verify path.
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_F32_TREE_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_f32_tree.hip");

/// Batched-sequential FP32 GDN recurrence — full-precision counterpart of
/// the Q8 batch kernel (GATED_DELTA_NET_Q8_SRC), with the same 32×32-tile
/// parallelism. Replaces the slow 128-thread single-token `gated_delta_net`
/// on the FP32 `StateQuant` batched prefill/verify path.
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_F32_BATCH_SEQ_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_f32_batch_seq.hip");

/// Chunked (parallel) FP32 GDN recurrence — chunked sibling of
/// GATED_DELTA_NET_F32_BATCH_SEQ_SRC. One wave32 workgroup per (head, chunk);
/// intra-chunk parallelism replaces the per-token serial inner loop.
/// Numerically EQUAL to the sequential recurrence (oracle gdn_chunked_f32).
/// Cross-chunk dependency is serialized host-side (one launch per chunk).
/// Grid: [n_heads, n_chunks]. Block: [32]. Default OFF (HIPFIRE_GDN_CHUNKED).
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_F32_CHUNKED_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_f32_chunked.hip");

/// GDN recurrence with Q4-quantized S state in VRAM.
/// State layout: unsigned char s_q4[n_heads][HD*HD/2] (nibble-packed) + float s_scales[n_heads*HD].
/// Symmetric 4-bit: values -8..+7, scale = absmax/7. Per-row scale.
/// 8x compression vs FP32 (8KB + 512B scales per head vs 64KB).
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_Q4_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_q4.hip");

/// Alpha gate compute on GPU: out[i] = softplus(alpha[i] + dt_bias[i]) * (-exp(a_log[i])).
/// Eliminates 85µs CPU roundtrip per DeltaNet layer.
#[cfg(feature = "deltanet")]
pub const ALPHA_GATE_SRC: &str = include_str!("../../../kernels/src/alpha_gate.hip");

/// Scale vector by constant: x[i] *= scale. Eliminates 48µs CPU roundtrip.
#[cfg(feature = "deltanet")]
pub const SCALE_F32_SRC: &str = include_str!("../../../kernels/src/scale_f32.hip");

/// Fused conv1d (kernel_size=4) + SiLU. Eliminates one kernel launch.
#[cfg(feature = "deltanet")]
pub const CONV1D_SILU_SRC: &str = include_str!("../../../kernels/src/conv1d_silu.hip");

/// Conv1d + SiLU + Q/K/V split fused into one kernel. Writes directly to
/// three separate destination buffers instead of producing a packed output
/// that needs three memcpys to split. Eliminates 3 DtoD copies per
/// linear-attention layer.
#[cfg(feature = "deltanet")]
pub const CONV1D_SILU_SPLIT_SRC: &str = include_str!("../../../kernels/src/conv1d_silu_split.hip");

pub const CONV1D_SILU_SPLIT_QKNORM_B32_SRC: &str = concat!(
    "#define HIPFIRE_CQN_BLOCK 32\n#define HIPFIRE_CQN_KERNEL conv1d_silu_split_qknorm_b32\n",
    include_str!("../../../kernels/src/conv1d_silu_split_qknorm.gfx1201.hip")
);
pub const CONV1D_SILU_SPLIT_QKNORM_B64_SRC: &str = concat!(
    "#define HIPFIRE_CQN_BLOCK 64\n#define HIPFIRE_CQN_KERNEL conv1d_silu_split_qknorm_b64\n",
    include_str!("../../../kernels/src/conv1d_silu_split_qknorm.gfx1201.hip")
);
pub const CONV1D_SILU_SPLIT_QKNORM_B128_SRC: &str = concat!(
    "#define HIPFIRE_CQN_BLOCK 128\n#define HIPFIRE_CQN_KERNEL conv1d_silu_split_qknorm_b128\n",
    include_str!("../../../kernels/src/conv1d_silu_split_qknorm.gfx1201.hip")
);
pub const CONV1D_SILU_SPLIT_QKNORM_B256_SRC: &str = concat!(
    "#define HIPFIRE_CQN_BLOCK 256\n#define HIPFIRE_CQN_KERNEL conv1d_silu_split_qknorm_b256\n",
    include_str!("../../../kernels/src/conv1d_silu_split_qknorm.gfx1201.hip")
);
pub const CONV1D_SILU_SPLIT_QKNORM_B256_SCALAR_PREP_SRC: &str = concat!(
    "#define HIPFIRE_CQN_BLOCK 256\n",
    "#define HIPFIRE_CQN_SCALAR_PREP 1\n",
    "#define HIPFIRE_CQN_KERNEL conv1d_silu_split_qknorm_b256_scalar_prep\n",
    include_str!("../../../kernels/src/conv1d_silu_split_qknorm.gfx1201.hip")
);
pub const CONV1D_SILU_SPLIT_QKNORM_B512_SRC: &str = concat!(
    "#define HIPFIRE_CQN_BLOCK 512\n#define HIPFIRE_CQN_KERNEL conv1d_silu_split_qknorm_b512\n",
    include_str!("../../../kernels/src/conv1d_silu_split_qknorm.gfx1201.hip")
);

/// Tree-aware variant of conv1d_silu_split. Each in-block token walks its
/// ancestor chain via parent_indices[] for the 3-tap causal window, falling
/// back to pre-block conv_state when the chain exits the block. Leaves
/// conv_state unchanged — caller runs linear conv1d on the accepted spine
/// post-acceptance to advance state.
///
/// Ported from SGLang's `causal_conv1d_update` HAS_EAGLE_TREE_CUSTOM_ATTN_MASK
/// branch, simplified to take a precomputed parent_indices[] (our tree layout
/// is materialized host-side by ddtree::linearize_tree).
#[cfg(feature = "deltanet")]
pub const CONV1D_SILU_SPLIT_TREE_SRC: &str =
    include_str!("../../../kernels/src/conv1d_silu_split_tree.hip");

/// GPU-side KV cache write using pos from a GPU buffer.
/// Copies kv_dim floats from src to dst at offset pos_buf[0] * kv_dim.
pub const KV_CACHE_WRITE_SRC: &str = include_str!("../../../kernels/src/kv_cache_write.hip");

/// Batched F32 KV cache write: scatter `batch_size` rows into the cache at
/// the absolute positions array, in one launch. Used by batched prefill.
pub const KV_CACHE_WRITE_F32_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_f32_batched.hip");

/// GPU-side top-K + top-P sampling. Eliminates 600KB logits download per token.
/// Single block, 256 threads. Returns token ID + RNG state (8 bytes vs 600KB).
///
/// Phase 1: Parallel max reduction over vocab_size logits.
/// Phase 2: Threshold filter — collect candidates within 30*temp of max (atomic shared counter).
/// Phase 3: Thread 0 softmax + sort + top-p + sample on the small candidate set.
pub const SAMPLE_TOP_P_SRC: &str = include_str!("../../../kernels/src/sample_top_p.hip");

/// Multi-workgroup parallel rewrite of `sample_top_p`: vocab top-K scan split
/// across N blocks (`sample_topk_partial`) + single-block merge/softmax/top-p
/// (`sample_topk_finalize`) + an in-place penalty prepass
/// (`sample_apply_repeat_penalty`). Byte-identical token to the single-block
/// kernel for distinct logits; recovers ~290 us/token on gfx1100 A3B decode.
pub const SAMPLE_TOP_P_PARALLEL_SRC: &str =
    include_str!("../../../kernels/src/sample_top_p_parallel.hip");

/// Product-semantics sampler for independent continuous-batch lanes.
pub const SAMPLE_ROWS_PF_SRC: &str = include_str!("../../../kernels/src/sample_rows_pf.hip");

/// Per-row temperature-scaled softmax probability gather. For each row r,
/// returns the softmax prob of `indices[r]` under temp-scaled row logits.
/// Used by MTP residual-acceptance sampling: gathers p_draft(c_k) and
/// p_target(c_k) without D2H-ing the full vocab logit row. Saves ~6 MB
/// D2H + ~4 ms host softmax per spec-decode cycle.
pub const SOFTMAX_PROB_GATHER_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/softmax_prob_gather_batched.hip");

/// Batched temperature-scaled softmax, out-of-place. One block per row;
/// writes `probs[r] = softmax(logits[r] / temp)` leaving `logits` untouched.
/// Distribution-parity (tree reduction) to the host `softmax_temp_into`;
/// used behind HIPFIRE_DFLASH_FAST_SAMPLE in the DFlash sampled path.
///
/// This file holds TWO entry points: `softmax_temp_batched_f32` (above) and
/// `softmax_temp_topp_batched_f32`, which additionally emits per-row nucleus
/// (top_p) `tau_cut`/`Z` outputs for AR-equivalent nucleus truncation on the
/// host (see `Gpu::softmax_temp_topp_batched_into_f32`).
pub const SOFTMAX_TEMP_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/softmax_temp_batched.hip");

/// GEMV Q4_F16_G64: matrix-vector multiply with on-the-fly Q4_F16 dequantization.
/// Block layout: f16 scale (2B) + f16 min (2B) + uint8 quants[32] (32B) = 36 bytes per 64 elements.
/// Dequant: weight = (_Float16)(nibble) * scale + min — single FP16 FMA on RDNA.
/// Thread tid reads quants[tid], processes both nibbles (elements tid and tid+32).
pub const GEMV_Q4F16_G64_SRC: &str = include_str!("../../../kernels/src/gemv_q4f16_g64.hip");

/// GEMV Q4_F16_G64 wide: 256 threads, element-strided access, shared memory reduction.
/// Matches F32 GEMV's occupancy pattern to test whether occupancy explains the 40% vs 48% gap.
/// Each thread processes elements tid, tid+256, tid+512, ... across the row.
pub const GEMV_Q4F16_G64_WIDE_SRC: &str =
    include_str!("../../../kernels/src/gemv_q4f16_g64_wide.hip");

/// GEMV Q4_F16_G32: matrix-vector multiply with Q4_F16 group-32 dequantization.
/// Block layout: f16 scale (2B) + f16 min (2B) + uint8 quants[16] (16B) = 20 bytes per 32 elements.
/// Thread tid reads quants[tid&15], extracts its nibble based on tid < 16 or >= 16.
pub const GEMV_Q4F16_G32_SRC: &str = include_str!("../../../kernels/src/gemv_q4f16_g32.hip");

/// Q8_0 embedding lookup: dequantize one row from a Q8_0 table to F32.
/// Block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements.
pub const EMBEDDING_Q8_SRC: &str = include_str!("../../../kernels/src/embedding_q8.hip");

/// Q4_K embedding lookup: dequantize one row from a Q4_K table to F32.
/// Avoids dequanting entire embedding to F32 (saves ~2GB for 150K+ vocabs).
/// 256 threads, one block, strided across the row's Q4_K blocks.
pub const EMBEDDING_Q4K_SRC: &str = include_str!("../../../kernels/src/embedding_q4k.hip");

/// HFQ4-G256 embedding lookup: dequantize one row from HFQ4-G256 table to F32.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 elements.
pub const EMBEDDING_HFQ4G256_SRC: &str =
    include_str!("../../../kernels/src/embedding_hfq4g256.hip");

/// Batched HFQ4-G256 embedding: dequantize N rows in one launch. Reads token ids
/// from a device buffer so the launch is hipGraph-captureable — update the buffer
/// between replays, replay the same graph. Writes into row-major `[N × dim]`.
pub const EMBEDDING_HFQ4G256_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/embedding_hfq4g256_batched.hip");

/// Batched Q8_0 embedding: same hipGraph-captureable pattern as the HFQ4-G256
/// variant. 27B MQ4 targets ship with Q8_0-quantized embedding tables, so the
/// verify hot path needs this variant to enable graph capture on that model.
pub const EMBEDDING_Q8_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/embedding_q8_batched.hip");

/// Batched F16 embedding: copies N rows of an F16 table into `[N × dim]` F32,
/// reading token ids from a device buffer. Keeps the DSpark markov head chain
/// GPU-resident (no per-slot D2H+H2D). The F16→F32 widening is exact, so this
/// is byte-identical to the host `f16_to_f32` fallback it replaces.
pub const EMBEDDING_F16_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/embedding_f16_batched.hip");

/// DSpark bidirectional staging assembly. Builds the per-stage attention
/// key/value buffer `staged[block, head_dim, stage_w]` on-GPU from the
/// committed main_kv ring + the block KV, replacing a host d2h+assemble+h2d
/// that forced 2 stream syncs per stage (3 stages → ~6 syncs/window).
pub const DSPARK_STAGE_KV_SRC: &str = include_str!("../../../kernels/src/dspark_stage_kv.hip");

/// HFQ4-G128 embedding lookup: dequantize one row from HFQ4-G128 table to F32.
pub const EMBEDDING_HFQ4G128_SRC: &str =
    include_str!("../../../kernels/src/embedding_hfq4g128.hip");

/// Batched HFQ4-G128 embedding lookup. Mirrors the G256 device-token-id
/// contract while using the 72-byte / 128-element wire layout.
pub const EMBEDDING_HFQ4G128_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/embedding_hfq4g128_batched.hip");

/// Q4_LUT GEMV: 4-bit with LDS codebook lookup.
/// Block: f16 codebook[16] (32 bytes) + u8 quants[16] (16 bytes) = 48 bytes per 32 elements.
/// Dequant: nibble → LDS[nibble] → f16 → FMA. No scale arithmetic per element.
/// 32 threads (single warp). Processes 8 blocks (256 elems) per outer iteration like Q8.
pub const GEMV_Q4LUT_SRC: &str = include_str!("../../../kernels/src/gemv_q4lut.hip");

/// Wave-cooperative Q4: use warp shuffle to distribute nibbles.
/// Same Q4_F16_G32 format (20 bytes/32 elem = 0.625 B/w).
/// 16 threads load 16 bytes, shuffle to give all 32 threads one nibble each.
/// Avoids the tid<16 conditional branch in the inner loop.
pub const GEMV_Q4WAVE_SRC: &str = include_str!("../../../kernels/src/gemv_q4wave.hip");

/// Q4 stored as Q8: 4-bit precision quantized but stored in int8 (1 byte per weight).
/// Same as Q8_0 format (34 bytes per 32 elements) but values clamped to [-8,7].
/// Gets Q8 occupancy (16 VGPRs, 84% peak BW) at 4-bit quality.
/// 1.0625 bytes/weight — only useful when VRAM is not the constraint.
pub const GEMV_Q4AS8_SRC: &str = include_str!("../../../kernels/src/gemv_q4as8.hip");

/// GEMV Q8_HFQ: split-metadata row layout — scales contiguous, then values contiguous.
/// Row layout: [f16 scales × n_groups | int8 values × K | padding to 128B]
/// Pure sequential value stream with no metadata gaps every 34 bytes.
/// Narrow variant: 32 threads (1 warp), 8x unrolled, warp shuffle reduction.
pub const GEMV_Q8HFQ_SRC: &str = include_str!("../../../kernels/src/gemv_q8hfq.hip");

/// GEMV Q8_HFQ wide: 2 warps per block, each processes one row independently.
/// Same split-metadata layout. 8x unrolled. Grid = ceil(M/2).
pub const GEMV_Q8HFQ_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_q8hfq_wide.hip");

/// Cross-entropy loss: -log(softmax[target]) computed entirely on GPU.
/// Input: logits[vocab_size], target_id (int). Output: loss (float).
/// Single block, 256 threads: parallel log-sum-exp reduction.
pub const CROSS_ENTROPY_LOSS_SRC: &str =
    include_str!("../../../kernels/src/cross_entropy_loss.hip");

/// GPU max-probability: compute max(softmax(logits)) entirely on GPU.
/// Output: single float = probability of the most likely token.
/// Used for early-exit confidence check (downloads 4 bytes instead of 600KB).
pub const MAX_PROB_SRC: &str = include_str!("../../../kernels/src/max_prob.hip");

/// GPU argmax: find index of maximum value.
pub const ARGMAX_SRC: &str = include_str!("../../../kernels/src/argmax.hip");

/// C8 Kernel 0: batched categorical sampler (one block per row).
/// Takes probs[batch*vocab] already softmax'd, tau_cut[batch], z[batch] from
/// softmax_temp_topp_batched_f32; outputs tokens[batch] (i32) and
/// prob_at_token[batch] (f32).  D2H: batch*8 bytes instead of batch*vocab*4.
pub const BATCHED_CATEGORICAL_SAMPLE_SRC: &str =
    include_str!("../../../kernels/src/batched_categorical_sample.hip");

/// C8 Kernel 1: on-GPU chain rejection-sampling accept.
/// Fused accept loop for the chain spec-decode temp>0 path.  Takes
/// tgt_probs[b*vocab], dft_probs[b*vocab], draft_tokens[b], draft_p_at_token[b],
/// tau/z arrays for both sides, rng_seed, cactus_delta; outputs
/// {accept_len, bonus_token, rejected_at, new_rng} in a 16-byte int[4].
pub const CHAIN_ACCEPT_SPEC_SRC: &str = include_str!("../../../kernels/src/chain_accept_spec.hip");

/// Batched argmax: one block per row, writes B indices with one kernel launch.
/// Used by DFlash verify to collapse the B × [vocab] logit download to B × 4 bytes.
pub const ARGMAX_BATCHED_SRC: &str = include_str!("../../../kernels/src/argmax_batched.hip");

/// Single-row argmax that writes the selected token into an on-device MTP
/// token chain, optionally remapping through a compressed-vocab sidecar.
pub const ARGMAX_TOKEN_CHAIN_SRC: &str =
    include_str!("../../../kernels/src/argmax_token_chain.hip");

/// Device-side greedy MTP accept prefix scan over verify argmaxes and draft
/// candidates. Writes compact `[accept_count, bonus_or_minus_one]` result.
pub const GREEDY_ACCEPT_SRC: &str = include_str!("../../../kernels/src/greedy_accept.hip");

/// Fused on-GPU sample+accept for DSpark temp>0 verify. Over resident target
/// logits `[n × vocab]` (batched lm-head), replays the single-block
/// `sample_top_p` draw per position, threads xorshift32 RNG, and lazily
/// early-exits on the first mismatch vs `draft[pos+1]` — replacing the
/// per-position `sample_top_p_pf` host loop (one 8-byte D2H/position) with one
/// launch + one `(n+1)×4`-byte D2H. Byte-identical token sequence to the
/// per-position single-block sampler.
pub const DSPARK_SAMPLE_ACCEPT_LAZY_SRC: &str =
    include_str!("../../../kernels/src/dspark_sample_accept_lazy.hip");

pub const DDTREE_SWOR_WALK_SRC: &str = include_str!("../../../kernels/src/ddtree_swor_walk.hip");

pub const DDTREE_GUMBEL_TOPK_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/ddtree_gumbel_topk_batched.hip");

/// Stage 3a: on-GPU ddtree attention-mask builder. Reads `parent_indices`
/// (already device-resident from H2D) and writes the `attn_bias[big_n ×
/// big_n]` f32 mask in-place, eliminating the ~15 KB H2D per cycle (D4).
/// Grid: [big_n,1,1]. Block: [big_n,1,1]. Each thread i walks the parent
/// chain upward, setting 0.0 for ancestor slots and -INF elsewhere.
pub const DDTREE_BUILD_ATTN_MASK_SRC: &str =
    include_str!("../../../kernels/src/ddtree_build_attn_mask.hip");

// ═══════════════════════════════════════════════════════════════════════════
// Vision encoder kernels (ViT: GEMM, LayerNorm, GELU, bias-add)
// ═══════════════════════════════════════════════════════════════════════════

/// Batched GEMV (= GEMM) for F16 weights, F32 activations.
/// Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T
/// Grid=[M,N], Block=[32]. Each warp computes one dot product via shuffle reduce.
/// DEPRECATED: Use gemm_f16_wmma on gfx1100+ for 10-50x better throughput.
pub const GEMM_F16_SRC: &str = include_str!("../../../kernels/src/gemm_f16.hip");
/// WMMA-accelerated F16×F32 batched GEMM for vision encoder (gfx1100+).
/// Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T.  Tiled 16x16 WMMA, ~10-50x vs naive gemm_f16.
/// Grid=[ceil(M/16), ceil(N/16)], Block=[32].
pub const GEMM_F16_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_f16_wmma.hip");
/// Fused-transpose WMMA GEMM: Y[N,M] = W_f16[M,K] @ X_f32[N,K]^T.
/// Writes transposed output directly (no separate transpose kernel).
/// MB=4: 4 N-subtiles per block (64 N-cols). Grid=[ceil(M/16), ceil(N/64)], block=[32].
pub const GEMM_F16_WMMA_MB4_SRC: &str = include_str!("../../../kernels/src/gemm_f16_wmma_mb4.hip");
/// MB=8: 8 N-subtiles per block (128 N-cols). Grid=[ceil(M/16), ceil(N/128)], block=[32].
pub const GEMM_F16_WMMA_MB8_SRC: &str = include_str!("../../../kernels/src/gemm_f16_wmma_mb8.hip");
/// gfx12/RDNA4 sibling of the MB=8 fused-transpose F16 WMMA GEMM.
pub const GEMM_F16_WMMA_MB8_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_f16_wmma_mb8.gfx12.hip");
/// Tiled F16 GEMM with shared memory (no WMMA dependency, works on all RDNA).
/// ~5-10x faster than naive gemm_f16 via LDS data reuse. Tile size 64K.
pub const GEMM_F16_TILED_SRC: &str = include_str!("../../../kernels/src/gemm_f16_tiled.hip");
/// Fused GEMM + bias: Y[N,M] = X[N,K] @ W_f16[M,K]^T + bias[M].
/// Eliminates transpose + bias_add kernel launches (~7MB saved per linear layer).
/// Grid=[N,1], Block=[256], 8-way unrolled.
pub const GEMM_F16_BIAS_SRC: &str = include_str!("../../../kernels/src/gemm_f16_bias.hip");
/// Optimized vision attention with tiled K/V loading and 4 queries per block.
/// ~3-5x faster than naive vit_attention_f32 via shared memory K/V reuse.
pub const VIT_ATTENTION_OPT_SRC: &str = include_str!("../../../kernels/src/vit_attention_opt.hip");

/// Batched GEMM for F32: Y[M,N] = A[M,K] @ B[N,K]^T
pub const GEMM_F32_SRC: &str = include_str!("../../../kernels/src/gemm_f32.hip");

/// LayerNorm with bias: out = gamma * (x - mean) / sqrt(var + eps) + beta
/// Grid=[batch], Block=[min(256, n)].
pub const LAYERNORM_SRC: &str = include_str!("../../../kernels/src/layernorm.hip");

/// GELU activation (tanh approximation, matches gelu_pytorch_tanh).
pub const GELU_TANH_SRC: &str = include_str!("../../../kernels/src/gelu_tanh.hip");

/// Gemma 4 E-series batched PLE activation with a strided per-layer input.
pub const GEMMA4_PLE_ACTIVATION_SRC: &str =
    include_str!("../../../kernels/src/gemma4_ple_activation.hip");

/// Gemma 4 final-logit soft-capping (in-place): x = tanh(x/cap)*cap. Applied to
/// the LM-head output before sampling. Gemma 4 has NO per-attention-layer softcap.
pub const LOGIT_SOFTCAP_SRC: &str = include_str!("../../../kernels/src/logit_softcap.hip");

/// Transpose: out[c, r] = in[r, c]. Converts [rows, cols] → [cols, rows].
pub const TRANSPOSE_SRC: &str = include_str!("../../../kernels/src/transpose.hip");

/// Fused ViT self-attention: Q@K^T → softmax → @V, reading QKV from [N, 3*hidden].
/// Grid=[n_heads, N]. Each block computes one (head, query_pos) output row.
pub const VIT_ATTENTION_SRC: &str = include_str!("../../../kernels/src/vit_attention.hip");

/// 2D rotary positional embedding for the Qwen3.5-VL vision tower. Rotates Q
/// and K halves of the packed QKV buffer in-place using per-token cos/sin of
/// size `head_dim/2`. See `kernels/src/apply_rope_2d_vision.hip`.
pub const APPLY_ROPE_2D_VISION_SRC: &str =
    include_str!("../../../kernels/src/apply_rope_2d_vision.hip");

/// DFlash draft cross-attention (non-causal, GQA): B queries attend to L
/// keys/values with no causal mask. Grid=[n_heads, B]. See
/// `kernels/src/attention_dflash.hip` for the full contract.
pub const ATTENTION_DFLASH_SRC: &str = include_str!("../../../kernels/src/attention_dflash.hip");

/// Bias-add: X[batch, n] += bias[n] (broadcast over batch dim)
pub const BIAS_ADD_SRC: &str = include_str!("../../../kernels/src/bias_add.hip");

/// Deinterleave: split [Q_h0, Gate_h0, Q_h1, Gate_h1, ...] into separate Q and Gate tensors.
pub const DEINTERLEAVE_SRC: &str = include_str!("../../../kernels/src/deinterleave.hip");

/// Batched deinterleave: same as DEINTERLEAVE but processes N tokens in one launch.
pub const DEINTERLEAVE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/deinterleave_batched.hip");

/// Single-token repeat-interleave Q and K key heads up to value heads count.
pub const REPEAT_INTERLEAVE_QK_SRC: &str =
    include_str!("../../../kernels/src/repeat_interleave_qk.hip");

/// Batched repeat-interleave Q and K key heads up to value heads count.
pub const REPEAT_INTERLEAVE_QK_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/repeat_interleave_qk_batched.hip");

/// PFlash per-block scoring kernel.
/// Reads Q8_0 K cache directly, dequantizes inline, computes per-block
/// mean K and cosine similarity vs the last position's K. Output: one
/// f32 score per block. Phase 2.1 of #93.
pub const PFLASH_SCORE_Q8_KV_SRC: &str =
    include_str!("../../../kernels/src/pflash_score_q8_kv.hip");

/// PFlash per-block scoring kernel — fwht3 K-cache variant.
/// Drop-in replacement for `pflash_score_q8_kv` when the drafter runs
/// with fwht3 KV (the LDS-cliff-free long-context path). Reads fwht3 K
/// cache (4 B cnorm + packed 3-bit TURBO_C3_256 codes), dequantizes
/// inline, and computes cosine in FWHT-rotated space — orthonormal
/// FWHT makes that exactly equal to the original-space cosine, with no
/// inverse FWHT needed in the scoring kernel.
pub const PFLASH_SCORE_FWHT3_KV_SRC: &str =
    include_str!("../../../kernels/src/pflash/score_fwht3_kv.hip");

/// PFlash per-block scoring — fwht4 variant. 4-bit codes / TURBO_C4 LUT.
/// 132 B/head at head_dim=256, two FWHT-128 halves per head. Higher
/// per-element precision than fwht3 (16 centroids vs 8) at the cost of
/// larger K storage. Shipped as a research / ablation variant — fwht3
/// is expected to dominate on the cosine-scorer's throughput/quality
/// curve.
pub const PFLASH_SCORE_FWHT4_KV_SRC: &str =
    include_str!("../../../kernels/src/pflash/score_fwht4_kv.hip");

/// PFlash per-block scoring — fwht2 variant. 2-bit codes / TURBO_C2 LUT.
/// 68 B/head at head_dim=256, two FWHT-128 halves per head. Most
/// aggressive K compression in the family; lowest precision (4
/// centroids). May regress NIAH needle recovery at long ctx — shipped
/// for ablation / lower-bound study.
pub const PFLASH_SCORE_FWHT2_KV_SRC: &str =
    include_str!("../../../kernels/src/pflash/score_fwht2_kv.hip");

// ─── DeepSeek V4 Flash (arch_id=9) — kernels ─────────────────────────────────
// All kernel sources required by the deepseek-v4-flash.mq2lloyd serving path.
// Registered as `pub const X_SRC: &str = include_str!(...)`.

/// MQ2-Lloyd MoE indexed family: routed-experts gate_up + down with
/// device-side topk routing + per-expert pointer table. Mirrors the HFQ4
/// MoE indexed kernels. X must be FWHT-pre-rotated by the caller.
pub const GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.gfx942.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_MFMA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_mfma.gfx942.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_MFMA_F16_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_mfma_f16.gfx942.hip");

pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed.gfx942.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_MFMA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed_mfma.gfx942.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_MFMA_F16_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed_mfma_f16.gfx942.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_ALLRANKS_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_allranks.gfx942.hip");
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_EXPANDED_K4_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4.gfx942.hip");

/// MQ3-Lloyd MoE indexed family (MiniMax-M2, arch_id=10): routed-experts
/// gate_up + down with device-side topk routing + per-expert pointer table.
/// Mirrors the MQ2-Lloyd MoE indexed kernels (3-bit + 8-entry codebook,
/// 112 B/group). X must be FWHT-pre-rotated by the caller.
pub const GEMV_MQ3G256_LLOYD_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_gate_up_indexed.hip");

pub const GEMV_MQ3G256_LLOYD_MOE_DOWN_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_down_indexed.hip");

/// Row-tiled (R=2 / R=4) siblings of
/// `gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed`, bit-exact per row
/// with the incumbent but launched on a `ceil(M/R)` grid so each wave carries R
/// rows and stages the whole quad-step codebook into disjoint LDS slots behind
/// one barrier. OPT-IN ONLY via `HIPFIRE_MQ3_DOWN_ROWS={2,4}`; unset (or 1)
/// keeps the incumbent. See the header comment in the .hip for the static
/// analysis that motivated it.
pub const GEMV_MQ3G256_LLOYD_MOE_DOWN_INDEXED_R4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_down_indexed_r4.hip");

/// Row-tiled MQ2-Lloyd MoE-down GEMV (R=2 / R=4). Same defect pair as the MQ3
/// sibling at the a3b decode shape: 16384 single-wave workgroups, and a K4 main
/// loop that is dead at K=moe_intermediate=512. OPT-IN, default OFF.
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_R4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed_r4.hip");

// ─── MQ*-G256-GL ("global Lloyd") MoE indexed family ─────────────────────────
//
// Same call shape as the MQ2/MQ3-Lloyd MoE indexed kernels above (device-side
// topk routing + per-expert pointer table, X must be FWHT-pre-rotated), with
// two differences that matter to the launchers in gemv.rs:
//
//   1. THE CODEBOOK IS A SCALAR KERNEL ARG, not a per-group fp16 header. The
//      MQ2-GL kernels take 4 floats (`cb0..cb3` = `GL_CB2`), the MQ3-GL kernels
//      take 8 (`cb0..cb7` = `GL_CB3`), inserted between the pointer args and
//      the trailing `int M, int K`.
//   2. THE BLOB IS SoA, two regions: `[M*gpr*IDX B indices][M*gpr*2 B fp16
//      scales]` with IDX = 64 (MQ2-GL) / 96 (MQ3-GL) — not an interleaved
//      per-group stride.
//
// Both DOWN kernels are ATOMIC SELF-COMBINING (they atomicAdd
// `topk_weights[krank] * acc` straight into `x_residual`), exactly like their
// Lloyd siblings — the caller MUST skip `moe_down_combine_k8_batched`.
pub const GEMV_MQ2G256GL_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256gl_moe_gate_up_indexed.hip");

pub const GEMV_MQ2G256GL_MOE_DOWN_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256gl_moe_down_indexed.hip");

pub const GEMV_MQ3G256GL_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256gl_moe_gate_up_indexed.hip");

pub const GEMV_MQ3G256GL_MOE_DOWN_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256gl_moe_down_indexed.hip");

/// Strict superset of fused_rmsnorm_mq_rotate that ALSO writes the
/// plain (non-FWHT) RMSNormed output to a second buffer. Eliminates the
/// follow-up rmsnorm_f32 / rmsnorm_batched launch in call sites that
/// consume both representations (Q8/F16 GEMV reads x_plain; MQ4 GEMV
/// reads x_rot).
pub const FUSED_RMSNORM_MQ_ROTATE_PLAIN_SRC: &str =
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate_plain.hip");

/// DeepSeek V4-asymmetric-clamped variant of `fused_silu_mul_mq_rotate`. Replaces
/// the DeepSeek V4 decode pair `deepseek4_silu_mul_clamp_f32` + `rotate_x_mq` with one
/// launch (saves 1 launch + 8 KB intermediate write/read per layer).
pub const V4F_FUSED_SILU_MUL_CLAMP_MQ_ROTATE_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_fused_silu_mul_clamp_mq_rotate.hip");

/// MQ2-Lloyd grouped GEMM with F16 WMMA — DeepSeek V4 MoE port of the
/// HFQ4 grouped pattern. Same scatter pipeline, codebook-lookup decode.
/// Gated by chunk_size ≥ 256 in DeepSeek V4 dispatch (per Gate 1: tile fill
/// crosses 50 % only above that batch size).
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_k2.hip");

/// gfx12 (RDNA4) sister of `GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC`.
/// Identical 9-arg kernarg contract and grid; the port deltas are the four
/// standard gfx11->gfx12 WMMA rules (half8_t operands, K split across wave
/// halves, `_gfx12` builtin/symbol suffix, 8-row-block accumulator). Landed
/// unwired in be65e2c2c; wired here so a UNIFORM MQ2-Lloyd routed gate_up can
/// take grouped-WMMA batched prefill on gfx1200/gfx1201 instead of falling back
/// to the per-token loop. Correctness bench:
/// `crates/rdna-compute/examples/bench_mq2g256_lloyd_grouped_gfx12.rs`.
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma.gfx12.hip");

/// `(entry_point_name, source)` for the uniform MQ2-Lloyd MoE grouped-WMMA
/// GEMM on the current arch. Single source of truth shared by
/// `Gpu::gemm_mq2g256_lloyd_moe_grouped_wmma` and its no-GPU resolution tests,
/// so a test can prove the launcher names a real `extern "C"` entry point in a
/// real registered source rather than a fallback or a typo.
///
/// The two module names MUST stay distinct: the JIT kernel cache is keyed by
/// module name only, so a shared name would silently make the second arch's
/// source dead code and run the wrong ISA.
pub fn mq2g256_lloyd_moe_grouped_wmma_source(is_gfx12: bool) -> (&'static str, &'static str) {
    if is_gfx12 {
        (
            "gemm_mq2g256_lloyd_moe_grouped_wmma_gfx12",
            GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_GFX12_SRC,
        )
    } else {
        (
            "gemm_mq2g256_lloyd_moe_grouped_wmma_k2",
            GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC,
        )
    }
}

/// MQ3-Lloyd twin of [`mq2g256_lloyd_moe_grouped_wmma_source`].
pub fn mq3g256_lloyd_moe_grouped_wmma_source(is_gfx12: bool) -> (&'static str, &'static str) {
    if is_gfx12 {
        (
            "gemm_mq3g256_lloyd_moe_grouped_wmma_gfx12",
            GEMM_MQ3G256_LLOYD_MOE_GROUPED_WMMA_GFX12_SRC,
        )
    } else {
        (
            "gemm_mq3g256_lloyd_moe_grouped_wmma_k2",
            GEMM_MQ3G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC,
        )
    }
}

/// 4-warp MoE-grouped MQ2-Lloyd WMMA GEMM for gfx1151 (RDNA3.5). 64-row
/// × 16-slot tile (vs 16×16 single-warp baseline), LDS-staged X shared
/// across 4 warps for 4× less B-fragment memory traffic per FLOP. Slot
/// dim stays at 16 due to expert-spanning constraint.
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2.hip");
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2.gfx12.hip");

/// Native gfx942 wave64 MFMA grouped-GEMM for DeepSeek4 MQ2-Lloyd prefill.
/// Chip-strict dispatch keeps CDNA operand/layout semantics out of RDNA paths.
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_MFMA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_mfma.gfx942.hip");

/// i8 WMMA MMQ MoE grouped-GEMM for MQ2-Lloyd weights on gfx1151. Decodes the
/// 2-bit Lloyd index via an in-kernel int8-codebook LUT and runs i8 WMMA at
/// ~2x the FP16 rate. Drop-in for the FP16 `_4w_k2` sibling (same kernarg
/// layout + trailing x_src_rows).
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_MMQ_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_mmq.gfx1151.hip");
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_MMQ_PERM_GFX1151_SRC: &str = concat!(
    "#define MQ2_GROUPED_MMQ_PERM_DECODE 1\n",
    "#define MQ2_GROUPED_MMQ_KERNEL_NAME gemm_mq2g256_lloyd_moe_grouped_mmq_perm_gfx1151\n",
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_mmq.gfx1151.hip")
);

pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_MMQ_PREQUANT_PERM_GFX1151_SRC: &str = concat!(
    "#define MQ2_GROUPED_MMQ_PERM_DECODE 1\n",
    "#define MQ2_GROUPED_MMQ_PREQUANT_CODEBOOK 1\n",
    "#define MQ2_GROUPED_MMQ_KERNEL_NAME gemm_mq2g256_lloyd_moe_grouped_mmq_prequant_perm_gfx1151\n",
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_mmq.gfx1151.hip")
);
/// Exact-gfx1030 sdot4 MMQ probe for MQ2-Lloyd grouped GEMM. Production MoE
/// dispatch does not reference this source.
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_MMQ_GFX1030_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_mmq.gfx1030.hip");
pub const GEMM_MQ3G256_LLOYD_MOE_GROUPED_MMQ_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq3g256_lloyd_moe_grouped_mmq.gfx1151.hip");

pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_N32_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32.hip");
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_N32_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32.gfx12.hip");

pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_CND_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd.hip");

pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_8W_K2_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2.hip");

/// F16-weight × F32-input GEMV. Used for full-precision MTP weights where
/// the WMMA F16×F16 path's F32→F16 input conversion loses precision.
pub const GEMV_F16_XF32_SRC: &str = include_str!("../../../kernels/src/gemv_f16_xf32.hip");
/// BF16-weight × F32-input GEMV. Native-bf16 reference path (KLD oracle) —
/// keeps the exact downloaded bf16 values (lossless widen = 16-bit shift),
/// unlike re-quantizing to f16. arch_id 12 (Cohere2-MoE).
pub const GEMV_BF16_XF32_SRC: &str = include_str!("../../../kernels/src/gemv_bf16_xf32.hip");

/// DeepSeek V4 SwiGLU with swiglu_limit clamp: silu(min(gate, L)) * clamp(up, ±L)
/// L = swiglu_limit (DeepSeek V4 config = 10.0).
pub const V4F_SILU_MUL_CLAMP_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_silu_mul_clamp.hip");

/// DeepSeek V4 MoE router: bias-aware top-K + normalized scaled weights, fully
/// GPU-side. Replaces the per-layer D2H/CPU/H2D round-trip.
pub const V4F_MOE_TOPK_BIAS_AWARE_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_moe_topk_bias_aware.hip");

// ─── DeepSeek V4 Flash (arch_id=7) — stub kernels ────────────────────────────
// All five are functional-stub implementations whose API contract is the
// signature; bodies are placeholder reference impls until DeepSeek V4 forward
// bring-up lands optimised versions. See `docs/plans/deepseek4-phase{2,3,4}-*.md`.
//
// Phase 2 — Compressed-KV indexer:
pub const INDEXER_COMPRESSED_K_SCORE_SRC: &str =
    include_str!("../../../kernels/src/indexer_compressed_k_score.hip");

pub const INDEXER_TOP_K_SRC: &str = include_str!("../../../kernels/src/indexer_top_k.hip");

pub const INDEXER_TOP_K_BUF_SRC: &str = include_str!("../../../kernels/src/indexer_top_k_buf.hip");
pub const INDEXER_TOP_K_BUF_PARALLEL_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_parallel.gfx1151.hip");
pub const INDEXER_TOP_K_BUF_BOUNDED_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_bounded.gfx1151.hip");
pub const INDEXER_TOP_K_BUF_UNROLLED_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_unrolled.gfx1151.hip");
pub const INDEXER_TOP_K_BUF_BOUNDED_GFX942_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_bounded.gfx942.hip");
/// Portable two-stage indexer top-K (chunk-sort + merge tree). Wave-size
/// agnostic: selected on gfx942 via F3 and gfx1151 via G1, compiled from one
/// source through `Gpu::indexer_top_k_two_stage` in attention.rs.
pub const INDEXER_TOPK_TWOSTAGE_SRC: &str =
    include_str!("../../../kernels/src/indexer_topk_twostage.hip");

pub const INDEXER_KV_GATHER_SRC: &str = include_str!("../../../kernels/src/indexer_kv_gather.hip");

// Phase 3 — Hyper-Connections:
pub const HC_COMPUTE_CONTROL_SRC: &str =
    include_str!("../../../kernels/src/hc_compute_control.hip");

/// Wide-block `hc_compute_control_vec4_finalize`: 1024 threads instead of 256.
///
/// The shipped 256-thread launch runs 24 blocks x 8 waves = 192 waves = 0.15
/// occupancy fills on gfx1151, and moves 786 KB at 71.8 GB/s against a
/// measured 207 GB/s ceiling — 35% efficiency, the worst of any kernel in the
/// ds4 AR decode step. Its critical path is one block's full 16384-element
/// reduction, because blocks 1..23 spin on block 0's rsqrt and the last block
/// to finish runs the sinkhorn epilogue; 1024 threads cuts that block's loop
/// from 16 iterations to 4 and lifts occupancy to 0.6 fills.
///
/// Distinct module AND symbol name: the JIT cache is keyed by module name, so
/// a variant loaded under the shipped name would silently be dead code.
///
/// NOT bit-exact with the 256-thread kernel — the LDS partial tree is 32-wide
/// instead of 8-wide, so the summation order differs. Validate through
/// `serve_harness.py`, not a ULP gate.
pub const HC_COMPUTE_CONTROL_T1024_SRC: &str = concat!(
    "#define THREADS_PER_BLOCK 1024\n",
    "#define HC_CTRL_FINALIZE_KERNEL hc_compute_control_vec4_finalize_t1024\n",
    include_str!("../../../kernels/src/hc_compute_control.hip")
);

pub const HC_SINKHORN_4X4_SRC: &str = include_str!("../../../kernels/src/hc_sinkhorn_4x4.hip");

pub const HC_MIX_4STREAM_SRC: &str = include_str!("../../../kernels/src/hc_mix_4stream.hip");

/// gfx1201 TP3/TP4 peer reduction fused into the Hyper-Connection consumer.
pub const HC_MIX_4STREAM_PEER4_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/hc_mix_4stream_peer4.gfx1201.hip");

/// gfx1201 TP3/TP4 graph-resident system-scope producer barrier.
pub const TP_GRAPH_SIGNAL_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/tp4_graph_signal.gfx1201.hip");

pub const HC_INPUT_MAP_SRC: &str = include_str!("../../../kernels/src/hc_input_map.hip");

pub const HC_APPLY_ALPHA_SRC: &str = include_str!("../../../kernels/src/hc_apply_alpha.hip");

pub const HC_FINALIZE_CONTROL_SRC: &str =
    include_str!("../../../kernels/src/hc_finalize_control.hip");
pub const HC_FINALIZE_INPUT_MAP_SRC: &str =
    include_str!("../../../kernels/src/hc_finalize_input_map.hip");

pub const ZERO_F32_SRC: &str = include_str!("../../../kernels/src/zero_f32.hip");

/// Zero inactive rows for EP routed partials / final hidden isolation.
/// Masked sibling writes exact +0.0f only to inactive rows (row = lane).
pub const ZERO_INACTIVE_ROWS_F32_SRC: &str =
    include_str!("../../../kernels/src/zero_inactive_rows_f32.hip");

pub const SQRT_SOFTPLUS_F32_SRC: &str = include_str!("../../../kernels/src/sqrt_softplus_f32.hip");

pub const V4F_ATTN_POS0_SRC: &str = include_str!("../../../kernels/src/deepseek4_attn_pos0.hip");

pub const V4F_ATTN_SWA_SRC: &str = include_str!("../../../kernels/src/deepseek4_attn_swa.hip");

/// HIP-graphs-safe twin of `deepseek4_attn_swa`: reads `n_valid` from a
/// device buffer instead of an i32 kernarg.
pub const V4F_ATTN_SWA_BUF_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_buf.hip");

/// DeepSeek V4 mHC pre+post sigmoid/scale fusion — replaces 3 element-wise
/// launches (sigmoid(pre), sigmoid(post), scale(post)) with 1.
pub const HC_PRE_POST_SIGMOID_SCALE_SRC: &str =
    include_str!("../../../kernels/src/hc_pre_post_sigmoid_scale.hip");

/// HIP-graphs-safe twin of compressor_softmax_pool_f32: reads
/// destination slot from a device buffer; early-returns on slot < 0
/// (so captured graph can include the commit kernels at every replay
/// while host gates on `commit_slot >= 0` only at actual commit positions).
pub const COMPRESSOR_SOFTMAX_POOL_BUF_SRC: &str =
    include_str!("../../../kernels/src/compressor_softmax_pool_buf.hip");

/// HIP-graphs-safe in-place RMSNorm at slot `slot_buf[0]` of a base
/// buffer; -1 sentinel → no-op. Twin of `rmsnorm_f32(kv_cache.sub_offset(slot*n, n))`.
pub const RMSNORM_AT_SLOT_BUF_SRC: &str =
    include_str!("../../../kernels/src/rmsnorm_at_slot_buf.hip");

/// DeepSeek V4 hash-routed MoE: GPU-side tid2eid lookup + score normalize +
/// route_scale multiply. Replaces the d2h+host+h2d round-trip in
/// `ffn_hash_routed` for hash-layered MoE dispatch.
pub const HASH_ROUTER_NORMALIZE_SRC: &str =
    include_str!("../../../kernels/src/hash_router_normalize_f32.hip");

/// HIP-graphs-safe twin of `HASH_ROUTER_NORMALIZE_SRC` — reads
/// `token_id` from a device buffer so the captured graph re-reads
/// it on every replay.
pub const HASH_ROUTER_NORMALIZE_BUF_SRC: &str =
    include_str!("../../../kernels/src/hash_router_normalize_f32_buf.hip");

/// Batched variant for the prefill `ffn_batched` hash-routed path:
/// per-batch tid2eid lookup + score gather + normalize + route_scale,
/// in one launch — eliminates the d2h(scores)+CPU+h2d round-trip per
/// hash-routed layer per chunk.
pub const HASH_ROUTER_NORMALIZE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hash_router_normalize_f32_batched.hip");

/// HIP-graphs-safe in-place YaRN-aware tail RoPE at slot `slot_buf[0]` of a
/// base buffer; -1 sentinel → no-op. Single-tensor (n_heads_q=1, n_heads_k=0).
/// Pass freq_scale=1.0, ext_factor=0.0 to recover plain rope_tail_interleaved.
pub const ROPE_TAIL_YARN_INTERLEAVED_AT_SLOT_BUF_SRC: &str =
    include_str!("../../../kernels/src/rope_tail_yarn_interleaved_at_slot_buf.hip");

/// HIP-graphs-safe ring write: src[proj_dim] → state[slot*proj_dim..]
/// with slot from `ring_slot_buf[0]`. Twin of the per-position
/// `memcpy_dtod_auto` writes in compressor_forward_impl.
pub const STATE_RING_WRITE_F32_BUF_SRC: &str =
    include_str!("../../../kernels/src/state_ring_write_f32_buf.hip");

/// HIP-graphs-safe overlap-shift: state[:ratio*proj_dim] = state[ratio*proj_dim:].
/// Gated by `commit_slot_buf[0] >= 0` so captured graphs only fire it on
/// commit positions. Twin of the post-commit memcpy_dtod_auto state shift.
pub const STATE_OVERLAP_SHIFT_F32_BUF_SRC: &str =
    include_str!("../../../kernels/src/state_overlap_shift_f32_buf.hip");

/// HIP-graphs-safe twin of `deepseek4_attn_swa_topk_f32`: reads `n_valid_swa`
/// + `n_active_topk` from device buffers.
pub const V4F_ATTN_SWA_TOPK_BUF_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_buf.hip");
pub const V4F_ATTN_SWA_TOPK_BUF_XLANE_GFX1151_SRC: &str = concat!(
    "#include <hip/hip_runtime.h>\n",
    include_str!("../../../kernels/src/gfx1151_xor_shuffle.h"),
    "\n#define HIPFIRE_GFX1151_ATTN_SCOREGRID_XLANE 1\n",
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_buf.hip")
);
pub const V4F_ATTN_SWA_TOPK_BUF_LARGE_SERIAL_GFX1151_SRC: &str = concat!(
    "#define HIPFIRE_GFX1151_ATTN_SCOREGRID_LARGE_SERIAL 1\n",
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_buf.hip")
);

/// HIP-graphs-safe twin of `deepseek4_topk_kv_gather_f32`: reads K + N_compressed
/// from device buffers. Launch with fixed grid = MAX_K; lanes beyond K
/// early-return.
pub const V4F_TOPK_KV_GATHER_BUF_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_buf.hip");
pub const V4F_TOPK_KV_GATHER_BUF_SHARDED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_buf_sharded.gfx1201.hip");
pub const V4F_TOPK_KV_GATHER_TILED_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_tiled.gfx1151.hip");

/// HIP-graphs-safe twin of `deepseek4_topk_kv_gather_identity_f32`.
pub const V4F_TOPK_KV_GATHER_IDENTITY_BUF_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_identity_buf.hip");
pub const V4F_TOPK_KV_GATHER_IDENTITY_BUF_SHARDED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_identity_buf_sharded.gfx1201.hip");

/// HIP-graphs-safe variant of swa_ring_write_f32: reads `slot` from a
/// device buffer instead of an i32 kernarg, so the captured kernel
/// picks up new positions on each replay without re-capture.
pub const SWA_RING_WRITE_BUF_SRC: &str =
    include_str!("../../../kernels/src/swa_ring_write_buf.hip");

/// Tail-only RoPE, INTERLEAVED pair convention (DeepSeek V4 upstream's
/// `torch.view_as_complex` variant, distinct from HF rotate_half).
pub const ROPE_TAIL_INTERLEAVED_SRC: &str =
    include_str!("../../../kernels/src/rope_tail_interleaved.hip");

/// Exact-gfx1201 head-parallel indexer-Q RoPE for DS4 H64/D128/R64.
pub const ROPE_TAIL_INTERLEAVED_H64D128R64_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/rope_tail_interleaved_h64d128r64.gfx1201.hip");

/// YaRN-aware tail-only RoPE for compressed-layer attention (DeepSeek V4).
/// Adds per-call freq_scale / ext_factor / attn_factor / corr_dims to
/// match antirez/ds4 rope_tail_ext_inplace. For dense (uncompressed)
/// layers, caller passes ext_factor=0 to disable YaRN — math collapses
/// to standard RoPE.
pub const ROPE_TAIL_YARN_INTERLEAVED_SRC: &str =
    include_str!("../../../kernels/src/rope_tail_yarn_interleaved.hip");

/// Tail-only RoPE — BATCHED (Phase B2, 2026-05-18). Per-batch positions
/// from a device array; rotation on the LAST n_rot dims of each head.
pub const ROPE_TAIL_INTERLEAVED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/rope_tail_interleaved_batched.hip");

/// YaRN-aware tail RoPE — BATCHED (Phase B2, 2026-05-18). Batched twin
/// of ROPE_TAIL_YARN_INTERLEAVED_SRC.
pub const ROPE_TAIL_YARN_INTERLEAVED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/rope_tail_yarn_interleaved_batched.hip");

/// HC control-vector — BATCHED (Phase B2, 2026-05-18). Per-batch dot
/// of streams[b] against the shared `hc_fn` weight + rsqrt mean + base.
pub const HC_COMPUTE_CONTROL_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hc_compute_control_batched.hip");
pub const HC_INV_RMS_BATCHED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/hc_inv_rms_batched.gfx1201.hip");
pub const HC_COMPUTE_CONTROL_BATCHED_FUSED24_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/hc_compute_control_batched_fused24.gfx1201.hip");
pub const HC_COMPUTE_CONTROL_WMMA_B1_GFX1201_SRC: &str = concat!(
    "#define HC_WMMA_BATCH_TILES 1\n",
    "#define HC_WMMA_KERNEL_NAME hc_compute_control_wmma_b1_gfx1201\n",
    include_str!("../../../kernels/src/hc_compute_control_wmma.gfx1201.hip")
);
pub const HC_COMPUTE_CONTROL_WMMA_B2_GFX1201_SRC: &str = concat!(
    "#define HC_WMMA_BATCH_TILES 2\n",
    "#define HC_WMMA_KERNEL_NAME hc_compute_control_wmma_b2_gfx1201\n",
    include_str!("../../../kernels/src/hc_compute_control_wmma.gfx1201.hip")
);
pub const HC_COMPUTE_CONTROL_WMMA_B4_GFX1201_SRC: &str = concat!(
    "#define HC_WMMA_BATCH_TILES 4\n",
    "#define HC_WMMA_KERNEL_NAME hc_compute_control_wmma_b4_gfx1201\n",
    include_str!("../../../kernels/src/hc_compute_control_wmma.gfx1201.hip")
);

/// HC α-scaling post-step — BATCHED (Phase B2, 2026-05-18). Per-batch
/// in-place rescale of c[b, 0..24] using the shared 3-segment α + base.
pub const HC_APPLY_ALPHA_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hc_apply_alpha_batched.hip");

/// HC Sinkhorn 4×4 — BATCHED (Phase B2, 2026-05-18). Per-batch
/// independent Sinkhorn iterations on each 4×4 matrix slot.
pub const HC_SINKHORN_4X4_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hc_sinkhorn_4x4_batched.hip");

/// HC split/finalize — BATCHED (Phase B2, 2026-05-18). Splits the
/// post-α-rescale c[B, 24] into contiguous pre/post/comb buffers with
/// sigmoid + scale already applied. Avoids strided sigmoid_f32 calls.
pub const HC_SPLIT_FINALIZE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hc_split_finalize_batched.hip");

/// SWA visibility staging — BATCHED (Phase B2, 2026-05-18). Per batch
/// position b at absolute position start_pos+b: builds the visibility
/// window from the pre-chunk SWA ring + within-chunk KV. Output feeds
/// deepseek4_attn_swa_topk_batched / deepseek4_attn_swa_batched.
pub const SWA_VISIBILITY_STAGE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/swa_visibility_stage_batched.hip");

/// Typed device-buffer copies used inside retained replay bodies. HIP memcpy
/// nodes are not kernel launches and therefore are not part of Redline's typed
/// launch recorder; these kernels make the dependency explicit and auditable.
pub const COPY_F32_BUFFER_SRC: &str = include_str!("../../../kernels/src/copy_f32_buffer.hip");

/// DeepSeek V4 top-K K/V gather — BATCHED (Phase B2, 2026-05-18). Per-batch
/// top-K gather from the shared main compressed-K cache into a
/// `[B, head_dim, out_stride]` buffer fed to deepseek4_attn_swa_topk_batched.
pub const V4F_TOPK_KV_GATHER_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_batched.hip");
pub const V4F_TOPK_KV_GATHER_BATCHED_TILED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_batched_tiled.gfx1201.hip");
pub const V4F_TOPK_KV_GATHER_BATCHED_TILED_SHARDED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_batched_tiled_sharded.gfx1201.hip");

/// DeepSeek V4 indexer score — BATCHED (Phase B2, 2026-05-18). Per-batch
/// score against the shared compressed-K cache.
pub const INDEXER_RELU_SCORE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/indexer_relu_score_batched.hip");

/// DeepSeek V4 indexer score — WMMA-accelerated BATCHED (Phase C1,
/// 2026-05-26). Replaces the F32 scalar one-thread-per-head baseline
/// with a 16×16×16 WMMA tile of Q·K^T per warp; 4 warps cover the
/// 64-head reduction in LDS.
pub const INDEXER_RELU_SCORE_WMMA_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/indexer_relu_score_wmma_batched.hip");
pub const INDEXER_RELU_SCORE_WMMA_BATCHED_GFX12_SRC: &str =
    include_str!("../../../kernels/src/indexer_relu_score_wmma_batched.gfx12.hip");
pub const INDEXER_RELU_SCORE_WMMA_BATCHED_SHARDED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/indexer_relu_score_wmma_batched_sharded.gfx1201.hip");

/// gfx1151 DeepSeek V4 indexer score with one cooperative LDS-staged K tile
/// shared by the four WMMA warps in each block.
pub const INDEXER_RELU_SCORE_WMMA_BATCHED_KLDS_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/indexer_relu_score_wmma_batched_klds.gfx1151.hip");

/// Wider-N Q8 WMMA: 16×64 output tile instead of 16×16, 4× weight
/// reuse per block. Same single-warp wave32 structure as
/// `gemm_q8_0_wmma`, but each K-step issues 4 back-to-back WMMA tiles
/// sharing one A (weight) fragment. Lands the structural lever
/// identified in llama.cpp issue 21284 (pedapudi) — the "wider tile"
/// gfx1151 prefill optimisation.
pub const GEMM_Q8_0_WMMA_X64_SRC: &str =
    include_str!("../../../kernels/src/gemm_q8_0_wmma_x64.hip");

/// SWA ring write — BATCHED (Phase B2, 2026-05-18). Advances the ring
/// at chunk end to include all B positions. Slot = (start_pos+b) % win.
pub const SWA_RING_WRITE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/swa_ring_write_batched.hip");

/// DeepSeek V4 identity gather — BATCHED (Phase B2, 2026-05-18). For ratio=128
/// layers that lack an indexer: copies kv_cache[0..K, :] into every
/// batch row's slab.
pub const V4F_TOPK_KV_GATHER_IDENTITY_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_identity_batched.hip");
pub const V4F_TOPK_KV_GATHER_IDENTITY_BATCHED_SHARDED_GFX1201_SRC: &str = include_str!(
    "../../../kernels/src/deepseek4_topk_kv_gather_identity_batched_sharded.gfx1201.hip"
);

/// DeepSeek V4 per-group O-LoRA batched GEMV — F32 (Phase B2, 2026-05-18).
/// Block-diagonal: wo_a[G, M, K] @ x_in[B, G, K] → y_out[B, G, M].
pub const WO_PER_GROUP_BATCHED_F32_SRC: &str =
    include_str!("../../../kernels/src/wo_per_group_batched_f32.hip");

/// DeepSeek V4 per-group O-LoRA batched GEMV for HFQ4G256-packed wo_a.
/// Single launch in place of B × G separate gemv_mq4g256_prerotated calls.
/// Collapses ~11k dispatch calls/chunk down to 43 in the DeepSeek V4 prefill path.
pub const WO_PER_GROUP_BATCHED_HFQ4G256_SRC: &str =
    include_str!("../../../kernels/src/wo_per_group_batched_hfq4g256.hip");

/// DeepSeek V4 per-group O-LoRA batched GEMV for Q8_0-packed wo_a (Phase D,
/// 2026-05-21). Sibling of `wo_per_group_batched_hfq4g256` for the
/// deepseek4-mq2lloyd-q8 build where wo_a is Q8_0. Single launch in place of
/// B × G `gemv_q8_0` calls — collapses ~32k per-chunk dispatches.
pub const WO_PER_GROUP_BATCHED_Q8_0_SRC: &str =
    include_str!("../../../kernels/src/wo_per_group_batched_q8_0.hip");

/// Multi-row Q8_0 variant (Lever 1). Same contract as the single-row
/// `wo_per_group_batched_q8_0` but with block processing R output rows
/// and hoisting x loads across rows. Grid = [ceil(M/R), B, G].
pub const WO_PER_GROUP_BATCHED_Q8_0_MULTIROW_SRC: &str =
    include_str!("../../../kernels/src/wo_per_group_batched_q8_0_multirow.hip");

/// MMQ-style preload variant of the 4-warp MoE grouped MQ2-Lloyd kernel.
/// Pre-loads all 8 index packs per K-group before the inner loop so the
/// hardware prefetcher starts on the second cache line earlier.
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_MMQLOAD_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload.hip");

/// Barrier-free variant of the mmqload kernel. Eliminates __syncthreads()
/// and LDS X staging. Each wave loads X directly from global memory.
pub const GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_MMQLOAD_NOSYNC_SRC: &str = include_str!(
    "../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync.hip"
);

/// WMMA Q8_0 GEMM for DeepSeek V4 O-LoRA's strided `[B, G, *]` layout.
pub const WO_PER_GROUP_BATCHED_Q8_0_WMMA_4W_SRC: &str =
    include_str!("../../../kernels/src/wo_per_group_batched_q8_0_wmma_4w.hip");
/// DeepSeek V4 MoE router top-K — BATCHED (Phase B2, 2026-05-18). Per-batch
/// bias-aware top-K + normalize + route_scale, one block per batch row.
pub const V4F_MOE_TOPK_BIAS_AWARE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_moe_topk_bias_aware_batched.hip");

/// WMMA F16 × F16 → F32 GEMM with (B, M) output layout.
/// Replaces gemm_f32_register_tiled for DeepSeek V4 compressor when weights
/// stay F16 on device. Targets gfx1100+ wave32 WMMA.
pub const GEMM_F16_X_F16_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_f16_x_f16_wmma.hip");

/// CDNA3 (gfx942) MFMA port of `GEMM_F16_X_F16_WMMA_SRC` — same math,
/// same `(A, X, Y, M, K, B)` signature, same `[B, M]` F32 output layout.
/// The WMMA original is wave32-only (`__builtin_amdgcn_wmma_*_w32` needs
/// `gfx11-insts,wavefrontsize32`) and cannot compile for gfx942; this uses
/// the wave64 `__builtin_amdgcn_mfma_f32_16x16x16f16` instead. Required by
/// the DeepSeek V4 DSpark prefill path on MI300X.
pub const GEMM_F16_X_F16_MFMA_GFX942_SRC: &str =
    include_str!("../../../kernels/src/gemm_f16_x_f16_mfma.gfx942.hip");

/// Bulk F32→F16 conversion for staging WMMA activations. Named
/// `deepseek4_convert_f32_to_f16` to avoid collision with the embedded
/// `convert_f32_to_f16` helper in `GEMM_HFQ4G256_RESIDUAL_FP16_SRC`
/// (different ABI: block=256, int n). See `gpu.deepseek4_convert_f32_to_f16`
/// for the DeepSeek V4 dispatcher.
pub const V4F_CONVERT_F32_TO_F16_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_convert_f32_to_f16.hip");

/// WMMA HFQ4G256 weight × F16 input → F32 output GEMM with (B, M)
/// output layout. Drop-in for `gemm_hfq4g256` (scalar FMA path).
pub const GEMM_HFQ4G256_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_wmma.hip");

/// gfx12 (RDNA4) sibling of GEMM_HFQ4G256_WMMA_SRC -- half8 operands,
/// _w32_gfx12 builtin, rows-contiguous C mapping. Plain store (no residual).
pub const GEMM_HFQ4G256_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_wmma.gfx12.hip");

/// Q8_0 indexed MoE gate_up GEMV (Gemma 4 26B-A4B with --expert-q8).
/// Same device-side expert-pointer table + topk_indices contract;
/// Q8_0 weight layout (34 B/block of 32 elements).
pub const GEMV_Q8_0_MOE_GATE_UP_K8_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_q8_0_moe_gate_up_k8_indexed.hip");

/// Q8_0 indexed MoE down-projection with fused scaled atomicAdd.
pub const GEMV_Q8_0_MOE_DOWN_RESIDUAL_SCALED_K8_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_q8_0_moe_down_residual_scaled_k8_indexed.hip");

// ─── Gemma 4 hd512 attention + KV write kernels ─────────────────────────
// (ROPE_PARTIAL_HALVED_SRC / LOGIT_SOFTCAP_SRC already defined above.)

pub const ATTENTION_FLASH_ASYM3_TILE_HD512_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym3_tile_hd512.hip");
pub const ATTENTION_FLASH_ASYM3_TILE_HD512_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_asym3_tile_hd512_batched.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS3_HD512_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens3_hd512.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS3_HD512_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_asym_k_givens3_hd512_batched.hip");
pub const KV_CACHE_WRITE_FWHT3_HD512_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_fwht3_hd512.hip");
pub const KV_CACHE_WRITE_FWHT3_HD512_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/kv_cache_write_fwht3_hd512_batched.hip");
pub const ATTENTION_FLASH_FWHT3_TILE_HD512_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht3_tile_hd512.hip");
pub const ATTENTION_FLASH_FWHT3_TILE_HD512_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/attention_flash_fwht3_tile_hd512_batched.hip");

/// DeepSeek V4 compressor batched ALIGNED compress events. Replaces the
/// 3-kernel per-event chain (overlap_concat × 2 + softmax_pool)
/// with a single launch over N_events. Handles both overlap=true
/// (ratio=4) and overlap=false (ratio=128) cases.
pub const COMPRESSOR_COMPRESS_ALIGNED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/compressor_compress_aligned_batched.hip");

/// DeepSeek V4 compressor batched ring-buffer write. Replaces B per-position
/// memcpy_dtod calls with a single scatter kernel.
pub const COMPRESSOR_RING_WRITE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/compressor_ring_write_batched.hip");

/// DeepSeek V4 compressor per-slot APE add over batched score buffer.
/// Mirrors the per-position add inside `compressor_forward_impl` so that
/// the batched-prefill compress path produces the same kv_cache entries
/// as the sequential per-position path.
pub const COMPRESSOR_ADD_APE_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/compressor_add_ape_batched.hip");

/// K4-unrolled batched MoE gate_up for MQ2-Lloyd (Phase 1, 2026-05-19).
/// 4 independent accumulators per thread for ILP; mirrors qwen35's
/// HFQ4 K4 unroll. Drop-in replacement for
/// gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched with FMA-order
/// epsilon drift.
pub const GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_BATCHED_K4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4.hip");

/// Fixed-K=4096 batched gate/up with one cooperative load of all 16 codebooks.
pub const GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_BATCHED_K4096_LDS_SRC: &str = include_str!(
    "../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4096_lds.hip"
);

/// DeepSeek V4 MoE down — POSITION-BATCHED MQ2-Lloyd indexed GEMV with K4-unrolled
/// accumulator and scaled residual atomicAdd. Sibling of qwen35's HFQ4 K4
/// unroll. Drop-in replacement for
/// gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched with
/// FMA-order epsilon drift.
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_BATCHED_K4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed_batched_k4.hip");

/// Position-batched form of the shipping K=2048 K8-all routed-down kernel.
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_BATCHED_K8ALL_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed_batched_k8all.hip");

pub const GEMV_MQ3G256_LLOYD_MOE_DOWN_INDEXED_BATCHED_K4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_down_indexed_batched_k4.hip");

/// DeepSeek V4 head HC mix — compute per-stream pre weights for the final
/// 4-stream → hidden projection before lm_head.
pub const HC_HEAD_COMPUTE_PRE_SRC: &str =
    include_str!("../../../kernels/src/hc_head_compute_pre.hip");

/// DeepSeek V4 Compressor softmax-weighted pool along window dim.
/// Used in Compressor.forward when should_compress fires every
/// `ratio` steps, to produce a single compressed KV vector from
/// T accumulated step values.
pub const COMPRESSOR_SOFTMAX_POOL_SRC: &str =
    include_str!("../../../kernels/src/compressor_softmax_pool.hip");

/// DeepSeek V4 Compressor overlap-transform concat. Builds the [2*ratio,
/// head_dim] view for compression from the [2*ratio, 2*head_dim]
/// kv_state / score_state buffer (overlap=true, ratio=4 case).
pub const COMPRESSOR_OVERLAP_CONCAT_SRC: &str =
    include_str!("../../../kernels/src/compressor_overlap_concat.hip");

pub const INDEXER_RELU_SCORE_BUF_SRC: &str =
    include_str!("../../../kernels/src/indexer_relu_score_buf.hip");
pub const INDEXER_RELU_SCORE_BUF_SHARDED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/indexer_relu_score_buf_sharded.gfx1201.hip");
pub const DEEPSEEK4_COMPRESSOR_CACHE_F16_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_compressor_cache_f16.hip");
pub const DEEPSEEK4_COMPRESSOR_COMMIT_STAGED_F16_SRC: &str =
    include_str!("../../../kernels/src/compressor_commit_staged_f16.hip");

/// DeepSeek V4 batched indexer-extended SWA attention (Phase A1, 2026-05-18).
/// Processes B query positions in parallel via grid dim Y. Each batch
/// position has its own SWA / top-K K/V slices and valid-count scalars.
pub const V4F_ATTN_SWA_TOPK_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_batched.hip");

/// DeepSeek V4 SWA + indexer top-K attention, direct main-KV variant.
pub const V4F_ATTN_SWA_TOPK_DIRECT_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_direct_batched.hip");

/// Head-batched f16-WMMA port of the direct-batched DSA attention (gfx1151).
/// Same joint-softmax math; 16 heads/block so the score/output GEMVs become
/// WMMA GEMMs reading the shared K/V once. See deepseek4_attn_swa_topk_direct_wmma.hip.
pub const V4F_ATTN_SWA_TOPK_DIRECT_WMMA_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_direct_wmma.hip");

/// Head-batched f16-WMMA port of the gathered DSA attention (top-K staged into
/// topk_kv[B,D,topk_win]). Sibling of the direct WMMA kernel.
pub const V4F_ATTN_SWA_TOPK_BATCHED_WMMA_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_batched_wmma.hip");

/// gfx1201-native sister of the gathered DSA WMMA kernel. Uses RDNA4's
/// half8 lane split and accepts a masked final head group for TP3's 24-head
/// ranks without changing the gfx11 symbol or route.
pub const V4F_ATTN_SWA_TOPK_BATCHED_WMMA_GFX12_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_batched_wmma.gfx12.hip");

/// DeepSeek V4 batched pure-SWA attention (Phase A2, 2026-05-18). Twin of
/// `V4F_ATTN_SWA_TOPK_BATCHED_SRC` for layers without an indexer top-K
/// path. Same launch shape and byte-equality contract at batch=1.
pub const V4F_ATTN_SWA_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_batched.hip");

/// DeepSeek V4 batched indexer top-K (Phase A3, 2026-05-18). Per (batch, head)
/// pair selects the top-K position indices from a score array. Grid
/// extends to `[n_idx_heads, batch, 1]`. Byte-identical to the
/// sequential indexer_top_k at batch=1.
pub const INDEXER_TOP_K_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_batched.hip");

/// gfx1151 exact bounded batched indexer top-K. One workgroup per
/// (batch, head) incrementally merges K-sized runs in static LDS, replacing
/// the portable O(N^2) rank-count once N exceeds K during DS4 prefill.
pub const INDEXER_TOP_K_BATCHED_BOUNDED_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_batched_bounded.gfx1151.hip");

/// gfx1201 exact bounded batched indexer top-K. Kept as a separate source and
/// symbol so admitting RDNA4 cannot alter gfx1151's certified code object.
pub const INDEXER_TOP_K_BATCHED_BOUNDED_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_batched_bounded.gfx1201.hip");

/// HC 4-stream residual mix — BATCHED (Phase A5, 2026-05-18). Twin of
/// HC_MIX_4STREAM_SRC; batch dim parallelizes cleanly across blockIdx.z.
pub const HC_MIX_4STREAM_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hc_mix_4stream_batched.hip");

/// HC input mapping — BATCHED (Phase A5, 2026-05-18). Twin of
/// HC_INPUT_MAP_SRC; batch dim parallelizes cleanly across blockIdx.y.
pub const HC_INPUT_MAP_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hc_input_map_batched.hip");

/// Broadcast batched embedding rows into the 4 HC residual streams
/// (Phase B2, 2026-05-18). Replaces the per-token loop of memcpys.
pub const HC_STREAMS_INIT_FROM_EMBED_BATCHED_SRC: &str =
    include_str!("../../../kernels/src/hc_streams_init_from_embed_batched.hip");

/// Debug-instrumented twin of deepseek4_attn_swa_batched. Same compute; also
/// writes max_score / sum_exp per (h, b) into per-block scratch buffers
/// for bisecting non-determinism inside the kernel.
pub const V4F_ATTN_SWA_BATCHED_DEBUG_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_batched_debug.hip");

/// Register-tiled F32 batched GEMM (Phase B2 perf, 2026-05-18).
/// Each block holds BATCH_TILE=8 accumulators in registers and reuses
/// each loaded weight tile across them — amortizes weight bandwidth.
/// Replaces gemm_f32_batched for prefill paths.
pub const GEMM_F32_REGISTER_TILED_SRC: &str =
    include_str!("../../../kernels/src/gemm_f32_register_tiled.hip");

/// Atomic-free MQ2-Lloyd K4 MoE down kernel — writes [N × K_TOP × M]
/// f32 with no atomicAdd contention. Pair with `moe_down_combine_k8_batched`
/// to fold K_TOP outputs into x_residual deterministically. Required by
/// the DeepSeek V4 MTP spec-decode draft/verify path: with the standard
/// non-deterministic K4 MoE-down, atomicAdd FP-reduction-order variance
/// between draft and verify passes makes top1 drift, causing spurious
/// rejection (~38% accept). Deterministic path is bit-reproducible →
/// matches memory-cited 84% accept on K=3.
pub const GEMV_MQ2G256_LLOYD_MOE_DOWN_EXPANDED_K4_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4.hip");

/// ParoQuant Givens rotation: apply learned pairwise rotations + channel scaling
/// to activations in-place. Called before each ParoQ4G128 GEMV.
pub const GIVENS_ROTATE_SRC: &str = include_str!("../../../kernels/src/givens_rotate.hip");

/// Host-side value-identity proof for the gfx1201 E8 decode rewrite.  These
/// helpers feed a byte-exact decode route, so close numerical agreement is not
/// sufficient: every produced coordinate and block scale must have identical
/// bits to the generic implementation.
#[cfg(test)]
mod gfx1201_e8_decode_identity {
    fn generic_cvt_e4m3(b: u8) -> f32 {
        let exp = ((b >> 3) & 0xF) as i32;
        let mant = (b & 0x7) as u32;
        if exp == 0 {
            return 0.015625 * (mant as f32) * 0.125;
        }
        if exp == 0xF && mant == 7 {
            return 448.0;
        }
        f32::from_bits(((exp + 120) as u32) << 23) * (1.0 + (mant as f32) * 0.125)
    }

    fn gfx1201_cvt_e4m3(b: u8) -> f32 {
        let u0 = (b as u32) & 0x7F;
        let u = u0.min(126);
        let normal = f32::from_bits((u + 960) << 20);
        let sub = (u as f32) * 0.001953125;
        if u < 8 {
            sub
        } else {
            normal
        }
    }

    fn generic_decode(idx: u32) -> [f32; 8] {
        let coset = (idx >> 31) & 1;
        let mut e = [0u32; 8];
        let mut sum = 0u32;
        for (i, slot) in e.iter_mut().take(7).enumerate() {
            *slot = (idx >> (4 * i as u32)) & 0xF;
            sum = sum.wrapping_add(*slot);
        }
        let p7 = ((idx >> 28) & 0x7) << 1;
        e[7] = p7 | (sum.wrapping_add(p7) & 1);
        let mut out = [0.0; 8];
        for i in 0..8 {
            let c = (e[i] as i32 - 7) as f32;
            out[i] = if coset != 0 { c + 0.5 } else { c };
        }
        out
    }

    fn gfx1201_decode(idx: u32) -> [f32; 8] {
        let lo = (idx & 0x0F0F_0F0F).to_le_bytes();
        let hi = ((idx >> 4) & 0x0F0F_0F0F).to_le_bytes();
        let e7 = ((idx >> 27) & 14) | ((idx & 0x0111_1111).count_ones() & 1);
        let off = if idx & 0x8000_0000 != 0 { -6.5 } else { -7.0 };
        [
            lo[0] as f32 + off,
            hi[0] as f32 + off,
            lo[1] as f32 + off,
            hi[1] as f32 + off,
            lo[2] as f32 + off,
            hi[2] as f32 + off,
            lo[3] as f32 + off,
            e7 as f32 + off,
        ]
    }

    #[test]
    fn block_scale_is_value_identical() {
        for b in 0u16..=255 {
            assert_eq!(
                generic_cvt_e4m3(b as u8).to_bits(),
                gfx1201_cvt_e4m3(b as u8).to_bits(),
                "E4M3 scale byte 0x{b:02x}"
            );
        }
    }

    #[test]
    fn lattice_coordinates_are_value_identical() {
        let mut cases = Vec::new();
        for top in 0u32..16 {
            for i in 0..7u32 {
                for v in 0u32..16 {
                    cases.push((top << 28) | (v << (4 * i)));
                }
            }
            cases.extend([
                top << 28,
                (top << 28) | 0x0FFF_FFFF,
                (top << 28) | 0x0111_1111,
            ]);
        }
        let mut state = 0x1234_5678u32;
        for _ in 0..200_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            cases.push(state);
        }
        for idx in cases {
            let generic = generic_decode(idx);
            let candidate = gfx1201_decode(idx);
            for coord in 0..8 {
                assert_eq!(
                    generic[coord].to_bits(),
                    candidate[coord].to_bits(),
                    "E8 codeword 0x{idx:08x} coordinate {coord}"
                );
            }
        }
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::arch_caps::ArchCaps;
    use crate::feature_flags::FeatureFlags;
    use std::sync::Arc;

    const ALL_ARCHS: &[&str] = &[
        "gfx906", "gfx908", "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032",
        "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
        "gfx1201", "gfx940", "gfx941", "gfx942",
    ];

    /// WMMA-capable archs (RDNA3 + RDNA4).
    const WMMA_ARCHS: &[&str] = &[
        "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
        "gfx1201",
    ];

    fn make_caps(arch: &str) -> ArchCaps {
        ArchCaps::new(arch, Arc::new(FeatureFlags::for_test(arch)))
    }

    #[test]
    fn gfx1100_q8_decode_fusions_are_translation_unit_isolated() {
        assert!(!KV_CACHE_WRITE_Q8_0_SRC.contains("kv_cache_write_q8_0_pair"));
        assert!(KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC.contains("kv_cache_write_q8_0_pair"));
        assert!(!ATTENTION_FLASH_Q8_0_REDUCE_SRC
            .contains("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"));
        assert!(ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1100_SRC
            .contains("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"));
        assert!(ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1151_SRC.starts_with(
            "#define HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151"
        ));
        assert!(GATED_NORM_MQ_ROTATE_GFX1151_SRC.starts_with(
            "#define HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL gated_norm_mq_rotate_gfx1151"
        ));
        assert!(MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1151_SRC.starts_with(
            "#define HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151"
        ));
        #[cfg(feature = "deltanet")]
        assert!(QWEN35_FA_PREP_GFX1151_SRC
            .starts_with("#define HIPFIRE_QWEN35_FA_PREP_KERNEL qwen35_fa_prep_gfx1151"));
    }

    #[test]
    fn qwen2_bias_symbols_are_isolated_from_existing_qkv_modules() {
        assert!(!FUSED_QKV_HFQ4G256_SRC.contains("#define HIPFIRE_QKV_WITH_BIAS 1"));
        assert!(FUSED_QKV_HFQ4G256_QWEN2_BIAS_SRC.contains("fused_qkv_hfq4g256_qwen2_bias"));
        assert!(FUSED_QKV_Q4K_QWEN2_BIAS_SRC.contains("fused_qkv_q4k_qwen2_bias"));
        assert!(FUSED_QKV_Q8_0_QWEN2_BIAS_SRC.contains("fused_qkv_q8_0_qwen2_bias"));

        for arch in ["gfx1030", "gfx1100", "gfx1151", "gfx1201"] {
            let caps = make_caps(arch);
            let (_, old_mq4) = fused_qkv_mq4g256_lloyd_for_arch(&caps, false);
            let (mq4_src, bias_mq4) = fused_qkv_mq4g256_lloyd_qwen2_bias_for_arch(&caps, false);
            assert_ne!(old_mq4, bias_mq4);
            assert!(mq4_src.contains("#define HIPFIRE_QKV_WITH_BIAS 1"));

            let (_, old_mq3) = fused_qkv_mq3g256_lloyd_for_arch(&caps, false);
            let (mq3_src, bias_mq3) = fused_qkv_mq3g256_lloyd_qwen2_bias_for_arch(&caps, false);
            assert_ne!(old_mq3, bias_mq3);
            assert!(mq3_src.contains("#define HIPFIRE_QKV_WITH_BIAS 1"));
        }
    }

    // ── MQ4G256-Lloyd family ─────────────────────────────────────

    mod mq4lloyd {
        use super::*;

        #[test]
        fn gemm_residual_wmma_selects_correct_variant() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                match arch {
                    "gfx1200" | "gfx1201" => {
                        let (src, mod_name) = gemm_mq4g256_lloyd_residual_wmma_for_arch(&caps);
                        assert!(
                            mod_name.contains("rdna4"),
                            "{arch}: expected rdna4, got {mod_name}"
                        );
                    }
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        let (src, mod_name) = gemm_mq4g256_lloyd_residual_wmma_for_arch(&caps);
                        assert!(
                            mod_name.contains("rdna3"),
                            "{arch}: expected rdna3, got {mod_name}"
                        );
                    }
                    _ => {
                        let result = std::panic::catch_unwind(|| {
                            gemm_mq4g256_lloyd_residual_wmma_for_arch(&make_caps(arch))
                        });
                        assert!(result.is_err(), "{arch}: should panic for unsupported arch");
                    }
                }
            }
        }

        #[test]
        fn gemm_residual_wmma_mb4_gfx11_only() {
            for &arch in WMMA_ARCHS {
                let caps = make_caps(arch);
                let result = std::panic::catch_unwind(|| {
                    gemm_mq4g256_lloyd_residual_wmma_mb4_for_arch(&caps)
                });
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(
                            mod_name.contains("mb4"),
                            "expected mb4 variant, got {mod_name}"
                        );
                    }
                    "gfx1103" | "gfx1150" | "gfx1152" | "gfx1200" | "gfx1201" => {
                        assert!(result.is_err(), "{arch} should panic (no mb4 variant)");
                    }
                    _ => {}
                }
            }
        }

        #[test]
        fn gemm_qkvza_wmma_gfx11_gfx12() {
            for &arch in WMMA_ARCHS {
                let caps = make_caps(arch);
                let result =
                    std::panic::catch_unwind(|| gemm_qkvza_mq4g256_lloyd_wmma_for_arch(&caps));
                match arch {
                    "gfx1200" | "gfx1201" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(mod_name.contains("rdna4"), "got {mod_name}");
                    }
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        let (_, mod_name) = result.unwrap();
                        if arch == "gfx1151" {
                            assert!(mod_name.contains("k4"), "got {mod_name}");
                        } else {
                            assert!(mod_name.contains("rdna3"), "got {mod_name}");
                        }
                    }
                    _ => assert!(result.is_err(), "{arch} should panic"),
                }
            }
        }

        #[test]
        fn gemm_qkv_wmma_gfx11_gfx12() {
            for &arch in WMMA_ARCHS {
                let caps = make_caps(arch);
                let result =
                    std::panic::catch_unwind(|| gemm_qkv_mq4g256_lloyd_wmma_for_arch(&caps));
                match arch {
                    "gfx1200" | "gfx1201" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(mod_name.contains("rdna4"), "got {mod_name}");
                    }
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(
                            mod_name.contains("rdna3") || mod_name.contains("k4"),
                            "got {mod_name}"
                        );
                    }
                    _ => assert!(result.is_err(), "{arch} should panic"),
                }
            }
        }

        #[test]
        fn gemm_gate_up_wmma_gfx11_gfx12() {
            for &arch in WMMA_ARCHS {
                let caps = make_caps(arch);
                let result =
                    std::panic::catch_unwind(|| gemm_gate_up_mq4g256_lloyd_wmma_for_arch(&caps));
                match arch {
                    "gfx1200" | "gfx1201" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(mod_name.contains("rdna4"), "got {mod_name}");
                    }
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(
                            mod_name.contains("rdna3") || mod_name.contains("k4"),
                            "got {mod_name}"
                        );
                    }
                    _ => assert!(result.is_err(), "{arch} should panic"),
                }
            }
        }

        #[test]
        fn gemv_baseline_or_k4() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_mq4g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "K4 variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => {
                        assert!(
                            mod_name.ends_with("lloyd"),
                            "baseline expected on {arch}, got {mod_name}"
                        );
                    }
                }
            }
        }

        #[test]
        fn gemv_force_baseline_overrides_k4() {
            for &arch in &["gfx1100", "gfx1151"] {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_mq4g256_lloyd_for_arch(&caps, true);
                assert!(
                    mod_name.ends_with("lloyd"),
                    "force_baseline should return lloyd, got {mod_name}"
                );
            }
        }

        #[test]
        fn gemv_residual_variant() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_mq4g256_lloyd_residual_for_arch(&caps, false);
                assert!(
                    mod_name.contains("residual"),
                    "expected residual, got {mod_name}"
                );
            }
        }

        #[test]
        fn fused_gate_up_arch_dispatch() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = fused_gate_up_mq4g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "fast variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => assert!(
                        mod_name.ends_with("lloyd"),
                        "baseline expected on {arch}, got {mod_name}"
                    ),
                }
            }
        }

        #[test]
        fn fused_qkvza_arch_dispatch() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = fused_qkvza_mq4g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "fast variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => assert!(
                        mod_name.ends_with("lloyd"),
                        "baseline expected on {arch}, got {mod_name}"
                    ),
                }
            }
        }

        #[test]
        fn fused_qkv_arch_dispatch() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = fused_qkv_mq4g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "fast variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => assert!(
                        mod_name.ends_with("lloyd"),
                        "baseline expected on {arch}, got {mod_name}"
                    ),
                }
            }
        }
    }

    // ── MQ3G256-Lloyd family ─────────────────────────────────────

    mod mq3lloyd {
        use super::*;

        #[test]
        fn gemm_residual_wmma_selects_correct_variant() {
            for &arch in WMMA_ARCHS {
                let caps = make_caps(arch);
                let result =
                    std::panic::catch_unwind(|| gemm_mq3g256_lloyd_residual_wmma_for_arch(&caps));
                match arch {
                    "gfx1200" | "gfx1201" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(mod_name.contains("rdna4"), "got {mod_name}");
                    }
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => {
                        let (_, mod_name) = result.unwrap();
                        assert!(mod_name.contains("rdna3"), "got {mod_name}");
                    }
                    _ => assert!(result.is_err(), "{arch} should panic"),
                }
            }
        }

        #[test]
        fn gemv_baseline_or_k4() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_mq3g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "K4 variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => {
                        assert!(
                            mod_name.ends_with("lloyd"),
                            "baseline expected on {arch}, got {mod_name}"
                        );
                    }
                }
            }
        }

        #[test]
        fn gemv_force_baseline_overrides_k4() {
            for &arch in &["gfx1100", "gfx1151"] {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_mq3g256_lloyd_for_arch(&caps, true);
                assert!(
                    mod_name.ends_with("lloyd"),
                    "force_baseline should return lloyd, got {mod_name}"
                );
            }
        }

        #[test]
        fn fused_gate_up_arch_dispatch() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = fused_gate_up_mq3g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(mod_name.contains("rdna3"), "got {mod_name}");
                    }
                    _ => assert!(mod_name.ends_with("lloyd"), "got {mod_name}"),
                }
            }
        }

        #[test]
        fn fused_qkvza_arch_dispatch() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = fused_qkvza_mq3g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(mod_name.contains("rdna3"), "got {mod_name}");
                    }
                    _ => assert!(mod_name.ends_with("lloyd"), "got {mod_name}"),
                }
            }
        }

        #[test]
        fn fused_qkv_arch_dispatch() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = fused_qkv_mq3g256_lloyd_for_arch(&caps, false);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" => {
                        assert!(mod_name.contains("rdna3"), "got {mod_name}");
                    }
                    _ => assert!(mod_name.ends_with("lloyd"), "got {mod_name}"),
                }
            }
        }
    }

    // ── Legacy flat-format family ────────────────────────────────

    mod flat {
        use super::*;

        #[test]
        fn gemv_hfq4g256_selects_correct_variant() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_hfq4g256_for_arch(&caps, None);
                match arch {
                    "gfx1030" | "gfx1031" => {
                        assert!(
                            mod_name.contains("rdna2"),
                            "RDNA2 variant expected on {arch}, got {mod_name}"
                        );
                    }
                    "gfx1100" | "gfx1101" | "gfx1102" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "RDNA3 K4 variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => {
                        assert_eq!(
                            mod_name, "gemv_hfq4g256",
                            "baseline expected on {arch}, got {mod_name}"
                        );
                    }
                }
            }
        }

        #[test]
        fn gemv_hfq4g256_rdna2_variant_selection() {
            for &variant in &[1u32, 2, 3, 4, 5] {
                let caps = make_caps("gfx1030");
                let (_, mod_name) = gemv_hfq4g256_for_arch(&caps, Some(variant));
                assert!(
                    mod_name.contains(&format!("rdna2v{}", variant)),
                    "variant {variant} expected rdna2v{variant}, got {mod_name}"
                );
            }
        }

        #[test]
        fn gemv_hfq4g256_default_rdna2_variant() {
            let caps = make_caps("gfx1030");
            // None should fall through to the baseline (variant 1)
            let (_, mod_name) = gemv_hfq4g256_for_arch(&caps, None);
            assert!(
                mod_name.contains("rdna2"),
                "expected RDNA2 variant, got {mod_name}"
            );
        }

        #[test]
        fn gemv_hfp4g32_selects_correct_variant() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_hfp4g32_for_arch(&caps);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "RDNA3 variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => {
                        assert_eq!(
                            mod_name, "gemv_hfp4g32",
                            "baseline expected on {arch}, got {mod_name}"
                        );
                    }
                }
            }
        }

        #[test]
        fn gemv_hfq4g256_residual_selects_correct_variant() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_hfq4g256_residual_for_arch(&caps);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "residual rdna3 expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => {
                        assert_eq!(
                            mod_name, "gemv_hfq4g256_residual",
                            "residual baseline on {arch}, got {mod_name}"
                        );
                    }
                }
            }
        }

        #[test]
        fn gemv_hfq3g256_selects_correct_variant() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_hfq3g256_for_arch(&caps);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "RDNA3 variant expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => {
                        assert_eq!(
                            mod_name, "gemv_hfq3g256",
                            "baseline expected on {arch}, got {mod_name}"
                        );
                    }
                }
            }
        }

        #[test]
        fn gemv_hfq3g256_residual_selects_correct_variant() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let (_, mod_name) = gemv_hfq3g256_residual_for_arch(&caps);
                match arch {
                    "gfx1100" | "gfx1101" | "gfx1102" => {
                        assert!(
                            mod_name.contains("rdna3"),
                            "residual rdna3 expected on {arch}, got {mod_name}"
                        );
                    }
                    _ => {
                        assert_eq!(
                            mod_name, "gemv_hfq3g256_residual",
                            "residual baseline on {arch}, got {mod_name}"
                        );
                    }
                }
            }
        }
    }

    // ── Module name invariants ───────────────────────────────────

    mod invariants {
        use super::*;

        #[test]
        fn module_names_are_valid_identifiers() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gemv_hfq4g256_for_arch(&caps, None)
                }));
                if let Ok((_src, mod_name)) = r {
                    assert!(!mod_name.contains(' '), "gemv_hfq4g256 on {arch}: spaces");
                    assert!(!mod_name.is_empty(), "gemv_hfq4g256 on {arch}: empty name");
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gemv_hfp4g32_for_arch(&caps)
                }));
                if let Ok((_src, mod_name)) = r {
                    assert!(!mod_name.contains(' '), "gemv_hfp4g32 on {arch}: spaces");
                    assert!(!mod_name.is_empty(), "gemv_hfp4g32 on {arch}: empty name");
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gemv_hfq4g256_residual_for_arch(&caps)
                }));
                if let Ok((_src, mod_name)) = r {
                    assert!(
                        !mod_name.contains(' '),
                        "gemv_hfq4g256_residual on {arch}: spaces"
                    );
                    assert!(
                        !mod_name.is_empty(),
                        "gemv_hfq4g256_residual on {arch}: empty name"
                    );
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gemv_hfq3g256_for_arch(&caps)
                }));
                if let Ok((_src, mod_name)) = r {
                    assert!(!mod_name.contains(' '), "gemv_hfq3g256 on {arch}: spaces");
                    assert!(!mod_name.is_empty(), "gemv_hfq3g256 on {arch}: empty name");
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gemv_hfq3g256_residual_for_arch(&caps)
                }));
                if let Ok((_src, mod_name)) = r {
                    assert!(
                        !mod_name.contains(' '),
                        "gemv_hfq3g256_residual on {arch}: spaces"
                    );
                    assert!(
                        !mod_name.is_empty(),
                        "gemv_hfq3g256_residual on {arch}: empty name"
                    );
                }
            }
        }

        #[test]
        fn source_is_non_empty() {
            for &arch in ALL_ARCHS {
                let caps = make_caps(arch);
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (gemv_hfq4g256_for_arch(&caps, None), "gemv_hfq4g256")
                }));
                if let Ok(((src, _), _name)) = r {
                    assert!(!src.is_empty(), "gemv_hfq4g256 on {arch}: empty source");
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (gemv_hfp4g32_for_arch(&caps), "gemv_hfp4g32")
                }));
                if let Ok(((src, _), _name)) = r {
                    assert!(!src.is_empty(), "gemv_hfp4g32 on {arch}: empty source");
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (
                        gemv_hfq4g256_residual_for_arch(&caps),
                        "gemv_hfq4g256_residual",
                    )
                }));
                if let Ok(((src, _), _name)) = r {
                    assert!(
                        !src.is_empty(),
                        "gemv_hfq4g256_residual on {arch}: empty source"
                    );
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (gemv_hfq3g256_for_arch(&caps), "gemv_hfq3g256")
                }));
                if let Ok(((src, _), _name)) = r {
                    assert!(!src.is_empty(), "gemv_hfq3g256 on {arch}: empty source");
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (
                        gemv_hfq3g256_residual_for_arch(&caps),
                        "gemv_hfq3g256_residual",
                    )
                }));
                if let Ok(((src, _), _name)) = r {
                    assert!(
                        !src.is_empty(),
                        "gemv_hfq3g256_residual on {arch}: empty source"
                    );
                }
            }
        }
    }
}
