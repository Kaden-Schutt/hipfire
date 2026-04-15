//! Built-in HIP kernel sources for inference operations.

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


/// HFQ4-G128 batched GEMM: same tiled approach as G256 but 72 bytes/group, 4 weights/thread.
pub const GEMM_HFQ4G128_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g128.hip");


/// HFQ2-G256: flat 2-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][64B data] = 72 bytes per 256 weights (0.28 B/w).
pub const GEMV_HFQ2G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq2g256.hip");


/// HFQ8-G256: flat 8-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][256B data] = 264 bytes per 256 weights (1.03 B/w).
pub const GEMV_HFQ8G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq8g256.hip");


/// HFQ6-G256: flat 6-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][192B data] = 200 bytes per 256 weights (0.78 B/w).
/// Packing: 4 weights per 3 bytes (24 bits = 4×6 bits).
pub const GEMV_HFQ6G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq6g256.hip");


/// HFQ3-G256: flat 3-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][96B data] = 104 bytes per 256 weights (0.41 B/w).
/// Packing: 8 weights per 3 bytes (24 bits = 8×3 bits).
pub const GEMV_HFQ3G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq3g256.hip");
pub const GEMV_HFQ3G128_SRC: &str = include_str!("../../../kernels/src/gemv_hfq3g128.hip");
pub const GEMV_MQ4G256_SRC: &str = include_str!("../../../kernels/src/gemv_mq4g256.hip");
pub const GEMV_MQ8G256_SRC: &str = include_str!("../../../kernels/src/gemv_mq8g256.hip");
/// MQ6-G256 GEMV: FWHT-rotated HFQ6 (6-bit, 200 B/group). Uses pre-rotated x.
pub const GEMV_MQ6G256_SRC: &str = include_str!("../../../kernels/src/gemv_mq6g256.hip");
pub const FUSED_RMSNORM_MQ_ROTATE_SRC: &str = include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate.hip");
pub const FUSED_SILU_MUL_MQ_ROTATE_SRC: &str = include_str!("../../../kernels/src/fused_silu_mul_mq_rotate.hip");




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

// ── RDNA2 (gfx1030) HFQ4-G256 variants ──
// 5 kernel variants exploring the occupancy/unroll/cache tradeoff space.
// Select via HIPFIRE_RDNA2_VARIANT=N env var (default: 1).
// v1: baseline-rdna2 — launch_bounds(32,16), 2x unroll, ~64 VGPRs
// v2: high-occupancy — launch_bounds(32,20), 2x unroll, ~51 VGPRs (scoped vars)
// v3: wide-unroll    — launch_bounds(32,12), 4x unroll, ~85 VGPRs
// v4: dp4a-packed    — launch_bounds(32,16), dp4a intrinsics, factored scale/zero
// v5: cache-aggressive — launch_bounds(32,16), 2x unroll, packed loads, factored math
pub const GEMV_HFQ4G256_GFX1100_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.gfx1100.hip");
pub const GEMV_HFQ4G256_RESIDUAL_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_residual.hip");
pub const GEMV_HFQ4G256_RESIDUAL_GFX1100_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_residual.gfx1100.hip");

/// HFQ4-G256 GEMV with fused SCALED residual: y[row] += scale * (A[row] · x).
/// Two flavors in one file: `_cpu` takes `scale` by kernarg, `_gpu` reads it
/// from a 1-element device buffer. Used by the MoE FFN accumulator — the
/// routed-expert variant scales by a CPU top-K weight, and the shared-expert
/// variant scales by an on-device sigmoid gate (no D2H sync).
pub const GEMV_HFQ4G256_RESIDUAL_SCALED_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_residual_scaled.hip");

/// MoE fused gate_up GEMV: runs 8 top-K experts' HFQ4-G256 GEMV in one
/// launch. Grid.y is the expert rank (0..7); each block selects its
/// expert's weight base from the W0..W7 kernarg array and runs the
/// standard HFQ4G256 body. Saves 7 launches per MoE layer.
pub const GEMV_HFQ4G256_MOE_GATE_UP_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up.hip");

/// MoE fused down GEMV with scaled residual: 8 experts' weighted
/// contributions accumulate into a single residual buffer via atomicAdd
/// in one kernel launch. Grid.y selects the expert. Saves 7 launches
/// per MoE layer.
pub const GEMV_HFQ4G256_MOE_DOWN_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_moe_down.hip");

/// GPU softmax + top-K + (optional) renormalize for the MoE router.
/// Reads [n_exp] logits, writes [k] indices and [k] weights to device
/// buffers. Eliminates the per-layer D2H sync the CPU-side top-K used
/// to need — required for hipGraph capture of MoE decode.
pub const MOE_SOFTMAX_TOPK_K8_SRC: &str = include_str!("../../../kernels/src/moe_softmax_topk_k8.hip");

/// Index-aware MoE gate_up GEMV — reads expert IDs from a device-side
/// topk_indices buffer and the per-expert weight base from an
/// expert-pointers table. hipGraph-capture-safe replacement for the
/// kernarg-pointer variant.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip");

/// CDNA3 (MI300X / gfx94x) wave64-native counterpart to the indexed
/// gate_up GEMV. Block=[64,1,1] with 2 rows per block (one per warp) —
/// halves the grid count vs the wave32 variant, which otherwise wastes
/// half a wave64 per workgroup. Byte-exact math.
pub const GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC: &str =
    include_str!("../../../kernels/src/gemv_hfq4g256_moe_gate_up_indexed_wave64.hip");

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

// Batched HFQ4-G256 GEMM with fused residual add. Processes N batch elements
// per launch with the same 4-accumulator interleave as the single-row GEMV, so
// output is bitwise identical to calling gemv_hfq4g256_residual N times. Used
// for batched prefill (FFN down + wo projection) where N prompt tokens share
// the same weight matrix.
pub const GEMM_HFQ4G256_RESIDUAL_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual.hip");

// CDNA3 wave64-native batched HFQ4-G256 residual GEMM. 2 rows per block
// (one per warp), halves grid.x. Byte-exact with the wave32 kernel.
pub const GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wave64.hip");

// FP16-packed variant: dequant to __half, v_pk_fma_f16 inner loop, FP32 accumulation.
// 2× throughput over FP32 on all RDNA. Same grid/block layout.
pub const GEMM_HFQ4G256_RESIDUAL_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_fp16.hip");

// WMMA variant: gfx1100+ only. Uses __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
// for 16×16 tiled matrix multiply. Same FP16 X input, FP32 Y output.
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA2_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma2.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_K2_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_k2.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_K2X32_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_k2x32.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_K4_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_k4.hip");
pub const GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_ksplit.hip");
pub const GEMM_MW16_RESIDUAL_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_mw16_residual_wmma.hip");
pub const DEQUANT_HFQ4G256_TO_F16_SRC: &str = include_str!("../../../kernels/src/dequant_hfq4g256_to_f16.hip");
pub const GEMM_GATE_UP_HFQ4G256_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_wmma.hip");
pub const GEMM_QKVZA_HFQ4G256_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_wmma.hip");
pub const GEMM_QKV_HFQ4G256_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq4g256_wmma.hip");

// Batched 4-way fused HFQ4-G256 GEMM (LA preamble: wqkv + wz + w_beta + w_alpha).
// Batched counterpart of fused_qkvza_hfq4g256 — byte-exact vs running that kernel
// N times on the same x[b]. Used for batched prefill of the LA layer projection.
pub const GEMM_QKVZA_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq4g256.hip");
// CDNA3 wave64-native batched 4-way fused LA GEMM. 2 rows per block via
// warp_id, halves grid.x. Byte-exact with wave32 base. Hottest DFlash
// verify kernel on MI300X — targeted first for this port.
pub const GEMM_QKVZA_HFQ4G256_WAVE64_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_wave64.hip");
// FP16 packed variant — RDNA1/2 fast path (no WMMA available).
pub const GEMM_QKVZA_HFQ4G256_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_fp16.hip");
// v_dot2_f32_f16 variant — emits v_dot2_f32_f16 on gfx1011/1012/1030-1032 and gfx11/12.
pub const GEMM_QKVZA_HFQ4G256_DOT2_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq4g256_dot2.hip");

// Batched 3-way fused HFQ4-G256 GEMM (FA preamble: wq + wk + wv).
// Batched counterpart of fused_qkv_hfq4g256 — byte-exact vs running that kernel
// N times on the same x[b]. Used for batched prefill of the FA layer projection.
pub const GEMM_QKV_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq4g256.hip");
// CDNA3 wave64-native batched 3-way fused FA preamble. 2 rows per block.
pub const GEMM_QKV_HFQ4G256_WAVE64_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq4g256_wave64.hip");
// FP16 packed variant — RDNA1/2 fast path (no WMMA available).
pub const GEMM_QKV_HFQ4G256_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq4g256_fp16.hip");
// v_dot2_f32_f16 variant — emits v_dot2_f32_f16 on gfx1011/1012/1030-1032 and gfx11/12.
pub const GEMM_QKV_HFQ4G256_DOT2_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq4g256_dot2.hip");

// Batched 2-way fused HFQ4-G256 GEMM (FFN preamble: w_gate + w_up).
// Batched counterpart of fused_gate_up_hfq4g256 — byte-exact vs running that kernel
// N times on the same x[b]. Used for batched prefill of the FFN gate/up projections.
pub const GEMM_GATE_UP_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq4g256.hip");
// FP16 packed variant — RDNA1/2 fast path (no WMMA available).
pub const GEMM_GATE_UP_HFQ4G256_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_fp16.hip");
// v_dot2_f32_f16 variant — emits v_dot2_f32_f16 on gfx1011/1012/1030-1032 and gfx11/12.
// Does NOT work on gfx1010 (5700 XT) or gfx1013 (BC-250 APU) — lack dot instructions.
pub const GEMM_GATE_UP_HFQ4G256_DOT2_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq4g256_dot2.hip");

// ── HFQ6-G256 batched GEMM (for MQ6 prefill) ──
pub const GEMM_HFQ6G256_RESIDUAL_SRC: &str = include_str!("../../../kernels/src/gemm_hfq6g256_residual.hip");
pub const GEMM_HFQ6G256_RESIDUAL_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_hfq6g256_residual_fp16.hip");
pub const GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC: &str = include_str!("../../../kernels/src/gemm_hfq6g256_residual_wmma_k2.hip");
pub const GEMM_QKVZA_HFQ6G256_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq6g256.hip");
pub const GEMM_QKVZA_HFQ6G256_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_fp16.hip");
pub const GEMM_QKVZA_HFQ6G256_DOT2_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_dot2.hip");
pub const GEMM_QKVZA_HFQ6G256_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_qkvza_hfq6g256_wmma.hip");
pub const GEMM_QKV_HFQ6G256_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq6g256.hip");
pub const GEMM_QKV_HFQ6G256_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq6g256_fp16.hip");
pub const GEMM_QKV_HFQ6G256_DOT2_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq6g256_dot2.hip");
pub const GEMM_QKV_HFQ6G256_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_qkv_hfq6g256_wmma.hip");
pub const GEMM_GATE_UP_HFQ6G256_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq6g256.hip");
pub const GEMM_GATE_UP_HFQ6G256_FP16_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_fp16.hip");
pub const GEMM_GATE_UP_HFQ6G256_DOT2_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_dot2.hip");
pub const GEMM_GATE_UP_HFQ6G256_WMMA_SRC: &str = include_str!("../../../kernels/src/gemm_gate_up_hfq6g256_wmma.hip");

// Multi-row GEMV variants: one warp computes R output rows at a time, sharing
// x register state across rows. Exposes R=2, R=4, R=8 extern "C" entry points
// from one source file. See kernel header for VGPR budget details.
pub const GEMV_HFQ4G256_MULTIROW_GFX1100_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_multirow.gfx1100.hip");
pub const GEMV_HFQ4G256_MULTIROW_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_multirow.hip");
pub const GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_residual_multirow.gfx1100.hip");

// 4-way fused HFQ4-G256 projection for Qwen3.5 DeltaNet LA preamble:
// wqkv + wz + w_beta + w_alpha in a single launch. Same 4x-unroll inner
// loop as gemv_hfq4g256.hip; grid = sum of the four projections' output
// row counts. Works on every RDNA generation — see the kernel header.
pub const FUSED_QKVZA_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/fused_qkvza_hfq4g256.hip");

// CDNA3 (MI300X / gfx94x) wave64-native counterpart: block=[64,1,1] with
// two fused-qkvza rows per block (one per warp). Grid halves from total_m
// to (total_m+1)/2. Byte-exact vs the wave32 base kernel.
pub const FUSED_QKVZA_HFQ4G256_WAVE64_SRC: &str = include_str!("../../../kernels/src/fused_qkvza_hfq4g256_wave64.hip");

// 3-way fused HFQ4-G256 projection for Qwen3.5 FullAttention preamble:
// wq + wk + wv in a single launch. Same 4x-unroll inner loop as the LA
// variant; grid = q_m + k_m + v_m. Cross-arch.
pub const FUSED_QKV_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/fused_qkv_hfq4g256.hip");

// CDNA3 (MI300X / gfx94x) wave64-native 3-way fused preamble — 2 rows per
// block via warp_id, halved grid. Byte-exact with the wave32 base kernel.
pub const FUSED_QKV_HFQ4G256_WAVE64_SRC: &str = include_str!("../../../kernels/src/fused_qkv_hfq4g256_wave64.hip");
// Note: 2-way fused gate+up uses the existing FUSED_GATE_UP_HFQ4G256_SRC
// constant declared further down (kernels/src/fused_gate_up_hfq4g256.hip).
pub const GEMV_HFQ4G256_GFX1030_V1_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v1.hip");
pub const GEMV_HFQ4G256_GFX1030_V2_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v2.hip");
pub const GEMV_HFQ4G256_GFX1030_V3_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v3.hip");
pub const GEMV_HFQ4G256_GFX1030_V4_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v4.hip");
pub const GEMV_HFQ4G256_GFX1030_V5_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.gfx1030.v5.hip");

/// Returns the HFQ4-G256 GEMV kernel source AND module name for the given arch.
/// On gfx1030/gfx1031 (RDNA2), selects variant via HIPFIRE_RDNA2_VARIANT env var.
/// Module name is variant-specific so each variant gets its own precompiled .hsaco blob.
/// The function name inside the .hsaco is always "gemv_hfq4g256" (the extern "C" symbol).
pub fn gemv_hfq4g256_for_arch(arch: &str) -> (&'static str, &'static str) {
    match arch {
        "gfx1030" | "gfx1031" => {
            let variant: u32 = std::env::var("HIPFIRE_RDNA2_VARIANT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            let names = ["", "baseline-rdna2", "high-occupancy", "wide-unroll", "dp4a-packed", "cache-aggressive"];
            let name = names.get(variant as usize).unwrap_or(&"baseline-rdna2");
            eprintln!("  RDNA2 GEMV variant: v{variant} ({name})");
            match variant {
                2 => (GEMV_HFQ4G256_GFX1030_V2_SRC, "gemv_hfq4g256_rdna2v2"),
                3 => (GEMV_HFQ4G256_GFX1030_V3_SRC, "gemv_hfq4g256_rdna2v3"),
                4 => (GEMV_HFQ4G256_GFX1030_V4_SRC, "gemv_hfq4g256_rdna2v4"),
                5 => (GEMV_HFQ4G256_GFX1030_V5_SRC, "gemv_hfq4g256_rdna2v5"),
                _ => (GEMV_HFQ4G256_GFX1030_V1_SRC, "gemv_hfq4g256_rdna2v1"),
            }
        }
        "gfx1100" | "gfx1101" | "gfx1102" => {
            (GEMV_HFQ4G256_GFX1100_SRC, "gemv_hfq4g256_rdna3")
        }
        // RDNA4 variants (existing)
        // "gfx1200" | "gfx1201" => ...,
        _ => (GEMV_HFQ4G256_SRC, "gemv_hfq4g256"), // gfx1010 baseline
    }
}

/// Same arch dispatch as `gemv_hfq4g256_for_arch` but returns the residual
/// variant (y[row] += A[row] · x instead of y[row] = ...). RDNA2 variants
/// fall back to the baseline residual kernel for now.
pub fn gemv_hfq4g256_residual_for_arch(arch: &str) -> (&'static str, &'static str) {
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => {
            (GEMV_HFQ4G256_RESIDUAL_GFX1100_SRC, "gemv_hfq4g256_residual_rdna3")
        }
        _ => (GEMV_HFQ4G256_RESIDUAL_SRC, "gemv_hfq4g256_residual"),
    }
}



/// HFQ2-G128: flat 2-bit with 128-weight groups. Finer granularity than G256.
/// [f32 scale (4B)][f32 zero (4B)][2-bit × 128 (32B)] = 40 bytes per 128 weights (0.3125 B/w).
/// 32 threads × 4 elements = 128 per group. Each thread reads 1 byte.
pub const GEMV_HFQ2G128_SRC: &str = include_str!("../../../kernels/src/gemv_hfq2g128.hip");


/// HFQ4-G256 wide GEMV: 2 rows per block (64 threads = 2 warps).
/// Each warp processes one row independently. Halves grid size.
pub const GEMV_HFQ4G256_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_wide.hip");


/// HFQ4-G256 batched GEMM: y[batch][row] = sum_k(A[row][k] * x[batch][k])
/// Loads weight data ONCE per group, multiplies against BATCH_TILE input vectors.
/// Grid: [M, ceil(batch_size/BATCH_TILE), 1]. Each block handles one row × BATCH_TILE batch elements.
/// x layout: [batch_size × K] row-major. y layout: [batch_size × M] row-major.
/// BATCH_TILE=8 keeps register pressure at ~26 VGPRs for good occupancy on RDNA.
pub const GEMM_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256.hip");
// CDNA3 wave64-native batched HFQ4-G256 GEMM (overwrite). 2 rows per block.
pub const GEMM_HFQ4G256_WAVE64_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_wave64.hip");

/// One-shot dequantize HFQ4-G256 matrix → FP16 row-major. Used when the
/// downstream prefill GEMM path uses rocBLAS MFMA kernels (CDNA3 only —
/// the FP16 shadow is 4× the MQ4 size, so the engine only allocates it on
/// large-VRAM GPUs). Launch grid = (M, K/256, 1), block = (128, 1, 1).
pub const HFQ4G256_DEQUANTIZE_TO_F16_SRC: &str = include_str!("../../../kernels/src/hfq4g256_dequantize_to_f16.hip");


/// Fused QKV Q4_K: three GEMVs in one kernel launch.
/// Grid = (q_m + k_m + v_m) blocks. Each block determines which matrix by blockIdx range.
/// All three projections read the same input x (cached). Saves 2 kernel launches per layer.
pub const FUSED_QKV_Q4K_SRC: &str = include_str!("../../../kernels/src/fused_qkv_q4k.hip");


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


pub const GEMV_Q8_0_SRC: &str = include_str!("../../../kernels/src/gemv_q8_0.hip");

/// Batched Q8_0 GEMM. Same per-row math as gemv_q8_0 but holds MAX_BATCH
/// per-row accumulators in registers, broadcasting each weight load across
/// all batch elements. Saves the (batch_size - 1)× weight re-reads of the
/// serial-GEMV loop for DFlash lm_heads.
pub const GEMM_Q8_0_BATCHED_SRC: &str = include_str!("../../../kernels/src/gemm_q8_0_batched.hip");


/// GEMV Q6_K: matrix-vector multiply with on-the-fly Q6_K dequantization.
/// Q6_K block: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per 256 elements.
pub const GEMV_Q6K_SRC: &str = include_str!("../../../kernels/src/gemv_q6k.hip");


/// RMSNorm: y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
pub const RMSNORM_SRC: &str = include_str!("../../../kernels/src/rmsnorm.hip");

/// TriAttention sidecar calibration: GPU band-statistics accumulator.
/// Replaces the CPU BandAccumulator loop (99% of sidecar cal wall time).
pub const TRIATTN_ACCUMULATE_SRC: &str = include_str!("../../../kernels/src/triattn_accumulate.hip");


/// Element-wise add
pub const ADD_SRC: &str = include_str!("../../../kernels/src/add.hip");


/// Element-wise in-place add: a[i] += b[i]
pub const ADD_INPLACE_SRC: &str = include_str!("../../../kernels/src/add_inplace.hip");


/// Scaled in-place add: y[i] += c * x[i] — one kernel for both
/// CPU-scalar (c via kernarg) and GPU-scalar (c via device buffer)
/// variants. Used in the MoE FFN accumulator to fuse the old
/// (scale_f32 + add_inplace_f32) pair.
pub const SCALED_ADD_INPLACE_SRC: &str = include_str!("../../../kernels/src/scaled_add_inplace.hip");


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


/// Fused Gate+Up HFQ4-G256: two GEMVs in one launch (saves 1 launch per layer).
/// Grid: [gate_m + up_m, 1, 1]. Each block processes one row from gate or up weight.
pub const FUSED_GATE_UP_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/fused_gate_up_hfq4g256.hip");


/// INT8 co-located KV v2: [f16 scale (2B)][padding (2B)][int8 × head_dim] = 132 bytes per head.
/// f16 scale matches Q8_0 but with one block per head. Padding for 4-byte alignment.
pub const KV_CACHE_WRITE_INT8C_F16_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_int8c_f16.hip");


/// Attention with INT8 co-located f16 scale KV.
pub const ATTENTION_INT8C_F16_KV_SRC: &str = include_str!("../../../kernels/src/attention_int8c_f16_kv.hip");


/// INT8 co-located KV: [f32 scale][int8 × head_dim] = 132 bytes per head.
/// Symmetric quantization, no zero point. Dequant: scale * (float)val.
/// Minimized VGPRs: no zero register, no nibble math.
pub const KV_CACHE_WRITE_INT8C_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_int8c.hip");


/// Attention with INT8 co-located KV. Deferred scale multiply, 4×32 unrolled inner loop.
/// Q preloaded into shared memory. Scale applied ONCE per position, not per element.
pub const ATTENTION_INT8C_KV_SRC: &str = include_str!("../../../kernels/src/attention_int8c_kv.hip");


/// HFQ8 KV: FP32 scale+zero per head, contiguous uint8 data. Asymmetric quantization.
/// Scales: [max_seq × n_kv_heads × 2] f32 (scale, zero pairs).
/// Data: [max_seq × n_kv_heads × head_dim] uint8.
pub const KV_CACHE_WRITE_HFQ8_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_hfq8.hip");


/// Attention with HFQ8 KV cache. Flat layout, FP32 scale+zero, contiguous uint8 data.
pub const ATTENTION_HFQ8_KV_SRC: &str = include_str!("../../../kernels/src/attention_hfq8_kv.hip");


/// INT8 KV with separate scale array. Contiguous int8 values, one f32 scale per head.
/// Keys: [max_seq × n_kv_heads × head_dim] int8, Scales: [max_seq × n_kv_heads] f32.
/// Write: one warp per head, find amax via shuffle, quantize 4 elements per thread.
pub const KV_CACHE_WRITE_INT8_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_int8.hip");


/// Attention with INT8 KV (separate scale array). Clean indexed access, no block math.
pub const ATTENTION_INT8_KV_SRC: &str = include_str!("../../../kernels/src/attention_int8_kv.hip");


/// Batched causal attention: all query positions attend to their causal context.
/// Grid: [n_heads, seq_len, 1]. Each block handles one (head, query_position) pair.
/// Q/K/V are FP32: [seq_len × n_heads × head_dim] or [seq_len × n_kv_heads × head_dim].
/// Output: [seq_len × n_heads × head_dim].
/// For prefill: Q/K/V come from batched projections. KV also written to cache.
pub const ATTENTION_CAUSAL_BATCHED_SRC: &str = include_str!("../../../kernels/src/attention_causal_batched.hip");


/// Batched Q8_0 KV cache write: quantize multiple positions at once.
/// src: [batch_size × kv_dim] FP32. positions: [batch_size] int32.
/// Grid: [total_blocks × batch_size]. Each block handles one Q8_0 group for one position.
pub const KV_CACHE_WRITE_Q8_0_BATCHED_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q8_0_batched.hip");


/// Quantize KV vector to Q8_0 format (same as GGML Q8_0 / existing GEMV kernels).
/// Block: [f16 scale (2B)][int8 × 32 (32B)] = 34 bytes per 32 elements.
/// head_dim=128 → 4 blocks × 34 = 136 bytes per head.
/// Layout: [max_seq × n_kv_heads × blocks_per_head × 34].
pub const KV_CACHE_WRITE_Q8_0_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q8_0.hip");


/// Attention with Q8_0 quantized KV cache — same format as GGML Q8_0.
/// K and V caches stored as [max_seq × n_kv_heads × blocks_per_head × 34].
pub const ATTENTION_Q8_0_KV_SRC: &str = include_str!("../../../kernels/src/attention_q8_0_kv.hip");

/// Batched counterpart of ATTENTION_Q8_0_KV_SRC. Processes N queries in
/// one launch with per-row causal windows from a positions[] array.
pub const ATTENTION_Q8_0_KV_BATCHED_SRC: &str = include_str!("../../../kernels/src/attention_q8_0_kv_batched.hip");

/// Phase-timed variant of ATTENTION_Q8_0_KV_SRC. Functionally equivalent
/// to the baseline kernel but instrumented with wall_clock64() around each
/// of the 3 internal phases (QK^T, softmax, V-weighted-sum). Writes per-head
/// cycle counts into an extra output buffer of length [n_heads * 3]. For
/// profiling/diagnostic use only.
pub const ATTENTION_Q8_0_KV_TIMED_SRC: &str = include_str!("../../../kernels/src/attention_q8_0_kv_timed.hip");

/// Flash attention tile kernel — zero LDS, online softmax, 32-thread WAVE32.
/// Grid: [n_heads, n_tiles]. Each block fuses QK-dot + softmax + V-accumulate
/// for its tile of positions, writing partials to global memory.
pub const ATTENTION_FLASH_Q8_0_TILE_SRC: &str = include_str!("../../../kernels/src/attention_flash_q8_0_tile.hip");

/// Flash attention reduce kernel — combines tile partials via online softmax
/// correction. Grid: [n_heads]. Reads per-tile {max, sum, out[head_dim]},
/// combines across tiles, normalizes, writes final output.
pub const ATTENTION_FLASH_Q8_0_REDUCE_SRC: &str = include_str!("../../../kernels/src/attention_flash_q8_0_reduce.hip");

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
pub const KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_asym_k_givens4.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_asym_k_givens3.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_asym_k_givens2.hip");
pub const ATTENTION_FLASH_ASYM4_TILE_SRC: &str = include_str!("../../../kernels/src/attention_flash_asym4_tile.hip");
pub const ATTENTION_FLASH_ASYM3_TILE_SRC: &str = include_str!("../../../kernels/src/attention_flash_asym3_tile.hip");
pub const ATTENTION_FLASH_ASYM2_TILE_SRC: &str = include_str!("../../../kernels/src/attention_flash_asym2_tile.hip");

// asym batched prefill variants: K rotated + V Q8 in one launch for N positions.
pub const KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_asym_k_givens4_batched.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_asym_k_givens3_batched.hip");
pub const KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_asym_k_givens2_batched.hip");
pub const ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC: &str = include_str!("../../../kernels/src/attention_flash_asym4_tile_batched.hip");
pub const ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC: &str = include_str!("../../../kernels/src/attention_flash_asym3_tile_batched.hip");
pub const ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC: &str = include_str!("../../../kernels/src/attention_flash_asym2_tile_batched.hip");
pub const ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC: &str = include_str!("../../../kernels/src/attention_flash_asym_reduce_batched.hip");

/// TriAttention scoring on Q8 post-RoPE K cache (arXiv:2604.04921).
pub const TRIATTN_SCORE_Q8_SRC: &str = include_str!("../../../kernels/src/triattn_score_q8.hip");

/// TriAttention scoring on asym3 (Givens-rotated 3-bit) K cache.
pub const TRIATTN_SCORE_ASYM3_SRC: &str = include_str!("../../../kernels/src/triattn_score_asym3.hip");

/// TriAttention scoring on asym4 (Givens-rotated 4-bit) K cache.
pub const TRIATTN_SCORE_ASYM4_SRC: &str = include_str!("../../../kernels/src/triattn_score_asym4.hip");

/// TriAttention scoring on asym2 (Givens-rotated 2-bit) K cache.
pub const TRIATTN_SCORE_ASYM2_SRC: &str = include_str!("../../../kernels/src/triattn_score_asym2.hip");

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
pub const KV_CACHE_WRITE_HFQ4_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_hfq4.hip");


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
pub const FUSED_QK_L2_NORM_SCALE_SRC: &str = include_str!("../../../kernels/src/fused_qk_l2_norm_scale.hip");

/// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Two back-to-back
/// scalar ops in the DeltaNet preamble merged into one launch.
#[cfg(feature = "deltanet")]
pub const FUSED_SIGMOID_ALPHA_GATE_SRC: &str = include_str!("../../../kernels/src/fused_sigmoid_alpha_gate.hip");

/// Fused sigmoid(gate) * x — the FA attention epilogue that used to be
/// `sigmoid_f32(gate)` + `mul_f32(attn_out, gate, attn_out)`.
pub const SIGMOID_MUL_SRC: &str = include_str!("../../../kernels/src/sigmoid_mul.hip");

/// Top-K=128 extraction over a logits vector. Lets the host sampler work
/// on a 1 KB GPU-side candidate set instead of DtoH'ing the full 600 KB
/// logits array. See kernel header for bit-exactness reasoning.
pub const TOPK_LOGITS_SRC: &str = include_str!("../../../kernels/src/topk_logits.hip");
pub const TOPK_LOGSUMEXP_BATCHED_SRC: &str = include_str!("../../../kernels/src/topk_logsumexp_batched.hip");


/// Partial interleaved RoPE: rotate only first n_rot dims, pairs are adjacent (d0,d1),(d2,d3),...
/// Dims >= n_rot pass through unchanged.
/// Grid: [n_rot/2]. Block: [1]. Each thread handles one rotation pair.
#[cfg(feature = "deltanet")]
pub const ROPE_PARTIAL_INTERLEAVED_SRC: &str = include_str!("../../../kernels/src/rope_partial_interleaved.hip");

/// Batched partial-interleaved RoPE — per-row positions read from a
/// positions[] array. Used by the batched prefill FA path.
#[cfg(feature = "deltanet")]
pub const ROPE_PARTIAL_INTERLEAVED_BATCHED_SRC: &str = include_str!("../../../kernels/src/rope_partial_interleaved_batched.hip");


/// 1D causal depthwise convolution (kernel_size=4) with persistent ring buffer state.
/// For decode: one token at a time. conv_state: [n_channels × 3] ring buffer.
/// out[c] = w[0]*x[c] + w[1]*state[c][0] + w[2]*state[c][1] + w[3]*state[c][2]
/// Then shift state: state = [x, state[0], state[1]].
#[cfg(feature = "deltanet")]
pub const CONV1D_DECODE_SRC: &str = include_str!("../../../kernels/src/conv1d_decode.hip");


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
pub const GATED_DELTA_NET_Q8_SRC: &str = include_str!("../../../kernels/src/gated_delta_net_q8.hip");

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
pub const GATED_DELTA_NET_Q8_TREE_SRC: &str = include_str!("../../../kernels/src/gated_delta_net_q8_tree.hip");


/// GDN recurrence with Q4-quantized S state in VRAM.
/// State layout: unsigned char s_q4[n_heads][HD*HD/2] (nibble-packed) + float s_scales[n_heads*HD].
/// Symmetric 4-bit: values -8..+7, scale = absmax/7. Per-row scale.
/// 8x compression vs FP32 (8KB + 512B scales per head vs 64KB).
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_Q4_SRC: &str = include_str!("../../../kernels/src/gated_delta_net_q4.hip");


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
pub const CONV1D_SILU_SPLIT_TREE_SRC: &str = include_str!("../../../kernels/src/conv1d_silu_split_tree.hip");


/// GPU-side KV cache write using pos from a GPU buffer.
/// Copies kv_dim floats from src to dst at offset pos_buf[0] * kv_dim.
pub const KV_CACHE_WRITE_SRC: &str = include_str!("../../../kernels/src/kv_cache_write.hip");


/// GPU-side top-K + top-P sampling. Eliminates 600KB logits download per token.
/// Single block, 256 threads. Returns token ID + RNG state (8 bytes vs 600KB).
///
/// Phase 1: Parallel max reduction over vocab_size logits.
/// Phase 2: Threshold filter — collect candidates within 30*temp of max (atomic shared counter).
/// Phase 3: Thread 0 softmax + sort + top-p + sample on the small candidate set.
pub const SAMPLE_TOP_P_SRC: &str = include_str!("../../../kernels/src/sample_top_p.hip");


/// GEMV Q4_F16_G64: matrix-vector multiply with on-the-fly Q4_F16 dequantization.
/// Block layout: f16 scale (2B) + f16 min (2B) + uint8 quants[32] (32B) = 36 bytes per 64 elements.
/// Dequant: weight = (_Float16)(nibble) * scale + min — single FP16 FMA on RDNA.
/// Thread tid reads quants[tid], processes both nibbles (elements tid and tid+32).
pub const GEMV_Q4F16_G64_SRC: &str = include_str!("../../../kernels/src/gemv_q4f16_g64.hip");


/// GEMV Q4_F16_G64 wide: 256 threads, element-strided access, shared memory reduction.
/// Matches F32 GEMV's occupancy pattern to test whether occupancy explains the 40% vs 48% gap.
/// Each thread processes elements tid, tid+256, tid+512, ... across the row.
pub const GEMV_Q4F16_G64_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_q4f16_g64_wide.hip");


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
pub const EMBEDDING_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/embedding_hfq4g256.hip");

/// Batched HFQ4-G256 embedding: dequantize N rows in one launch. Reads token ids
/// from a device buffer so the launch is hipGraph-captureable — update the buffer
/// between replays, replay the same graph. Writes into row-major `[N × dim]`.
pub const EMBEDDING_HFQ4G256_BATCHED_SRC: &str = include_str!("../../../kernels/src/embedding_hfq4g256_batched.hip");

/// Batched Q8_0 embedding: same hipGraph-captureable pattern as the HFQ4-G256
/// variant. 27B MQ4 targets ship with Q8_0-quantized embedding tables, so the
/// verify hot path needs this variant to enable graph capture on that model.
pub const EMBEDDING_Q8_BATCHED_SRC: &str = include_str!("../../../kernels/src/embedding_q8_batched.hip");


/// HFQ4-G128 embedding lookup: dequantize one row from HFQ4-G128 table to F32.
pub const EMBEDDING_HFQ4G128_SRC: &str = include_str!("../../../kernels/src/embedding_hfq4g128.hip");


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
pub const CROSS_ENTROPY_LOSS_SRC: &str = include_str!("../../../kernels/src/cross_entropy_loss.hip");


/// GPU max-probability: compute max(softmax(logits)) entirely on GPU.
/// Output: single float = probability of the most likely token.
/// Used for early-exit confidence check (downloads 4 bytes instead of 600KB).
pub const MAX_PROB_SRC: &str = include_str!("../../../kernels/src/max_prob.hip");


/// GPU argmax: find index of maximum value.
pub const ARGMAX_SRC: &str = include_str!("../../../kernels/src/argmax.hip");

/// Batched argmax: one block per row, writes B indices with one kernel launch.
/// Used by DFlash verify to collapse the B × [vocab] logit download to B × 4 bytes.
pub const ARGMAX_BATCHED_SRC: &str = include_str!("../../../kernels/src/argmax_batched.hip");


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


/// Transpose: out[c, r] = in[r, c]. Converts [rows, cols] → [cols, rows].
pub const TRANSPOSE_SRC: &str = include_str!("../../../kernels/src/transpose.hip");


/// Fused ViT self-attention: Q@K^T → softmax → @V, reading QKV from [N, 3*hidden].
/// Grid=[n_heads, N]. Each block computes one (head, query_pos) output row.
pub const VIT_ATTENTION_SRC: &str = include_str!("../../../kernels/src/vit_attention.hip");

/// DFlash draft cross-attention (non-causal, GQA): B queries attend to L
/// keys/values with no causal mask. Grid=[n_heads, B]. See
/// `kernels/src/attention_dflash.hip` for the full contract.
pub const ATTENTION_DFLASH_SRC: &str = include_str!("../../../kernels/src/attention_dflash.hip");


/// Bias-add: X[batch, n] += bias[n] (broadcast over batch dim)
pub const BIAS_ADD_SRC: &str = include_str!("../../../kernels/src/bias_add.hip");




/// Deinterleave: split [Q_h0, Gate_h0, Q_h1, Gate_h1, ...] into separate Q and Gate tensors.
pub const DEINTERLEAVE_SRC: &str = include_str!("../../../kernels/src/deinterleave.hip");

/// Batched deinterleave: same as DEINTERLEAVE but processes N tokens in one launch.
pub const DEINTERLEAVE_BATCHED_SRC: &str = include_str!("../../../kernels/src/deinterleave_batched.hip");

/// Single-token repeat-interleave Q and K key heads up to value heads count.
pub const REPEAT_INTERLEAVE_QK_SRC: &str = include_str!("../../../kernels/src/repeat_interleave_qk.hip");

/// Batched repeat-interleave Q and K key heads up to value heads count.
pub const REPEAT_INTERLEAVE_QK_BATCHED_SRC: &str = include_str!("../../../kernels/src/repeat_interleave_qk_batched.hip");

