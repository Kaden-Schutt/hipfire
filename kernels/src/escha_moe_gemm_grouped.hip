// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Escha-W2 routed-expert GROUPED GEMM (Phase 3) — decode each expert's trellis
// code ONCE per (layer, batch) and multiply it against every token that routed
// to it, instead of once per (token, expert) slot.
//
// # The measurement this exists for
//
// `escha_moe_gemv_native.hip` is SLOT-parallel: `krank = blockIdx.y` is one
// (token, expert) pair, so a batch of `n` tokens at `k = 8` re-reads and
// re-decodes each selected expert's whole code `n*k / n_exp` times over. On a
// warm 256-token prefill chunk of the shipped A3B file (`n_exp = 256`,
// `slots = 2048`), `rocprofv3 --kernel-trace` put the two routed GEMV launches
// at 1 779 ms of a 2 370 ms prefill — 75.1% of all GPU time — and prefill
// ms/token was FLAT from n=128 to n=2048 (4.83 / 4.53 / 4.44), i.e. the batch
// bought nothing at all.
//
// Two things are amortised here, and both mattered:
//
//   * WEIGHT TRAFFIC. Logical expert bytes per (layer, chunk) go from
//     `slots * expert_bytes` to `sum_e ceil(G_e / ROWS) * expert_bytes`, where
//     `G_e` is the number of this chunk's slots that chose expert `e`.
//   * DECODE ALU. `escha_native_cba` is five ops and the window extraction is
//     two loads plus a shift; the slot-parallel kernel pays that per (weight,
//     slot), this one pays it per (weight, ROWS slots).
//
// # Why a block owns CTILES tile columns, not one
//
// The slot-parallel kernel's block owns ONE 16-wide tile column, so every slot
// re-reads its whole `x` row once per tile column — `(M/16) * slots * K * 4`
// bytes, which at the gate_up shape is 1.07 GB per (layer, chunk), the SAME
// order as the expert traffic the grouping removes. Grouping alone would have
// left that term untouched and capped the win at ~1.8x.
//
// A block that owns CTILES ADJACENT tile columns reads `x` once for all of
// them: the `x` term divides by CTILES while the expert term is unchanged
// (there are CTILES-times fewer blocks, each reading CTILES-times more code).
// The decode slot descriptor (`word_a`, `word_b`, `shift`) depends only on
// (row-in-tile, column-in-tile), so the extra tile columns cost only their own
// code loads — no extra slot maths, no extra registers beyond the accumulators.
//
// # Registers
//
// A lane holds `ROWS * 2 * CTILES` f32 accumulators (two output columns per
// warp — `w` and `w + 8` share a decode window). ROWS=8, CTILES=4 is 64, which
// fits with room for the `x` values in flight; the host wrapper picks the
// instantiation and the tuning sweep behind
// `HIPFIRE_ESCHA_GROUPED_TILE` is what chose it.
//
// # No codebook lookup table, still
//
// `escha_native_cba` is five inline ops. 65 536 fp16 entries would be 128 KB
// against gfx1151's 64 KB of LDS, so there is no table here either — same
// constraint, same answer, as `escha_decode_tiles.hip` and
// `escha_moe_gemv_native.hip`.
//
// # K=2 and K=3 stay separate
//
// The two trellis orders extract their 16-bit window from structurally
// different stream geometries. Everything that decides WHICH two words and
// WHICH shift is duplicated per K and resolved at compile time, exactly as in
// `escha_moe_gemv_native.hip`; neither pays a runtime branch.
//
// # DELIBERATE DUPLICATION
//
// `escha_native_cba`, `escha_native_word`, the lane/j inversion and the two
// slot descriptors below are copied VERBATIM from
// `kernels/src/escha_moe_gemv_native.hip`, which in turn reproduces
// `escha_decode_tiles.hip` and `hipfire_quantize::escha_ref`. That chain is
// what G2 and G7 assert bit-exact against the frozen CPU oracle. Sharing the
// source between the kernel under test and the kernel it is tested against
// would make those gates circular. Do not deduplicate.
//
// # Numerics — this kernel is NOT bit-identical to the slot-parallel one for
// # every projection, and that is stated rather than hidden
//
// Per (token, output row) this kernel keeps the slot-parallel NARROW form
// exactly: the same lane -> contraction-index map (`lane` strided by 32), the
// same `bi` order, ONE sequential accumulator, and the same `__shfl_down`
// ladder. So for a projection the slot-parallel path also runs narrow
// (`K > 1536` — the shipped gate_up, K = 2048) the two agree BIT FOR BIT.
//
// For a projection the slot-parallel path runs WIDE (`K <= 1536` — the shipped
// down, K = 512) it does not: that form folds FOUR interleaved accumulators as
// `(acc0+acc1)+(acc2+acc3)`, and reproducing it here would cost 4x the
// accumulator registers (ROWS*2*CTILES*4 = 256 at the shipped tile) and force
// CTILES back to 1, which is the term that had to shrink. The grouped kernel
// therefore sums that projection as one sequential chain. The difference is a
// different PARTITION of the same contraction, not a different contraction:
// both are exact-order f32 sums of the identical 512 products. The measured
// cost is reported by the G4 gate's grouped arm.

// Contraction-loop unroll factor. 1 — i.e. NO unroll — and that is a MEASURED
// choice, not an oversight.
//
// The slot-parallel narrow kernel needs `#pragma unroll 4` because it has one
// FMA chain and one memory round trip of latency to hide per iteration, with
// nothing to hide it with. This kernel already carries ROWS*2*CTILES = 64
// independent chains and CTILES*2 independent code loads per iteration, so the
// ILP is there without unrolling. Unrolling on top of it only widens the live
// range of the in-flight loads: measured on the shipped gate_up shape (drift
// corrected against the untouched slot-parallel control in the same process),
// unroll 2 took r8c4 from 2.38x to 2.00x and left r8c8 unchanged.
//
// Overridable at build time so a future shape can re-derive the choice rather
// than inherit it.
#ifndef ESCHA_GROUPED_UNROLL
#define ESCHA_GROUPED_UNROLL 1
#endif

// ── the codebook, inline ─────────────────────────────────────────────────────
__device__ __forceinline__ __half escha_native_cba(unsigned short state) {
    unsigned int r = ((unsigned int)state * 0xCBAC1FEDu) & 0x8FFF8FFFu;
    r ^= 0x3B603B60u;
    __half lo = __ushort_as_half((unsigned short)(r & 0xFFFFu));
    __half hi = __ushort_as_half((unsigned short)(r >> 16));
    return __hadd(lo, hi);  // fp16 RNE add — matches the reference exactly
}

// ADDRESS SPACE 1, explicitly — `expert_ptrs[...]` is a runtime 64-bit integer,
// so without the cast the compiler emits `flat_load_b32` and pays an aperture
// check the whole inner loop then waits on. See the same typedef in
// `escha_moe_gemv_native.hip`, where it was measured worth 1.6x on its own.
typedef __attribute__((address_space(1))) const unsigned int* EschaCodePtr;

__device__ __forceinline__ unsigned int escha_native_word(EschaCodePtr tile, int i) {
    return tile[i];
}

__device__ __forceinline__ int escha_native_lane_of(int row, int col) {
    return ((col >> 1) & 3) * 8 + (col & 1) * 4 + ((row & 7) >> 1);
}
__device__ __forceinline__ int escha_native_j_of(int row, int col) {
    return ((col >> 3) & 1) * 4 + ((row >> 3) & 1) * 2 + (row & 1);
}

__device__ __forceinline__ void escha_native_slot_k2(
    int row, int col, int* word_a, int* word_b, int* shift) {
    const int lane = escha_native_lane_of(row, col);
    const int j = escha_native_j_of(row, col);
    const int t_off = lane * 8;
    const int i1 = t_off >> 4;
    const int i0 = (i1 + 15) & 15;
    *word_a = i0;
    *word_b = i1;
    *shift = (((~t_off) & 8) << 1) + 2 * (7 - j);
}

__device__ __forceinline__ void escha_native_slot_k3(
    int row, int col, int* word_a, int* word_b, int* shift) {
    const int BITS = 3;
    const int lane = escha_native_lane_of(row, col);
    const int j = escha_native_j_of(row, col);
    const int t_off = lane * 8;
    const int b1 = (t_off + 257) * BITS;
    const int b0 = b1 - 16;
    const int b2 = b1 + BITS * 7;
    const int i0 = b0 >> 5;
    const int i2 = (b2 - 1) >> 5;
    const int s2 = ((i2 + 1) << 5) - b2;
    *word_a = i0 % 24;
    *word_b = i2 % 24;
    *shift = s2 + (j < 4 ? BITS * 4 : 0) + BITS * (3 - (j & 3));
}

template <int TRELLIS_K>
__device__ __forceinline__ void escha_native_slot(
    int row, int col, int* word_a, int* word_b, int* shift) {
    if (TRELLIS_K == 2) escha_native_slot_k2(row, col, word_a, word_b, shift);
    else                escha_native_slot_k3(row, col, word_a, word_b, shift);
}

// ── the grouped body ─────────────────────────────────────────────────────────
//
// Grid: (M / (16*CTILES), n_exp, 1). Block: 256 = eight warps; warp `w` owns
// output columns `w` and `w + 8` of each of the block's CTILES tile columns.
//
// `expert_offsets[e] .. expert_offsets[e+1]` is expert `e`'s half-open range in
// the SORTED slot order, and `sorted_slot_index[p]` is the flat slot at sorted
// position `p`. Both come from `moe_scatter_fused_k8` run with `block_m = 1`,
// i.e. WITHOUT the grouped-WMMA path's BLOCK_M padding: padding would cost real
// `x` reads and real FMAs for rows that are then discarded, and at the shipped
// shape (256 experts, 2048 slots, mean group 8) a BLOCK_M of 8 would have
// inflated the `x` term — the dominant one — by ~40%. Ranging over the exact
// group instead costs one extra `expert_offsets` load and a runtime loop bound.
//
// An expert with no slots this chunk exits before touching `expert_ptrs`, so a
// batch that lights up 30 of 256 experts does 30 experts' work.
template <int TRELLIS_K, int ROWS, int CTILES>
__device__ __forceinline__ void escha_gemm_grouped_body(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ expert_offsets,    // [n_exp + 1] exclusive scan
    const int* __restrict__ sorted_slot_index, // [slots] sorted pos -> flat slot
    const float* __restrict__ x_batch,         // [slots, K]
    float* __restrict__ y_batch,               // [slots, M]
    int M, int K
) {
    const int e = blockIdx.y;
    const int g_start = expert_offsets[e];
    const int g_end = expert_offsets[e + 1];
    if (g_start >= g_end) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // Loop-invariant decode geometry. `in_row` is this lane's row inside a
    // tile, `tile_hi` says which of the iteration's two stacked tiles it falls
    // in — identical to the slot-parallel kernel, because the lane -> weight
    // map is what the bit-exactness contract is about.
    const int tn = M >> 4;
    const int in_row = lane & 15;
    const int tile_hi = lane >> 4;
    int word_a, word_b, shift_lo, shift_hi;
    int word_a2, word_b2;
    escha_native_slot<TRELLIS_K>(in_row, warp, &word_a, &word_b, &shift_lo);
    escha_native_slot<TRELLIS_K>(in_row, warp + 8, &word_a2, &word_b2, &shift_hi);
    // Columns `c` and `c + 8` provably share a decode window (see
    // `escha_moe_gemv_native.hip`). If that ever stops holding the two columns
    // would silently read each other's weights, so poison loudly instead —
    // every escha gate asserts finiteness and nothing else manufactures a NaN.
    if (word_a2 != word_a || word_b2 != word_b) {
        if (lane == 0) {
            for (int p = g_start; p < g_end; p++) {
                const int s = sorted_slot_index[p];
                if (s < 0) continue;
                #pragma unroll
                for (int c = 0; c < CTILES; c++) {
                    const int row = (blockIdx.x * CTILES + c) * 16 + warp;
                    y_batch[(size_t)s * M + row] = __builtin_nanf("");
                    y_batch[(size_t)s * M + row + 8] = __builtin_nanf("");
                }
            }
        }
        return;
    }

    EschaCodePtr code = (EschaCodePtr)(expert_ptrs[e]);
    // Tile `kt` of tile column `nt` starts at `(kt*tn + nt) * 8*TRELLIS_K`
    // DWORDS. This lane's tile in iteration `bi` is `kt = bi*2 + tile_hi`.
    const size_t strip = (size_t)tn * (8 * TRELLIS_K);
    const int nt0 = blockIdx.x * CTILES;
    EschaCodePtr strip0 = code + (size_t)nt0 * (8 * TRELLIS_K) + (size_t)tile_hi * strip;

    const int blocks_per_row = K / 32;

    for (int g = g_start; g < g_end; g += ROWS) {
        // The chunk's rows. Out-of-range lanes carry -1 and are skipped by a
        // BLOCK-UNIFORM branch (every lane sees the same `slot[r]`), so this is
        // a scalar branch, not divergence.
        int slot[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int p = g + r;
            slot[r] = (p < g_end) ? sorted_slot_index[p] : -1;
        }

        float acc[ROWS][2 * CTILES];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            #pragma unroll
            for (int c = 0; c < 2 * CTILES; c++) acc[r][c] = 0.0f;
        }

        #pragma unroll ESCHA_GROUPED_UNROLL
        for (int bi = 0; bi < blocks_per_row; bi++) {
            // Decode this iteration's CTILES weights pairs ONCE, then spend
            // them across every row of the chunk. This is the whole point of
            // the kernel: `ROWS * CTILES * 2` FMAs per `CTILES * 2` decodes.
            float w_lo[CTILES], w_hi[CTILES];
            #pragma unroll
            for (int c = 0; c < CTILES; c++) {
                EschaCodePtr tile =
                    strip0 + (size_t)c * (8 * TRELLIS_K) + (size_t)bi * 2 * strip;
                const unsigned long long merged =
                    ((unsigned long long)escha_native_word(tile, word_a) << 32) |
                    escha_native_word(tile, word_b);
                w_lo[c] = __half2float(escha_native_cba((unsigned short)(merged >> shift_lo)));
                w_hi[c] = __half2float(escha_native_cba((unsigned short)(merged >> shift_hi)));
            }
            const int xo = bi * 32 + lane;
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (slot[r] < 0) continue;
                const float xv = x_batch[(size_t)slot[r] * K + xo];
                #pragma unroll
                for (int c = 0; c < CTILES; c++) {
                    acc[r][2 * c] += w_lo[c] * xv;
                    acc[r][2 * c + 1] += w_hi[c] * xv;
                }
            }
        }

        // The slot-parallel kernel's reduction, verbatim, once per (row,
        // column). Every value lane 0 consumes comes from an in-range lane, so
        // the ladder is a balanced binary tree over the 32 partial sums.
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (slot[r] < 0) continue;
            #pragma unroll
            for (int c = 0; c < CTILES; c++) {
                float sum_lo = acc[r][2 * c];
                float sum_hi = acc[r][2 * c + 1];
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sum_lo += __shfl_down(sum_lo, offset);
                    sum_hi += __shfl_down(sum_hi, offset);
                }
                if (lane == 0) {
                    const int row = (nt0 + c) * 16 + warp;
                    y_batch[(size_t)slot[r] * M + row] = sum_lo;
                    y_batch[(size_t)slot[r] * M + row + 8] = sum_hi;
                }
            }
        }
    }
}

// ── entry points ─────────────────────────────────────────────────────────────
//
// One per (trellis order, tile shape). The host wrapper enforces
// `M % (16*CTILES) == 0` and `K % 32 == 0`, and picks the shape; the set below
// is what the sweep in `bench_escha_grouped_gemm` covers.

#define ESCHA_GROUPED_GEMM(NAME, TK, ROWS, CTILES)                             \
    __launch_bounds__(256)                                                     \
    extern "C" __global__ void NAME(                                           \
        const unsigned long long* __restrict__ expert_ptrs,                    \
        const int* __restrict__ expert_offsets,                                \
        const int* __restrict__ sorted_slot_index,                             \
        const float* __restrict__ x_batch,                                     \
        float* __restrict__ y_batch,                                           \
        int M, int K                                                           \
    ) {                                                                        \
        escha_gemm_grouped_body<TK, ROWS, CTILES>(                             \
            expert_ptrs, expert_offsets, sorted_slot_index, x_batch, y_batch,  \
            M, K);                                                             \
    }

ESCHA_GROUPED_GEMM(escha_gemm_grouped_k2_r4_c2, 2, 4, 2)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k2_r8_c2, 2, 8, 2)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k2_r8_c4, 2, 8, 4)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k2_r16_c2, 2, 16, 2)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k2_r16_c4, 2, 16, 4)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k2_r8_c8, 2, 8, 8)

ESCHA_GROUPED_GEMM(escha_gemm_grouped_k3_r4_c2, 3, 4, 2)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k3_r8_c2, 3, 8, 2)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k3_r8_c4, 3, 8, 4)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k3_r16_c2, 3, 16, 2)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k3_r16_c4, 3, 16, 4)
ESCHA_GROUPED_GEMM(escha_gemm_grouped_k3_r8_c8, 3, 8, 8)
