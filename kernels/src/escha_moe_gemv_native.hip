// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Escha-W2 routed-expert GEMVs that read the TRELLIS CODE DIRECTLY (Phase 2).
//
// # Why
//
// Phase 1 decoded escha's codes to Q8_0 at load: 1.0625 B/weight resident and
// 1.0625 B/weight across the bus on every token. The codes themselves are
// 0.25 B/weight (K=2) and 0.375 B/weight (K=3). At A3B shapes the routed half
// of a decode token moves 1.07 GB as Q8_0 against 0.294 GB native, and the
// whole model is 3.01 GB/token against 2.23 — a 69 tok/s roofline against 94
// on this box's measured 209 GB/s. Q8_0 could not reach the comparable mq4r
// SKU at ANY efficiency. So the expanded copy has to stop existing: these
// kernels decode inside the GEMV and the code is what crosses the bus.
//
// # The one structural constraint: a block owns a whole 16-wide tile column
//
// Escha's code is a 16x16 TRELLIS tile: 256 weights encoded as one 512-bit
// (K=2) or 768-bit (K=3) sliding-window bit stream. Any single weight needs a
// 16-bit window out of that stream, so a kernel that wants ONE weight from a
// tile must still touch 4 bytes of it. Decoding a tile for one output column
// therefore costs ~12 B of the 64 B tile for 16 weights (0.75 B/weight) —
// three times the format's own 0.25, and no better than Q8_0.
//
// A block that owns all SIXTEEN columns of a tile column-strip reads each tile
// exactly once and uses all 256 of its weights: 0.25 B/weight, the format's
// floor. That is why the block is 16 warps and `blockIdx.x` is a TILE column
// (`out/16` of them) rather than an output row.
//
// # Bit-exactness contract (load-bearing)
//
// Warp `w` of the block computes output row `blockIdx.x*16 + w` with the SAME
// lane->contraction-index mapping, the SAME accumulator count, the SAME loop
// order and the SAME final reduction as `escha_moe_gemv_k8_indexed.hip`. Only
// the weight's provenance changes: `__half2float(escha_cba(state))` instead of
// `d * (float)qval`. So this is bit-identical to running the Q8_0 kernel's
// arithmetic on exactly-decoded fp16 weights — which is exactly what
// `escha_gemv_f16_*` at the bottom of this file does, and what the gate
// `rdna-compute/examples/test_escha_native_gemv_gpu_vs_cpu.rs` asserts to the
// bit against both that kernel and `escha_ref`, the frozen CPU oracle.
//
// The narrow/wide split mirrors the Q8_0 kernels' `k <= 1536` rule for the
// same reason it exists there: the wide form folds FOUR interleaved
// accumulators and the narrow one, so picking the other variant for a given
// projection changes the answer. The host wrapper re-uses that threshold.
//
// # No codebook table, still
//
// `escha_cba` is five inline ops. A 65536-entry fp16 LUT would be 128 KB and
// gfx1151 has 64 KB of LDS, so there is no table here either — see
// `escha_decode_tiles.hip`, whose per-lane maths this reproduces.
//
// # K=2 and K=3 are separate, deliberately
//
// The two orders extract their 16-bit window from structurally different
// stream geometries (16 32-bit words with a 2-bit stride vs 24 words with a
// 3-bit stride and a modular wrap). They share the final "merge two words,
// shift, truncate" step because that step IS the same; everything that decides
// WHICH two words and WHICH shift is duplicated per K on purpose, and each K
// gets its own kernel entry points so neither pays a runtime branch.

// ── the codebook, inline ─────────────────────────────────────────────────────
// Byte-for-byte the same five ops as `escha_decode_tiles.hip` and
// `escha_ref::cba`. Duplicated rather than shared: G2 asserts the GPU decode
// bit-exact against the Rust oracle, and generating either from the other
// would make that gate circular. Same reasoning here.
__device__ __forceinline__ __half escha_native_cba(unsigned short state) {
    unsigned int r = ((unsigned int)state * 0xCBAC1FEDu) & 0x8FFF8FFFu;
    r ^= 0x3B603B60u;
    __half lo = __ushort_as_half((unsigned short)(r & 0xFFFFu));
    __half hi = __ushort_as_half((unsigned short)(r >> 16));
    return __hadd(lo, hi);  // fp16 RNE add — matches the reference exactly
}

// Load tile word `i` (a little-endian pair of int16s), as in
// `escha_decode_tiles::escha_load_word`. `off` is the word's SHORT offset
// (`2 * i`), precomputed once per lane because it is loop-invariant.
__device__ __forceinline__ unsigned int escha_native_word(
    const short* __restrict__ tile, int off) {
    return ((unsigned int)(unsigned short)tile[off]) |
           (((unsigned int)(unsigned short)tile[off + 1]) << 16);
}

// Where the state for tile element (row, col) lives.
//
// `escha_decode_tiles` runs 32 lanes x 8 values FORWARD: lane `l` decodes 8
// states out of one 32-bit window and scatters them to 8 (row, col) slots. A
// GEMV needs the INVERSE — "which window, which shift, for THIS (row, col)" —
// because the lane that must own a weight is fixed by the contraction index,
// not by the tile geometry.
//
// Inverting the forward map (`escha_decode_tiles.hip:96-103`):
//   row = 2*(l & 3) + (j & 1) + 8*((j >> 1) & 1)
//   col = 2*((l >> 3) + 4*(j >= 4)) + ((l >> 2) & 1)
// gives, uniquely,
//   l = ((col >> 1) & 3)*8 + (col & 1)*4 + ((row & 7) >> 1)
//   j = ((col >> 3) & 1)*4 + ((row >> 3) & 1)*2 + (row & 1)
// which is what both helpers below start from. A wrong inversion yields a
// full-rank, finite, plausible weight matrix — never a fault — so it is gated
// against `escha_ref` rather than eyeballed.
__device__ __forceinline__ int escha_native_lane_of(int row, int col) {
    return ((col >> 1) & 3) * 8 + (col & 1) * 4 + ((row & 7) >> 1);
}
__device__ __forceinline__ int escha_native_j_of(int row, int col) {
    return ((col >> 3) & 1) * 4 + ((row >> 3) & 1) * 2 + (row & 1);
}

// K=2: 16 words per tile, `escha_decode8_k2`'s geometry.
//
// That routine merges words (i0, i1) into a 64-bit value and takes a 32-bit
// window at `shift` (16 for even lanes, 0 for odd), then value `j` is the
// 16 bits at `2*(7-j)` inside the window. Composing the two shifts is exact:
// the widest total is 30, and 30 + 16 <= 64, so the truncation to
// `unsigned short` selects the identical bits.
__device__ __forceinline__ void escha_native_slot_k2(
    int row, int col, int* off_a, int* off_b, int* shift) {
    const int lane = escha_native_lane_of(row, col);
    const int j = escha_native_j_of(row, col);
    const int t_off = lane * 8;
    const int i1 = t_off >> 4;
    const int i0 = (i1 + 15) & 15;
    *off_a = 2 * i0;                        // the high half of `merged`
    *off_b = 2 * i1;                        // the low half
    *shift = (((~t_off) & 8) << 1) + 2 * (7 - j);
}

// K=3: 24 words per tile, `escha_decode8_k3`'s geometry — a 3-bit stride with
// a modular wrap, not a scaled copy of K=2.
//
// There the merge is of words (i0, i2), the two 32-bit windows are at `s2`
// (values 4..7) and `s2 + 12` (values 0..3), and value `j` is the 16 bits at
// `3*(3 - (j & 3))` inside its window. `s2` is in 1..32, so the largest total
// shift is 32 + 12 + 9 = 53 — under 64, so composing is defined, and the bits
// past 63 that the original's 32-bit truncation dropped are zeros either way.
__device__ __forceinline__ void escha_native_slot_k3(
    int row, int col, int* off_a, int* off_b, int* shift) {
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
    *off_a = 2 * (i0 % 24);                 // the high half of `merged`
    *off_b = 2 * (i2 % 24);                 // the low half
    *shift = s2 + (j < 4 ? BITS * 4 : 0) + BITS * (3 - (j & 3));
}

// ── the two accumulation forms ───────────────────────────────────────────────
//
// Both are transcriptions of `escha_moe_gemv_k8_indexed.hip`. The ONLY edit is
// the weight expression. Everything that decides the value of the sum — which
// lane owns which contraction index, the iteration order, the accumulator
// count, the `(acc0+acc1)+(acc2+acc3)` fold, the `__shfl_down` ladder — is
// copied. `TRELLIS_K` is a template parameter so the slot maths is resolved at
// compile time and the inner loop carries no branch on it.
//
// `code` addresses tiles as `[in/16][out/16][16*TRELLIS_K]` int16 — escha's
// own IN-major grid, unchanged from the file. `nt` is this block's tile
// column, `tn = M/16` the grid's width.

template <int TRELLIS_K>
__device__ __forceinline__ void escha_native_slot(
    int row, int col, int* off_a, int* off_b, int* shift) {
    if (TRELLIS_K == 2) escha_native_slot_k2(row, col, off_a, off_b, shift);
    else                escha_native_slot_k3(row, col, off_a, off_b, shift);
}

// Narrow form (Q8_0 twin: `escha_gemv_q8_0_moe_k8_indexed_batched`, K > 1536).
// One accumulator; lane `t` owns contraction indices t, t+32, t+64, ...
template <int TRELLIS_K>
__device__ __forceinline__ void escha_gemv_native_narrow_body(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_batch,
    float* __restrict__ y_batch,
    int M, int K
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 16 + warp;
    if (row >= M) return;
    const int krank = blockIdx.y;

    const short* __restrict__ code =
        reinterpret_cast<const short*>(expert_ptrs[topk_indices[krank]]);
    const float* __restrict__ x = x_batch + (size_t)krank * K;
    float* __restrict__ y = y_batch + (size_t)krank * M;

    // Loop-invariant: the block's tile column, the lane's row inside a tile,
    // and which of the iteration's two tiles the lane falls in. The slot
    // descriptor depends only on (row-in-tile, column-in-tile), so it is
    // hoisted out of the contraction loop entirely.
    const int tn = M >> 4;
    const int nt = blockIdx.x;
    const int in_row = lane & 15;
    const int tile_hi = lane >> 4;
    int off_a, off_b, shift;
    escha_native_slot<TRELLIS_K>(in_row, warp, &off_a, &off_b, &shift);

    // Tile `kt` of column `nt` starts at `(kt*tn + nt) * 16*TRELLIS_K` shorts.
    // This lane's tile in iteration `bi` is `kt = bi*2 + tile_hi`, so the base
    // is a loop-invariant origin plus `bi` times two strips.
    const size_t strip = (size_t)tn * (16 * TRELLIS_K);
    const short* __restrict__ strip0 =
        code + (size_t)nt * (16 * TRELLIS_K) + (size_t)tile_hi * strip;

    const int blocks_per_row = K / 32;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks_per_row; bi++) {
        const short* __restrict__ tile = strip0 + (size_t)bi * 2 * strip;
        const unsigned long long merged =
            ((unsigned long long)escha_native_word(tile, off_a) << 32) |
            escha_native_word(tile, off_b);
        const __half w = escha_native_cba((unsigned short)(merged >> shift));
        sum += __half2float(w) * x[bi * 32 + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}

// Wide form (Q8_0 twin: `escha_gemv_q8_0_wide_moe_k8_indexed_batched`,
// K <= 1536). Four interleaved accumulators with the `(acc0+acc1)+(acc2+acc3)`
// fold. The tail's `acc[t]` assignment is copied verbatim, including the fact
// that `t == 3` is unreachable.
template <int TRELLIS_K>
__device__ __forceinline__ void escha_gemv_native_wide_body(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_batch,
    float* __restrict__ y_batch,
    int M, int K
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 16 + warp;
    if (row >= M) return;
    const int krank = blockIdx.y;

    const short* __restrict__ code =
        reinterpret_cast<const short*>(expert_ptrs[topk_indices[krank]]);
    const float* __restrict__ x = x_batch + (size_t)krank * K;
    float* __restrict__ y = y_batch + (size_t)krank * M;

    const int tn = M >> 4;
    const int nt = blockIdx.x;
    const int in_row = lane & 15;
    const int tile_hi = lane >> 4;
    int off_a, off_b, shift;
    escha_native_slot<TRELLIS_K>(in_row, warp, &off_a, &off_b, &shift);

    const size_t strip = (size_t)tn * (16 * TRELLIS_K);
    const short* __restrict__ strip0 =
        code + (size_t)nt * (16 * TRELLIS_K) + (size_t)tile_hi * strip;

    const int blocks_per_row = K / 32;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = blocks_per_row >> 2;
    const int tail = blocks_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int bi = q << 2;
        const short* __restrict__ t0 = strip0 + (size_t)(bi + 0) * 2 * strip;
        const short* __restrict__ t1 = strip0 + (size_t)(bi + 1) * 2 * strip;
        const short* __restrict__ t2 = strip0 + (size_t)(bi + 2) * 2 * strip;
        const short* __restrict__ t3 = strip0 + (size_t)(bi + 3) * 2 * strip;

        const unsigned long long m0 =
            ((unsigned long long)escha_native_word(t0, off_a) << 32) |
            escha_native_word(t0, off_b);
        const unsigned long long m1 =
            ((unsigned long long)escha_native_word(t1, off_a) << 32) |
            escha_native_word(t1, off_b);
        const unsigned long long m2 =
            ((unsigned long long)escha_native_word(t2, off_a) << 32) |
            escha_native_word(t2, off_b);
        const unsigned long long m3 =
            ((unsigned long long)escha_native_word(t3, off_a) << 32) |
            escha_native_word(t3, off_b);

        const __half w0 = escha_native_cba((unsigned short)(m0 >> shift));
        const __half w1 = escha_native_cba((unsigned short)(m1 >> shift));
        const __half w2 = escha_native_cba((unsigned short)(m2 >> shift));
        const __half w3 = escha_native_cba((unsigned short)(m3 >> shift));

        acc0 += __half2float(w0) * x[(bi + 0) * 32 + lane];
        acc1 += __half2float(w1) * x[(bi + 1) * 32 + lane];
        acc2 += __half2float(w2) * x[(bi + 2) * 32 + lane];
        acc3 += __half2float(w3) * x[(bi + 3) * 32 + lane];
    }

    for (int t = 0; t < tail; t++) {
        const int bi = (quads << 2) + t;
        const short* __restrict__ tp = strip0 + (size_t)bi * 2 * strip;
        const unsigned long long m =
            ((unsigned long long)escha_native_word(tp, off_a) << 32) |
            escha_native_word(tp, off_b);
        const __half w = escha_native_cba((unsigned short)(m >> shift));
        const float contrib = __half2float(w) * x[bi * 32 + lane];
        if (t == 0) acc0 += contrib;
        else if (t == 1) acc1 += contrib;
        else if (t == 2) acc2 += contrib;
    }

    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}

// ── entry points ─────────────────────────────────────────────────────────────
// Grid: (M/16, slots, 1). Block: 512 (16 warps, one per column of the strip).
// The host wrapper enforces `M % 16 == 0` and `K % 32 == 0`.

#define ESCHA_NATIVE_GEMV(NAME, FORM, TK)                                      \
    __launch_bounds__(512)                                                  \
    extern "C" __global__ void NAME(                                           \
        const unsigned long long* __restrict__ expert_ptrs,                    \
        const int* __restrict__ topk_indices,                                  \
        const float* __restrict__ x_batch,                                     \
        float* __restrict__ y_batch,                                           \
        int M, int K                                                           \
    ) {                                                                        \
        FORM<TK>(expert_ptrs, topk_indices, x_batch, y_batch, M, K);           \
    }

ESCHA_NATIVE_GEMV(escha_gemv_native_k2_moe_k8_indexed_batched,
                  escha_gemv_native_narrow_body, 2)
ESCHA_NATIVE_GEMV(escha_gemv_native_k3_moe_k8_indexed_batched,
                  escha_gemv_native_narrow_body, 3)
ESCHA_NATIVE_GEMV(escha_gemv_native_k2_wide_moe_k8_indexed_batched,
                  escha_gemv_native_wide_body, 2)
ESCHA_NATIVE_GEMV(escha_gemv_native_k3_wide_moe_k8_indexed_batched,
                  escha_gemv_native_wide_body, 3)

// ── the F16 reference arm ────────────────────────────────────────────────────
//
// These read an OUT-major `[M, K]` fp16 expert slot — exactly what
// `escha_bare_to_f16` writes, i.e. the exactly-decoded weights with nothing
// re-quantised. Their whole purpose is to be the thing the fused kernels above
// are compared against: same grid, same block, same lane mapping, same
// accumulator structure, same reduction, and the weight expression written
// character-for-character the same (`__half2float(w) * x[...]`), so the ONLY
// difference between the two is where `w` came from. Equality is therefore a
// statement about the DECODE, not about floating-point luck.
//
// They are a gate arm, not a production route: `EschaWeightStore::F16` still
// runs host-routed. Admitting F16 to the indexed path would change the arm the
// G5 KLD reference is built from, and that reference has published numbers.

template <bool WIDE>
__device__ __forceinline__ void escha_gemv_f16_body(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_batch,
    float* __restrict__ y_batch,
    int M, int K
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 16 + warp;
    if (row >= M) return;
    const int krank = blockIdx.y;

    const __half* __restrict__ A =
        reinterpret_cast<const __half*>(expert_ptrs[topk_indices[krank]]);
    const float* __restrict__ x = x_batch + (size_t)krank * K;
    float* __restrict__ y = y_batch + (size_t)krank * M;
    const __half* __restrict__ row_data = A + (size_t)row * K;

    const int blocks_per_row = K / 32;
    float sum = 0.0f;
    if (!WIDE) {
        for (int bi = 0; bi < blocks_per_row; bi++) {
            const __half w = row_data[bi * 32 + lane];
            sum += __half2float(w) * x[bi * 32 + lane];
        }
    } else {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const int quads = blocks_per_row >> 2;
        const int tail = blocks_per_row & 3;
        for (int q = 0; q < quads; q++) {
            const int bi = q << 2;
            const __half w0 = row_data[(bi + 0) * 32 + lane];
            const __half w1 = row_data[(bi + 1) * 32 + lane];
            const __half w2 = row_data[(bi + 2) * 32 + lane];
            const __half w3 = row_data[(bi + 3) * 32 + lane];
            acc0 += __half2float(w0) * x[(bi + 0) * 32 + lane];
            acc1 += __half2float(w1) * x[(bi + 1) * 32 + lane];
            acc2 += __half2float(w2) * x[(bi + 2) * 32 + lane];
            acc3 += __half2float(w3) * x[(bi + 3) * 32 + lane];
        }
        for (int t = 0; t < tail; t++) {
            const int bi = (quads << 2) + t;
            const __half w = row_data[bi * 32 + lane];
            const float contrib = __half2float(w) * x[bi * 32 + lane];
            if (t == 0) acc0 += contrib;
            else if (t == 1) acc1 += contrib;
            else if (t == 2) acc2 += contrib;
        }
        sum = (acc0 + acc1) + (acc2 + acc3);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}

__launch_bounds__(512)
extern "C" __global__ void escha_gemv_f16_moe_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_batch,
    float* __restrict__ y_batch,
    int M, int K
) {
    escha_gemv_f16_body<false>(expert_ptrs, topk_indices, x_batch, y_batch, M, K);
}

__launch_bounds__(512)
extern "C" __global__ void escha_gemv_f16_wide_moe_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_batch,
    float* __restrict__ y_batch,
    int M, int K
) {
    escha_gemv_f16_body<true>(expert_ptrs, topk_indices, x_batch, y_batch, M, K);
}
