#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// HFQ4v4 iu4 K=32 residual GEMM for RDNA4 (gfx1200/gfx1201).
//
// Sister of `gemm_hfq4g256_residual_iu4.gfx12.hip` (the v1 path that quality-
// failed coherence; PR #140). Same outer recipe (128x128 MMQ tile, Q4_1
// activations, iu4 K=32 wmma) but with a redesigned weight format:
//
//   v1 (HFQ4-G256 + iu4):
//     [4 B FP32 scale][4 B FP32 zero][128 B nibbles] per K=256 group.
//     Per-element noise ~14% from Q4_1 activations destroyed coherence on
//     all 4 model sizes (PR #140 / project_gfx12_iu4_breakthrough_2026_05_04.md).
//
//   v4 (HFQ4v4):
//     [16 B nibbles][2 B FP16 d] per K=32 group. Per-row FP16 mu sidecar.
//     SmoothQuant-style: per-row mu absorbs the per-output-channel
//     activation×weight mean; the K=32 group `d` quantizes the residual.
//     With the post-wmma correction `mu_w[r] * sum_q_a[c]`, the activation
//     distribution per-channel is mean-centered → symmetric Q4 [-7, 7] now
//     captures real dynamic range instead of clipping fat tails.
//
// Mu correction:
//   out[m, n] = sum_g d_w[m, g] * d_a[n, sub] * wmma_acc(W_iu4, X_iu4)
//             + mu_w[m] * sum_q_a_full[n]
//   where:
//     - d_w[m, g] is the FP16 scale for K=32 group g of row m
//     - d_a[n, sub] is the Q4_1 activation FP16 scale for col n, K=32 sub-block
//     - wmma_acc is the iu4 K=32 INT32 inner product
//     - mu_w[m] is the per-row FP16 mu (the SmoothQuant transfer)
//     - sum_q_a_full[n] = sum_k q_a[k] across the FULL K of col n
//
// We accumulate sum_q_a_full[n] on the fly as we walk through K — each
// Q4_1 sub-block's `ds.y / d_a` recovers the per-K=32 sum_q_a, which we
// add into a per-(thread, n) running sum. After the K-loop, one final
// MAD per (m, n) per lane folds in the mu correction.
//
// Builtin contract (same as v1, validated by probe_wmma_iu4_k32_layout):
//   __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
//       a_signed=false (HFQ4v4 nibbles are signed [-7,7] but we pretend
//                       they're unsigned bit patterns; we manually sign-
//                       extend when reading via byte<<4>>4, which becomes
//                       signed at the wmma input level. NOTE: per the
//                       layout probe the iu4 builtin reads each nibble's
//                       low 4 bits as the unsigned/signed value depending
//                       on the flag. To make our signed Q4 weights work,
//                       we must set a_signed=true).
//       a, b_signed=true, b, acc, clamp=true)
//
// Acc layout (8-row-block, validated by probe_wmma_iu4_k32_layout on
// hiptrx silicon at commit 03a5856):
//   lane l, slot j ∈ [0, 8) → C[8*(l>>4) + j][l & 0xF]
//
// Per-lane operand: int32x2 = 8 bytes = 16 INT4 (K-half of K=32). Lane l:
//   row m = l & 0xF, k-half = l >> 4 (so lane l holds K=[16*k_half,
//                                            16*k_half + 16)).

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#define MMQ_X 128
#define MMQ_Y 128
#define MMQ_NWARPS 8
#define WARP_SIZE 32

// HFQ4v4 layout constants:
//   - Group K = 32, 16 B nibbles + 2 B FP16 d = 18 B per group per row.
//   - Nibbles are SIGNED 4-bit packed 2/byte: low nibble = even index,
//     high nibble = odd index. Sign extension by `(int8_t)(byte << 4) >> 4`
//     and `(int8_t)(byte) >> 4`.
//   - 8 nibbles fit in one i32 (8 * 4 bits = 32 bits). One K=32 group in 4
//     contiguous i32s. To match the iu4 K=32 wmma layout (per-lane int32x2
//     covers K-half = 16 nibbles = 2 i32), we pack the K=32 group's 4 i32s
//     in K-order: i32[0..2) = first K-half, i32[2..4) = second K-half.
//
// LDS row stride for nibbles: tile_x_qs row stride = (k_per_stage / 8) i32.
// We stage K=128 per kb iter → 16 i32 per row.
#define MMQ_TILE_NE_K 32       // K covered per Q4_1 sub-block
#define K_PER_STAGE 128         // K per kb iter (= 4 K=32 sub-blocks)
#define MMQ_TILE_X_K_QS 16      // i32 per row in tile_x_qs (= K=128 / 8)
#define MMQ_TILE_X_K_DM 4       // half d per row per kb iter (= 4 K=32 d's per K=128)
#define MMQ_TILE_Y_K_Q4_1 20    // i32 per col in tile_y (= 80 B = block_q4_1_mmq)

// HFQ4v4 row stride in bytes (= groups_per_row * 18). Computed in Rust;
// passed in via `row_bytes_w` kernel arg.

// Q4_1 activation block layout — same as v1 path. block_q4_1_mmq covers
// K=128 of one token: 16 B half2[4] (d, d*sum_q) + 64 B INT4 packed.
struct block_q4_1_mmq {
    half2 ds[4];
    int8_t qs[64];
};

static_assert(sizeof(block_q4_1_mmq) == 80, "bad block_q4_1_mmq size");

#if !HIPFIRE_GFX12

extern "C" __global__ void gemm_hfq4v4_residual_iu4_gfx12(
    const char*,
    const __half*,
    const block_q4_1_mmq*,
    float*,
    int,
    int,
    int,
    int,
    int
) {}

#else

// ---------------------------------------------------------------------------
// HFQ4v4 weight tile loader.
//
// Per kb iter (K_PER_STAGE = 128) we load:
//   - Per row: 16 i32 of nibbles (= K=128 / 8 nibbles-per-i32) into tile_x_qs
//   - Per row: 4 FP16 d's (one per K=32 group) into tile_x_dm
//
// On-disk layout per row, per K=32 group: [16 B nibbles][2 B FP16 d] = 18 B.
// For groups covering K=[kb*128, kb*128 + 128) we read 4 contiguous groups
// = 4 * 18 = 72 bytes = 18 i32 per row per kb iter (16 of nibbles + 2 of
// d-padding-or-the-fp16-d-bytes).
//
// We fan out the read so each thread of a warp pulls 1 i32 of nibbles per
// outer iter (32 threads × 1 row → 32 i32 = 64 nibbles = K=64 covered, so
// 2 outer-iter steps per K=128). And the d's are read by 2 threads/row (8
// bytes each).
//
// To keep this simple, we read the 18 B-per-group structure unaligned via
// per-byte reads collapsed into u32 reads. Specifically, for each K=32
// group g in [0, 4), thread t in [0, 16) reads 1 byte of nibbles
// (nibbles cover bytes [0..16)); thread t in [16, 18) reads the 2 d-bytes.
// 18 threads handle one (row, group); we fan out across the 32-thread warp
// by handling 1 group per group of 18 threads per outer iter, and have
// MMQ_NWARPS warps cover MMQ_Y rows.
//
// Simpler: load the entire row's K-stride (= 4 groups × 18 B = 72 B) as
// 18 i32 = 72 B for a single row, with 18 threads doing 1 i32 each. The
// remaining 32-18=14 threads idle. Across 8 warps × 1 row → 8 rows per
// outer iter, so 16 outer iters cover 128 rows. Acceptable load cost.
// ---------------------------------------------------------------------------
template <bool FULL>
static __device__ __forceinline__ void load_hfq4v4_tile_iu4(
    const char* __restrict__ A,        // base pointer to weight blob
    int row_bytes_w,                    // bytes per row (= groups_per_row * 18)
    int group0_byte_off,                // byte offset within a row to start of K=128 stage
    int* __restrict__ tile_x_qs,        // [MMQ_Y][MMQ_TILE_X_K_QS] i32
    half* __restrict__ tile_x_dm,       // [MMQ_Y][MMQ_TILE_X_K_DM] FP16
    int row0,
    int M
) {
    // Each kb iter spans 4 K=32 groups = 72 bytes per row. We split the
    // 72-byte read into:
    //   - 16 i32 of nibbles per row (4 groups × 4 i32 each)
    //   - 4 FP16 d's per row (4 groups × 2 bytes each)
    //
    // For each K=32 group g:
    //   bytes [g*18, g*18+16) = 16 nibble bytes = 4 i32
    //   bytes [g*18+16, g*18+18) = 2 d bytes = 1 FP16
    //
    // The i32 reads land at byte offsets that are NOT 4-byte aligned (e.g.,
    // group 1 nibbles start at byte 18, not byte 16). We use unaligned
    // byte loads via memcpy-style char-pointer access, which the compiler
    // turns into the appropriate hardware load.

    // Outer loop: handle MMQ_Y / MMQ_NWARPS rows per warp.
    constexpr int rows_per_warp = MMQ_Y / MMQ_NWARPS;  // 16 rows per warp
    const int txi = threadIdx.x;

    #pragma unroll
    for (int ri = 0; ri < rows_per_warp; ++ri) {
        const int i = threadIdx.y * rows_per_warp + ri;  // row in tile (0..127)
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + (long long)row * row_bytes_w + group0_byte_off;

        // 16 threads handle the 16 i32 of nibbles (4 groups × 4 i32 each).
        // Layout: i32 idx = g * 4 + j (g in [0,4), j in [0,4)).
        //   byte offset within row stage = g * 18 + j * 4.
        if (txi < 16) {
            const int g = txi / 4;
            const int j = txi % 4;
            const int byte_off = g * 18 + j * 4;
            unsigned int v;
            // Unaligned-safe: use memcpy of a u32 from the byte pointer.
            __builtin_memcpy(&v, gp + byte_off, sizeof(unsigned int));
            tile_x_qs[i * MMQ_TILE_X_K_QS + txi] = (int)v;
        }

        // 4 threads handle the 4 FP16 d's. Threads 16..19 → d for groups
        // 0..3. Byte offset within stage = g * 18 + 16.
        if (txi >= 16 && txi < 20) {
            const int g = txi - 16;
            const int byte_off = g * 18 + 16;
            unsigned short raw;
            __builtin_memcpy(&raw, gp + byte_off, sizeof(unsigned short));
            half d_val = __ushort_as_half(raw);
            tile_x_dm[i * MMQ_TILE_X_K_DM + g] = d_val;
        }
    }
}

// ---------------------------------------------------------------------------
// gfx12 iu4 K=32 dot product over a K=128 stage cycle, HFQ4v4 variant.
//
// Differences vs v1 (`vec_dot_q4_1_q4_iu4_mma_gfx12`):
//   - Weight is signed Q4 (a_signed=true, was false for unsigned HFQ4 nibs).
//   - Per-K=32 group has its own FP16 d (4 d's per K=128 stage), not a
//     per-K=256 (scale, zero) pair.
//   - No zero_w * dsB.y term — the zero is gone (absorbed into per-row mu,
//     applied once after the full K-loop completes).
//   - Per-lane sum_q_a accumulator: one float per (col-block tile, n-slot)
//     that accumulates dsB.y / dsB.x summed over all sub-blocks. This is
//     used after the K-loop to apply the mu correction.
//
// Per stage cycle = 4 K=32 sub-blocks. Per (col-block, m-tile, K=32):
//   - load int32x2 a-frag (K-half of K=32) from tile_x_qs (signed Q4 weights)
//   - load int32x2 b-frag from tile_y (signed Q4 activations)
//   - issue 1 wmma call (a_signed=true, b_signed=true)
//   - apply per-K=32 (d_w[g], d_a[sub]) scale into the persistent sum[].
// After the full K-loop, one MAD per (m, n) folds in mu_w[m] * sum_q_a_full[n].
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void vec_dot_hfq4v4_q4_iu4_mma_gfx12(
    const int* __restrict__ x_qs,
    const half* __restrict__ x_dm,        // [MMQ_Y][MMQ_TILE_X_K_DM]
    const int* __restrict__ y,
    float* __restrict__ sum,              // [n_blocks * ntx * tile_C_ne]
    float* __restrict__ sum_x_acc       // [n_blocks * tile_C_ne]
) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    constexpr int tile_C_I = 16;
    constexpr int tile_C_J = 16;
    constexpr int tile_C_ne = 8;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C_I;  // 2 m-tiles per warp

    y += (threadIdx.y % ntx) * (tile_C_J * MMQ_TILE_Y_K_Q4_1);
    const int* y_qs_base = y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    const int tid    = (int)threadIdx.x;
    const int m_lane = tid & 0xF;
    const int k_grp  = tid >> 4;

    constexpr int K_I32_PER_SUB = MMQ_TILE_NE_K / 8;  // 4 i32 per K=32 sub-block
    constexpr int N_SUBS = K_PER_STAGE / MMQ_TILE_NE_K;  // 4 sub-blocks per stage
    constexpr int N_J0_STEPS = MMQ_X / (ntx * tile_C_J);  // = 4 col-blocks per warp

    #pragma unroll
    for (int sub_idx = 0; sub_idx < N_SUBS; ++sub_idx) {
        // For HFQ4v4, the K=32 group's 4 i32 are contiguous within the
        // per-row stage (no inter-group padding). i32 base for sub_idx
        // within the K=128 stage = sub_idx * 4 i32.
        const int k0_i32 = sub_idx * K_I32_PER_SUB;

        int32x2_t a_frag[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            const int row_in_tile = n * tile_C_I + m_lane;
            const int* row_ptr = x_qs + (i0 + row_in_tile) * MMQ_TILE_X_K_QS + k0_i32;
            a_frag[n][0] = row_ptr[2 * k_grp + 0];
            a_frag[n][1] = row_ptr[2 * k_grp + 1];
        }

        // Step counter for j0 (avoids j0/tile_C_J giving sparse indices
        // into sum_x_acc[]).
        int j0_step = 0;
        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C_J, ++j0_step) {
            const int col_in_tile = m_lane;
            const int* col_qs_ptr = y_qs_base
                + (j0 + col_in_tile) * MMQ_TILE_Y_K_Q4_1
                + sub_idx * K_I32_PER_SUB;
            int32x2_t b_v;
            b_v[0] = col_qs_ptr[2 * k_grp + 0];
            b_v[1] = col_qs_ptr[2 * k_grp + 1];

            const int j = j0 + col_in_tile;
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K_Q4_1 + sub_idx]);
            // dsB.x = d_a (FP16 → FP32), dsB.y = d_a * sum_q_a (per K=32).
            // For the mu correction we need
            //   sum_k X[c,k] = sum_g d_a[c,g] * sum_q_a[c,g] = sum_g dsB.y
            // so we accumulate dsB.y directly. (NOT dsB.y / dsB.x — that
            // would drop the activation-scale factor.)
            const float dsB_y = dsB.y;

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                int32x8_t acc = {0,0,0,0,0,0,0,0};
                // Both A and B are signed Q4: a_signed=true, b_signed=true.
                acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    /*a_signed*/ true, a_frag[n],
                    /*b_signed*/ true, b_v,
                    acc, /*clamp*/ true);

                #pragma unroll
                for (int l = 0; l < tile_C_ne; ++l) {
                    const int i_in_tile = n * tile_C_I + 8 * k_grp + l;
                    const int row_idx = i0 + i_in_tile;
                    const half d_w_h = x_dm[row_idx * MMQ_TILE_X_K_DM + sub_idx];
                    const float d_w = (float)d_w_h;
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l] +=
                        d_w * dsB.x * (float)acc[l];
                }
            }

            // Accumulate sum_x_full[c] = sum_g dsB.y[c, g] for the col
            // c = j0 + m_lane that THIS lane handles. Only added once per
            // (sub_idx, j0) — outside the inner `for (n)` m-tile loop.
            // Both k_grp lanes (0 and 1) end up with identical values
            // (dsB.y has no k_grp dimension); each lane uses its own
            // register copy in the post-K-loop mu correction.
            sum_x_acc[j0_step] += dsB_y;
        }
    }
}

// ---------------------------------------------------------------------------
// Main HFQ4v4 iu4 residual GEMM kernel for gfx12.
//
// Grid: ((M+127)/128, (N+127)/128, 1)   block: (32, 8, 1)
//
// Per workgroup processes a 128×128 output tile. Outer kb loop iterates over
// `groups_per_row / 4` kb iters (each kb iter covers K=128 = 4 K=32 groups).
//
// Args:
//   A          : HFQ4v4 weight blob (m × row_bytes_w bytes)
//   mu_row     : per-row FP16 mu sidecar (m FP16 values)
//   Xq         : pre-quantized Q4_1 activations (block_q4_1_mmq layout)
//   Y          : output [N × M] FP32 column-major
//   M, K, N    : matrix dims
//   row_bytes_w: bytes per row in A (= groups_per_row * 18)
//   add        : if 1, Y += result; if 0, Y = result
//
// LDS budget:
//   tile_y:    128 cols × 20 i32 × 4 B = 10 KB
//   tile_x_qs: 128 rows × 16 i32 × 4 B = 8 KB
//   tile_x_dm: 128 rows × 4 half × 2 B = 1 KB
//   total ~ 19 KB (well under gfx12's ~64 KB ceiling)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4v4_residual_iu4_gfx12(
    const char* __restrict__ A,
    const __half* __restrict__ mu_row,
    const block_q4_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int row_bytes_w,
    int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int kb_iters = K / K_PER_STAGE;  // 1 kb iter per K=128

    extern __shared__ int smem[];
    int*   tile_y    = smem;
    int*   tile_x_qs = tile_y + MMQ_X * MMQ_TILE_Y_K_Q4_1;
    half*  tile_x_dm = (half*)(tile_x_qs + MMQ_Y * MMQ_TILE_X_K_QS);

    // Per-lane accumulators.
    constexpr int n_blocks = MMQ_X / (16 * 2);  // = 4 col-blocks per warp's n-tiles
    constexpr int per_lane_sum = n_blocks * 8;  // = 32 floats per lane (= 4 * 2 m-tiles * 8 slots)
    // Actually we keep the v1 layout: 4 blocks × 2 m-tiles × 8 slots = 64 per lane.
    constexpr int sum_len = 4 * 2 * 8;
    float sum[sum_len] = {0.0f};
    // sum_q_a: one per col-block, lane-local. 4 entries per lane.
    float sum_x_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < kb_iters; ++kb) {
        // Stage HFQ4v4 weight tile for this K=128 kb iter.
        // Per kb iter we span 4 K=32 groups × 18 B per row = 72 B per row.
        // Group 0 of this kb starts at byte offset (kb * 4) * 18 within the
        // row.
        const int group0_byte = (kb * 4) * 18;
        load_hfq4v4_tile_iu4<false>(
            A, row_bytes_w, group0_byte, tile_x_qs, tile_x_dm, row0, M);

        // Stage tile_y for this K=128 (same as v1 — block_q4_1_mmq).
        const int* by = (const int*)(Xq + (long long)kb * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_Q4_1; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            if (l < MMQ_X * MMQ_TILE_Y_K_Q4_1) {
                const int j = l / MMQ_TILE_Y_K_Q4_1;
                tile_y[l] = (col0 + j < N) ? by[l] : 0;
            }
        }
        __syncthreads();
        vec_dot_hfq4v4_q4_iu4_mma_gfx12(
            tile_x_qs, tile_x_dm, tile_y, sum, sum_x_acc);
        __syncthreads();
    }

    // ── Apply mu correction ─────────────────────────────────────────────
    // For each lane's (m-tile, n-block) acc slot:
    //   out[m, n] = sum[...] + mu_w[m] * sum_q_a[n]
    //
    // mu is read per-row from mu_row. sum_q_a per col j is the lane-local
    // accumulator (the lane corresponds to one specific col j = j0 +
    // col_in_tile via m_lane).
    constexpr int tile_C_I = 16;
    constexpr int tile_C_J = 16;
    constexpr int tile_C_ne = 8;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C_I;

    const int tid = (int)threadIdx.x;
    const int m_lane = tid & 0xF;
    const int k_grp = tid >> 4;
    const int i0 = (int)(threadIdx.y / ntx) * rows_per_warp;
    const int j_offset_in_warp = (int)(threadIdx.y % ntx) * tile_C_J;

    int j0_step = 0;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C_J, ++j0_step) {
        const float sqa = sum_x_acc[j0_step];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C_ne; ++l) {
                const int i = i0 + n * tile_C_I + 8 * k_grp + l;
                const int row_idx_global = row0 + i;
                if (row_idx_global < M) {
                    const float mu = (float)mu_row[row_idx_global];
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l] += mu * sqa;
                }
            }
        }
    }

    // ── C-store ────────────────────────────────────────────────────────
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C_J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C_ne; ++l) {
                const int i = i0 + n * tile_C_I + 8 * k_grp + l;
                const int j = j0 + j_offset_in_warp + m_lane;
                if (row0 + i < M && col0 + j < N) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C_J + n) * tile_C_ne + l];
                    if (add) {
                        *yp += v;
                    } else {
                        *yp = v;
                    }
                }
            }
        }
    }
}

#endif
