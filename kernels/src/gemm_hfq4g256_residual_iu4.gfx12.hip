#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// Experimental HFQ4-G256 iu4 K=32 residual GEMM for RDNA4 (gfx1200/gfx1201).
//
// Sister of `gemm_hfq4g256_residual_mmq.gfx12.hip` (the iu8 K=16 port). Same
// outer recipe (128x128 MMQ tile, HFQ4 weights staged in LDS, activation pre-
// quantized into a packed scratch buffer) but uses the iu4 K=32 wmma builtin:
//
//   __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
//       bool a_signed, v2i a, bool b_signed, v2i b, v8i acc, bool clamp)
//
// Why iu4 K=32 over iu8 K=16:
//   - 2x K-per-call (32 vs 16) → half the LDS-read traffic per K-tile and
//     half the wmma launch count. AMD spec'd 4x FP16 throughput on RDNA4 vs
//     2x for iu8.
//   - HFQ4 nibbles map NATIVELY to iu4 unsigned indices [0..15] — no weight
//     conversion, just byte-repack into iu4 lane-form.
//   - Activation pre-quant produces SIGNED INT4 in [-7,7] (Q4_1 form). The
//     iu8 MMQ reference uses A=weight / B=activation; we mirror that, so:
//     `a_signed=false` (HFQ4 weight A side, unsigned [0..15]) and
//     `b_signed=true` (Q4_1 activation B side, signed [-7..7]).
//
// Layout contract (locked by probe_wmma_iu4_k32_layout.gfx12.hip on hiptrx
// silicon at commit 03a5856):
//   - Per-lane operand: int32x2 = 8 bytes = 16 INT4 = K-half of K=32. Layout
//     lane l: row m = l & 0xF, k-half = l >> 4 (so lane l holds K=[16*k_half,
//     16*k_half+16)).
//   - Acc layout (8-row-block): lane l, slot j ∈ [0,8) → C[8*(l>>4) + j][l & 0xF].
//
// Algebraic correction per K=32 sub-block (Q4_1 activation, HFQ4 weight):
//   wmma returns int_acc = sum_k (q_a[k] * q_w[k]) over K=32, signed*unsigned.
//   True dot:
//     w[m][k] = scale_w * q_w[m][k] + zero_w        (HFQ4 group of K=256)
//     x[n][k] = d_a    * q_a[n][k]                  (Q4_1 sub-block of K=32)
//   sum_k w*x = scale_w * d_a * int_acc + zero_w * d_a * sum_k(q_a[k])
//             = scale_w * dsB.x * int_acc + zero_w * dsB.y
//   where dsB = (d_a, d_a * sum_q_a) is the half2 from block_q4_1_mmq.ds[sub].
//
// Note that scale_w / zero_w are CONSTANT over the K=256 HFQ4 group, while
// dsB varies per K=32 sub-block. The inner loop fetches dsB per sub-block
// and uses the per-row (scale_w, zero_w) once per group iter.
//
// Skips the `_full_add` and `_full_set` boundary-elided variants — only the
// basic `_residual_iu4` with a runtime `add` flag. Those can be added later
// once the basic variant is validated.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#define MMQ_X 128
#define MMQ_Y 128
#define MMQ_NWARPS 8
#define WARP_SIZE 32

// One iu4 K=32 wmma call covers K=32. Per kb iter (= half of a 256-elem HFQ4
// group), we cover K=128 — so 4 K=32 sub-blocks per kb-iter.
#define MMQ_TILE_NE_K 32        // K covered per Q4_1 sub-block (per inner-loop step)
#define K_PER_STAGE 128         // K covered per stage-and-dispatch (= 4 sub-blocks)

// LDS row strides:
//   tile_x_qs row stride = 32 i32 (covers K=256 HFQ4 group worth of nibbles,
//                                   8 nibbles per i32). NO padding — gfx12
//                                   wave32 handles tile_x bank conflicts via
//                                   per-row distinct cache lines, and the
//                                   inner-loop access pattern is K-strided.
//   tile_x_dm  row stride = 1 half2 per row per group iter (HFQ4 sc/zp).
//   tile_y     row stride = 20 i32 = sizeof(block_q4_1_mmq) / 4 (per stage).
#define MMQ_TILE_X_K_QS 32      // i32 per row in tile_x_qs (K=256 group)
#define MMQ_TILE_Y_K_Q4_1 20    // i32 per col in tile_y (one block_q4_1_mmq = 80 B)

// block_q4_1_mmq layout (declared here for the GEMM-side cast). Mirrors the
// layout produced by quantize_q4_1_mmq_ds4_gfx12 in
// quantize_q4_1.gfx12.hip — kept in sync by hand (same struct, both files).
struct block_q4_1_mmq {
    half2 ds[4];        // (d, d*sum_q) per K=32 sub-block
    int8_t qs[64];      // 4 sub-blocks * 32 INT4 packed 2-per-byte
};

static_assert(sizeof(block_q4_1_mmq) == 80, "bad block_q4_1_mmq size");

#if !HIPFIRE_GFX12

extern "C" __global__ void gemm_hfq4g256_residual_iu4_gfx12(
    const char*,
    const block_q4_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

#else

// ---------------------------------------------------------------------------
// HFQ4 weight tile loader. Stages K=256 worth of HFQ4 nibbles + the per-group
// (scale, zero) FP16 metadata into LDS, ONCE per kb iter (covers both halves
// of the K=128 inner stage cycles).
//
// Per row, 32 threads collaboratively read 32 i32 of nibble data (= K=256 of
// nibbles) directly into LDS — no per-byte unpacking needed. The HFQ4 disk
// format already has the nibbles tightly packed with low-nibble = even K,
// high-nibble = odd K — exactly the layout iu4 wmma expects per int32 (8
// nibbles per i32, k0..k7 in order). So this is a straight memcpy.
//
// LDS layout in tile_x_qs: row * MMQ_TILE_X_K_QS + k_i32, where k_i32 in
// [0, 32) covers K = [0, 256) of one HFQ4 group.
// ---------------------------------------------------------------------------
template <bool FULL>
static __device__ __forceinline__ void load_hfq4_tile_iu4(
    const char* __restrict__ A,
    int* __restrict__ tile_x_qs,
    half2* __restrict__ tile_x_dm,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    constexpr int rows_per_warp = 1;
    const int txi = threadIdx.x;

    // Stage 32 i32 per row (= K=256 nibbles, the full HFQ4 group). 32 threads
    // per warp, 1 row per warp per outer iter, MMQ_NWARPS warps → cover MMQ_Y
    // rows in MMQ_Y / MMQ_NWARPS outer iters. Each thread reads 1 i32 at
    // column offset txi * 4 within the per-row nibble payload.
    for (int i0 = 0; i0 < MMQ_Y; i0 += rows_per_warp * MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
        // Skip 8 B header (FP32 sc + FP32 zp), then read txi-th i32 of nibbles.
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + txi * 4);
        tile_x_qs[i * MMQ_TILE_X_K_QS + txi] = (int)qs0;
    }

    // Stage the per-group (scale, zero) half2. One half2 per row is enough
    // (the inner loop reads dmA once per group). Use 1 thread per row from
    // a subset of warps to avoid contention; 8 warps × 16 rows-per-warp =
    // 128 rows in 1 outer iter.
    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        // Only thread x = 0 (within each pair) writes — the /2 above means
        // pairs of threads target the same row. We could write redundantly
        // across both, but a guarded write keeps things explicit.
        if (i < MMQ_Y && (threadIdx.x & 1) == 0) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            tile_x_dm[i] = make_half2((_Float16)sc, (_Float16)zp);
        }
    }
}

// ---------------------------------------------------------------------------
// gfx12 iu4 K=32 dot product over a K=128 stage cycle.
//
// Per warp covers 32 rows (rows_per_warp = 2 * tile_C_I) and iterates over
// 4 col-blocks of 16 cols each → MMQ_X / (ntx * tile_C_J) = 128 / 32 = 4.
// Per stage cycle = 4 K=32 sub-blocks. Per (col-block, m-tile, K=32 sub-block):
//   - load 1 int32x2 a-frag (16 INT4 = K-half of K=32) for the activation
//   - load 1 int32x2 b-frag for the weight (per col)
//   - issue 1 wmma call (iu4 K=32 — twice the K-coverage of iu8 K=16)
//   - apply per-row (scale_w, zero_w) and per-(col, K=32) (d_a, d_a*sum_q_a)
//     correction, add into the persistent `sum[]` accumulator.
//
// k00 selects the K-base offset within the staged tile_x_qs (in i32 stride):
//   k00 = 0  → first half of the HFQ4 group (K=[0..128))
//   k00 = 16 → second half (K=[128..256))     [16 i32 = K=128 nibbles]
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void vec_dot_q4_1_q4_iu4_mma_gfx12(
    const int* __restrict__ x_qs,
    const half2* __restrict__ x_dm,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00_i32
) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    constexpr int tile_C_I = 16;
    constexpr int tile_C_J = 16;
    constexpr int tile_C_ne = 8;  // 8 acc slots per lane (8-row-block layout)
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C_I;  // = 2 m-tiles per warp

    // Per-warp partition into the K-axis stage. y is the LDS pointer to
    // tile_y (cols start). We offset by (warp_id_within_n) * MMQ_TILE_Y_K_Q4_1
    // * tile_C_J so each warp's "n" pair handles its assigned col block. This
    // mirrors the iu8 reference's `y += (threadIdx.y % ntx) * (tile_C::J *
    // MMQ_TILE_Y_K)` partition.
    y += (threadIdx.y % ntx) * (tile_C_J * MMQ_TILE_Y_K_Q4_1);
    const int* y_qs_base = y + 4;        // first 4 i32 = 4 half2 = ds[]
    const half2* y_dm = (const half2*)y; // ds[4]

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    // Per-lane decomposition (8-row-block + iu4 K=32):
    //   m_lane = tid & 0xF   → row within 16-row tile (also col for B)
    //   k_grp  = tid >> 4    → selects K-half (K=0..15 vs K=16..31) of K=32 sub-tile
    const int tid    = (int)threadIdx.x;
    const int m_lane = tid & 0xF;
    const int k_grp  = tid >> 4;

    // Per K=32 sub-block (4 sub-blocks per K=128 stage cycle):
    //   - k01 ∈ {0, 8, 16, 24} (i32 stride within the K=128 stage; 8 i32 per K=32)
    //     OH WAIT — i32 packs 8 nibbles, so K=32 = 32/8 = 4 i32 per row stride.
    //     k01 i32-stride steps: 0, 4, 8, 12.
    //   - sub_idx ∈ {0, 1, 2, 3} (Q4_1 sub-block index within the K=128 stage)
    constexpr int K_I32_PER_SUB = MMQ_TILE_NE_K / 8;  // = 4 i32 per K=32 sub-block
    constexpr int N_SUBS = K_PER_STAGE / MMQ_TILE_NE_K;  // = 4 sub-blocks per stage

    #pragma unroll
    for (int sub_idx = 0; sub_idx < N_SUBS; ++sub_idx) {
        const int k0_i32 = k00_i32 + sub_idx * K_I32_PER_SUB;

        // Per-m-tile activation pre-load. iu4 K=32 wants int32x2 per lane =
        // 2 i32 (= 16 INT4 = K-half of K=32). Lane k_grp picks the K-half:
        //   k_grp=0 → i32 [0,1] of the K=32 sub-block (K=0..15)
        //   k_grp=1 → i32 [2,3] of the K=32 sub-block (K=16..31)
        //
        // For tile_x_qs (HFQ4 weight): row m = i0 + n*16 + m_lane.
        int32x2_t a_frag[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            const int row_in_tile = n * tile_C_I + m_lane;
            const int* row_ptr = x_qs + (i0 + row_in_tile) * MMQ_TILE_X_K_QS + k0_i32;
            a_frag[n][0] = row_ptr[2 * k_grp + 0];
            a_frag[n][1] = row_ptr[2 * k_grp + 1];
        }

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C_J) {
            // B fragment: per-col K=32 sub-block. Lane (m_lane = col_in_tile)
            // pulls 2 i32 from one col's (sub_idx)-th K=32 sub-block.
            // tile_y col c is laid out as [4 half2 ds + 16 i32 qs]. The qs
            // i32 base for sub_idx is at i32 offset (4 + sub_idx * K_I32_PER_SUB)
            // = 4 + 4*sub_idx within the col's MMQ_TILE_Y_K_Q4_1 block.
            //
            // Above we already advanced y_qs_base = y + 4 (skipping ds), so
            // we just add sub_idx * 4 i32. Lane k_grp picks K-half within
            // that K=32: 2 i32 starting at offset 2*k_grp.
            const int col_in_tile = m_lane;
            const int* col_qs_ptr = y_qs_base
                + (j0 + col_in_tile) * MMQ_TILE_Y_K_Q4_1
                + sub_idx * K_I32_PER_SUB;
            int32x2_t b_v;
            b_v[0] = col_qs_ptr[2 * k_grp + 0];
            b_v[1] = col_qs_ptr[2 * k_grp + 1];

            // dsB = (d_a, d_a * sum_q_a) for this (col, K=32 sub-block).
            const int j = j0 + col_in_tile;
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K_Q4_1 + sub_idx]);

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                int32x8_t acc = {0,0,0,0,0,0,0,0};
                // A = weight (HFQ4 nibbles, unsigned [0..15]), staged in x_qs.
                // B = activation (Q4_1, signed [-7,7]), staged in y_qs_base.
                // Matches the iu8 MMQ reference's A=weight / B=activation
                // convention. Acc layout (8-row-block) reads C[m_block][n_col]
                // — same orientation as the C-store loop below.
                acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    /*a_signed*/ false, a_frag[n],   // HFQ4 weight, unsigned [0..15]
                    /*b_signed*/ true,  b_v,         // Q4_1 activation, signed [-7,7]
                    acc, /*clamp*/ true);

                // Apply HFQ4 (scale, zero) + Q4_1 (d, d*sum_q) correction.
                //   sum[m,n] += scale_w * dsB.x * acc + zero_w * dsB.y
                //
                // Per slot j ∈ [0,8): C-row within the m-tile = 8*k_grp + j;
                // C-col within the n-tile = m_lane (== col_in_tile above).
                // The (scale_w, zero_w) is per-row (one HFQ4 group covers
                // K=256 = the entire kb-iter, so it's constant across all 8
                // K=32 sub-blocks of this kb).
                #pragma unroll
                for (int l = 0; l < tile_C_ne; ++l) {
                    const int i_in_tile = n * tile_C_I + 8 * k_grp + l;
                    const int i = i0 + i_in_tile;
                    const float2 dmA = __half22float2(x_dm[i]);
                    const float scale_w = dmA.x;
                    const float zero_w  = dmA.y;
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l] +=
                        scale_w * dsB.x * (float)acc[l] + zero_w * dsB.y;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main iu4 residual GEMM kernel for gfx12.
//
// Grid: ((M+127)/128, (N+127)/128, 1)   block: (32, 8, 1)  warps: 8
//
// Per workgroup processes a 128 (rows) × 128 (cols) output tile. Outer kb
// loop iterates over `groups_per_row` (= K/256). Each kb iter:
//   1. Stage HFQ4 weights for the 128-row-block × K=256-group into LDS.
//   2. Stage tile_y (block_q4_1_mmq) for the K=128 LOW half of the kb,
//      vec_dot, sync.
//   3. Stage tile_y for the K=128 HIGH half of the kb, vec_dot, sync.
//
// LDS budget:
//   tile_x_qs: MMQ_Y rows × 32 i32 stride = 128 × 32 × 4 B = 16 KB
//   tile_x_dm: MMQ_Y half2                = 128 × 4 B       = 512 B
//   tile_y:    MMQ_X cols × 20 i32 stride = 128 × 20 × 4 B = 10 KB
//   total ~ 26.5 KB (well under gfx12's ~64 KB ceiling)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_iu4_gfx12(
    const char* __restrict__ A,
    const block_q4_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    // Layout: [tile_y (10 KB)] [tile_x_qs (16 KB)] [tile_x_dm (512 B)]
    int*   tile_y    = smem;
    int*   tile_x_qs = tile_y + MMQ_X * MMQ_TILE_Y_K_Q4_1;
    half2* tile_x_dm = (half2*)(tile_x_qs + MMQ_Y * MMQ_TILE_X_K_QS);

    // Per-lane sum buffer:
    //   total elements = MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)
    //                  = 128 * 128 / 256 = 64
    // Layout: 4 n-blocks × 2 m-tiles × 8 slots per lane = 64 floats per lane.
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        // Stage HFQ4 weight tile for this K=256 group ONCE.
        load_hfq4_tile_iu4<false>(A, tile_x_qs, tile_x_dm, row0, kb, M, groups_per_row);

        // ---- Low half of the group (K=[0..128) within this kb) ----
        // Source: Xq + (2*kb) * N (block index 2*kb). Each block_q4_1_mmq is
        // 80 B = 20 i32 = MMQ_TILE_Y_K_Q4_1.
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_Q4_1; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            if (l < MMQ_X * MMQ_TILE_Y_K_Q4_1) {
                const int j = l / MMQ_TILE_Y_K_Q4_1;
                tile_y[l] = (col0 + j < N) ? by0[l] : 0;
            }
        }
        __syncthreads();
        // k00_i32 = 0 → low half of the K=256 group in tile_x_qs (= K=[0..128)).
        vec_dot_q4_1_q4_iu4_mma_gfx12(tile_x_qs, tile_x_dm, tile_y, sum, 0);
        __syncthreads();

        // ---- High half of the group (K=[128..256) within this kb) ----
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_Q4_1; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (l < MMQ_X * MMQ_TILE_Y_K_Q4_1) {
                    const int j = l / MMQ_TILE_Y_K_Q4_1;
                    tile_y[l] = (col0 + j < N) ? by1[l] : 0;
                }
            }
            __syncthreads();
            // k00_i32 = 16 → high half of the K=256 group in tile_x_qs
            //              (= K=[128..256), i.e. i32 offset 16 in MMQ_TILE_X_K_QS=32).
            vec_dot_q4_1_q4_iu4_mma_gfx12(tile_x_qs, tile_x_dm, tile_y, sum, 16);
            __syncthreads();
        }
    }

    // C-store. 8-row-block acc layout (matches iu8 / FP8 mmq):
    //   lane l, slot j → C[8*(l>>4) + j][l & 0xF] within each 16x16 tile.
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
