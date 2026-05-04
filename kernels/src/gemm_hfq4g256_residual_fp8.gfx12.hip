#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// Experimental HFQ4-G256 FP8 (E4M3) residual GEMM for RDNA4 (gfx1200/gfx1201).
//
// Sister of `gemm_hfq4g256_residual_mmq.gfx12.hip` (the i8 MMQ port). Both
// kernels compute the same residual GEMM (Y[m,n] = sum_k W[m,k] * X[n,k] +
// Y[m,n]) but trade the i8-WMMA + Q8_1 scale-correction recipe for a
// straight FP8-WMMA + per-tile-col activation scale.
//
// Why FP8 over i8 MMQ on RDNA4:
//   - FP8 wmma accumulates into FP32 directly — no INT32->FP32 cast and no
//     per-K=32 dmA*dsB scale-correction loop in the inner dot.
//   - Activation is FP32 in, cast to FP8 INSIDE this kernel — we drop the
//     separate `quantize_q8_1_mmq_ds4_gfx12` pre-pass kernel entirely.
//   - HFQ4 dequant ( w = scale*q + zero ) is BAKED into the FP8 cast — no
//     metadata in LDS for HFQ4 scale/zero; only the activation per-col
//     idScale is staged.
//   - Same 2x peak vs FP16 on RDNA4 matrix cores as i8 (1024 vs 512 ops/
//     cycle/CU per AMD spec) but with the metadata overheads gone.
//
// Layout contract (locked by probe_wmma_fp8_layout.gfx12.hip on R9700 silicon
// at commit 7984e38):
//   - Builtin: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc)`
//     with signature `(int32x2 a, int32x2 b, float8 acc) -> float8`. NO clamp/
//     sign args (unlike iu4/iu8). Acc is FP32 x8 per lane.
//   - A/B operand: int32x2 per lane = 8 FP8 (E4M3) bytes = K-half of one row.
//     lane l: row m = l & 0xF, k-half = 8*(l>>4) within the K=16 sub-tile.
//   - Acc layout: lane l, slot j in [0,8) -> C[8*(l>>4) + j][l & 0xF]. Same
//     8-row-block as iu4/iu8, just with FP32 slots instead of i32.
//   - FP32->FP8 packing: `__builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, word_sel)`
//     packs 2 FP32 -> 2 FP8 bytes into a 16-bit half of `old`. word_sel=false
//     -> low 16 bits, true -> high 16 bits. 4 calls fill 8 FP8 = 1 int32x2.
//
// LDS budget (gfx12 effective ceiling ~64 KB observed via launch-failure
// boundary on iu8 MMQ port):
//   - tile_x: 128 rows x K=128 FP8 = 16 KB  (= 128 rows x 32 i32 stride)
//   - tile_y: 128 cols x K=128 FP8 = 16 KB
//   - tile_idS: 128 floats = 0.5 KB           (per-col activation idScale)
//   Total ~32.5 KB - well under the ceiling. Per kb iteration covers a K=128
//   half-group (so each HFQ4 group of K=256 takes 2 kb iters, sharing the
//   same scale/zero across both halves).
//
// First-cut variant only: basic `_residual_fp8` with runtime `add` flag.
// Boundary-elided `_full_add` / `_full_set` are RDNA3 / iu8 reserved follow-
// ups once correctness lands.

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_GFX12 1
#else
#define HIPFIRE_GFX12 0
#endif

#define MMQ_X 128
#define MMQ_Y 128
#define MMQ_NWARPS 8
#define WARP_SIZE 32
#define FP8_K_PER_KB 128       // K positions covered per kb iteration (= half-group)
#define FP8_K_I32_STRIDE 32    // K=128 / 4 fp8-per-i32

#if !HIPFIRE_GFX12

extern "C" __global__ void gemm_hfq4g256_residual_fp8_gfx12(
    const char*,
    const float*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_hfq4g256_residual_fp8_preconv_gfx12(
    const unsigned char*,
    const float*,
    float*,
    int,
    int,
    int,
    int
) {}

#else

// E4M3 max representable magnitude (per OCP / IEEE FP8 spec).
__device__ __forceinline__ float fp8_e4m3_max() { return 448.0f; }

// ---------------------------------------------------------------------------
// HFQ4 -> FP8 weight tile loader. Dequants 8 nibbles -> 8 FP32 (applying the
// per-group scale + zero) -> 8 FP8 bytes (= int32x2 LDS slot) per thread.
//
// Layout in LDS: tile_x[row * 32 + k_i32] where k_i32 in [0..32) covers K=
// [0..128) of this kb iter. 4 FP8 bytes per i32, so k_i32 = K/4.
//
// HFQ4 group on disk is 256 elems with one shared (FP32 scale, FP32 zero).
// kb iteration covers 128 of those 256 elems; the high/low half is selected
// by (kb & 1). The group index on disk is (kb >> 1).
// ---------------------------------------------------------------------------
template <bool FULL>
static __device__ __forceinline__ void load_hfq4_tile_fp8(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    // 256 threads = 16 rows-per-iter x 16 nibble-u32-per-row.
    // Each thread loads 1 u32 of nibbles (= 8 nibbles = K=8 worth) and writes
    // 2 i32 of FP8 (= 8 FP8 bytes) at tile_x[row * 32 + 2 * k_chunk + 0..1].
    const int tid = (int)(threadIdx.y * WARP_SIZE + threadIdx.x);
    const int row_in_iter = tid >> 4;          // 0..15
    const int k_chunk = tid & 0xF;             // 0..15, indexes 4-byte nibble cluster within K=128
    const int kb_group = kb >> 1;              // HFQ4 group index on disk
    const int kb_half = kb & 1;                // 0 = K=[0..127], 1 = K=[128..255] within group

    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += 16) {
        const int i = i0 + row_in_iter;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb_group) * 136;

        // Group metadata is constant per row across all k_chunks. Compiler
        // hoists this out via a uniform-sub-warp-dispatch pattern; if not, the
        // duplicated loads land in L0 anyway.
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        // Nibble offset within the group: skip 8B header, then 64 B for the
        // low half if we want the high half.
        const int nibble_offset = 8 + kb_half * 64 + k_chunk * 4;
        const unsigned int qs0 = *(const unsigned int*)(gp + nibble_offset);

        // Dequant 8 nibbles -> 8 FP32 (BAKE scale + zero into the FP8 value).
        const float v0 = sc * (float)((qs0      ) & 0xFu) + zp;
        const float v1 = sc * (float)((qs0 >>  4) & 0xFu) + zp;
        const float v2 = sc * (float)((qs0 >>  8) & 0xFu) + zp;
        const float v3 = sc * (float)((qs0 >> 12) & 0xFu) + zp;
        const float v4 = sc * (float)((qs0 >> 16) & 0xFu) + zp;
        const float v5 = sc * (float)((qs0 >> 20) & 0xFu) + zp;
        const float v6 = sc * (float)((qs0 >> 24) & 0xFu) + zp;
        const float v7 = sc * (float)((qs0 >> 28) & 0xFu) + zp;

        // Cast 8 FP32 -> 2 i32 (= 8 FP8 bytes, low-byte-order: v0,v1,...,v7).
        unsigned int w0 = 0u;
        w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, w0, false);
        w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, w0, true);
        unsigned int w1 = 0u;
        w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v4, v5, w1, false);
        w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v6, v7, w1, true);

        tile_x[i * FP8_K_I32_STRIDE + 2 * k_chunk + 0] = (int)w0;
        tile_x[i * FP8_K_I32_STRIDE + 2 * k_chunk + 1] = (int)w1;
    }
}

// ---------------------------------------------------------------------------
// Activation FP32 -> FP8 staging with per-(tile_y col) dynamic scale.
//
// Strategy: 1 warp owns 16 cols (warp_id * 16 + col_in_warp). Lane l within
// warp -> (col_in_warp = l & 0xF, k_half = l >> 4). 2 lanes share the same
// col with different K-halves (K=[0..63] vs K=[64..127] of this kb iter).
//
// Pass 1: read X, compute amax over the lane's 64 K positions.
// Combine across the 2 lanes via __shfl_xor(amax, 16). amax reduction is
// per-col now.
//
// scale_b[col] = E4M3_MAX / max(|x|, eps), inv_scale_b[col] = 1 / scale_b.
// Lane k_half==0 stores inv_scale_b to tile_idS[col].
//
// Pass 2: re-read X (relies on L1/L2 hit), multiply by scale_b, cast to
// FP8 via cvt_pk_fp8_f32, store into tile_y[col * 32 + k_half * 16 + j].
// ---------------------------------------------------------------------------
template <bool FULL>
static __device__ __forceinline__ void load_activation_tile_fp8(
    const float* __restrict__ X,
    int* __restrict__ tile_y,
    float* __restrict__ tile_idS,
    int col0,
    int kb,
    int K,
    int N
) {
    const int warp_id = (int)threadIdx.y;
    const int lane = (int)threadIdx.x;
    const int col_in_warp = lane & 0xF;
    const int k_half = lane >> 4;
    const int col = warp_id * 16 + col_in_warp;
    const int col_global = col0 + col;
    const int kb_K_offset = kb * FP8_K_PER_KB;  // base K offset for this kb iter
    const int k_base = kb_K_offset + k_half * 64;

    const bool col_in_range = FULL ? true : (col_global < N);

    // ---- Pass 1: amax over 64 K positions of one (col, k_half).
    float amax = 0.0f;
    if (col_in_range) {
        const float* xp = X + (long long)col_global * K + k_base;
        #pragma unroll
        for (int j = 0; j < 64; ++j) {
            const int k = k_base + j;
            const float v = (k < K) ? xp[j] : 0.0f;
            amax = fmaxf(amax, fabsf(v));
        }
    }

    // Reduce amax across the 2 lanes (k_half = 0 vs 1) sharing this col.
    amax = fmaxf(amax, __shfl_xor(amax, 16, WARP_SIZE));

    const float E4M3_MAX = fp8_e4m3_max();
    const float scale_b = (amax > 1.0e-20f) ? (E4M3_MAX / amax) : 0.0f;
    const float inv_scale_b = (amax > 1.0e-20f) ? (amax / E4M3_MAX) : 0.0f;

    // Lane 0..15 of warp writes inv_scale_b for its col. Both lanes (k_half=0
    // and k_half=1) have the same value post-reduce, so writing once suffices.
    // For out-of-range cols, write 0.0 — the C-store boundary check skips
    // writing those Y entries, but tile_idS still feeds the inner-dot
    // accumulator and we don't want to pull garbage from prior workgroup state.
    if (k_half == 0) {
        tile_idS[col] = col_in_range ? inv_scale_b : 0.0f;
    }

    // ---- Pass 2: scale + cast + store. 64 floats / 4 fp32-per-pack = 16
    // packs per lane, each producing 1 i32 of FP8. Stored at
    //   tile_y[col * 32 + k_half * 16 + j]  for j in [0..16).
    if (col_in_range) {
        const float* xp = X + (long long)col_global * K + k_base;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int kbase = k_base + 4 * j;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (kbase + 0 < K) v0 = xp[4 * j + 0];
            if (kbase + 1 < K) v1 = xp[4 * j + 1];
            if (kbase + 2 < K) v2 = xp[4 * j + 2];
            if (kbase + 3 < K) v3 = xp[4 * j + 3];
            unsigned int w = 0u;
            w = __builtin_amdgcn_cvt_pk_fp8_f32(v0 * scale_b, v1 * scale_b, w, false);
            w = __builtin_amdgcn_cvt_pk_fp8_f32(v2 * scale_b, v3 * scale_b, w, true);
            tile_y[col * FP8_K_I32_STRIDE + k_half * 16 + j] = (int)w;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            tile_y[col * FP8_K_I32_STRIDE + k_half * 16 + j] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// gfx12 FP8 WMMA dot product over a K=128 kb iter.
//
// Per warp covers 32 rows (rows_per_warp = 2 * tile_C_I) and iterates over
// 4 col-blocks of 16 cols each -> MMQ_X / (ntx * tile_C_J) = 4 col-blocks.
// Per kb iter = 8 K=16 sub-tiles. Per (col-block, m-tile, K=16 sub-tile):
//   - load 1 int32x2 a-frag and 1 int32x2 b-frag from LDS (8 FP8 each)
//   - issue 1 wmma call (FP8 wmma is K=16, NOT K=32 like iu8/iu4 effectively)
//   - apply the per-col activation inv_scale (constant within kb) and add
//     into the persistent `sum[]` accumulator
//
// Per kb iter total: 4 col-blocks * 2 m-tiles * 8 K=16 sub-tiles = 64 wmma
// calls per warp. The sum buffer persists across kb iters; the per-kb-iter
// inv_scale is applied per K=16 sub-tile (mathematically equivalent because
// scale is constant within kb).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void vec_dot_fp8_fp8_mma_gfx12(
    const int* __restrict__ x,
    const int* __restrict__ y,
    const float* __restrict__ idS,
    float* __restrict__ sum
) {
    using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using float32x8_t = __attribute__((__vector_size__(8 * sizeof(float)))) float;

    constexpr int tile_C_I = 16;
    constexpr int tile_C_J = 16;
    constexpr int tile_C_ne = 8;  // 8 acc slots per lane (8-row-block layout)
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C_I;  // = 2 m-tiles per warp
    constexpr int K_subtiles = FP8_K_PER_KB / tile_C_I;  // 8 K=16 sub-tiles

    const int tid = (int)threadIdx.x;
    const int m_lane = tid & 0xF;
    const int k_grp = tid >> 4;

    const int i0 = (int)(threadIdx.y / ntx) * rows_per_warp;
    const int j_offset_in_warp = (int)(threadIdx.y % ntx) * tile_C_J;
    (void)j_offset_in_warp;  // doc-only — used inline as `(threadIdx.y % ntx) * tile_C_J`
                             // in j_for_b below to keep the expression close to the read.

    const int* x_qs = x;
    const int* y_qs = y;

    #pragma unroll
    for (int k_sub = 0; k_sub < K_subtiles; ++k_sub) {
        // A fragment: per (m-tile, lane) load 2 i32 (= 8 FP8 = K-half of K=16).
        // Stride within row = 4 i32 per K=16 sub-tile, lane k-half offset = 2*k_grp.
        int32x2_t a_frag[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            const int row_in_tile = n * tile_C_I + m_lane;
            const int* row_ptr = x_qs + (i0 + row_in_tile) * FP8_K_I32_STRIDE + 4 * k_sub;
            a_frag[n][0] = row_ptr[2 * k_grp + 0];
            a_frag[n][1] = row_ptr[2 * k_grp + 1];
        }

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C_J) {
            // B fragment: lane (m_lane = col_in_tile) loads 2 i32 from one col.
            const int col_in_tile = m_lane;
            // The col within tile_y for this warp's n-tile contribution.
            // Each warp handles ntx m-tiles; the j-axis still iterates the
            // FULL 4 col-blocks (n_block_idx = j0 / 32). The B-frag we load
            // is the same for both m-tiles within a warp, hoisted out of
            // the n-loop below.
            const int j_for_b = j0 + (int)(threadIdx.y % ntx) * tile_C_J + col_in_tile;
            const int* col_ptr = y_qs + j_for_b * FP8_K_I32_STRIDE + 4 * k_sub;
            int32x2_t b_v;
            b_v[0] = col_ptr[2 * k_grp + 0];
            b_v[1] = col_ptr[2 * k_grp + 1];

            // Per-col inv_scale (activation un-scaling). Same for all 8 acc
            // slots of this lane within this n-tile (all share col=m_lane).
            const float invs = idS[j_for_b];

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                float32x8_t acc = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
                acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_frag[n], b_v, acc);

                // Apply per-kb inv_scale and add into persistent sum buffer.
                // Acc layout: lane l, slot j -> C[8*k_grp + j][m_lane] within
                // the (n*16, j_for_b)-tile. The sum-buffer index matches the
                // iu8 MMQ kernel: (j0/16 + n) * 8 + slot.
                #pragma unroll
                for (int l = 0; l < tile_C_ne; ++l) {
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l] += acc[l] * invs;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 weight tile loader — preconv variant.
//
// Reads pre-converted FP8 weights from a [M × K] byte buffer (1 byte per
// element, populated by ensure_fp8_shadow at model load via
// dequant_hfq4g256_to_fp8.gfx12.hip). Unlike load_hfq4_tile_fp8, no in-kernel
// dequant — just a direct global → LDS memcpy. Eliminates the per-prefill
// 8 cvt_pk_fp8_f32 calls + 8 FP32 mul-adds per thread per kb iter.
//
// Same LDS layout as the dequant variant: tile_x[row * 32 + k_i32] where
// k_i32 in [0..32) covers K=[0..128) of this kb iter. 4 FP8 bytes per i32.
// kb iter k covers FP8 bytes [k*128, (k+1)*128) per row.
//
// 256 threads = 16 rows-per-iter × 16 chunks-per-row. Each thread loads 8
// FP8 bytes (= 2 int32) from global, writes 2 int32 to LDS.
// ---------------------------------------------------------------------------
template <bool FULL>
static __device__ __forceinline__ void load_fp8_tile_preconv(
    const unsigned char* __restrict__ A_fp8,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int K
) {
    const int tid = (int)(threadIdx.y * WARP_SIZE + threadIdx.x);
    const int row_in_iter = tid >> 4;          // 0..15
    const int k_chunk = tid & 0xF;             // 0..15, indexes 8-byte FP8 chunk in K=128

    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += 16) {
        const int i = i0 + row_in_iter;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        // Direct FP8 read: row stride = K bytes, kb stride = 128 bytes,
        // chunk stride = 8 bytes. 8-byte aligned (k_chunk * 8).
        const int* fp8_i32 = (const int*)(A_fp8 + (long long)row * K
                                           + kb * FP8_K_PER_KB
                                           + k_chunk * 8);
        tile_x[i * FP8_K_I32_STRIDE + 2 * k_chunk + 0] = fp8_i32[0];
        tile_x[i * FP8_K_I32_STRIDE + 2 * k_chunk + 1] = fp8_i32[1];
    }
}

// ---------------------------------------------------------------------------
// Main FP8 residual GEMM kernel for gfx12 — preconv variant.
//
// Consumes pre-converted FP8 weights (from ensure_fp8_shadow). Identical to
// gemm_hfq4g256_residual_fp8_gfx12 except for the weight load path:
//   - that variant: HFQ4 packed (136 B / 256 elements) -> in-kernel dequant
//     -> FP8 LDS. Costs 8 cvt + 8 muladd per thread per kb iter.
//   - this variant: FP8 byte buffer (1 B / element) -> direct LDS memcpy.
// All other code (activation cast, wmma inner loop, C-store) is shared.
//
// Trade-off vs the dequant variant:
//   (+) eliminates the dequant arithmetic in the inner kb loop
//   (-) doubles weight bandwidth (FP8 1.0 B/w vs HFQ4 0.531 B/w on disk)
// Net win on compute-bound regimes; depends on bw saturation otherwise.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_fp8_preconv_gfx12(
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int kb_iters = K / FP8_K_PER_KB;

    extern __shared__ int smem[];
    int* tile_x = smem;
    int* tile_y = tile_x + MMQ_Y * FP8_K_I32_STRIDE;
    float* tile_idS = (float*)(tile_y + MMQ_X * FP8_K_I32_STRIDE);

    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < kb_iters; ++kb) {
        load_fp8_tile_preconv<false>(A_fp8, tile_x, row0, kb, M, K);
        load_activation_tile_fp8<false>(X, tile_y, tile_idS, col0, kb, K, N);
        __syncthreads();
        vec_dot_fp8_fp8_mma_gfx12(tile_x, tile_y, tile_idS, sum);
        __syncthreads();
    }

    // C-store identical to dequant variant.
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
                    if (add) { *yp += v; } else { *yp = v; }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main FP8 residual GEMM kernel for gfx12 — original (in-kernel dequant) variant.
// Kept for the case where ensure_fp8_shadow isn't pre-warmed and we want a
// one-launch path that doesn't need the FP8 shadow buffer. Production callers
// should prefer the _preconv variant above when the shadow is available.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_hfq4g256_residual_fp8_gfx12(
    const char* __restrict__ A,
    const float* __restrict__ X,
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
    const int kb_iters = K / FP8_K_PER_KB;  // 2 * groups_per_row

    extern __shared__ int smem[];
    int* tile_x = smem;                                                       // 128 rows x 32 i32 = 16 KB
    int* tile_y = tile_x + MMQ_Y * FP8_K_I32_STRIDE;                          // 128 cols x 32 i32 = 16 KB
    float* tile_idS = (float*)(tile_y + MMQ_X * FP8_K_I32_STRIDE);            // 128 floats = 512 B

    // Per-lane sum buffer. Layout matches the iu8 MMQ kernel:
    //   4 n-blocks x 2 m-tiles x 8 slots = 64 floats per lane.
    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < kb_iters; ++kb) {
        load_hfq4_tile_fp8<false>(A, tile_x, row0, kb, M, groups_per_row);
        load_activation_tile_fp8<false>(X, tile_y, tile_idS, col0, kb, K, N);
        __syncthreads();
        vec_dot_fp8_fp8_mma_gfx12(tile_x, tile_y, tile_idS, sum);
        __syncthreads();
    }

    // C-store. 8-row-block acc layout (matches iu8 MMQ):
    //   lane l, slot j -> C[8*(l>>4) + j][l & 0xF] within each 16x16 tile.
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
