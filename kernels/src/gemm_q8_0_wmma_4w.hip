// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Multi-warp Q8 WMMA GEMM for gfx1151 (RDNA3.5). 4 cooperating warps,
// 64×64 output tile per block, LDS-staged X. Follows the llama.cpp MMQ
// pattern documented in pedapudi's gist for issue ggml-org/llama.cpp#21284
// (`mmq_x=48, mmq_y=64, nwarps=4` was their tuned config).
//
// Y[N, M] = X[N, K] @ A_q8[M, K]^T
//   A: Q8_0 packed weights. Per K=32 block: [2 B fp16 scale | 32 B int8 weights],
//      34 bytes/block, K must be a multiple of 32.
//   X: FP16 [N, K] row-major (caller stages F32 → F16 via convert_f32_to_f16).
//   Y: FP32 [N, M] row-major.
//
// Layout:
//   - Block: 128 threads (4 warps of 32 lanes).
//   - Output tile: 64 rows (M) × 64 cols (N/batch).
//   - Warp_id 0..3 owns M-rows [warp_id*16 .. (warp_id+1)*16].
//     Each warp produces 4 WMMA tiles along the N direction
//     (batches 0..15, 16..31, 32..47, 48..63 of the block).
//
// K loop:
//   For each 32-K block (one Q8 block):
//     - All 128 threads cooperatively load the X[64 batches × 32 K]
//       slice into LDS (one HFP16 = 2 B × 64 × 32 = 4 KB).
//     - Each warp loads its 16 weight rows' Q8 block (16 × 34 B = 544 B).
//     - Per warp: dequant int8 → fp16 inline, run 8 WMMA tiles
//       (2 K-tiles × 4 N-tiles = 8 mma per K-block per warp).
//
// LDS: 4 KB per block for X staging. Well under 64 KB cap.
//
// Grid: [ceil(M/64), ceil(N/64), 1]
// Block: [128, 1, 1]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE 64
#define N_TILE 64
#define WARPS  4
#define M_PER_WARP 16    // M_TILE / WARPS
#define N_TILES_PER_WARP 4 // N_TILE / 16

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_q8_0_wmma_4w(
    const char*     __restrict__ A_q8,    // [M, K/32 * 34]
    const _Float16* __restrict__ X,       // [N, K] FP16
    float*          __restrict__ Y,       // [N, M] FP32
    int M, int K, int N
) {
    const int block_m = blockIdx.x * M_TILE;
    const int block_n = blockIdx.y * N_TILE;

    if (block_m >= M || block_n >= N) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;             // 0..3
    const int lane    = tid & 31;             // 0..31
    const int lane16  = lane & 15;            // 0..15 (WMMA fragment lane)

    // Each warp owns M-rows [warp_id*16, warp_id*16 + 16).
    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row = warp_m_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    // 4 accumulators per warp (one per N-tile in the 64-col output).
    float8_t acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc2 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc3 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A_q8 + (long long)safe_row * row_stride;

    // LDS X stage: [64 batches × 32 K] = 64 × 32 × 2 B = 4 KB.
    // Layout: lds_x[n_in_tile * 32 + k_in_block]  (row-major over N).
    __shared__ _Float16 lds_x[N_TILE * 32];

    for (int bi = 0; bi < blocks_per_row; bi++) {
        // ── Cooperative X load: 128 threads load 64 batches × 32 K ints.
        // Total floats to load: 64 × 32 = 2048 fp16. 128 threads × 16 each.
        // Each thread loads 16 fp16 = one half16_t (32 bytes, aligned).
        //
        // Thread i covers a contiguous chunk of (n_row, k_off) defined by
        // a flat index 0..2047 stepping by 128 × 16 = 2048 per pass.
        // i.e. each thread loads exactly one half16_t.
        {
            const int flat = tid * 16;                  // 0, 16, 32, ..., 2032
            const int n_row = flat >> 5;                // 0..63
            const int k_off = flat & 31;                // 0 or 16
            const int gn = block_n + n_row;
            const int safe_gn = (gn < N) ? gn : (N - 1);
            const long long x_off = (long long)safe_gn * K + bi * 32 + k_off;
            half16_t v = *(const half16_t*)(X + x_off);
            *(half16_t*)(lds_x + n_row * 32 + k_off) = v;
        }
        __syncthreads();

        // ── Each warp: load its row's Q8 block (warp-private, 16 rows × 34 B).
        // Per-lane (lane16): load this lane's row's scale + 32 int8 weights.
        const char* bp = row_base + bi * 34;
        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        // WMMA tile 0 (K-elements 0..16) and tile 1 (K-elements 16..32).
        const signed char* w0 = (const signed char*)(bp + 2);
        const signed char* w1 = (const signed char*)(bp + 2 + 16);

        half16_t a0, a1;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

        // For each of the 4 N-tiles (batches 0..15, 16..31, 32..47, 48..63
        // within the block), load B fragment from LDS and run 2 WMMAs.
        // B fragment shape per RDNA3 wave32 WMMA spec: lane l holds 16 K
        // values from row l of the tile (n = block_n + tile_n*16 + l).
        #pragma unroll
        for (int n_tile = 0; n_tile < N_TILES_PER_WARP; n_tile++) {
            const int n_row_in_tile = n_tile * 16 + lane16;
            const _Float16* lds_row = lds_x + n_row_in_tile * 32;
            half16_t b0 = *(const half16_t*)(lds_row);
            half16_t b1 = *(const half16_t*)(lds_row + 16);

            if (n_tile == 0) {
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc0);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc0);
            } else if (n_tile == 1) {
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc1);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc1);
            } else if (n_tile == 2) {
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc2);
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc2);
            } else {
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc3);
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc3);
            }
        }
        __syncthreads();
    }

    // Output writes. Per RDNA3 wave32 WMMA: acc[j] on lane L holds
    // C[2*j + (L >> 4)][L & 15].
    // Each warp owns M-rows [warp_m_start, warp_m_start + 16); each acc
    // covers a 16-N-tile starting at block_n + n_tile_offset.
    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES_PER_WARP; n_tile++) {
        const int out_col = block_n + n_tile * 16 + lane16;
        if (out_col >= N) continue;
        float8_t acc;
        if      (n_tile == 0) acc = acc0;
        else if (n_tile == 1) acc = acc1;
        else if (n_tile == 2) acc = acc2;
        else                  acc = acc3;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = warp_m_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
