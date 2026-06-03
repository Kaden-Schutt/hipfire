// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Register-blocked WMMA F16×F32 GEMM for the vision encoder. gfx1100+ (RDNA3
// wave32 WMMA). Two wins over the naive `gemm_f16_wmma`:
//
//   1. **NB=4 register blocking over N.** Each block computes one 16-row
//      M-tile × FOUR 16-col N-subtiles (a 16×64 output panel). The W A-tile
//      (16×16) is loaded from DRAM ONCE per K-step and reused across all four
//      B-subtiles. The naive kernel reloads each W M-tile for every one of
//      N/16 N-tiles — at the vision shapes (M≈1536, N≈19520) the weight is
//      re-streamed ~1220× and the kernel runs at ~0.5% of WMMA peak
//      (DRAM-bound). NB=4 cuts that redundant weight traffic 4×.
//   2. **Transposed output.** Writes Y as [N, M] (batch-major) directly, so
//      the caller drops the separate `transpose_f32` fixup (one per GEMM,
//      ~4s of the vision wall).
//
// Computes (math): Y[n][m] = sum_k W_f16[m][k] * X_f32[n][k]
//   W is row-major F16 [M, K]   (weight)
//   X is row-major F32 [N, K]   (batch of input vectors, cast F32→F16 lane-side)
//   Y is row-major F32 [N, M]   (output, TRANSPOSED vs gemm_f16_wmma's [M,N])
//
// WMMA register layout is identical to gemm_f16_wmma (lane t holds row t of
// the 16×16 operand; lanes 16..31 mirror 0..15; accumulator acc[j] at lane t
// maps to A-row 2*j+(t>>4), B-row t&15). See that kernel's header for the
// RDNA3 wave32 derivation.
//
// Grid: [ceil(M/16), ceil(N/64)]   Block: [32]   LDS: 0

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define NB 4

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_f16_wmma_mb4(
    const _Float16* __restrict__ W,   // [M, K] row-major, F16
    const float*    __restrict__ X,   // [N, K] row-major, F32
    float*          __restrict__ Y,   // [N, M] row-major, F32  (transposed)
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;            // M-tile origin
    const int col_start = blockIdx.y * (16 * NB);     // N-tile-group origin
    const int tid = threadIdx.x;

    if (row_start >= M || col_start >= N) return;

    const int my_a_row = row_start + (tid & 15);      // this lane's W row (M axis)
    const bool a_in_bounds = (my_a_row < M);

    float8_t acc[NB];
    #pragma unroll
    for (int nb = 0; nb < NB; nb++)
        acc[nb] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Load the W A-tile ONCE for this K-step (reused across all NB subtiles).
        half16_t a_reg;
        if (a_in_bounds) {
            const _Float16* src = W + (long long)my_a_row * K + k0;
            #pragma unroll
            for (int j = 0; j < 16; j++)
                a_reg[j] = (k0 + j < K) ? src[j] : (_Float16)0.0f;
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j++) a_reg[j] = (_Float16)0.0f;
        }

        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            const int my_b_row = col_start + nb * 16 + (tid & 15);  // X row (N axis)
            half16_t b_reg;
            if (my_b_row < N) {
                const float* src = X + (long long)my_b_row * K + k0;
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    b_reg[j] = (k0 + j < K) ? (_Float16)src[j] : (_Float16)0.0f;
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) b_reg[j] = (_Float16)0.0f;
            }
            acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[nb]);
        }
    }

    // Output, transposed [N, M]: acc[nb][j] at lane t → (M-row = 2*j+(t>>4),
    // N-col = col_start + nb*16 + (t&15)). Write Y[N-col * M + M-row].
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int out_col = col_start + nb * 16 + (tid & 15);   // N position
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_row = row_start + 2 * j + (tid >> 4);    // M position
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[nb][j];
                }
            }
        }
    }
}
