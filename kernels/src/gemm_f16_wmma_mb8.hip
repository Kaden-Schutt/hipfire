// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MB8 variant of gemm_f16_wmma_mb4: 8 subtiles per block (16×128 output panel)
//
// Same WMMA layout as MB4 (lane t holds row t&15 of A/B; lanes 16..31
// mirror 0..15), just double the register blocking over N. Each K-step
// loads the weight A-tile once from DRAM and applies it to EIGHT X
// B-subtiles — 2× fewer weight re-reads than MB4 at the cost of +8
// float8_t accumulators (~64 VGPRs).
//
// Expected VGPR pressure: MB4 uses ~60 VGPRs, MB8 uses ~90-120 VGPRs.
// At occupancy cap (LDS=0) this allows 8-10 waves per SIMD — ample.
//
// Grid: [ceil(M/16), ceil(N/128)]   Block: [32]   LDS: 0

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define NB 8

__launch_bounds__(32, 8)   // 8 waves/SIMD (vs MB4's 16 waves — higher register pressure)
extern "C" __global__ void gemm_f16_wmma_mb8(
    const _Float16* __restrict__ W,   // [M, K] row-major F16 (weights)
    const float*    __restrict__ X,   // [N, K] row-major F32 (input)
    float*          __restrict__ Y,   // [N, M] row-major F32 (transposed output)
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;            // M-tile origin
    const int col_start = blockIdx.y * (16 * NB);     // N-tile-group origin
    const int tid = threadIdx.x;

    if (row_start >= M || col_start >= N) return;

    const int my_a_row = row_start + (tid & 15);      // lane's A-row (M axis)
    const bool a_in_bounds = (my_a_row < M);

    float8_t acc[NB];
    #pragma unroll
    for (int nb = 0; nb < NB; nb++)
        acc[nb] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Load W A-tile ONCE (reused across all 8 subtiles).
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
            const int my_b_row = col_start + nb * 16 + (tid & 15);
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

    // Transposed output: acc[nb][j] at lane t → (M-row = 2*j+(t>>4),
    // N-col = col_start + nb*16 + (t&15)).
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int out_col = col_start + nb * 16 + (tid & 15);
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_row = row_start + 2 * j + (tid >> 4);
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[nb][j];
                }
            }
        }
    }
}
