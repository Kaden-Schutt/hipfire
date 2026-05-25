// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA GEMM: K2 pipelining + 32-row block (vs 16 in k2).
//
// Processes 32 rows × 16 batches per block. Each wave32 computes two 16-row
// output tiles by issuing two WMMAs per K-tile against the SAME batch X
// fragment. Halves the block count for huge-M lm_head (M=248K) and amortizes
// X loads across two weight rows.
//
// Shape: W [M,K] row-major (HFQ4G256), X [N,K] FP16 row-major, Y [N,M] FP32.
//
// Grid: [ceil(M/32), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_residual_wmma_k2x32(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start0 = blockIdx.x * 32;
    const int row_start1 = row_start0 + 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start0 >= M || batch_start >= batch_size) return;

    const int lane = tid & 15;
    // Safe-row clamps each thread's row index into the valid [0,M) range so
    // the hipMemcpy-free tail handling (read from M-1) is safe for both
    // tile groups even when the block straddles M boundary.
    const int safe_row0 = (row_start0 + lane < M) ? (row_start0 + lane) : (M - 1);
    const int safe_row1 = (row_start1 + lane < M) ? (row_start1 + lane) : (M - 1);
    const int safe_batch = (batch_start + lane < batch_size) ? (batch_start + lane) : 0;

    const int groups_per_row = K / 256;
    const char* row_base0 = A + (long long)safe_row0 * groups_per_row * 136;
    const char* row_base1 = A + (long long)safe_row1 * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp0 = row_base0 + g * 136;
        const char* gp1 = row_base1 + g * 136;
        _Float16 sc0_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0));
        _Float16 zp0_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        _Float16 sc1_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1));
        _Float16 zp1_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        const _Float16* xg = x_base + g * 256;

        // Process 2 K-tiles per iteration (8 iterations covers 16 tiles)
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // Shared X loads (one batch tile, used by both weight rows)
            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            // Row group 0, tile A
            unsigned int pk0_0a = *(const unsigned int*)(gp0 + 8 + k_off_a / 2);
            unsigned int pk1_0a = *(const unsigned int*)(gp0 + 8 + k_off_a / 2 + 4);
            // Row group 1, tile A
            unsigned int pk0_1a = *(const unsigned int*)(gp1 + 8 + k_off_a / 2);
            unsigned int pk1_1a = *(const unsigned int*)(gp1 + 8 + k_off_a / 2 + 4);
            // Row group 0, tile B
            unsigned int pk0_0b = *(const unsigned int*)(gp0 + 8 + k_off_b / 2);
            unsigned int pk1_0b = *(const unsigned int*)(gp0 + 8 + k_off_b / 2 + 4);
            // Row group 1, tile B
            unsigned int pk0_1b = *(const unsigned int*)(gp1 + 8 + k_off_b / 2);
            unsigned int pk1_1b = *(const unsigned int*)(gp1 + 8 + k_off_b / 2 + 4);

            half16_t a;
            #define DQ(reg, i, pk, sh, sc, zp) reg[i] = (sc) * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + (zp)

            // ── Tile A ─────────────────────────────────────────────────
            // Row group 0
            DQ(a, 0,  pk0_0a,  0, sc0_h, zp0_h); DQ(a, 1,  pk0_0a,  4, sc0_h, zp0_h);
            DQ(a, 2,  pk0_0a,  8, sc0_h, zp0_h); DQ(a, 3,  pk0_0a, 12, sc0_h, zp0_h);
            DQ(a, 4,  pk0_0a, 16, sc0_h, zp0_h); DQ(a, 5,  pk0_0a, 20, sc0_h, zp0_h);
            DQ(a, 6,  pk0_0a, 24, sc0_h, zp0_h); DQ(a, 7,  pk0_0a, 28, sc0_h, zp0_h);
            DQ(a, 8,  pk1_0a,  0, sc0_h, zp0_h); DQ(a, 9,  pk1_0a,  4, sc0_h, zp0_h);
            DQ(a, 10, pk1_0a,  8, sc0_h, zp0_h); DQ(a, 11, pk1_0a, 12, sc0_h, zp0_h);
            DQ(a, 12, pk1_0a, 16, sc0_h, zp0_h); DQ(a, 13, pk1_0a, 20, sc0_h, zp0_h);
            DQ(a, 14, pk1_0a, 24, sc0_h, zp0_h); DQ(a, 15, pk1_0a, 28, sc0_h, zp0_h);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_a, acc0);

            // Row group 1
            DQ(a, 0,  pk0_1a,  0, sc1_h, zp1_h); DQ(a, 1,  pk0_1a,  4, sc1_h, zp1_h);
            DQ(a, 2,  pk0_1a,  8, sc1_h, zp1_h); DQ(a, 3,  pk0_1a, 12, sc1_h, zp1_h);
            DQ(a, 4,  pk0_1a, 16, sc1_h, zp1_h); DQ(a, 5,  pk0_1a, 20, sc1_h, zp1_h);
            DQ(a, 6,  pk0_1a, 24, sc1_h, zp1_h); DQ(a, 7,  pk0_1a, 28, sc1_h, zp1_h);
            DQ(a, 8,  pk1_1a,  0, sc1_h, zp1_h); DQ(a, 9,  pk1_1a,  4, sc1_h, zp1_h);
            DQ(a, 10, pk1_1a,  8, sc1_h, zp1_h); DQ(a, 11, pk1_1a, 12, sc1_h, zp1_h);
            DQ(a, 12, pk1_1a, 16, sc1_h, zp1_h); DQ(a, 13, pk1_1a, 20, sc1_h, zp1_h);
            DQ(a, 14, pk1_1a, 24, sc1_h, zp1_h); DQ(a, 15, pk1_1a, 28, sc1_h, zp1_h);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_a, acc1);

            // ── Tile B ─────────────────────────────────────────────────
            // Row group 0
            DQ(a, 0,  pk0_0b,  0, sc0_h, zp0_h); DQ(a, 1,  pk0_0b,  4, sc0_h, zp0_h);
            DQ(a, 2,  pk0_0b,  8, sc0_h, zp0_h); DQ(a, 3,  pk0_0b, 12, sc0_h, zp0_h);
            DQ(a, 4,  pk0_0b, 16, sc0_h, zp0_h); DQ(a, 5,  pk0_0b, 20, sc0_h, zp0_h);
            DQ(a, 6,  pk0_0b, 24, sc0_h, zp0_h); DQ(a, 7,  pk0_0b, 28, sc0_h, zp0_h);
            DQ(a, 8,  pk1_0b,  0, sc0_h, zp0_h); DQ(a, 9,  pk1_0b,  4, sc0_h, zp0_h);
            DQ(a, 10, pk1_0b,  8, sc0_h, zp0_h); DQ(a, 11, pk1_0b, 12, sc0_h, zp0_h);
            DQ(a, 12, pk1_0b, 16, sc0_h, zp0_h); DQ(a, 13, pk1_0b, 20, sc0_h, zp0_h);
            DQ(a, 14, pk1_0b, 24, sc0_h, zp0_h); DQ(a, 15, pk1_0b, 28, sc0_h, zp0_h);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_b, acc0);

            // Row group 1
            DQ(a, 0,  pk0_1b,  0, sc1_h, zp1_h); DQ(a, 1,  pk0_1b,  4, sc1_h, zp1_h);
            DQ(a, 2,  pk0_1b,  8, sc1_h, zp1_h); DQ(a, 3,  pk0_1b, 12, sc1_h, zp1_h);
            DQ(a, 4,  pk0_1b, 16, sc1_h, zp1_h); DQ(a, 5,  pk0_1b, 20, sc1_h, zp1_h);
            DQ(a, 6,  pk0_1b, 24, sc1_h, zp1_h); DQ(a, 7,  pk0_1b, 28, sc1_h, zp1_h);
            DQ(a, 8,  pk1_1b,  0, sc1_h, zp1_h); DQ(a, 9,  pk1_1b,  4, sc1_h, zp1_h);
            DQ(a, 10, pk1_1b,  8, sc1_h, zp1_h); DQ(a, 11, pk1_1b, 12, sc1_h, zp1_h);
            DQ(a, 12, pk1_1b, 16, sc1_h, zp1_h); DQ(a, 13, pk1_1b, 20, sc1_h, zp1_h);
            DQ(a, 14, pk1_1b, 24, sc1_h, zp1_h); DQ(a, 15, pk1_1b, 28, sc1_h, zp1_h);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_b, acc1);
            #undef DQ
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // acc0 → rows [row_start0, row_start0+16), acc1 → rows [row_start1, row_start1+16).
    const int out_col = batch_start + lane;
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int dr = 2 * j + (tid >> 4);
            const int r0 = row_start0 + dr;
            const int r1 = row_start1 + dr;
            if (r0 < M) Y[(long long)out_col * M + r0] += acc0[j];
            if (r1 < M) Y[(long long)out_col * M + r1] += acc1[j];
        }
    }
}
