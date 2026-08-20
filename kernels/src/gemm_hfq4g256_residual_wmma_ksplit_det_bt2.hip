// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batch-tiled B=2 variant of the deterministic K-split residual GEMM.
// 2 independent acc chains reuse weight loads across 2 batch (N) tiles
// per block, halving the N-grid. Weights+dequant loaded once per K-tile,
// reused across both N-tiles. Each chain writes its own partials.
//
// Grid: [ceil(M/16), ceil(N/32), K_SPLITS]. Block: [32]. LDS: 0.
// K_SPLITS MUST match gemm_ksplit_det_finalize.hip.
//
// Derived from gemm_hfq4g256_residual_wmma_ksplit_det.hip (Apache-2.0).

#define K_SPLITS 4

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_residual_wmma_ksplit_det_bt2(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ partials,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 32;
    const int ksplit = blockIdx.z;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);

    const int groups_per_row = K / 256;
    const int groups_per_split = groups_per_row / K_SPLITS;
    const int g_start = ksplit * groups_per_split;
    const int g_end   = (ksplit == K_SPLITS - 1) ? groups_per_row
                                                 : g_start + groups_per_split;

    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Two batch indices
    int sb0 = batch_start + (tid & 15);
    int sb1 = batch_start + 16 + (tid & 15);
    if (sb0 >= batch_size) sb0 = 0;
    if (sb1 >= batch_size) sb1 = 0;

    const _Float16* xg0_base = X + (long long)sb0 * K;
    const _Float16* xg1_base = X + (long long)sb1 * K;

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = g_start; g < g_end; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg0 = xg0_base + g * 256;
        const _Float16* xg1 = xg1_base + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b0_a = *(const half16_t*)(xg0 + k_off_a);
            half16_t b0_b = *(const half16_t*)(xg0 + k_off_b);
            half16_t b1_a = *(const half16_t*)(xg1 + k_off_a);
            half16_t b1_b = *(const half16_t*)(xg1 + k_off_b);

            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b0_a, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b1_a, acc1);

            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b0_b, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b1_b, acc1);
        }
    }

    // Write partials for both batch tiles
    const long long split_base = (long long)ksplit * batch_size * M;
    const int out_col0 = batch_start + (tid & 15);
    const int out_col1 = batch_start + 16 + (tid & 15);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < M) {
            if (out_col0 < batch_size)
                partials[split_base + (long long)out_col0 * M + out_row] = acc0[j];
            if (out_col1 < batch_size)
                partials[split_base + (long long)out_col1 * M + out_row] = acc1[j];
        }
    }
}
