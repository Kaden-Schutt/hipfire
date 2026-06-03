// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// DETERMINISTIC K-split, phase 1 (partial):
// NOTE: K_SPLITS here MUST match K_SPLITS in gemm_ksplit_det_finalize.hip. identical compute to
// gemm_hfq4g256_residual_wmma_ksplit, but instead of atomicAdd'ing each
// K-split's partial into Y (race-ordered, non-deterministic fp accum),
// it writes each partial to a SEPARATE scratch slice partials[ksplit].
// A second kernel (gemm_ksplit_det_finalize) then sums the K_SPLITS
// partials + residual in FIXED index order → bit-reproducible.
//
// Keeps ksplit's grid.z occupancy win (4× blocks → ~13/CU on gfx1100's
// 96 CUs, vs 3.3/CU for the un-split k2 grid) — the perf edge is the
// occupancy, NOT the atomicAdd, so determinism is recovered at ~zero
// throughput cost (plain store < atomicAdd RMW; +one tiny finalize pass).
//
// partials layout: [K_SPLITS][batch_size][M], mirroring Y's
// [batch_size][M] (index out_col*M + out_row). Each valid cell is written
// exactly once per ksplit; finalize reads all K_SPLITS in fixed order.
//
// Grid: [ceil(M/16), ceil(N/16), K_SPLITS]. Block: [32]. LDS: 0.

#define K_SPLITS 4

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_residual_wmma_ksplit_det(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ partials,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int ksplit = blockIdx.z;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const int groups_per_split = groups_per_row / K_SPLITS;
    const int g_start = ksplit * groups_per_split;
    const int g_end   = (ksplit == K_SPLITS - 1) ? groups_per_row
                                                 : g_start + groups_per_split;

    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = g_start; g < g_end; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // Write this ksplit's partial to its OWN slice (plain store, no atomic,
    // no residual — finalize adds residual + all partials in fixed order).
    const int out_col = batch_start + (tid & 15);
    if (out_col < batch_size) {
        const long long split_base = (long long)ksplit * batch_size * M;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                partials[split_base + (long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
