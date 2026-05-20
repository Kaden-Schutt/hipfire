// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA GEMM v5: Multi-wave cooperative block.
//
// 64 threads = 2 waves per block. Each wave computes a different 16×16 output
// tile (wave 0: rows 0-15, wave 1: rows 16-31). Both waves share the same
// batch tile (16 batch elements).
//
// Latency hiding: wave 0 loads its weights while wave 1 executes WMMA (and
// vice versa). The GPU scheduler interleaves the two waves' instructions on
// the same SIMD, hiding memory stalls behind compute from the other wave.
//
// Weight loading: each wave loads its own 16 rows independently (no LDS
// sharing needed — the rows are different). The latency hiding comes from
// the hardware scheduler, not explicit synchronization.
//
// B (X) loading: both waves load the same batch elements but at different
// K offsets within the same iteration — no conflict, L1-cached after first
// wave loads it.
//
// Grid: [ceil(M/32), ceil(N/16)]. Block: [64]. LDS: 0.
// Each block covers 32 output rows × 16 batch elements.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(64, 1)
extern "C" __global__ void gemm_hfq4g256_residual_wmma2(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;           // 0 or 1
    const int lane = tid & 31;              // 0..31 within wave
    const int row_start = blockIdx.x * 32 + wave_id * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (lane & 15) < M)
                         ? (row_start + (lane & 15)) : (M - 1);
    const int safe_batch = (batch_start + (lane & 15) < batch_size)
                           ? (batch_start + (lane & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;

            unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ

            half16_t b_reg = *(const half16_t*)(xg + k_off);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    // Output: each wave writes its own 16×16 tile
    const int out_row = row_start + (lane & 15);
    const int out_batch_base = batch_start + (lane >> 4) * 8;
    if (out_row < M) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int bidx = out_batch_base + j;
            if (bidx < batch_size)
                Y[(long long)bidx * M + out_row] += acc[j];
        }
    }
}
