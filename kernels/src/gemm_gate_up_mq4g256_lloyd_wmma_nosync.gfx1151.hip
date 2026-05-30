// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Barrier-free variant of gemm_gate_up_mq4g256_lloyd_wmma — gfx1151 K4 variant.
// Each lane loads its own row's codebook from global, eliminating both
// __syncthreads() calls.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_mq4g256_lloyd_wmma_nosync(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_tile_row = tid & 15;
    const int my_row = row_start + my_tile_row;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate; local_row = safe_row;
    } else {
        A = A_up;   local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + my_tile_row < N)
                           ? (batch_start + my_tile_row) : 0;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        // Each lane loads its own row's codebook directly from global.
        const __half* cb_h = (const __half*)(row_base + g * 160);
        half16_t cb = *(const half16_t*)cb_h;

        const char* gp = row_base + g * 160;
        const unsigned char* dp = (const unsigned char*)(gp + 32);
        const _Float16* xg = X + (long long)safe_batch * K + g * 256;

        for (int kt = 0; kt < 16; kt += 4) {
            const unsigned char* dp4 = dp + kt * 8;

            unsigned int pk0a = *(const unsigned int*)dp4;
            unsigned int pk1a = *(const unsigned int*)(dp4 + 4);
            unsigned int pk0b = *(const unsigned int*)(dp4 + 8);
            unsigned int pk1b = *(const unsigned int*)(dp4 + 12);
            unsigned int pk0c = *(const unsigned int*)(dp4 + 16);
            unsigned int pk1c = *(const unsigned int*)(dp4 + 20);
            unsigned int pk0d = *(const unsigned int*)(dp4 + 24);
            unsigned int pk1d = *(const unsigned int*)(dp4 + 28);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = cb[(((pk) >> (sh)) & 0xFu)]

            // === Tile kt+0 ===
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);
            half16_t b_a = *(const half16_t*)(xg + kt * 16);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            // === Tile kt+1 ===
            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            half16_t b_b = *(const half16_t*)(xg + (kt + 1) * 16);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);

            // === Tile kt+2 ===
            DQ(0,  pk0c,  0); DQ(1,  pk0c,  4); DQ(2,  pk0c,  8); DQ(3,  pk0c, 12);
            DQ(4,  pk0c, 16); DQ(5,  pk0c, 20); DQ(6,  pk0c, 24); DQ(7,  pk0c, 28);
            DQ(8,  pk1c,  0); DQ(9,  pk1c,  4); DQ(10, pk1c,  8); DQ(11, pk1c, 12);
            DQ(12, pk1c, 16); DQ(13, pk1c, 20); DQ(14, pk1c, 24); DQ(15, pk1c, 28);
            half16_t b_c = *(const half16_t*)(xg + (kt + 2) * 16);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_c, acc);

            // === Tile kt+3 ===
            DQ(0,  pk0d,  0); DQ(1,  pk0d,  4); DQ(2,  pk0d,  8); DQ(3,  pk0d, 12);
            DQ(4,  pk0d, 16); DQ(5,  pk0d, 20); DQ(6,  pk0d, 24); DQ(7,  pk0d, 28);
            DQ(8,  pk1d,  0); DQ(9,  pk1d,  4); DQ(10, pk1d,  8); DQ(11, pk1d, 12);
            DQ(12, pk1d, 16); DQ(13, pk1d, 20); DQ(14, pk1d, 24); DQ(15, pk1d, 28);
            #undef DQ
            half16_t b_d = *(const half16_t*)(xg + (kt + 3) * 16);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_d, acc);
        }
    }

    const int out_col = batch_start + my_tile_row;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate; proj_row = out_row;            out_stride = gate_m;
                } else {
                    Y = Y_up;   proj_row = out_row - gate_m;   out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}

#else

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_mq4g256_lloyd_wmma_nosync(
    const char*, const char*, const _Float16*, float*, float*,
    int, int, int, int) {}

#endif
