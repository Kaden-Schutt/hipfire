// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 per-group O-LoRA batched GEMM — Q8_0 weights, gfx11 WMMA.
//
// Computes the block-diagonal WO projection:
//   wo_a[g, M, K] @ x[B, g, K] -> y[B, g, M]
//
// This is the strided-layout sibling of `gemm_q8_0_wmma_4w`: X and Y keep
// the production `[B, G, *]` layout, while grid.z selects the O-group.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE 64
#define N_TILE 64
#define WARPS  4
#define M_PER_WARP 16
#define N_TILES_PER_WARP 4

__launch_bounds__(128, 2)
extern "C" __global__ void wo_per_group_batched_q8_0_wmma_4w(
    const char*     __restrict__ A_q8,    // [G, M, K/32*34]
    const _Float16* __restrict__ X,       // [B, G, K] FP16
    float*          __restrict__ Y,       // [B, G, M] FP32
    int G, int M, int K, int N
) {
    const int group = blockIdx.z;
    const int block_m = blockIdx.x * M_TILE;
    const int block_n = blockIdx.y * N_TILE;

    if (group >= G || block_m >= M || block_n >= N) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int lane16  = lane & 15;

    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row = warp_m_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    float8_t acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc2 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc3 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const long long group_stride = (long long)M * row_stride;
    const char* row_base = A_q8 + (long long)group * group_stride + (long long)safe_row * row_stride;

    __shared__ _Float16 lds_x[N_TILE * 32];

    for (int bi = 0; bi < blocks_per_row; bi++) {
        {
            const int flat = tid * 16;
            const int n_row = flat >> 5;
            const int k_off = flat & 31;
            const int gn = block_n + n_row;
            const int safe_gn = (gn < N) ? gn : (N - 1);
            const long long x_off =
                ((long long)safe_gn * G + group) * K + bi * 32 + k_off;
            half16_t v = *(const half16_t*)(X + x_off);
            *(half16_t*)(lds_x + n_row * 32 + k_off) = v;
        }
        __syncthreads();

        const char* bp = row_base + bi * 34;
        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const signed char* w0 = (const signed char*)(bp + 2);
        const signed char* w1 = (const signed char*)(bp + 2 + 16);

        half16_t a0, a1;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

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
                Y[((long long)out_col * G + group) * M + out_row] = acc[j];
            }
        }
    }
}
