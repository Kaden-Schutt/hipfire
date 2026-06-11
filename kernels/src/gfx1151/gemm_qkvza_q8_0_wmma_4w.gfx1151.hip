// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire -- see LICENSE and NOTICE in the project root.
//
// gfx1151 4-warp fused Q8_0 QKVZA GEMM for Qwen3.5/3.6 DeltaNet prefill.
// This is the fused-projection sibling of gemm_q8_0_wmma_4w: four warps
// cooperate on a 64-row x 64-token output tile, staging X in LDS so each
// packed Q8 weight block is reused across 64 batch positions.

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
extern "C" __global__ void gemm_qkvza_q8_0_wmma_4w_gfx1151(
    const char*     __restrict__ A_qkv,
    const char*     __restrict__ A_z,
    const char*     __restrict__ A_beta,
    const char*     __restrict__ A_alpha,
    const _Float16* __restrict__ X,         // [N, K] FP16
    float*          __restrict__ Y_qkv,     // [N, qkv_m]
    float*          __restrict__ Y_z,       // [N, z_m]
    float*          __restrict__ Y_beta,    // [N, beta_m]
    float*          __restrict__ Y_alpha,   // [N, alpha_m]
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int block_m = blockIdx.x * M_TILE;
    const int block_n = blockIdx.y * N_TILE;
    if (block_m >= total_m || block_n >= N) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int lane16  = lane & 15;

    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row = warp_m_start + lane16;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < qkv_m) {
        A = A_qkv;   local_row = safe_row;
    } else if (safe_row < qkv_m + z_m) {
        A = A_z;     local_row = safe_row - qkv_m;
    } else if (safe_row < qkv_m + z_m + beta_m) {
        A = A_beta;  local_row = safe_row - (qkv_m + z_m);
    } else {
        A = A_alpha; local_row = safe_row - (qkv_m + z_m + beta_m);
    }

    float8_t acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc2 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t acc3 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)local_row * row_stride;

    __shared__ _Float16 lds_x[N_TILE * 32];

    for (int bi = 0; bi < blocks_per_row; bi++) {
        {
            const int flat = tid * 16;
            const int n_row = flat >> 5;
            const int k_off = flat & 31;
            const int gn = block_n + n_row;
            const int safe_gn = (gn < N) ? gn : (N - 1);
            const long long x_off = (long long)safe_gn * K + bi * 32 + k_off;
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
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int proj_m;
                if (out_row < qkv_m) {
                    Y = Y_qkv;   proj_row = out_row;                       proj_m = qkv_m;
                } else if (out_row < qkv_m + z_m) {
                    Y = Y_z;     proj_row = out_row - qkv_m;                proj_m = z_m;
                } else if (out_row < qkv_m + z_m + beta_m) {
                    Y = Y_beta;  proj_row = out_row - (qkv_m + z_m);        proj_m = beta_m;
                } else {
                    Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m); proj_m = alpha_m;
                }
                Y[(long long)out_col * proj_m + proj_row] = acc[j];
            }
        }
    }
}
