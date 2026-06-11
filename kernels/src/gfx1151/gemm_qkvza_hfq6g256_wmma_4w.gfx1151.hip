// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.
//
// gfx1151 4-warp fused HFQ6-G256 QKVZA GEMM for Qwen3.5/3.6 prefill.
// Four warps cooperate on a 64-row x 64-token output tile. Each block
// stages one 64-token x 256-K HFQ6 group of X in LDS, then reuses each
// dequantized weight row across four 16-token WMMA tiles.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE 64
#define N_TILE 64
#define WARPS  4
#define M_PER_WARP 16
#define N_TILES_PER_WARP 4
#define GROUP_K 256

__launch_bounds__(128, 1)
extern "C" __global__ void gemm_qkvza_hfq6g256_wmma_4w_gfx1151(
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

    const int groups_per_row = K / GROUP_K;
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + (long long)local_row * row_stride;

    __shared__ _Float16 lds_x[N_TILE * GROUP_K];

    for (int g = 0; g < groups_per_row; g++) {
        #pragma unroll
        for (int pass = 0; pass < 8; pass++) {
            const int flat = pass * 2048 + tid * 16;
            const int n_row = flat >> 8;
            const int k_off = flat & 255;
            const int gn = block_n + n_row;
            const int safe_gn = (gn < N) ? gn : (N - 1);
            const long long x_off = (long long)safe_gn * K + g * GROUP_K + k_off;
            half16_t v = *(const half16_t*)(X + x_off);
            *(half16_t*)(lds_x + n_row * GROUP_K + k_off) = v;
        }
        __syncthreads();

        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h

        #pragma unroll
        for (int kt = 0; kt < 16; kt += 2) {
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + kt * 12);
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + (kt + 1) * 12);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d2a = *(const unsigned int*)(dp_a + 6);
            unsigned int d3a = *(const unsigned int*)(dp_a + 9);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d2b = *(const unsigned int*)(dp_b + 6);
            unsigned int d3b = *(const unsigned int*)(dp_b + 9);

            half16_t a_reg;
            DQ6(0,d0a,0); DQ6(1,d0a,6); DQ6(2,d0a,12); DQ6(3,d0a,18);
            DQ6(4,d1a,0); DQ6(5,d1a,6); DQ6(6,d1a,12); DQ6(7,d1a,18);
            DQ6(8,d2a,0); DQ6(9,d2a,6); DQ6(10,d2a,12); DQ6(11,d2a,18);
            DQ6(12,d3a,0); DQ6(13,d3a,6); DQ6(14,d3a,12); DQ6(15,d3a,18);

            #pragma unroll
            for (int n_tile = 0; n_tile < N_TILES_PER_WARP; n_tile++) {
                const int n_row_in_tile = n_tile * 16 + lane16;
                const _Float16* lds_row = lds_x + n_row_in_tile * GROUP_K + kt * 16;
                half16_t b = *(const half16_t*)(lds_row);
                if (n_tile == 0) {
                    acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc0);
                } else if (n_tile == 1) {
                    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc1);
                } else if (n_tile == 2) {
                    acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc2);
                } else {
                    acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc3);
                }
            }

            DQ6(0,d0b,0); DQ6(1,d0b,6); DQ6(2,d0b,12); DQ6(3,d0b,18);
            DQ6(4,d1b,0); DQ6(5,d1b,6); DQ6(6,d1b,12); DQ6(7,d1b,18);
            DQ6(8,d2b,0); DQ6(9,d2b,6); DQ6(10,d2b,12); DQ6(11,d2b,18);
            DQ6(12,d3b,0); DQ6(13,d3b,6); DQ6(14,d3b,12); DQ6(15,d3b,18);

            #pragma unroll
            for (int n_tile = 0; n_tile < N_TILES_PER_WARP; n_tile++) {
                const int n_row_in_tile = n_tile * 16 + lane16;
                const _Float16* lds_row = lds_x + n_row_in_tile * GROUP_K + (kt + 1) * 16;
                half16_t b = *(const half16_t*)(lds_row);
                if (n_tile == 0) {
                    acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc0);
                } else if (n_tile == 1) {
                    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc1);
                } else if (n_tile == 2) {
                    acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc2);
                } else {
                    acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b, acc3);
                }
            }
        }
        #undef DQ6
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
                    Y = Y_qkv;   proj_row = out_row;                         proj_m = qkv_m;
                } else if (out_row < qkv_m + z_m) {
                    Y = Y_z;     proj_row = out_row - qkv_m;                 proj_m = z_m;
                } else if (out_row < qkv_m + z_m + beta_m) {
                    Y = Y_beta;  proj_row = out_row - (qkv_m + z_m);         proj_m = beta_m;
                } else {
                    Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m); proj_m = alpha_m;
                }
                Y[(long long)out_col * proj_m + proj_row] = acc[j];
            }
        }
    }
}
