// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Non-grouped HFQ4-G256 x Q8_1 i8-WMMA MMQ for gfx1151.
//
// This is the G256 sibling of gemm_hfq4g128_mmq.gfx1151.hip: one wave32
// computes a 16-row x 16-token tile with unsigned i8 WMMA. It avoids the
// generic RDNA2-style LDS-tiled sdot4 MMQ path on Strix Halo for aligned
// Qwen3.5/3.6 prefill set/residual projections.

#if defined(__gfx1151__)

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(16))) int8x16_t;
typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

template <bool ADD>
__device__ __forceinline__ void gemm_hfq4g256_mmq_body(
    const char*           __restrict__ A,        // [M x (K/256) x 136]
    const block_q8_1_mmq* __restrict__ X_q8_src, // [K/128 x N]
    float*                __restrict__ Y,        // [N x M]
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int slot_start = blockIdx.y * 16;
    if (row_start >= M || slot_start >= N) return;

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    const int x_row = (slot_idx < N) ? slot_idx : -1;

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        float d_x[8];
        float sum_x[8];
        #pragma unroll
        for (int sb = 0; sb < 8; sb++) {
            const int kb_q8 = 2 * g + (sb >> 2);
            const int ds = sb & 3;
            float2 dm = make_float2(0.0f, 0.0f);
            if (x_row >= 0) {
                const block_q8_1_mmq* xb = X_q8_src + (long long)kb_q8 * N + x_row;
                dm = __half22float2(xb->ds4[ds]);
            }
            d_x[sb] = dm.x;
            sum_x[sb] = dm.y;
        }

        int32x4_t a_i32[16];
        int32x4_t b_i32[16];

        #pragma unroll
        for (int kt_idx = 0; kt_idx < 16; kt_idx++) {
            const int sb = kt_idx >> 1;
            const int kt = kt_idx & 1;
            const int k_off = sb * 32 + kt * 16;

            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);

            int8x16_t a_vec;
            a_vec[0]  = (int8_t)((pk0      ) & 0xFu);
            a_vec[1]  = (int8_t)((pk0 >>  4) & 0xFu);
            a_vec[2]  = (int8_t)((pk0 >>  8) & 0xFu);
            a_vec[3]  = (int8_t)((pk0 >> 12) & 0xFu);
            a_vec[4]  = (int8_t)((pk0 >> 16) & 0xFu);
            a_vec[5]  = (int8_t)((pk0 >> 20) & 0xFu);
            a_vec[6]  = (int8_t)((pk0 >> 24) & 0xFu);
            a_vec[7]  = (int8_t)((pk0 >> 28) & 0xFu);
            a_vec[8]  = (int8_t)((pk1      ) & 0xFu);
            a_vec[9]  = (int8_t)((pk1 >>  4) & 0xFu);
            a_vec[10] = (int8_t)((pk1 >>  8) & 0xFu);
            a_vec[11] = (int8_t)((pk1 >> 12) & 0xFu);
            a_vec[12] = (int8_t)((pk1 >> 16) & 0xFu);
            a_vec[13] = (int8_t)((pk1 >> 20) & 0xFu);
            a_vec[14] = (int8_t)((pk1 >> 24) & 0xFu);
            a_vec[15] = (int8_t)((pk1 >> 28) & 0xFu);

            int8x16_t b_vec;
            if (x_row >= 0) {
                const int kb_q8 = 2 * g + (sb >> 2);
                const int ds = sb & 3;
                const block_q8_1_mmq* xb = X_q8_src + (long long)kb_q8 * N + x_row;
                b_vec = *(const int8x16_t*)(xb->qs + ds * 32 + kt * 16);
            } else {
                b_vec = (int8x16_t){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            }

            a_i32[kt_idx] = *(const int32x4_t*)&a_vec;
            b_i32[kt_idx] = *(const int32x4_t*)&b_vec;
        }

        int32x8_t cacc0 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc1 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc2 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc3 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc4 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc5 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc6 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t cacc7 = {0, 0, 0, 0, 0, 0, 0, 0};

        cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[0],  true, b_i32[0],  cacc0, true);
        cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[1],  true, b_i32[1],  cacc0, true);
        cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[2],  true, b_i32[2],  cacc1, true);
        cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[3],  true, b_i32[3],  cacc1, true);
        cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[4],  true, b_i32[4],  cacc2, true);
        cacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[5],  true, b_i32[5],  cacc2, true);
        cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[6],  true, b_i32[6],  cacc3, true);
        cacc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[7],  true, b_i32[7],  cacc3, true);
        cacc4 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[8],  true, b_i32[8],  cacc4, true);
        cacc4 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[9],  true, b_i32[9],  cacc4, true);
        cacc5 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[10], true, b_i32[10], cacc5, true);
        cacc5 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[11], true, b_i32[11], cacc5, true);
        cacc6 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[12], true, b_i32[12], cacc6, true);
        cacc6 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[13], true, b_i32[13], cacc6, true);
        cacc7 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[14], true, b_i32[14], cacc7, true);
        cacc7 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_i32[15], true, b_i32[15], cacc7, true);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[j] += sc_row[j] * d_x[0] * (float)cacc0[j] + zp_row[j] * sum_x[0];
            acc[j] += sc_row[j] * d_x[1] * (float)cacc1[j] + zp_row[j] * sum_x[1];
            acc[j] += sc_row[j] * d_x[2] * (float)cacc2[j] + zp_row[j] * sum_x[2];
            acc[j] += sc_row[j] * d_x[3] * (float)cacc3[j] + zp_row[j] * sum_x[3];
            acc[j] += sc_row[j] * d_x[4] * (float)cacc4[j] + zp_row[j] * sum_x[4];
            acc[j] += sc_row[j] * d_x[5] * (float)cacc5[j] + zp_row[j] * sum_x[5];
            acc[j] += sc_row[j] * d_x[6] * (float)cacc6[j] + zp_row[j] * sum_x[6];
            acc[j] += sc_row[j] * d_x[7] * (float)cacc7[j] + zp_row[j] * sum_x[7];
        }
    }

    const int out_col = slot_start + lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                float* yp = Y + (long long)out_col * M + out_row;
                if constexpr (ADD) {
                    *yp += acc[j];
                } else {
                    *yp = acc[j];
                }
            }
        }
    }
}

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_mmq_gfx1151(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ X_q8_src,
    float* __restrict__ Y,
    int M, int K, int N, int add
) {
    if (add) {
        gemm_hfq4g256_mmq_body<true>(A, X_q8_src, Y, M, K, N);
    } else {
        gemm_hfq4g256_mmq_body<false>(A, X_q8_src, Y, M, K, N);
    }
}

#else

extern "C" __global__ void gemm_hfq4g256_mmq_gfx1151(
    const char*, const void*, float*, int, int, int, int
) {}

#endif
