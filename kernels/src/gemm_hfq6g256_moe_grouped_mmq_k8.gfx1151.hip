// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// i8 WMMA MMQ MoE grouped-GEMM for gfx1151 HFQ6/MQ6 experts. Mirrors the
// HFQ4 K8 kernel but decodes 6-bit weight tiles from the 200 B/group layout.

#if defined(__gfx1151__)

#define QK8_1 32

struct block_q8_1_mmq_ds8 {
    half2 ds8[8];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq_ds8) == 160, "bad block_q8_1_mmq_ds8 size");

typedef int8_t  __attribute__((ext_vector_type(16))) int8x16_t;
typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_mmq_k8_gfx1151(
    const unsigned long long*  __restrict__ expert_weight_ptrs,
    const int*                 __restrict__ expert_tile_ids,
    const int*                 __restrict__ sorted_slot_index,
    const block_q8_1_mmq_ds8*  __restrict__ X_q8_src,
    float*                     __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int blocks128 = K / 128;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            const block_q8_1_mmq_ds8* xb = (x_row >= 0)
                ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
                : nullptr;

            #pragma unroll
            for (int kt_idx = 0; kt_idx < 8; kt_idx++) {
                const int tile = kb_in_group * 8 + kt_idx;
                const int byte_off = tile * 12;

                const unsigned char* dp = (const unsigned char*)(gp + 8 + byte_off);
                unsigned int d0 = *(const unsigned int*)(dp);
                unsigned int d1 = *(const unsigned int*)(dp + 3);
                unsigned int d2 = *(const unsigned int*)(dp + 6);
                unsigned int d3 = *(const unsigned int*)(dp + 9);

                int8x16_t a_vec;
                a_vec[0]  = (int8_t)((d0 >> 0)  & 63u);
                a_vec[1]  = (int8_t)((d0 >> 6)  & 63u);
                a_vec[2]  = (int8_t)((d0 >> 12) & 63u);
                a_vec[3]  = (int8_t)((d0 >> 18) & 63u);
                a_vec[4]  = (int8_t)((d1 >> 0)  & 63u);
                a_vec[5]  = (int8_t)((d1 >> 6)  & 63u);
                a_vec[6]  = (int8_t)((d1 >> 12) & 63u);
                a_vec[7]  = (int8_t)((d1 >> 18) & 63u);
                a_vec[8]  = (int8_t)((d2 >> 0)  & 63u);
                a_vec[9]  = (int8_t)((d2 >> 6)  & 63u);
                a_vec[10] = (int8_t)((d2 >> 12) & 63u);
                a_vec[11] = (int8_t)((d2 >> 18) & 63u);
                a_vec[12] = (int8_t)((d3 >> 0)  & 63u);
                a_vec[13] = (int8_t)((d3 >> 6)  & 63u);
                a_vec[14] = (int8_t)((d3 >> 12) & 63u);
                a_vec[15] = (int8_t)((d3 >> 18) & 63u);

                float sum_w_self = 0.0f;
                #pragma unroll
                for (int t = 0; t < 16; t++) {
                    sum_w_self += (float)a_vec[t];
                }
                float sum_w_row[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int src_lane = 2 * j + (tid >> 4);
                    sum_w_row[j] = __shfl(sum_w_self, src_lane);
                }

                int8x16_t b_vec;
                if (xb != nullptr) {
                    const int8_t* qs_p = xb->qs + kt_idx * 16;
                    b_vec = *(const int8x16_t*)qs_p;
                } else {
                    b_vec = (int8x16_t){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                }

                int32x4_t a_i32 = *(const int32x4_t*)&a_vec;
                int32x4_t b_i32 = *(const int32x4_t*)&b_vec;
                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};
                cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, a_i32, true, b_i32, cacc, true
                );

                float d_x = 0.0f;
                float mean_x = 0.0f;
                if (xb != nullptr) {
                    float2 dm_f = __half22float2(xb->ds8[kt_idx]);
                    d_x = dm_f.x;
                    mean_x = dm_f.y;
                }

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[j] += sc_row[j] * (d_x * (float)cacc[j] + mean_x * sum_w_row[j])
                            + zp_row[j] * (mean_x * 16.0f);
                }
            }
        }
    }

    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_hfq6g256_moe_grouped_mmq_k8_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
