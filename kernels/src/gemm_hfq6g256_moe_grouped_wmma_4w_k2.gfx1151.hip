// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// 4-warp gfx1151 HFQ6/MQ6 grouped-MoE WMMA. Same contract as the HFQ4
// 4w sibling, with 200-byte HFQ6/MQ6 groups.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE         64
#define N_TILE         16
#define WARPS          4
#define M_PER_WARP     16
#define K_GROUP        256
#define SLOTS_PER_WARP 4

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_4w_k2_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int block_m = blockIdx.x * M_TILE;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * N_TILE;

    if (block_m >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;

    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row = warp_m_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    __shared__ int slot_x_row[N_TILE];
    if (tid < N_TILE) {
        const int slot_idx = slot_start + tid;
        int x_row = -1;
        if (slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0) {
                x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
            }
        }
        slot_x_row[tid] = x_row;
    }

    __shared__ _Float16 lds_x[N_TILE * K_GROUP];

    const int groups_per_row = K / K_GROUP;
    const long long row_stride = (long long)groups_per_row * 200;
    const char* row_base = A + (long long)safe_row * row_stride;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b =
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    __syncthreads();

    for (int g = 0; g < groups_per_row; g++) {
        {
            const int slot_in_block = warp_id * SLOTS_PER_WARP + (lane >> 3);
            const int k_off = (lane & 7) * 32;
            const int x_row = slot_x_row[slot_in_block];
            half16_t va, vb;
            if (x_row >= 0) {
                const long long x_off = (long long)x_row * K + g * K_GROUP + k_off;
                va = *(const half16_t*)(X_src + x_off);
                vb = *(const half16_t*)(X_src + x_off + 16);
            } else {
                va = zero_b;
                vb = zero_b;
            }
            _Float16* dst = lds_x + slot_in_block * K_GROUP + k_off;
            *(half16_t*)(dst) = va;
            *(half16_t*)(dst + 16) = vb;
        }
        __syncthreads();

        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        for (int kt = 0; kt < 16; kt += 2) {
            const int byte_off_a = kt * 12;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + byte_off_a);
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = *(const unsigned int*)(dp_a + 3);
            unsigned int d2a = *(const unsigned int*)(dp_a + 6);
            unsigned int d3a = *(const unsigned int*)(dp_a + 9);

            const int byte_off_b = (kt + 1) * 12;
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + byte_off_b);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = *(const unsigned int*)(dp_b + 3);
            unsigned int d2b = *(const unsigned int*)(dp_b + 6);
            unsigned int d3b = *(const unsigned int*)(dp_b + 9);

            const _Float16* lds_slot = lds_x + lane16 * K_GROUP;
            half16_t b_a = *(const half16_t*)(lds_slot + kt * 16);
            half16_t b_b = *(const half16_t*)(lds_slot + (kt + 1) * 16);

            half16_t a_reg;
#define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(0,  d0a, 0);  DQ6(1,  d0a, 6);
            DQ6(2,  d0a, 12); DQ6(3,  d0a, 18);
            DQ6(4,  d1a, 0);  DQ6(5,  d1a, 6);
            DQ6(6,  d1a, 12); DQ6(7,  d1a, 18);
            DQ6(8,  d2a, 0);  DQ6(9,  d2a, 6);
            DQ6(10, d2a, 12); DQ6(11, d2a, 18);
            DQ6(12, d3a, 0);  DQ6(13, d3a, 6);
            DQ6(14, d3a, 12); DQ6(15, d3a, 18);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ6(0,  d0b, 0);  DQ6(1,  d0b, 6);
            DQ6(2,  d0b, 12); DQ6(3,  d0b, 18);
            DQ6(4,  d1b, 0);  DQ6(5,  d1b, 6);
            DQ6(6,  d1b, 12); DQ6(7,  d1b, 18);
            DQ6(8,  d2b, 0);  DQ6(9,  d2b, 6);
            DQ6(10, d2b, 12); DQ6(11, d2b, 18);
            DQ6(12, d3b, 0);  DQ6(13, d3b, 6);
            DQ6(14, d3b, 12); DQ6(15, d3b, 18);
#undef DQ6
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
        __syncthreads();
    }

    const int out_col = slot_start + lane16;
    if (out_col < m_total) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = warp_m_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
