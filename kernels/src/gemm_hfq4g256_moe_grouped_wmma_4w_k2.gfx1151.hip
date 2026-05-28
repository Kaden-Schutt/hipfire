// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// 4-warp gfx1151 HFQ4 grouped-MoE WMMA. Slot tile stays 16 wide so one
// block still maps to one expert tile; M tile widens from 16 to 64 rows.
// The 16x256 X slice is staged once in LDS and reused by four warps.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE         64
#define N_TILE         16
#define WARPS          4
#define M_PER_WARP     16
#define K_GROUP        256
#define SLOTS_PER_WARP 4

__launch_bounds__(128, 2)
extern "C" __global__ void gemm_hfq4g256_moe_grouped_wmma_4w_k2_gfx1151(
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
    const long long row_stride = (long long)groups_per_row * 136;
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

        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            const _Float16* lds_slot = lds_x + lane16 * K_GROUP;
            half16_t b_a = *(const half16_t*)(lds_slot + k_off_a);
            half16_t b_b = *(const half16_t*)(lds_slot + k_off_b);

            half16_t a_reg;
#define DQ4(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ4(a_reg, 0,  pk0a,  0); DQ4(a_reg, 1,  pk0a,  4);
            DQ4(a_reg, 2,  pk0a,  8); DQ4(a_reg, 3,  pk0a, 12);
            DQ4(a_reg, 4,  pk0a, 16); DQ4(a_reg, 5,  pk0a, 20);
            DQ4(a_reg, 6,  pk0a, 24); DQ4(a_reg, 7,  pk0a, 28);
            DQ4(a_reg, 8,  pk1a,  0); DQ4(a_reg, 9,  pk1a,  4);
            DQ4(a_reg, 10, pk1a,  8); DQ4(a_reg, 11, pk1a, 12);
            DQ4(a_reg, 12, pk1a, 16); DQ4(a_reg, 13, pk1a, 20);
            DQ4(a_reg, 14, pk1a, 24); DQ4(a_reg, 15, pk1a, 28);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ4(a_reg, 0,  pk0b,  0); DQ4(a_reg, 1,  pk0b,  4);
            DQ4(a_reg, 2,  pk0b,  8); DQ4(a_reg, 3,  pk0b, 12);
            DQ4(a_reg, 4,  pk0b, 16); DQ4(a_reg, 5,  pk0b, 20);
            DQ4(a_reg, 6,  pk0b, 24); DQ4(a_reg, 7,  pk0b, 28);
            DQ4(a_reg, 8,  pk1b,  0); DQ4(a_reg, 9,  pk1b,  4);
            DQ4(a_reg, 10, pk1b,  8); DQ4(a_reg, 11, pk1b, 12);
            DQ4(a_reg, 12, pk1b, 16); DQ4(a_reg, 13, pk1b, 20);
            DQ4(a_reg, 14, pk1b, 24); DQ4(a_reg, 15, pk1b, 28);
#undef DQ4
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
