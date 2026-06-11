// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1151 routed MoE grouped GEMM for raw FP16 expert weights.
// Contract matches the MQ grouped kernels:
//   A expert weights: [M, K] fp16 row-major, one pointer per expert
//   X_src:            [x_src_rows, K] fp16, staged by the Rust wrapper
//   Y_grouped:        [m_total, M] fp32
//   x_row_div:        K_TOP for gate_up, 1 for down

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE         128
#define N_TILE         16
#define WARPS          8
#define M_PER_WARP     16
#define K_BLOCK        256
#define SLOTS_PER_WARP 2

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_f16_moe_grouped_wmma_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int row_start = blockIdx.x * M_TILE;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * N_TILE;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const _Float16* __restrict__ A =
        reinterpret_cast<const _Float16*>(expert_weight_ptrs[expert_id]);

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

    __shared__ _Float16 lds_x[N_TILE * K_BLOCK];

    const int warp_row_start = row_start + warp_id * M_PER_WARP;
    const int my_row = warp_row_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);
    const _Float16* w_row = A + (long long)safe_row * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const half16_t zero_b = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    __syncthreads();

    for (int k_base = 0; k_base < K; k_base += K_BLOCK) {
        const int slot_in_block = warp_id * SLOTS_PER_WARP + (lane >> 4);
        const int k_off = (lane & 15) * 16;
        const int x_row = slot_x_row[slot_in_block];
        half16_t x_frag = zero_b;
        if (x_row >= 0 && k_base + k_off < K) {
            x_frag = *(const half16_t*)(X_src + (long long)x_row * K + k_base + k_off);
        }
        *(half16_t*)(lds_x + slot_in_block * K_BLOCK + k_off) = x_frag;

        __syncthreads();

        const int group_k = ((K - k_base) < K_BLOCK) ? (K - k_base) : K_BLOCK;
        for (int kt = 0; kt < group_k; kt += 16) {
            half16_t a_frag = *(const half16_t*)(w_row + k_base + kt);
            half16_t b_frag = *(const half16_t*)(lds_x + lane16 * K_BLOCK + kt);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    const int out_col = slot_start + lane16;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = warp_row_start + 2 * j + (lane >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#undef M_TILE
#undef N_TILE
#undef WARPS
#undef M_PER_WARP
#undef K_BLOCK
#undef SLOTS_PER_WARP
