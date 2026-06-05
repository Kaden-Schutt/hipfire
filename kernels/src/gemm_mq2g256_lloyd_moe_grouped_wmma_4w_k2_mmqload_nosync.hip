// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Barrier-free variant of the MMQ-style preload 4-warp MoE grouped MQ2-Lloyd
// WMMA GEMM. Eliminates __syncthreads() and LDS X staging. Each wave loads
// X directly from global memory, removing wave-level serialization at the
// cost of 4x more global X reads.
//
// Same grid/block layout as gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload.
// Output byte-identical (same arithmetic, different memory schedule).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define M_TILE         64
#define N_TILE         16
#define WARPS          4
#define M_PER_WARP     16
#define K_GROUP        256
#define SLOTS_PER_WARP 4

__launch_bounds__(128, 4)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int block_m   = blockIdx.x * M_TILE;
    const int tile_y    = blockIdx.y;
    const int slot_start = tile_y * N_TILE;

    if (block_m >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int lane16  = lane & 15;

    const int warp_m_start = block_m + warp_id * M_PER_WARP;
    const int my_row       = warp_m_start + lane16;
    const int safe_row     = (my_row < M) ? my_row : (M - 1);

    // Per-lane slot X row (replaces __shared__ slot_x_row[N_TILE] + syncthreads).
    int my_slot_x_row = -1;
    {
        const int slot_idx = slot_start + lane16;
        if (slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0) {
                my_slot_x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
            }
        }
    }

    const int groups_per_row    = K / K_GROUP;
    const long long row_stride  = (long long)groups_per_row * 72;
    const char* row_base        = A + (long long)safe_row * row_stride;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const half16_t zero_b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int g = 0; g < groups_per_row; g++) {
        // MMQ-style: pre-load ALL 8 index packs per K-group.
        const char* gp = row_base + g * 72;
        unsigned int packs[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            packs[i] = *(const unsigned int*)(gp + 8 + i * 4);
        }

        // Codebook.
        const __half* cb_h = reinterpret_cast<const __half*>(gp);
        const _Float16  cb0   = (_Float16)__half2float(cb_h[0]);
        const _Float16  cb1   = (_Float16)__half2float(cb_h[1]);
        const _Float16  cb2   = (_Float16)__half2float(cb_h[2]);
        const _Float16  cb3   = (_Float16)__half2float(cb_h[3]);
        const _Float16  d01   = cb1 - cb0;
        const _Float16  d02   = cb2 - cb0;
        const _Float16  d_xor = cb3 - cb2 - cb1 + cb0;

        // Inner K-tile loop — load B (X data) from global instead of LDS.
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            const unsigned int pa = packs[kt / 2];
            const unsigned int pb = packs[(kt + 1) / 2];

            half16_t b_a, b_b;
            if (my_slot_x_row >= 0) {
                const long long x_off = (long long)my_slot_x_row * K + g * K_GROUP + k_off_a;
                b_a = *(const half16_t*)(X_src + x_off);
                b_b = *(const half16_t*)(X_src + x_off + 16);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_a;
            #define DQ(reg, i, pk, sh) do { \
                _Float16 qlo = (_Float16)(int)(((pk) >> (sh))     & 0x1u); \
                _Float16 qhi = (_Float16)(int)(((pk) >> ((sh)+1)) & 0x1u); \
                reg[i] = cb0 + qlo * d01 + qhi * d02 + (qlo * qhi) * d_xor; \
            } while (0)

            DQ(a_a, 0,  pa,  0); DQ(a_a, 1,  pa,  2); DQ(a_a, 2,  pa,  4); DQ(a_a, 3,  pa,  6);
            DQ(a_a, 4,  pa,  8); DQ(a_a, 5,  pa, 10); DQ(a_a, 6,  pa, 12); DQ(a_a, 7,  pa, 14);
            DQ(a_a, 8,  pa, 16); DQ(a_a, 9,  pa, 18); DQ(a_a, 10, pa, 20); DQ(a_a, 11, pa, 22);
            DQ(a_a, 12, pa, 24); DQ(a_a, 13, pa, 26); DQ(a_a, 14, pa, 28); DQ(a_a, 15, pa, 30);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ(a_a, 0,  pb,  0); DQ(a_a, 1,  pb,  2); DQ(a_a, 2,  pb,  4); DQ(a_a, 3,  pb,  6);
            DQ(a_a, 4,  pb,  8); DQ(a_a, 5,  pb, 10); DQ(a_a, 6,  pb, 12); DQ(a_a, 7,  pb, 14);
            DQ(a_a, 8,  pb, 16); DQ(a_a, 9,  pb, 18); DQ(a_a, 10, pb, 20); DQ(a_a, 11, pb, 22);
            DQ(a_a, 12, pb, 24); DQ(a_a, 13, pb, 26); DQ(a_a, 14, pb, 28); DQ(a_a, 15, pb, 30);
            #undef DQ
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
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
