// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1151 (Strix Halo) HFQ6/MQ6 sister of
// `gemm_hfq4g256_moe_grouped_wmma_k2`. This keeps the existing MoE Path 2
// grouped-dispatch contract and swaps only the inner dequant from HFQ4
// stride-136 to HFQ6/MQ6 stride-200.
//
// Per 256-value group:
//   [0..4]    f32 scale
//   [4..8]    f32 zero
//   [8..200]  192 B packed 6-bit payload
//
// gfx1151 uses the gfx11 wave32 WMMA ABI:
//   - half16_t A/B operands
//   - base `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
//   - output mapping `acc[j] = C[2*j + (tid>>4)][tid & 15]`

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N x K], x_row_div = K_TOP -> x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots x K], x_row_div = 1 -> x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_k2_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows x K]
    float*                    __restrict__ Y_grouped,          // [m_total x M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int lane = tid & 15;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base =
        (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b =
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

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

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + kt * 16);
                b_b = *(const half16_t*)(xg + (kt + 1) * 16);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_reg;
#define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(0,  d0a, 0);  DQ6(1,  d0a, 6);  DQ6(2,  d0a, 12); DQ6(3,  d0a, 18);
            DQ6(4,  d1a, 0);  DQ6(5,  d1a, 6);  DQ6(6,  d1a, 12); DQ6(7,  d1a, 18);
            DQ6(8,  d2a, 0);  DQ6(9,  d2a, 6);  DQ6(10, d2a, 12); DQ6(11, d2a, 18);
            DQ6(12, d3a, 0);  DQ6(13, d3a, 6);  DQ6(14, d3a, 12); DQ6(15, d3a, 18);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ6(0,  d0b, 0);  DQ6(1,  d0b, 6);  DQ6(2,  d0b, 12); DQ6(3,  d0b, 18);
            DQ6(4,  d1b, 0);  DQ6(5,  d1b, 6);  DQ6(6,  d1b, 12); DQ6(7,  d1b, 18);
            DQ6(8,  d2b, 0);  DQ6(9,  d2b, 6);  DQ6(10, d2b, 12); DQ6(11, d2b, 18);
            DQ6(12, d3b, 0);  DQ6(13, d3b, 6);  DQ6(14, d3b, 12); DQ6(15, d3b, 18);
#undef DQ6
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
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
