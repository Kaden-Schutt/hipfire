// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Native CDNA3 lowering of DeepSeek4's grouped MQ2-Lloyd MoE prefill GEMM.
// One wave64 computes a 16-output-row x 16-routed-slot tile with
// V_MFMA_F32_16X16X16_F16. The frozen MQ2R payload stays in its row-major
// 72-byte/G256 layout and is decoded directly into MFMA A fragments.

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float vfloat4 __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(64)
void gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const half_t* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int lane = threadIdx.x;
    const int n_lane = lane & 15;
    const int strip = lane >> 4;
    const int row_start = blockIdx.x * 16;
    const int slot_start = blockIdx.y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[blockIdx.y];
    if (expert_id < 0) return;

    const int my_row = row_start + n_lane;
    const int safe_row = my_row < M ? my_row : M - 1;
    const int slot = slot_start + n_lane;
    int x_row = -1;
    if (slot < m_total) {
        const int flat = sorted_slot_index[slot];
        if (flat >= 0) x_row = x_row_div > 1 ? flat / x_row_div : flat;
    }

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)safe_row * groups_per_row * 72;
    vfloat4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_ptr + g * 72;
        const half_t* cb = reinterpret_cast<const half_t*>(gp);

        #pragma unroll
        for (int ki = 0; ki < 256; ki += 16) {
            // A lane owns one output row and four adjacent K values. Four
            // 2-bit Lloyd indices occupy exactly one byte.
            const unsigned char packed =
                *(reinterpret_cast<const unsigned char*>(gp + 8 + (ki + strip * 4) / 4));
            half4 a_reg;
            a_reg[0] = cb[(packed >> 0) & 3u];
            a_reg[1] = cb[(packed >> 2) & 3u];
            a_reg[2] = cb[(packed >> 4) & 3u];
            a_reg[3] = cb[(packed >> 6) & 3u];

            half4 b_reg = {(half_t)0, (half_t)0, (half_t)0, (half_t)0};
            if (x_row >= 0) {
                const long long x_off =
                    (long long)x_row * K + g * 256 + ki + strip * 4;
                b_reg = *reinterpret_cast<const half4*>(X_src + x_off);
            }

            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
                a_reg, b_reg, acc, 0, 0, 0);
        }
    }

    // MFMA result layout: lane (strip,n) owns four M rows for column n.
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int out_row = row_start + strip * 4 + r;
        if (slot < m_total && out_row < M) {
            Y_grouped[(long long)slot * M + out_row] = acc[r];
        }
    }
}
