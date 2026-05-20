// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 4-way fused Q8_0 GEMM for the Qwen3.5
// DeltaNet LA preamble (wqkv + wz + w_beta + w_alpha projections).
//
// **gfx12 (RDNA4) variant — K4 unroll**. Same pattern as the gate_up
// sibling: 2 Q8 blocks (= 4 WMMA tiles) per outer-loop iter, all loads
// up-front, 4 dequant+WMMA pairs back-to-back. Validated numerically
// identical to the single-block port via test_gemm_q8_qkvza_wmma.
//
// Block-pair plus tail: 2 Q8 blocks per K4-unrolled body, plus a single-
// block tail for odd blocks_per_row (correctness on small/odd K).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_q8_0_wmma_gfx12(
    const char*     __restrict__ A_qkv,
    const char*     __restrict__ A_z,
    const char*     __restrict__ A_beta,
    const char*     __restrict__ A_alpha,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y_qkv,
    float*          __restrict__ Y_z,
    float*          __restrict__ Y_beta,
    float*          __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m     = qkv_m + z_m + beta_m + alpha_m;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N)     return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row   = row_start + m_lane;
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

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)local_row * row_stride;

    const int safe_batch = ((batch_start + m_lane) < N)
                           ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #define DQ8_Q(pk, sc) \
        a_reg[0] = (sc) * (_Float16)(float)(signed char)(((pk) >>  0) & 0xFFu); \
        a_reg[1] = (sc) * (_Float16)(float)(signed char)(((pk) >>  8) & 0xFFu); \
        a_reg[2] = (sc) * (_Float16)(float)(signed char)(((pk) >> 16) & 0xFFu); \
        a_reg[3] = (sc) * (_Float16)(float)(signed char)(((pk) >> 24) & 0xFFu); \
        a_reg[4] = (sc) * (_Float16)(float)(signed char)(((pk) >> 32) & 0xFFu); \
        a_reg[5] = (sc) * (_Float16)(float)(signed char)(((pk) >> 40) & 0xFFu); \
        a_reg[6] = (sc) * (_Float16)(float)(signed char)(((pk) >> 48) & 0xFFu); \
        a_reg[7] = (sc) * (_Float16)(float)(signed char)(((pk) >> 56) & 0xFFu)

    int bi = 0;
    for (; bi + 1 < blocks_per_row; bi += 2) {
        const char* bp_a = row_base + bi * 34;
        const char* bp_b = row_base + (bi + 1) * 34;

        _Float16 sc_a, sc_b;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp_a, 2); __builtin_memcpy(&sc_a, &s_bits, 2);
            __builtin_memcpy(&s_bits, bp_b, 2); __builtin_memcpy(&sc_b, &s_bits, 2);
        }

        const _Float16* x_a = X + (long long)safe_batch * K + bi * 32;
        const _Float16* x_b = X + (long long)safe_batch * K + (bi + 1) * 32;

        unsigned long long pk0a, pk1a, pk0b, pk1b;
        __builtin_memcpy(&pk0a, bp_a + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1a, bp_a + 2 + 16 + k_grp * 8, 8);
        __builtin_memcpy(&pk0b, bp_b + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1b, bp_b + 2 + 16 + k_grp * 8, 8);

        half8_t b0a = *(const half8_t*)(x_a +  0 + k_grp * 8);
        half8_t b1a = *(const half8_t*)(x_a + 16 + k_grp * 8);
        half8_t b0b = *(const half8_t*)(x_b +  0 + k_grp * 8);
        half8_t b1b = *(const half8_t*)(x_b + 16 + k_grp * 8);

        half8_t a_reg;
        DQ8_Q(pk0a, sc_a);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b0a, acc);
        DQ8_Q(pk1a, sc_a);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b1a, acc);
        DQ8_Q(pk0b, sc_b);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b0b, acc);
        DQ8_Q(pk1b, sc_b);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b1b, acc);
    }

    // Tail: one unpaired block if blocks_per_row is odd.
    if (bi < blocks_per_row) {
        const char* bp = row_base + bi * 34;
        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const _Float16* x_base = X + (long long)safe_batch * K + bi * 32;

        unsigned long long pk0, pk1;
        __builtin_memcpy(&pk0, bp + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1, bp + 2 + 16 + k_grp * 8, 8);

        half8_t b0 = *(const half8_t*)(x_base +  0 + k_grp * 8);
        half8_t b1 = *(const half8_t*)(x_base + 16 + k_grp * 8);

        half8_t a_reg;
        DQ8_Q(pk0, sc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b0, acc);
        DQ8_Q(pk1, sc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b1, acc);
    }
    #undef DQ8_Q

    // gfx12 C-output mapping: acc[j] = C[8*(tid>>4) + j][tid & 15]
    const int out_col = batch_start + m_lane;
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 8 * k_grp + j;
        if (out_row < total_m) {
            float* Y;
            int proj_row;
            int proj_m;
            if (out_row < qkv_m) {
                Y = Y_qkv;   proj_row = out_row;                         proj_m = qkv_m;
            } else if (out_row < qkv_m + z_m) {
                Y = Y_z;     proj_row = out_row - qkv_m;                  proj_m = z_m;
            } else if (out_row < qkv_m + z_m + beta_m) {
                Y = Y_beta;  proj_row = out_row - (qkv_m + z_m);          proj_m = beta_m;
            } else {
                Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m); proj_m = alpha_m;
            }
            Y[(long long)out_col * proj_m + proj_row] = acc[j];
        }
    }
}
