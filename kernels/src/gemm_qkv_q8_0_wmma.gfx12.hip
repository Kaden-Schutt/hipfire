// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 3-way fused Q8_0 GEMM for the Qwen3.5
// FullAttention preamble (Q + K + V projections from one normed X).
//
// **gfx12 (RDNA4) variant — K4 unroll**. Mirrors the HFQ4 gfx12 sibling
// pattern (issue #65 lever 2). Original single-block port was
// silicon-validated on R9700 in commit 79976dac; this K-unrolled
// version preserves the same C-output mapping and validates
// numerically identical via test_gemm_q8_qkv_wmma.
//
// Pattern: outer loop processes 2 Q8 blocks (= 4 WMMA-K-tiles) per
// body. All 4 X reads + 4 weight reads (uint64 packs of 8 int8 each)
// issued up-front; 4 dequant+WMMA pairs follow back-to-back so the
// compiler can deep-pipeline VMEM behind WMMA execution.
//
// Block-pair plus tail: the K4 unroll processes 2 Q8 blocks per body,
// and a final single-block tail handles odd `blocks_per_row`. Production
// 9B shapes (K=4096, blocks=128) take the fast path with no tail; the
// tail exists for correctness on small/odd K test shapes (e.g. the
// every-int8-value-once pattern at K=32 → blocks_per_row=1).

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_q8_0_wmma_gfx12(
    const char*     __restrict__ A_q,    // [q_m, K/32 * 34] Q8_0
    const char*     __restrict__ A_k,    // [k_m, K/32 * 34] Q8_0
    const char*     __restrict__ A_v,    // [v_m, K/32 * 34] Q8_0
    const _Float16* __restrict__ X,      // [N, K] FP16
    float*          __restrict__ Y_q,    // [N, q_m] FP32
    float*          __restrict__ Y_k,    // [N, k_m] FP32
    float*          __restrict__ Y_v,    // [N, v_m] FP32
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m     = q_m + k_m + v_m;
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
    if (safe_row < q_m) {
        A = A_q;  local_row = safe_row;
    } else if (safe_row < q_m + k_m) {
        A = A_k;  local_row = safe_row - q_m;
    } else {
        A = A_v;  local_row = safe_row - q_m - k_m;
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

    // Tail: one unpaired block if blocks_per_row is odd. Production
    // shapes (K=4096, blocks=128) never enter this path.
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
            if (out_row < q_m) {
                Y = Y_q;  proj_row = out_row;                 proj_m = q_m;
            } else if (out_row < q_m + k_m) {
                Y = Y_k;  proj_row = out_row - q_m;            proj_m = k_m;
            } else {
                Y = Y_v;  proj_row = out_row - q_m - k_m;      proj_m = v_m;
            }
            Y[(long long)out_col * proj_m + proj_row] = acc[j];
        }
    }
}
