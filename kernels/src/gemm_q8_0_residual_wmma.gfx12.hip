// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched Q8_0 GEMM with fused residual add.
//
// **gfx12 (RDNA4) variant — K4 unroll**. Same pattern as the other Q8
// gfx12 siblings: 2 Q8 blocks (= 4 WMMA tiles) per loop iter, all loads
// up-front, 4 dequant+WMMA pairs back-to-back. Validated numerically
// identical to the single-block port via test_gemm_q8_residual_wmma.
//
// Y[N, M] += X[N, K] @ A_q8[M, K]^T
//
// === Non-overlapping-write invariant (DO NOT BREAK) ===
// The `Y[i] += acc[j]` at the end is **NOT atomic**. Race-free only
// because each (out_row, out_col) cell is written by exactly one lane
// in exactly one block under this kernel's grid. With the gfx12 mapping
//   acc[j] → C[8*(tid>>4) + j][tid & 15]
// lane group 0 writes rows 0..7, lane group 1 writes rows 8..15 — also
// non-overlapping by construction. Any K-split sibling would still
// need atomicAdd.
//
// Block-pair plus tail: 2 Q8 blocks per K4-unrolled body, plus a single-
// block tail for odd blocks_per_row (correctness on small/odd K).
// Production 9B shapes (wo K=4096 → 128 blocks; w_down K=11008 → 344
// blocks) take the fast path with no tail.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_q8_0_residual_wmma_gfx12(
    const char*     __restrict__ A_q8,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y,      // caller seeds with residual
    int M, int K, int N
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start >= M)   return;
    if (batch_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row   = row_start + m_lane;
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A_q8 + (long long)safe_row * row_stride;

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
    // += semantics fuses the residual.
    const int out_col = batch_start + m_lane;
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 8 * k_grp + j;
        if (out_row < M) {
            Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
