// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused Q8_0 GEMM for the Qwen3.5 FFN
// preamble (w_gate + w_up projections from the same normed X).
//
// **gfx12 (RDNA4) variant — K4 unroll**. Mirrors the HFQ4 sibling
// `gemm_gate_up_hfq4g256_wmma.gfx12.hip` (issue #65 lever 2), which
// measured 2.0× on R9700 9B pp=128 gate_up (1842 → 922 µs/call,
// 39 → 80 GiB/s effective). The same pattern should help Q8: the
// kernel_atlas profile on R9700 (2026-05-14, single-block loop)
// showed only 191 GiB/s effective (30% of peak) and 51.7% of total
// prefill time — deep-pipelining the loads above the WMMA chain is
// the obvious next lever.
//
// Pattern (same shape as the HFQ4 K4 unroll):
//   1. Outer loop iterates 2 Q8 blocks (= 4 WMMA-K-tiles) per body
//   2. All 4 X reads + all 4 weight reads issued up-front
//   3. Weight bytes are loaded as uint64 packs (8 int8 per pack) so
//      each WMMA tile = 1 VMEM op instead of 8 byte loads
//   4. 4 dequant + WMMA pairs follow back-to-back; s_waitcnt vmcnt(>0)
//      lets the compiler overlap VMEM with WMMA
//
// Block-pair plus tail: the K4 unroll processes 2 Q8 blocks per body,
// and a final single-block tail handles odd `blocks_per_row`. Production
// 9B shapes (K=4096 → 128 blocks, K=11008 → 344 blocks) take the fast
// path with no tail; the tail exists for correctness on small/odd K.
//
// gfx12 vs gfx11 differences (carried over from the single-block port):
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//   2. half8_t A/B operands instead of half16_t
//   3. K dim split across 2 lane-groups (k_grp = tid >> 4)
//   4. C-output mapping: acc[j] = C[8*(tid>>4) + j][tid & 15]
//      (silicon-validated on R9700, 2026-05-14)
//
// Grid: [ceil((gate_m + up_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_q8_0_wmma_gfx12(
    const char*     __restrict__ A_gate,
    const char*     __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y_gate,
    float*          __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m     = gate_m + up_m;
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
    if (safe_row < gate_m) {
        A = A_gate;  local_row = safe_row;
    } else {
        A = A_up;    local_row = safe_row - gate_m;
    }

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)local_row * row_stride;

    const int safe_batch = ((batch_start + m_lane) < N)
                           ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Dequant macro: unpack one uint64 of 8 int8 weights, multiply by scale.
    // Cast through `signed char` so the high bit is sign-extended (Q8_0 is
    // signed [-128, 127]).
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

        // 4 weight packs (one uint64 = 8 int8 weights per WMMA tile), all up-front.
        unsigned long long pk0a, pk1a, pk0b, pk1b;
        __builtin_memcpy(&pk0a, bp_a + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1a, bp_a + 2 + 16 + k_grp * 8, 8);
        __builtin_memcpy(&pk0b, bp_b + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1b, bp_b + 2 + 16 + k_grp * 8, 8);

        // 4 X tiles, all up-front.
        half8_t b0a = *(const half8_t*)(x_a +  0 + k_grp * 8);
        half8_t b1a = *(const half8_t*)(x_a + 16 + k_grp * 8);
        half8_t b0b = *(const half8_t*)(x_b +  0 + k_grp * 8);
        half8_t b1b = *(const half8_t*)(x_b + 16 + k_grp * 8);

        // 4 dequant + WMMA pairs, back-to-back.
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
    // shapes never enter this path.
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

    // gfx12 C-output mapping (silicon-validated 2026-05-14):
    //   acc[j] = C[8*(tid>>4) + j][tid & 15]
    const int out_col = batch_start + m_lane;
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 8 * k_grp + j;
        if (out_row < total_m) {
            float* Y;
            int proj_row;
            int proj_m;
            if (out_row < gate_m) {
                Y = Y_gate; proj_row = out_row;            proj_m = gate_m;
            } else {
                Y = Y_up;   proj_row = out_row - gate_m;   proj_m = up_m;
            }
            Y[(long long)out_col * proj_m + proj_row] = acc[j];
        }
    }
}
