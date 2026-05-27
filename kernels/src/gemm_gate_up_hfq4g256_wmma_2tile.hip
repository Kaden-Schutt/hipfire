// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// 2tile variant of gemm_gate_up_hfq4g256_wmma. Doubles the per-block
// output tile in the M dimension (32 rows × 16 cols vs base 16 × 16),
// halving the grid (1728 → 864 blocks at M=27648). Two wave32 waves
// per block, each owning its own 16-row half-tile and computing an
// independent WMMA accumulator against the SAME 16-col X tile.
//
// Why: base kernel runs at 32% of 960 GB/s peak BW on gfx1100 at
// M=27648 N=3. k4 4-tile pipeline (+3% per-call only) and ldscoop LDS
// staging (-14%) both failed to capture the ceiling. Both attacked
// "weight load latency" or "weight coalescing" — neither was the
// bottleneck. Hypothesis here: per-block X load amortization is the
// lever — X is FP16 [N×K], ~160 KB per group * groups. Both waves in
// the same block share the same X tile, so cache (L0/L1) absorbs the
// second wave's loads ~for free, halving effective X traffic.
//
// Each wave independently:
//   - lane in [0,31] = tid & 31
//   - my_row = row_start + wave_id*16 + (lane & 15)
//   - safe_batch = batch_start + (lane & 15), clamped
//   - issues its own WMMA chain producing acc[0..7]
//
// Same WMMA C-mapping per wave as base (canonical wave32 layout). Byte
// -exact-up-to-fp-rounding vs base (single wave per row tile, same
// accumulation order within a row).
//
// Grid: [ceil((gate_m + up_m)/32), ceil(N/16)]. Block: [64]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(64, 2)
extern "C" __global__ void gemm_gate_up_hfq4g256_wmma_2tile(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 32;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;   // 0 or 1
    const int lane = tid & 31;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    // Each wave owns 16 rows; this wave's rows = [wave_row_base, +15].
    const int wave_row_base = row_start + wave_id * 16;
    const int my_row = wave_row_base + (lane & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < gate_m) { A = A_gate; local_row = safe_row; }
    else                   { A = A_up;   local_row = safe_row - gate_m; }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + (lane & 15) < N)
                           ? (batch_start + (lane & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;

        // k2 software pipeline (same as base).
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b_a = *(const half16_t*)(x_base + k_off_a);
            half16_t b_b = *(const half16_t*)(x_base + k_off_b);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0a,  0); DQ(1,  pk0a,  4); DQ(2,  pk0a,  8); DQ(3,  pk0a, 12);
            DQ(4,  pk0a, 16); DQ(5,  pk0a, 20); DQ(6,  pk0a, 24); DQ(7,  pk0a, 28);
            DQ(8,  pk1a,  0); DQ(9,  pk1a,  4); DQ(10, pk1a,  8); DQ(11, pk1a, 12);
            DQ(12, pk1a, 16); DQ(13, pk1a, 20); DQ(14, pk1a, 24); DQ(15, pk1a, 28);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc);

            DQ(0,  pk0b,  0); DQ(1,  pk0b,  4); DQ(2,  pk0b,  8); DQ(3,  pk0b, 12);
            DQ(4,  pk0b, 16); DQ(5,  pk0b, 20); DQ(6,  pk0b, 24); DQ(7,  pk0b, 28);
            DQ(8,  pk1b,  0); DQ(9,  pk1b,  4); DQ(10, pk1b,  8); DQ(11, pk1b, 12);
            DQ(12, pk1b, 16); DQ(13, pk1b, 20); DQ(14, pk1b, 24); DQ(15, pk1b, 28);
            #undef DQ
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc);
        }
    }

    // Output: per-wave 16x16 WMMA tile mapping.
    // acc[j] = C[2*j + (lane>>4)][lane & 15] within this wave's tile,
    // offset by wave_row_base in the m-dim.
    const int out_col = batch_start + (lane & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = wave_row_base + 2 * j + (lane >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate;  proj_row = out_row;          out_stride = gate_m;
                } else {
                    Y = Y_up;    proj_row = out_row - gate_m; out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
