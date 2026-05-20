// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA GEMM with K-split: attacks grid-starvation on small-M WMMA GEMMs.
//
// Problem: the plain WMMA GEMM has grid = [M/16, N/16]. For the Qwen3.5-27B
// FA wo_residual (M=5120, N=16) that's only 320 blocks — ~3.3 per CU on
// gfx1100 (96 CUs). Too few waves in flight to hide memory latency, so the
// kernel runs at ~26% of peak BW (247 GB/s vs 960 GB/s peak).
//
// Fix: split the K-loop across K_SPLITS blocks on grid.z. Each block processes
// K/K_SPLITS of the K range, atomicAdd's its partial into Y. Grid becomes
// [M/16, N/16, K_SPLITS] = 4× more blocks → ~13 blocks/CU, enough concurrent
// waves to saturate the memory subsystem.
//
// Y MUST contain the residual summand on entry (unchanged from the
// non-k-split variant). First block (blockIdx.z == 0) is written as a
// regular add to preserve Y's residual; subsequent blocks atomicAdd their
// partials on top. This is WHY each block atomicAdds instead of racing:
// the residual is summed exactly once.
//
// Byte-exact-up-to-fp-rounding with the non-split k2 kernel; the reduce
// order changes, so non-deterministic accumulation in fp32 gives tiny
// differences. For greedy decode this is well under argmax-tie margins.
//
// Grid: [ceil(M/16), ceil(N/16), K_SPLITS]. Block: [32]. LDS: 0.

#define K_SPLITS 4

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g256_residual_wmma_ksplit(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int ksplit = blockIdx.z;

    if (row_start >= M || batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const int groups_per_split = groups_per_row / K_SPLITS;
    const int g_start = ksplit * groups_per_split;
    const int g_end   = (ksplit == K_SPLITS - 1) ? groups_per_row
                                                 : g_start + groups_per_split;

    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = g_start; g < g_end; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;

        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);

            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b_a = *(const half16_t*)(xg + k_off_a);
            half16_t b_b = *(const half16_t*)(xg + k_off_b);

            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    const int out_col = batch_start + (tid & 15);
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                // atomicAdd: 4 partials from 4 K-splits race into the same Y
                // cell. Y already has the residual from upstream, and we need
                // sum-with-residual → atomicAdd is the simplest correct reduce.
                atomicAdd(&Y[(long long)out_col * M + out_row], acc[j]);
            }
        }
    }
}
