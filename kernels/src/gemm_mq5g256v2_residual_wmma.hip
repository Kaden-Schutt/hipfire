// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched MQ5G256V2 (qt=48) GEMM with fused residual add.
//
// gfx11 (RDNA3 / RDNA3.5) variant — sister of
// gemm_mq5g256v2_residual_wmma.gfx12.hip (RDNA4) and architecture peer of
// gemm_mq4g256v2_residual_wmma.hip (MQ4V2 gfx11).
//
// Same geometry as the MQ4V2 gfx11 residual kernel:
//   Each wave32 block computes a 16×16 output tile:
//     C[16 rows, 16 batch] += W[16 rows, 16 K] * X[16 K, 16 batch]
//   iterated over K in chunks of 16, grouped by 256-element weight groups.
//   Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.
//
// MQ5V2 group contract (168 B, dual fp16 headers, 5-bit payload):
//   [0..4)   : s0 (fp16 lo) | z0 (fp16 hi) → governs weights 0..127
//   [4..8)   : s1 (fp16 lo) | z1 (fp16 hi) → governs weights 128..255
//   [8..168) : 160 bytes of 5-bit packs (8 weights / 5 bytes = 256 weights)
//   Each K-tile is 16 weights = 10 bytes at payload offset kt*10.
//   The 16 K-tiles split at kt=8: tiles 0..7 use (s0,z0), 8..15 use (s1,z1).
//
// WMMA contracts (gfx11 half16):
//   half16_t A/B operands, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32,
//   float8_t accumulator, interleaved C mapping
//   acc[j]=C[2*j+(tid>>4)][tid&15].
//
// Compile with `hipcc --offload-arch=gfx1100` (or 1151).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void gemm_mq5g256v2_residual_wmma(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= M) return;
    if (batch_start >= N) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < N) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 168;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 168;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const _Float16* xg = x_base + g * 256;

        #pragma unroll
        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;

            // 16 × 5-bit = 10 bytes per K-tile.
            const unsigned char* dp = (const unsigned char*)(gp + 8 + kt * 10);
            const uint32_t d0 = *(const uint32_t*)(dp);
            const uint8_t  d4 = dp[4];
            const uint64_t bits0 = ((uint64_t)d0) | ((uint64_t)d4 << 32);
            const uint32_t d5 = *(const uint32_t*)(dp + 5);
            const uint8_t  d9 = dp[9];
            const uint64_t bits1 = ((uint64_t)d5) | ((uint64_t)d9 << 32);

            half16_t a_reg;
            #define DQ5(i, bits, sh) a_reg[i] = sc_h * (_Float16)(float)(((bits) >> (sh)) & 31u) + zp_h
            DQ5(0,  bits0,  0); DQ5(1,  bits0,  5); DQ5(2,  bits0, 10); DQ5(3,  bits0, 15);
            DQ5(4,  bits0, 20); DQ5(5,  bits0, 25); DQ5(6,  bits0, 30); DQ5(7,  bits0, 35);
            DQ5(8,  bits1,  0); DQ5(9,  bits1,  5); DQ5(10, bits1, 10); DQ5(11, bits1, 15);
            DQ5(12, bits1, 20); DQ5(13, bits1, 25); DQ5(14, bits1, 30); DQ5(15, bits1, 35);
            #undef DQ5

            half16_t b_reg = *(const half16_t*)(xg + k_off);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    const int out_col = batch_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
