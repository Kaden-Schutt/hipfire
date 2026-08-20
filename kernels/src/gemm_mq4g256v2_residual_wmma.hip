// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched MQ4G256V2 (qt=44) GEMM with fused residual add.
//
// gfx11 (RDNA3 / RDNA3.5) variant — sister of gemm_hfq4g256_residual_wmma.hip
// (HFQ4-G256) and gemm_mq4g256v2_residual_wmma.gfx12.hip (RDNA4).
//
// Same geometry as the HFQ4 gfx11 kernel:
//   Each wave32 block computes a 16×16 output tile:
//     C[16 rows, 16 batch] += W[16 rows, 16 K] * X[16 K, 16 batch]
//   iterated over K in chunks of 16, grouped by 256-element weight groups.
//   Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.
//
// Differences from the HFQ4 gfx11 kernel — MQ4V2 header contract only:
//   136 B row groups, dual fp16 headers [s0,z0,s1,z1], payload at +8.
//   Header bytes:
//     [0..4)  : s0 (fp16 lo) | z0 (fp16 hi)  → governs weights 0..127 of the group
//     [4..8)  : s1 (fp16 lo) | z1 (fp16 hi)  → governs weights 128..255 of the group
//     [8..136): 128 bytes of 4-bit nibbles (256 weights), identical packing to v1.
//   The 16 K-tiles inside a group split at kt=8: tiles 0..7 use (s0,z0), 8..15 use (s1,z1).
//   Implemented as `kt < 8 ? sc0 : sc1` / `kt < 8 ? zp0 : zp1`.
//
// WMMA contracts unchanged from the HFQ4 gfx11 kernel:
//   half16_t A/B operands (16 fp16 per lane, full 16-K per lane, kABKLane=1),
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 (no gfx12 suffix),
//   float8_t accumulator, interleaved C mapping acc[j]=C[2*j+(tid>>4)][tid&15].
//
// Reference: gemm_hfq4g256_residual_wmma.hip (gfx11) and
// gemm_mq4g256v2_residual_wmma.gfx12.hip (gfx12 sibling).
// Compile with `hipcc --offload-arch=gfx1100` (or 1151).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void gemm_mq4g256v2_residual_wmma(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= M) return;
    if (batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
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

            unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ

            half16_t b_reg = *(const half16_t*)(xg + k_off);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    const int out_row = row_start + (tid & 15);
    const int out_batch_base = batch_start + (tid >> 4) * 8;

    if (out_row < M) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int bidx = out_batch_base + j;
            if (bidx < batch_size)
                Y[(long long)bidx * M + out_row] += acc[j];
        }
    }
}

