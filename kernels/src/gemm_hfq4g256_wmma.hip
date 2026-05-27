// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated GEMM for HFQ4G256 weights + F16 input → F32 output.
// gfx1100+ only (RDNA3 / RDNA3.5 wave32 WMMA).
//
// Drop-in for `gemm_hfq4g256` (scalar FMA path) when the activation
// has been F32→F16 staged. HFQ4G256 packs 8 weights per u32 with
// f32 scale + f32 zero per 256-element group (136 bytes/group).
//
// Layout:
//   A : HFQ4G256-packed [M, K/256 × 136] bytes
//   X : [B, K]  F16 row-major (caller stages F32→F16)
//   Y : [B, M]  F32 row-major
//
// Grid:  [ceil(M / 16), ceil(B / 16)]
// Block: [32]
//
// Per K=16 step we dequant a 16-element column of A from one packed
// u32 chunk (8 weights → 16 because we walk two sub-groups per WMMA
// iter for cache-line efficiency). Each thread holds 16 F16 elements
// of the A fragment (its row of the 16×16 tile). Same wave-32 WMMA
// duplication semantics as gemm_f16_x_f16_wmma.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_hfq4g256_wmma(
    const char*     __restrict__ A,    // HFQ4G256 packed
    const _Float16* __restrict__ X,    // [B, K] F16
    float*          __restrict__ Y,    // [B, M] F32
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= M || batch_start >= B) return;

    const int my_m   = row_start + lane;
    const int my_b   = batch_start + lane;
    const int safe_m = (my_m < M) ? my_m : (M - 1);
    const int safe_b = (my_b < B) ? my_b : (B - 1);

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* a_row = A + (long long)safe_m * row_stride;
    const _Float16* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // For each HFQ4 group of 256 K elements, run 16 WMMA mma_f16 ops
    // (one per K=16 sub-tile). The packed u32 layout encodes 8 nibbles
    // per dword × 32 dwords per group; per K=16 step we use 2 dwords.
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = a_row + g * 136;
        _Float16 sc = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned int* pk = (const unsigned int*)(gp + 8);

        #pragma unroll 1
        for (int kt = 0; kt < 16; kt++) {
            // Each K=16 sub-tile consumes 2 u32 dwords from this row's
            // packed nibbles (16 nibbles total).
            unsigned int pk0 = pk[kt * 2];
            unsigned int pk1 = pk[kt * 2 + 1];

            half16_t a_frag;
            #define DQ(i, p, sh) a_frag[i] = sc * (_Float16)(float)(((p) >> (sh)) & 0xFu) + zp
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ

            const int k = g * 256 + kt * 16;
            half16_t x_frag = *(const half16_t*)(x_row + k);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
        }
    }

    // Store outputs — row = 2*j + (tid>>4), col (= batch) = tid & 15.
    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
