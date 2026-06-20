// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library: WMMA GEMM, BF16 inputs -> BF16 output.
// gfx1103 (RDNA3 UMA) wave32, register-tiled, ZERO LDS (gfx1103 LDS-hang
// lesson). Layout matches gemm_bf16_x_bf16_wmma (the bf16->F32 sibling):
//   A : [M, K] BF16 row-major (raw uint16_t BF16 payload)
//   X : [B, K] BF16 row-major
//   Y : [B, M] BF16 row-major
//
// Accumulation is in F32 via v_wmma_f32_16x16x16_bf16 (the result is rounded
// to BF16 only on store). This is both more accurate and more robust than the
// 16-bit-output WMMA (v_wmma_bf16_16x16x16_bf16): the latter re-feeds a packed
// BF16 C accumulator whose slot layout does not line up with the D write for a
// general K>16 GEMM, so chained accumulation silently drops magnitude. F32
// accumulate + BF16 store is the production-correct choice. acc[j] at lane t
// maps to (row=2*j+(t>>4), col=t&15), identical to the F32 sibling.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

// F32 -> BF16, round-to-nearest-even.
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (unsigned short)(b >> 16); // NaN
    unsigned int bias = 0x7fffu + ((b >> 16) & 1u);
    return (unsigned short)((b + bias) >> 16);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16_bf16_wmma(
    const unsigned short* __restrict__ A,   // [M, K] BF16
    const unsigned short* __restrict__ X,   // [B, K] BF16
    unsigned short*       __restrict__ Y,   // [B, M] BF16
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= M || batch_start >= B) return;

    int my_m = row_start + lane;
    int my_b = batch_start + lane;
    int safe_m = (my_m < M) ? my_m : (M - 1);
    int safe_b = (my_b < B) ? my_b : (B - 1);

    const unsigned short* w_row = A + (long long)safe_m * K;
    const unsigned short* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k = 0; k < K; k += 16) {
        b16x16_t a_frag = *(const b16x16_t*)(w_row + k);
        b16x16_t x_frag = *(const b16x16_t*)(x_row + k);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, x_frag, acc);
    }

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = f32_to_bf16_rne(acc[j]);
            }
        }
    }
}
