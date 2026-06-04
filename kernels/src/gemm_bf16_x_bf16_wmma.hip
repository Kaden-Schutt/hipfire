// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated GEMM: BF16 weight x BF16 input -> F32 output.
// gfx1100/gfx1151 wave32 path, matching gemm_f16_x_f16_wmma's output
// layout:
//   A : [M, K] BF16 row-major (raw uint16_t BF16 payload)
//   X : [B, K] BF16 row-major
//   Y : [B, M] F32 row-major

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16_x_bf16_wmma(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ X,
    float*               __restrict__ Y,
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
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
