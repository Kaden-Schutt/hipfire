// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library: WMMA GEMM, signed INT8 inputs -> INT32 output.
// gfx1103 (RDNA3 UMA) wave32, register-tiled, ZERO LDS (gfx1103 LDS-hang
// lesson). Same tile/layout as the f16/bf16 siblings:
//   A : [M, K] int8 row-major
//   X : [B, K] int8 row-major
//   Y : [B, M] int32 row-major
//
// Uses v_wmma_i32_16x16x16_iu8 with both operands SIGNED and clamp=false (the
// i32 accumulator must not saturate across K-tiles). Each lane loads its 16
// int8 K-elements as one int32x4_t. acc[j] at lane t maps to
// (row=2*j+(t>>4), col=t&15), identical to the float siblings.
//
// Requires K % 16 == 0. For unsigned int8, flip the signedness flags (a future
// gemm_u8_i32_wmma variant); this kernel is the signed path.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_iu8_i32_wmma(
    const int8_t* __restrict__ A,   // [M, K] int8 (signed)
    const int8_t* __restrict__ X,   // [B, K] int8 (signed)
    int32_t*      __restrict__ Y,   // [B, M] int32
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

    const int8_t* w_row = A + (long long)safe_m * K;
    const int8_t* x_row = X + (long long)safe_b * K;

    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k = 0; k < K; k += 16) {
        int32x4_t a_frag = *(const int32x4_t*)(w_row + k);
        int32x4_t x_frag = *(const int32x4_t*)(x_row + k);
        // signed A, signed B, clamp=false (no i32 saturation mid-accumulation).
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_frag, true, x_frag, acc, false);
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
