// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library: WMMA GEMM, F16 inputs -> F16 output.
// gfx1103 (RDNA3 UMA) wave32, register-tiled, ZERO LDS (gfx1103 LDS-hang
// lesson). Layout matches gemm_f16_x_f16_wmma (the F16->F32 sibling):
//   A : [M, K] F16 row-major
//   X : [B, K] F16 row-major
//   Y : [B, M] F16 row-major
//
// Accumulation is in F32 via v_wmma_f32_16x16x16_f16, rounded to F16 only on
// store. The 16-bit-output WMMA (v_wmma_f16_16x16x16_f16) is avoided for the
// same reason as the bf16 sibling: its packed F16 C accumulator does not chain
// across K-tiles (drops magnitude for K>16). "->F16" denotes output dtype, not
// accumulation precision. acc[j] at lane t maps to (row=2*j+(t>>4), col=t&15).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16_f16_wmma(
    const _Float16* __restrict__ A,    // [M, K] F16
    const _Float16* __restrict__ X,    // [B, K] F16
    _Float16*       __restrict__ Y,    // [B, M] F16
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

    const _Float16* w_row = A + (long long)safe_m * K;
    const _Float16* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k = 0; k < K; k += 16) {
        half16_t a_frag = *(const half16_t*)(w_row + k);
        half16_t x_frag = *(const half16_t*)(x_row + k);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
    }

    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = (_Float16)acc[j];
            }
        }
    }
}
