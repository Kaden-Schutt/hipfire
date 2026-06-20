// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Generic kernel library GEMV: BF16 weight × BF16 vector → F32.
// y[m] = sum_k W_bf16[m,k] * x_bf16[k]. gfx1103 wave32, zero LDS, F32 accumulate.
// grid=[M], block=[32], wave-shuffle reduction. BF16 payloads are raw uint16.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__launch_bounds__(32)
extern "C" __global__ void gemv_bf16_f32(
    const unsigned short* __restrict__ W,   // [M, K] BF16
    const unsigned short* __restrict__ x,   // [K] BF16
    float*                __restrict__ y,   // [M] F32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const unsigned short* w_row = W + (size_t)row * K;

    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += bf16_to_f32(w_row[i]) * bf16_to_f32(x[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
