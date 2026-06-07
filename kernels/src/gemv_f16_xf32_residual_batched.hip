// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched F16-weight x F32-input GEMV with fused residual:
// Y[b, row] += W[row] @ X[b].
__launch_bounds__(32)
extern "C" __global__ void gemv_f16_xf32_residual_batched(
    const _Float16* __restrict__ W,
    const float*    __restrict__ X,
    float*          __restrict__ Y,
    int M,
    int K,
    int B
) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    if (row >= M || b >= B) return;
    const int tid = threadIdx.x;

    const _Float16* w_row = W + (size_t)row * K;
    const float* x_row = X + (size_t)b * K;
    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * x_row[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) Y[(size_t)b * M + row] += sum;
}
