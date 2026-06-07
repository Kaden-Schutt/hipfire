// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// F16-weight x F32-input GEMV with fused residual add: y[row] += W[row] @ x.
__launch_bounds__(32)
extern "C" __global__ void gemv_f16_xf32_residual(
    const _Float16* __restrict__ W,
    const float*    __restrict__ x,
    float*          __restrict__ y,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const _Float16* w_row = W + (size_t)row * K;
    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * x[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] += sum;
}
