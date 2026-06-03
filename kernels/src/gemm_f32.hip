// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 20)
__global__ void gemm_f32_batched(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int M, int K, int N
) {
    int m = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    if (m >= M || n >= N) return;

    const float* a_row = A + (size_t)m * K;
    const float* b_row = B + (size_t)n * K;

    float sum = 0.0f;
    for (int k = tid; k < K; k += 32) {
        sum += a_row[k] * b_row[k];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0)
        Y[(size_t)n * M + m] = sum;
}
