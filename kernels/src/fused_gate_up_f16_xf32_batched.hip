// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched two-way F16-weight x F32-input GEMV.
__launch_bounds__(32)
extern "C" __global__ void fused_gate_up_f16_xf32_batched(
    const _Float16* __restrict__ Wgate,
    const _Float16* __restrict__ Wup,
    const float*    __restrict__ X,
    float*          __restrict__ Ygate,
    float*          __restrict__ Yup,
    int Mgate,
    int Mup,
    int K,
    int B
) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int total = Mgate + Mup;
    if (row >= total || b >= B) return;

    const _Float16* W = Wgate;
    float* Y = Ygate;
    int local = row;
    int M = Mgate;
    if (local >= Mgate) {
        local -= Mgate;
        W = Wup;
        Y = Yup;
        M = Mup;
        if (local >= Mup) return;
    }

    const _Float16* w_row = W + (size_t)local * K;
    const float* x_row = X + (size_t)b * K;
    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * x_row[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) Y[(size_t)b * M + local] = sum;
}
