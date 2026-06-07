// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Two independent F16-weight x F32-input GEMVs in one launch.
__launch_bounds__(32)
extern "C" __global__ void fused_gate_up_f16_xf32(
    const _Float16* __restrict__ Wgate,
    const _Float16* __restrict__ Wup,
    const float*    __restrict__ x,
    float*          __restrict__ ygate,
    float*          __restrict__ yup,
    int Mgate,
    int Mup,
    int K
) {
    int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int total = Mgate + Mup;
    if (row >= total) return;

    const _Float16* W = Wgate;
    float* y = ygate;
    int local = row;
    if (local >= Mgate) {
        local -= Mgate;
        W = Wup;
        y = yup;
        if (local >= Mup) return;
    }

    const _Float16* w_row = W + (size_t)local * K;
    float sum = 0.0f;
    for (int i = tid; i < K; i += 32) {
        sum += (float)w_row[i] * x[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[local] = sum;
}
