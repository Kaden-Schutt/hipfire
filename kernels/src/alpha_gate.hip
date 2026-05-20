// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void alpha_gate_f32(
    float* __restrict__ alpha,          // in: raw projection output, out: gate values
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float biased = alpha[i] + dt_bias[i];
    float sp = (biased > 20.0f) ? biased : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
    alpha[i] = sp * (-expf(a_log[i]));
}
