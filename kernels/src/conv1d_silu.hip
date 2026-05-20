// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void conv1d_silu_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int n_channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float x = input[c];
    float s0 = state[c * 3];
    float s1 = state[c * 3 + 1];
    float s2 = state[c * 3 + 2];

    float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
            + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

    // SiLU fused
    output[c] = y / (1.0f + expf(-y));

    state[c * 3 + 2] = s1;
    state[c * 3 + 1] = s0;
    state[c * 3]     = x;
}
