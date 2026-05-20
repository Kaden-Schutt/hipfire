// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void conv1d_decode_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,   // [4 × n_channels]
    float* __restrict__ state,          // [n_channels × 3]
    int n_channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float x = input[c];
    float s0 = state[c * 3];
    float s1 = state[c * 3 + 1];
    float s2 = state[c * 3 + 2];

    // weight layout: [n_channels, 4] from safetensors [channels, 1, kernel_size]
    // PyTorch conv1d: weight[0] = oldest, weight[3] = newest (current)
    output[c] = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
              + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

    state[c * 3 + 2] = s1;
    state[c * 3 + 1] = s0;
    state[c * 3]     = x;
}
