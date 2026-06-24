// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// ReLU-squared activation: out = max(0, x)^2. nemotron_h dense MLP uses
// `mlp_hidden_act: relu2` (down(relu(up(x))^2)) instead of SwiGLU. Elementwise,
// one thread per element.

#include <hip/hip_runtime.h>

extern "C" __global__ void relu2_f32(
    float* __restrict__ out,
    const float* __restrict__ in,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = fmaxf(in[i], 0.0f);
    out[i] = v * v;
}
