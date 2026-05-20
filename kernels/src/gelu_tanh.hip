// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void gelu_tanh_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanhf(inner));
}
