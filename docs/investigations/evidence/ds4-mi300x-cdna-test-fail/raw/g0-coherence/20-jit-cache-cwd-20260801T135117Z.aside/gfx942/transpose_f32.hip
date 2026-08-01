// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void transpose_f32(
    const float* __restrict__ in_data,
    float* __restrict__ out_data,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int r = idx / cols;
    int c = idx % cols;
    out_data[c * rows + r] = in_data[idx];
}
