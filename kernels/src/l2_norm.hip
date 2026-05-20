// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void l2_norm_f32(
    float* __restrict__ x, int n_heads, int head_dim, float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    float* head = x + h * head_dim;

    float sq_sum = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        sq_sum += head[i] * head[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);

    float inv_norm = rsqrtf(sq_sum + eps);
    for (int i = tid; i < head_dim; i += 32)
        head[i] *= inv_norm;
}
