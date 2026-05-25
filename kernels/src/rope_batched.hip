// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,  // [batch_size]
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    float freq_base,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // freq index
    int b = blockIdx.y;  // batch index
    int half = head_dim / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b];
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;

    for (int h = 0; h < n_heads_q; h++) {
        int base = b * q_stride + h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_val - q1 * sin_val;
        q[base + i + half] = q0 * sin_val + q1 * cos_val;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = b * k_stride + h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_val - k1 * sin_val;
        k[base + i + half] = k0 * sin_val + k1 * cos_val;
    }
}
