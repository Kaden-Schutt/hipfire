// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
extern "C" __global__ void rope_partial_interleaved_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,  // device buffer, single i32 — graph-safe
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,        // number of dims to rotate (64 for Qwen3.5)
    float freq_base
) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_rot / 2;
    if (pair >= n_pairs) return;

    // Interleaved: pair i rotates dims (2*i, 2*i+1)
    int d0 = pair * 2;
    int d1 = pair * 2 + 1;

    int pos = *pos_buf;
    // Frequency for this pair (same formula as standard RoPE but indexed by pair)
    float freq = 1.0f / powf(freq_base, (float)(2 * pair) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float v0 = q[base + d0];
        float v1 = q[base + d1];
        q[base + d0] = v0 * cos_a - v1 * sin_a;
        q[base + d1] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float v0 = k[base + d0];
        float v1 = k[base + d1];
        k[base + d0] = v0 * cos_a - v1 * sin_a;
        k[base + d1] = v0 * sin_a + v1 * cos_a;
    }
}
