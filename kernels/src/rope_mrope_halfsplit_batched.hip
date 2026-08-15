// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched 3D mrope, half-split convention. Twin of
// rope_partial_halfsplit_batched_f32; see rope_mrope_halfsplit.hip for the
// band-mapping rationale.
//
// Grid: [ceil(n_rot/2 / block), batch_size, 1]
// Q/K are [batch_size * n_heads * head_dim] row-major.
// positions is [batch_size][3] row-major: positions[b*3 + axis].
extern "C" __global__ void rope_mrope_halfsplit_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,  // [batch_size][3]
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base,
    int batch_size,
    int pos_offset,
    int sec_h,
    int sec_w
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (i >= half || b >= batch_size) return;

    int m = i % 3;
    int axis = (m == 1 && i < 3 * sec_h) ? 1
             : (m == 2 && i < 3 * sec_w) ? 2
             : 0;
    int pos = positions[b * 3 + axis] + pos_offset;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    for (int h = 0; h < n_heads_q; h++) {
        int base = (b * n_heads_q + h) * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_a - q1 * sin_a;
        q[base + i + half] = q0 * sin_a + q1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = (b * n_heads_k + h) * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_a - k1 * sin_a;
        k[base + i + half] = k0 * sin_a + k1 * cos_a;
    }
}
