// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 3D mrope, half-split convention. Twin of rope_partial_halfsplit_f32,
// differing ONLY in which position each frequency index reads.
//
// HF apply_interleaved_mrope produces band layout [THWTHW...TT]: frequency d
// reads H if (d%3==1 && d < 3*sec_h), W if (d%3==2 && d < 3*sec_w), else T.
// A chunked [TTT...HHH...WWW] layout is WRONG.
//
// pos_buf3 holds exactly 3 int32: [t, h, w].
extern "C" __global__ void rope_mrope_halfsplit_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf3,   // [3] = (t, h, w)
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base,
    int sec_h,      // mrope_section[1]
    int sec_w       // mrope_section[2]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int m = i % 3;
    int axis = (m == 1 && i < 3 * sec_h) ? 1
             : (m == 2 && i < 3 * sec_w) ? 2
             : 0;
    int pos = pos_buf3[axis];

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_a - q1 * sin_a;
        q[base + i + half] = q0 * sin_a + q1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_a - k1 * sin_a;
        k[base + i + half] = k0 * sin_a + k1 * cos_a;
    }
}
