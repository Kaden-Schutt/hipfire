// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Tail-only RoPE, INTERLEAVED pair convention (DeepSeek V4 upstream).
//
// Upstream DeepSeek V4 uses `torch.view_as_complex(x.unflatten(-1, (-1, 2)))`
// which treats consecutive dim pairs (2i, 2i+1) as (real, imag) of a
// complex number. The rotation is x' = x * e^(i·angle):
//
//   for i in 0..n_rot/2:
//     d_even = tail_off + 2*i
//     d_odd  = tail_off + 2*i + 1
//     x[d_even] = x[d_even] * cos(angle) - x[d_odd] * sin(angle)
//     x[d_odd]  = x[d_even] * sin(angle) + x[d_odd] * cos(angle)
//     angle    = pos * (1 / base^(2i/n_rot))
//
// Note: the HF `rotate_half` variant pairs (i, i + n_rot/2) — a
// different memory layout but equivalent algebra. The DeepSeek V4 training
// uses interleaved, so for cross-checkpoint correctness we need this
// variant.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_interleaved_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int pos = *pos_buf;
    float freq  = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int tail_off = head_dim - n_rot;
    int d_even = tail_off + 2 * i;
    int d_odd  = d_even + 1;

    for (int h = 0; h < n_heads_q; ++h) {
        int base = h * head_dim;
        float x0 = q[base + d_even];
        float x1 = q[base + d_odd];
        q[base + d_even] = x0 * cos_a - x1 * sin_a;
        q[base + d_odd]  = x0 * sin_a + x1 * cos_a;
    }
    for (int h = 0; h < n_heads_k; ++h) {
        int base = h * head_dim;
        float x0 = k[base + d_even];
        float x1 = k[base + d_odd];
        k[base + d_even] = x0 * cos_a - x1 * sin_a;
        k[base + d_odd]  = x0 * sin_a + x1 * cos_a;
    }
}
