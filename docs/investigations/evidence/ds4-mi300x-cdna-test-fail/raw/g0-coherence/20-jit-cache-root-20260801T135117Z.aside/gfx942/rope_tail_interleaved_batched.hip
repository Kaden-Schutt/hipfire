// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Tail-only RoPE — BATCHED variant (Phase B2, 2026-05-18).
//
// Batched twin of `rope_tail_interleaved_f32`. Each batch row b has its
// own absolute position read from positions[b]; rotation applies to the
// LAST n_rot dims of each head (dims [head_dim - n_rot, head_dim)).
//
// Grid:  [ceil(n_rot/2 / block), batch_size, 1]
// Block: [block, 1, 1]
//
// Byte-identical to rope_tail_interleaved_f32 at batch_size == 1 with
// positions[0] == pos_buf[0].

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_interleaved_batched_f32(
    float* __restrict__ q,           // [batch, n_heads_q, head_dim]
    float* __restrict__ k,           // [batch, n_heads_k, head_dim]
    const int* __restrict__ positions,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b];
    float freq  = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int tail_off = head_dim - n_rot;
    int d_even = tail_off + 2 * i;
    int d_odd  = d_even + 1;

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;
    int q_base_b = b * q_stride;
    int k_base_b = b * k_stride;

    for (int h = 0; h < n_heads_q; ++h) {
        int base = q_base_b + h * head_dim;
        float x0 = q[base + d_even];
        float x1 = q[base + d_odd];
        q[base + d_even] = x0 * cos_a - x1 * sin_a;
        q[base + d_odd]  = x0 * sin_a + x1 * cos_a;
    }
    for (int h = 0; h < n_heads_k; ++h) {
        int base = k_base_b + h * head_dim;
        float x0 = k[base + d_even];
        float x1 = k[base + d_odd];
        k[base + d_even] = x0 * cos_a - x1 * sin_a;
        k[base + d_odd]  = x0 * sin_a + x1 * cos_a;
    }
}
