// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 YaRN-aware tail RoPE — BATCHED variant (Phase B2, 2026-05-18).
//
// Batched twin of `rope_tail_yarn_interleaved_f32`. Each batch row b has
// its own absolute position from positions[b]; same YaRN blend semantics
// applied per (batch, dim-pair) cell. Rotation applies to the LAST n_rot
// dims of each head.
//
// Grid:  [ceil(n_rot/2 / block), batch_size, 1]
// Block: [block, 1, 1]
//
// Byte-identical to rope_tail_yarn_interleaved_f32 at batch_size == 1
// with positions[0] == pos_buf[0].

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_yarn_interleaved_batched_f32(
    float* __restrict__ q,           // [batch, n_heads_q, head_dim]
    float* __restrict__ k,           // [batch, n_heads_k, head_dim]
    const int* __restrict__ positions,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float corr_low,
    float corr_high,
    int inverse,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b];
    int idx_pair = i * 2;

    float freq = 1.0f / powf(freq_base, (float)idx_pair / (float)n_rot);
    float theta_extrap = (float)pos * freq;
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;

    if (ext_factor != 0.0f) {
        float y = ((float)i - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
    float cos_a = cosf(theta) * mscale;
    float sin_a = sin_sign * sinf(theta) * mscale;

    int tail_off = head_dim - n_rot;
    int d_even = tail_off + idx_pair;
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
