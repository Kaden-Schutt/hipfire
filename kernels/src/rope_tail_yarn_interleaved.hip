// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 YaRN-aware tail RoPE (INTERLEAVED pair convention, single-tensor).
//
// Mirrors antirez/ds4 rope_tail_ext_inplace (ds4.c:4694) for layers that
// need YaRN scaling. Caller picks per-layer freq_base / freq_scale /
// ext_factor / attn_factor / corr_low / corr_high based on whether the
// layer is compressed (ratio>0) or dense (ratio=0).
//
// Constants for DeepSeek V4 (from antirez constants):
//   Dense layer (ratio=0):
//     freq_base = 10000, freq_scale = 1.0, ext_factor = 0.0,
//     attn_factor = 1.0
//   Compressed layer (ratio > 0):
//     freq_base = 160000, freq_scale = 1/16 = 0.0625, ext_factor = 1.0,
//     attn_factor = 1.0 / (1.0 + 0.1 * log(1/freq_scale))  = 1.0 / (1.0 + 0.1 * log(16))
//                 ≈ 1.0 / 1.27726 ≈ 0.7830
//   But then the inner mscale multiply: mscale *= 1.0 + 0.1*log(1/freq_scale)
//   So the net attn_factor seen by rotation = original_attn_factor * (1+...) = 1.0.
//   Caller passes the COMPUTED FINAL attn_factor (1.0 for both cases).
//
// YaRN ramp blend per dim-pair index i:
//   ramp = yarn_ramp(corr_low, corr_high, i) * ext_factor
//   theta = theta_interp * (1 - ramp) + theta_extrap * ramp
// where:
//   theta_extrap = pos * freq_base^(-2i/n_rot)
//   theta_interp = freq_scale * theta_extrap
//
// For ext_factor = 0: theta = theta_interp (standard interpolation).
// For ext_factor = 1: blend between interp (long-context) and extrap
// (short-context) per-dim.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_yarn_interleaved_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
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
    int inverse  // 0 = forward, 1 = inverse (negate sin)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int pos = *pos_buf;
    int idx_pair = i * 2;  // dim-pair start (even index)

    // Standard RoPE freq for this pair.
    float freq = 1.0f / powf(freq_base, (float)idx_pair / (float)n_rot);
    float theta_extrap = (float)pos * freq;
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;

    if (ext_factor != 0.0f) {
        // YaRN ramp: linear blend between corr_low and corr_high.
        float y = ((float)i - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        // mscale absorbs the log(1/freq_scale) correction.
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
    float cos_a = cosf(theta) * mscale;
    float sin_a = sin_sign * sinf(theta) * mscale;

    int tail_off = head_dim - n_rot;
    int d_even = tail_off + idx_pair;
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
