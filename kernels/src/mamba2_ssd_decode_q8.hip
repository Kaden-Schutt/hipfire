// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.
//
// Mamba-2 SSD single-token decode with Q8 persistent SSM state.
//
// The arithmetic mirrors mamba2_ssd_decode_f32, but the recurrent state row is
// stored as signed int8 plus one f32 scale per (head, head_dim position). The
// output uses the updated f32 state before requantization; the q8 store is the
// hand-off state for the next token.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float softplus_stable_q8(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ signed char quantize_i8_sym(float v, float inv_scale) {
    float qf = floorf(v * inv_scale + 0.5f);
    qf = fminf(fmaxf(qf, -127.0f), 127.0f);
    return (signed char)qf;
}

extern "C" __global__ void mamba2_ssd_decode_q8(
    float* __restrict__ y,                 // [num_heads * head_dim]
    signed char* __restrict__ state_q8,    // [num_heads * head_dim * state_size]
    float* __restrict__ state_scales,      // [num_heads * head_dim]
    const float* __restrict__ x,           // [num_heads * head_dim]
    const float* __restrict__ B,           // [n_groups * state_size]
    const float* __restrict__ C,           // [n_groups * state_size]
    const float* __restrict__ dt_raw,      // [num_heads]
    const float* __restrict__ A_log,       // [num_heads]
    const float* __restrict__ D,           // [num_heads]
    const float* __restrict__ dt_bias,     // [num_heads]
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    float dt_min,
    float dt_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    int grp = head / (num_heads / n_groups);

    float a = -expf(A_log[head]);
    float dt = softplus_stable_q8(dt_raw[head] + dt_bias[head]);
    dt = fminf(fmaxf(dt, dt_min), dt_max);
    float dA = expf(dt * a);

    float xp = x[idx];
    float dbx_scale = dt * xp;
    const float* Bg = B + grp * state_size;
    const float* Cg = C + grp * state_size;
    signed char* qrow = state_q8 + (size_t)idx * state_size;
    float old_scale = state_scales[idx];

    float acc = 0.0f;
    float amax = 0.0f;
    for (int n = 0; n < state_size; ++n) {
        float old_s = old_scale * (float)qrow[n];
        float s = dA * old_s + dbx_scale * Bg[n];
        acc += Cg[n] * s;
        amax = fmaxf(amax, fabsf(s));
    }
    y[idx] = acc + D[head] * xp;

    float new_scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    for (int n = 0; n < state_size; ++n) {
        float old_s = old_scale * (float)qrow[n];
        float s = dA * old_s + dbx_scale * Bg[n];
        qrow[n] = quantize_i8_sym(s, inv_scale);
    }
    state_scales[idx] = new_scale;
}
