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
//
// One block per head (same layout as mamba2_ssd_decode_f32): the group-shared
// B/C row is staged in LDS and the per-head scalars computed once, so the two
// state passes (acc/amax, then requantize) read B/C from LDS instead of global.
// Arithmetic is unchanged → bit-faithful to the old flat-grid q8 kernel.

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
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int grp = head / (num_heads / n_groups);

    extern __shared__ float lds[];
    float* Bs = lds;                 // [state_size]
    float* Cs = lds + state_size;    // [state_size]
    __shared__ float sh_dt, sh_dA, sh_D;

    const float* Bg = B + grp * state_size;
    const float* Cg = C + grp * state_size;
    for (int n = threadIdx.x; n < state_size; n += blockDim.x) {
        Bs[n] = Bg[n];
        Cs[n] = Cg[n];
    }
    if (threadIdx.x == 0) {
        float a = -expf(A_log[head]);
        float dt = softplus_stable_q8(dt_raw[head] + dt_bias[head]);
        dt = fminf(fmaxf(dt, dt_min), dt_max);
        sh_dt = dt;
        sh_dA = expf(dt * a);
        sh_D = D[head];
    }
    __syncthreads();

    float dt = sh_dt;
    float dA = sh_dA;
    float Dh = sh_D;

    for (int p = threadIdx.x; p < head_dim; p += blockDim.x) {
        int idx = head * head_dim + p;
        float xp = x[idx];
        float dbx_scale = dt * xp;
        signed char* qrow = state_q8 + (size_t)idx * state_size;
        float old_scale = state_scales[idx];

        float acc = 0.0f;
        float amax = 0.0f;
        for (int n = 0; n < state_size; ++n) {
            float old_s = old_scale * (float)qrow[n];
            float s = dA * old_s + dbx_scale * Bs[n];
            acc += Cs[n] * s;
            amax = fmaxf(amax, fabsf(s));
        }
        y[idx] = acc + Dh * xp;

        float new_scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        for (int n = 0; n < state_size; ++n) {
            float old_s = old_scale * (float)qrow[n];
            float s = dA * old_s + dbx_scale * Bs[n];
            qrow[n] = quantize_i8_sym(s, inv_scale);
        }
        state_scales[idx] = new_scale;
    }
}
