// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 SSD (selective state-space duality) — single-token DECODE step, f32.
// The reference floor for nemotron_h Mamba-2 mixer layers. Mirrors the CPU
// oracle `hipfire_arch_nemotron::ssd::ssd_decode_step` exactly (gpu-vs-cpu
// tested). One thread per (head, head_dim position); it loops over the per-head
// SSM state width N (state_size), reading the group-shared B/C.
//
//   dt = clamp(softplus(dt_raw[head] + dt_bias[head]), dt_min, dt_max)
//   A  = -exp(A_log[head])                       // per-head scalar decay
//   dA = exp(dt * A)
//   state[(head*HD + p)*N + n] = dA*state + (dt*B[n])*x[head*HD+p]
//   y[head*HD + p]             = Σ_n C[n]*state  + D[head]*x[head*HD+p]
//
// B/C are shared across the heads of a group: group = head % n_groups (HF tile).

#include <hip/hip_runtime.h>

__device__ __forceinline__ float softplus_stable(float x) {
    // log(1 + exp(x)), stable for large x
    return x > 20.0f ? x : log1pf(expf(x));
}

extern "C" __global__ void mamba2_ssd_decode_f32(
    float* __restrict__ y,            // [num_heads * head_dim]
    float* __restrict__ state,        // [num_heads * head_dim * state_size]
    const float* __restrict__ x,      // [num_heads * head_dim]
    const float* __restrict__ B,      // [n_groups * state_size]
    const float* __restrict__ C,      // [n_groups * state_size]
    const float* __restrict__ dt_raw, // [num_heads]
    const float* __restrict__ A_log,  // [num_heads]
    const float* __restrict__ D,      // [num_heads]
    const float* __restrict__ dt_bias,// [num_heads]
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    float dt_min,
    float dt_max
) {
    // One thread per (head, p) inner-dim element.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    int p = idx - head * head_dim;

    // HF expands B/C groups via `B.repeat(num_heads/n_groups)` — a tile, so
    // head h uses group h % n_groups (NOT the interleave h/(num_heads/n_groups)).
    int grp = head % n_groups;

    float a = -expf(A_log[head]);
    float dt = softplus_stable(dt_raw[head] + dt_bias[head]);
    dt = fminf(fmaxf(dt, dt_min), dt_max);
    float dA = expf(dt * a);

    float xp = x[idx];
    float dbx_scale = dt * xp;

    const float* Bg = B + grp * state_size;
    const float* Cg = C + grp * state_size;
    float* srow = state + (size_t)idx * state_size;

    float acc = 0.0f;
    for (int n = 0; n < state_size; ++n) {
        float s = dA * srow[n] + dbx_scale * Bg[n];
        srow[n] = s;
        acc += Cg[n] * s;
    }
    y[idx] = acc + D[head] * xp;
}
