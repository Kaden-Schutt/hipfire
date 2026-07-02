// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 SSD (selective state-space duality) — single-token DECODE step, f32.
// The reference floor for nemotron_h Mamba-2 mixer layers. Mirrors the CPU
// oracle `hipfire_arch_nemotron::ssd::ssd_decode_step` exactly (gpu-vs-cpu
// tested).
//
//   dt = clamp(softplus(dt_raw[head] + dt_bias[head]), dt_min, dt_max)
//   A  = -exp(A_log[head])                       // per-head scalar decay
//   dA = exp(dt * A)
//   state[(head*HD + p)*N + n] = dA*state + (dt*B[n])*x[head*HD+p]
//   y[head*HD + p]             = Σ_n C[n]*state  + D[head]*x[head*HD+p]
//
// B/C are shared across the heads of a group: group = head / (num_heads/n_groups)
// (mamba-ssm fast path / HF decode convention).
//
// Layout: ONE BLOCK PER HEAD. The group-shared B/C row (2*N floats) is staged
// into LDS once per block, then reused by every head_dim thread instead of being
// re-read from global N times per channel (the dominant decode-bandwidth waste —
// only G*N unique B/C values exist but the flat-grid kernel read H*P*N of them).
// The per-head scalars (dt, dA, D) are computed once by lane 0 rather than
// recomputed by all P channels. Arithmetic order is unchanged → bit-faithful to
// the CPU oracle and the old flat-grid kernel.

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
    // One block per head; threads cover the head_dim channels (strided).
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int grp = head / (num_heads / n_groups);

    // Stage the group-shared B/C row in LDS (2*state_size floats), then compute
    // the per-head scalars once in lane 0.
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
        float dt = softplus_stable(dt_raw[head] + dt_bias[head]);
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
        float* srow = state + (size_t)idx * state_size;

        float acc = 0.0f;
        for (int n = 0; n < state_size; ++n) {
            float s = dA * srow[n] + dbx_scale * Bs[n];
            srow[n] = s;
            acc += Cs[n] * s;
        }
        y[idx] = acc + Dh * xp;
    }
}
