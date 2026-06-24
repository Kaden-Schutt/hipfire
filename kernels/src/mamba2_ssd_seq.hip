// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 SSD (selective state-space duality) — multi-token PREFILL scan, f32.
// N6 throughput kernel for nemotron_h: instead of `seq_len` separate
// `mamba2_ssd_decode_f32` launches (O(prompt_len) launch-bound), ONE launch
// processes the whole prompt. One thread per (head, head_dim position) walks the
// sequence in registers/global, carrying the per-channel SSM state forward — so
// it is bit-faithful to the sequential decode (`ssd::ssd_sequence`, i.e. the
// decode step repeated). The genuinely-parallel intra-chunk matmul
// decomposition (`ssd::ssd_chunked`, C·B Gram + L decay) is a later optimisation
// layered on this correct floor.
//
// Per position t, per (head, p):
//   dt = clamp(softplus(dt_raw[t,head] + dt_bias[head]), dt_min, dt_max)
//   A  = -exp(A_log[head]);  dA = exp(dt * A)
//   state[n] = dA*state[n] + (dt*B[t,n])*x[t,head,p]      // recurrence carry
//   y[t,head,p] = Σ_n C[t,n]*state[n] + D[head]*x[t,head,p]
//
// `state` is updated in place to the post-sequence value so prefill hands off to
// the single-token decode cleanly. B/C use the interleave group
// head/(num_heads/n_groups) (mamba-ssm fast path / HF decode convention),
// matching `mamba2_ssd_decode.hip`.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float softplus_stable(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

extern "C" __global__ void mamba2_ssd_seq_f32(
    float* __restrict__ y,             // [seq * num_heads * head_dim]
    float* __restrict__ state,         // [num_heads * head_dim * state_size] (in/out)
    const float* __restrict__ x,       // [seq * num_heads * head_dim]
    const float* __restrict__ B,       // [seq * n_groups * state_size]
    const float* __restrict__ C,       // [seq * n_groups * state_size]
    const float* __restrict__ dt_raw,  // [seq * num_heads]
    const float* __restrict__ A_log,   // [num_heads]
    const float* __restrict__ D,       // [num_heads]
    const float* __restrict__ dt_bias, // [num_heads]
    int seq,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    float dt_min,
    float dt_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // (head, p)
    int total = num_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    int grp = head / (num_heads / n_groups);

    float a = -expf(A_log[head]);
    float Dh = D[head];
    float dtb = dt_bias[head];

    const int xd = num_heads * head_dim; // per-position x/y stride
    const int bd = n_groups * state_size; // per-position B/C stride
    float* srow = state + (size_t)idx * state_size;

    for (int t = 0; t < seq; ++t) {
        float dt = softplus_stable(dt_raw[t * num_heads + head] + dtb);
        dt = fminf(fmaxf(dt, dt_min), dt_max);
        float dA = expf(dt * a);

        float xp = x[(size_t)t * xd + idx];
        float scale = dt * xp;
        const float* Bg = B + (size_t)t * bd + grp * state_size;
        const float* Cg = C + (size_t)t * bd + grp * state_size;

        float acc = 0.0f;
        for (int n = 0; n < state_size; ++n) {
            float s = dA * srow[n] + scale * Bg[n];
            srow[n] = s;
            acc += Cg[n] * s;
        }
        y[(size_t)t * xd + idx] = acc + Dh * xp;
    }
}
