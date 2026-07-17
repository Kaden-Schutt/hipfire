// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact Qwen3.5-27B full-attention normalization producer for gfx1201.
//
// The stock decode path materializes Q/Gate with deinterleave_f32, then runs
// separate Q and K rmsnorm_f32 dispatches.  This fixed-shape kernel performs
// those three operations in one head-local dispatch while deliberately leaving
// RoPE in the stock kernel.  Keeping RoPE separate preserves its one-frequency-
// per-dimension execution order; each RMS workgroup below is byte-for-byte the
// same 256-thread reduction tree used by rmsnorm_f32 for head_dim=256.
//
// Admitted shape: Qwen3.5-27B, Q=24 heads, K=4 heads, head_dim=256.
extern "C" __launch_bounds__(256, 1)
__global__ void qwen35_fa_norm_prep_gfx1201(
    const float* __restrict__ q_interleaved,
    float* __restrict__ q,
    float* __restrict__ gate,
    float* __restrict__ k,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    float eps)
{
    constexpr int NQ = 24;
    constexpr int HD = 256;

    __shared__ float sdata[HD];

    int head = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_q = head < NQ;

    float v;
    const float* weight;
    float* out;
    if (is_q) {
        const int src = head * HD * 2 + tid;
        v = q_interleaved[src];
        gate[head * HD + tid] = q_interleaved[src + HD];
        weight = q_weight;
        out = q + head * HD;
    } else {
        head -= NQ;
        out = k + head * HD;
        v = out[tid];
        weight = k_weight;
    }

    sdata[tid] = v * v;
    __syncthreads();
    for (int s = HD / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(sdata[0] / (float)HD + eps);
    out[tid] = v * weight[tid] * rms;
}
