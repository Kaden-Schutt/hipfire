// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused Q-head + K-head RMSNorm for the decode attention preamble.
//
// The Qwen3.5 (and kin) attention preamble applies a per-head RMSNorm to the
// query heads (weight = q_norm[head_dim]) and, independently, to the key heads
// (weight = k_norm[head_dim]) before RoPE. In the baseline that is two separate
// `rmsnorm_f32` launches (grid = n_q_heads, then grid = n_kv_heads). This kernel
// merges them into ONE dispatch: the grid covers n_q_heads + n_kv_heads blocks,
// and a workgroup-id branch selects the Q side (blockIdx.x < n_q_heads) or the
// K side (the remainder), each with its own x / weight / out pointer.
//
// VALUE-PRESERVING: each head's normalization is fully independent, and this
// body is a byte-for-byte copy of `rmsnorm_f32` — identical reduction order
// (block-wide tree in LDS), identical `rsqrtf(sum/n + eps)`, identical
// `x * weight * rms` multiplication order. The only change is *which* row a
// given block touches (a local head index) and *which* buffer/weight it reads.
// Both sides share the same feature length `n` (= head_dim) so the launch
// geometry (block_size, shared_mem) is identical to the two split launches.
extern "C" __global__ void rmsnorm_qk_fused(
    const float* __restrict__ xq,
    const float* __restrict__ wq,
    float* __restrict__ outq,
    const float* __restrict__ xk,
    const float* __restrict__ wk,
    float* __restrict__ outk,
    int n,
    int n_q_heads,
    float eps
) {
    extern __shared__ float sdata[];

    // Select side by workgroup id. Q heads occupy [0, n_q_heads); K heads
    // occupy [n_q_heads, n_q_heads + n_kv_heads).
    const float* __restrict__ x;
    const float* __restrict__ weight;
    float* __restrict__ out;
    int head = blockIdx.x;
    if (head < n_q_heads) {
        x = xq;
        weight = wq;
        out = outq;
    } else {
        head -= n_q_heads;
        x = xk;
        weight = wk;
        out = outk;
    }

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[head * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = head * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
}
