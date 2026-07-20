// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_cache_init_f32(
    float* __restrict__ cos_cache,
    float* __restrict__ sin_cache,
    int max_seq,
    int head_dim,
    float freq_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int n = max_seq * half;
    if (idx >= n) return;

    int pos = idx / half;
    int i = idx - pos * half;
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    cos_cache[idx] = cosf(val);
    sin_cache[idx] = sinf(val);
}

extern "C" __global__ void lfm_qk_norm_rope_cached_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    float eps
) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    bool is_q = head < n_heads_q;
    int local_head = is_q ? head : head - n_heads_q;
    float* x = (is_q ? q : k) + local_head * head_dim;
    const float* weight = is_q ? q_weight : k_weight;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)head_dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float normalized = x[i] * weight[i] * rms;
        x[i] = normalized;
        sdata[i] = normalized;
    }
    __syncthreads();

    int half = head_dim / 2;
    int pos = pos_buf[0];
    int cache_base = pos * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float cos_val = cos_cache[cache_base + i];
        float sin_val = sin_cache[cache_base + i];
        float x0 = sdata[i];
        float x1 = sdata[i + half];
        x[i] = x0 * cos_val - x1 * sin_val;
        x[i + half] = x0 * sin_val + x1 * cos_val;
    }
}
