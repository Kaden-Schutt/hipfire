// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// GQA-aware split-K flash decode partial: one block per (kv_head, chunk).
// Each block reads the chunk's K/V ONCE and computes all `kv_group` query
// heads that share this KV head, so the long F32 KV cache is traversed
// n_kv_heads× instead of n_heads× (6× less for 12:2). partials layout is
// identical to attention_flash so attention_flash_reduce is reused.
extern "C" __global__ void attention_flash_gqa_partial(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partials,       // [n_heads, n_chunks, 2 + head_dim]
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    int max_seq, float scale, int chunk_size
) {
    const int kv_h = blockIdx.x;
    const int chunk_id = blockIdx.y;
    if (kv_h >= n_kv_heads) return;
    const int kv_group = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > seq_len) chunk_end = seq_len;
    if (chunk_start >= seq_len) return;
    const int chunk_len = chunk_end - chunk_start;

    extern __shared__ float sdata[];
    float* scores = sdata;                 // [chunk_size]
    float* ws = sdata + chunk_size;        // [nthreads]
    const int n_chunks = gridDim.y;
    const int stride = 2 + head_dim;
    const int kd = n_kv_heads * head_dim;  // cache row stride

    for (int g = 0; g < kv_group; g++) {
        const int h = kv_h * kv_group + g;
        const float* q_head = q + h * head_dim;

        float local_max = -1e30f;
        for (int t = tid; t < chunk_len; t += nthreads) {
            const float* k_t = k_cache + (chunk_start + t) * kd + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q_head[d] * k_t[d];
            float s = dot * scale;
            scores[t] = s;
            local_max = fmaxf(local_max, s);
        }
        ws[tid] = local_max; __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]); __syncthreads(); }
        float max_val = ws[0]; __syncthreads();

        float local_sum = 0.0f;
        for (int t = tid; t < chunk_len; t += nthreads) { float e = expf(scores[t] - max_val); scores[t] = e; local_sum += e; }
        ws[tid] = local_sum; __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] += ws[tid + s]; __syncthreads(); }
        float sum_val = ws[0]; __syncthreads();

        float* partial = partials + (h * n_chunks + chunk_id) * stride;
        for (int d = tid; d < head_dim; d += nthreads) {
            float val = 0.0f;
            for (int t = 0; t < chunk_len; t++) val += scores[t] * v_cache[(chunk_start + t) * kd + kv_h * head_dim + d];
            partial[2 + d] = val;
        }
        if (tid == 0) { partial[0] = max_val; partial[1] = sum_val; }
        __syncthreads();
    }
}
