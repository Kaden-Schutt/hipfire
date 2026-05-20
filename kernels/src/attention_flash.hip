// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_flash_partial(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partials,       // [n_heads, n_chunks, 2 + head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale,
    int chunk_size
) {
    const int h = blockIdx.x;
    const int chunk_id = blockIdx.y;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > seq_len) chunk_end = seq_len;
    if (chunk_start >= seq_len) return;
    const int chunk_len = chunk_end - chunk_start;

    const float* q_head = q + h * head_dim;

    // Compute scores for this chunk
    extern __shared__ float sdata[];
    float* scores = sdata;  // [chunk_size]

    float local_max = -1e30f;
    for (int t = tid; t < chunk_len; t += nthreads) {
        int pos = chunk_start + t;
        const float* k_t = k_cache + pos * n_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_t[d];
        }
        float s = dot * scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    // Reduce max
    float* ws = sdata + chunk_size;
    ws[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
        __syncthreads();
    }
    float max_val = ws[0];
    __syncthreads();

    // Exp + sum
    float local_sum = 0.0f;
    for (int t = tid; t < chunk_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    ws[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] += ws[tid + s];
        __syncthreads();
    }
    float sum_val = ws[0];
    __syncthreads();

    // Weighted sum of values for this chunk
    int n_chunks = gridDim.y;
    int partial_stride = 2 + head_dim;  // [max, sum, output[head_dim]]
    float* partial = partials + (h * n_chunks + chunk_id) * partial_stride;

    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < chunk_len; t++) {
            val += scores[t] * v_cache[(chunk_start + t) * n_kv_heads * head_dim + kv_h * head_dim + d];
        }
        partial[2 + d] = val;  // unnormalized weighted sum
    }
    if (tid == 0) {
        partial[0] = max_val;
        partial[1] = sum_val;
    }
}

// Phase 2: reduce partials across chunks using online softmax correction
extern "C" __global__ void attention_flash_reduce(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int n_heads,
    int n_chunks,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    int partial_stride = 2 + head_dim;

    // Online softmax: combine chunks
    float global_max = -1e30f;
    float global_sum = 0.0f;

    // First pass: find global max
    for (int c = 0; c < n_chunks; c++) {
        const float* p = partials + (h * n_chunks + c) * partial_stride;
        float m = p[0];
        if (m > global_max) global_max = m;
    }

    // Second pass: compute corrected sum and weighted output
    float* out_head = out + h * head_dim;
    // Initialize output to zero
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_head[d] = 0.0f;
    }
    __syncthreads();

    for (int c = 0; c < n_chunks; c++) {
        const float* p = partials + (h * n_chunks + c) * partial_stride;
        float chunk_max = p[0];
        float chunk_sum = p[1];
        float correction = expf(chunk_max - global_max);
        float corrected_sum = chunk_sum * correction;
        global_sum += corrected_sum;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_head[d] += p[2 + d] * correction;
        }
        __syncthreads();
    }

    // Final normalize
    float inv_sum = 1.0f / global_sum;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_head[d] *= inv_sum;
    }
}
