// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1201 batch-1 DS4 decode screen: split the joint SWA + top-K attention
// position axis across blocks, then merge per-chunk softmax state. The current
// one-block-per-head route exposes only H=24/16 blocks per TP3 rank; this route
// makes H*ceil(640/chunk_size) independent blocks available to 64 CUs.

#include <hip/hip_runtime.h>
#include <math.h>

#define DS4_SPLIT_MAX_CHUNK 256
#define DS4_SPLIT_MAX_CHUNKS 16
#define DS4_SPLIT_HEAD_DIM 512

extern "C" __global__ __launch_bounds__(256, 2)
void deepseek4_attn_swa_topk_split_partial_gfx1201(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_k,
    const float* __restrict__ topk_v,
    const float* __restrict__ attn_sink,
    float*       __restrict__ partials, // [H, n_chunks, 2 + D]
    const int*   __restrict__ n_valid_swa_buf,
    const int*   __restrict__ n_active_topk_buf,
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window,
    int n_chunks,
    int chunk_size
) {
    const int h = blockIdx.x;
    const int chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (h >= n_heads || chunk >= n_chunks) return;

    const int n_valid_swa = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const int start = chunk * chunk_size;
    const int end = min(start + chunk_size, n_total);
    const int len = max(0, end - start);
    const int stride = 2 + head_dim;
    float* partial = partials + ((size_t)h * n_chunks + chunk) * stride;

    __shared__ float q_lds[DS4_SPLIT_HEAD_DIM];
    __shared__ float scores[DS4_SPLIT_MAX_CHUNK];
    __shared__ float reduce[256];

    for (int d = tid; d < head_dim; d += blockDim.x)
        q_lds[d] = q[(size_t)h * head_dim + d];
    __syncthreads();

    if (len == 0) {
        for (int d = tid; d < head_dim; d += blockDim.x)
            partial[2 + d] = 0.0f;
        if (tid == 0) {
            partial[0] = -INFINITY;
            partial[1] = 0.0f;
        }
        return;
    }

    float local_max = -INFINITY;
    if (tid < len) {
        const int p = start + tid;
        const float* k_col;
        int k_stride;
        int col;
        if (p < n_valid_swa) {
            k_col = swa_k;
            k_stride = swa_window;
            col = p;
        } else {
            k_col = topk_k;
            k_stride = topk_window;
            col = p - n_valid_swa;
        }
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += q_lds[d] * k_col[(size_t)d * k_stride + col];
        local_max = dot * rsqrtf((float)head_dim);
        scores[tid] = local_max;
    }
    if (chunk == 0 && tid == 0)
        local_max = fmaxf(local_max, attn_sink[h]);
    reduce[tid] = local_max;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) reduce[tid] = fmaxf(reduce[tid], reduce[tid + step]);
        __syncthreads();
    }
    const float chunk_max = reduce[0];

    float local_sum = 0.0f;
    if (tid < len) {
        const float value = expf(scores[tid] - chunk_max);
        scores[tid] = value;
        local_sum = value;
    }
    if (chunk == 0 && tid == 0)
        local_sum += expf(attn_sink[h] - chunk_max);
    reduce[tid] = local_sum;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) reduce[tid] += reduce[tid + step];
        __syncthreads();
    }

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float value = 0.0f;
        for (int local = 0; local < len; ++local) {
            const int p = start + local;
            const float v = p < n_valid_swa
                ? swa_v[(size_t)d * swa_window + p]
                : topk_v[(size_t)d * topk_window + (p - n_valid_swa)];
            value += scores[local] * v;
        }
        partial[2 + d] = value;
    }
    if (tid == 0) {
        partial[0] = chunk_max;
        partial[1] = reduce[0];
    }
}

extern "C" __global__ __launch_bounds__(512, 1)
void deepseek4_attn_swa_topk_split_reduce_gfx1201(
    const float* __restrict__ partials,
    float*       __restrict__ attn_out,
    int n_heads,
    int head_dim,
    int n_chunks
) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    if (h >= n_heads) return;

    const int stride = 2 + head_dim;
    const float* head_partials = partials + (size_t)h * n_chunks * stride;
    __shared__ float correction[DS4_SPLIT_MAX_CHUNKS];
    __shared__ float inverse_sum;

    if (tid == 0) {
        float global_max = -INFINITY;
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const float* partial = head_partials + chunk * stride;
            if (partial[1] > 0.0f)
                global_max = fmaxf(global_max, partial[0]);
        }
        float global_sum = 0.0f;
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const float* partial = head_partials + chunk * stride;
            const float corr = partial[1] > 0.0f
                ? expf(partial[0] - global_max)
                : 0.0f;
            correction[chunk] = corr;
            global_sum += partial[1] * corr;
        }
        inverse_sum = 1.0f / global_sum;
    }
    __syncthreads();

    if (tid < head_dim) {
        float value = 0.0f;
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const float* partial = head_partials + chunk * stride;
            value += partial[2 + tid] * correction[chunk];
        }
        attn_out[(size_t)h * head_dim + tid] = value * inverse_sum;
    }
}
