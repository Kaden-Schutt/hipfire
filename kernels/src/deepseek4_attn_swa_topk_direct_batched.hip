// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 indexer-extended SWA attention — direct top-K KV variant.
//
// This is the fused sibling of `deepseek4_attn_swa_topk_batched_f32` for
// ratio=4 indexed layers. It reads top-K slots directly from the compressed
// main KV cache using `topk_idx`, avoiding the separate
// `deepseek4_topk_kv_gather_batched_f32` staging pass.

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_TOTAL 1024
#define HEAD_DIM_LDS 512
#define MAX_WARPS 16

extern "C" __global__ __launch_bounds__(512, 1) void deepseek4_attn_swa_topk_direct_batched_f32(
    const float* __restrict__ q,                 // [batch, n_heads, head_dim]
    const float* __restrict__ swa_k,             // [batch, head_dim, swa_window]
    const float* __restrict__ swa_v,             // [batch, head_dim, swa_window]
    const float* __restrict__ kv_cache,          // [n_compressed, head_dim]
    const int*   __restrict__ topk_idx,          // [batch, topk_window]
    const float* __restrict__ attn_sink,         // [n_heads]
    const int*   __restrict__ n_valid_swa_arr,   // [batch]
    const int*   __restrict__ n_active_topk_arr, // [batch]
    float*       __restrict__ attn_out,          // [batch, n_heads, head_dim]
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window,
    int n_compressed,
    int batch_size
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= n_heads || b >= batch_size) return;

    const int n_valid_swa = n_valid_swa_arr[b];
    const int n_active_topk = n_active_topk_arr[b];
    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];

    const size_t q_base = (size_t)(b * n_heads + h) * head_dim;
    const size_t swa_base = (size_t)b * head_dim * swa_window;
    const size_t out_base = (size_t)(b * n_heads + h) * head_dim;
    const int* topk_row = topk_idx + (size_t)b * topk_window;

    if (t < head_dim) {
        q_lds[t] = q[q_base + t];
    }
    __syncthreads();

    for (int p = t; p < n_total; p += blockDim.x) {
        float s = 0.0f;
        if (p < n_valid_swa) {
            const float* k_col = swa_k + swa_base;
            const int col = p;
            for (int i = 0; i < head_dim; ++i) {
                s += q_lds[i] * k_col[i * swa_window + col];
            }
        } else {
            const int tk = p - n_valid_swa;
            const int idx = topk_row[tk];
            if (idx >= 0 && idx < n_compressed) {
                const float* k_row = kv_cache + (size_t)idx * head_dim;
                for (int i = 0; i < head_dim; ++i) {
                    s += q_lds[i] * k_row[i];
                }
            } else {
                s = -1e30f;
            }
        }
        scores_lds[p] = s * inv_scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off));
    }
    const int warp_id = t >> 5;
    const int lane_id = t & 31;
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_max;
    }
    __syncthreads();

    float max_score;
    if (warp_id == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float v = (lane_id < n_warps) ? warp_scratch[lane_id] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down(v, off));
        }
        if (lane_id == 0) {
            warp_scratch[0] = v;
        }
    }
    __syncthreads();
    max_score = warp_scratch[0];

    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();

    float thread_sum = 0.0f;
    for (int p = t; p < n_total; p += blockDim.x) {
        thread_sum += scores_lds[p];
    }

    const int warp_id4 = t >> 5;
    const int lane_id4 = t & 31;
    float ws = 0.0f;
    #pragma unroll
    for (int src = 0; src < 32; ++src) {
        ws += __shfl(thread_sum, src);
    }
    if (lane_id4 == 0) {
        warp_scratch[warp_id4] = ws;
    }
    __syncthreads();

    float sum_exp;
    if (t == 0) {
        float s = expf(attn_sink[h] - max_score);
        const int n_warps_p4 = (blockDim.x + 31) >> 5;
        for (int w = 0; w < n_warps_p4; ++w) {
            s += warp_scratch[w];
        }
        warp_scratch[0] = s;
    }
    __syncthreads();
    sum_exp = warp_scratch[0];

    const float inv_sum = 1.0f / sum_exp;
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    if (t >= head_dim) return;
    const int d = t;
    const float* swa_v_b = swa_v + swa_base;
    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += scores_lds[p] * swa_v_b[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        const int idx = topk_row[p];
        if (idx >= 0 && idx < n_compressed) {
            acc += scores_lds[n_valid_swa + p] * kv_cache[(size_t)idx * head_dim + d];
        }
    }
    attn_out[out_base + d] = acc;
}
