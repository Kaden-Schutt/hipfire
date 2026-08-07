// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <math.h>

#define SCORE_TILE 128
#define VALUE_TILE 128
#define HEAD_DIM 512
#define SOFTMAX_THREADS 512
#define MAX_WARPS 16

// Exact-gfx1201 occupancy screen for DS4 TP3 decode. The incumbent uses one
// 512-thread block per head for score, softmax, and value. TP3 has only 16 or
// 24 local heads on a 64-CU GPU. These three kernels retain the scalar score
// and value accumulation order but expose independent score/value tiles.

extern "C" __global__ __launch_bounds__(SCORE_TILE, 8)
void deepseek4_attn_swa_topk_score_split_gfx1201(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ topk_k,
    float* __restrict__ scores,
    const int* __restrict__ n_valid_swa_buf,
    const int* __restrict__ n_active_topk_buf,
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= n_heads || head_dim != HEAD_DIM) return;
    const int n_valid_swa = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const int stride_total = swa_window + topk_window;
    const int p = blockIdx.x * SCORE_TILE + t;

    __shared__ float q_lds[HEAD_DIM];
#pragma unroll
    for (int i = t; i < HEAD_DIM; i += SCORE_TILE) {
        q_lds[i] = q[h * HEAD_DIM + i];
    }
    __syncthreads();

    if (p >= n_total) return;
    const float* k_col;
    int stride;
    int col;
    if (p < n_valid_swa) {
        k_col = swa_k;
        stride = swa_window;
        col = p;
    } else {
        k_col = topk_k;
        stride = topk_window;
        col = p - n_valid_swa;
    }
    float sum = 0.0f;
    for (int i = 0; i < HEAD_DIM; ++i) {
        sum += q_lds[i] * k_col[i * stride + col];
    }
    scores[h * stride_total + p] = sum * rsqrtf((float)HEAD_DIM);
}

extern "C" __global__ __launch_bounds__(SOFTMAX_THREADS, 1)
void deepseek4_attn_swa_topk_softmax_split_gfx1201(
    float* __restrict__ scores,
    const float* __restrict__ attn_sink,
    const int* __restrict__ n_valid_swa_buf,
    const int* __restrict__ n_active_topk_buf,
    int n_heads,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;
    const int n_total = n_valid_swa_buf[0] + n_active_topk_buf[0];
    const int stride_total = swa_window + topk_window;
    float* head_scores = scores + h * stride_total;
    __shared__ float warp_scratch[MAX_WARPS];

    float local_max = -1e30f;
    for (int p = t; p < n_total; p += SOFTMAX_THREADS) {
        local_max = fmaxf(local_max, head_scores[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off, 32));
    }
    const int warp_id = t >> 5;
    const int lane_id = t & 31;
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        float value = lane_id < MAX_WARPS ? warp_scratch[lane_id] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            value = fmaxf(value, __shfl_down(value, off, 32));
        }
        if (lane_id == 0) {
            warp_scratch[0] = value;
        }
    }
    __syncthreads();
    const float max_score = warp_scratch[0];

    for (int p = t; p < n_total; p += SOFTMAX_THREADS) {
        head_scores[p] = expf(head_scores[p] - max_score);
    }
    __syncthreads();
    float thread_sum = 0.0f;
    for (int p = t; p < n_total; p += SOFTMAX_THREADS) {
        thread_sum += head_scores[p];
    }
    float warp_sum = 0.0f;
#pragma unroll
    for (int source = 0; source < 32; ++source) {
        warp_sum += __shfl(thread_sum, source, 32);
    }
    if (lane_id == 0) {
        warp_scratch[warp_id] = warp_sum;
    }
    __syncthreads();
    if (t == 0) {
        float total = expf(attn_sink[h] - max_score);
        for (int w = 0; w < MAX_WARPS; ++w) {
            total += warp_scratch[w];
        }
        warp_scratch[0] = total;
    }
    __syncthreads();
    const float inv_sum = 1.0f / warp_scratch[0];
    for (int p = t; p < n_total; p += SOFTMAX_THREADS) {
        head_scores[p] *= inv_sum;
    }
}

extern "C" __global__ __launch_bounds__(VALUE_TILE, 8)
void deepseek4_attn_swa_topk_value_split_gfx1201(
    const float* __restrict__ scores,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_v,
    float* __restrict__ attn_out,
    const int* __restrict__ n_valid_swa_buf,
    const int* __restrict__ n_active_topk_buf,
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.y;
    const int d = blockIdx.x * VALUE_TILE + threadIdx.x;
    if (h >= n_heads || d >= head_dim) return;
    const int n_valid_swa = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int stride_total = swa_window + topk_window;
    const float* head_scores = scores + h * stride_total;
    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += head_scores[p] * swa_v[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        acc += head_scores[n_valid_swa + p] * topk_v[d * topk_window + p];
    }
    attn_out[h * head_dim + d] = acc;
}
