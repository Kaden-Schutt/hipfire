// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HIP-graphs-safe twin of deepseek4_attn_swa_topk_f32. Reads `n_valid_swa`
// and `n_active_topk` from device buffers instead of i32 kernargs, so
// the captured launch picks up updated positions on each graph replay
// without re-capture.
//
// Body math is byte-identical to deepseek4_attn_swa_topk_f32 — only the
// kernarg layout changes.

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_TOTAL 1024
#define HEAD_DIM_LDS 512
#define MAX_WARPS 16

extern "C" __global__ __launch_bounds__(512, 1) void deepseek4_attn_swa_topk_f32_buf(
    const float* __restrict__ q,
    const float* __restrict__ swa_k,
    const float* __restrict__ swa_v,
    const float* __restrict__ topk_k,
    const float* __restrict__ topk_v,
    const float* __restrict__ attn_sink,
    float*       __restrict__ attn_out,
    const int*   __restrict__ n_valid_swa_buf,    // [1]
    const int*   __restrict__ n_active_topk_buf,  // [1]
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window
) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;

    const int n_valid_swa   = n_valid_swa_buf[0];
    const int n_active_topk = n_active_topk_buf[0];
    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];

    if (t < head_dim) {
        q_lds[t] = q[h * head_dim + t];
    }
    __syncthreads();

    for (int p = t; p < n_total; p += blockDim.x) {
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
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += q_lds[i] * k_col[i * stride + col];
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

    // Phase 4a: parallel exp into scores_lds.
    //
    // Phase 4 history mirrors deepseek4_attn_swa_topk_batched:
    //   v1: __shfl_down reduction — non-deterministic FP order (commit 734e371).
    //   v2: thread-0 serial sum — deterministic but serial ~n_total adds.
    //   v3 (THIS): per-thread slice → __shfl broadcast warp sum (literal src) →
    //   thread-0 cross-warp sum. ~47 deterministic adds vs v2's ~n_total.
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();

    // Phase 4b: per-thread partial sum in fixed stride order.
    float thread_sum = 0.0f;
    for (int p = t; p < n_total; p += blockDim.x) {
        thread_sum += scores_lds[p];
    }

    // Phase 4c: warp-wide deterministic sum via __shfl broadcasts in fixed
    // lane order 0..31. Top-of-kernel `if (h >= n_heads) return;` is
    // block-uniform, so all wave lanes participate here.
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

    // Phase 4d: thread 0 sums warp partials in fixed warp order + attn_sink.
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
    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += scores_lds[p] * swa_v[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        acc += scores_lds[n_valid_swa + p] * topk_v[d * topk_window + p];
    }
    attn_out[h * head_dim + d] = acc;
}
