// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 MoE router top-K (bias-aware, normalize, scale) — BATCHED
// variant (Phase B2, 2026-05-18).
//
// Per batch position b, runs the same per-block argmax-poison-loop as
// the sequential deepseek4_moe_topk_bias_aware_f32 against its own
// scores[b, :] row. Bias is a shared model weight, unchanged across
// batch positions. Outputs:
//   indices[b, k_top]
//   weights[b, k_top]
//
// Grid:  [batch_size, 1, 1]
// Block: [256, 1, 1]  — n_exp must be ≤ 256 (DeepSeek V4: n_exp=256 exactly)
//
// Byte-identical to deepseek4_moe_topk_bias_aware_f32 at batch_size == 1.

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_K_TOP 32
#define WARP_SIZE 32

extern "C" __global__ void deepseek4_moe_topk_bias_aware_batched_f32(
    const float* __restrict__ scores,      // [B, n_exp]
    const float* __restrict__ bias,        // [n_exp]
    int*         __restrict__ indices,     // [B, k_top]
    float*       __restrict__ weights,     // [B, k_top]
    int   n_exp,
    int   k_top,
    float route_scale,
    int   batch_size
) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    __shared__ float s_score[1024];
    __shared__ float s_biased[1024];
    __shared__ int   s_idx[MAX_K_TOP];
    __shared__ float s_w_unsum[MAX_K_TOP];
    __shared__ float s_sum;

    // Phase 1: cooperative load this batch row's scores.
    if (tid < n_exp) {
        const float sc = scores[(size_t)b * n_exp + tid];
        s_score[tid]  = sc;
        s_biased[tid] = sc + bias[tid];
    }
    __syncthreads();

    // Phase 2: K rounds of block-wide argmax with poison.
    __shared__ float s_warp_max[WARP_SIZE];
    __shared__ int   s_warp_arg[WARP_SIZE];

    for (int k = 0; k < k_top; ++k) {
        float v  = (tid < n_exp) ? s_biased[tid] : -1e30f;
        int   ix = tid;

        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down(v,  off);
            int   oi = __shfl_down(ix, off);
            if (ov > v) { v = ov; ix = oi; }
        }
        if (lane_id == 0) {
            s_warp_max[warp_id] = v;
            s_warp_arg[warp_id] = ix;
        }
        __syncthreads();

        if (warp_id == 0) {
            const int n_warps = (blockDim.x + 31) >> 5;
            float ww = (lane_id < n_warps) ? s_warp_max[lane_id] : -1e30f;
            int   wi = (lane_id < n_warps) ? s_warp_arg[lane_id] : 0;
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down(ww, off);
                int   oi = __shfl_down(wi, off);
                if (ov > ww) { ww = ov; wi = oi; }
            }
            if (lane_id == 0) {
                s_idx[k] = wi;
                s_w_unsum[k] = s_score[wi];
                s_biased[wi] = -1e30f;
            }
        }
        __syncthreads();
    }

    // Phase 3: sum.
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < k_top; ++k) sum += s_w_unsum[k];
        s_sum = fmaxf(sum, 1e-12f);
    }
    __syncthreads();

    // Phase 4: write per-batch outputs.
    if (tid < k_top) {
        const float w = (s_w_unsum[tid] / s_sum) * route_scale;
        indices[(size_t)b * k_top + tid] = s_idx[tid];
        weights[(size_t)b * k_top + tid] = w;
    }
}
