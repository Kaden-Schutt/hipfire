// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// GQA sliding-window batched attention over a STAGED window cache.
//
// Adapted from deepseek4_attn_swa_batched for the GQA arches (gemma3): each
// query head h reads the K/V of its group's kv head `kv_h = h / (n_heads /
// n_kv_heads)`, so the staged cache is per-kv-head ([batch, n_kv, head_dim,
// window]) rather than deepseek4's MLA shared-KV ([batch, head_dim, window]).
// No attention sink (gemma3 has none). Softmax scale is passed in (gemma3 bakes
// its query_pre_attn_scalar into q_norm, so the caller passes 1/sqrt(head_dim)).
//
// The staged cache is produced by swa_visibility_stage_batched (once per kv
// head): for batch row b at absolute position p, columns [0, n_valid_arr[b])
// hold the visible window keys in ascending age; deeper columns are ignored.
// Attention is order-invariant over keys, so the exact column order is
// irrelevant — only the set matters.
//
// Grid:  [n_heads, batch_size, 1]
// Block: [head_dim, 1, 1]   (head_dim ≤ 512)
//
// LDS: q_lds[head_dim] + scores_lds[window] + warp_scratch[warps] — bounded
// (window ≤ 1024), not seq-len-scaled; safe on the gfx1103 UMA path.

#include <hip/hip_runtime.h>
#include <math.h>

#define HIPFIRE_SWA_HEAD_DIM_LDS 512
#define HIPFIRE_SWA_MAX_WINDOW 1024
#define HIPFIRE_SWA_MAX_WARPS 16

extern "C" __launch_bounds__(512, 1)
__global__ void attention_swa_gqa_batched(
    const float* __restrict__ q,        // [batch, n_heads, head_dim]
    const float* __restrict__ k_cache,  // [batch, n_kv_heads, head_dim, window]
    const float* __restrict__ v_cache,  // [batch, n_kv_heads, head_dim, window]
    const int*   __restrict__ n_valid_arr, // [batch]
    float*       __restrict__ attn_out, // [batch, n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int window,
    int batch_size,
    float scale
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= n_heads || b >= batch_size) return;

    const int kv_h = h / (n_heads / n_kv_heads);
    const int n_valid = n_valid_arr[b];

    __shared__ float q_lds[HIPFIRE_SWA_HEAD_DIM_LDS];
    __shared__ float scores_lds[HIPFIRE_SWA_MAX_WINDOW];
    __shared__ float warp_scratch[HIPFIRE_SWA_MAX_WARPS];

    const size_t q_base = (size_t)(b * n_heads + h) * head_dim;
    // Per-(batch, kv-head) staged window base.
    const size_t kv_base = (size_t)((b * n_kv_heads) + kv_h) * head_dim * window;
    const size_t out_base = (size_t)(b * n_heads + h) * head_dim;

    const float* k_b = k_cache + kv_base;
    const float* v_b = v_cache + kv_base;

    // Phase 1: load Q[b, h] into LDS once.
    if (t < head_dim) {
        q_lds[t] = q[q_base + t];
    }
    __syncthreads();

    // Phase 2: scores = scale * (q · k_p) over valid window columns.
    for (int p = t; p < n_valid; p += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += q_lds[i] * k_b[(size_t)i * window + p];
        }
        scores_lds[p] = s * scale;
    }
    __syncthreads();

    // Phase 3: max reduction over scores (no sink).
    float local_max = -1e30f;
    for (int p = t; p < n_valid; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
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
    const float max_score = warp_scratch[0];

    // Phase 4: exp + deterministic sum (fixed-order __shfl broadcasts, matching
    // deepseek4_attn_swa_batched's determinism contract).
    for (int p = t; p < n_valid; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();

    float thread_sum = 0.0f;
    for (int p = t; p < n_valid; p += blockDim.x) {
        thread_sum += scores_lds[p];
    }
    float ws = 0.0f;
#pragma unroll
    for (int src = 0; src < 32; ++src) {
        ws += __shfl(thread_sum, src);
    }
    if (lane_id == 0) {
        warp_scratch[warp_id] = ws;
    }
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        const int n_warps_p4 = (blockDim.x + 31) >> 5;
        for (int w = 0; w < n_warps_p4; ++w) {
            s += warp_scratch[w];
        }
        warp_scratch[0] = s;
    }
    __syncthreads();
    const float sum_exp = warp_scratch[0];

    // Phase 5: normalise.
    const float inv_sum = 1.0f / sum_exp;
    for (int p = t; p < n_valid; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    // Phase 6: V accumulation (one output dim per thread).
    if (t >= head_dim) return;
    const int d = t;
    float acc = 0.0f;
    for (int p = 0; p < n_valid; ++p) {
        acc += scores_lds[p] * v_b[(size_t)d * window + p];
    }
    attn_out[out_base + d] = acc;
}
