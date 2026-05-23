// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 SWA-windowed attention — BATCHED variant (Phase A2, 2026-05-18).
//
// Twin of deepseek4_attn_swa for pure-SWA layers (no top-K indexer path).
// Processes a batch of B query positions in parallel through one launch.
// Each batch position has its own:
//   - Q row              q[b, h, *]
//   - SWA K/V slice      k_cache[b, h_dim, *], v_cache[b, h_dim, *]
//   - valid SWA count    n_valid_arr[b]
// attn_sink is shared per-head across the batch.
//
// At batch_size == 1 this kernel reduces to byte-identical math vs the
// sequential `deepseek4_attn_swa` (same FMA order, same reductions).
//
// Grid:  [n_heads, batch_size, 1]
// Block: [head_dim, 1, 1] = [512, 1, 1]

#include <hip/hip_runtime.h>
#include <math.h>

#define HEAD_DIM_LDS 512
#define MAX_WINDOW 1024
#define MAX_WARPS 16

extern "C" __global__ __launch_bounds__(512, 1) void deepseek4_attn_swa_batched(
    const float* __restrict__ q,           // [batch, n_heads, head_dim]
    const float* __restrict__ k_cache,     // [batch, head_dim, window]
    const float* __restrict__ v_cache,     // [batch, head_dim, window]
    const float* __restrict__ attn_sink,   // [n_heads]
    const int*   __restrict__ n_valid_arr, // [batch]
    float*       __restrict__ attn_out,    // [batch, n_heads, head_dim]
    int n_heads,
    int head_dim,
    int o_groups,                           // unused — kept for ABI parity
    int window,
    int batch_size
) {
    (void)o_groups;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= n_heads || b >= batch_size) return;

    const int n_valid     = n_valid_arr[b];
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_WINDOW];
    __shared__ float warp_scratch[MAX_WARPS];

    const size_t q_base  = (size_t)(b * n_heads + h) * head_dim;
    const size_t kv_base = (size_t)b * head_dim * window;
    const size_t out_base = (size_t)(b * n_heads + h) * head_dim;

    const float* k_b = k_cache + kv_base;
    const float* v_b = v_cache + kv_base;

    // Phase 1: load Q[b, h] into LDS once.
    if (t < head_dim) {
        q_lds[t] = q[q_base + t];
    }
    __syncthreads();

    // Phase 2: parallel score compute.
    for (int p = t; p < n_valid; p += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += q_lds[i] * k_b[i * window + p];
        }
        scores_lds[p] = s * inv_scale;
    }
    __syncthreads();

    // Phase 3: parallel max-reduction over scores + sink.
    float local_max = -1e30f;
    for (int p = t; p < n_valid; p += blockDim.x) {
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

    // Phase 4: parallel exp + DETERMINISTIC serial sum.
    //
    // Earlier this kernel used a __shfl_down warp-reduction (matching the
    // Phase 3 max-reduce structure). The MAX path is determinstic because
    // fmaxf returns one of its args exactly; the SUM path is not, because
    // FP + is order-sensitive. In-kernel bisect (commit 734e371) confirmed
    // Phase 3 max is bit-exact across twin-runs while Phase 4 sum_exp
    // differs in ~0.05 % of (h, b) pairs — enough to flip greedy decode
    // after a couple of sentences.
    //
    // Fix: each thread writes its exp into scores_lds; thread 0 then
    // sums scores_lds[0..n_valid] in serial deterministic order, plus
    // the attn_sink contribution. Cost is ~n_valid FP adds on lane 0
    // (~150 ns at n_valid=128), block-level — negligible at GPU scale.
    for (int p = t; p < n_valid; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();
    if (t == 0) {
        float s = expf(attn_sink[h] - max_score);
        for (int p = 0; p < n_valid; ++p) {
            s += scores_lds[p];
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

    // Phase 6: V accumulation.
    if (t >= head_dim) return;
    const int d = t;
    float acc = 0.0f;
    for (int p = 0; p < n_valid; ++p) {
        acc += scores_lds[p] * v_b[d * window + p];
    }
    attn_out[out_base + d] = acc;
}
