// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 indexer-extended SWA attention — BATCHED variant (Phase A1, 2026-05-18).
//
// Processes a batch of B query positions in parallel through one launch.
// Each batch position has its own:
//   - Q row              q[b, h, *]
//   - SWA K/V slice      swa_k[b, h_dim, *],  swa_v[b, h_dim, *]
//   - top-K K/V slice    topk_k[b, h_dim, *], topk_v[b, h_dim, *]
//   - valid SWA count    n_valid_swa[b]
//   - active topK count  n_active_topk[b]
// attn_sink is shared per-head across the batch.
//
// At batch_size == 1 this kernel reduces to byte-identical math vs the
// sequential `deepseek4_attn_swa_topk_f32` (same FMA order, same reductions).
//
// Grid:  [n_heads, batch_size, 1]
// Block: [head_dim, 1, 1] = [512, 1, 1]
//
// LDS:   q_lds[512] + scores_lds[1024] + warp_scratch[16] (static)
//        ≈ 6.2 KB per block. With launch_bounds 512×1, occupancy is
//        bounded by VGPR/LDS pressure on the gfx target; the per-block
//        footprint matches the sequential kernel exactly (a single block
//        handles one (h, b) pair — the batch dim parallelizes via grid).

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_TOTAL 1024
#define HEAD_DIM_LDS 512
#define MAX_WARPS 16    // 512-thread block / 32-thread warp

extern "C" __global__ __launch_bounds__(512, 1) void deepseek4_attn_swa_topk_batched_f32(
    const float* __restrict__ q,                 // [batch, n_heads, head_dim]
    const float* __restrict__ swa_k,             // [batch, head_dim, swa_window]
    const float* __restrict__ swa_v,             // [batch, head_dim, swa_window]
    const float* __restrict__ topk_k,            // [batch, head_dim, topk_window]
    const float* __restrict__ topk_v,            // [batch, head_dim, topk_window]
    const float* __restrict__ attn_sink,         // [n_heads]
    const int*   __restrict__ n_valid_swa_arr,   // [batch]
    const int*   __restrict__ n_active_topk_arr, // [batch]
    float*       __restrict__ attn_out,          // [batch, n_heads, head_dim]
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window,
    int batch_size
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= n_heads || b >= batch_size) return;

    const int n_valid_swa  = n_valid_swa_arr[b];
    const int n_active_topk = n_active_topk_arr[b];
    const int n_total      = n_valid_swa + n_active_topk;
    const float inv_scale  = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];

    // Per-batch base offsets.
    // Q layout: q[(b * n_heads + h) * head_dim + d]
    const size_t q_base = (size_t)(b * n_heads + h) * head_dim;
    // SWA/topK layout: cache_b[d * stride + col]; base offset per batch row
    // is b * head_dim * stride.
    const size_t swa_base  = (size_t)b * head_dim * swa_window;
    const size_t topk_base = (size_t)b * head_dim * topk_window;
    // Output: attn_out[(b * n_heads + h) * head_dim + d]
    const size_t out_base = (size_t)(b * n_heads + h) * head_dim;

    // Phase 1: load Q[b, h] into LDS once.
    if (t < head_dim) {
        q_lds[t] = q[q_base + t];
    }
    __syncthreads();

    // Phase 2: each thread computes scores for its stride-blockDim subset of
    // positions. Per position: full head_dim FMA chain over LDS Q.
    for (int p = t; p < n_total; p += blockDim.x) {
        const float* k_col;
        int stride;
        int col;
        if (p < n_valid_swa) {
            k_col = swa_k + swa_base;
            stride = swa_window;
            col = p;
        } else {
            k_col = topk_k + topk_base;
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

    // Phase 3: parallel max-reduction over scores + sink.
    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    // Warp-shuffle reduction within each 32-thread warp.
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

    // Phase 4: parallel exp + DETERMINISTIC serial sum on thread 0.
    // (Same fix as deepseek4_attn_swa_batched. Warp-shuffle reduce with FP +
    //  is order-sensitive at the hardware level; bit-determinism at
    //  temp=0 requires a fixed-order reduction. See commit 734e371 and
    //  feedback_deepseek4_prefill_nondeterministic_at_temp0 for the trail.)
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] = expf(scores_lds[p] - max_score);
    }
    __syncthreads();
    float sum_exp;
    if (t == 0) {
        float s = expf(attn_sink[h] - max_score);
        for (int p = 0; p < n_total; ++p) {
            s += scores_lds[p];
        }
        warp_scratch[0] = s;
    }
    __syncthreads();
    sum_exp = warp_scratch[0];

    // Phase 5: normalise probabilities.
    const float inv_sum = 1.0f / sum_exp;
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    // Phase 6: V accumulation. Each thread d computes one output element.
    if (t >= head_dim) return;
    const int d = t;
    const float* swa_v_b  = swa_v + swa_base;
    const float* topk_v_b = topk_v + topk_base;
    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += scores_lds[p] * swa_v_b[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        acc += scores_lds[n_valid_swa + p] * topk_v_b[d * topk_window + p];
    }
    attn_out[out_base + d] = acc;
}
