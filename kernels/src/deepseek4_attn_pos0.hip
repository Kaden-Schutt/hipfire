// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 position-0 attention with attn_sink (single-position, single KV head).
//
// Output is PER-HEAD (no group reduction). Upstream DeepSeek V4's `o_groups`
// structures the wo_a O-LoRA projection, NOT a head-reduction. Each
// of the 64 query heads attends only to the current token's K=V vector:
//
//   score_h    = Q[h, :] · K / sqrt(head_dim) - attn_sink[h]
//   prob_h     = sigmoid(score_h)
//   attn_out[h, d] = prob_h * V[d]
//
// Output shape: [n_heads, head_dim] = [64, 512] = 32768 elements.
// Downstream wo_a (per-group [o_lora_rank, heads_per_group*head_dim]
// GEMV) and wo_b ([hidden, n_groups*o_lora_rank] GEMV) project this
// to the residual contribution.

#include <hip/hip_runtime.h>

extern "C" __global__ void deepseek4_attn_pos0(
    const float* __restrict__ q,           // [n_heads, head_dim] = [64, 512]
    const float* __restrict__ kv,          // [head_dim] = [512] (tied K=V)
    const float* __restrict__ attn_sink,   // [n_heads] = [64]
    float*       __restrict__ attn_out,    // [n_heads, head_dim] = [64, 512]
    int n_heads,
    int head_dim,
    int o_groups                            // unused — kept in signature for ABI compat
) {
    (void)o_groups;
    const int h = blockIdx.x;             // head index 0..n_heads
    const int d = threadIdx.x;            // dim index 0..head_dim
    if (h >= n_heads || d >= head_dim) return;

    // Compute the per-head probability once (thread 0), broadcast via shared.
    __shared__ float shared_prob;
    if (d == 0) {
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            dot += q[h * head_dim + i] * kv[i];
        }
        float inv_scale = rsqrtf((float)head_dim);
        float score = dot * inv_scale - attn_sink[h];
        shared_prob = 1.0f / (1.0f + __expf(-score));
    }
    __syncthreads();

    attn_out[h * head_dim + d] = shared_prob * kv[d];
}
