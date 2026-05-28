// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 hash-routed MoE — BATCHED variant (Phase D, 2026-05-21).
//
// Per-batch GPU-side tid2eid lookup + score gather + normalize +
// route_scale. Replaces the d2h(scores) + per-batch CPU loop + h2d in
// `ffn_batched` for hash-routed layers (DeepSeek V4 layers 0..num_hash_layers).
//
// Per batch row b:
//   ids[b, i]     = tid2eid[token_id[b], i]      i in 0..k
//   ws[b, i]      = scores[b, ids[b, i]]
//   sum[b]        = Σ ws[b, i]
//   if sum > 0: topk_w[b, i] = ws[b, i] * route_scale / sum
//   else:       topk_w[b, i] = 0  (zero contribution, layer no-ops)
//
// Inputs:
//   tid2eid       — [vocab_size, k] u32 lookup table (per hash layer)
//   scores        — [batch, n_exp] f32 router scores (unbiased)
//   token_ids     — [batch] i32 token IDs
//   n_exp, k      — routed expert count and top-k per token
//   route_scale   — weight scale
//   batch_size    — number of batch rows
//
// Outputs:
//   topk_idx      — [batch, k] i32 expert IDs
//   topk_w        — [batch, k] f32 normalized weights
//
// Grid: [batch_size, 1, 1].  Block: [1, 1, 1] — k is tiny (DeepSeek V4: 8).

#include <hip/hip_runtime.h>

#define HASH_ROUTER_BATCHED_K_MAX 16

extern "C" __global__ void hash_router_normalize_f32_batched(
    const unsigned int* __restrict__ tid2eid,    // [vocab, k]
    const float*        __restrict__ scores,     // [batch, n_exp]
    const int*          __restrict__ token_ids,  // [batch]
    int*                __restrict__ topk_idx,   // [batch, k]
    float*              __restrict__ topk_w,     // [batch, k]
    int n_exp, int k, float route_scale, int batch_size
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    if (threadIdx.x != 0) return;

    const int token_id = token_ids[b];
    int ids[HASH_ROUTER_BATCHED_K_MAX];
    float ws[HASH_ROUTER_BATCHED_K_MAX];
    float sum = 0.0f;

    const size_t tid_row_off  = (size_t)token_id * (size_t)k;
    const size_t score_off    = (size_t)b * (size_t)n_exp;
    const size_t out_row_off  = (size_t)b * (size_t)k;

    for (int i = 0; i < k; ++i) {
        int eid = (int)tid2eid[tid_row_off + i];
        if (eid >= n_exp) eid = n_exp - 1;
        ids[i] = eid;
        float s = scores[score_off + eid];
        ws[i] = s;
        sum += s;
    }

    if (sum > 0.0f) {
        float inv = route_scale / sum;
        for (int i = 0; i < k; ++i) {
            topk_idx[out_row_off + i] = ids[i];
            topk_w[out_row_off + i] = ws[i] * inv;
        }
    } else {
        for (int i = 0; i < k; ++i) {
            topk_idx[out_row_off + i] = ids[i];
            topk_w[out_row_off + i] = 0.0f;
        }
    }
}
