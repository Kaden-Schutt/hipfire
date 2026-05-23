// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 hash-routed MoE: GPU-side tid2eid lookup + score normalize +
// route_scale multiply. Replaces the d2h+host-compute+h2d round-trip
// in `ffn_hash_routed`:
//
//   d2h scores → host gather scores[tid2eid[token_id][0..k]]
//             → host normalize (sum, divide)
//             → host multiply by route_scale
//             → h2d topk_idx + topk_w
//
// Single thread is sufficient — k is tiny (DeepSeek V4: 8). The kernel writes
// the same outputs the host loop produced.
//
// Inputs:
//   tid2eid       — [vocab_size, k] u32 lookup table for this hash layer
//   scores        — [n_exp] router scores (unbiased)
//   token_id      — single i32 (passed as kernarg; for graph capture,
//                   pass via a device buffer with a `_buf` variant)
//   n_exp         — number of routed experts (256 on DeepSeek V4)
//   k             — number of experts per token (8 on DeepSeek V4)
//   route_scale   — multiplier (DeepSeek V4 default 2.2)
//
// Outputs:
//   topk_idx[k]   — i32 expert IDs from tid2eid[token_id][0..k]
//   topk_w[k]     — f32 normalized + route-scaled weights. When the
//                   gathered-score sum is non-positive (degenerate
//                   router), all weights are written as 0.0 — the
//                   downstream MoE accumulator multiplies by 0 and
//                   produces no contribution (matches host fallback
//                   "return Ok(())" behaviour without needing a
//                   conditional dispatch skip).

#include <hip/hip_runtime.h>

// k_max = 16 is plenty for any current arch (DeepSeek V4 uses 8).
#define HASH_ROUTER_K_MAX 16

extern "C" __global__ void hash_router_normalize_f32(
    const unsigned int* __restrict__ tid2eid,  // [vocab, k]
    const float*        __restrict__ scores,   // [n_exp]
    int*                __restrict__ topk_idx, // [k]
    float*              __restrict__ topk_w,   // [k]
    int token_id, int n_exp, int k, float route_scale
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int ids[HASH_ROUTER_K_MAX];
    float ws[HASH_ROUTER_K_MAX];
    float sum = 0.0f;

    const size_t row_off = (size_t)token_id * (size_t)k;
    for (int i = 0; i < k; ++i) {
        int eid = (int)tid2eid[row_off + i];
        if (eid >= n_exp) eid = n_exp - 1;
        ids[i] = eid;
        float s = scores[eid];
        ws[i] = s;
        sum += s;
    }

    if (sum > 0.0f) {
        float inv = route_scale / sum;
        for (int i = 0; i < k; ++i) {
            topk_idx[i] = ids[i];
            topk_w[i] = ws[i] * inv;
        }
    } else {
        // Degenerate: write valid eids + zero weights. Downstream MoE
        // accumulator multiplies by 0 so the layer no-ops.
        for (int i = 0; i < k; ++i) {
            topk_idx[i] = ids[i];
            topk_w[i] = 0.0f;
        }
    }
}
