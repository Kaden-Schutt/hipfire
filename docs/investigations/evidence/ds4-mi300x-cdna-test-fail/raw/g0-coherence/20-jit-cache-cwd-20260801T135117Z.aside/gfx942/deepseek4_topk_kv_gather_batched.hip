// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 top-K K/V gather — BATCHED variant (Phase B2, 2026-05-18).
//
// Batched twin of `deepseek4_topk_kv_gather_f32`. Per batch position b uses
// its own top-K index list (from `indexer_top_k_batched`) and writes
// into its own slice of the staged output:
//
//   out[b, d, col_offset + k] = kv_cache[topk_idx[b, k], d] * scale
//
// kv_cache is shared across batch (single global compressed-K cache).
// topk_idx is per-batch [B, K] from the batched indexer.
//
// Output layout `[B, head_dim, out_stride]` matches what
// deepseek4_attn_swa_topk_batched expects as the topk_k / topk_v buffer.
//
// Sentinel topk_idx[b, k] = -1 (or out-of-range) writes zeros so the
// downstream softmax handles degenerate slots gracefully.
//
// Grid:  [K, batch_size, 1]
// Block: [head_dim, 1, 1]

#include <hip/hip_runtime.h>

extern "C" __global__ void deepseek4_topk_kv_gather_batched_f32(
    const float* __restrict__ kv_cache,   // [N_compressed, head_dim] shared
    const int*   __restrict__ topk_idx,   // [B, K]
    float*       __restrict__ out,        // [B, head_dim, out_stride]
    int K,
    int head_dim,
    int N_compressed,
    int out_stride,
    int col_offset,
    float scale,
    int batch_size
) {
    const int k = blockIdx.x;
    const int b = blockIdx.y;
    const int d = threadIdx.x;
    if (k >= K || b >= batch_size || d >= head_dim) return;

    const int idx = topk_idx[(size_t)b * K + k];
    float val = 0.0f;
    if (idx >= 0 && idx < N_compressed) {
        val = kv_cache[(size_t)idx * head_dim + d] * scale;
    }
    out[((size_t)b * head_dim + d) * out_stride + col_offset + k] = val;
}
