// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Indexer top-k selection (DeepSeek V4 Phase 2).
//
// Per head, select the top-K position indices from the score array
// produced by `indexer_compressed_k_score`. K = `index_topk = 512` on
// DeepSeek V4.
//
// Signature (subject to revision):
//
//   scores       : [n_idx_heads, n_compressed]   (fp32, in)
//   top_indices  : [n_idx_heads, k]              (i32, out)
//   top_scores   : [n_idx_heads, k]              (fp32, out, optional)
//
// Strategy: each block processes one head. Two-pass radix:
//   Pass 1: threshold-binary-search over the histogram of scores to
//           find a cutoff value such that the count of `score > cutoff`
//           equals k.
//   Pass 2: emit indices where score > cutoff.
//
// For Phase 2 stub: simple partial-sort baseline (O(n_compressed · k)).
// Real impl will use bitonic-select for n_compressed up to ~16K, then
// radix for the long-context (ratio=4 at 1M ctx) case where
// n_compressed is ~250K.
//
// Status: STUB. Correctness via the simple baseline; perf tuning is
// Phase 2 work proper.

#include <hip/hip_runtime.h>
#include <limits.h>
#include <float.h>

extern "C" __global__ void indexer_top_k(
    const float* __restrict__ scores,        // [H, N]
    int*         __restrict__ top_indices,    // [H, K]
    int n_idx_heads,                         // H
    int n_compressed,                         // N
    int k                                     // K (= index_topk = 512)
) {
    const int head = blockIdx.x;
    if (head >= n_idx_heads) return;
    if (threadIdx.x != 0) return;  // stub: single-thread per head, replace in real impl.

    // Simple partial-sort: K passes over N positions, each finding the
    // largest unselected score. O(N·K). At N=8K, K=512 → ~4M ops/head.
    // O(64M ops total) for DeepSeek V4's 64 heads — workable but suboptimal.
    extern __shared__ unsigned char taken_smem[];
    char* taken = (char*)taken_smem;
    for (int i = 0; i < n_compressed; ++i) taken[i] = 0;

    for (int rank = 0; rank < k; ++rank) {
        float best = -FLT_MAX;
        int best_idx = -1;
        for (int i = 0; i < n_compressed; ++i) {
            if (taken[i]) continue;
            float s = scores[head * n_compressed + i];
            if (s > best) { best = s; best_idx = i; }
        }
        if (best_idx >= 0) {
            top_indices[head * k + rank] = best_idx;
            taken[best_idx] = 1;
        } else {
            top_indices[head * k + rank] = -1;
        }
    }
}
