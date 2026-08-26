// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Indexer top-K selection — BATCHED variant (Phase A3, 2026-05-18).
//
// Per batch position and per head, select the top-K position indices
// from the score array produced by indexer_compressed_k_score. Layout:
//   scores      : [batch, n_idx_heads, n_compressed]  fp32 in
//   top_indices : [batch, n_idx_heads, k]             i32  out
//
// Grid:  [n_idx_heads, batch, 1]
// Block: same as sequential (single-thread stub; dynamic LDS for `taken`)
//
// At batch_size == 1 this is byte-identical to `indexer_top_k`.

#include <hip/hip_runtime.h>
#include <float.h>

// `n_stride` and `k_stride` are the STORAGE strides (allocated row size
// of `scores` and `top_indices`). `n_iter` and `k_fill` are the actual
// work bounds.
//
// Fast path: when n_iter <= k_fill (which holds for all batch positions
// where n_compressed <= topk_max — i.e. context ≤ 8 K with DeepSeek V4 defaults)
// the top-K is "use all of them"; downstream attention is order-
// invariant w.r.t. topk_idx ordering, so we can write [0, 1, …,
// n_iter-1, sentinel-padded] in parallel and skip the O(N^2) loop.
//
// Slow path: still single-thread selection sort when n_iter > k_fill.
// Same caveat as indexer_top_k_buf — port to parallel top-K when long-
// context prefill at >8 K matters.
extern "C" __global__ void indexer_top_k_batched(
    const float* __restrict__ scores,
    int*         __restrict__ top_indices,
    int n_idx_heads,
    int n_stride,
    int n_iter,
    int k_stride,
    int k_fill,
    int batch_size
) {
    const int head = blockIdx.x;
    const int b    = blockIdx.y;
    if (head >= n_idx_heads || b >= batch_size) return;

    const size_t out_base    = (size_t)b * n_idx_heads * k_stride
                             + (size_t)head * k_stride;

    // Fast path: n_iter <= k_fill. Identity-write across all threads.
    if (n_iter <= k_fill) {
        for (int i = threadIdx.x; i < k_stride; i += blockDim.x) {
            top_indices[out_base + i] = (i < n_iter) ? i : -1;
        }
        return;
    }

    // Slow path: n_iter > k_fill. EXACT top-k_fill via block-parallel rank.
    // Each position i's rank = #positions that outrank it (strictly higher
    // score, OR equal score and lower index — a stable tiebreak). Positions
    // with rank < k_fill are the top-k; each writes itself to output slot
    // `rank`, so the result is score-sorted and BYTE-IDENTICAL to the old
    // single-thread selection sort — but every position's rank is computed
    // in parallel. O(n^2) cached score reads instead of the O(k*n) serial
    // loop that dominated long-context decode (>69% of decode GPU time @2.6K).
    const size_t scores_base = (size_t)b * n_idx_heads * n_stride
                             + (size_t)head * n_stride;
    const int T   = blockDim.x;
    const int tid = threadIdx.x;

    for (int i = tid; i < n_iter; i += T) {
        const float si = scores[scores_base + i];
        int rank = 0;
        for (int j = 0; j < n_iter; ++j) {
            const float sj = scores[scores_base + j];
            rank += (sj > si) || (sj == si && j < i);
        }
        if (rank < k_fill) top_indices[out_base + rank] = i;
    }
    // n_iter > k_fill guarantees exactly k_fill positions have rank < k_fill,
    // so slots [0, k_fill) are all written; pad [k_fill, k_stride) with -1.
    for (int r = k_fill + tid; r < k_stride; r += T) top_indices[out_base + r] = -1;
}
