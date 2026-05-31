// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HIP-graphs-safe twin of indexer_top_k.
// Reads `n_compressed` (= N) and `k` from device buffers instead of
// i32 kernargs.
//
// Fast path: when n_compressed <= max_k (the common case until the
// cache fills at ratio=4 layers, i.e. for positions <= 8K) the
// selection is "use all of them" — every compressed slot lands in the
// top-K. The downstream gather + softmax-attention are order-
// invariant w.r.t. the topk_idx ordering, so we can write the slots
// in IDENTITY order (0, 1, …, n_compressed-1, sentinel-padded) and
// skip the O(N^2) selection sort entirely. Parallelised across the
// whole block.
//
// Slow path: when n_compressed > max_k (positions > 8K with DeepSeek V4
// defaults), single-thread selection sort still runs. Worth porting
// to a parallel top-K when long-context perf matters.
//
// Body math is byte-equivalent to indexer_top_k on the fast path
// (same SET of selected indices, just different ORDER — downstream
// kernels treat the K-axis as a bag).

#include <hip/hip_runtime.h>
#include <limits.h>
#include <float.h>

extern "C" __global__ void indexer_top_k_buf(
    const float* __restrict__ scores,
    int*         __restrict__ top_indices,
    const int*   __restrict__ n_compressed_buf,  // [1]
    const int*   __restrict__ k_buf,             // [1]
    int n_idx_heads,
    int max_k                                    // upper bound on k for output stride
) {
    const int head = blockIdx.x;
    if (head >= n_idx_heads) return;

    const int n_compressed = n_compressed_buf[0];
    const int k = k_buf[0];

    // Fast path: n_compressed <= max_k → top-K is the full set. Each
    // thread writes its own slot of the output (identity for the first
    // n_compressed entries, -1 sentinel for the rest up to max_k).
    if (n_compressed <= max_k) {
        for (int i = threadIdx.x; i < max_k; i += blockDim.x) {
            top_indices[head * max_k + i] = (i < n_compressed) ? i : -1;
        }
        return;
    }

    // Slow path: actual selection. Single-thread per block, O(N*K)
    // selection sort with a "taken" bitmap in shared memory.
    if (threadIdx.x != 0) return;

    extern __shared__ unsigned char taken_smem[];
    char* taken = (char*)taken_smem;
    for (int i = 0; i < n_compressed; ++i) taken[i] = 0;

    for (int rank = 0; rank < max_k; ++rank) {
        if (rank >= k) {
            top_indices[head * max_k + rank] = -1;
            continue;
        }
        float best = -FLT_MAX;
        int best_idx = -1;
        for (int i = 0; i < n_compressed; ++i) {
            if (taken[i]) continue;
            float s = scores[head * n_compressed + i];
            if (s > best) { best = s; best_idx = i; }
        }
        if (best_idx >= 0) {
            top_indices[head * max_k + rank] = best_idx;
            taken[best_idx] = 1;
        } else {
            top_indices[head * max_k + rank] = -1;
        }
    }
}
