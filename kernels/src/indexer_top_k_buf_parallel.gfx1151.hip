// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Exact block-parallel replacement for DeepSeek V4 MQ2R's graph-safe
// indexer top-K at full compressed-cache occupancy.
//
// Each candidate computes its stable rank by counting scores that are
// strictly greater, plus equal scores at lower indices. Ranks below K write
// their original index directly. This is byte-for-byte index-order equivalent
// to the portable serial selection sort, including ties, without LDS or a
// workgroup barrier.

#include <hip/hip_runtime.h>

extern "C" __global__ void indexer_top_k_buf_parallel(
    const float* __restrict__ scores,
    int*         __restrict__ top_indices,
    const int*   __restrict__ n_compressed_buf,
    const int*   __restrict__ k_buf,
    int n_idx_heads,
    int max_k
) {
    const int head = blockIdx.x;
    if (head >= n_idx_heads) return;

    const int n_compressed = n_compressed_buf[0];
    const int k = k_buf[0];
    const size_t output_base = (size_t)head * max_k;

    if (n_compressed <= max_k) {
        for (int i = threadIdx.x; i < max_k; i += blockDim.x)
            top_indices[output_base + i] = (i < n_compressed) ? i : -1;
        return;
    }

    const size_t scores_base = (size_t)head * n_compressed;
    for (int i = threadIdx.x; i < n_compressed; i += blockDim.x) {
        const float score_i = scores[scores_base + i];
        int rank = 0;
        for (int j = 0; j < n_compressed; ++j) {
            const float score_j = scores[scores_base + j];
            rank += (score_j > score_i) || (score_j == score_i && j < i);
        }
        if (rank < k) top_indices[output_base + rank] = i;
    }

    for (int rank = k + threadIdx.x; rank < max_k; rank += blockDim.x)
        top_indices[output_base + rank] = -1;
}
