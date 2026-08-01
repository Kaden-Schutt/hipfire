// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// E3(a) candidate for DeepSeek V4's graph-safe exact top-K.
//
// Cache two kpad-sized runs in LDS, sort each run by the stable score/index
// order, then bitonic-merge only the 2*kpad working set. Repeat that merge for
// every input tile, retaining the best kpad entries after each tile. Network
// width is therefore bounded by padded K rather than padded N, including
// N=2,048. k=1 retains the dedicated block argmax reduction from E2.
//
// The exported symbol deliberately remains `indexer_top_k_buf_parallel` so an
// E2 fresh-process A/B keeps the retained tape identity unchanged. The host
// selects a distinct module/cache key; both implementations must never be
// loaded under that symbol in one process.

#include <hip/hip_runtime.h>
#include <float.h>
#include <limits.h>

static constexpr int TOPK_MAX_KPAD = 512;

__device__ __forceinline__ bool topk_better(
    float lhs_score,
    int lhs_index,
    float rhs_score,
    int rhs_index
) {
    return (lhs_score > rhs_score)
        || (lhs_score == rhs_score && lhs_index < rhs_index);
}

__device__ __forceinline__ void topk_swap(
    float* scores,
    int* indices,
    int lhs,
    int rhs
) {
    const float score = scores[lhs];
    const int index = indices[lhs];
    scores[lhs] = scores[rhs];
    indices[lhs] = indices[rhs];
    scores[rhs] = score;
    indices[rhs] = index;
}

__device__ __forceinline__ void topk_bitonic_sort_desc(
    float* scores,
    int* indices,
    int base,
    int width
) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for (int size = 2; size <= width; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int local = tid; local < width; local += threads) {
                const int mate = local ^ stride;
                if (mate <= local) continue;
                const int lhs = base + local;
                const int rhs = base + mate;
                const bool descending = (local & size) == 0;
                const bool swap = descending
                    ? topk_better(scores[rhs], indices[rhs], scores[lhs], indices[lhs])
                    : topk_better(scores[lhs], indices[lhs], scores[rhs], indices[rhs]);
                if (swap) topk_swap(scores, indices, lhs, rhs);
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void topk_bitonic_merge_desc(
    float* scores,
    int* indices,
    int width
) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        for (int lhs = tid; lhs < width; lhs += threads) {
            const int rhs = lhs ^ stride;
            if (rhs <= lhs) continue;
            if (topk_better(scores[rhs], indices[rhs], scores[lhs], indices[lhs]))
                topk_swap(scores, indices, lhs, rhs);
        }
        __syncthreads();
    }
}

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

    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int n_compressed = n_compressed_buf[0];
    const int k = k_buf[0];
    const size_t output_base = (size_t)head * max_k;

    if (n_compressed <= 0 || k <= 0) {
        for (int rank = tid; rank < max_k; rank += threads)
            top_indices[output_base + rank] = -1;
        return;
    }

    if (n_compressed <= k) {
        for (int rank = tid; rank < max_k; rank += threads)
            top_indices[output_base + rank] =
                (rank < n_compressed) ? rank : -1;
        return;
    }

    const size_t scores_base = (size_t)head * n_compressed;
    __shared__ float cached_scores[2 * TOPK_MAX_KPAD];
    __shared__ int cached_indices[2 * TOPK_MAX_KPAD];

    if (k == 1) {
        float best_score = -FLT_MAX;
        int best_index = INT_MAX;
        for (int index = tid; index < n_compressed; index += threads) {
            const float score = scores[scores_base + index];
            if (topk_better(score, index, best_score, best_index)) {
                best_score = score;
                best_index = index;
            }
        }
        cached_scores[tid] = best_score;
        cached_indices[tid] = best_index;
        __syncthreads();
        for (int stride = threads >> 1; stride > 0; stride >>= 1) {
            if (tid < stride
                && topk_better(
                    cached_scores[tid + stride],
                    cached_indices[tid + stride],
                    cached_scores[tid],
                    cached_indices[tid]
                )) {
                cached_scores[tid] = cached_scores[tid + stride];
                cached_indices[tid] = cached_indices[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) top_indices[output_base] = cached_indices[0];
        for (int rank = 1 + tid; rank < max_k; rank += threads)
            top_indices[output_base + rank] = -1;
        return;
    }

    int kpad = 1;
    while (kpad < k) kpad <<= 1;
    if (kpad <= TOPK_MAX_KPAD) {
        for (int local = tid; local < kpad; local += threads) {
            const int index = local;
            cached_scores[local] =
                (index < n_compressed) ? scores[scores_base + index] : -INFINITY;
            cached_indices[local] = (index < n_compressed) ? index : INT_MAX;
        }
        __syncthreads();
        topk_bitonic_sort_desc(cached_scores, cached_indices, 0, kpad);

        for (int tile = kpad; tile < n_compressed; tile += kpad) {
            for (int local = tid; local < kpad; local += threads) {
                const int index = tile + local;
                cached_scores[kpad + local] =
                    (index < n_compressed) ? scores[scores_base + index] : -INFINITY;
                cached_indices[kpad + local] =
                    (index < n_compressed) ? index : INT_MAX;
            }
            __syncthreads();
            topk_bitonic_sort_desc(cached_scores, cached_indices, kpad, kpad);

            for (int local = tid; local < kpad / 2; local += threads)
                topk_swap(
                    cached_scores,
                    cached_indices,
                    kpad + local,
                    2 * kpad - 1 - local
                );
            __syncthreads();
            topk_bitonic_merge_desc(cached_scores, cached_indices, 2 * kpad);
        }

        for (int rank = tid; rank < max_k; rank += threads)
            top_indices[output_base + rank] =
                (rank < k) ? cached_indices[rank] : -1;
        return;
    }

    // Exact fallback for unsupported K > TOPK_MAX_KPAD.
    for (int index = tid; index < n_compressed; index += threads) {
        const float score_i = scores[scores_base + index];
        int rank = 0;
        for (int other = 0; other < n_compressed; ++other) {
            const float score_j = scores[scores_base + other];
            rank += (score_j > score_i)
                || (score_j == score_i && other < index);
        }
        if (rank < k) top_indices[output_base + rank] = index;
    }
    for (int rank = k + tid; rank < max_k; rank += threads)
        top_indices[output_base + rank] = -1;
}
