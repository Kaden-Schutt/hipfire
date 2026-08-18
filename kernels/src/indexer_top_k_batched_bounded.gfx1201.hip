// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1201 exact batched top-K for DeepSeek V4 prefill.
//
// One workgroup owns one (batch row, indexer head). Batched prefill already
// supplies hundreds of independent rows, so the decode-only multi-launch tree
// is unnecessary here: each workgroup keeps one K-sized winner run plus one
// incoming run in 8 KiB of LDS and incrementally bitonic-merges input tiles.
// Work is O(N log^2 K), replacing the portable O(N^2) rank-count slow path.
//
// Ordering is exactly the incumbent contract: score descending, then source
// index ascending. Real -INFINITY values remain eligible (the incumbent ranks
// them); only out-of-range tile padding uses INT_MAX as a sentinel.

// This is deliberately a distinct gfx1201 symbol. The gfx1151 source, code
// object, dispatch method, and route stay untouched.

#include <hip/hip_runtime.h>
#include <float.h>
#include <limits.h>

static constexpr int BATCHED_TOPK_MAX_KPAD = 512;

__device__ __forceinline__ bool batched_topk_better_gfx1201(
    float lhs_score,
    int lhs_index,
    float rhs_score,
    int rhs_index
) {
    return (lhs_score > rhs_score)
        || (lhs_score == rhs_score && lhs_index < rhs_index);
}

__device__ __forceinline__ void batched_topk_swap_gfx1201(
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

__device__ __forceinline__ void batched_topk_bitonic_sort_desc_gfx1201(
    float* scores,
    int* indices,
    int base,
    int width
) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= width; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int local = tid; local < width; local += blockDim.x) {
                const int mate = local ^ stride;
                if (mate <= local) continue;
                const int lhs = base + local;
                const int rhs = base + mate;
                const bool descending = (local & size) == 0;
                const bool swap = descending
                    ? batched_topk_better_gfx1201(
                        scores[rhs], indices[rhs], scores[lhs], indices[lhs])
                    : batched_topk_better_gfx1201(
                        scores[lhs], indices[lhs], scores[rhs], indices[rhs]);
                if (swap) batched_topk_swap_gfx1201(scores, indices, lhs, rhs);
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void batched_topk_bitonic_merge_desc_gfx1201(
    float* scores,
    int* indices,
    int width
) {
    const int tid = threadIdx.x;
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        for (int lhs = tid; lhs < width; lhs += blockDim.x) {
            const int rhs = lhs ^ stride;
            if (rhs <= lhs) continue;
            if (batched_topk_better_gfx1201(
                    scores[rhs], indices[rhs], scores[lhs], indices[lhs])) {
                batched_topk_swap_gfx1201(scores, indices, lhs, rhs);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void indexer_top_k_batched_bounded_gfx1201(
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
    const int batch = blockIdx.y;
    if (head >= n_idx_heads || batch >= batch_size) return;

    const int tid = threadIdx.x;
    const size_t out_base = (size_t)batch * n_idx_heads * k_stride
                          + (size_t)head * k_stride;

    if (n_iter <= k_fill) {
        for (int rank = tid; rank < k_stride; rank += blockDim.x)
            top_indices[out_base + rank] = (rank < n_iter) ? rank : -1;
        return;
    }

    const size_t scores_base = (size_t)batch * n_idx_heads * n_stride
                             + (size_t)head * n_stride;
    __shared__ float cached_scores[2 * BATCHED_TOPK_MAX_KPAD];
    __shared__ int cached_indices[2 * BATCHED_TOPK_MAX_KPAD];

    int kpad = 1;
    while (kpad < k_fill) kpad <<= 1;
    if (kpad <= BATCHED_TOPK_MAX_KPAD) {
        for (int local = tid; local < kpad; local += blockDim.x) {
            const int index = local;
            cached_scores[local] =
                (index < n_iter) ? scores[scores_base + index] : -INFINITY;
            cached_indices[local] = (index < n_iter) ? index : INT_MAX;
        }
        __syncthreads();
        batched_topk_bitonic_sort_desc_gfx1201(
            cached_scores, cached_indices, 0, kpad);

        for (int tile = kpad; tile < n_iter; tile += kpad) {
            for (int local = tid; local < kpad; local += blockDim.x) {
                const int index = tile + local;
                cached_scores[kpad + local] =
                    (index < n_iter) ? scores[scores_base + index] : -INFINITY;
                cached_indices[kpad + local] =
                    (index < n_iter) ? index : INT_MAX;
            }
            __syncthreads();
            batched_topk_bitonic_sort_desc_gfx1201(
                cached_scores, cached_indices, kpad, kpad);

            for (int local = tid; local < kpad / 2; local += blockDim.x) {
                batched_topk_swap_gfx1201(
                    cached_scores,
                    cached_indices,
                    kpad + local,
                    2 * kpad - 1 - local
                );
            }
            __syncthreads();
            batched_topk_bitonic_merge_desc_gfx1201(
                cached_scores, cached_indices, 2 * kpad);
        }

        for (int rank = tid; rank < k_stride; rank += blockDim.x) {
            top_indices[out_base + rank] =
                (rank < k_fill && cached_indices[rank] != INT_MAX)
                    ? cached_indices[rank]
                    : -1;
        }
        return;
    }

    for (int rank = tid; rank < k_stride; rank += blockDim.x)
        top_indices[out_base + rank] = -1;
    __syncthreads();
    for (int index = tid; index < n_iter; index += blockDim.x) {
        const float score_i = scores[scores_base + index];
        int rank = 0;
        for (int other = 0; other < n_iter; ++other) {
            const float score_j = scores[scores_base + other];
            rank += (score_j > score_i)
                || (score_j == score_i && other < index);
        }
        if (rank < k_fill) top_indices[out_base + rank] = index;
    }
}
