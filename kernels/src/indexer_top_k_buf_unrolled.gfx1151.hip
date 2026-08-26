// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// E4 candidate for DeepSeek V4's graph-safe exact top-K.
//
// This is the E3 tiled, K-bounded network with KPAD and wave width promoted to
// compile-time constants. The power-of-two dispatch ladder makes every
// compare-exchange chain statically sized so clang can fully unroll it.

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

template<int WIDTH>
__device__ __forceinline__ void topk_bitonic_sort_desc(
    float* scores,
    int* indices,
    int base
) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
#pragma unroll
    for (int size = 2; size <= WIDTH; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int local = tid; local < WIDTH; local += threads) {
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

template<int WIDTH>
__device__ __forceinline__ void topk_bitonic_merge_desc(
    float* scores,
    int* indices
) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
#pragma unroll
    for (int stride = WIDTH >> 1; stride > 0; stride >>= 1) {
        for (int lhs = tid; lhs < WIDTH; lhs += threads) {
            const int rhs = lhs ^ stride;
            if (rhs <= lhs) continue;
            if (topk_better(scores[rhs], indices[rhs], scores[lhs], indices[lhs]))
                topk_swap(scores, indices, lhs, rhs);
        }
        __syncthreads();
    }
}

template<int KPAD, int WAVE_WIDTH>
__device__ __forceinline__ void topk_tiled(
    const float* scores,
    int* top_indices,
    float* cached_scores,
    int* cached_indices,
    size_t scores_base,
    size_t output_base,
    int n_compressed,
    int k,
    int max_k
) {
    static_assert((KPAD & (KPAD - 1)) == 0, "KPAD must be a power of two");
    static_assert(WAVE_WIDTH == 32, "gfx1151 route requires wave32");
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    for (int local = tid; local < KPAD; local += threads) {
        const int index = local;
        cached_scores[local] =
            (index < n_compressed) ? scores[scores_base + index] : -INFINITY;
        cached_indices[local] = (index < n_compressed) ? index : INT_MAX;
    }
    __syncthreads();
    topk_bitonic_sort_desc<KPAD>(cached_scores, cached_indices, 0);

    for (int tile = KPAD; tile < n_compressed; tile += KPAD) {
        for (int local = tid; local < KPAD; local += threads) {
            const int index = tile + local;
            cached_scores[KPAD + local] =
                (index < n_compressed) ? scores[scores_base + index] : -INFINITY;
            cached_indices[KPAD + local] = (index < n_compressed) ? index : INT_MAX;
        }
        __syncthreads();
        topk_bitonic_sort_desc<KPAD>(cached_scores, cached_indices, KPAD);

        for (int local = tid; local < KPAD / 2; local += threads)
            topk_swap(
                cached_scores,
                cached_indices,
                KPAD + local,
                2 * KPAD - 1 - local
            );
        __syncthreads();
        topk_bitonic_merge_desc<2 * KPAD>(cached_scores, cached_indices);
    }

    for (int rank = tid; rank < max_k; rank += threads)
        top_indices[output_base + rank] =
            (rank < k) ? cached_indices[rank] : -1;
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
    switch (kpad) {
        case 2:
            topk_tiled<2, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 4:
            topk_tiled<4, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 8:
            topk_tiled<8, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 16:
            topk_tiled<16, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 32:
            topk_tiled<32, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 64:
            topk_tiled<64, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 128:
            topk_tiled<128, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 256:
            topk_tiled<256, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        case 512:
            topk_tiled<512, 32>(
                scores, top_indices, cached_scores, cached_indices,
                scores_base, output_base, n_compressed, k, max_k);
            return;
        default:
            break;
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
