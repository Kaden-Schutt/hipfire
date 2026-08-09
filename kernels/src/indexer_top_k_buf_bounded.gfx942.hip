// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx942 bounded-width indexer top-K for DeepSeek V4 MQ2R (lever F2, opt-in).
//
// Drop-in replacement for indexer_top_k_buf_parallel_gfx942 with identical
// dispatch shape: grid [n_idx_heads,1,1], block [256,1,1], no dynamic shared
// memory, no new buffers, graph-capture safe. The reference is O(N^2): every
// eligible candidate counts strictly-greater eligible scores across all N
// rows, and ds4 launches with n_idx_heads == 1, so the whole N^2 cost lands
// on ONE 256-thread workgroup of the 304-CU MI300X. Measured on the campaign
// box (MI300X, HIP 7.0, ds4_longctx_probe, synthetic position, 3 decode
// reps, peak memory flat-to-mildly-rising so memory is not the limit):
//   cap  2048 @ pos   8191 -> 25.070    tok/s (  39.89 ms/tok)
//   cap  8192 @ pos  32767 ->  3.914    tok/s ( 255.49 ms/tok)
//   cap 32768 @ pos 131071 ->  0.274202 tok/s (3646.95 ms/tok)
// Fitting T(N) = A + B*N^2 on the two largest points gives A = 29.40 ms for
// everything that is not top-K and attributes 35% / 88.5% / 99.2% of decode
// at N = 2048 / 8192 / 32768 to this kernel. Extrapolated to 1M context
// (N = 262144 compressed rows at ratio 4) the reference alone costs
// ~232 s/token.
//
// Algorithm: port of the gfx1151 tile-merge bitonic in
// indexer_top_k_buf_bounded.gfx1151.hip. Cache two kpad-sized runs in LDS,
// bitonic-sort each run by the stable (score desc, index asc) order, then
// bitonic-merge only the 2*kpad working set per input tile, retaining the
// best kpad entries after each tile. Network width is bounded by padded K
// (kpad <= TOPK_MAX_KPAD = 512) instead of padded N: O(N log^2 K) instead of
// O(N^2). k == 1 deliberately flows through the general kpad == 1 path; the
// gfx1151 block-argmax special case is not carried over (the general path is
// still exact and one serial O(N) merge per candidate).
//
// The exported symbol `indexer_top_k_buf_parallel_gfx942_bounded` is
// deliberately distinct from BOTH the gfx1151 symbol
// (`indexer_top_k_buf_parallel`) and the existing gfx942 symbol
// (`indexer_top_k_buf_parallel_gfx942`) so the three implementations can
// never collide in kernel traces or code-object caches; the host selects
// this source/symbol pair behind a distinct module cache key.
//
// Known remaining limit: the grid is still one workgroup because ds4 passes
// n_idx_heads == 1, so the tile loop is serial on a single CU. A multi-block
// two-stage top-K (per-block partial top-K plus a merge pass) is the next
// lever if this kernel still dominates at very long context.

#include <hip/hip_runtime.h>
#include <float.h>
#include <limits.h>

static constexpr int TOPK_MAX_KPAD = 512;

// Sort/merge helpers are verbatim copies of the proven gfx1151 network.

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

extern "C" __global__ void indexer_top_k_buf_parallel_gfx942_bounded(
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

    // Identity fast path — BYTE-EXACT vs indexer_top_k_buf_parallel_gfx942.
    // The condition is n_compressed <= max_k (the gfx1151 kernel uses <= k)
    // and it writes (i < n_compressed) ? i : -1 with NO eligibility filter.
    // The gfx942 default kernel behaves exactly this way, so this opt-in
    // replacement must too: "fixing" the condition or adding the filter here
    // would diverge from the A/B reference for shapes with
    // k < n_compressed <= max_k.
    if (n_compressed <= max_k) {
        for (int i = tid; i < max_k; i += threads)
            top_indices[output_base + i] = (i < n_compressed) ? i : -1;
        return;
    }

    // Wave64 / barrier uniformity: n_compressed, k, max_k and every kpad/tile
    // bound below are kernarg/buffer values identical for every thread in the
    // block. The identity early-return above is therefore uniform across the
    // whole workgroup — either every lane takes it, or none do. Every
    // __syncthreads() below sits OUTSIDE per-lane predicates, with loop
    // bounds derived only from those uniform values: inside the sort/merge
    // helpers the barrier is outside the per-lane strided loop, so lanes that
    // take `continue` or run zero inner iterations still arrive. No wave of
    // the 4-wave (4 x 64) 256-thread CDNA block can reach a different barrier
    // count than its peers, so all barriers are safe on wave64.
    //
    // Eligibility (ds4 initialises unfilled indexer score-pool slots to -inf,
    // which must get zero softmax weight and must never be selected): a
    // candidate is eligible only when score > -FLT_MAX — by IEEE this
    // excludes NaN, -INFINITY and -FLT_MAX itself. Canonicalise at LOAD: an
    // ineligible score is cached as (-INFINITY, INT_MAX), identical to
    // out-of-range padding, so ineligible entries sort deterministically to
    // the tail of the retained run and NaN never enters the comparator (a
    // bitonic network needs a valid total order). Filter at WRITE: a claimed
    // rank whose cached index is INT_MAX was never an eligible candidate and
    // writes -1; ranks past k also write -1. The selected set and rank order
    // are byte-identical to the reference rank-count kernel.
    //
    // LDS budget: 2 * TOPK_MAX_KPAD floats + 2 * TOPK_MAX_KPAD ints = 8 KiB,
    // statically declared; dynamic shared memory stays 0.
    const size_t scores_base = (size_t)head * n_compressed;
    __shared__ float cached_scores[2 * TOPK_MAX_KPAD];
    __shared__ int cached_indices[2 * TOPK_MAX_KPAD];

    int kpad = 1;
    while (kpad < k) kpad <<= 1;

    if (kpad <= TOPK_MAX_KPAD) {
        // Tile 0: candidates [0, kpad), canonicalised at load.
        for (int local = tid; local < kpad; local += threads) {
            const float score =
                (local < n_compressed) ? scores[scores_base + local] : -INFINITY;
            const bool eligible = score > -FLT_MAX;
            cached_scores[local] = eligible ? score : -INFINITY;
            cached_indices[local] = eligible ? local : INT_MAX;
        }
        __syncthreads();
        topk_bitonic_sort_desc(cached_scores, cached_indices, 0, kpad);

        // Each further tile: sort the canonicalised run into the upper half,
        // reverse it so the concatenation is bitonic, then merge and retain
        // the best kpad entries in the lower half.
        for (int tile = kpad; tile < n_compressed; tile += kpad) {
            for (int local = tid; local < kpad; local += threads) {
                const int index = tile + local;
                const float score =
                    (index < n_compressed) ? scores[scores_base + index] : -INFINITY;
                const bool eligible = score > -FLT_MAX;
                cached_scores[kpad + local] = eligible ? score : -INFINITY;
                cached_indices[kpad + local] = eligible ? index : INT_MAX;
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

        // Write-time eligibility filter: INT_MAX can only come from a
        // canonicalised ineligible/padding entry, never from a real index.
        for (int rank = tid; rank < max_k; rank += threads) {
            int selected = -1;
            if (rank < k) {
                const int index = cached_indices[rank];
                selected = (index == INT_MAX) ? -1 : index;
            }
            top_indices[output_base + rank] = selected;
        }
        return;
    }

    // Exact fallback for K > TOPK_MAX_KPAD: the reference O(N^2) rank-count
    // with the same eligibility filter as indexer_top_k_buf_parallel_gfx942
    // and a full -1 prefill so ranks left unclaimed by eligibility
    // (n_eligible < k) never leak uninitialised slots. Byte-identical to the
    // default kernel at reference cost.
    //
    // Barrier uniformity: the prefill bound max_k is uniform and the single
    // __syncthreads() is outside every per-lane predicate; the rank-count
    // loop contains no barrier, so its per-lane `continue` is safe.
    for (int rank = tid; rank < max_k; rank += threads)
        top_indices[output_base + rank] = -1;
    __syncthreads();
    for (int i = tid; i < n_compressed; i += threads) {
        const float score_i = scores[scores_base + i];
        // Reference predicate: best = -FLT_MAX, select only if s > best.
        if (!(score_i > -FLT_MAX)) continue;
        int rank = 0;
        for (int j = 0; j < n_compressed; ++j) {
            const float score_j = scores[scores_base + j];
            if (!(score_j > -FLT_MAX)) continue;
            rank += (score_j > score_i) || (score_j == score_i && j < i);
        }
        if (rank < k) top_indices[output_base + rank] = i;
    }
}
