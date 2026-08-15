// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Exact block-parallel replacement for DeepSeek V4 MQ2R's graph-safe
// indexer top-K at full compressed-cache occupancy.
//
// Each ELIGIBLE candidate computes its stable rank by counting other ELIGIBLE
// scores that are strictly greater, plus equal scores at lower indices. Ranks
// below K write their original index directly. This is byte-for-byte
// index-order equivalent to the portable serial selection sort, including ties.
//
// Eligibility filter (added 2026-08-03): a candidate is eligible only when
// `score > -FLT_MAX`, which by IEEE excludes NaN, -INFINITY and -FLT_MAX. DS4
// initialises unfilled indexer score-pool slots to -inf, so they must get zero
// softmax weight and must never be selected. The gfx942 sibling
// (`indexer_top_k_buf_parallel.gfx942.hip`) has always carried this filter;
// gfx1151 did not, and the raw-i32 channel (`test_indexer_top_k_buf`) caught
// three real failures on hardware because of it:
//   nonfinite_n513_neginf_pool / _nan / _neg_flt_max  -> DIFFER vs SERIAL,
//   and the NaN case reported unwritten=2.
// The NaN mechanism is the dangerous one: with NaN both `score_j > score_i`
// and `score_j == score_i` are false, so every NaN candidate computes rank 0,
// they all collide on slot 0, and two output ranks are never written at all —
// uninitialised indices feeding a KV gather is silent corruption, not a fault.
// Hence also the full `max_k` prefill below (the old code prefilled only
// `rank >= k`, which cannot cover ranks left unclaimed by eligibility).
//
// In the normal pipeline `indexer_relu_score_f32_buf` writes -inf only beyond
// the live N and this kernel bounds by `n_compressed`, so every scanned score
// is finite and the filter is a no-op. It exists for the paths where that
// invariant does not hold.

#include <hip/hip_runtime.h>
#include <float.h>

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

    // Deterministic pad: every rank starts as -1. Required once eligibility can
    // leave ranks unclaimed (n_eligible < k). The bound is uniform across the
    // block and the barrier sits outside every per-lane predicate, so it is
    // safe on wave32 (8 waves of 32 in a 256-thread block).
    for (int rank = threadIdx.x; rank < max_k; rank += blockDim.x)
        top_indices[output_base + rank] = -1;
    __syncthreads();

    for (int i = threadIdx.x; i < n_compressed; i += blockDim.x) {
        const float score_i = scores[scores_base + i];
        // Outer filter only. An ineligible candidate must never claim a rank,
        // so this one is load-bearing.
        if (!(score_i > -FLT_MAX)) continue;
        int rank = 0;
        for (int j = 0; j < n_compressed; ++j) {
            const float score_j = scores[scores_base + j];
            // No inner eligibility test: it is provably redundant here and
            // costs ~12% because it runs N^2 times. Given score_i > -FLT_MAX,
            // every ineligible score_j already contributes exactly 0 under
            // IEEE comparison:
            //   score_j == -INFINITY : -inf > score_i is false (score_i is
            //                          strictly greater than -FLT_MAX > -inf),
            //                          and -inf == score_i is false.
            //   score_j == -FLT_MAX  : -FLT_MAX > score_i is false and
            //                          -FLT_MAX == score_i is false, both
            //                          because score_i > -FLT_MAX.
            //   score_j is NaN       : every ordered and equality comparison
            //                          against NaN is false.
            // So the rank a surviving candidate computes is identical with or
            // without the test. Measured on gfx1151: 143.4 ms -> 160.9 ms at
            // N=32768 with the test, back to baseline without it.
            rank += (score_j > score_i) || (score_j == score_i && j < i);
        }
        if (rank < k) top_indices[output_base + rank] = i;
    }
}
