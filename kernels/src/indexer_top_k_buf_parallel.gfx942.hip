// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx942-only block-parallel indexer top-K for DeepSeek V4 MQ2R.
//
// Byte-exact analogue of the portable serial slow path (strict `s > best`
// with `best = -FLT_MAX`): each ELIGIBLE candidate computes its stable rank
// by counting other ELIGIBLE scores that are strictly greater, plus equal
// scores at lower indices. Ranks below K write their original index.
//
// Eligibility filter exists because DS4 initialises unfilled indexer score
// pool slots to -inf so they must get zero softmax weight and must never be
// selected. Eligible means `score > -FLT_MAX` (excludes NaN, -INFINITY, and
// -FLT_MAX via IEEE). Ineligible candidates never occupy an output slot;
// unfilled ranks stay the -1 prefill.
//
// Distinct entry symbol from the gfx1151 shipping kernel so the two cannot
// collide in kernel traces or code-object caches.

#include <hip/hip_runtime.h>
#include <float.h>

extern "C" __global__ void indexer_top_k_buf_parallel_gfx942(
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

    // Identity fast path: N <= max_k is unchanged vs serial/gfx1151.
    if (n_compressed <= max_k) {
        for (int i = threadIdx.x; i < max_k; i += blockDim.x)
            top_indices[output_base + i] = (i < n_compressed) ? i : -1;
        return;
    }

    // Deterministic pad: every rank starts as -1. Required once eligibility
    // can leave ranks unclaimed (n_eligible < k).
    //
    // Wave64 / barrier uniformity: n_compressed and max_k are kernarg/buffer
    // values identical for every thread in the block. The early return above
    // is therefore uniform across the whole workgroup — either every lane
    // takes it, or none do. All threads that reach this slow path execute the
    // prefill loop and the __syncthreads() together. The barrier is NOT inside
    // divergent control flow (no per-lane predicate around it), so it is safe
    // on wave64 CDNA (4 waves of 64 in a 256-thread block): every wave arrives.
    for (int rank = threadIdx.x; rank < max_k; rank += blockDim.x)
        top_indices[output_base + rank] = -1;
    __syncthreads();

    const size_t scores_base = (size_t)head * n_compressed;
    for (int i = threadIdx.x; i < n_compressed; i += blockDim.x) {
        const float score_i = scores[scores_base + i];
        // SERIAL: best = -FLT_MAX; select only if s > best. NaN/-inf/-FLT_MAX
        // never satisfy the strict greater-than, so they are ineligible.
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
