// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Portable two-stage (merge-tree) indexer top-K for DeepSeek V4 MQ2R —
// levers F3 (gfx942) and G1 (gfx1151), both opt-in behind separate env
// flags. PORTABLE: the file and its symbols carry NO arch suffix; the same
// source compiles for gfx942 (wave64, CDNA3) and gfx1151 (wave32, RDNA3.5).
// See "Wave-size independence" below for why that is safe.
//
// WHAT IT REPLACES AND WHY
// ------------------------
// F2 shipped `indexer_top_k_buf_parallel_gfx942_bounded`, a tile-merge
// bitonic that cut the O(N^2) rank-count to O(N log^2 K) and took the full
// model at cap 32768 from 0.274 to 13.051 tok/s. But its grid is
// [n_idx_heads,1,1] and ds4 passes n_idx_heads == 1, so its tile loop —
// (G-1) sequential tile merges — is SERIAL on one CU of both the 304-CU
// MI300X and the ~40-CU Strix Halo. gfx1151 is hit identically: its
// promoted default is still the O(N^2) rank-count on one workgroup, and its
// bounded variant has the same single-workgroup serial tile loop. The
// single-CU serial wall is arch-independent.
//
// This file replaces the serial tile loop with a parallel merge tree:
//   stage 1  indexer_topk_chunk_sort — G = ceil(max_compressed/TOPK_RUN)
//            blocks, each bitonic-sorts its own TOPK_RUN-candidate chunk.
//   rounds   indexer_topk_merge_round — R = ceil(log2 G) rounds; round r
//            has ceil(G/2^(r+1)) blocks, each merging two sorted
//            TOPK_RUN-runs and keeping the better TOPK_RUN. Every block in
//            a round is independent and parallel.
//   finalize indexer_topk_finalize — one block writes run 0 out as
//            top_indices.
// Critical path collapses from 46 + (G-1)*57 barrier-separated passes
// (F2: 1 load + 45 sort passes for tile 0, then 57 per tile: 1 load +
// 45 sort + 1 reverse + 10 merge) to 46 + R*11 (per round: 1 fused
// load+reverse + log2(2*TOPK_RUN) = 10 merge stages). Calibrated against
// F2's MEASURED kernel times (the pass model reproduces them within 1.5%
// at N=8192 and N=32768 with 0.528 us/pass from 15.405 ms / 29173 passes
// at N=262144):
//
//   N        G    R   F2 passes  F3 passes  ratio   F2 ms    F3 ms (proj)
//   8192     16   4         901         94    10x    0.483    0.068
//   32768    64   6        3637        118    31x    1.938    0.086
//   262144  512   9       29173        154   189x   15.405    0.114
//
// (The shared-context projection assumed 12 passes per round — an unfused
// reversal. The reversal is fused into the load here, so the achieved
// count is 11 per round: 46 + R*11 = 90 / 112 / 145 passes at the three
// N above, i.e. marginally better than the table, ~0.077 ms at N=262144.)
// Model-level projection (21 ratio-4 indexer layers; non-top-K residual
// measured at 25.9 ms + 0.3067 ms per 1000 rows):
//   cap  8192: 25.94 -> 33.52 tok/s | cap 32768: 13.05 -> 26.48 |
//   cap 262144: 2.33 -> 9.20
//
// EXACTNESS — byte-identical IN ORDER to the incumbent rank-count kernel
// for every input. Three load-bearing rules, identical to F2:
//   1. Total order (score desc, index asc), via the VERBATIM helpers
//      topk_better / topk_swap / topk_bitonic_sort_desc /
//      topk_bitonic_merge_desc copied from
//      indexer_top_k_buf_bounded.gfx942.hip (themselves verbatim from the
//      proven gfx1151 network). Not re-derived.
//   2. Eligibility: a candidate is eligible iff score > -FLT_MAX (IEEE
//      excludes NaN, -INFINITY, -FLT_MAX; ds4 initialises unfilled indexer
//      score-pool slots to -inf). Canonicalise at LOAD to
//      (-INFINITY, INT_MAX), identical to out-of-range padding, so NaN
//      never reaches the comparator (a bitonic network needs a valid
//      total order). Filter at WRITE: a rank whose cached index is
//      INT_MAX was never an eligible candidate and writes -1.
//   3. Identity fast path: finalize short-circuits on
//      n_compressed <= max_k (NOT <= k) writing (i < n_compressed) ? i : -1
//      with NO eligibility filter, byte-exact with the gfx942 default
//      kernel's fast path.
// Decomposition correctness: requires k <= TOPK_RUN (same bound as F2's
// kpad <= TOPK_MAX_KPAD fast path; ds4 ships k <= 512, proven by F2's
// fast-path measurements). An element of the global top-K is then in the
// top-K of its own TOPK_RUN chunk, so the union of the per-chunk sorted
// runs contains the global top-TOPK_RUN. Each merge round keeps the
// top-TOPK_RUN of the union of its two input runs (both already sorted
// descending — the precondition the fused load reversal relies on to make
// the concatenation bitonic), so by induction run 0 after R rounds holds
// the global top-TOPK_RUN in order. Chunks beyond the live n_compressed
// contain only pads, which sort to run tails and filter to -1 at write.
//
// GRAPH SAFETY
// ------------
// The host computes n_runs = ceil(max_compressed/TOPK_RUN) and
// R = ceil(log2 n_runs) from HOST-KNOWN max_compressed and issues exactly
// 1 + R + 1 launches with grids [n_runs,1,1] / [n_pairs,1,1] / [1,1,1].
// Launch count and every grid derive ONLY from max_compressed, so a
// captured hipGraph is fixed; the live n_compressed reaches the device
// solely through n_compressed_buf, read on-device at replay. No host
// readback, no dynamic shared memory, no atomics — deterministic and
// tape-replay safe.
//
// SINGLE HEAD
// -----------
// ds4 passes n_idx_heads == 1. The workspace (ws_scores/ws_indices,
// [max_compressed] each) has no head dimension and the kernels take no
// head argument: this path must only ever be selected for the single-head
// case.
//
// WAVE-SIZE INDEPENDENCE
// ----------------------
// Block is 256 threads = 8 waves of 32 on gfx1151 (RDNA) or 4 waves of 64
// on gfx942 (CDNA); both fine. Correctness never depends on wave size:
//   * No warp-synchronous intrinsics anywhere — no __shfl*, no ballots,
//     no assumed synchronous lanes. All cross-lane exchange goes through
//     LDS + __syncthreads().
//   * Every __syncthreads() sits OUTSIDE per-lane predicates with
//     block-uniform loop bounds. The helpers' barrier is outside the
//     strided per-lane loop, so lanes that `continue` or run zero inner
//     iterations still arrive; widths are compile-time constants
//     (TOPK_RUN / 2*TOPK_RUN). Early returns are taken only on conditions
//     derived from blockIdx.x, kernargs, or buffer values identical for
//     every lane of the block (n_compressed_buf[0] is one address with no
//     concurrent writer), so either the whole block returns before any
//     barrier or none does.
//
// LDS BUDGET
// ----------
// chunk_sort: TOPK_RUN floats + TOPK_RUN ints = 4 KiB static.
// merge_round: 2*TOPK_RUN floats + 2*TOPK_RUN ints = 8 KiB static.
// finalize: none. Dynamic shared memory is 0 everywhere. 8 KiB fits the
// 64 KiB per-CU group-segment budget on BOTH arches with room for high
// occupancy — the entire point versus the portable serial
// indexer_top_k_buf, whose `taken` bitmap is n_compressed BYTES of dynamic
// LDS (attention.rs:9578-9579) and therefore hard-caps at N = 65536.

#include <hip/hip_runtime.h>
#include <float.h>
#include <limits.h>

static constexpr int TOPK_RUN = 512;

// Sort/merge helpers — VERBATIM copies from
// indexer_top_k_buf_bounded.gfx942.hip (themselves verbatim from the
// proven gfx1151 network). Wave-size agnostic as shipped: the strided
// per-lane loops use threadIdx.x/blockDim.x with `continue` skips, and
// every __syncthreads() is outside the per-lane loop with bounds derived
// only from the uniform `width` argument, so no wave can reach a
// different barrier count than its peers on either wave32 or wave64.

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

// Stage 1 — per-chunk sort. Block g loads candidates
// [g*TOPK_RUN, (g+1)*TOPK_RUN), canonicalising at load: index >=
// n_compressed OR !(score > -FLT_MAX) becomes (-INFINITY, INT_MAX),
// identical to padding, so ineligible entries sort deterministically to
// the run tail and NaN never enters the comparator. Sorts the chunk
// descending by (score, index asc) and stores it to ws_*[g*TOPK_RUN ..].
// Barrier-separated passes: 1 (post-load) + 45 (bitonic sort of
// TOPK_RUN = 512: sum_{s=1..9} s) = 46.
extern "C" __global__ void indexer_topk_chunk_sort(
    const float* __restrict__ scores,
    float*       __restrict__ ws_scores,
    int*         __restrict__ ws_indices,
    const int*   __restrict__ n_compressed_buf,
    int n_runs
) {
    const int g = blockIdx.x;
    // Block-uniform guard: grid is exactly [n_runs,1,1]; this only
    // hardens against a misconfigured launch and is taken by the whole
    // block or not at all, before any barrier.
    if (g >= n_runs) return;

    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int n_compressed = n_compressed_buf[0];
    const size_t run_base = (size_t)g * TOPK_RUN;

    __shared__ float cached_scores[TOPK_RUN];
    __shared__ int cached_indices[TOPK_RUN];

    for (int local = tid; local < TOPK_RUN; local += threads) {
        const int index = g * TOPK_RUN + local;
        const float score =
            (index < n_compressed) ? scores[index] : -INFINITY;
        const bool eligible = score > -FLT_MAX;
        cached_scores[local] = eligible ? score : -INFINITY;
        cached_indices[local] = eligible ? index : INT_MAX;
    }
    __syncthreads();

    topk_bitonic_sort_desc(cached_scores, cached_indices, 0, TOPK_RUN);
    // The sort ends with a barrier after its final compare-exchange, so
    // the LDS run is fully visible here; the store needs no extra sync.

    for (int local = tid; local < TOPK_RUN; local += threads) {
        ws_scores[run_base + local] = cached_scores[local];
        ws_indices[run_base + local] = cached_indices[local];
    }
}

// Merge rounds — round r (run_step = 2^r) has n_pairs = ceil(n_runs /
// 2^(r+1)) blocks. Run j lives at element offset j*TOPK_RUN. Pair p
// merges run lo_run = p*2*run_step with run lo_run+run_step into run
// lo_run, retaining the better TOPK_RUN of the union. Runs skipped by the
// hi_run >= n_runs guard are left untouched in place and are picked up by
// a later round with a larger run_step, so odd run counts merge
// correctly. Barrier-separated passes per round: 1 (post-load) +
// log2(2*TOPK_RUN) = 10 merge stages = 11.
extern "C" __global__ void indexer_topk_merge_round(
    float* __restrict__ ws_scores,
    int*   __restrict__ ws_indices,
    int n_runs,
    int run_step,
    int n_pairs
) {
    const int p = blockIdx.x;
    // Both guards are BLOCK-UNIFORM: they derive only from blockIdx.x and
    // the kernargs run_step/n_runs/n_pairs, identical for every lane of
    // the block. This is the one spot a reader would suspect divergence —
    // a block that returns here never reaches a __syncthreads() while a
    // sibling block does — but __syncthreads() is block-scoped, every
    // lane of THIS block takes the same branch, and no other block's
    // barrier depends on this block. There is no barrier hazard.
    if (p >= n_pairs) return;
    const int lo_run = p * 2 * run_step;
    const int hi_run = lo_run + run_step;
    if (hi_run >= n_runs) return;  // low run carries forward untouched

    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const size_t lo_base = (size_t)lo_run * TOPK_RUN;
    const size_t hi_base = (size_t)hi_run * TOPK_RUN;

    __shared__ float cached_scores[2 * TOPK_RUN];
    __shared__ int cached_indices[2 * TOPK_RUN];

    // Both input runs are already sorted descending (chunk_sort for
    // round 0; the previous round's writeback thereafter) — the
    // precondition that makes [lo desc, reverse(hi) asc] a bitonic
    // sequence. The reversal of the hi run is FUSED INTO THE LOAD: the
    // store index TOPK_RUN + local reads hi element TOPK_RUN-1-local.
    // This is exactly the same permutation as the load-then-reverse pass
    // in indexer_top_k_buf_bounded.gfx942.hip (swap(kpad+local,
    // 2*kpad-1-local)), hence bit-identical, and it saves one barrier
    // per round.
    for (int local = tid; local < TOPK_RUN; local += threads) {
        cached_scores[local] = ws_scores[lo_base + local];
        cached_indices[local] = ws_indices[lo_base + local];
    }
    for (int local = tid; local < TOPK_RUN; local += threads) {
        const int reversed = TOPK_RUN - 1 - local;
        cached_scores[TOPK_RUN + local] = ws_scores[hi_base + reversed];
        cached_indices[TOPK_RUN + local] = ws_indices[hi_base + reversed];
    }
    __syncthreads();

    topk_bitonic_merge_desc(cached_scores, cached_indices, 2 * TOPK_RUN);
    // The merge ends with a barrier after its final compare-exchange, so
    // the lower TOPK_RUN entries — the better half of the union, in
    // descending order — are fully visible to the writeback.

    for (int local = tid; local < TOPK_RUN; local += threads) {
        ws_scores[lo_base + local] = cached_scores[local];
        ws_indices[lo_base + local] = cached_indices[local];
    }
}

// Finalize — one block writes the surviving run 0 out as top_indices.
// Identity fast path: n_compressed <= max_k (NOT <= k) writes
// (i < n_compressed) ? i : -1 with NO eligibility filter — byte-exact
// with the gfx942 default kernel, whose fast path behaves exactly so.
// Otherwise rank < k reads run 0, mapping the INT_MAX sentinel
// (canonicalised ineligible/padding entry) to -1 and ranks >= k to -1.
// ws_scores is unused: the total order is fully encoded in the sorted
// index run after the merge tree.
extern "C" __global__ void indexer_topk_finalize(
    const float* __restrict__ ws_scores,
    const int*   __restrict__ ws_indices,
    int*         __restrict__ top_indices,
    const int*   __restrict__ n_compressed_buf,
    const int*   __restrict__ k_buf,
    int max_k
) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int n_compressed = n_compressed_buf[0];
    const int k = k_buf[0];
    (void)ws_scores;

    // Uniform early-out: n_compressed and max_k are buffer/kernarg values
    // identical for every lane of the block. No barrier exists on either
    // path, so there is nothing to diverge around.
    if (n_compressed <= max_k) {
        for (int i = tid; i < max_k; i += threads)
            top_indices[i] = (i < n_compressed) ? i : -1;
        return;
    }

    for (int rank = tid; rank < max_k; rank += threads) {
        int selected = -1;
        // rank < TOPK_RUN is a determinism clamp, not a behaviour change:
        // the merge tree retains only TOPK_RUN candidates, so selection
        // requires k <= TOPK_RUN (the same bound as F2's kpad fast path;
        // ds4 ships k <= 512). In that domain the clamp is a no-op.
        // Outside it, ranks past run 0 would read stale workspace from
        // other runs — clamp them to the deterministic -1 instead.
        if (rank < k && rank < TOPK_RUN) {
            const int index = ws_indices[rank];
            selected = (index == INT_MAX) ? -1 : index;
        }
        top_indices[rank] = selected;
    }
}
