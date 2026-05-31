// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Indexer KV-gather (DeepSeek V4 Phase 2).
//
// Gathers raw K and V rows from the main attention KV cache at the
// position indices produced by `indexer_top_k`. The gathered rows
// are written to a per-step scratch buffer that the main attention
// kernel then reads alongside the SWA window.
//
// Signature (subject to revision):
//
//   k_main_cache   : [n_kv_heads, head_dim, max_seq]   (fp16, in)
//   v_main_cache   : [n_kv_heads, head_dim, max_seq]   (fp16, in)
//   top_indices    : [n_idx_heads, k]                  (i32, in)
//                      — each entry is a position in the compressed
//                        space; multiply by compress_ratio to recover
//                        the main-cache row index.
//   compress_ratio : scalar (per-layer)
//   k_gathered     : [n_kv_heads, head_dim, n_unique_indices]  (fp16, out)
//   v_gathered     : [n_kv_heads, head_dim, n_unique_indices]  (fp16, out)
//
// Open question (per design doc): top_indices is per-indexer-head
// (64 heads). For gather we either:
//   (a) gather UNION of all heads' top-k → up to 64·k unique positions
//   (b) gather PER-HEAD top-k separately
// (a) is what the DeepSeek V4 paper suggests; (b) wastes BW. Default to (a).
//
// Strategy: one warp per gathered row. The warp coalesced-reads the
// head_dim elements from main_cache at the indicated position and
// writes them to the scratch buffer in order.
//
// Status: STUB. Real impl needs:
//   - Union-of-indices preprocessing kernel (dedup + sort).
//   - This kernel takes the dedup'd list and does the gather.
//   - Currently the stub is a placeholder signature so the dispatch
//     wrapper can be wired against it.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define THREADS_PER_BLOCK 64

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void indexer_kv_gather(
    const __half* __restrict__ k_main_cache,   // [Hkv, D, max_seq]
    const __half* __restrict__ v_main_cache,   // [Hkv, D, max_seq]
    const int*    __restrict__ unique_indices, // [n_unique] (dedup'd union)
    __half*       __restrict__ k_gathered,      // [Hkv, D, n_unique]
    __half*       __restrict__ v_gathered,      // [Hkv, D, n_unique]
    int n_kv_heads,                            // Hkv (DeepSeek V4: 1)
    int head_dim,                              // D
    int max_seq,                               // KV cache capacity
    int n_unique,                              // length of unique_indices
    int compress_ratio                         // per-layer
) {
    const int row     = blockIdx.x;   // index into unique_indices
    const int kv_head = blockIdx.y;
    if (row >= n_unique || kv_head >= n_kv_heads) return;
    const int tid = threadIdx.x;

    const int pos_compressed = unique_indices[row];
    if (pos_compressed < 0) return;
    const int pos_main = pos_compressed * compress_ratio;
    if (pos_main >= max_seq) return;

    // Each thread copies a chunk of head_dim from main_cache to gathered.
    for (int d = tid; d < head_dim; d += THREADS_PER_BLOCK) {
        const long long src_k = ((long long)kv_head * head_dim + d) * max_seq + pos_main;
        const long long src_v = ((long long)kv_head * head_dim + d) * max_seq + pos_main;
        const long long dst_k = ((long long)kv_head * head_dim + d) * n_unique + row;
        const long long dst_v = ((long long)kv_head * head_dim + d) * n_unique + row;
        k_gathered[dst_k] = k_main_cache[src_k];
        v_gathered[dst_v] = v_main_cache[src_v];
    }
}
