// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// KV-cache compaction: gather retained positions into a contiguous prefix.
//
// Used by TriAttention eviction (arXiv:2604.04921 §4.3): after scoring and
// top-B selection, copy the retained position rows from `src` to `dst`
// at new cache indices 0..B-1. The source byte layout is opaque here —
// we just copy `bytes_per_pos` bytes per new position — so the same
// kernel handles Q8_0, asym3, V-cache, etc.
//
// Post-RoPE correctness: new-position compaction is safe without
// re-RoPE because RoPE phase is already baked into the stored K when
// it was written. The flash attention reads K_post[j] which still
// represents a key post-rotated at its original absolute position;
// causal distance is implicit in the phase difference, not in the
// cache index.
//
// Grid: [budget, 1, 1]. Block: [N threads, 1, 1]. Picks one retained
// row per block, parallelizes the byte copy across threads.
#include <hip/hip_runtime.h>

extern "C" __global__ void kv_compact_gather(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const int* __restrict__ retain_indices,   // [budget] src position for each new slot
    int bytes_per_pos,
    int budget
) {
    const int new_pos = blockIdx.x;
    if (new_pos >= budget) return;
    const int src_pos = retain_indices[new_pos];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const unsigned char* sp = src + (size_t)src_pos * bytes_per_pos;
    unsigned char* dp = dst + (size_t)new_pos * bytes_per_pos;

    // Bulk copy via 16-byte uint4 loads/stores where the prefix allows.
    // Bytes_per_pos typically 1088 (Q8_0 9B), 800 (asym3 9B), etc. — all
    // ≥ 16 and usually divisible by 16, but we guard the tail either way.
    const int n_u4 = bytes_per_pos / 16;
    const uint4* sp4 = (const uint4*)sp;
    uint4* dp4 = (uint4*)dp;
    for (int i = tid; i < n_u4; i += nthreads) {
        dp4[i] = sp4[i];
    }
    const int tail_start = n_u4 * 16;
    for (int i = tail_start + tid; i < bytes_per_pos; i += nthreads) {
        dp[i] = sp[i];
    }
}
