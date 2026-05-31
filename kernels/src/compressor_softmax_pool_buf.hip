// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HIP-graphs-safe twin of compressor_softmax_pool_f32. Reads the
// destination slot index from a device buffer; computes its own write
// pointer = kv_cache + slot * head_dim. Early-returns on slot < 0
// (sentinel for non-commit positions), letting the captured graph
// safely include the commit kernels at every replay even though the
// host only wants the actual commit on (pos+1)%ratio == 0.
//
// T (window length) and head_dim are constants (per layer's compress
// ratio), so they remain i32 kernargs — only the destination slot is
// state-derived.

#include <hip/hip_runtime.h>

#define BLOCK_HIDDEN 256

extern "C" __global__ void compressor_softmax_pool_f32_buf(
    const float* __restrict__ kv_state,    // [T, head_dim]
    const float* __restrict__ score_state, // [T, head_dim]
    float*       __restrict__ kv_cache,    // base pointer; [max_slots, head_dim]
    const int*   __restrict__ slot_buf,    // [1] — -1 to skip
    int T,
    int head_dim
) {
    const int slot = slot_buf[0];
    if (slot < 0) return;  // non-commit position: no-op
    const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
    if (d >= head_dim) return;

    // Pass 1: max over T for this d
    float max_s = score_state[0 * head_dim + d];
    for (int t = 1; t < T; ++t) {
        float s = score_state[t * head_dim + d];
        if (s > max_s) max_s = s;
    }

    // Pass 2: sum_exp and weighted-sum-of-kv together (one pass)
    float sum_exp = 0.0f;
    float weighted = 0.0f;
    for (int t = 0; t < T; ++t) {
        float e = __expf(score_state[t * head_dim + d] - max_s);
        sum_exp += e;
        weighted += e * kv_state[t * head_dim + d];
    }

    kv_cache[(size_t)slot * head_dim + d] = weighted / sum_exp;
}
