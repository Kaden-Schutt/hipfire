// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Twin of swa_ring_write_f32, but reads `slot` from a device buffer
// instead of taking it as an i32 kernarg. HIP-graphs-safe — the slot
// value can be updated host-side between graph_launch() calls without
// re-capturing the graph.
//
// `slot_buf` points to a single i32 (often a slice of a larger state
// buffer like state.attn_state_buf[0]).
extern "C" __global__ void swa_ring_write_f32_buf(
    const float* __restrict__ kv,          // [head_dim]
    float*       __restrict__ cache,       // [n_kv_heads, head_dim, window]
    const int*   __restrict__ slot_buf,    // [1] i32
    int n_kv_heads,
    int head_dim,
    int window
) {
    const int slot = slot_buf[0];
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    for (int h = 0; h < n_kv_heads; ++h) {
        cache[(h * head_dim + d) * window + slot] = kv[d];
    }
}
