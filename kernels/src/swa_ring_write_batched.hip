// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// SWA ring write — BATCHED (Phase B2, 2026-05-18).
//
// After the chunk-forward attention dispatch completes, advance the
// layer's SWA ring buffer to include this chunk's KVs so that future
// decode steps (or subsequent chunks) see the most-recent `window`
// positions in the ring.
//
// For each batch position b at absolute position p = start_pos + b:
//   slot = (start_pos + b) % window
//   ring[:, d, slot] = kv_batch[b, d]   for d in 0..head_dim
//
// If B > window, positions earlier in the chunk get overwritten by
// later ones — that's correct: only the last `window` positions
// should remain in the ring.
//
// Grid:  [ceil(head_dim/block), batch_size, 1]
// Block: [block, 1, 1]
//
// DeepSeek V4 has n_kv_heads = 1 so the loop is trivial; the kernel still
// loops to mirror the sequential signature.

#include <hip/hip_runtime.h>

#define BLOCK 256

extern "C" __global__ void swa_ring_write_batched_f32(
    const float* __restrict__ kv_batch,   // [B, head_dim]
    float*       __restrict__ cache,      // [n_kv_heads, head_dim, window]
    int n_kv_heads,
    int head_dim,
    int window,
    int start_pos,
    int batch_size
) {
    const int d = blockIdx.x * BLOCK + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= head_dim || b >= batch_size) return;

    const int slot = (start_pos + b) % window;
    const float v = kv_batch[(size_t)b * head_dim + d];
    for (int h = 0; h < n_kv_heads; ++h) {
        cache[((size_t)h * head_dim + d) * window + slot] = v;
    }
}
