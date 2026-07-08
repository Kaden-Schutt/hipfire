// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// DSpark bidirectional staging assembly (replaces a host d2h+assemble+h2d).
//
// Assemble staged[b, d, c] of shape [block, head_dim, stage_w] from:
//   - the committed main_kv ring (layout [n_kv=1, head_dim, win], so the
//     committed entry at column c lives at ring[d*win + c]); and
//   - the block KV [block, kv_dim] (kv_dim == head_dim for n_kv==1), so
//     block_kv[j*kv_dim + d].
//
// Column map (all `block` rows are IDENTICAL — bidirectional attention):
//   c in 0..n_committed                  -> ring[d*win + c]
//   c in n_committed..n_committed+block  -> block_kv[(c-n_committed)*kv_dim + d]
//   c >= n_committed+block               -> 0  (zeroed tail)
//
// Grid: (head_dim, block). Each thread owns one (d, b) and writes the full
// stage_w-column row for that (b, d). All threads in a block share the same b
// (blockIdx.y), so block_kv / ring reads coalesce well across d.
extern "C" __global__ void dspark_stage_kv(
    const float* __restrict__ ring,      // [head_dim * win]
    const float* __restrict__ block_kv,  // [block * kv_dim]
    float* __restrict__ staged,          // [block * head_dim * stage_w]
    int win,
    int kv_dim,
    int head_dim,
    int n_committed,
    int block,
    int stage_w
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= head_dim || b >= block) return;

    float* dst = staged + ((size_t)b * head_dim + d) * stage_w;

    for (int c = 0; c < n_committed; ++c) {
        dst[c] = ring[(size_t)d * win + c];
    }
    for (int j = 0; j < block; ++j) {
        dst[n_committed + j] = block_kv[(size_t)j * kv_dim + d];
    }
    for (int c = n_committed + block; c < stage_w; ++c) {
        dst[c] = 0.0f;
    }
}
