// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Indexer compressed-K scoring kernel (DeepSeek V4 Phase 2).
//
// Computes per-head dot products of the current Q_idx against every
// position in the per-layer compressed-K cache. Output feeds into
// `indexer_top_k` selection, which feeds `kv_gather` from the main
// KV cache.
//
// Signature (subject to revision once Phase 2 forward wiring lands):
//
//   q_idx       : [n_idx_heads, idx_head_dim]      (fp16, current step)
//   k_idx_cache : [n_idx_heads, idx_head_dim, n_compressed]  (fp16)
//   scores      : [n_idx_heads, n_compressed]      (fp32, output)
//
//   n_idx_heads, idx_head_dim, n_compressed: scalar ints
//
// Grid:
//   block.x = 1 (per-head)
//   grid.x  = n_idx_heads        (parallel across heads)
//   grid.y  = ceil(n_compressed / TILE)   (split work along positions)
//
// Block layout:
//   THREADS_PER_BLOCK = 64
//   Each thread handles `idx_head_dim / 64` mantissa accumulations and
//   strides across `TILE` positions per outer pass.
//
// Status: STUB. Phase 2 work will (a) tune TILE for gfx1151 LDS budget,
// (b) add a wave-shuffle reduction across the head_dim accumulator,
// (c) micro-benchmark vs a Q8·Q8 reference. The kernel signature is
// the contract that `IndexerLayerState` and `kv_gather` will bind to.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define THREADS_PER_BLOCK 64
#define TILE_POSITIONS 8

__launch_bounds__(THREADS_PER_BLOCK, 8)
extern "C" __global__ void indexer_compressed_k_score(
    const __half* __restrict__ q_idx,        // [H, D]
    const __half* __restrict__ k_idx_cache,  // [H, D, N]
    float*        __restrict__ scores,        // [H, N]
    int n_idx_heads,                         // H
    int idx_head_dim,                        // D
    int n_compressed                          // N
) {
    const int head = blockIdx.x;
    const int tid  = threadIdx.x;
    if (head >= n_idx_heads) return;

    // Outer loop over position tiles.
    for (int n0 = blockIdx.y * TILE_POSITIONS;
         n0 < (blockIdx.y + 1) * TILE_POSITIONS && n0 < n_compressed;
         ++n0) {
        // Each thread accumulates a portion of the head_dim dot product
        // for ONE position.
        float acc = 0.0f;
        for (int d = tid; d < idx_head_dim; d += THREADS_PER_BLOCK) {
            const __half q_val = q_idx[head * idx_head_dim + d];
            const __half k_val = k_idx_cache[(head * idx_head_dim + d) * n_compressed + n0];
            acc += __half2float(q_val) * __half2float(k_val);
        }
        // Wave-shuffle reduction (wave32 on gfx11xx).
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) {
            scores[head * n_compressed + n0] = acc;
        }
    }
}
