// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention reduce kernel for the lloyd-V (FWHT-rotated centroid) path.
//
// Based on attention_flash_q8_0_reduce, with two differences:
//   1. The lloyd tile kernels now write their V partials in ROTATED space
//      (the per-tile inverse FWHT was removed). This reduce applies the
//      inverse exactly ONCE, after the cross-tile combine.
//   2. Because fwht_shfl_inverse_256 requires the 256-wide head distributed
//      as 8 contiguous dims per lane (d0 = tid*8) and uses cross-lane
//      shuffles across all 32 lanes cooperatively, the Pass-2 combine here
//      uses a single tid*8 pass over all 256 dims (head_dim is fixed at 256
//      for lloyd-V) — NOT the per-half (tid*4) loop the Q8/asym reduce uses.
//
// Math: the tile partials g_tile are rotated (g_tile ≈ H·Σ_tile w·v). The
// cross-tile combine forms g = Σ_tile corr·g_tile (a weighted sum) and the
// final normalize multiplies by the scalar inv_sum. Both the weighted sum
// and the scalar multiply commute with the linear inverse H⁻¹, so
//   H⁻¹(Σ corr·g_tile) · inv_sum == (Σ corr·H⁻¹(g_tile)) · inv_sum,
// i.e. applying the inverse once at the end is byte-equivalent in exact
// arithmetic to applying it per tile (and numerically equivalent up to FP
// reassociation that is no worse than the existing 2-pass accumulation).
//
// 2-pass numerics are preserved EXACTLY (global max in pass 1; single
// corr = expf(tile_max - global_max) per tile in pass 2; tile_sum <= 0 skip;
// global_sum += tile_sum*corr). Do NOT switch to online softmax.
//
// Grid: [n_heads, 1, 1]. Block: [32, 1, 1] (one WAVE32). Zero shared memory.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_lloyd_reduce(
    const float* __restrict__ partials,  // [n_heads, max_tiles, 2 + head_dim] (ROTATED V)
    float* __restrict__ out,             // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    const int* __restrict__ pos_buf,     // device buffer: pos_buf[0] = last position
    int tile_size,                       // tile_size used by tile kernel (= blockDim.x of tile)
    int max_tiles,                       // partials buffer capacity per head
    const float* __restrict__ signs1,    // FWHT sign vectors (same as tile launch)
    const float* __restrict__ signs2
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = 2 + head_dim;
    const int seq_len = pos_buf[0] + 1;
    const int n_tiles = min((seq_len + tile_size - 1) / tile_size, max_tiles);

    // ═══ Pass 1: find global max across all tiles ═══
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = partials + (h * max_tiles + tile) * stride;
        float tile_sum = p[1];
        if (tile_sum > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // ═══ Pass 2: accumulate (ROTATED) with single correction factor per tile ═══
    //
    // head_dim is 256 for lloyd-V. Thread tid owns the 8 contiguous dims
    // d0 = tid*8 .. d0+7, covering all 256 dims in ONE pass (no n_halves
    // loop) — this is the layout fwht_shfl_inverse_256 requires.
    const int d0 = tid * 8;
    float global_sum = 0.0f;
    float g[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = partials + (h * max_tiles + tile) * stride;
        float tile_max = p[0];
        float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;

        // Single correction: exp(tile_max - global_max).
        // Exact: no running corrections, no compounding error.
        float corr = expf(tile_max - global_max);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            g[i] += p[2 + d0 + i] * corr;
        global_sum += tile_sum * corr;
    }

    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // g is the cross-tile-combined accumulator in ROTATED space. Undo the
    // FWHT once (cooperatively across all 32 lanes), then normalize.
    fwht_shfl_inverse_256(g[0], g[1], g[2], g[3],
                          g[4], g[5], g[6], g[7],
                          signs1, signs2, tid);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        out[h * head_dim + d0 + i] = g[i] * inv_sum;
}
