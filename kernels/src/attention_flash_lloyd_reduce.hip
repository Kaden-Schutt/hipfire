// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention reduce kernel for the lloyd-V (FWHT-rotated centroid) path.
//
// V2 split-norm layout: tile kernels write rotated-V partials with two
// independent 128-wide halves at d = half*128 + tid*4 + i. This reduce
// combines those partials, then applies fwht_shfl_inverse once per half
// (two independent FWHT-128 inverses over four registers each) after the
// cross-tile combine — never fwht_shfl_inverse_256.
//
// Math: the tile partials g_tile are rotated per half (g_tile ≈ H₁₂₈·Σ_tile w·v).
// The cross-tile combine forms g = Σ_tile corr·g_tile (a weighted sum) and the
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
    // V2: two independent 128-wide halves. Thread tid owns four dims per half
    // at d = half*128 + tid*4 + {0,1,2,3} — layout shared with tile writers.
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
        for (int half = 0; half < 2; half++) {
            const int d = half * 128 + tid * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                g[half * 4 + i] += p[2 + d + i] * corr;
        }
        global_sum += tile_sum * corr;
    }

    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // g is the cross-tile-combined accumulator in ROTATED space. Undo each
    // half's FWHT-128 independently (cooperatively across 32 lanes), then
    // normalize. signs1/signs2[0..127] shared by both halves.
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float a = g[half * 4 + 0];
        float b = g[half * 4 + 1];
        float c = g[half * 4 + 2];
        float dv = g[half * 4 + 3];
        fwht_shfl_inverse(a, b, c, dv, signs1, signs2, tid);
        const int d0 = half * 128 + tid * 4;
        out[h * head_dim + d0 + 0] = a * inv_sum;
        out[h * head_dim + d0 + 1] = b * inv_sum;
        out[h * head_dim + d0 + 2] = c * inv_sum;
        out[h * head_dim + d0 + 3] = dv * inv_sum;
    }
}
