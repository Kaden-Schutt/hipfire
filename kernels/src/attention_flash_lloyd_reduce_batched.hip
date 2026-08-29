// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention reduce for the lloyd-V (FWHT-rotated centroid) path.
//
// V2 split-norm layout: tile kernels write rotated-V partials with two
// independent 128-wide halves at d = half*128 + tid*4 + i. This reduce
// combines those partials, then applies fwht_shfl_inverse once per half
// (two independent FWHT-128 inverses over four registers each) after the
// cross-tile combine — never fwht_shfl_inverse_256.
//
// Math: tile partials are rotated per half (g_tile ≈ H₁₂₈·Σ_tile w·v). The
// cross-tile combine forms g = Σ_tile corr·g_tile (weighted sum); the inverse
// H⁻¹ is linear and corr/inv_sum are scalars, so H⁻¹(Σ corr·g_tile)·inv_sum
// equals (Σ corr·H⁻¹(g_tile))·inv_sum — applying inverse once at the end is
// equivalent to per-tile.
//
// 2-pass numerics preserved EXACTLY (global max pass 1; single corr per tile
// pass 2; tile_sum <= 0 skip). Tree-mode (block_cols > 0) logic preserved.
//
// Grid: [n_heads, sub_batch_size, 1]. Block: [32, 1, 1]. Zero LDS.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_lloyd_reduce_batched(
    const float* __restrict__ partials,        // [sub_batch × n_heads × max_tiles × stride] (ROTATED V)
    float* __restrict__ out,                   // [sub_batch × n_heads × head_dim]
    const int* __restrict__ positions,         // [total_batch]
    int n_heads,
    int head_dim,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,                           // ignored when block_cols == 0
    int block_cols,                            // 0 = legacy causal, >0 = tree mode
    const float* signs1,                       // FWHT sign vectors (same as tile launch)
    const float* signs2
) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = 2 + head_dim;

    const int global_bid = batch_offset + local_bid;
    // hipGraph-safe block_start: under captured tree-verify replay the
    // `block_start` scalar kernarg is baked at capture time and goes stale as
    // the committed prefix grows. positions[] is re-uploaded (uncaptured) every
    // cycle and the linearized tree root sits at positions[0] == start_pos ==
    // block_start, so derive it from the device buffer instead of the scalar.
    // Eager (non-capture) value is identical, so this is a no-op off-graph. The
    // tile kernel applies the same fix; n_tiles here must match its seq_len.
    const int eff_block_start = (block_cols > 0) ? positions[0] : block_start;
    const int seq_len = (block_cols > 0) ? (eff_block_start + block_cols)
                                          : (positions[global_bid] + 1);
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;

    const float* my_partials = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride;

    // Pass 1: global max
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Pass 2: accumulate ROTATED dims — V2 half layout d = half*128 + tid*4 + i
    float global_sum = 0.0f;
    float g[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        float tile_max = p[0];
        float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
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
    const int q_dim = n_heads * head_dim;
    float* my_out = out + local_bid * q_dim + h * head_dim;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float a = g[half * 4 + 0];
        float b = g[half * 4 + 1];
        float c = g[half * 4 + 2];
        float dv = g[half * 4 + 3];
        fwht_shfl_inverse(a, b, c, dv, signs1, signs2, tid);
        const int d0 = half * 128 + tid * 4;
        my_out[d0 + 0] = a * inv_sum;
        my_out[d0 + 1] = b * inv_sum;
        my_out[d0 + 2] = c * inv_sum;
        my_out[d0 + 3] = dv * inv_sum;
    }
}
