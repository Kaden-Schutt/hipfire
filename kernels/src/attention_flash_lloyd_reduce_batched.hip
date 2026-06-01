// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention reduce for the lloyd-V (FWHT-rotated centroid) path.
//
// Based on attention_flash_asym_reduce_batched (which is already tid*8 with
// g[8] and d_base = tid*8). The one difference: the lloyd tile kernels now
// write their V partials in ROTATED space (the per-tile inverse FWHT was
// removed), so this reduce applies the inverse FWHT exactly ONCE, after the
// cross-tile combine and before the scalar normalize.
//
// Math: tile partials are rotated (g_tile ≈ H·Σ_tile w·v). The cross-tile
// combine forms g = Σ_tile corr·g_tile (weighted sum); the inverse H⁻¹ is
// linear and corr/inv_sum are scalars, so H⁻¹(Σ corr·g_tile)·inv_sum equals
// (Σ corr·H⁻¹(g_tile))·inv_sum — applying inverse once at the end is
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
    const int seq_len = (block_cols > 0) ? (block_start + block_cols)
                                          : (positions[global_bid] + 1);
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;

    const float* my_partials = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride;

    // Thread tid owns 8 contiguous dims (d_base = tid*8), covering all 256
    // dims of the lloyd-V head — the layout fwht_shfl_inverse_256 requires.
    const int d_base = tid * 8;

    // Pass 1: global max
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Pass 2: accumulate 8 (ROTATED) dims per thread
    float global_sum = 0.0f;
    float g[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        float tile_max = p[0];
        float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        float corr = expf(tile_max - global_max);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            g[i] += p[2 + d_base + i] * corr;
        global_sum += tile_sum * corr;
    }

    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // g is the cross-tile-combined accumulator in ROTATED space. Undo the
    // FWHT once (cooperatively across all 32 lanes), then normalize.
    fwht_shfl_inverse_256(g[0], g[1], g[2], g[3],
                          g[4], g[5], g[6], g[7],
                          signs1, signs2, tid);

    const int q_dim = n_heads * head_dim;
    float* my_out = out + local_bid * q_dim + h * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        my_out[d_base + i] = g[i] * inv_sum;
}
