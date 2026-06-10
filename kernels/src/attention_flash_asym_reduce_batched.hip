// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention reduce for asym KV modes (asym4 / asym3 / asym2).
// V is in NORMAL (un-rotated) space — no inverse rotation needed.
// Shared across all asym bit-widths because partials layout is dim-indexed
// (slot n holds the accumulated value for dim n regardless of which tid
// produced it).
//
// Grid: [n_heads, sub_batch_size, 1]. Block: [32, 1, 1]. Zero LDS.
#include <hip/hip_runtime.h>

// When `block_cols > 0` the reduce runs in DDTree tree-attention mode and
// sets seq_len = `block_start + block_cols` (matching the tile kernel).
// Otherwise it uses `positions[global_bid] + 1` (legacy causal). The tile
// kernel's bias is already baked into the partials, so reduce only needs
// to know how many tiles to combine.
extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_asym_reduce_batched(
    const float* __restrict__ partials,        // [sub_batch × n_heads × max_tiles × stride]
    float* __restrict__ out,                   // [sub_batch × n_heads × head_dim]
    const int* __restrict__ positions,         // [total_batch]
    int n_heads,
    int head_dim,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,                           // ignored when block_cols == 0
    int block_cols                             // 0 = legacy causal, >0 = tree mode
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

    // Pass 1: global max
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Pass 2: accumulate 8 dims per thread, looping head_dim/256 chunks.
    //
    // head_dim coverage: 32 threads × 8 dims = 256 dims per chunk. Gemma 4
    // full-attention layers use head_dim=512, so loop `n_chunks = head_dim/256`
    // chunks (matching the asym3 hd512 tile's lo/hi-chunk layout). Without
    // this loop, dims 256..511 were never written — the upper half of every
    // head's attention output silently kept the buffer's previous contents.
    // (Same bug class as the q8 single-token reduce fix that loops
    // n_halves=head_dim/128.) global_sum is identical across chunks; compute
    // it once during chunk 0.
    const int n_chunks = (head_dim + 255) / 256;
    const int q_dim = n_heads * head_dim;
    float* my_out = out + local_bid * q_dim + h * head_dim;
    float global_sum = 0.0f;
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        const int d_base = chunk * 256 + tid * 8;
        float g[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = my_partials + tile * stride;
            float tile_max = p[0];
            float tile_sum = p[1];
            if (tile_sum <= 0.0f) continue;
            float corr = expf(tile_max - global_max);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                if (d_base + i < head_dim) g[i] += p[2 + d_base + i] * corr;
            if (chunk == 0) global_sum += tile_sum * corr;
        }

        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            if (d_base + i < head_dim) my_out[d_base + i] = g[i] * inv_sum;
    }
}
