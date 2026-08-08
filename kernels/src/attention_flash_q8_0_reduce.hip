// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention reduce kernel — 2-pass combination of tile partials.
//
// Grid: [n_heads, 1, 1]. Block: [256, 1, 1] (eight WAVE32s).
// Dynamic shared memory: max_tiles floats for the cross-tile correction
// factors. One thread retains the original sequential scalar reduction order;
// the wider block parallelizes the independent output dimensions.
//
// CRITICAL: uses 2-pass reduction (find global max first, then accumulate)
// instead of online softmax. Online softmax introduces ~1e-5 error per
// tile correction which is enough to flip argmax at decision boundaries.
// The 2-pass approach computes the exact same result as if all positions
// were processed in one pass (baseline-equivalent numerics).
//
// Pass 1: find global_max = max of all tile_max values
// Pass 2: accumulate exp(tile_max - global_max) * tile_out and sum
// Final: normalize by global_sum
//
// This is numerically equivalent to baseline because:
// - global_max = max(all scores) — exact, same as baseline
// - each tile_out = Σ exp(score - tile_max) * V
// - rescaled: tile_out * exp(tile_max - global_max)
//   = Σ exp(score - tile_max + tile_max - global_max) * V
//   = Σ exp(score - global_max) * V — exact
// - global_sum = Σ tile_sum * exp(tile_max - global_max) — exact
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(256, 4)
__global__ void attention_flash_q8_0_reduce(
    const float* __restrict__ partials,  // [n_heads, max_tiles, 2 + head_dim]
    float* __restrict__ out,             // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    const int* __restrict__ pos_buf,     // device buffer: pos_buf[0] = last position
    int tile_size,                       // tile_size used by tile kernel (= blockDim.x of tile)
    int max_tiles                        // partials buffer capacity per head
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = 2 + head_dim;
    const int seq_len = pos_buf[0] + 1;
    const int n_tiles = min((seq_len + tile_size - 1) / tile_size, max_tiles);

    extern __shared__ float corr_lut[]; // [max_tiles]
    __shared__ float global_max_shared;
    __shared__ float global_sum_shared;

    // Preserve the reference kernel's increasing-tile fmax order exactly.
    if (tid == 0) {
        float global_max = -1e30f;
        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            if (p[1] > 0.0f)
                global_max = fmaxf(global_max, p[0]);
        }
        global_max_shared = global_max;
    }
    __syncthreads();

    // Compute each correction once. The old 32-thread kernel recomputed the
    // same exp for every lane and for both 128-dimension halves.
    for (int tile = tid; tile < n_tiles; tile += blockDim.x) {
        const float* p = partials + (h * max_tiles + tile) * stride;
        corr_lut[tile] = p[1] > 0.0f ? expf(p[0] - global_max_shared) : 0.0f;
    }
    __syncthreads();

    // Preserve the reference kernel's increasing-tile global-sum order.
    if (tid == 0) {
        float global_sum = 0.0f;
        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            global_sum += p[1] * corr_lut[tile];
        }
        global_sum_shared = global_sum;
    }
    __syncthreads();

    const float inv_sum = global_sum_shared > 0.0f ? 1.0f / global_sum_shared : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float g = 0.0f;
        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            g += p[2 + d] * corr_lut[tile];
        }
        out[h * head_dim + d] = g * inv_sum;
    }
}
