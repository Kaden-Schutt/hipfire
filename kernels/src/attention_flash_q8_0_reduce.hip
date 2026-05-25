// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention reduce kernel — 2-pass combination of tile partials.
//
// Grid: [n_heads, 1, 1]. Block: [32, 1, 1] (one WAVE32).
// Zero shared memory.
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

extern "C" __launch_bounds__(32, 16)
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

    // ═══ Pass 1: find global max across all tiles ═══
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = partials + (h * max_tiles + tile) * stride;
        float tile_sum = p[1];
        if (tile_sum > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // ═══ Pass 2: accumulate with single correction factor per tile ═══
    //
    // head_dim coverage: 32 threads × 4 dims = 128 dims per half. Qwen 3.5
    // uses head_dim=256, so we loop over `n_halves = head_dim / 128` and
    // cover both halves. Without this loop, dims 128..255 were never written
    // to the output — attention's top half was effectively zeroed, which
    // tolerated on some weight distributions (4B) but collapses output
    // into the pad token on others (9B, 27B). Matches the 2-pass pattern
    // used by kv_cache_write_asym_k_givens4_batched.
    //
    // global_sum is scalar and identical across halves — compute it once
    // during the half=0 pass and reuse for half>0.
    const int n_halves = head_dim / 128;
    float global_sum = 0.0f;
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;

        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            float tile_max = p[0];
            float tile_sum = p[1];
            if (tile_sum <= 0.0f) continue;

            // Single correction: exp(tile_max - global_max).
            // This is exact: no running corrections, no compounding error.
            float corr = expf(tile_max - global_max);

            g0 += p[2 + d_base + 0] * corr;
            g1 += p[2 + d_base + 1] * corr;
            g2 += p[2 + d_base + 2] * corr;
            g3 += p[2 + d_base + 3] * corr;
            if (half == 0) global_sum += tile_sum * corr;
        }

        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        out[h * head_dim + d_base + 0] = g0 * inv_sum;
        out[h * head_dim + d_base + 1] = g1 * inv_sum;
        out[h * head_dim + d_base + 2] = g2 * inv_sum;
        out[h * head_dim + d_base + 3] = g3 * inv_sum;
    }
}
