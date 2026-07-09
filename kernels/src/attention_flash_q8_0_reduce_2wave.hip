// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Flash attention reduce kernel — SPLIT-K 2-WAVE-CONCURRENT variant.
//
// Value-preserving perf variant of attention_flash_q8_0_reduce, selected by
// HIPFIRE_SPLITK_2WAVE=1 at dispatch. Produces BIT-IDENTICAL output to the
// single-wave baseline; it only changes HOW the head_dim halves are mapped to
// hardware, not the arithmetic.
//
// Grid: [n_heads, 1, 1]. Block: [64, 1, 1] (TWO WAVE32s). Zero shared memory.
//
// Baseline (attention_flash_q8_0_reduce) uses one WAVE32 and iterates the
// `n_halves = head_dim / 128` halves SERIALLY on the same 32 lanes. For
// head_dim=256 (n_halves=2) that is two serial passes on the same wave. Here we
// launch two waves and give each wave one half: wave0 → half0, wave1 → half1,
// running CONCURRENTLY. The per-lane math is unchanged, so the result is
// identical to the baseline's serial reduce.
//
// Bit-identity argument (vs attention_flash_q8_0_reduce):
//   - global_max = max over tiles of p[0] (tile_sum>0). Order-independent and
//     exact; each wave recomputes it redundantly → same value in both waves.
//   - global_sum = Σ_tiles tile_sum · exp(tile_max − global_max), accumulated
//     in tile order 0..n_tiles-1 with the same tile_sum<=0 skip. The baseline
//     accumulates this exact sum, in this exact order, during its half==0 pass.
//     Recomputing it here in a dedicated tile-order loop yields the identical
//     floating-point result; both waves compute the same bits.
//   - g0..g3 per half = Σ_tiles p[2+d_base+i] · exp(tile_max − global_max),
//     same expression, same tile order, same correction factor as baseline.
//   - out = g · (1/global_sum). Identical.
//
// For head_dim=128 (n_halves=1) wave1's half-loop has zero iterations and wave0
// reproduces the baseline exactly; the second wave is simply idle.
//
// Works on RDNA1 (gfx1010), RDNA2 (gfx1030), RDNA3 (gfx1100), RDNA4 (gfx1200+).
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(64, 16)
__global__ void attention_flash_q8_0_reduce_2wave(
    const float* __restrict__ partials,  // [n_heads, max_tiles, 2 + head_dim]
    float* __restrict__ out,             // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    const int* __restrict__ pos_buf,     // device buffer: pos_buf[0] = last position
    int tile_size,                       // tile_size used by tile kernel
    int max_tiles                        // partials buffer capacity per head
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;         // 0..63
    const int wave = tid >> 5;           // 0 or 1
    const int lane = tid & 31;           // 0..31
    const int stride = 2 + head_dim;
    const int seq_len = pos_buf[0] + 1;
    const int n_tiles = min((seq_len + tile_size - 1) / tile_size, max_tiles);
    const int n_halves = head_dim / 128;

    // ═══ Pass 1: global max across all tiles (redundant per wave) ═══
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = partials + (h * max_tiles + tile) * stride;
        float tile_sum = p[1];
        if (tile_sum > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // ═══ global_sum: accumulate in tile order (bit-identical to the
    //     single-wave baseline's half==0 accumulation) ═══
    float global_sum = 0.0f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = partials + (h * max_tiles + tile) * stride;
        float tile_max = p[0];
        float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        global_sum += tile_sum * expf(tile_max - global_max);
    }
    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // ═══ Pass 2: each wave owns halves { wave, wave+2, ... } ═══
    // head_dim=256 → wave0 does half0, wave1 does half1 concurrently.
    for (int half = wave; half < n_halves; half += 2) {
        const int d_base = half * 128 + lane * 4;
        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;

        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = partials + (h * max_tiles + tile) * stride;
            float tile_max = p[0];
            float tile_sum = p[1];
            if (tile_sum <= 0.0f) continue;

            float corr = expf(tile_max - global_max);
            g0 += p[2 + d_base + 0] * corr;
            g1 += p[2 + d_base + 1] * corr;
            g2 += p[2 + d_base + 2] * corr;
            g3 += p[2 + d_base + 3] * corr;
        }

        out[h * head_dim + d_base + 0] = g0 * inv_sum;
        out[h * head_dim + d_base + 1] = g1 * inv_sum;
        out[h * head_dim + d_base + 2] = g2 * inv_sum;
        out[h * head_dim + d_base + 3] = g3 * inv_sum;
    }
}
