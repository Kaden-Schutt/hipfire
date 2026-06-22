// SPDX-License-Identifier: Apache-2.0
// hipfire — deferred-hierarchical KV compression, Phase 2b: hot-tier (m,l) extract.
//
// The batched KVarN/asym flash writes per-tile partials [m, l, acc...] and a
// separate reduce (attention_flash_asym_reduce_batched) folds them into the
// normalized output. The reduce computes the final softmax stats global_max and
// global_sum internally but discards them. To let the hot flash tier participate
// in flash_tier_merge (hot ⊕ cold), we need those two scalars per (pos, head).
//
// This kernel re-reads the SAME partials buffer the flash already filled (it is
// caller-owned scratch, untouched by the reduce) and emits (m, l) — exactly the
// reduce's pass-1 max and rebased denom — without recomputing the output. Layout
// mirrors attention_flash_asym_reduce_batched verbatim:
//   partials = [sub_batch × n_heads × max_tiles × (2 + head_dim)], tile=[m,l,acc].
// Zero LDS — one thread per (head, pos). Cheap: decode is a handful of tiles.
//
// Grid: [n_heads, sub_batch, 1]. Block: [1, 1, 1].

#include <hip/hip_runtime.h>

extern "C" __global__ void flash_partials_ml(
    const float* __restrict__ partials,   // [sub_batch × n_heads × max_tiles × (2+head_dim)]
    const int*   __restrict__ positions,  // [total_batch]
    float* __restrict__ M_OUT,            // [sub_batch × n_heads] flash max
    float* __restrict__ L_OUT,            // [sub_batch × n_heads] flash denom (rebased to max)
    int n_heads, int head_dim, int tile_size, int max_tiles,
    int batch_offset, int block_start, int block_cols
) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    if (h >= n_heads) return;
    const int stride = 2 + head_dim;

    const int global_bid = batch_offset + local_bid;
    const int seq_len = (block_cols > 0) ? (block_start + block_cols)
                                          : (positions[global_bid] + 1);
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;

    const float* mp = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride;

    // Pass 1: global max over live tiles (l > 0).
    float gmax = -1e30f;
    for (int t = 0; t < n_tiles; ++t) {
        const float* p = mp + t * stride;
        if (p[1] > 0.0f) gmax = fmaxf(gmax, p[0]);
    }
    // Pass 2: rebased denom.
    float gsum = 0.0f;
    for (int t = 0; t < n_tiles; ++t) {
        const float* p = mp + t * stride;
        const float l = p[1];
        if (l <= 0.0f) continue;
        gsum += l * expf(p[0] - gmax);
    }

    const int idx = local_bid * n_heads + h;
    M_OUT[idx] = gmax;
    L_OUT[idx] = gsum;
}
