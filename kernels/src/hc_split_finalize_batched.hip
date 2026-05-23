// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HC split + finalize — BATCHED (Phase B2, 2026-05-18).
//
// Combines the per-segment transforms applied after `hc_apply_alpha`
// in the sequential mhc_pre path:
//
//   pre[b, h]    = sigmoid(c[b, 0..4][h])
//   post[b, h]   = post_scale * sigmoid(c[b, 4..8][h])
//   comb[b, i]   = c[b, 8..24][i]
//
// where c is the post-α-rescale control vector. The split-into-contiguous
// outputs makes downstream kernels (hc_sinkhorn_4x4_batched on comb,
// hc_input_map_4stream_batched on pre, hc_mix_4stream_batched on post/
// comb) addressable without strided views.
//
// Grid:  [batch_size, 1, 1]
// Block: [24, 1, 1]   — one thread per c[b, i].
//
// At batch_size == 1 the math matches the sequential sigmoid+scale
// followed by the comb-view feeding sinkhorn.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_split_finalize_batched(
    const float* __restrict__ c,        // [B, 24]
    float*       __restrict__ pre,      // [B, 4]
    float*       __restrict__ post,     // [B, 4]
    float*       __restrict__ comb,     // [B, 16]
    float        post_scale,
    int          batch_size
) {
    const int b = blockIdx.x;
    const int i = threadIdx.x;
    if (b >= batch_size || i >= 24) return;

    const float v = c[(size_t)b * 24 + i];
    if (i < 4) {
        const float s = 1.0f / (1.0f + expf(-v));
        pre[(size_t)b * 4 + i] = s;
    } else if (i < 8) {
        const float s = 1.0f / (1.0f + expf(-v));
        post[(size_t)b * 4 + (i - 4)] = s * post_scale;
    } else {
        comb[(size_t)b * 16 + (i - 8)] = v;
    }
}
