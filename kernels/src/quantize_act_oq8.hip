// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Opus Quant W8A8: dynamic per-token, per-group INT8 activation quantizer.
// Quantizes an already-rotated/smoothed f32 activation matrix into the signed
// int8 + per-group-scale format that `gemm_oq8_grouped_wmma` consumes. gfx1103
// wave32, ZERO LDS (wave-shuffle absmax reduction). The int8 generalization of
// quantize_act_oq4 — one int8 per element (no nibble packing), qmax 127.
//
//   X  : [B, K] f32 activations (assumed already FWHT-rotated / SmoothQuant'd)
//   Xq : [B, K] int8 (signed)
//   Xs : [B, K/group] f32 per-group scales
//
// grid = [K/group, B], block = [32]. group must be a multiple of 32; each lane
// owns group/32 contiguous elements.
// scale = absmax(group)/127 ; q = round(v/scale) clamped to [-127, 127].

#include <hip/hip_runtime.h>
#include <stdint.h>

__launch_bounds__(32)
extern "C" __global__ void quantize_act_oq8(
    const float*   __restrict__ X,
    int8_t*        __restrict__ Xq,
    float*         __restrict__ Xs,
    int B, int K, int group
) {
    const int b = blockIdx.y;
    const int g = blockIdx.x;
    if (b >= B) return;
    const int lane = threadIdx.x;
    const int per_lane = group / 32;
    const int n_groups = K / group;
    const long long g0 = (long long)g * group;
    const float* row = X + (long long)b * K + g0;

    // Wave-cooperative absmax over the group.
    float amax = 0.0f;
    for (int i = 0; i < per_lane; ++i) {
        amax = fmaxf(amax, fabsf(row[lane * per_lane + i]));
    }
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down(amax, off));
    }
    amax = __shfl(amax, 0); // broadcast group absmax from lane 0

    const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    if (lane == 0) {
        Xs[(long long)b * n_groups + g] = scale;
    }

    // Quantize this lane's elements (one int8 each).
    int8_t* out = Xq + (long long)b * K + g0;
    for (int i = 0; i < per_lane; ++i) {
        const int idx = lane * per_lane + i;
        int q = (int)lroundf(row[idx] * inv);
        q = max(-127, min(127, q));
        out[idx] = (int8_t)q;
    }
}
