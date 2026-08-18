// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// DeepSeek V4 mHC pre+post sigmoid/scale fusion. Replaces three separate
// element-wise launches (sigmoid(pre), sigmoid(post), scale(post))
// per mhc_pre call with a single 8-thread launch. mhc_pre fires twice
// per layer (attn-side + ffn-side) × 43 layers = 86 launches/token;
// each fused launch saves ~10μs of launch overhead = ~860μs/decode
// (~1.5% improvement on a 56ms decode).
//
// Layout of hc_c[24]:
//   [0..4]   pre  — input mixing weights:  sigmoid(c[i]) + hc_eps
//   [4..8]   post — output scaling factors: post_scale * sigmoid(c[i])
//   [8..24]  comb — 4x4 sinkhorn matrix; UNTOUCHED (sinkhorn iterates)
extern "C" __global__ void hc_pre_post_sigmoid_scale_f32(
    float* __restrict__ hc_c,
    float hc_eps,
    float post_scale
) {
    const int i = threadIdx.x;
    if (i < 4) {
        // pre: sigmoid + eps
        float z = hc_c[i];
        hc_c[i] = 1.0f / (1.0f + __expf(-z)) + hc_eps;
    } else if (i < 8) {
        // post: scaled sigmoid
        float z = hc_c[i];
        hc_c[i] = post_scale * (1.0f / (1.0f + __expf(-z)));
    }
    // i >= 8: comb region — left alone for sinkhorn pass.
}
