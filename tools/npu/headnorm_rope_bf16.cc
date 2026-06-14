// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 fused per-head QK norm + RoPE rotation kernel.
//
// Replaces two sequential NPU dispatches (headnorm + rope) with a single
// kernel invocation per head.  At 28 layers × 4 dispatches saved per layer,
// this recovers ~9.5 ms per decode step on Phoenix APU.
//
// Inputs / tensor param layout:
//   input           : [head_dim] bf16   — one head (tiled by IRON)
//   output          : [head_dim] bf16
//   packed_weight_cs: [head_dim + n_rot] bf16 — tensor param acquired once
//                       [0, head_dim)  = per-head norm weight
//                       [head_dim, head_dim+n_rot) = cos/sin buffer
//                       (n_rot = head_dim / 4; Qwen3.5 partial_rotary_factor=0.25)
//   head_dim        : int32 (auto-appended by IRON as 'n')
//
// cos/sin buffer layout: [cos_0..cos_{n_rot2-1}, sin_0..sin_{n_rot2-1}]
// where n_rot2 = n_rot / 2 = head_dim / 8.
//
// Algorithm per head:
//   Pass 1: compute sum(x_i^2) for RMS → inv_rms = 1/sqrt(mean_sq + eps)
//   Pass 2 (rotation region [0, n_rot)):
//     x_n   = x[i]       * weight[i]       * inv_rms
//     y_n   = x[i+n_rot2]* weight[i+n_rot2]* inv_rms
//     out[i]       = x_n*cos[i] - y_n*sin[i]
//     out[i+n_rot2] = y_n*cos[i] + x_n*sin[i]
//   Pass 3 (passthrough region [n_rot, head_dim)):
//     out[j] = x[j] * weight[j] * inv_rms
//
// Constraints (all hold for Qwen3.5 default head_dim=256, n_rot=64):
//   head_dim divisible by 16; n_rot2 divisible by 16; (head_dim-n_rot) divisible by 16.

#include <aie_api/aie.hpp>

static constexpr unsigned VEC = 16;

static void headnorm_rope_impl(
    const bfloat16 *restrict input,
    bfloat16       *restrict output,
    const bfloat16 *restrict packed_weight_cs,
    const int32_t   head_dim)
{
    const int32_t n_rot  = head_dim >> 2;   // head_dim / 4
    const int32_t n_rot2 = n_rot >> 1;      // head_dim / 8
    const bfloat16 *restrict weight = packed_weight_cs;
    const bfloat16 *restrict cs     = packed_weight_cs + head_dim;

    // Pass 1: accumulate sum of squares for RMS normalization
    float sum_sq = 0.0f;
    for (int32_t i = 0; i < head_dim; i += VEC) {
        auto v = aie::load_v<VEC>(input + i);
        sum_sq += aie::reduce_add(aie::mul(v, v).to_vector<float>());
    }
    float inv_rms = aie::invsqrt(
        aie::broadcast<float, VEC>(sum_sq / (float)head_dim + 1e-5f))[0];
    bfloat16 inv_rms_bf = (bfloat16)inv_rms;

    // Pass 2: normalize and rope-rotate the rotation region [0, n_rot)
    // Half-split layout: x-pair at (i, i+n_rot2), cos/sin at (i, i+n_rot2).
    for (int32_t i = 0; i < n_rot2; i += VEC) {
        auto xv = aie::load_v<VEC>(input  + i);
        auto yv = aie::load_v<VEC>(input  + i + n_rot2);
        auto wx = aie::load_v<VEC>(weight + i);
        auto wy = aie::load_v<VEC>(weight + i + n_rot2);
        auto cv = aie::load_v<VEC>(cs + i);
        auto sv = aie::load_v<VEC>(cs + i + n_rot2);

        // Normalize: x_n = x * inv_rms * weight
        auto xn = aie::mul(aie::mul(xv, inv_rms_bf).to_vector<bfloat16>(), wx)
                      .to_vector<bfloat16>();
        auto yn = aie::mul(aie::mul(yv, inv_rms_bf).to_vector<bfloat16>(), wy)
                      .to_vector<bfloat16>();

        // RoPE rotate: [x_n*c - y_n*s, y_n*c + x_n*s]
        aie::store_v(output + i,
                     aie::sub(aie::mul(xn, cv), aie::mul(yn, sv))
                         .to_vector<bfloat16>());
        aie::store_v(output + i + n_rot2,
                     aie::add(aie::mul(yn, cv), aie::mul(xn, sv))
                         .to_vector<bfloat16>());
    }

    // Pass 3: normalize the passthrough region [n_rot, head_dim) without rotation
    for (int32_t i = n_rot; i < head_dim; i += VEC) {
        auto v = aie::load_v<VEC>(input  + i);
        auto w = aie::load_v<VEC>(weight + i);
        aie::store_v(output + i,
                     aie::mul(aie::mul(v, inv_rms_bf).to_vector<bfloat16>(), w)
                         .to_vector<bfloat16>());
    }
}

extern "C" {

void headnorm_rope_bf16(bfloat16       *restrict input,
                         bfloat16       *restrict output,
                         const bfloat16 *restrict packed_weight_cs,
                         const int32_t   head_dim) {
    headnorm_rope_impl(input, output, packed_weight_cs, head_dim);
}

} // extern "C"
