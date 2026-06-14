// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 row softmax kernel for attention scores.
//
// Computes: output[i] = exp(input[i] - max) / sum_j(exp(input[j] - max))
// Caller pre-masks invalid positions with a large negative value; those
// produce 0 in the output via exp underflow.
//
// Uses the scalar polynomial exp from the official mlir-aie bf16_softmax.cc
// example, extended with:
//   1. A max-finding pre-pass for numerical stability.
//   2. An explicit underflow clamp (ix < -127 → 0) to avoid UB on the
//      bit-packed 2^ix computation and ensure masked -inf positions give 0.
//   3. eps guard on the sum to prevent NaN on all-masked inputs.
//
// n (tile_size, auto-appended by IRON) must be a multiple of 16 and ≤ 512.

#include <aie_api/aie.hpp>
#include <stdint.h>

static void softmax_bf16_impl(bfloat16 *restrict input,
                               bfloat16 *restrict output, const int32_t n) {
    // --- Pass 1: find max ---
    float max_val = (float)input[0];
    for (int32_t i = 1; i < n; i++) {
        float v = (float)input[i];
        if (v > max_val) max_val = v;
    }

    // --- Pass 2: exp(x - max) → output[], accumulate sum ---
    // exp(x) = 2^floor(x*log2e) * 2^{x*log2e - floor(x*log2e)}
    // 2^iy packed into IEEE-754 exponent field; clamped to 0 for iy < -127.
    // 2^fy via degree-2 polynomial (valid for fy ∈ (-1, 0]).
    // From: mlir-aie/include/aie_kernels/aie2/bf16_softmax.cc (Apache-2.0)
    const float log2e  = 1.442695040888963f;
    const float ln2    = 0.6931471805599453f;
    const float ln2_sq = 0.2401598148889220f;
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        float x = (float)input[i] - max_val;
        float y = x * log2e;
        int32_t ix = (int32_t)y;         // truncate toward zero
        float fx = y - (float)ix;        // fractional part ∈ (-1, 0]
        float result;
        if (ix < -127) {
            result = 0.0f;               // underflow; handles masked -inf inputs
        } else {
            ix = (ix + 127) << 23;       // pack into IEEE-754 float exponent
            float pow2_ix;
            memcpy(&pow2_ix, &ix, sizeof(float));
            float pow2_fx = 1.0f + ln2 * fx + ln2_sq * fx * fx;
            result = pow2_ix * pow2_fx;
        }
        output[i] = (bfloat16)result;
        sum += result;
    }

    // --- Pass 3: normalize ---
    const float eps = 1e-7f;             // prevents NaN on all-masked rows
    float inv_sum = 1.0f / (sum + eps);
    for (int32_t i = 0; i < n; i++) {
        output[i] = (bfloat16)((float)output[i] * inv_sum);
    }
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t n) {
    softmax_bf16_impl(input, output, n);
}

} // extern "C"
