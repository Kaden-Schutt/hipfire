// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 kernel: out[i] = (x[i] / rms(x)) * weight[i]
//
// Weighted RMSNorm — adds a learned per-element scale (gamma) to the
// mlir_aie reference aie2p/rms_norm.cc which hardcodes gamma=1.
// Ported from the AIE2P reference; works on AIE2 as-is since
// aie::invsqrt() is available on AIE2, AIE2P, and AIE1 — no arch guard needed.
// cols must be a multiple of 16.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

static const float RMSNorm_EPS = 1.0e-5f;

template <typename T, unsigned VecSize>
static void rms_norm_weighted_impl(T *restrict input, T *restrict weight,
                                    T *restrict output, int32_t cols) {
    // Pass 1: accumulate sum(x²) into a float32 accumulator.
    // aie::mul_square on bfloat16 returns accum<accfloat,N> — float32 precision.
    aie::accum<accfloat, VecSize> acc = aie::zeros<accfloat, VecSize>();
    auto it1 = aie::begin_restrict_vector<VecSize>(input);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < cols; i += VecSize) {
        aie::vector<T, VecSize> v = *it1++;
        acc = aie::add(acc, aie::mul_square(v));
    }
    float sum_sq = aie::reduce_add(acc.template to_vector<float>());
    // Scalar aie::invsqrt(float) calls sqrtf which is unavailable bare-metal.
    // Instead broadcast the scalar into a float vector and use the vector
    // hardware RSQRT instruction, then extract element 0.
    aie::vector<float, VecSize> inv_vec =
        aie::invsqrt(aie::broadcast<float, VecSize>(sum_sq / float(cols) + RMSNorm_EPS));
    float inv_rms = inv_vec[0];

    // Pass 2: out[i] = input[i] * inv_rms * weight[i]
    // Broadcast inv_rms in bfloat16 (7 mantissa bits sufficient for the scale).
    aie::vector<T, VecSize> inv_rms_vec = aie::broadcast<T, VecSize>(T(inv_rms));

    auto it_in  = aie::begin_restrict_vector<VecSize>(input);
    auto it_w   = aie::begin_restrict_vector<VecSize>(weight);
    auto it_out = aie::begin_restrict_vector<VecSize>(output);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < cols; i += VecSize) {
        aie::vector<T, VecSize> x = *it_in++;
        aie::vector<T, VecSize> w = *it_w++;
        // x * inv_rms → still T (mul of two bfloat16 vecs returns accum → vector)
        aie::vector<T, VecSize> xn = aie::mul(x, inv_rms_vec);
        // * weight
        auto result = aie::mul(xn, w);
        *it_out++ = result.template to_vector<T>();
    }
}

extern "C" {

void rms_norm_weighted_bf16(bfloat16 *restrict input, bfloat16 *restrict weight,
                             bfloat16 *restrict output, const int32_t cols) {
    rms_norm_weighted_impl<bfloat16, 16>(input, weight, output, cols);
}

} // extern "C"
