// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 kernel: per-head RMSNorm (QK head norm)
//
// Applies weighted RMSNorm to a single head's data [head_dim] using a shared
// weight vector that is the same for all heads. The IRON _transform_gen
// framework calls this once per head; the weight is a tensor param acquired
// once for all head iterations (rather than a tiled input duplicated n_heads
// times).
//
// C signature follows the _transform_gen arg order:
//   (input=elem_in, output=elem_out, weight=param0, head_dim=auto-n)
// Compare rms_norm_weighted_bf16.cc which uses (input, weight, output, cols)
// — input and weight are both tiled there; here weight is a tensor param.
//
// Same math as rms_norm_weighted_bf16: two-pass, float32 accumulator.
// head_dim must be a multiple of 16.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

static const float HeadNorm_EPS = 1.0e-5f;

template <typename T, unsigned VecSize>
static void rms_norm_head_impl(T *restrict input, T *restrict output,
                                const T *restrict weight, int32_t head_dim) {
    // Pass 1: accumulate sum(x²) in float32
    aie::accum<accfloat, VecSize> acc = aie::zeros<accfloat, VecSize>();
    auto it1 = aie::begin_restrict_vector<VecSize>(input);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < head_dim; i += VecSize) {
        aie::vector<T, VecSize> v = *it1++;
        acc = aie::add(acc, aie::mul_square(v));
    }
    float sum_sq = aie::reduce_add(acc.template to_vector<float>());
    // Vector invsqrt — scalar version calls sqrtf which is unavailable bare-metal.
    aie::vector<float, VecSize> inv_vec =
        aie::invsqrt(aie::broadcast<float, VecSize>(sum_sq / float(head_dim) + HeadNorm_EPS));
    float inv_rms = inv_vec[0];
    aie::vector<T, VecSize> inv_rms_vec = aie::broadcast<T, VecSize>(T(inv_rms));

    // Pass 2: output[i] = input[i] * inv_rms * weight[i]
    auto it_in  = aie::begin_restrict_vector<VecSize>(input);
    auto it_w   = aie::begin_restrict_vector<VecSize>(weight);
    auto it_out = aie::begin_restrict_vector<VecSize>(output);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < head_dim; i += VecSize) {
        aie::vector<T, VecSize> x = *it_in++;
        aie::vector<T, VecSize> w = *it_w++;
        aie::vector<T, VecSize> xn = aie::mul(x, inv_rms_vec);
        auto result = aie::mul(xn, w);
        *it_out++ = result.template to_vector<T>();
    }
}

extern "C" {

void rms_norm_head_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                         const bfloat16 *restrict weight, const int32_t head_dim) {
    rms_norm_head_impl<bfloat16, 16>(input, output, weight, head_dim);
}

} // extern "C"
