// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 kernel: out[i] = sigmoid(gate[i]) * x[i]
//
// Used for the Qwen3.5 attention output gate:
//   gate_vec = sigmoid(gate_vec)   [in-place on GPU: gpu.sigmoid_f32]
//   attn_out = attn_out * gate_vec [gpu.mul_f32]
// Here they are fused into a single NPU dispatch.
//
// sigmoid(g) = 0.5 * (1 + tanh(0.5 * g))
//   AIE2  (__AIE_ARCH__==20): getLUT-based getTanhBf16
//   AIE2P (__AIE_ARCH__==21): hardware aie::tanh
//
// Contrast with silu_mul_bf16.cc (SwiGLU):
//   SwiGLU: out = (gate * sigmoid(gate)) * up   [2 muls per element]
//   This:   out = sigmoid(gate) * x             [1 mul per element]
//
// tile_size (n) must be a multiple of 16.
// Total elements (n_heads × head_dim) must be a multiple of tile_size × 4.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#if __AIE_ARCH__ == 20
#  include "lut_based_ops.h"
#  include "lut_based_ops.cpp"
#endif
#include <stdint.h>

using namespace aie;

static void sigmoid_mul_bf16_inner(bfloat16 *restrict gate, bfloat16 *restrict x,
                                    bfloat16 *restrict output, const int32_t n) {
    auto it_gate = aie::begin_restrict_vector<16>(gate);
    auto it_x    = aie::begin_restrict_vector<16>(x);
    auto it_out  = aie::begin_restrict_vector<16>(output);

    aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
    aie::vector<bfloat16, 16> one  = aie::broadcast<bfloat16, 16>(1.0f);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < n; i += 16) {
        aie::vector<bfloat16, 16> g = *it_gate++;
        aie::vector<bfloat16, 16> v = *it_x++;

        // sigmoid(g) = 0.5 * (1 + tanh(0.5 * g))
#if __AIE_ARCH__ == 20
        aie::vector<bfloat16, 16> half_g     = aie::mul(g, half);
        aie::vector<bfloat16, 16> tanh_hg    = getTanhBf16(half_g);
        aie::vector<bfloat16, 16> tanh_plus1 = aie::add(tanh_hg, one);
        aie::vector<bfloat16, 16> sig_g      = aie::mul(tanh_plus1, half);
#else
        auto half_g_acc  = aie::mul(g, half);
        auto tanh_hg     = aie::tanh<bfloat16>(half_g_acc.to_vector<float>());
        auto tanh_plus1  = aie::add(tanh_hg, one);
        aie::vector<bfloat16, 16> sig_g = aie::mul(tanh_plus1, half);
#endif

        // out = sigmoid(gate) * x
        auto result = aie::mul(sig_g, v);
        *it_out++ = result.to_vector<bfloat16>();
    }
}

extern "C" {

void sigmoid_mul_bf16(bfloat16 *restrict gate, bfloat16 *restrict x,
                       bfloat16 *restrict output, const int32_t n) {
    sigmoid_mul_bf16_inner(gate, x, output, n);
}

} // extern "C"
