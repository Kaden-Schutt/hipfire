// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 kernel: RoPE rotation (half-split convention)
//
// Adapted from the mlir_aie aie2p/rope.cc reference, which uses interleaved
// pair layout. This kernel uses the **half-split** layout matching hipfire's
// GPU rope_partial_halfsplit_f32 kernel:
//   - Positions [0, n_rot/2): the "x" component of each pair
//   - Positions [n_rot/2, n_rot): the "y" component of each pair
//   - Positions [n_rot, head_dim): pass-through (no rotation)
//
// cos/sin are pre-computed by the caller and passed in cs[]:
//   cs[0..n_rot/2)       = cos values for each pair
//   cs[n_rot/2..n_rot)   = sin values for each pair
//
// Qwen3.5-specific: n_rot = head_dim * partial_rotary_factor = head_dim / 4.
// The IRON _transform_gen framework does not support runtime scalar params beyond
// the auto-appended tile size, so n_rot is derived at runtime from head_dim.
// head_dim must be a multiple of 4 (holds for all Qwen3.5 configs).
//
// Called per head: the IRON framework iterates over all n_heads / n_kv_heads
// invocations, delivering one head (head_dim elements) per call.
// No __AIE_ARCH__ guard needed — no trig, no tanh, no arch-specific ops.

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, unsigned N>
static void rope_rotate_impl(const T *restrict input, T *restrict output,
                              const T *restrict cs,
                              int32_t head_dim) {
    // n_rot = head_dim * 0.25 (Qwen3.5 partial_rotary_factor = 0.25)
    const int32_t n_rot  = head_dim / 4;
    const int32_t n_rot2 = n_rot / 2;  // number of pairs

    // Rotation region: apply half-split RoPE to positions [0, n_rot)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t i = 0; i < n_rot2; i += N) {
        ::aie::vector<T, N> xv = ::aie::load_v<N>(input + i);
        ::aie::vector<T, N> yv = ::aie::load_v<N>(input + i + n_rot2);
        ::aie::vector<T, N> cv = ::aie::load_v<N>(cs + i);
        ::aie::vector<T, N> sv = ::aie::load_v<N>(cs + i + n_rot2);

        // x_rot = x * cos - y * sin
        ::aie::vector<T, N> xcos = ::aie::mul(xv, cv);
        ::aie::vector<T, N> ysin = ::aie::mul(yv, sv);
        ::aie::vector<T, N> xrot = ::aie::sub(xcos, ysin);

        // y_rot = y * cos + x * sin
        ::aie::vector<T, N> ycos = ::aie::mul(yv, cv);
        ::aie::vector<T, N> xsin = ::aie::mul(xv, sv);
        ::aie::vector<T, N> yrot = ::aie::add(ycos, xsin);

        ::aie::store_v(output + i,        xrot);
        ::aie::store_v(output + i + n_rot2, yrot);
    }

    // Pass-through region: copy positions [n_rot, head_dim) unchanged
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t j = n_rot; j < head_dim; j += N) {
        ::aie::store_v(output + j, ::aie::load_v<N>(input + j));
    }
}

extern "C" {

void rope_rotate_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                      bfloat16 *restrict cs,
                      int32_t head_dim) {
    rope_rotate_impl<bfloat16, 16>(input, output, cs, head_dim);
}

} // extern "C"
