// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 kernel: out[i] = silu(gate[i]) * up[i]
//
// Used in the Qwen3.5 dense FFN NPU path after the gate/up projections have
// already been computed on the GPU.  Adapted from the aie2/swiglu.cc kernel
// shipped with mlir_aie; simplified to remove the weight-matmul fuse.
// tile_size (n) must be a multiple of 16.
//
// AIE2 note: uses getTanhBf16() (LUT-based) not aie::tanh() (AIE2P hardware).

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
// AIE2 (NPU1, __AIE_ARCH__==20): getTanhBf16 is LUT-based.
// Include the .h (declares getTanhBf16 inline + extern table decls) then the
// .cpp (provides the actual table definitions) — the .cpp does NOT include the
// .h itself so both are required.
// AIE2P (NPU2, __AIE_ARCH__==21): hardware aie::tanh — no LUT needed.
#if __AIE_ARCH__ == 20
#  include "lut_based_ops.h"
#  include "lut_based_ops.cpp"
#endif
#include <stdint.h>

using namespace aie;

static void silu_mul_bf16_inner(bfloat16 *restrict gate, bfloat16 *restrict up,
                                bfloat16 *restrict output, const int32_t n) {
  auto it_gate = aie::begin_restrict_vector<16>(gate);
  auto it_up   = aie::begin_restrict_vector<16>(up);
  auto it_out  = aie::begin_restrict_vector<16>(output);

  aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 16> one  = aie::broadcast<bfloat16, 16>(1.0f);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < n; i += 16) {
    aie::vector<bfloat16, 16> g = *it_gate++;
    aie::vector<bfloat16, 16> u = *it_up++;

    // sigmoid(g) = 0.5 * (1 + tanh(0.5 * g))
    // Use explicit vector<bfloat16,16> for all intermediates — aie::mul returns
    // accum<__accfloat,16> and using auto keeps it as accum, which breaks chained
    // mul/add calls (no vec×accum overload exists).
    // sigmoid(g) = 0.5 * (1 + tanh(0.5*g))
    //
    // The two paths have different intermediate types so the whole block is
    // gated.  AIE2: aie::mul returns accum but we must use explicit
    // vector<bfloat16,16> for all intermediates to keep the chess scheduler
    // happy (no accum→vector bypass stalls pipeline).  AIE2P: auto is fine;
    // the hardware tanh takes vector<float> in and returns vector<bfloat16>
    // out, and to_vector<float>() is a method on accum not on vector.
#if __AIE_ARCH__ == 20
    aie::vector<bfloat16, 16> half_g     = aie::mul(g, half);
    aie::vector<bfloat16, 16> tanh_hg    = getTanhBf16(half_g);
    aie::vector<bfloat16, 16> tanh_plus1 = aie::add(tanh_hg, one);
    aie::vector<bfloat16, 16> sig_g      = aie::mul(tanh_plus1, half);
#else
    // Use auto for tanh/add intermediates (they're vectors); force explicit
    // vector<bfloat16,16> for sig_g so the subsequent mul(g, sig_g) resolves —
    // aie::mul(vec,vec) returns accum and there's no mul(vec,accum) overload.
    auto half_g_acc  = aie::mul(g, half);
    auto tanh_hg     = aie::tanh<bfloat16>(half_g_acc.to_vector<float>());
    auto tanh_plus1  = aie::add(tanh_hg, one);
    aie::vector<bfloat16, 16> sig_g = aie::mul(tanh_plus1, half);
#endif

    // silu(g) = g * sigmoid(g)
    aie::vector<bfloat16, 16> silu_g = aie::mul(g, sig_g);

    // out = silu(gate) * up
    auto result = aie::mul(silu_g, u);
    *it_out++ = result.to_vector<bfloat16>();
  }
}

extern "C" {

void silu_mul_bf16(bfloat16 *restrict gate, bfloat16 *restrict up,
                   bfloat16 *restrict output, const int32_t n) {
  silu_mul_bf16_inner(gate, up, output, n);
}

} // extern "C"
