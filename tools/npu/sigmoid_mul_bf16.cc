// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 attention output gate kernel: out[i] = sigmoid(gate[i]) * x[i]
//
// Fuses two GPU ops that the Qwen3.5 FullAttention path runs after attention:
//   gpu.sigmoid_f32(gate)          [in-place sigmoid on gate vector]
//   gpu.mul_f32(attn_out, gate)    [elementwise multiply]
// Both are replaced by a single NPU dispatch of this kernel.
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   gate   : [n_heads × head_dim] bf16   — attention gate vector (tiled)
//   x      : [n_heads × head_dim] bf16   — attention output to scale (tiled)
//   out    : [n_heads × head_dim] bf16   — sigmoid(gate) * x
//   n      : int32                       — tile_size (auto-appended by IRON)
//
// ── Computation ──────────────────────────────────────────────────────────────
//   sigmoid(g) = 0.5 * (1 + tanh(0.5 * g))
//   out[i]     = sigmoid(gate[i]) * x[i]
//
//   Contrast with silu_mul_bf16.cc (SwiGLU):
//     SwiGLU: out = (g * sigmoid(g)) * up   — gate multiplied back in (2 muls)
//     This:   out = sigmoid(gate) * x        — plain sigmoid gate (1 mul)
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   transform_parallel_binary — work split across all 4 NPU columns.
//   Default tile_size=16; (n_heads × head_dim) must be divisible by 64.
//   For 0.8B: 8h × 256d = 2048 elements, 2048 % 64 == 0 ✓
//   For 4B/9B: 16h × 256d = 4096 elements, 4096 % 64 == 0 ✓
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   tile_size              % 16 == 0   (vector alignment)
//   n_heads × head_dim     % 64 == 0   (tile_size × 4 columns)
//
// ── Built xclbins (target/npu/) ──────────────────────────────────────────────
//   qwen35-attn-gate-8h256d.xclbin   — 0.8B/1.5B/2B  (8 Q heads, 256 head_dim)
//   qwen35-attn-gate-16h256d.xclbin  — 4B/9B          (16 Q heads, 256 head_dim)
//   Naming: qwen35-attn-gate-{n_heads}h{head_dim}d.xclbin + -instr.bin
//
// ── Integration status ───────────────────────────────────────────────────────
//   Wired: try_npu_attn_gate() in qwen35.rs FullAttention path (after attention
//   output, before residual add). Active when xclbin is present in target/npu/.
//
// ── AIE2 vs AIE2P ───────────────────────────────────────────────────────────
//   AIE2  (__AIE_ARCH__==20, NPU1/Phoenix): getTanhBf16() LUT-based tanh
//   AIE2P (__AIE_ARCH__==21, NPU2/Strix+): aie::tanh() hardware instruction

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
