// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 weighted RMSNorm kernel (hidden layer norm)
//
// Weighted RMSNorm — adds a learned per-element scale (gamma) to the
// mlir_aie reference aie2p/rms_norm.cc which hardcodes gamma=1.
// Ported from the AIE2P reference; works on AIE2 as-is since
// aie::invsqrt() is available on both generations.
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   input  : [hidden_size] bf16   — layer input (entire row in one tile)
//   weight : [hidden_size] bf16   — learned RMSNorm scale (gamma)
//   output : [hidden_size] bf16   — (x / rms(x)) * weight
//   cols   : int32                — hidden_size (auto-appended by IRON)
//
// ── Computation ──────────────────────────────────────────────────────────────
//   rms(x)    = sqrt(mean(x²) + ε),   ε = 1e-5
//   out[i]    = (x[i] / rms(x)) * weight[i]
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   _transform_gen, single-core: tile_size = hidden_size (entire row per tile).
//   RMSNorm requires a global reduction over all elements before any output
//   can be written, so the work cannot be split across columns. With
//   tile_size=hidden_size the AIE core holds the full row in its local memory
//   and computes sum(x²) before the normalize pass.
//   Memory budget: 3 × hidden_size × 2 B = 9 KB at 1536, 21 KB at 3584 —
//   both fit in the 32 KB AIE2 local memory.
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   hidden_size % 16 == 0   (vector width)
//   hidden_size × 6 B ≤ 32 KB   (3 buffers fit in AIE local memory)
//
// ── Built xclbins (target/npu/) ──────────────────────────────────────────────
//   qwen35-rmsnorm-1024.xclbin  — 0.8B  hidden size
//   qwen35-rmsnorm-1536.xclbin  — 1.5B  hidden size
//   qwen35-rmsnorm-2048.xclbin  — 2B    hidden size
//   qwen35-rmsnorm-2560.xclbin  — 4B    hidden size
//   qwen35-rmsnorm-3584.xclbin  — 7B    hidden size (also used for future sizes)
//   qwen35-rmsnorm-4096.xclbin  — 9B    hidden size
//   Naming: qwen35-rmsnorm-{hidden_size}.xclbin + -instr.bin
//
// ── Integration status ───────────────────────────────────────────────────────
//   NOT wired in the active decode path. The MQ4 hot path uses fused
//   fused_rmsnorm_rotate_mq (GPU), which combines rmsnorm + FWHT rotation.
//   This kernel is only useful when weights are non-MQ (BF16 hidden norms)
//   and the GPU is saturated. See project_npu_kernel_map.md for economics.
//
//   Compare rms_norm_head_bf16.cc: that kernel takes weight as a tensor param
//   (same weight reused for all heads); here both input and weight are tiled.

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
