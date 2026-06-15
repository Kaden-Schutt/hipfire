// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 per-head RMSNorm kernel (QK head norm)
//
// Applies weighted RMSNorm independently to each head in Q or K using a
// shared weight vector that is the same for every head. The IRON
// _transform_gen framework calls this once per head; the weight is a tensor
// param acquired once and reused across all head iterations, rather than
// being duplicated n_heads times in the input stream.
//
// This is a stepping-stone kernel superseded by headnorm_rope_bf16.cc, which
// fuses this norm with the subsequent RoPE rotation in a single dispatch.
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   input    : [head_dim] bf16   — one attention head's activations (tiled)
//   output   : [head_dim] bf16   — (x / rms(x)) * weight
//   weight   : [head_dim] bf16   — learned norm scale (tensor param, const)
//   head_dim : int32             — tile size (auto-appended by IRON as 'n')
//
//   C signature: rms_norm_head_bf16(input, output, weight, head_dim)
//   Note: arg order differs from rms_norm_weighted_bf16(input, weight, output, cols)
//   because _transform_gen places the tensor param after both I/O pointers.
//
// ── Computation ──────────────────────────────────────────────────────────────
//   rms(x)   = sqrt(mean(x²) + ε),   ε = 1e-5
//   out[i]   = (x[i] / rms(x)) * weight[i]
//   Same math as rms_norm_weighted_bf16; only the IRON wiring differs.
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   _transform_gen, single-core: tile_size = head_dim, N_div_n = n_heads or n_kv_heads.
//   Weight is a tensor param (buffer acquired once before the head loop, held
//   across all N_div_n invocations). Contrast with rms_norm_weighted_bf16.cc
//   where both input AND weight are tiled in the input stream.
//   IRON calls this kernel N_div_n times with:
//     - different input/output pointers (one head each)
//     - the same weight pointer (tensor param, no copy)
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   head_dim % 16 == 0   (vector width)
//
// ── Built xclbins (target/npu/) ──────────────────────────────────────────────
//   Q norm (n_heads Q heads):
//     qwen35-headnorm-q-8h256d.xclbin   — 0.8B/1.5B/2B  (8 Q heads, head_dim=256)
//     qwen35-headnorm-q-16h256d.xclbin  — 4B/9B          (16 Q heads, head_dim=256)
//   K norm (n_kv_heads KV heads):
//     qwen35-headnorm-k-2h256d.xclbin   — 0.8B/1.5B/2B  (2 KV heads, head_dim=256)
//     qwen35-headnorm-k-4h256d.xclbin   — 4B/9B          (4 KV heads, head_dim=256)
//   Naming: qwen35-headnorm-{q|k}-{n_heads}h{head_dim}d.xclbin + -instr.bin
//
// ── Integration status ───────────────────────────────────────────────────────
//   NOT wired in the active path. Superseded by headnorm_rope_bf16.cc (#5),
//   which fuses this norm with RoPE rotation in one dispatch and avoids a
//   second xclbin load plus separate host round-trip for cos/sin.
//   At 28 layers × 4 heads × 2 (Q+K) = 224 fewer dispatches per decode step.

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
