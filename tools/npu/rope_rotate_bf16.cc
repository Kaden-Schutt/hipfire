// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 RoPE rotation kernel (half-split convention)
//
// Adapted from the mlir_aie aie2p/rope.cc reference, which uses interleaved
// pair layout. This kernel uses the half-split layout matching hipfire's GPU
// rope_partial_halfsplit_f32 kernel so that the cos/sin tensor param and the
// output can be compared byte-for-byte with GPU reference runs.
//
// Qwen3.5-specific: partial_rotary_factor = 0.25, so only the first n_rot =
// head_dim / 4 dimensions are rotated. The remaining [n_rot, head_dim) are
// copied unchanged (passthrough). n_rot is derived at runtime from head_dim;
// no separate runtime parameter is needed.
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   input    : [head_dim] bf16   — one attention head (tiled by IRON)
//   output   : [head_dim] bf16   — rotated head
//   cs       : [n_rot]    bf16   — cos/sin tensor param (const, acquired once)
//   head_dim : int32             — tile size (auto-appended by IRON as 'n')
//
//   n_rot  = head_dim / 4       (partial_rotary_factor = 0.25)
//   n_rot2 = n_rot / 2          (number of (x, y) pairs)
//
// ── cos/sin buffer layout ────────────────────────────────────────────────────
//   Half-split: all cos first, all sin second.
//   cs[0 .. n_rot2)       = cos values for positions 0 .. n_rot2-1
//   cs[n_rot2 .. n_rot)   = sin values for positions 0 .. n_rot2-1
//
//   This matches the GPU rope_partial_halfsplit_f32 tensor format produced by
//   qwen35.rs `build_rope_cs()` so the same precomputed cs buffer can be used
//   on both GPU and NPU without reformatting.
//
// ── Rotation math (per pair i) ───────────────────────────────────────────────
//   x  = input[i],         y  = input[i + n_rot2]
//   x' = x * cos[i] - y * sin[i]
//   y' = y * cos[i] + x * sin[i]
//   output[i]        = x'
//   output[i+n_rot2] = y'
//   output[n_rot..head_dim] = input[n_rot..head_dim]   (passthrough)
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   _transform_gen, single-core: tile_size = head_dim, N_div_n = n_heads or n_kv_heads.
//   cs is a tensor param (same cos/sin for all heads at this token position,
//   acquired once before the head loop). No arch-specific guard needed — no
//   trig, no tanh, only arithmetic that is identical on AIE2 and AIE2P.
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   n_rot2            % 16 == 0   (rotation region, vector width)
//   head_dim - n_rot  % 16 == 0   (passthrough region, vector width)
//   For Qwen3.5 head_dim=256: n_rot2=32, passthrough=192 — both ✓
//
// ── Built xclbins (target/npu/) ──────────────────────────────────────────────
//   Q rope (n_heads Q heads):
//     qwen35-rope-q-8h256d.xclbin   — 0.8B/1.5B/2B  (8 Q heads, head_dim=256)
//     qwen35-rope-q-16h256d.xclbin  — 4B/9B          (16 Q heads, head_dim=256)
//   K rope (n_kv_heads KV heads):
//     qwen35-rope-k-2h256d.xclbin   — 0.8B/1.5B/2B  (2 KV heads, head_dim=256)
//     qwen35-rope-k-4h256d.xclbin   — 4B/9B          (4 KV heads, head_dim=256)
//   Naming: qwen35-rope-{q|k}-{n_heads}h{head_dim}d.xclbin + -instr.bin
//
// ── Integration status ───────────────────────────────────────────────────────
//   NOT wired in the active path. Was wired via try_npu_rope() but superseded
//   by headnorm_rope_bf16.cc, which fuses this rotation with the preceding
//   QK head norm in a single dispatch. Keeping these xclbins around as a
//   fallback for ablation studies or models without QK norm.

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
