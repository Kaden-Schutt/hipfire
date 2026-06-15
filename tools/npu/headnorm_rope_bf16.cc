// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 fused per-head QK norm + RoPE rotation kernel (ACTIVE)
//
// Fuses two operations that would otherwise require separate NPU dispatches
// (headnorm then rope) into a single kernel invocation per head. At 28 layers
// × 4 dispatches saved per layer, this recovers ~9.5 ms per decode step on
// Phoenix APU relative to running rms_norm_head_bf16 + rope_rotate_bf16.
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   input           : [head_dim] bf16   — one attention head (tiled by IRON)
//   output          : [head_dim] bf16   — normed + rotated head
//   packed_weight_cs: [head_dim + n_rot] bf16 — tensor param, acquired once:
//                       [0, head_dim)            = per-head norm weight (gamma)
//                       [head_dim, head_dim+n_rot) = cos/sin buffer
//   head_dim        : int32             — tile size (auto-appended by IRON)
//
//   n_rot  = head_dim / 4   (Qwen3.5 partial_rotary_factor = 0.25)
//   n_rot2 = n_rot / 2      (head_dim / 8; number of (x, y) pairs)
//
// ── cos/sin sub-buffer layout ────────────────────────────────────────────────
//   cs[0 .. n_rot2)         = cos values for positions 0 .. n_rot2-1
//   cs[n_rot2 .. n_rot)     = sin values for positions 0 .. n_rot2-1
//   Half-split layout, matching the GPU rope_partial_halfsplit_f32 format
//   produced by qwen35.rs build_rope_cs().
//
// ── Algorithm per head ───────────────────────────────────────────────────────
//   Pass 1: sum(x[i]²) for i in [0, head_dim) → inv_rms = 1/sqrt(mean + ε)
//   Pass 2 (rotation region, i in [0, n_rot2)):
//     x_n = x[i]       * weight[i]       * inv_rms
//     y_n = x[i+n_rot2]* weight[i+n_rot2]* inv_rms
//     output[i]        = x_n * cos[i] - y_n * sin[i]
//     output[i+n_rot2] = y_n * cos[i] + x_n * sin[i]
//   Pass 3 (passthrough region, i in [n_rot, head_dim)):
//     output[i] = x[i] * weight[i] * inv_rms
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   _transform_gen, single-core: tile_size = head_dim,
//   N_div_n = n_heads (for Q) or n_kv_heads (for K).
//   packed_weight_cs is a tensor param — same buffer reused for all heads.
//   The norm weight is shared across all Q (or K) heads; cos/sin is per-token
//   and updated by the host before each decode step.
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   head_dim           % 16 == 0   (VEC=16, all passes)
//   n_rot2 = head_dim/8 % 16 == 0  (rotation region in pass 2)
//   head_dim - n_rot   % 16 == 0   (passthrough region in pass 3)
//   For Qwen3.5 head_dim=256: n_rot2=32, passthrough=192 — all ✓
//
// ── Built xclbins (target/npu/) ──────────────────────────────────────────────
//   Q (n_heads Q heads):
//     qwen35-headnorm-rope-q-8h256d.xclbin   — 0.8B/1.5B/2B (8 Q, head_dim=256)
//     qwen35-headnorm-rope-q-16h256d.xclbin  — 4B/9B         (16 Q, head_dim=256)
//   K (n_kv_heads KV heads):
//     qwen35-headnorm-rope-k-2h256d.xclbin   — 0.8B/1.5B/2B (2 KV, head_dim=256)
//     qwen35-headnorm-rope-k-4h256d.xclbin   — 4B/9B         (4 KV, head_dim=256)
//   Naming: qwen35-headnorm-rope-{q|k}-{n_heads}h{head_dim}d.xclbin + -instr.bin
//
// ── Integration status ───────────────────────────────────────────────────────
//   ACTIVE — wired via try_npu_headnorm_rope() in qwen35.rs FullAttention path.
//   Called at 4 sites per layer (Q-pre-rope, K-pre-rope dispatches for both
//   Q and K tensors). Data path: GPU F32 → BF16 → NPU → BF16 → F32 → GPU.
//   Confirmed working: 0.8B, 4B, 9B BF16 and MQ4.
//
// ── AIE2 vs AIE2P ───────────────────────────────────────────────────────────
//   No arch-specific code. aie::invsqrt() is available on AIE2 (NPU1/Phoenix),
//   AIE2P (NPU2/Strix), and AIE1. No tanh/exp — only arithmetic operations.

#include <aie_api/aie.hpp>

static constexpr unsigned VEC = 16;

static void headnorm_rope_impl(
    const bfloat16 *restrict input,
    bfloat16       *restrict output,
    const bfloat16 *restrict packed_weight_cs,
    const int32_t   head_dim)
{
    const int32_t n_rot  = head_dim >> 2;   // head_dim / 4
    const int32_t n_rot2 = n_rot >> 1;      // head_dim / 8
    const bfloat16 *restrict weight = packed_weight_cs;
    const bfloat16 *restrict cs     = packed_weight_cs + head_dim;

    // Pass 1: accumulate sum of squares for RMS normalization
    float sum_sq = 0.0f;
    for (int32_t i = 0; i < head_dim; i += VEC) {
        auto v = aie::load_v<VEC>(input + i);
        sum_sq += aie::reduce_add(aie::mul(v, v).to_vector<float>());
    }
    float inv_rms = aie::invsqrt(
        aie::broadcast<float, VEC>(sum_sq / (float)head_dim + 1e-5f))[0];
    bfloat16 inv_rms_bf = (bfloat16)inv_rms;

    // Pass 2: normalize and rope-rotate the rotation region [0, n_rot)
    // Half-split layout: x-pair at (i, i+n_rot2), cos/sin at (i, i+n_rot2).
    for (int32_t i = 0; i < n_rot2; i += VEC) {
        auto xv = aie::load_v<VEC>(input  + i);
        auto yv = aie::load_v<VEC>(input  + i + n_rot2);
        auto wx = aie::load_v<VEC>(weight + i);
        auto wy = aie::load_v<VEC>(weight + i + n_rot2);
        auto cv = aie::load_v<VEC>(cs + i);
        auto sv = aie::load_v<VEC>(cs + i + n_rot2);

        // Normalize: x_n = x * inv_rms * weight
        auto xn = aie::mul(aie::mul(xv, inv_rms_bf).to_vector<bfloat16>(), wx)
                      .to_vector<bfloat16>();
        auto yn = aie::mul(aie::mul(yv, inv_rms_bf).to_vector<bfloat16>(), wy)
                      .to_vector<bfloat16>();

        // RoPE rotate: [x_n*c - y_n*s, y_n*c + x_n*s]
        aie::store_v(output + i,
                     aie::sub(aie::mul(xn, cv), aie::mul(yn, sv))
                         .to_vector<bfloat16>());
        aie::store_v(output + i + n_rot2,
                     aie::add(aie::mul(yn, cv), aie::mul(xn, sv))
                         .to_vector<bfloat16>());
    }

    // Pass 3: normalize the passthrough region [n_rot, head_dim) without rotation
    for (int32_t i = n_rot; i < head_dim; i += VEC) {
        auto v = aie::load_v<VEC>(input  + i);
        auto w = aie::load_v<VEC>(weight + i);
        aie::store_v(output + i,
                     aie::mul(aie::mul(v, inv_rms_bf).to_vector<bfloat16>(), w)
                         .to_vector<bfloat16>());
    }
}

extern "C" {

void headnorm_rope_bf16(bfloat16       *restrict input,
                         bfloat16       *restrict output,
                         const bfloat16 *restrict packed_weight_cs,
                         const int32_t   head_dim) {
    headnorm_rope_impl(input, output, packed_weight_cs, head_dim);
}

} // extern "C"
