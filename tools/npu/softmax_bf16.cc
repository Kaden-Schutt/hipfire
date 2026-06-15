// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// AIE2 BF16 row softmax kernel for attention scores.
//
// Numerically-stable row softmax for one row of an [n_heads × ctx_len] QK^T
// score matrix. Each call processes one head's score row of length ctx_len.
// The IRON _transform_gen framework iterates over n_heads rows; ctx_len is
// the tile_size passed as the auto-appended 'n' parameter.
//
// Caller is responsible for pre-masking future positions with a large negative
// value (e.g. -1e9). Masked positions produce ~0 via exp underflow; the
// underflow clamp prevents undefined behavior in the bit-packing step.
//
// Based on the scalar polynomial exp from the official mlir-aie
// bf16_softmax.cc example (Apache-2.0), extended with:
//   (1) a max-finding pre-pass for numerical stability (shift by max)
//   (2) an explicit underflow clamp (ix < -127 → 0) to handle -inf mask values
//   (3) an eps guard on the sum to prevent NaN on all-masked rows
//
// ── Inputs / output ──────────────────────────────────────────────────────────
//   input  : [ctx_len] bf16   — attention scores for one head (tiled by IRON)
//   output : [ctx_len] bf16   — softmax probabilities
//   n      : int32            — ctx_len / tile_size (auto-appended by IRON)
//
// ── Computation ──────────────────────────────────────────────────────────────
//   max_val      = max(input[0..n))
//   exp_i        = exp(input[i] - max_val)           [see polynomial below]
//   output[i]    = exp_i / (sum(exp_j) + ε)
//
//   exp(x) approximation — 3-pass scalar:
//     y  = x * log2(e)
//     ix = floor(y)                      → packed into IEEE-754 exponent
//     fx = y - ix ∈ (-1, 0]
//     exp(x) ≈ 2^ix * (1 + ln2*fx + (ln2²/2)*fx²)   [degree-2 polynomial]
//   Accuracy: ~5 ULP, adequate for softmax normalization with BF16 output.
//
// ── IRON dispatch pattern ────────────────────────────────────────────────────
//   _transform_gen, single-core: tile_size = ctx_len (full row per tile).
//   Softmax requires a global max and global sum before any output can be
//   written — cannot split across columns. Each invocation processes one head.
//   N_div_n = n_heads.
//
// ── Shape constraints ────────────────────────────────────────────────────────
//   ctx_len % 16 == 0   (vector alignment, for future vectorization)
//   ctx_len ≤ 512       (fits in AIE local memory as 3 × ctx_len × 2 B)
//
// ── Built xclbins (target/npu/) ──────────────────────────────────────────────
//   qwen35-softmax-8h64ctx.xclbin    — 8 heads, ctx_len=64
//   qwen35-softmax-8h128ctx.xclbin   — 8 heads, ctx_len=128
//   qwen35-softmax-8h256ctx.xclbin   — 8 heads, ctx_len=256
//   qwen35-softmax-8h512ctx.xclbin   — 8 heads, ctx_len=512
//   Naming: qwen35-softmax-{n_heads}h{ctx_len}ctx.xclbin + -instr.bin
//
// ── Integration status ───────────────────────────────────────────────────────
//   NOT wired. Attention is handled by DFlash (GPU Flash Attention) on the
//   active decode path. This kernel would only be relevant for a small fixed
//   prefill context on the NPU, or if GPU attention were bypassed entirely.
//   The ctx_len=512 cap rules out production prefill lengths.
//
// ── AIE2 vs AIE2P ───────────────────────────────────────────────────────────
//   No arch-specific code. The exp polynomial uses scalar float arithmetic
//   (memcpy for bit-packing) which compiles identically on AIE2 and AIE2P.
//   No SIMD exp intrinsic is used — the polynomial is the portable path.

#include <aie_api/aie.hpp>
#include <stdint.h>

static void softmax_bf16_impl(bfloat16 *restrict input,
                               bfloat16 *restrict output, const int32_t n) {
    // --- Pass 1: find max ---
    float max_val = (float)input[0];
    for (int32_t i = 1; i < n; i++) {
        float v = (float)input[i];
        if (v > max_val) max_val = v;
    }

    // --- Pass 2: exp(x - max) → output[], accumulate sum ---
    // exp(x) = 2^floor(x*log2e) * 2^{x*log2e - floor(x*log2e)}
    // 2^iy packed into IEEE-754 exponent field; clamped to 0 for iy < -127.
    // 2^fy via degree-2 polynomial (valid for fy ∈ (-1, 0]).
    // From: mlir-aie/include/aie_kernels/aie2/bf16_softmax.cc (Apache-2.0)
    const float log2e  = 1.442695040888963f;
    const float ln2    = 0.6931471805599453f;
    const float ln2_sq = 0.2401598148889220f;
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        float x = (float)input[i] - max_val;
        float y = x * log2e;
        int32_t ix = (int32_t)y;         // truncate toward zero
        float fx = y - (float)ix;        // fractional part ∈ (-1, 0]
        float result;
        if (ix < -127) {
            result = 0.0f;               // underflow; handles masked -inf inputs
        } else {
            ix = (ix + 127) << 23;       // pack into IEEE-754 float exponent
            float pow2_ix;
            memcpy(&pow2_ix, &ix, sizeof(float));
            float pow2_fx = 1.0f + ln2 * fx + ln2_sq * fx * fx;
            result = pow2_ix * pow2_fx;
        }
        output[i] = (bfloat16)result;
        sum += result;
    }

    // --- Pass 3: normalize ---
    const float eps = 1e-7f;             // prevents NaN on all-masked rows
    float inv_sum = 1.0f / (sum + eps);
    for (int32_t i = 0; i < n; i++) {
        output[i] = (bfloat16)((float)output[i] * inv_sum);
    }
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t n) {
    softmax_bf16_impl(input, output, n);
}

} // extern "C"
