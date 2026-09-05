// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Round-trip an F32 router-logits array through f16 in-place
// (F32 -> f16 -> F32, round-to-nearest-even).
//
// EschaLabs' runtime computes MoE router logits as `f16(x @ gate_w.T)` and
// only THEN widens back to F32 to run top-k. hipfire's router keeps logits
// in F32 end to end. The two selection functions differ whenever two
// experts' F32 logits round to the same f16 value AND straddle the top-k
// boundary (measured ~0.42% of router decisions, concentrated in a handful
// of layers). Escha's recovery fine-tune was trained and validated against
// the f16-rounding runtime, so hipfire's escha decode path must reproduce
// that rounding before top-k to avoid a silent, unexplained expert-choice
// divergence.
//
// Caller applies this ONLY on the escha decode path (gated on the model's
// routed-expert dtype being Escha2T16/Escha3T16 — see
// `MoeDtypes::has_escha_experts` in hipfire-dispatch), applied to the F32
// `router_logits` buffer BEFORE any top-k selection kernel runs (fused
// exact-wave64 kernel or the softmax_f32 + moe_topk_renorm_k8 fallback
// pair) so both routes see identically-rounded logits. Every other model /
// dtype takes the unmodified F32 path, byte-for-byte unchanged.
//
// Reuses the `f2h_rne` workaround pattern from escha_h128.hip: on this
// ROCm 7.2.2/gfx1151 build, `__float2half` loses the sign of a RUNTIME
// MULTIPLY-PRODUCED exact zero (see that file's comment for the full
// characterisation). Router logits here come from a GEMV reduction, not a
// bare multiply, so they are not known to hit that exact shape — but the
// exact-zero guard below is correct and free for any provenance, so it is
// applied unconditionally rather than relying on that distinction holding.
// `__float2half` itself is round-to-nearest-even for every other input, so
// no other special-casing is needed to satisfy the RNE requirement.
//
// Layout: in-place, single 1-D buffer of `n` F32 elements (one router's
// logits for one decode token). Grid: ceil(n / 256), Block: 256.
__device__ __forceinline__ __half f2h_rne_logit(float v) {
    if (v == 0.0f) {
        unsigned int bits = __float_as_uint(v);
        return __ushort_as_half((unsigned short)(bits >> 16));
    }
    return __float2half(v);
}

extern "C" __global__ void router_logits_round_f16_rne(
    float* __restrict__ logits,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = logits[i];
    if (!isfinite(v)) return;
    __half h = f2h_rne_logit(v);
    logits[i] = __half2float(h);
}
