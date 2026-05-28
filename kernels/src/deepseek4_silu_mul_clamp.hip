// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 SwiGLU with swiglu_limit clamp.
//
// Per upstream `inference/model.py:Expert.forward`:
//   if swiglu_limit > 0:
//     up   = clamp(up,   -limit, +limit)   (symmetric)
//     gate = clamp(gate,         +limit)   (one-sided, cap from above only)
//   out  = silu(gate) * up
//
// DeepSeek V4 config: swiglu_limit = 10.0. With FP4-quantized routed weights and
// 4096-dim input dots, unclamped values can reach magnitudes that drive
// silu(gate) into a regime where small numerical noise causes attention/
// FFN feedback attractors. The clamp is part of DeepSeek V4's faithful math, not
// a numerical hack — `original_scores` of the router depend on these
// bounded magnitudes propagating through the residual stream.
//
// Drop-in replacement for `silu_mul_f32` when the layer has swiglu_limit
// > 0. Shared expert and each routed expert both benefit.
//
// Batched mode (2026-05-18): callers can dispatch with grid.y = batch_size
// to process `batch_size` independent streams in one launch. Each stream
// occupies `n` consecutive elements at stride `n` in the gate / up / out
// buffers. When the dispatch passes grid.y = 1 the kernel runs in its
// original single-stream mode (batch_off = 0, byte-identical).

#include <hip/hip_runtime.h>

extern "C" __global__ void deepseek4_silu_mul_clamp_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int n,
    float swiglu_limit
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long batch_off = (long long)blockIdx.y * n;
    float g = gate[batch_off + i];
    float u = up[batch_off + i];
    // DeepSeek V4 asymmetric clamp.
    if (g > swiglu_limit) g = swiglu_limit;
    if (u >  swiglu_limit) u =  swiglu_limit;
    if (u < -swiglu_limit) u = -swiglu_limit;
    float silu_g = g / (1.0f + __expf(-g));
    out[batch_off + i] = silu_g * u;
}
