// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Shared-expert-folded combine for the MoE decode down path.
//
// Identical to `moe_down_combine_k8_batched` (kernels/src/
// moe_down_combine_k8_batched.hip) except it also folds the always-active
// shared-expert down output into the residual in the SAME launch:
//
//   x_residual[n,m] = (x_residual[n,m] + shared_out[n,m])
//                     + Σ_{k<K_TOP} topk_weights[n,k] * expert_outputs[n,k,m]
//
// `shared_out` holds the shared-expert down projection already scaled by
// sigmoid(shared_gate·x) — produced by `gemv_hfq4g256_sigmoid_scaled_write_gpu`
// (a bit-for-bit copy of the shared-expert residual GEMV that WRITES rather
// than accumulates). The shared term is added into the residual BEFORE the
// routed sum, exactly mirroring the baseline launch order (separate shared
// residual GEMV, then the routed combine). The routed reduction is the same
// unrolled-8 short-chain sum as the baseline kernel, so this composed path is
// byte-identical to the baseline while touching the residual stream with one
// read-modify-write instead of two.
//
// Shapes:
//   expert_outputs : [N × K_TOP × M] (f32)
//   topk_weights   : [N × K_TOP]     (f32)
//   shared_out     : [N × M]         (f32, sigmoid-scaled shared-expert down)
//   x_residual     : [N × M]         (f32, in-place += result)
//
// Grid: ((M + BLOCK_M - 1) / BLOCK_M, N). Block: BLOCK_M = 256 threads.

extern "C" __global__ void moe_down_combine_k8_shared_batched(
    const float* __restrict__ expert_outputs, // [N × K_TOP × M]
    const float* __restrict__ topk_weights,   // [N × K_TOP]
    const float* __restrict__ shared_out,     // [N × M]
    float*       __restrict__ x_residual,     // [N × M] accumulated in place
    int M, int K_TOP
) {
    const int bid = blockIdx.y;          // token id
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // m column
    if (col >= M) return;

    const long long token_base   = (long long)bid * K_TOP * M;
    const int       weights_base = bid * K_TOP;
    const long long residual_off = (long long)bid * M + col;

    // K_TOP is statically 8 per the indexed-batched kernel family contract.
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < K_TOP) {
            float w  = topk_weights[weights_base + k];
            float v  = expert_outputs[token_base + (long long)k * M + col];
            acc += w * v;
        }
    }

    // Shared-expert term folded first (matches baseline: shared residual GEMV
    // ran before the routed combine), then the routed sum. Same associativity
    // as the two-launch baseline → byte-identical.
    float base = x_residual[residual_off] + shared_out[residual_off];
    x_residual[residual_off] = base + acc;
}
