// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused sigmoid(gate) * x: replaces the FA attention epilogue sequence
//
//   sigmoid_f32(fa_gate);                    // gate = sigmoid(gate)
//   mul_f32(fa_attn_out, fa_gate, fa_attn_out); // out = out * gate
//
// with one launch. Saves one kernel per full-attention layer.
//
// Bit-exact with the separate kernels: each element is computed as
// `out[i] = out[i] * sigmoid(gate[i])` using the same sigmoid formula
// as sigmoid.hip and the same in-place multiplication as mul.hip.

extern "C" __global__ void sigmoid_mul_f32(
    float* __restrict__ out,         // in/out: out[i] *= sigmoid(gate[i])
    const float* __restrict__ gate,  // read-only gate
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = 1.0f / (1.0f + expf(-g));
    out[i] = out[i] * s;
}
