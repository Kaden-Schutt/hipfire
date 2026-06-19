// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fused GeGLU: out = gelu_tanh(gate) * up. The gemma family uses the
// tanh-approximation GELU (`gelu_pytorch_tanh`) in its gated MLP, the GeGLU
// analogue of silu_mul.hip's SwiGLU. Same launch shape as silu_mul_f32.

#include <hip/hip_runtime.h>

extern "C" __global__ void gelu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        // gelu_pytorch_tanh: 0.5 v (1 + tanh(sqrt(2/pi) (v + 0.044715 v^3)))
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        float g = 0.5f * v * (1.0f + tanhf(inner));
        out[i] = g * up[i];
    }
}
