// SPDX-License-Identifier: Apache-2.0
// hipfire-train — SwiGLU forward + backward (fp32, un-fused training path).
//
// Forward:  out = silu(gate) * up,  silu(g) = g * sigmoid(g)
// Backward (let s = sigmoid(gate)):
//   d_up   = d_out * silu(gate) = d_out * gate * s
//   d_gate = d_out * up * silu'(gate),  silu'(g) = s * (1 + g * (1 - s))
//
// Pure elementwise, one thread per element.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void swiglu_train_fwd(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g * sigmoidf(g)) * up[i];
}

extern "C" __global__ void swiglu_train_bwd(
    const float* __restrict__ d_out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ d_gate,
    float* __restrict__ d_up,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = sigmoidf(g);
    float silu = g * s;
    float dsilu = s * (1.0f + g * (1.0f - s));
    float go = d_out[i];
    d_up[i] = go * silu;
    d_gate[i] = go * up[i] * dsilu;
}
