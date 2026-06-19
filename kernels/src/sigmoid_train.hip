// SPDX-License-Identifier: Apache-2.0
// hipfire-train — sigmoid forward + backward (fp32, un-fused training path).
// Used for the GLA-lite forget gate: g = sigmoid(g_pre).
//
// Forward:  out = sigmoid(x) = 1/(1+e^-x)
// Backward: d_x = d_out * out * (1 - out)
//
// Pure elementwise, one thread per element.

#include <hip/hip_runtime.h>

extern "C" __global__ void sigmoid_train_fwd(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / (1.0f + expf(-x[i]));
}

// `out` is the forward sigmoid output (saved), so backward needs no recompute.
extern "C" __global__ void sigmoid_train_bwd(
    const float* __restrict__ d_out,
    const float* __restrict__ out,
    float* __restrict__ d_x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = out[i];
    d_x[i] = d_out[i] * s * (1.0f - s);
}
