// SPDX-License-Identifier: Apache-2.0
// hipfire-train — AdamW optimizer step (fp32), decoupled weight decay.
//
// Per parameter element (m, v are the fp32 moment buffers, persisted across
// steps; bc1 = 1-β1^t, bc2 = 1-β2^t are the host-computed bias corrections):
//   m = β1·m + (1-β1)·g
//   v = β2·v + (1-β2)·g²
//   p -= lr·( (m/bc1) / (sqrt(v/bc2) + eps) + wd·p )
//
// Matches sft.py's adamw_torch_fused (decoupled wd; wd=0 for Phase 0 LoRA).

#include <hip/hip_runtime.h>

extern "C" __global__ void adamw_step(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    int n,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float mhat = mi / bc1;
    float vhat = vi / bc2;
    float pi = p[i];
    p[i] = pi - lr * (mhat / (sqrtf(vhat) + eps) + wd * pi);
}
