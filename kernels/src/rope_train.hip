// SPDX-License-Identifier: Apache-2.0
// hipfire-train — RoPE forward + backward (fp32), HF-Llama half-split convention.
//
// Layout: x is [rows, D] with rows = seq*n_heads (row = t*n_heads + h); the
// position of a row is pos[row / n_heads]. For pair i in [0, D/2):
//   theta_i = base^(-2i/D),  angle = pos * theta_i,  c=cos, s=sin
// Forward (rotation by +angle):
//   out[i]      = a*c - b*s
//   out[i+D/2]  = b*c + a*s        with a=x[i], b=x[i+D/2]
// Backward (Rᵀ = rotation by -angle), a=dout[i], b=dout[i+D/2]:
//   dx[i]       = a*c + b*s
//   dx[i+D/2]   = -a*s + b*c
//
// RoPE is parameter-free, so only the input gradient is produced.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float rope_angle(float pos, int i, int D, float base) {
    float exponent = -2.0f * (float)i / (float)D;
    float theta = expf(exponent * logf(base));
    return pos * theta;
}

extern "C" __global__ void rope_train_fwd(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ pos, // [seq]
    int rows, int n_heads, int D, float base
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int half = D / 2;
    int t = r / n_heads;
    float p = pos[t];
    const float* xr = x + (size_t)r * D;
    float* orr = out + (size_t)r * D;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float ang = rope_angle(p, i, D, base);
        float c = cosf(ang), s = sinf(ang);
        float a = xr[i], b = xr[i + half];
        orr[i] = a * c - b * s;
        orr[i + half] = b * c + a * s;
    }
}

extern "C" __global__ void rope_train_bwd(
    const float* __restrict__ d_out,
    float* __restrict__ dx,
    const float* __restrict__ pos,
    int rows, int n_heads, int D, float base
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int half = D / 2;
    int t = r / n_heads;
    float p = pos[t];
    const float* dor = d_out + (size_t)r * D;
    float* dxr = dx + (size_t)r * D;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float ang = rope_angle(p, i, D, base);
        float c = cosf(ang), s = sinf(ang);
        float a = dor[i], b = dor[i + half];
        dxr[i] = a * c + b * s;
        dxr[i + half] = -a * s + b * c;
    }
}
