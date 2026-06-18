// SPDX-License-Identifier: Apache-2.0
// hipfire-train — RMSNorm forward + backward (fp32, un-fused training path).
//
// Forward:  y[t,i] = x[t,i] / r[t] * w[i],  r[t] = sqrt(mean_i x[t,i]^2 + eps)
// Backward (per row t, hidden dim H):
//   c[t]    = Σ_i dy[t,i] * w[i] * x[t,i]
//   dx[t,i] = (1/r[t]) * ( dy[t,i]*w[i] - x[t,i]*c[t] / (H * r[t]^2) )
//   dw[i]   = Σ_t dy[t,i] * x[t,i] / r[t]      (accumulated across rows)
//
// One block per row, wave32-strided reduction over H. dw uses atomicAdd since
// every row contributes to every weight element.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(64)
__global__ void rmsnorm_train_fwd(
    const float* __restrict__ x,   // [rows, H]
    const float* __restrict__ w,   // [H]
    float* __restrict__ y,         // [rows, H]
    float* __restrict__ rinv,      // [rows]  out: 1/r per row (saved for bwd)
    int rows, int H, float eps
) {
    int t = blockIdx.x;
    if (t >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* xr = x + (size_t)t * H;
    float* yr = y + (size_t)t * H;

    float ss = 0.0f;
    for (int i = tid; i < H; i += nthreads) ss += xr[i] * xr[i];
    // block reduce via shared mem
    __shared__ float red[64];
    red[tid] = ss;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float ri = rsqrtf(red[0] / (float)H + eps);
    if (tid == 0) rinv[t] = ri;
    for (int i = tid; i < H; i += nthreads) yr[i] = xr[i] * ri * w[i];
}

extern "C" __launch_bounds__(64)
__global__ void rmsnorm_train_bwd(
    const float* __restrict__ dy,   // [rows, H]
    const float* __restrict__ x,    // [rows, H]
    const float* __restrict__ w,    // [H]
    const float* __restrict__ rinv, // [rows]  (1/r from forward)
    float* __restrict__ dx,         // [rows, H]  out
    float* __restrict__ dw,         // [H]        out (atomic-accumulated)
    int rows, int H
) {
    int t = blockIdx.x;
    if (t >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* dyr = dy + (size_t)t * H;
    const float* xr = x + (size_t)t * H;
    float* dxr = dx + (size_t)t * H;
    float ri = rinv[t];

    // c = Σ_i dy*w*x
    float c = 0.0f;
    for (int i = tid; i < H; i += nthreads) c += dyr[i] * w[i] * xr[i];
    __shared__ float red[64];
    red[tid] = c;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float csum = red[0];
    float coef = csum * ri * ri / (float)H; // c / (H r^2)

    for (int i = tid; i < H; i += nthreads) {
        float dxi = ri * (dyr[i] * w[i] - xr[i] * coef);
        dxr[i] = dxi;
        if (dw) atomicAdd(&dw[i], dyr[i] * xr[i] * ri);
    }
}
