// SPDX-License-Identifier: Apache-2.0
// hipfire-train — row-softmax forward + backward (fp32, un-fused training path).
//
// Forward (per row, length N):  p_i = exp(s_i - max) / Σ exp(s_j - max)
// Backward:  ds_i = p_i * ( dy_i - Σ_j dy_j p_j )
//
// The forward writes p into `y`; the backward consumes that same `y` (=p), so
// nothing is recomputed. One block per row, shared-mem reduction.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(64)
__global__ void softmax_train_fwd(
    const float* __restrict__ s, // [rows, N]
    float* __restrict__ y,       // [rows, N]  out (= p)
    int rows, int N
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* sr = s + (size_t)r * N;
    float* yr = y + (size_t)r * N;

    __shared__ float red[64];
    // max
    float m = -3.402823e38f;
    for (int i = tid; i < N; i += nt) m = fmaxf(m, sr[i]);
    red[tid] = m;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads();
    // sum exp
    float sum = 0.0f;
    for (int i = tid; i < N; i += nt) sum += expf(sr[i] - mx);
    red[tid] = sum;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int i = tid; i < N; i += nt) yr[i] = expf(sr[i] - mx) * inv;
}

extern "C" __launch_bounds__(64)
__global__ void softmax_train_bwd(
    const float* __restrict__ dy, // [rows, N]  upstream dL/dp
    const float* __restrict__ p,  // [rows, N]  saved softmax output
    float* __restrict__ ds,       // [rows, N]  out dL/ds
    int rows, int N
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* dyr = dy + (size_t)r * N;
    const float* pr = p + (size_t)r * N;
    float* dsr = ds + (size_t)r * N;

    __shared__ float red[64];
    float dot = 0.0f;
    for (int i = tid; i < N; i += nt) dot += dyr[i] * pr[i];
    red[tid] = dot;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float d = red[0];
    for (int i = tid; i < N; i += nt) dsr[i] = pr[i] * (dyr[i] - d);
}
