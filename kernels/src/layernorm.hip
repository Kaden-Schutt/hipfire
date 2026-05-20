// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void layernorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* xi = x + (size_t)row * n;
    float* oi = out + (size_t)row * n;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += xi[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = xi[i] - mean;
        var_sum += d * d;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        oi[i] = gamma[i] * (xi[i] - mean) * inv_std + beta[i];
    }
}
