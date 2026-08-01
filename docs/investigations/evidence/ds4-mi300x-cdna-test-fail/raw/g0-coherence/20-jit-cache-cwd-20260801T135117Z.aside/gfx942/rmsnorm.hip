// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
}

// gfx1151 decode variant: replace the full 256-entry LDS reduction and eight
// workgroup barriers with a wave-first reduction through eight LDS scalars.
// Two independent sum chains expose the same within-wave ILP lever that won
// for fused RMSNorm+FWHT in the gfx1151 autoresearch campaign.
extern "C" __global__ __launch_bounds__(256, 1)
void rmsnorm_f32_warp_reduce(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n,
    float eps
) {
    extern __shared__ float warp_sums[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    const int row = blockIdx.x * n;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    int i = tid;
    for (; i + blockDim.x < n; i += blockDim.x * 2) {
        const float v0 = x[row + i];
        const float v1 = x[row + i + blockDim.x];
        sum0 += v0 * v0;
        sum1 += v1 * v1;
    }
    if (i < n) {
        const float v = x[row + i];
        sum0 += v * v;
    }

    float sum = sum0 + sum1;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    float total = tid < n_warps ? warp_sums[tid] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down(total, offset);
        }
        if (tid == 0) {
            warp_sums[0] = rsqrtf(total / (float)n + eps);
        }
    }
    __syncthreads();
    const float rms = warp_sums[0];

    for (int d = tid; d < n; d += blockDim.x) {
        out[row + d] = x[row + d] * weight[d] * rms;
    }
}
