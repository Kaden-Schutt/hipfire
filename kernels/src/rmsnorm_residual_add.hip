// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused sandwich post-norm + residual-add:
//   out[r,i] = residual[r,i] + rmsnorm(x[r,i], weight[i])
//
// Collapses the 3-launch (rmsnorm -> memcpy(out<-residual) -> add_inplace)
// pattern into ONE launch. One block per row, mirrors rmsnorm_f32's reduction.
// Math is identical to rmsnorm_f32 followed by elementwise add; only the ULP
// rounding of the (x*w*rms + residual) accumulation differs from doing the add
// in a separate kernel (both round in f32). gemma4's near-tie argmaxes may flip
// on one ULP — coherence-validated, not byte-identical.
extern "C" __global__ void rmsnorm_residual_add_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int base = row * n;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[base + i];
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
        int idx = base + i;
        out[idx] = residual[idx] + x[idx] * weight[i] * rms;
    }
}
