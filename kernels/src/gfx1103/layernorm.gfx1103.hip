// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1103 (Phoenix) LayerNorm with wave-reduced mean/variance.
//
// Same ABI and math contract as kernels/src/layernorm.hip. The generic kernel
// stores one partial per thread in LDS and reduces through a synchronized
// halving ladder over blockDim.x floats, twice (mean, then variance). gfx1103
// is wave32: each wave reduces its strided sum with shuffles and only one
// float per wave touches LDS. Keeps the Phoenix UMA path off heavy LDS.

#if defined(__gfx1103__)
#define HIPFIRE_GFX1103 1
#else
#define HIPFIRE_GFX1103 0
#endif

#if HIPFIRE_GFX1103
__device__ __forceinline__ float hipfire_wave_sum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 32);
    }
    return v;
}

// Block-wide sum reduce over up to 32 wave32 waves. Broadcasts the total to
// every thread via scratch[0]. Trailing barrier lets the caller reuse
// `scratch` for a subsequent reduction.
__device__ __forceinline__ float hipfire_block_sum(
    float local, float* scratch, int tid, int num_waves
) {
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const float wsum = hipfire_wave_sum32(local);
    if (lane == 0) scratch[wave] = wsum;
    __syncthreads();
    if (wave == 0) {
        float total = (lane < num_waves) ? scratch[lane] : 0.0f;
        total = hipfire_wave_sum32(total);
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();
    return result;
}
#endif

extern "C" __launch_bounds__(256, 1)
__global__ void layernorm_f32_gfx1103(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int n,
    float eps
) {
#if HIPFIRE_GFX1103
    extern __shared__ float wave_sums[];
    const int tid = threadIdx.x;
    const int num_waves = (blockDim.x + 31) >> 5;

    const int row = blockIdx.x;
    const float* xi = x + (size_t)row * n;
    float* oi = out + (size_t)row * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += xi[i];
    const float mean = hipfire_block_sum(sum, wave_sums, tid, num_waves) / (float)n;

    float var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = xi[i] - mean;
        var_sum += d * d;
    }
    const float inv_std =
        rsqrtf(hipfire_block_sum(var_sum, wave_sums, tid, num_waves) / (float)n + eps);

    for (int i = tid; i < n; i += blockDim.x) {
        oi[i] = gamma[i] * (xi[i] - mean) * inv_std + beta[i];
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n == 0) {
        out[0] = x[0] + gamma[0] + beta[0] + eps;
    }
#endif
}
