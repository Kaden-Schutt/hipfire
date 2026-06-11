// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 RMSNorm with wave-reduced sum-of-squares.
//
// Same ABI and math contract as kernels/src/rmsnorm.hip. The generic kernel
// stores one partial per thread in LDS and reduces through a synchronized
// halving ladder. gfx1151 is wave32, so each wave can reduce its strided sum
// with shuffles; only one float per wave is written to LDS.

#if defined(__gfx1151__)
#define HIPFIRE_GFX1151 1
#else
#define HIPFIRE_GFX1151 0
#endif

__device__ __forceinline__ float hipfire_wave_sum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 32);
    }
    return v;
}

extern "C" __launch_bounds__(256, 1)
__global__ void rmsnorm_f32_gfx1151(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n,
    float eps
) {
#if HIPFIRE_GFX1151
    extern __shared__ float wave_sums[];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int wave_id = tid >> 5;
    const int num_waves = (blockDim.x + 31) >> 5;

    const long long row_off = (long long)blockIdx.x * n;
    const float* x_row = x + row_off;
    float* out_row = out + row_off;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x_row[i];
        local_sum += v * v;
    }

    const float wave_sum = hipfire_wave_sum32(local_sum);
    if (lane_id == 0) {
        wave_sums[wave_id] = wave_sum;
    }
    __syncthreads();

    float total = 0.0f;
    if (wave_id == 0) {
        total = (lane_id < num_waves) ? wave_sums[lane_id] : 0.0f;
        total = hipfire_wave_sum32(total);
        if (lane_id == 0) {
            wave_sums[0] = total;
        }
    }
    __syncthreads();

    const float inv_rms = rsqrtf(wave_sums[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * weight[i] * inv_rms;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n == 0) {
        out[0] = x[0] + weight[0] + eps;
    }
#endif
}
