// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1103 (Phoenix) in-place row softmax with wave-reduced max/sum.
//
// Same ABI and math contract as kernels/src/softmax.hip. The generic kernel
// reduces max then sum through a blockDim.x LDS halving ladder. gfx1103 is
// wave32: each wave reduces with shuffles and only one float per wave touches
// LDS. Keeps the Phoenix UMA path off heavy LDS.

#if defined(__gfx1103__)
#define HIPFIRE_GFX1103 1
#else
#define HIPFIRE_GFX1103 0
#endif

#if HIPFIRE_GFX1103
__device__ __forceinline__ float hipfire_wave_sum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
    return v;
}
__device__ __forceinline__ float hipfire_wave_max32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down(v, offset, 32));
    return v;
}
__device__ __forceinline__ float hipfire_block_reduce(
    float local, float* scratch, int tid, int num_waves, bool is_max
) {
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const float wred = is_max ? hipfire_wave_max32(local) : hipfire_wave_sum32(local);
    if (lane == 0) scratch[wave] = wred;
    __syncthreads();
    if (wave == 0) {
        float ident = is_max ? -1e30f : 0.0f;
        float total = (lane < num_waves) ? scratch[lane] : ident;
        total = is_max ? hipfire_wave_max32(total) : hipfire_wave_sum32(total);
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();
    return result;
}
#endif

extern "C" __launch_bounds__(256, 1)
__global__ void softmax_f32_gfx1103(
    float* __restrict__ x,
    int n
) {
#if HIPFIRE_GFX1103
    extern __shared__ float wave_scratch[];
    const int tid = threadIdx.x;
    const int num_waves = (blockDim.x + 31) >> 5;
    float* row_data = x + (size_t)blockIdx.x * n;

    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x)
        local_max = fmaxf(local_max, row_data[i]);
    const float max_val = hipfire_block_reduce(local_max, wave_scratch, tid, num_waves, true);

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float e = expf(row_data[i] - max_val);
        row_data[i] = e;
        local_sum += e;
    }
    const float sum = hipfire_block_reduce(local_sum, wave_scratch, tid, num_waves, false);

    const float inv_sum = 1.0f / sum;
    for (int i = tid; i < n; i += blockDim.x) row_data[i] *= inv_sum;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n == 0) x[0] = x[0];
#endif
}
