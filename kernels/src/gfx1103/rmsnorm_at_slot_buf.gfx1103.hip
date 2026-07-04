// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) in-place RMSNorm at a device-buffer-indexed slot.
// Wave-reduced twin of kernels/src/rmsnorm_at_slot_buf.hip: same ABI and math,
// same single-block launch (head_dim ≤ 512, so ≤ 16 wave32 waves), but reduces
// sum-of-squares with shuffles and only one float per wave in LDS. Keeps the
// Phoenix UMA path off heavy LDS. slot_buf[0] < 0 → no-op.
//
// Grid:  [1, 1, 1]
// Block: [BLOCK, 1, 1]   BLOCK = next-pow2 ≥ head_dim (typically 256/512)

#include <hip/hip_runtime.h>

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
#endif

extern "C" __launch_bounds__(512, 1)
__global__ void rmsnorm_f32_at_slot_buf_gfx1103(
    float* __restrict__ base,           // [max_slots, n]
    const float* __restrict__ weight,   // [n]
    const int*   __restrict__ slot_buf, // [1]
    int n,
    float eps
) {
#if HIPFIRE_GFX1103
    extern __shared__ float wave_sums[];
    const int slot = slot_buf[0];
    if (slot < 0) return;
    float* row = base + (size_t)slot * n;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int num_waves = (blockDim.x + 31) >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = row[i];
        sum_sq += v * v;
    }
    const float wsum = hipfire_wave_sum32(sum_sq);
    if (lane == 0) wave_sums[wave] = wsum;
    __syncthreads();
    if (wave == 0) {
        float total = (lane < num_waves) ? wave_sums[lane] : 0.0f;
        total = hipfire_wave_sum32(total);
        if (lane == 0) wave_sums[0] = total;
    }
    __syncthreads();

    const float rms = rsqrtf(wave_sums[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        row[i] = row[i] * weight[i] * rms;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n == 0) {
        base[0] = weight[0] + eps + (float)slot_buf[0];
    }
#endif
}
