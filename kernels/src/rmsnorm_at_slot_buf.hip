// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// In-place RMSNorm at a device-buffer-indexed slot of a base buffer.
// HIP-graphs-safe twin of the `rmsnorm_f32(kv_cache.sub_offset(...))`
// pattern used during DeepSeek V4 compressor commit. Reads the slot index from
// `slot_buf[0]`; -1 sentinel → no-op.
//
// Grid:  [1, 1, 1]   (single block — head_dim ≤ 512)
// Block: [BLOCK, 1, 1]   BLOCK = next-pow2 ≥ head_dim (typically 256/512)

#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_f32_at_slot_buf(
    float* __restrict__ base,           // [max_slots, n]
    const float* __restrict__ weight,   // [n]
    const int*   __restrict__ slot_buf, // [1]
    int n, float eps
) {
    extern __shared__ float sdata[];
    const int slot = slot_buf[0];
    if (slot < 0) return;
    float* row = base + (size_t)slot * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
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
        row[i] = row[i] * weight[i] * rms;
    }
}
