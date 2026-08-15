// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1201 batched Hyper-Connections row RMS factors.

#include <hip/hip_runtime.h>

#define HC_RMS_THREADS 256

// One block computes the shared normalization factor for one token row.  The
// established scalar control kernel redundantly recomputes this same value in
// every one of the 24 control-row workgroups.
extern "C" __global__ __launch_bounds__(HC_RMS_THREADS)
void hc_inv_rms_batched_gfx1201(
    const float* __restrict__ x,
    float* __restrict__ inv_rms,
    int x_dim,
    int batch_size
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    const float* row = x + (size_t)b * x_dim;
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
    for (int d = tid * 4; d < x_dim; d += HC_RMS_THREADS * 4) {
        const float4 v = *reinterpret_cast<const float4*>(row + d);
        sq0 += v.x * v.x;
        sq1 += v.y * v.y;
        sq2 += v.z * v.z;
        sq3 += v.w * v.w;
    }
    float sum = (sq0 + sq1) + (sq2 + sq3);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float partial[HC_RMS_THREADS / 32];
    if ((tid & 31) == 0) partial[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < HC_RMS_THREADS / 32; ++i) total += partial[i];
        inv_rms[b] = rsqrtf(total / (float)x_dim + 1.0e-6f);
    }
}
