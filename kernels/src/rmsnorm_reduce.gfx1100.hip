// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// First half of the gfx1100 decode split. This is the reduction tree from
// fused_rmsnorm_mq_rotate verbatim: 256 original thread streams, five wave32
// shuffle stages, then the eight wave partials folded by lanes 0..7.

__launch_bounds__(256, 1)
extern "C" __global__ void rmsnorm_reduce_gfx1100(
    const float* __restrict__ x,
    float* __restrict__ rms_out,
    int K,
    float eps
) {
    __shared__ float reduce[8];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    float local_sum = 0.0f;
    for (int i = tid; i < K; i += 256) {
        const float v = x[i];
        local_sum += v * v;
    }

    float warp_sum = local_sum;
    warp_sum += __shfl_down(warp_sum, 16);
    warp_sum += __shfl_down(warp_sum, 8);
    warp_sum += __shfl_down(warp_sum, 4);
    warp_sum += __shfl_down(warp_sum, 2);
    warp_sum += __shfl_down(warp_sum, 1);
    if (lane_id == 0) reduce[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        warp_sum = tid < 8 ? reduce[tid] : 0.0f;
        warp_sum += __shfl_down(warp_sum, 4);
        warp_sum += __shfl_down(warp_sum, 2);
        warp_sum += __shfl_down(warp_sum, 1);
        if (tid == 0) rms_out[0] = rsqrtf(warp_sum / (float)K + eps);
    }
}
