// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void max_prob(
    const float* __restrict__ logits,
    float* __restrict__ result,  // single float output
    int vocab_size
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Find max logit
    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += nthreads)
        local_max = fmaxf(local_max, logits[i]);
    smem[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    // Sum exp(logit - max)
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads)
        local_sum += expf(logits[i] - max_val);
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // max_prob = exp(max - max) / sum = 1.0 / sum
    if (tid == 0) result[0] = 1.0f / smem[0];
}
