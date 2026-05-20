// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void cross_entropy_loss(
    const float* __restrict__ logits,
    const int* __restrict__ target_buf,  // single int: target token id
    float* __restrict__ loss_buf,        // single float: output loss
    int vocab_size
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int target = target_buf[0];

    // Phase 1: find max logit (for numerical stability)
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

    // Phase 2: sum of exp(logit - max)
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads)
        local_sum += expf(logits[i] - max_val);
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum_exp = smem[0];

    // Phase 3: loss = -(logits[target] - max - log(sum_exp))
    if (tid == 0) {
        float log_prob = logits[target] - max_val - logf(sum_exp);
        loss_buf[0] = -log_prob;
    }
}
