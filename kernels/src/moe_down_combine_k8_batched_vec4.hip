// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Four-column specialization of moe_down_combine_k8_batched. Each thread
// owns four adjacent output columns and preserves the scalar kernel's k=0..7
// accumulation order independently for every column. The launcher admits this
// variant only for K_TOP=8 and M%4=0, so all float4 accesses are aligned and
// there is no tail or dynamic loop branch in the hot path.

__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_k8_batched_vec4(
    const float* __restrict__ expert_outputs,
    const float* __restrict__ topk_weights,
    float*       __restrict__ x_residual,
    int M, int K_TOP
) {
    (void)K_TOP;
    const int bid = blockIdx.y;
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (col >= M) return;

    const long long token_base = (long long)bid * 8 * M;
    const int weights_base = bid * 8;
    const long long residual_off = (long long)bid * M + col;

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const float w = topk_weights[weights_base + k];
        const float4 v = *reinterpret_cast<const float4*>(
            expert_outputs + token_base + (long long)k * M + col);
        acc.x += w * v.x;
        acc.y += w * v.y;
        acc.z += w * v.z;
        acc.w += w * v.w;
    }

    float4 residual = *reinterpret_cast<const float4*>(x_residual + residual_off);
    residual.x += acc.x;
    residual.y += acc.y;
    residual.z += acc.z;
    residual.w += acc.w;
    *reinterpret_cast<float4*>(x_residual + residual_off) = residual;
}
