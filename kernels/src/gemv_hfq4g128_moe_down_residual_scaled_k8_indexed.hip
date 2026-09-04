// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Device-routed MoE down projection for HFQ4-G128 expert weights.  Expert
// selection and mixture weights remain device-resident so the launch sequence
// and kernargs are stable across generated tokens.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ per_expert_scale,
    const float* __restrict__ hidden_batch,
    float* __restrict__ x_residual,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = (K + 127) / 128;
    float mixed = 0.0f;
    // Fixed rank order makes HIP, kernarg-blob and retained PM4 bit-exact.
    #pragma unroll
    for (int krank = 0; krank < 8; ++krank) {
        const int expert_id = topk_indices[krank];
        const char* __restrict__ A =
            reinterpret_cast<const char*>(expert_ptrs[expert_id]);
        const float* __restrict__ x = hidden_batch + (size_t)krank * K;
        const char* row_ptr = A + (long long)row * groups_per_row * 72;
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
            const char* gptr = row_ptr + g * 72;
            const float scale = __builtin_bit_cast(float, *(const unsigned int*)gptr);
            const float zero = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
            const unsigned char* nibbles = (const unsigned char*)(gptr + 8);
            const int base = g * 128 + tid * 4;
            const int byte_off = tid * 2;
            const unsigned char b0 = nibbles[byte_off];
            const unsigned char b1 = nibbles[byte_off + 1];
            const float v0 = scale * (float)(b0 & 0xf) + zero;
            const float v1 = scale * (float)(b0 >> 4) + zero;
            const float v2 = scale * (float)(b1 & 0xf) + zero;
            const float v3 = scale * (float)(b1 >> 4) + zero;
            if (base < K) acc += v0 * x[base];
            if (base + 1 < K) acc += v1 * x[base + 1];
            if (base + 2 < K) acc += v2 * x[base + 2];
            if (base + 3 < K) acc += v3 * x[base + 3];
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0)
            mixed += topk_weights[krank] * per_expert_scale[expert_id] * acc;
    }
    if (tid == 0) x_residual[row] = mixed;
}
