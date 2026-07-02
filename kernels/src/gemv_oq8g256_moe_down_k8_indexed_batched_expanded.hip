// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Atomic-free batched indexed MoE down GEMV for OQ8G256 — OQ8 sibling of
// gemv_oq4g256_moe_down_k8_indexed_batched_expanded. Writes each (token, krank)
// result to expert_outputs[N × K_TOP × M]; reuse moe_down_combine_k8_batched to
// fold into the residual. 260 B blocks [f32 scale | 256 int8], dequant
// w = sc * (int8). Grid (M, K_TOP, N), block = 32. No atomics.

#define OQ8_BLK 260

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq8g256_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * OQ8_BLK;
    const int boff = tid * 8;

    float acc = 0.0f;
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * OQ8_BLK;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const signed char* w8 = (const signed char*)(gp + 4 + boff);
        const int base = g * 256 + boff;
        float gsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            gsum += (float)w8[n] * x[base + n];
        }
        acc += sc * gsum;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const long long out_offset =
            ((long long)bid * K_TOP + krank) * (long long)M + row;
        expert_outputs[out_offset] = acc;
    }
}
