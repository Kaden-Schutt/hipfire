// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Index-aware MoE down-projection GEMV for Q8_0 quantized expert weights,
// with fused scaled atomicAdd into the residual stream.
//
// For each (row, krank) pair, computes:
//   acc = down_proj[expert_id] @ hidden[krank]
//   atomicAdd(&x_residual[row], topk_weights[krank] * per_expert_scale[expert_id] * acc)
//
// Q8_0 layout: 34 bytes per block of 32 elements (2B f16 scale + 32B int8).
//
// Expert ptrs: [n_exp] array of `const char*` weight-base pointers.
// Grid: (M, K_TOP, 1). Block = 32 threads.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_q8_0_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs,   // [n_exp]
    const int* __restrict__ topk_indices,                   // [k_top]
    const float* __restrict__ topk_weights,                 // [k_top]
    const float* __restrict__ per_expert_scale,             // [n_exp]
    const float* __restrict__ hidden_batch,                 // [k_top × K]
    float* __restrict__ x_residual,                         // [M] — accumulator
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const unsigned char* __restrict__ A =
        reinterpret_cast<const unsigned char*>(expert_ptrs[expert_id]);

    // Each krank's hidden input is at hidden_batch[krank * K ... (krank+1) * K].
    const float* __restrict__ x = hidden_batch + (size_t)krank * K;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + (size_t)oi * 8 * 34;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Remaining blocks (K not multiple of 256).
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + (size_t)bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    // Warp shuffle reduction.
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0) {
        float scale = topk_weights[krank] * per_expert_scale[expert_id];
        atomicAdd(&x_residual[row], scale * sum);
    }
}
