// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Index-aware fused MoE gate_up GEMV for Q8_0 quantized expert weights.
// Reads expert IDs from a device-side topk_indices buffer and the per-expert
// weight base from a device-side expert pointer table. No CPU kernargs depend
// on the routing result, so the whole kernel replays cleanly under hipGraph.
//
// Q8_0 layout: 34 bytes per block of 32 elements (2B f16 scale + 32B int8).
//
// Expert ptrs: [n_exp] array of `const char*` weight-base pointers
// (packed as `unsigned long long`, 8B each).
//
// Output: y_gate[K_TOP × MI] and y_up[K_TOP × MI], feeding batched
// gelu_tanh + mul.
//
// Grid: (M, K_TOP, 1). Block = 32 threads. K_TOP is hardcoded to 8.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_q8_0_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    // Read expert_id from device and index into the pointer table.
    const int expert_id = topk_indices[krank];
    const unsigned char* __restrict__ A =
        reinterpret_cast<const unsigned char*>(expert_ptrs[expert_id]);

    // Q8_0: 34 bytes per block of 32 elements.
    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    // Process 8 Q8_0 blocks (256 elements) per outer iteration.
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

    // Handle remaining blocks (if K is not multiple of 256).
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
        // Split gate_up output: first MI rows → y_gate, second MI → y_up.
        // krank determines which expert's contribution, row < MI → gate,
        // row >= MI → up.
        int MI = M / 2;
        if (row < MI) {
            y_gate[krank * MI + row] = sum;
        } else {
            y_up[krank * MI + (row - MI)] = sum;
        }
    }
}
