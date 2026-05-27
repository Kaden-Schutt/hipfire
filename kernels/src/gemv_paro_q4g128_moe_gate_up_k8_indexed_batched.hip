// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched-over-tokens indexed MoE gate_up GEMV for HFQ4-G128 (ParoQuant).
// Sister of gemv_hfq4g256_moe_gate_up_k8_indexed_batched
// (kernels/src/gemv_hfq4g256_moe_gate_up_indexed_batched.hip) with 72 B/group
// stride instead of 136 B/group.
//
// Shapes:
//   expert_ptrs : [n_exp] device pointer table, shared across tokens.
//   topk_indices: [N × K_TOP]
//   x           : [N × K]                           (Givens-rotated post-RMSNorm)
//   y_gate      : [N × K_TOP × MI]                  (MI = M / 2)
//   y_up        : [N × K_TOP × MI]
//
// Grid: (M, K_TOP, N). Each block computes one output row for one expert
// for one token. Block = 32 threads, one warp per output cell.
//
// Per HFQ4G128 layout (see gemv_hfq4g128.hip + gemv_paro_q4g128_moe_gate_up_indexed.hip):
//   - 128 elements per group
//   - 8-byte header: scale F32 + zero F32
//   - 64-byte payload: 64 packed 4-bit nibbles
//   - 32 threads × 4 elements per thread per group = 128 ✓
//
// The Givens rotation is applied UPSTREAM to x (the caller calls
// givens_rotate_to with the layer's shared gate_up sidecar). This kernel is
// rotation-agnostic — it just reads HFQ4G128 nibbles + the rotated x_row.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    // Per-token slices into routing + input.
    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 128 weights / 32 threads = 4 weights per thread
        int base_idx = g * 128 + tid * 4;
        int byte_off = tid * 2;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];

        float v0 = scale * (float)(b0 & 0xF) + zero;
        float v1 = scale * (float)(b0 >> 4)  + zero;
        float v2 = scale * (float)(b1 & 0xF) + zero;
        float v3 = scale * (float)(b1 >> 4)  + zero;

        acc += v0 * x_row[base_idx]     + v1 * x_row[base_idx + 1]
             + v2 * x_row[base_idx + 2] + v3 * x_row[base_idx + 3];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
