// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HFQ6G256 counterpart of `gemv_hfq4g256_moe_gate_up_k8_indexed_batched`
// (kernels/src/gemv_hfq4g256_moe_gate_up_indexed_batched.hip).
//
// Batched N-token indexed MoE gate_up GEMV. Identical kernarg signature,
// expert indexing (expert_ptrs[topk_indices[bid*K_TOP + krank]]), grid
// mapping (M, K_TOP, N), group-size-256 loop, gate/up output split
// (M = 2*MI: first half → y_gate, second half → y_up), batch handling,
// and wave32 reduction. The ONLY difference is the per-group weight
// dequant: 4-bit (136 B/group) → 6-bit (200 B/group).
//
// 6-bit dequant copied verbatim from kernels/src/gemv_hfq6g256.hip
// (lines 24-60: 4 B fp32 scale + 4 B fp32 zero + 192 B of 6-bit packed
// quants for 256 weights; 8 weights/thread = 6 bytes/thread). 256*6/8 = 192,
// + 8 B header = 200 B/group.
//
// Note: weights post-PR-199 are typically MQ6G256 (FWHT-rotated). HFQ6G256
// (non-rotated) shares the same byte layout — the kernel is rotation-
// agnostic, the caller decides whether to pass FWHT(x) or x_norm via the
// `x` parameter (same contract as the HFQ4 sibling).

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
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

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread
        // 4 weights per 3 bytes → 8 weights = 6 bytes per thread
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 6;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];
        unsigned char b3 = data[byte_off + 3];
        unsigned char b4 = data[byte_off + 4];
        unsigned char b5 = data[byte_off + 5];

        // 4 weights from first 3 bytes
        int q0 = b0 & 63;
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        int q3 = b2 >> 2;
        // 4 weights from next 3 bytes
        int q4 = b3 & 63;
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        int q7 = b5 >> 2;

        acc += (scale * (float)q0 + zero) * x_row[base_idx]
             + (scale * (float)q1 + zero) * x_row[base_idx + 1]
             + (scale * (float)q2 + zero) * x_row[base_idx + 2]
             + (scale * (float)q3 + zero) * x_row[base_idx + 3]
             + (scale * (float)q4 + zero) * x_row[base_idx + 4]
             + (scale * (float)q5 + zero) * x_row[base_idx + 5]
             + (scale * (float)q6 + zero) * x_row[base_idx + 6]
             + (scale * (float)q7 + zero) * x_row[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        // Output layout matches gemv_hfq4g256_moe_gate_up_k8_indexed_batched:
        // M = 2 * MI, first half is gate, second half is up.
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
