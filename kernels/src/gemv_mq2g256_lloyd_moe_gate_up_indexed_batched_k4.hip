// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// K4-unrolled batched MoE gate_up GEMV for MQ2-Lloyd. Same shape contract
// as `gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched` but with 4
// independent accumulators per thread so ILP can hide codebook-lookup
// + dequant + FMA latency. Mirrors qwen35's HFQ4 K4 unroll pattern.
//
// Shapes (unchanged):
//   expert_ptrs : [n_exp]   device-pointer table, shared across tokens
//   topk_indices: [N × K_TOP]
//   x           : [N × K]                  (FWHT-rotated post-RMSNorm)
//   y_gate      : [N × K_TOP × MI]         (MI = M / 2)
//   y_up        : [N × K_TOP × MI]
//
// Grid: (M, K_TOP, N). Block = 32 threads.
//
// Numerical contract:
// FMA reordering between this (4 acc) and the single-acc variant means
// the result is NOT bit-identical to the older kernel but is within
// FMA-order epsilon.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x_rot,                     // [N × K] FWHT-rotated
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
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
    const float* __restrict__ x_row = x_rot + (long long)bid * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;
    const int xoff = tid * 8;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    auto group_fma = [&] (const char* gptr, int base, float& acc) {
        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb[4] = {
            __half2float(cb_h[0]),
            __half2float(cb_h[1]),
            __half2float(cb_h[2]),
            __half2float(cb_h[3]),
        };
        const unsigned char* data = (const unsigned char*)(gptr + 8);
        unsigned char b0 = data[boff];
        unsigned char b1 = data[boff + 1];
        acc += cb[b0 & 3]        * x_row[base]
            +  cb[(b0 >> 2) & 3] * x_row[base + 1]
            +  cb[(b0 >> 4) & 3] * x_row[base + 2]
            +  cb[b0 >> 6]       * x_row[base + 3]
            +  cb[b1 & 3]        * x_row[base + 4]
            +  cb[(b1 >> 2) & 3] * x_row[base + 5]
            +  cb[(b1 >> 4) & 3] * x_row[base + 6]
            +  cb[b1 >> 6]       * x_row[base + 7];
    };

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        const int base = g * 256 + xoff;
        group_fma(gp0, base,         acc0);
        group_fma(gp1, base + 256,   acc1);
        group_fma(gp2, base + 512,   acc2);
        group_fma(gp3, base + 768,   acc3);
    }
    // Tail groups (1..3 leftover, when groups_per_row is not a multiple of 4).
    const int tail_start = quads << 2;
    for (int g = tail_start; g < tail_start + tail; g++) {
        const char* gp = row_ptr + g * 72;
        const int base = g * 256 + xoff;
        group_fma(gp, base, acc0);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long stride = (long long)K_TOP * mi;
        if (row < mi) {
            y_gate[(long long)bid * stride + krank * mi + row] = acc;
        } else {
            y_up[(long long)bid * stride + krank * mi + (row - mi)] = acc;
        }
    }
}
