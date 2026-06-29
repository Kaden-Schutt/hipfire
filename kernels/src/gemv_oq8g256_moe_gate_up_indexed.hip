// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Index-aware fused MoE gate_up GEMV for OQ8G256 (Opus Quant W8 experts).
// Sibling of gemv_oq4g256_moe_gate_up_k8_indexed for the int8 runtime weight the
// OQ+ magnitude-tiered (OqPlusCompact, top-w8_frac int8 outliers) experts expand
// to at load. Per-group block = 260 B [f32 scale | 256 int8], dequant
// w = sc * (int8) (symmetric). W8A16 at decode: int8 weight x f32 activation.
//
// Grid (M, K_TOP=8, 1), block = 32 threads (8 int8 / lane).

#define OQ8_BLK 260

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_oq8g256_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
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

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * OQ8_BLK;
    const int boff = tid * 8; // 8 int8 / lane

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
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
