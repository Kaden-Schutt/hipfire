// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HC Sinkhorn normaliser — BATCHED variant — single-warp per batch row.
//
// Grid:  [batch_size, 1, 1]
// Block: [32, 1, 1]   (one wave, 16 active lanes for 4×4 matrix)
//
// Per batch row b, this matches the math in hc_sinkhorn_4x4.hip exactly
// (row-softmax + alternating col/row normalisations) and is byte-identical
// to that kernel at batch_size == 1.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32, 1)
void hc_sinkhorn_4x4_batched(
    float* __restrict__ matrix,  // [B, 16] row-major 4×4 per batch
    float  eps,
    int    iters,
    int    batch_size
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;
    if (tid >= 16) return;

    float* mat = matrix + (size_t)b * 16;
    float v = mat[tid];

    // Step 1: row-softmax + eps (row partners differ by XOR with 1 and 2).
    float rm = v;
    rm = fmaxf(rm, __shfl_xor(rm, 1));
    rm = fmaxf(rm, __shfl_xor(rm, 2));
    v = __expf(v - rm);
    float rs = v;
    rs += __shfl_xor(rs, 1);
    rs += __shfl_xor(rs, 2);
    v = v / rs + eps;

    // Step 2: column normalize (col partners differ by XOR with 4 and 8).
    float cs = v;
    cs += __shfl_xor(cs, 4);
    cs += __shfl_xor(cs, 8);
    cs += eps;
    v = v / cs;

    // Step 3: remaining iters of alternating row/col.
    for (int it = 0; it < iters - 1; ++it) {
        rs = v;
        rs += __shfl_xor(rs, 1);
        rs += __shfl_xor(rs, 2);
        rs += eps;
        v = v / rs;

        cs = v;
        cs += __shfl_xor(cs, 4);
        cs += __shfl_xor(cs, 8);
        cs += eps;
        v = v / cs;
    }

    mat[tid] = v;
}
