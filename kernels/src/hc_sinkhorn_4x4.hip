// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Hyper-Connections Sinkhorn normaliser (DeepSeek V4 Phase 3) — single-warp variant.
//
// Mirrors upstream `hc_split_sinkhorn_kernel` exactly:
//   1. comb = softmax(comb, dim=-1) + eps    (row-wise softmax + eps)
//   2. comb = comb / (comb.sum(-2) + eps)    (column normalize)
//   3. for sinkhorn_iters - 1 more iters:
//        comb = comb / (comb.sum(-1) + eps)  (row normalize)
//        comb = comb / (comb.sum(-2) + eps)  (column normalize)
//
// One lane per matrix cell (16 lanes active, 16 idle in the wave). All
// reductions use `__shfl_xor` within the wave; no LDS needed.
//
// Pre-fix this kernel ran single-threaded on a single block. PMC showed
// OccupancyPercent ≈ 0.5 % (one lane out of ~30 expected per CU). The
// single-warp variant lifts occupancy to ~25 %+ on this 16-element
// problem (limit is the matrix size, not the chip).

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32, 1)
void hc_sinkhorn_4x4(
    float* __restrict__ matrix,  // [16]: row-major 4×4
    float  eps,
    int    iters
) {
    const int tid = threadIdx.x;
    if (blockIdx.x != 0 || tid >= 16) return;

    // tid = r*4 + c, so r = tid >> 2, c = tid & 3.
    float v = matrix[tid];

    // Step 1: row-softmax + eps.
    //   Row partners differ in lane index by XOR with 1 (pair (0,1),(2,3),…)
    //   and XOR with 2 (pair (0,2),(1,3),…). After both shuffles every
    //   active lane has the row-max / row-sum.
    float rm = v;
    rm = fmaxf(rm, __shfl_xor(rm, 1));
    rm = fmaxf(rm, __shfl_xor(rm, 2));
    v = __expf(v - rm);
    float rs = v;
    rs += __shfl_xor(rs, 1);
    rs += __shfl_xor(rs, 2);
    v = v / rs + eps;

    // Step 2: column normalize.
    //   Column partners differ in lane index by XOR with 4 (pairs with same
    //   c) and XOR with 8 (same).
    float cs = v;
    cs += __shfl_xor(cs, 4);
    cs += __shfl_xor(cs, 8);
    cs += eps;
    v = v / cs;

    // Step 3: (iters - 1) more iterations alternating row/column.
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

    matrix[tid] = v;
}
