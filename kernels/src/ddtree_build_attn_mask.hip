// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stage 3a: on-GPU ddtree attention-mask builder.
//
// Replaces the host `linearize_tree_with_parents` mask block + H2D upload
// with a single GPU kernel that reads `parent_indices` (device-resident
// from the preceding H2D in Stage 3a) and writes `attn_bias` directly into
// the per-cycle scratch buffer that the FA verify kernel reads.
//
// Algorithm (per thread i):
//   Fill row i with -INF, then walk i → parent_indices[i] → ... → -1,
//   setting each visited slot j to 0.0. Exactly mirrors the host
//   `visibility[i][j] = visibility[parent_slot][j] || (j == i)` bottom-up
//   pass in ddtree.rs:346-364 + the row-major flatten in
//   linearize_tree_with_parents:988-994.
//
// Grid: [big_n, 1, 1]. Block: [big_n, 1, 1].
// `big_n` ≤ 1 + max_budget ≤ 61 (default). One thread per row.

#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__ void ddtree_build_attn_mask_f32(
    const int* __restrict__ parent_indices, // [big_n] i32
    float*     __restrict__ attn_bias,      // [big_n * big_n] f32, row-major
    int big_n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= big_n) return;

    float* row = attn_bias + (size_t)i * big_n;

    // Fill entire row with -INF.
    for (int k = 0; k < big_n; ++k) {
        row[k] = -INFINITY;
    }

    // Walk ancestor chain from i upward and mark each visited slot as 0.0.
    // parent_indices[0] = -1 (root has no parent).
    // parent_indices[i] for i>0 is either 0 (direct child of root) or
    // node.parent_index + 1.
    // Walk stops when j < 0.
    int j = i;
    while (j >= 0) {
        row[j] = 0.0f;
        j = parent_indices[j];
    }
}
