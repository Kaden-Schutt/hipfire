// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Zero inactive rows of a 2D F32 tensor.
// Tensor is row-major [rows, cols] flattened as data[row*cols + col].
// `active_mask` lane bit i selects row i as active. Only inactive rows are
// written, with exact +0.0f (no -0.0, no contamination of active rows).
// Full-mask callers skip this dispatch entirely.
extern "C" __global__ void zero_inactive_rows_f32(
    float* __restrict__ data,
    int rows,
    int cols,
    unsigned long long active_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    if (((active_mask >> row) & 1ULL) != 0) return;
    data[idx] = 0.0f;
}
