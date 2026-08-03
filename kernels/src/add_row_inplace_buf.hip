// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// `a[i] += rows[(row_idx_buf[0] + idx_bias) * n + i]`
//
// HIP-graphs-safe twin of `add_inplace_f32` for the compressor's APE add.
//
// The host version selects the APE row by computing `pos % ratio` and handing
// `add_inplace_f32` a `sub_offset` view — i.e. a raw device pointer baked at
// call time. Under graph capture that pointer is recorded ONCE, so every
// replay adds the capture-time position's APE row. The score buffer is only
// consumed at a compress event, so the error stays latent for a few steps and
// then corrupts the compressed KV cache: exactly the observed
// graph-vs-direct divergence.
//
// Here the row index is read from a DEVICE buffer, so a captured node picks up
// the current position's value on every replay. `idx_bias` is a per-layer
// constant (the compressor packs `ring_slot = ratio + pos % ratio` for the
// overlap ratio, so the APE row is `ring_slot - ratio`), which is safe to bake.

#include <hip/hip_runtime.h>

extern "C" __global__ void add_row_inplace_f32_buf(
    float*       __restrict__ a,            // [n]
    const float* __restrict__ rows,         // [num_rows, n]
    const int*   __restrict__ row_idx_buf,  // [1] — device-side row selector
    int idx_bias,
    int num_rows,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = row_idx_buf[0] + idx_bias;
    // Clamp rather than read out of bounds. A wrong-but-in-range row is still
    // wrong, but an out-of-range read is undefined; the caller's invariant is
    // 0 <= r < num_rows.
    if (r < 0) r = 0;
    if (r >= num_rows) r = num_rows - 1;
    a[i] += rows[(long long)r * (long long)n + (long long)i];
}
