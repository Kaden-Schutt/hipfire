// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 Compressor overlap-transform concat (for overlap=true / ratio=4).
//
// Builds the [2*ratio, head_dim] compressed view from a [2*ratio,
// 2*head_dim] kv_state buffer:
//   For rows 0..ratio (old window): take cols 0..head_dim (first half).
//   For rows ratio..2*ratio (current window): take cols head_dim..2*head_dim
//     (second half).
//
// Equivalent to upstream `cat([kv_state[:ratio, :d], kv_state[ratio:, d:]], dim=1)`.
//
// The same kernel reads both kv_state and score_state into separate
// concat outputs (input/output ptrs differ; logic identical).
//
// Grid: 1 block per row in [0, 2*ratio). Each thread handles one
// head_dim element. coff*ratio is small (8 for ratio=4 overlap).

#include <hip/hip_runtime.h>

extern "C" __global__ void compressor_overlap_concat_f32(
    const float* __restrict__ src,     // [2*ratio, 2*head_dim]
    float*       __restrict__ dst,     // [2*ratio, head_dim]
    int ratio,
    int head_dim
) {
    const int r = blockIdx.x;
    const int d = threadIdx.x;
    if (r >= 2 * ratio || d >= head_dim) return;

    // Pick which half-of-cols based on row:
    //   r < ratio  → cols [0..head_dim) of src
    //   r >= ratio → cols [head_dim..2*head_dim) of src
    int src_col_off = (r < ratio) ? 0 : head_dim;
    int src_idx = r * (2 * head_dim) + src_col_off + d;
    int dst_idx = r * head_dim + d;
    dst[dst_idx] = src[src_idx];
}
