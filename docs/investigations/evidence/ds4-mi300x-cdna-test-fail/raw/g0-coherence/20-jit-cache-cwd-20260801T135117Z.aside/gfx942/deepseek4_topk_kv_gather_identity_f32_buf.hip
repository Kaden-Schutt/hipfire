// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// HIP-graphs-safe twin of deepseek4_topk_kv_gather_identity_f32: reads `K`
// from a device buffer. Identity gather (no top-K indexer): copies the
// first K rows of kv_cache into a transposed column-major slab.
//
// Launch with fixed grid [MAX_K, 1, 1]; blocks beyond K_buf[0] return.
extern "C" __global__ void deepseek4_topk_kv_gather_identity_f32_buf(
    const float* __restrict__ kv_cache,  // [N_compressed, head_dim]
    float*       __restrict__ out,       // [head_dim, out_stride]
    const int*   __restrict__ K_buf,     // [1]
    int head_dim,
    int out_stride
) {
    const int K = K_buf[0];
    const int k = blockIdx.x;
    const int d = threadIdx.x;
    if (k >= K || d >= head_dim) return;
    out[d * out_stride + k] = kv_cache[k * head_dim + d];
}
