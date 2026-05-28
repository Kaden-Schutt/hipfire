// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 mixed-attention identity gather — BATCHED (Phase B2, 2026-05-18).
//
// Batched twin of `deepseek4_topk_kv_gather_identity_f32`. For ratio=128
// layers that lack an indexer, all compressed slots are attended to —
// no per-batch index variation. So this kernel copies the SAME
// [0..K, :] slice of kv_cache into EVERY batch row's slab.
//
// Grid:  [K, batch_size, 1]
// Block: [head_dim, 1, 1]

#include <hip/hip_runtime.h>

extern "C" __global__ void deepseek4_topk_kv_gather_identity_batched_f32(
    const float* __restrict__ kv_cache,  // [N_compressed, head_dim] shared
    float*       __restrict__ out,       // [B, head_dim, out_stride]
    int K,
    int head_dim,
    int out_stride,
    int batch_size
) {
    const int k = blockIdx.x;
    const int b = blockIdx.y;
    const int d = threadIdx.x;
    if (k >= K || b >= batch_size || d >= head_dim) return;
    out[((size_t)b * head_dim + d) * out_stride + k] = kv_cache[(size_t)k * head_dim + d];
}
