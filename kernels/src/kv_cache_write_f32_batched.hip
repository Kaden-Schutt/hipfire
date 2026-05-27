// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched F32 KV-cache write: scatter `batch_size` rows of `src` into the
// F32 cache at the absolute positions in `positions`, in one launch.
// Replaces the per-position `kv_cache_write` loop used by batched prefill.
//
//   dst[positions[b] * kv_dim + i] = src[b * kv_dim + i]
//
// Grid: [ceil(kv_dim / block), batch_size, 1]; block [256, 1, 1].
extern "C" __global__ void kv_cache_write_f32_batched(
    float* __restrict__ dst,           // cache: [max_seq * kv_dim]
    const float* __restrict__ src,     // [batch_size * kv_dim]
    const int* __restrict__ positions, // [batch_size]
    int kv_dim,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // element within kv_dim
    int b = blockIdx.y;                             // batch row index
    if (b >= batch_size || i >= kv_dim) return;
    int pos = positions[b];
    dst[(size_t)pos * kv_dim + i] = src[(size_t)b * kv_dim + i];
}
