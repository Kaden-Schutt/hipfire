// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Routed batched F32 KV-cache write for independent request sessions.
//
// Existing kv_cache_write_f32_batched writes all rows into one cache:
//
//   dst[positions[b] * kv_dim + i] = src[b * kv_dim + i]
//
// This variant routes each row through a per-session destination pointer table:
//
//   dst = dst_ptrs[row_session_indices[b] * ptr_layer_stride + layer_index]
//   dst[positions[b] * kv_dim + i] = src[b * kv_dim + i]
//
// Grid: [ceil(kv_dim / block), batch_size, 1]; block [256, 1, 1].
extern "C" __global__ void kv_cache_write_f32_routed_batched(
    const uint64_t* __restrict__ dst_ptrs,       // [num_sessions * ptr_layer_stride] device ptrs
    const float* __restrict__ src,               // [batch_size * kv_dim]
    const int* __restrict__ row_session_indices, // [batch_size]
    const int* __restrict__ positions,           // [batch_size]
    int ptr_layer_stride,
    int layer_index,
    int kv_dim,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // element within kv_dim
    int b = blockIdx.y;                             // routed batch row index
    if (b >= batch_size || i >= kv_dim) return;

    int session_idx = row_session_indices[b];
    float* dst = reinterpret_cast<float*>(
        static_cast<uintptr_t>(dst_ptrs[(size_t)session_idx * ptr_layer_stride + layer_index])
    );
    int pos = positions[b];
    dst[(size_t)pos * kv_dim + i] = src[(size_t)b * kv_dim + i];
}
