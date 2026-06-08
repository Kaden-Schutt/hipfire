// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Routed batched Q8_0 KV-cache write for independent request sessions.
//
// Existing kv_cache_write_q8_0_batched writes all rows into one quantized KV
// cache. This variant routes each row through a per-session destination pointer
// table:
//
//   dst = dst_ptrs[row_session_indices[b] * ptr_layer_stride + layer_index]
//
// Grid: [n_kv_heads * (head_dim / 32), batch_size, 1]; block [32, 1, 1].
extern "C" __global__ void kv_cache_write_q8_0_routed_batched(
    const uint64_t* __restrict__ dst_ptrs,       // [num_sessions * ptr_layer_stride]
    const float* __restrict__ src,               // [batch_size * kv_dim]
    const int* __restrict__ row_session_indices, // [batch_size]
    const int* __restrict__ positions,           // [batch_size]
    int ptr_layer_stride,
    int layer_index,
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int gid = blockIdx.x;
    const int b = blockIdx.y;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int session_idx = row_session_indices[b];
    unsigned char* dst = reinterpret_cast<unsigned char*>(
        static_cast<uintptr_t>(dst_ptrs[(size_t)session_idx * ptr_layer_stride + layer_index])
    );

    const int pos = positions[b];
    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int kv_dim = n_kv_heads * head_dim;
    const int elem_offset = b * kv_dim + head_idx * head_dim + block_idx * 32 + tid;

    float val = src[elem_offset];
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset));
    }

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    unsigned char* out = dst + (size_t)pos * total_blocks * 34 + gid * 34;
    if (tid == 0) *((_Float16*)(out)) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}
