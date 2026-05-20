// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_q8_0(
    unsigned char* __restrict__ dst,     // quantized cache
    const float* __restrict__ src,       // [kv_dim] FP32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    // Grid: [n_kv_heads × blocks_per_head, 1, 1]. Block: [32, 1, 1].
    // Each threadblock quantizes one Q8_0 block (32 elements).
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    const int pos = pos_buf[0];

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int elem_offset = head_idx * head_dim + block_idx * 32 + tid;

    float val = src[elem_offset];

    // Warp reduction for max absolute value
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    // Output: pos * total_blocks * 34 + gid * 34
    unsigned char* out = dst + (size_t)pos * total_blocks * 34 + gid * 34;

    // Thread 0 writes the f16 scale
    if (tid == 0) {
        *((_Float16*)(out)) = (_Float16)scale;
    }
    // All threads write their int8 value
    out[2 + tid] = (unsigned char)(signed char)q;
}
