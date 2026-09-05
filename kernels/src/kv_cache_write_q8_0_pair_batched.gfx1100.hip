// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Paired K/V sibling of kv_cache_write_q8_0_batched (legacy single-arena
// addressing). The two cache writes are independent and use the same Q8_0
// arithmetic as the two calls it replaces; folding them into the x-grid only
// removes the launch boundary between them.
//
// Grid: [2 * total_blocks, batch_size, 1] where
// total_blocks = n_kv_heads * (head_dim / 32). Block: [32, 1, 1].
// The low half of the x-grid writes K, the high half writes V.
//
// Keep this gfx1100-only body in its own translation unit: compiling it
// beside the portable writer perturbs LLVM codegen for gfx12 even though
// dispatch is architecture-gated.
//
// Bit-exactness: per (block, batch row) the element offset, warp-shuffle
// amax reduction, scale/quantize rounding, and destination bytes are
// statement-identical to kv_cache_write_q8_0_batched with null slot
// descriptors (dst + pos * per_pos_bytes + gid * 34).
extern "C" __global__ void kv_cache_write_q8_0_pair_batched_gfx1100(
    unsigned char* __restrict__ k_dst,
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ k_src,   // [batch_size x kv_dim]
    const float* __restrict__ v_src,   // [batch_size x kv_dim]
    const int* __restrict__ positions, // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size)
{
    const int combined_gid = blockIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x; // 0..31

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (combined_gid >= total_blocks * 2) return;

    const bool is_v = combined_gid >= total_blocks;
    const int gid = is_v ? combined_gid - total_blocks : combined_gid;
    unsigned char* dst = is_v ? v_dst : k_dst;
    const float* src = is_v ? v_src : k_src;

    const int pos = positions[bid];
    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int kv_dim = n_kv_heads * head_dim;
    const int elem_offset = bid * kv_dim + head_idx * head_dim + block_idx * 32 + tid;

    float val = src[elem_offset];

    // Warp max absolute value
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    const int per_pos_bytes = total_blocks * 34;
    unsigned char* out = dst + (unsigned long long)pos * (unsigned long long)per_pos_bytes + gid * 34;
    if (tid == 0) *((_Float16*)(out)) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}
