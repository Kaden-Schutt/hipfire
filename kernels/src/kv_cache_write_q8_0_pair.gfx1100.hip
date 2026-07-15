// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact paired K/V sibling for single-token decode. The two cache writes are
// independent and use the same Q8_0 arithmetic as kv_cache_write_q8_0; folding
// them into the x-grid only removes the launch boundary between them.
//
// Keep this gfx1100-only body in its own translation unit: compiling it beside
// the portable writer perturbs LLVM codegen for gfx12 even though dispatch is
// architecture-gated.
extern "C" __global__ void kv_cache_write_q8_0_pair(
    unsigned char* __restrict__ k_dst,
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int combined_gid = blockIdx.x;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (combined_gid >= total_blocks * 2) return;

    const bool is_v = combined_gid >= total_blocks;
    const int gid = is_v ? combined_gid - total_blocks : combined_gid;
    unsigned char* dst = is_v ? v_dst : k_dst;
    const float* src = is_v ? v_src : k_src;

    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int elem_offset = head_idx * head_dim + block_idx * 32 + tid;

    float val = src[elem_offset];
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    unsigned char* out = dst + (size_t)pos * total_blocks * 34 + gid * 34;
    if (tid == 0) {
        *((_Float16*)(out)) = (_Float16)scale;
    }
    out[2 + tid] = (unsigned char)(signed char)q;
}
