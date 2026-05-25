// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Split a fused interleaved QKV buffer into three separate Q, K, V
// buffers — used by the dots.ocr vision encoder when feeding the
// non-causal `attention_dflash_f32` attention kernel (which takes
// Q/K/V as separate flat `[N, n_heads * head_dim]` buffers).
//
// Why not just sub_offset into the interleaved buffer? The fused
// layout has row stride `3 * hidden` (Q in cols 0..hidden, K in
// hidden..2*hidden, V in 2*hidden..3*hidden), so each of Q/K/V is
// non-contiguous when viewed as `[n_patches, hidden]`. Slicing
// requires an actual copy with strided source reads.
//
// Layout:
//   qkv : [n_patches, 3 * hidden] row-major (hidden = n_heads * head_dim)
//   q   : [n_patches,     hidden] row-major  — write-only
//   k   : [n_patches,     hidden] row-major  — write-only
//   v   : [n_patches,     hidden] row-major  — write-only
//
// Grid:  [n_patches, ceil(hidden / 256), 1]
// Block: [256, 1, 1]
//
// Each thread (patch, j) copies one element of each of Q/K/V (3
// coalesced reads + 3 coalesced writes per thread).
extern "C" __global__ void qkv_split_interleaved_f32(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int n_patches,
    int hidden
) {
    int patch = blockIdx.x;
    int j     = blockIdx.y * blockDim.x + threadIdx.x;
    if (patch >= n_patches || j >= hidden) return;

    int stride = 3 * hidden;
    int dst_off = patch * hidden + j;
    int src_base = patch * stride + j;

    q[dst_off] = qkv[src_base];
    k[dst_off] = qkv[src_base + hidden];
    v[dst_off] = qkv[src_base + 2 * hidden];
}
