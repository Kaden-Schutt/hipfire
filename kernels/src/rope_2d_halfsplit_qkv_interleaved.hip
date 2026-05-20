// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 2-D spatial RoPE (halfsplit convention) applied IN-PLACE to the Q and K
// slices of a FUSED interleaved QKV buffer, using PRECOMPUTED per-patch
// cos/sin tables.
//
// Companion to rope_2d_halfsplit_f32 (which takes Q and K as separate
// tight-stride [N, n_heads, head_dim] buffers). The fused-QKV variant
// matches the natural output layout of a single QKV GEMM:
//   qkv[N, 3 * hidden] row-major, with each row laid out as
//     [Q-all-heads (hidden), K-all-heads (hidden), V-all-heads (hidden)]
// which is also the layout `vit_attention_opt` expects, so the encoder
// path is:
//   single QKV GEMM → this in-place RoPE → vit_attention_opt
// without any split/merge copies.
//
// V is left untouched (RoPE applies to Q and K only).
//
// Layout:
//   qkv      : [n_patches, 3 * hidden] row-major, hidden = n_heads * head_dim
//   cos, sin : [n_patches, head_dim] row-major (quarter-repeat per dots.ocr;
//              see kernels/src/rope_2d_halfsplit.hip header for the
//              [hc, wc, hc, wc] layout invariant)
//
// Grid:  [n_patches, n_heads, 1]
// Block: [head_dim / 2, 1, 1]
//
// Each thread (patch, head, d) rotates the pair (d, d + head_dim/2) of
// BOTH Q[patch, head, :] and K[patch, head, :] in-place.
extern "C" __global__ void rope_2d_halfsplit_qkv_interleaved_f32(
    float* __restrict__ qkv,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int n_patches,
    int n_heads,
    int head_dim
) {
    int patch = blockIdx.x;
    int head  = blockIdx.y;
    int d     = threadIdx.x;
    int half  = head_dim / 2;
    if (d >= half) return;
    if (patch >= n_patches || head >= n_heads) return;

    int e = d + half;
    int hidden = n_heads * head_dim;
    int stride = 3 * hidden;  // row stride: Q (hidden) + K (hidden) + V (hidden)

    // Per-pair cos/sin. cos[patch, d] == cos[patch, e] in dots.ocr's
    // quarter-repeat layout, but we read both indices to match the
    // separate-buffer kernel's contract.
    int table_base = patch * head_dim;
    float cd = cos_table[table_base + d];
    float sd = sin_table[table_base + d];
    float ce = cos_table[table_base + e];
    float se = sin_table[table_base + e];

    // Q at offset 0 within the patch row.
    int q_base = patch * stride + head * head_dim;
    {
        float qd = qkv[q_base + d];
        float qe = qkv[q_base + e];
        qkv[q_base + d] = qd * cd - qe * sd;
        qkv[q_base + e] = qe * ce + qd * se;
    }

    // K at offset hidden within the patch row.
    int k_base = patch * stride + hidden + head * head_dim;
    {
        float kd = qkv[k_base + d];
        float ke = qkv[k_base + e];
        qkv[k_base + d] = kd * cd - ke * sd;
        qkv[k_base + e] = ke * ce + kd * se;
    }
}
