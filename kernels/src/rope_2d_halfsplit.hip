// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 2-D spatial RoPE (halfsplit convention) with PRECOMPUTED per-patch
// cos/sin tables.
//
// Differs from rope_partial_halfsplit_f32 in three ways:
// 1. Takes cos/sin tables of shape [n_patches, head_dim] (one row per
//    patch) instead of a single position counter + freq_base. The
//    host side precomputes the tables for the patch grid and uploads
//    them once per image.
// 2. Operates on N independent positions in a single launch (the
//    "batch" axis = patches), each with its own frequency tables.
//    rope_partial_halfsplit_f32 operates on a single position.
// 3. Rotates the FULL head_dim (no partial rotary) — vision RoPE in
//    dots.ocr doesn't use a partial rotary factor.
//
// Used by hipfire-arch-dots-ocr for the 42-block DotsVisionTransformer.
// See docs/plans/qwen_2.0_vlm_plus_dots_ocr.md §2.6 for the cos/sin
// table layout (quarter-repeat: [hc, wc, hc, wc] within head_dim) and
// crates/hipfire-arch-dots-ocr/src/rope.rs for the host builder.
//
// Halfsplit convention: pairs are (d, d + head_dim/2). For pair (d, e),
// the rotation is:
//     q_out[d] = q[d] * cos[d] - q[e] * sin[d]
//     q_out[e] = q[e] * cos[e] + q[d] * sin[e]
// In the dots.ocr 2-D layout, cos[d] == cos[e] for all d (the table's
// second half is a copy of the first); same for sin. The kernel reads
// both indices to stay generic — kernels with the quarter-repeat
// invariant pay only one extra L1 load per thread.
//
// Layout:
//   q : [n_patches, n_heads_q, head_dim] row-major
//   k : [n_patches, n_heads_k, head_dim] row-major
//   cos, sin : [n_patches, head_dim] row-major
//
// Grid:  [n_patches, max(n_heads_q, n_heads_k), 1]
// Block: [head_dim / 2, 1, 1]
//
// Thread (patch, head, d) rotates the pair (d, d + head_dim/2) of
// Q[patch, head, :] if head < n_heads_q, and of K[patch, head, :] if
// head < n_heads_k. dots.ocr's vision attention has n_heads_q ==
// n_heads_k (no GQA), so the two checks both pass for every thread.
extern "C" __global__ void rope_2d_halfsplit_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int n_patches,
    int n_heads_q,
    int n_heads_k,
    int head_dim
) {
    int patch = blockIdx.x;
    int head  = blockIdx.y;
    int d     = threadIdx.x;
    int half  = head_dim / 2;
    if (d >= half) return;
    if (patch >= n_patches) return;

    int e = d + half;

    // Per-pair cos/sin. cos[patch, d] and cos[patch, e] are equal in
    // the dots.ocr layout (quarter-repeat), but we read both so the
    // kernel works for any "halfsplit with per-position tables" case.
    int table_base = patch * head_dim;
    float cd = cos_table[table_base + d];
    float sd = sin_table[table_base + d];
    float ce = cos_table[table_base + e];
    float se = sin_table[table_base + e];

    if (head < n_heads_q) {
        int q_base = (patch * n_heads_q + head) * head_dim;
        float qd = q[q_base + d];
        float qe = q[q_base + e];
        q[q_base + d] = qd * cd - qe * sd;
        q[q_base + e] = qe * ce + qd * se;
    }
    if (head < n_heads_k) {
        int k_base = (patch * n_heads_k + head) * head_dim;
        float kd = k[k_base + d];
        float ke = k[k_base + e];
        k[k_base + d] = kd * cd - ke * sd;
        k[k_base + e] = ke * ce + kd * se;
    }
}
