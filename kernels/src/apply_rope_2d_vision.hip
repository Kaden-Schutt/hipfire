// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// 2D rotary positional embedding for the Qwen3.5-VL vision tower.
//
// HF reference: `apply_rotary_pos_emb_vision` in
// `transformers/models/qwen3_5/modeling_qwen3_5.py`. Rotates Q and K in-place
// in the packed QKV buffer; V is untouched.
//
// Each block handles a (head, token) pair; each thread handles a (d, d + half)
// pair where d ∈ [0, head_dim/2). cos/sin are stored once per (token, d) since
// HF concatenates `(rotary_pos_emb, rotary_pos_emb)` before computing the
// trig table — so the same scalar applies to both halves.
//
//   q[d]      <- q[d]      * c - q[d+half] * s
//   q[d+half] <- q[d+half] * c + q[d]      * s
//
// (matches `rotate_half(x) = cat([-x2, x1])` applied as `q*cos + rh(q)*sin`.)

#include <hip/hip_runtime.h>

// `__launch_bounds__(N, M)` is an occupancy hint: the kernel will be launched
// with at most N threads/block and the compiler may assume M blocks/CU. For
// the current Qwen3.5-VL family head_dim ≤ 72, so head_dim/2 ≤ 36 (under 1
// wave32). Cap at 64 to stay generous for any future variant with head_dim
// up to 128 while still letting the compiler keep register pressure low —
// the body does ~6 fma per thread, so this is mostly documentation.
extern "C" __launch_bounds__(64, 8) __global__ void apply_rope_2d_vision_f32(
    float* __restrict__ qkv,        // [N, 3 * hidden] in-place
    const float* __restrict__ cos_t, // [N, head_dim / 2]
    const float* __restrict__ sin_t, // [N, head_dim / 2]
    int N, int hidden, int num_heads, int head_dim
) {
    int token = blockIdx.y;
    int head  = blockIdx.x;
    int d     = threadIdx.x;
    int half  = head_dim >> 1;
    if (token >= N || head >= num_heads || d >= half) return;

    int stride = 3 * hidden;
    int q_base = token * stride + head * head_dim;
    int k_base = token * stride + hidden + head * head_dim;
    int cs_base = token * half;

    float c = cos_t[cs_base + d];
    float s = sin_t[cs_base + d];

    // Q
    float qa = qkv[q_base + d];
    float qb = qkv[q_base + d + half];
    qkv[q_base + d]        = qa * c - qb * s;
    qkv[q_base + d + half] = qb * c + qa * s;

    // K
    float ka = qkv[k_base + d];
    float kb = qkv[k_base + d + half];
    qkv[k_base + d]        = ka * c - kb * s;
    qkv[k_base + d + half] = kb * c + ka * s;
}
