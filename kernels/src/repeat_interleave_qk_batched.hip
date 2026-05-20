// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched repeat-interleave Q and K key heads up to value heads count.
//
// For each of N batch elements:
//   q_dst[b, (kh*ratio+r), d] = q_src[b, kh, d]
//   k_dst[b, (kh*ratio+r), d] = k_src[b, kh, d]
//
// Grid: [ceil(n_key_heads * ratio * head_dim / 256), N, 1]
// Block: [256, 1, 1]
//
// Replaces the per-token gather/repeat_interleave/scatter loop in the
// batched LA prefill path for GQA models (n_key_heads < n_value_heads).

extern "C" __launch_bounds__(256)
__global__ void repeat_interleave_qk_f32_batched(
    const float* __restrict__ q_src,  // [N × n_key_heads × head_dim]
    const float* __restrict__ k_src,  // [N × n_key_heads × head_dim]
    float* __restrict__ q_dst,         // [N × n_key_heads × ratio × head_dim]
    float* __restrict__ k_dst,         // [N × n_key_heads × ratio × head_dim]
    int n_key_heads,
    int ratio,
    int head_dim,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = n_key_heads * ratio * head_dim;
    if (idx >= total_out) return;
    int b = blockIdx.y;
    if (b >= N) return;

    int d = idx % head_dim;
    int kh_r = idx / head_dim;     // kh * ratio + r
    int kh = kh_r / ratio;          // source key-head index

    int src_stride = n_key_heads * head_dim;
    int dst_stride = total_out;

    int src_off = (long long)b * src_stride + kh * head_dim + d;
    int dst_off = (long long)b * dst_stride + idx;

    q_dst[dst_off] = q_src[src_off];
    k_dst[dst_off] = k_src[src_off];
}
