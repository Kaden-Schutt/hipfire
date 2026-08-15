// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Repeat-interleave Q and K key heads up to value heads count.
// Replaces the per-head memcpy loop in DeltaNet for asymmetric MQA configs.
//
// Input layout (Q and K each):
//   src[kh * head_dim + d]              for kh in [0, n_key_heads), d in [0, head_dim)
// Output layout (Q and K each):
//   dst[(kh * ratio + r) * head_dim + d] = src[kh * head_dim + d]
//     for kh in [0, n_key_heads), r in [0, ratio), d in [0, head_dim)
//
// Each thread writes one element of the output. Block 256 threads, grid
// covers n_key_heads * ratio * head_dim total elements per buffer (Q+K share
// the same indexing).
extern "C" __launch_bounds__(256)
__global__ void repeat_interleave_qk_f32(
    const float* __restrict__ q_src,  // [n_key_heads * head_dim]
    const float* __restrict__ k_src,  // [n_key_heads * head_dim]
    float* __restrict__ q_dst,         // [n_key_heads * ratio * head_dim]
    float* __restrict__ k_dst,         // [n_key_heads * ratio * head_dim]
    int n_key_heads,
    int ratio,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_key_heads * ratio * head_dim;
    if (idx >= total) return;

    // idx = ((kh * ratio + r) * head_dim) + d
    int d = idx % head_dim;
    int kh_r = idx / head_dim;     // kh * ratio + r
    int kh = kh_r / ratio;          // source key-head index
    // (r is implicit — it's just kh_r % ratio, but we don't need it)

    int src_off = kh * head_dim + d;
    q_dst[idx] = q_src[src_off];
    k_dst[idx] = k_src[src_off];
}
