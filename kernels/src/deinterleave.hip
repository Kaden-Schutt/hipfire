// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

/// Deinterleave Q and Gate from interleaved layout.
/// Input:  [Q_h0(hd), Gate_h0(hd), Q_h1(hd), Gate_h1(hd), ...]
/// Output: Q = [Q_h0(hd), Q_h1(hd), ...], Gate = [Gate_h0(hd), Gate_h1(hd), ...]
extern "C" __launch_bounds__(256)
__global__ void deinterleave_f32(
    const float* __restrict__ interleaved,  // [n_heads * head_dim * 2]
    float* __restrict__ out_a,              // [n_heads * head_dim] (Q)
    float* __restrict__ out_b,              // [n_heads * head_dim] (Gate)
    int n_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * head_dim;
    if (idx >= total) return;

    int h = idx / head_dim;
    int d = idx % head_dim;
    int src_a = h * head_dim * 2 + d;         // Q offset in interleaved
    int src_b = h * head_dim * 2 + head_dim + d;  // Gate offset in interleaved

    out_a[idx] = interleaved[src_a];
    out_b[idx] = interleaved[src_b];
}
