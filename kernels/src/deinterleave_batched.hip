// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched deinterleave Q and Gate from interleaved layout.
//
// Input:  [N × n_heads × head_dim × 2]  (interleaved: Q_h0, Gate_h0, Q_h1, Gate_h1, ...)
// Output: Q    = [N × n_heads × head_dim]
//         Gate = [N × n_heads × head_dim]
//
// Grid: [ceil(n_heads*head_dim / 256), N, 1]
// Block: [256, 1, 1]
//
// Replaces the per-token gather/deinterleave/scatter loop in the batched
// FA prefill path, eliminating 4N kernel launches per FA layer.

extern "C" __launch_bounds__(256)
__global__ void deinterleave_f32_batched(
    const float* __restrict__ interleaved,  // [N × n_heads × head_dim × 2]
    float* __restrict__ out_q,              // [N × n_heads × head_dim]
    float* __restrict__ out_gate,           // [N × n_heads × head_dim]
    int n_heads,
    int head_dim,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * head_dim;
    if (idx >= total) return;
    int b = blockIdx.y;
    if (b >= N) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    // Per-batch offsets into the contiguous [N × ...] tensors.
    const float* inp_b = interleaved + (long long)b * total * 2;
    float* q_b = out_q + (long long)b * total;
    float* g_b = out_gate + (long long)b * total;

    int src_q = h * head_dim * 2 + d;              // Q offset in interleaved
    int src_g = h * head_dim * 2 + head_dim + d;   // Gate offset in interleaved

    q_b[idx] = inp_b[src_q];
    g_b[idx] = inp_b[src_g];
}
