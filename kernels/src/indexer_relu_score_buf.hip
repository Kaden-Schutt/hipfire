// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// HIP-graphs-safe twin of indexer_relu_score_f32.
// Reads `N` (number of filled compressed slots) from a device buffer
// instead of an i32 kernarg. Launch with a FIXED grid sized to max_N;
// blocks beyond N_buf[0] write 0.0 (sentinel) so downstream top_k
// (which also reads N from buf) ignores them.
//
// Body math is byte-identical to indexer_relu_score_f32 — only N comes
// from a buffer and out-of-range writes a sentinel score.

#include <hip/hip_runtime.h>

#define MAX_HEADS 64

extern "C" __global__ __launch_bounds__(MAX_HEADS, 4)
void indexer_relu_score_f32_buf(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ weights,
    float*       __restrict__ scores,
    const int*   __restrict__ N_buf,        // [1]
    int H,
    int D
) {
    const int N = N_buf[0];
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    if (n >= N) {
        // Sentinel: write -inf so top_k never selects this slot.
        if (tid == 0) scores[n] = -1e30f;
        return;
    }
    if (tid >= H) return;

    const float* qh = q + tid * D;
    const float* kt = k_cache + n * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += qh[d] * kt[d];
    }
    float v = (dot > 0.0f ? dot : 0.0f) * weights[tid];

    __shared__ float reduce[MAX_HEADS];
    reduce[tid] = v;
    __syncthreads();
    for (int s = MAX_HEADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[n] = reduce[0];
}
