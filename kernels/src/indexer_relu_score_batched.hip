// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 indexer score — BATCHED variant (Phase B2, 2026-05-18).
//
// Per batch position b, score every compressed slot t against the
// batch row's Q via:
//
//   scores[b, t] = sum_h relu(Q[b, h, :] · K_cache[t, :]) * weights[b, h]
//
// K_cache is shared across batch (single global compressed-K cache).
// Q and weights are per-batch.
//
// Grid:  [N, batch_size, 1]  — one block per (compressed slot, batch)
// Block: [H, 1, 1]            — one thread per head, reduce via LDS
//
// Byte-identical to indexer_relu_score_f32 at batch_size == 1.

#include <hip/hip_runtime.h>

#define MAX_HEADS 64

extern "C" __global__ __launch_bounds__(MAX_HEADS, 4)
void indexer_relu_score_batched_f32(
    const float* __restrict__ q,         // [B, H, D]
    const float* __restrict__ k_cache,   // [N_max, D] shared
    const float* __restrict__ weights,   // [B, H]
    const int*   __restrict__ n_per_batch, // [B] per-batch valid prefix
    float*       __restrict__ scores,    // [B, N_max]
    int H,
    int D,
    int N_max,
    int batch_size
) {
    const int n = blockIdx.x;
    const int b = blockIdx.y;
    if (n >= N_max || b >= batch_size) return;
    const int tid = threadIdx.x;
    if (tid >= H) return;

    // Per-batch causal cutoff: cache slots ≥ n_per_batch[b] are commits
    // made by LATER batch positions within this chunk and must not be
    // visible to batch row b. Write -inf so top-K skips them.
    const int n_valid_b = n_per_batch[b];
    if (n >= n_valid_b) {
        if (tid == 0) scores[(size_t)b * N_max + n] = -1e30f;
        return;
    }

    const float* qh = q + ((size_t)b * H + tid) * D;
    const float* kt = k_cache + (size_t)n * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += qh[d] * kt[d];
    }
    float v = (dot > 0.0f ? dot : 0.0f) * weights[(size_t)b * H + tid];

    __shared__ float reduce[MAX_HEADS];
    reduce[tid] = v;
    __syncthreads();
    for (int s = MAX_HEADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[(size_t)b * N_max + n] = reduce[0];
}
