// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// BATCH_TILE=32 sibling for large-vocab reference/eval lm_head work.
// This is an experiment in reducing CTA count further than the B16 variant.
// Keep it isolated from the production BATCH_TILE=8 prefill path because the
// larger accumulator array may reduce occupancy or spill on some arches.

#include <hip/hip_runtime.h>

#define BATCH_TILE 32

extern "C" __launch_bounds__(32, 4)
__global__ void gemm_f32_register_tiled_b32(
    const float* __restrict__ A,    // [M, K]
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const float* a_row = A + (size_t)row * K;

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int k = tid; k < K; k += 32) {
        const float w = a_row[k];
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b < local_bs) {
                acc[b] += w * x[(size_t)(batch_start + b) * K + k];
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) continue;
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset);
        }
        if (tid == 0) {
            y[(size_t)(batch_start + b) * M + row] = val;
        }
    }
}
