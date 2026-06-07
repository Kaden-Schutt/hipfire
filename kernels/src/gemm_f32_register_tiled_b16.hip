// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// BATCH_TILE=16 sibling of gemm_f32_register_tiled. This is intended for
// large-vocab reference/eval lm_head work where launch geometry dominates:
// it halves the batch-tile grid dimension and reuses each F32 weight row
// across 16 activation rows. It is kept as a separate entrypoint so the
// BATCH_TILE=8 production prefill path can remain unchanged.

#include <hip/hip_runtime.h>

#define BATCH_TILE 16

extern "C" __launch_bounds__(32, 8)
__global__ void gemm_f32_register_tiled_b16(
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
