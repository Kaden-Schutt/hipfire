// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched Q8_0 GEMM: Y[batch, M] = X[batch, K] @ W[M, K]^T where W is Q8_0
// quantized (34 bytes per group of 32 elements: 2 byte FP16 scale + 32 INT8).
//
// For DFlash speculative decode the same Q8 lm_head runs against B = 15
// (draft positions) or B = 16 (target verify positions) hidden vectors.
// Serial-GEMV reads the weight matrix `batch` times; this kernel reads it
// once and broadcasts each weight load across MAX_BATCH per-row registers.
//
// On the 7900 XTX (gfx1100) this kernel benchmarks roughly equal to the
// serial-GEMV-with-staged-output path on a 4B Q8 lm_head (B=15-16,
// K=2560, M=248K) — the GPU pipelines async kernel launches efficiently
// enough that the per-launch overhead isn't the dominant cost. Kept here
// as a substrate for further tuning (smarter X layout, shared-memory
// weight broadcast) without disrupting the dispatch surface.
//
// Grid:  [M, 1, 1]    — one block per output row.
// Block: [32, 1, 1]   — one wave per row.
//
// FP-reduction order matches gemv_q8_0 (single-accumulator FMA loop) so
// the per-batch slice of the output equals what the per-row GEMV would
// produce — preserves greedy parity for spec decode.

#define MAX_BATCH 64

extern "C" __global__ void gemm_q8_0_batched(
    const unsigned char* __restrict__ A_q8,  // [M, K/32 * 34] packed Q8_0
    const float* __restrict__ X,              // [batch_size, K] row-major
    float* __restrict__ Y,                    // [batch_size, M] row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sums[MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < MAX_BATCH; b++) sums[b] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        float dq = d * (float)qval;
        const int x_off = bi * 32 + tid;

        #pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) {
            if (b < batch_size) {
                sums[b] += dq * X[(size_t)b * K + x_off];
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < MAX_BATCH; b++) {
        if (b < batch_size) {
            float s = sums[b];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down(s, offset);
            if (tid == 0) Y[(size_t)b * M + row] = s;
        }
    }
}
