// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched sibling of gemv_q8_0_wide. It preserves the four-accumulator
// FP32 arithmetic contract while sharing each Q8_0 weight load across a
// small batch. MAX_BATCH=8 bounds the per-lane live accumulator set.

#define MAX_BATCH 8

extern "C" __global__ void gemm_q8_0_batched_wide_exact(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float acc0[MAX_BATCH] = {0.0f};
    float acc1[MAX_BATCH] = {0.0f};
    float acc2[MAX_BATCH] = {0.0f};
    float acc3[MAX_BATCH] = {0.0f};
    const int quads = blocks_per_row >> 2;
    const int tail = blocks_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int bi = q << 2;
        const unsigned char* bp0 = row_data + (bi + 0) * 34;
        const unsigned char* bp1 = row_data + (bi + 1) * 34;
        const unsigned char* bp2 = row_data + (bi + 2) * 34;
        const unsigned char* bp3 = row_data + (bi + 3) * 34;

        const float d0 = (float)*((const _Float16*)bp0);
        const float d1 = (float)*((const _Float16*)bp1);
        const float d2 = (float)*((const _Float16*)bp2);
        const float d3 = (float)*((const _Float16*)bp3);
        const signed char q0 = (signed char)bp0[2 + lane];
        const signed char q1 = (signed char)bp1[2 + lane];
        const signed char q2 = (signed char)bp2[2 + lane];
        const signed char q3 = (signed char)bp3[2 + lane];

        #pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) {
            if (b < batch_size) {
                const float* x = X + (size_t)b * K;
                acc0[b] += d0 * (float)q0 * x[(bi + 0) * 32 + lane];
                acc1[b] += d1 * (float)q1 * x[(bi + 1) * 32 + lane];
                acc2[b] += d2 * (float)q2 * x[(bi + 2) * 32 + lane];
                acc3[b] += d3 * (float)q3 * x[(bi + 3) * 32 + lane];
            }
        }
    }

    for (int t = 0; t < tail; t++) {
        const int bi = (quads << 2) + t;
        const unsigned char* bp = row_data + bi * 34;
        const float d = (float)*((const _Float16*)bp);
        const signed char qv = (signed char)bp[2 + lane];
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) {
            if (b < batch_size) {
                const float contribution =
                    d * (float)qv * X[(size_t)b * K + bi * 32 + lane];
                if (t == 0) acc0[b] += contribution;
                else if (t == 1) acc1[b] += contribution;
                else if (t == 2) acc2[b] += contribution;
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < MAX_BATCH; b++) {
        if (b < batch_size) {
            float sum = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down(sum, offset);
            if (lane == 0) Y[(size_t)b * M + row] = sum;
        }
    }
}
