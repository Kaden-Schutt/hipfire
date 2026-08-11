// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched MQ6G256 GEMM for pre-rotated activations (FWHT already applied).
// Inner loop is identical to plain HFQ6-G256 — zero rotation overhead.
//
// Packing: 4 weights per 3 bytes; 256 weights per group; per-group header
// [f32 scale][f32 zero] (8 bytes) + 192 bytes packed data = 200 bytes/group.
// 256 weights / 32 threads = 8 weights = 6 bytes per thread per group.
//
// Signature matches the MQ4 analogue `gemm_hfq4g256_batched_lmhead`:
//   a_raw : [M, K] HFQ6/MQ6 packed (M * (K/256) * 200 bytes)
//   x_rot : [batch_size, K] f32 row-major (already FWHT-rotated)
//   y     : [batch_size, M] f32 row-major output (y = A * x)
//   M, K, batch_size
//
// Grid: (M rows) x (ceil(batch_size / 8) batch-tiles) with BATCH_TILE=8.
// Each block handles 1 row x up-to-8 batch elements, reusing the dequantized
// weights across the batch dimension (same amortization as gemm_hfq4g256).
// No LDS, no WMMA, scalar FP32 FMA — correctness-first baseline.

#define BATCH_TILE 8

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_mq6g256(
    const char* __restrict__ A,
    const float* __restrict__ x,    // [batch_size, K] pre-rotated
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    float acc[BATCH_TILE];
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        int base_k = g * 256 + tid * 8;
        int byte_off = tid * 6;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];
        unsigned char b3 = data[byte_off + 3];
        unsigned char b4 = data[byte_off + 4];
        unsigned char b5 = data[byte_off + 5];

        // 4 weights from first 3 bytes
        int q0 = b0 & 63;
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        int q3 = b2 >> 2;
        // 4 weights from next 3 bytes
        int q4 = b3 & 63;
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        int q7 = b5 >> 2;

        float w0 = scale * (float)q0 + zero;
        float w1 = scale * (float)q1 + zero;
        float w2 = scale * (float)q2 + zero;
        float w3 = scale * (float)q3 + zero;
        float w4 = scale * (float)q4 + zero;
        float w5 = scale * (float)q5 + zero;
        float w6 = scale * (float)q6 + zero;
        float w7 = scale * (float)q7 + zero;

        for (int b = 0; b < local_bs; b++) {
            const float* xb = x + (long long)(batch_start + b) * K;
            acc[b] += w0 * xb[base_k]     + w1 * xb[base_k + 1]
                    + w2 * xb[base_k + 2] + w3 * xb[base_k + 3]
                    + w4 * xb[base_k + 4] + w5 * xb[base_k + 5]
                    + w6 * xb[base_k + 6] + w7 * xb[base_k + 7];
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) y[(long long)(batch_start + b) * M + row] = val;
    }
}
