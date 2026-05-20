// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define BATCH_TILE 8

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq4g256(
    const char* __restrict__ A,
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    // How many batch elements this block handles
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    float acc[BATCH_TILE];
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_k = g * 256 + tid * 8;
        int byte_off = tid * 4;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];

        float w0 = scale * (float)(b0 & 0xF) + zero;
        float w1 = scale * (float)(b0 >> 4)  + zero;
        float w2 = scale * (float)(b1 & 0xF) + zero;
        float w3 = scale * (float)(b1 >> 4)  + zero;
        float w4 = scale * (float)(b2 & 0xF) + zero;
        float w5 = scale * (float)(b2 >> 4)  + zero;
        float w6 = scale * (float)(b3 & 0xF) + zero;
        float w7 = scale * (float)(b3 >> 4)  + zero;

        for (int b = 0; b < local_bs; b++) {
            const float* xb = x + (batch_start + b) * K;
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
        if (tid == 0) y[(batch_start + b) * M + row] = val;
    }
}
