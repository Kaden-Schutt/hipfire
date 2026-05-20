// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g512(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 512;
    const int row_bytes = groups_per_row * 264;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 264;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 512 weights / 32 threads = 16 weights per thread = 8 bytes
        int base_idx = g * 512 + tid * 16;
        int byte_off = tid * 8;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];
        unsigned char b4 = nibbles[byte_off + 4];
        unsigned char b5 = nibbles[byte_off + 5];
        unsigned char b6 = nibbles[byte_off + 6];
        unsigned char b7 = nibbles[byte_off + 7];

        acc += (scale * (float)(b0 & 0xF) + zero) * x[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x[base_idx + 7]
             + (scale * (float)(b4 & 0xF) + zero) * x[base_idx + 8]
             + (scale * (float)(b4 >> 4)  + zero) * x[base_idx + 9]
             + (scale * (float)(b5 & 0xF) + zero) * x[base_idx + 10]
             + (scale * (float)(b5 >> 4)  + zero) * x[base_idx + 11]
             + (scale * (float)(b6 & 0xF) + zero) * x[base_idx + 12]
             + (scale * (float)(b6 >> 4)  + zero) * x[base_idx + 13]
             + (scale * (float)(b7 & 0xF) + zero) * x[base_idx + 14]
             + (scale * (float)(b7 >> 4)  + zero) * x[base_idx + 15];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
