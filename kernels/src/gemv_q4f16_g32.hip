// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q4_F16 group-32 GEMV: y = A * x
// 32 threads (single warp), 1 element per thread per block.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4f16_g32(
    const unsigned char* __restrict__ A_q4f16,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q4f16 + (size_t)row * blocks_per_row * 20;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 20;

        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);

        // Threads 0-15 read lower nibble, 16-31 read upper nibble of same byte
        unsigned char qbyte = block[4 + (tid & 15)];
        unsigned int nibble = (tid < 16) ? (qbyte & 0xF) : (qbyte >> 4);

        _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
        sum += (float)w * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
