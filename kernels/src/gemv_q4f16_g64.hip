// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q4_F16 group-64 GEMV: y = A * x
// 32 threads (single warp), warp shuffle reduction, no shared memory.
// FP16 dequant via native _Float16 FMA, FP32 accumulate for precision.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4f16_g64(
    const unsigned char* __restrict__ A_q4f16,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 64;
    const unsigned char* row_data = A_q4f16 + (size_t)row * blocks_per_row * 36;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 36;

        // Load scale and min — broadcast across warp via L0 cache
        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);

        // Each thread reads one byte, extracts both nibbles
        unsigned char qbyte = block[4 + tid];

        // Lower nibble -> element tid within block
        unsigned int nib_lo = qbyte & 0xF;
        _Float16 w0 = (_Float16)((unsigned short)nib_lo) * scale + mn;
        sum += (float)w0 * x[bi * 64 + tid];

        // Upper nibble -> element tid+32 within block
        unsigned int nib_hi = qbyte >> 4;
        _Float16 w1 = (_Float16)((unsigned short)nib_hi) * scale + mn;
        sum += (float)w1 * x[bi * 64 + 32 + tid];
    }

    // Warp shuffle reduction (5 steps for 32 threads)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
