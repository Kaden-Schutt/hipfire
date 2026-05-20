// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4wave(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A + (size_t)row * blocks_per_row * 20;

    float sum = 0.0f;

    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 20;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 20;
            _Float16 scale = *(const _Float16*)(block);
            _Float16 mn    = *(const _Float16*)(block + 2);

            // Only threads 0-15 load a byte from memory
            unsigned int byte_val = (tid < 16) ? (unsigned int)block[4 + tid] : 0u;
            // Shuffle: thread t gets the byte that thread (t&15) loaded
            unsigned int shared_byte = __shfl(byte_val, tid & 15);
            // Extract: tid<16 gets low nibble, tid>=16 gets high nibble
            unsigned int nibble = (tid < 16) ? (shared_byte & 0xF) : (shared_byte >> 4);

            _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
            sum += (float)w * xb[sub * 32 + tid];
        }
    }

    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 20;
        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);
        unsigned int byte_val = (tid < 16) ? (unsigned int)block[4 + tid] : 0u;
        unsigned int shared_byte = __shfl(byte_val, tid & 15);
        unsigned int nibble = (tid < 16) ? (shared_byte & 0xF) : (shared_byte >> 4);
        _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
        sum += (float)w * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
