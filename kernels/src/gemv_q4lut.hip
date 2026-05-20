// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// LDS: 16 f16 codebook entries per block = 32 bytes. For 8 blocks = 256 bytes.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4lut(
    const unsigned char* __restrict__ A_q4lut,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    // 256 bytes of LDS for 8 codebooks (one per sub-block in the unrolled iteration)
    __shared__ _Float16 codebook[8][16];

    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q4lut + (size_t)row * blocks_per_row * 48;

    float sum = 0.0f;

    // Process 8 blocks (256 elements) per outer iteration
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 48;
        const float* xb = x + oi * 256;

        // Phase 1: Load 8 codebooks into LDS. 32 threads, 128 entries = 4 per thread.
        // Each codebook is 16 entries of f16 = 32 bytes, at offset block*48.
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int flat = tid * 4 + i;        // 0..127
            int blk = flat / 16;           // which of 8 blocks
            int entry = flat % 16;         // which codebook entry
            const _Float16* cb = (const _Float16*)(base + blk * 48);
            codebook[blk][entry] = cb[entry];
        }
        // No __syncthreads needed — single warp, lockstep execution on RDNA

        // Phase 2: Each thread processes 8 elements (one per block).
        // Packing: byte[i] = elem[i](lo nibble) | elem[i+16](hi nibble)
        // Thread tid: tid<16 reads low nibble of byte[tid], tid>=16 reads high nibble of byte[tid-16]
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* quants = base + sub * 48 + 32;
            unsigned char qbyte = quants[tid & 15];
            unsigned int nibble = (tid < 16) ? (qbyte & 0xF) : (qbyte >> 4);
            float w = (float)codebook[sub][nibble];
            sum += w * xb[sub * 32 + tid];
        }
    }

    // Handle remaining blocks
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 48;
        // Load codebook for this single block
        if (tid < 16) {
            codebook[0][tid] = ((const _Float16*)block)[tid];
        }
        const unsigned char* quants = block + 32;
        unsigned char qbyte = quants[tid & 15];
        unsigned int nibble = (tid < 16) ? (qbyte & 0xF) : (qbyte >> 4);
        float w = (float)codebook[0][nibble];
        sum += w * x[bi * 32 + tid];
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
