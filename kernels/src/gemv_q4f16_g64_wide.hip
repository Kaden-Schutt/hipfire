// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_q4f16_g64_wide(
    const unsigned char* __restrict__ A_q4f16,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    if (row >= M) return;

    const int blocks_per_row = K / 64;
    const unsigned char* row_data = A_q4f16 + (size_t)row * blocks_per_row * 36;

    float sum = 0.0f;

    // Element-strided: thread tid handles elements tid, tid+blockDim.x, ...
    for (int elem = threadIdx.x; elem < K; elem += blockDim.x) {
        int block_idx = elem >> 6;           // elem / 64 (shift since 64 = 2^6)
        int within = elem & 63;              // elem % 64

        const unsigned char* block = row_data + block_idx * 36;
        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);

        // Nibble extraction: elements 0-31 use lower nibble, 32-63 use upper
        int byte_idx = within & 31;          // within < 32 ? within : within - 32
        unsigned char qbyte = block[4 + byte_idx];
        unsigned int nibble = (within < 32) ? (qbyte & 0xF) : (qbyte >> 4);

        _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
        sum += (float)w * x[elem];
    }

    // Shared memory tree reduction (same as F32 GEMV)
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}
