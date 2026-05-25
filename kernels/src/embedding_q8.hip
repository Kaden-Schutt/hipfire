// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_q8(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int block_idx = elem / 32;
        int within = elem % 32;
        const unsigned char* block = row + block_idx * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + within];
        output[elem] = d * (float)qval;
    }
}
