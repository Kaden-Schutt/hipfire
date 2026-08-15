// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched Q8_0 embedding lookup. Dequantizes N rows of the Q8 embedding
// table into `output[n, dim]` in a single launch, reading token ids from
// a device buffer so the launch is hipGraph-captureable.
extern "C" __global__ void embedding_q8_batched(
    const unsigned char* __restrict__ table,    // [vocab × bytes_per_row]
    float* __restrict__ output,                 // [n × dim] row-major
    const int* __restrict__ token_ids,          // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const int token_id = token_ids[batch_idx];
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int block_idx = elem / 32;
        int within = elem % 32;
        const unsigned char* block = row + block_idx * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + within];
        out_row[elem] = d * (float)qval;
    }
}
