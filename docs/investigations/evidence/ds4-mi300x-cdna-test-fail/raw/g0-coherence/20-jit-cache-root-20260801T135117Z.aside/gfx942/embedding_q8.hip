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

// Buffer-driven retained-replay adapter for DeepSeek V4. The token id is
// staged by a synchronous four-byte H2D copy before PM4 submission, then this
// first tape dispatch dequantizes the row directly into all HC streams. This
// replaces one external kernel, four D2D copies, and the full-device
// HIP-to-HSA handoff synchronization.
extern "C" __global__ void embedding_q8_buf_broadcast(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    const int* __restrict__ token_id_buf,
    int dim,
    int copies
) {
    const int tid = threadIdx.x;
    const int token_id = token_id_buf[0];
    const int blocks_per_row = dim / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int block_idx = elem / 32;
        int within = elem % 32;
        const unsigned char* block = row + block_idx * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + within];
        float value = d * (float)qval;
        for (int copy = 0; copy < copies; ++copy) {
            output[copy * dim + elem] = value;
        }
    }
}
