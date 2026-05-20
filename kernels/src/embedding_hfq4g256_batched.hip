// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched HFQ4-G256 embedding lookup. Dequantizes `n` rows of the embedding
// table into `output[n, dim]` in a single launch. Grid.x = batch row, threads
// cooperate across `dim` within a block.
//
// Reads token ids from a device buffer so the launch is hipGraph-captureable:
// the caller updates `token_ids[..]` on the stream between replays, and the
// same graph can embed any batch of `n` tokens.
extern "C" __global__ void embedding_hfq4g256_batched(
    const unsigned char* __restrict__ table,    // [vocab × bytes_per_row]
    float* __restrict__ output,                 // [n × dim] row-major
    const int* __restrict__ token_ids,          // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 256;
    const int bytes_per_row = groups_per_row * 136;
    const int token_id = token_ids[batch_idx];
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 256;
        int within = elem % 256;
        const unsigned char* gptr = row + group * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        int byte_idx = within / 2;
        unsigned char byte_val = gptr[8 + byte_idx];
        int nibble = (within % 2 == 0) ? (byte_val & 0xF) : (byte_val >> 4);
        out_row[elem] = scale * (float)nibble + zero;
    }
}
