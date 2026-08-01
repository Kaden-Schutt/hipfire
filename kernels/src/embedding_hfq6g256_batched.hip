// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched HFQ6-G256 embedding lookup. Dequantizes `n` rows of the embedding
// table into `output[n, dim]` in a single launch. Grid.x = batch row, threads
// cooperate across `dim` within a block.
//
// Reads token ids from a device buffer so the launch is hipGraph-captureable:
// the caller updates `token_ids[..]` on the stream between replays, and the
// same graph can embed any batch of `n` tokens.
//
// Group layout / 6-bit unpack is byte-identical to gemv_hfq6g256.hip
// (200 B / G256, 4 weights per 3 bytes).
extern "C" __global__ void embedding_hfq6g256_batched(
    const unsigned char* __restrict__ table,    // [vocab × bytes_per_row]
    float* __restrict__ output,                 // [n × dim] row-major
    const int* __restrict__ token_ids,          // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 256;
    const int bytes_per_row = groups_per_row * 200;
    const int token_id = token_ids[batch_idx];
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 256;
        int within = elem % 256;
        const unsigned char* gptr = row + group * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = gptr + 8;

        // 4 weights per 3 bytes — bit-exact with gemv_hfq6g256.hip
        int pack = within / 4;
        int slot = within % 4;
        const unsigned char* p = data + pack * 3;
        unsigned char b0 = p[0];
        unsigned char b1 = p[1];
        unsigned char b2 = p[2];
        int q;
        if (slot == 0)      q = b0 & 63;
        else if (slot == 1) q = (b0 >> 6) | ((b1 & 0xF) << 2);
        else if (slot == 2) q = (b1 >> 4) | ((b2 & 3) << 4);
        else                q = b2 >> 2;

        out_row[elem] = scale * (float)q + zero;
    }
}
