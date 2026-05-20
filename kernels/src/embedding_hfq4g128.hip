// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_hfq4g128(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 128;
    const int bytes_per_row = groups_per_row * 72;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 128;
        int within = elem % 128;
        const unsigned char* gptr = row + group * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        int byte_idx = within / 2;
        unsigned char byte_val = gptr[8 + byte_idx];
        int nibble = (within % 2 == 0) ? (byte_val & 0xF) : (byte_val >> 4);
        output[elem] = scale * (float)nibble + zero;
    }
}
