// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched HFQ4-G128 embedding lookup. Layout per group:
// [f32 scale][f32 zero][64 B packed nibbles] = 72 B / 128 elements.
// One block owns one token row; device-resident token ids keep graph capture
// semantics identical to the existing HFQ4-G256 batched embedding kernel.
extern "C" __global__ void embedding_hfq4g128_batched(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    const int* __restrict__ token_ids,
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 128;
    const int bytes_per_row = groups_per_row * 72;
    const int token_id = token_ids[batch_idx];
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        const int group = elem / 128;
        const int within = elem % 128;
        const unsigned char* gptr = row + group * 72;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        const float zero = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char packed = gptr[8 + within / 2];
        const int nibble = (within & 1) == 0 ? (packed & 0xF) : (packed >> 4);
        out_row[elem] = scale * (float)nibble + zero;
    }
}
