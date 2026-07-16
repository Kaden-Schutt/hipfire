// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Compute-queue copy used inside retained MTP tapes. Offsets and length are in
// 32-bit words; every MTP activation copy is naturally aligned.
extern "C" __global__ void copy_u32_range(
    const unsigned int* __restrict__ src,
    unsigned int* __restrict__ dst,
    int src_word_offset,
    int dst_word_offset,
    int words
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < words) {
        dst[dst_word_offset + i] = src[src_word_offset + i];
    }
}
