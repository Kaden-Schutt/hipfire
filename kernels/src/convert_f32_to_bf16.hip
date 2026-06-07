// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Standalone f32 -> bf16 elementwise cast with round-to-nearest-even.
// The output is raw IEEE BF16 bits stored in uint16_t lanes.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void convert_f32_to_bf16(
    const float*    __restrict__ in,
    unsigned short* __restrict__ out,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;

    uint32_t bits = __float_as_uint(in[i]);
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounded = bits + 0x7fffu + lsb;
    out[i] = (unsigned short)(rounded >> 16);
}
