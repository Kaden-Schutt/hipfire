// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Bulk F32 → BF16 conversion with round-to-nearest-even.
//
// Matches host reference `round_to_bf16` in
// `crates/hipfire-arch-deepseek4/src/parent/codec.rs:92-110`
// (which is the `pack_f32_to_bf16_bytes` path in
// `crates/hipfire-arch-deepseek4/src/parent/compressor.rs:1491-1499`).
//
// BF16 is the top 16 bits of F32. RNE on the discarded 16:
//   lsb       = (bits >> 16) & 1
//   round_bit = (bits & 0x8000) >> 15
//   sticky    = (bits & 0x7fff) != 0
//   round up when round_bit==1 && (sticky==1 || lsb==1)
// NaN is canonicalized to quiet BF16 NaN with original sign
// (sign | 0x7fc0), matching the host.

extern "C" __global__ void convert_f32_to_bf16(
    const float* __restrict__ src,
    __bf16* __restrict__ dst,
    int nelems
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < nelems; i += stride) {
        float v = src[i];
        uint32_t bits = __builtin_bit_cast(uint32_t, v);
        uint32_t top;
        // NaN: exponent all-ones and mantissa != 0 (bits & 0x7fffffff) > 0x7f800000
        if ((bits & 0x7fffffffu) > 0x7f800000u) {
            uint32_t sign = (bits >> 16) & 0x8000u;
            top = sign | 0x7fc0u;
        } else {
            uint32_t lsb = (bits >> 16) & 1u;
            uint32_t lower = bits & 0xffffu;
            uint32_t round_bit = (lower >> 15) & 1u;
            uint32_t sticky = (lower & 0x7fffu) != 0u ? 1u : 0u;
            top = bits >> 16;
            if (round_bit == 1u && (sticky == 1u || lsb == 1u)) {
                top = top + 1u;
            }
        }
        // Store BF16 bits via uint16 pun (avoids __bf16 bit_cast issues).
        ((uint16_t*)dst)[i] = (uint16_t)(top & 0xffffu);
    }
}
