// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx942 MFP4G32E8SOA dequantize-to-FP16 staging kernel.
//
// Input row layout (qt=35):
//   [16B header][K/32 E4M3 scales, padded to 16B][K/32 * 16B E8 codewords]
// Output: row-major FP16 [M, K] in the already-rotated domain.
//
// One native wave64 expands sixteen 32-value blocks (512 values). This is a
// model-lifetime staging operation for the rocBLAS prefill path, not a decode
// kernel: the expanded tensor is cached after the first call.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float e8_soa_scale_to_f32_dq(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * (float)mant * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f;
    }
    const float pow2 = __builtin_bit_cast(
        float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + (float)mant * 0.125f);
}

__device__ __forceinline__ void e8_soa_decode_dq(
    unsigned int idx,
    float p[8]
) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sum = 0u;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sum += e[i];
    }
    const unsigned int p7 = ((idx >> 28) & 0x7u) << 1;
    e[7] = p7 | ((sum + p7) & 1u);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float centered = (float)((int)e[i] - 7);
        p[i] = coset ? centered + 0.5f : centered;
    }
}

extern "C" __global__ __launch_bounds__(64, 16)
void dequantize_mfp4g32_e8_soa_to_f16_gfx942(
    const char* __restrict__ packed,
    _Float16* __restrict__ expanded,
    int M,
    int K
) {
    const int row = (int)blockIdx.x;
    if (row >= M) {
        return;
    }

    const int n_blocks = K / 32;
    const int block = (int)blockIdx.y * 16 + ((int)threadIdx.x >> 2);
    if (block >= n_blocks) {
        return;
    }
    const int codeword_slot = (int)threadIdx.x & 3;
    const int scale_padded = (n_blocks + 15) & ~15;
    const long long row_bytes =
        16LL + (long long)scale_padded + (long long)n_blocks * 16;
    const char* row_ptr = packed + (long long)row * row_bytes;
    const float row_scale = (float)__builtin_bit_cast(
        _Float16, *(const unsigned short*)row_ptr);
    const unsigned char* scales =
        reinterpret_cast<const unsigned char*>(row_ptr + 16);
    const char* codewords = row_ptr + 16 + scale_padded;

    const float scale = row_scale * e8_soa_scale_to_f32_dq(scales[block]) * 0.88f;
    const unsigned int codeword = *reinterpret_cast<const unsigned int*>(
        codewords + (long long)block * 16 + codeword_slot * 4);
    float values[8];
    e8_soa_decode_dq(codeword, values);

    _Float16* out = expanded + (long long)row * K
        + (long long)block * 32 + codeword_slot * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = (_Float16)(scale * values[i]);
    }
}
