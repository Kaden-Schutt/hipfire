// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — MFP4-E8 SoA decode GEMV, gfx942 wave64 lowering.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float cvt_e4m3_scale_to_f32_soa_gfx942(
    unsigned char b
) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(
        float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index_soa_gfx942(
    unsigned int idx,
    float p[8]
) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; ++i) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sl += e[i];
    }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; ++i) {
        const float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

// One native wave64 computes two independent rows. The width=32 shuffle
// keeps the two half-wave reductions isolated while doubling the row-level
// work available to each scheduled workgroup.
extern "C" __global__ __launch_bounds__(64)
void gemv_mfp4g32_e8_soa_gfx942(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = (int)blockIdx.x * 2 + (tid >> 5);
    if (row >= M) return;

    const int n_blocks = K / 32;
    const int groups_per_row = K / 256;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total =
        16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = A + (long long)row * row_total;
    const float row_scale = (float)__builtin_bit_cast(
        _Float16, *(const unsigned short*)hdr);
    const char* scale_base = hdr + 16;
    const char* cw_base = scale_base + scale_padded;

    const int block_in_group = lane >> 2;
    const int cw_slot = lane & 3;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail = groups_per_row & 1;

    for (int p = 0; p < pairs; ++p) {
        const int g = p << 1;
        const int b0 = g * 8 + block_in_group;
        const int b1 = b0 + 8;
        const float s0 = row_scale
            * cvt_e4m3_scale_to_f32_soa_gfx942(
                *(const unsigned char*)(scale_base + b0))
            * 0.88f;
        const float s1 = row_scale
            * cvt_e4m3_scale_to_f32_soa_gfx942(
                *(const unsigned char*)(scale_base + b1))
            * 0.88f;
        const unsigned int ew0 = *(const unsigned int*)(
            cw_base + (long long)b0 * 16 + cw_slot * 4);
        const unsigned int ew1 = *(const unsigned int*)(
            cw_base + (long long)b1 * 16 + cw_slot * 4);
        const int base0 = g * 256 + lane * 8;
        const int base1 = base0 + 256;

        float pe[8];
        e8_decode_index_soa_gfx942(ew0, pe);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0 += s0 * pe[i] * x[base0 + i];
        }
        e8_decode_index_soa_gfx942(ew1, pe);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc1 += s1 * pe[i] * x[base1 + i];
        }
    }

    if (tail) {
        const int g = pairs << 1;
        const int b = g * 8 + block_in_group;
        const float scale = row_scale
            * cvt_e4m3_scale_to_f32_soa_gfx942(
                *(const unsigned char*)(scale_base + b))
            * 0.88f;
        const unsigned int ew = *(const unsigned int*)(
            cw_base + (long long)b * 16 + cw_slot * 4);
        const int base = g * 256 + lane * 8;
        float pe[8];
        e8_decode_index_soa_gfx942(ew, pe);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0 += scale * pe[i] * x[base + i];
        }
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, 32);
    }
    if (lane == 0) y[row] = acc;
}
