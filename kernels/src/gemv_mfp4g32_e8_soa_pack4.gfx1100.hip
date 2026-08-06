// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1100 four-wave workgroup MFP4-E8 SoA GEMV screen.

#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ float e4m3_scale_to_f32_pack4(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(
        float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + (float)mant * 0.125f);
}

__device__ __forceinline__ void e8_decode_pack4(
    unsigned int idx,
    float p[8]
) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sum_low = 0u;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sum_low += e[i];
    }
    const unsigned int high = (idx >> 28) & 0x7u;
    const unsigned int even_part = high << 1;
    e[7] = even_part | ((sum_low + even_part) & 1u);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

} // namespace

// The incumbent uses one wave32 per 32-thread workgroup. gfx1100 admits at
// most 16 workgroups per CU, which can cap this one-wave shape below the
// hardware wave limit. This screen packs four otherwise-independent incumbent
// waves into one 128-thread workgroup. Each wave keeps the exact row mapping,
// load order, accumulation order, and width-32 reduction.
extern "C" __global__ __launch_bounds__(128)
void gemv_mfp4g32_e8_soa_pack4_gfx1100_candidate(
    const char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int row = (int)blockIdx.x * 4 + (tid >> 5);
    if (row >= rows) return;

    const int blocks = cols / 32;
    const int groups = cols / 256;
    const int scale_bytes = ((blocks + 15) >> 4) << 4;
    const long long row_bytes = 16LL + scale_bytes + (long long)blocks * 16;
    const char* row_base = weights + (long long)row * row_bytes;
    const float row_scale = (float)__builtin_bit_cast(
        _Float16, *(const unsigned short*)row_base);
    const char* scale_base = row_base + 16;
    const char* codeword_base = scale_base + scale_bytes;
    const int block_in_group = lane >> 2;
    const int codeword_slot = lane & 3;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int pairs = groups >> 1;
    const int tail = groups & 1;
    for (int pair = 0; pair < pairs; ++pair) {
        const int group = pair << 1;
        const int block0 = group * 8 + block_in_group;
        const int block1 = block0 + 8;
        const float scale0 = row_scale
            * e4m3_scale_to_f32_pack4(
                *(const unsigned char*)(scale_base + block0))
            * 0.88f;
        const float scale1 = row_scale
            * e4m3_scale_to_f32_pack4(
                *(const unsigned char*)(scale_base + block1))
            * 0.88f;
        const unsigned int codeword0 = *(const unsigned int*)(
            codeword_base + (long long)block0 * 16 + codeword_slot * 4);
        const unsigned int codeword1 = *(const unsigned int*)(
            codeword_base + (long long)block1 * 16 + codeword_slot * 4);
        const int x0 = group * 256 + lane * 8;
        const int x1 = x0 + 256;
        float p[8];
        e8_decode_pack4(codeword0, p);
        acc0 += scale0*p[0]*x[x0  ] + scale0*p[1]*x[x0+1]
              + scale0*p[2]*x[x0+2] + scale0*p[3]*x[x0+3]
              + scale0*p[4]*x[x0+4] + scale0*p[5]*x[x0+5]
              + scale0*p[6]*x[x0+6] + scale0*p[7]*x[x0+7];
        e8_decode_pack4(codeword1, p);
        acc1 += scale1*p[0]*x[x1  ] + scale1*p[1]*x[x1+1]
              + scale1*p[2]*x[x1+2] + scale1*p[3]*x[x1+3]
              + scale1*p[4]*x[x1+4] + scale1*p[5]*x[x1+5]
              + scale1*p[6]*x[x1+6] + scale1*p[7]*x[x1+7];
    }
    if (tail) {
        const int group = pairs << 1;
        const int block = group * 8 + block_in_group;
        const float scale = row_scale
            * e4m3_scale_to_f32_pack4(
                *(const unsigned char*)(scale_base + block))
            * 0.88f;
        const unsigned int codeword = *(const unsigned int*)(
            codeword_base + (long long)block * 16 + codeword_slot * 4);
        const int xb = group * 256 + lane * 8;
        float p[8];
        e8_decode_pack4(codeword, p);
        acc0 += scale*p[0]*x[xb  ] + scale*p[1]*x[xb+1]
              + scale*p[2]*x[xb+2] + scale*p[3]*x[xb+3]
              + scale*p[4]*x[xb+4] + scale*p[5]*x[xb+5]
              + scale*p[6]*x[xb+6] + scale*p[7]*x[xb+7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, 32);
    }
    if (lane == 0) y[row] = acc;
}
