// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 decode-GEMV, gfx1151-specific (Strix Halo, RDNA3.5 APU).
//
// ONLY for gfx1151. Other archs use gemv_mfp4g32_e8 (generic).
//
// Variant A: sequential decode-accumulate + 2-way acc interleave.
// Goal: reduce VGPR from 94 to ~64-72, improving occupancy 5→7 waves/SIMD32.
// Key change: decode ONE codeword at a time into float p[8] (reuse registers),
// 2-way acc interleave instead of 4-way.
//
// Bit-exact with original gfx1151 kernel (same math, same __launch_bounds__).

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * ((float)mant) * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f; // defensive: NaN code -> max finite (never emitted)
    }
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}
#endif

// Sequential decode: decode ONE E8 codeword into p[8], then dot-product, then discard.
// Bit-exact with e8_decode_index in generic kernel.
__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sl += e[i];
    }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) {
        float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

__launch_bounds__(32, 7)
extern "C" __global__ void gemv_mfp4g32_e8_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;

    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter = tid >> 2;
    const int cw_off = (tid & 3) * 4;
    const int thread_block_off = block_in_iter * 17;

    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail  = groups_per_row & 1;

    for (int p = 0; p < pairs; p++) {
        const int g = p << 1;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;

        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * 0.88f;

        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);

        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;

        float pe[8];
        e8_decode_index(ew0, pe);
        acc0 += ssc0*pe[0]*x[base0  ] + ssc0*pe[1]*x[base0+1]
              + ssc0*pe[2]*x[base0+2] + ssc0*pe[3]*x[base0+3]
              + ssc0*pe[4]*x[base0+4] + ssc0*pe[5]*x[base0+5]
              + ssc0*pe[6]*x[base0+6] + ssc0*pe[7]*x[base0+7];

        e8_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x[base1  ] + ssc1*pe[1]*x[base1+1]
              + ssc1*pe[2]*x[base1+2] + ssc1*pe[3]*x[base1+3]
              + ssc1*pe[4]*x[base1+4] + ssc1*pe[5]*x[base1+5]
              + ssc1*pe[6]*x[base1+6] + ssc1*pe[7]*x[base1+7];
    }

    if (tail) {
        const int g = pairs << 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8];
        e8_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1]
              + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5]
              + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
