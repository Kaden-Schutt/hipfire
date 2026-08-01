// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

// mfp4+P GEMV — mfp4 (E2M1 4-bit element + FP16 per-row scale + offline FWHT)
// with the per-32-block UE8M0 scale PROMOTED to E4M3 (FP8, non-power-of-2).
//
// Byte layout is BYTE-IDENTICAL to mfp4/MFP4G32 (NO codebook prefix):
//   per-row 16-B header (row_scale_a:f16 @0, ..., flags:u8 @6 = 0x05)
//   then (K/32) blocks × 17 B (E4M3 scale_byte:u8 + 16 B packed E2M1 nibbles).
// Per element:  value = row_scale_a * e4m3_decode(scale_byte) * E2M1_LUT[nibble]
//
// The ONLY differences from gemv_mfp4g32_lloyd.hip:
//   (1) NO 32-B codebook prefix → row_off has no +32.
//   (2) the per-block scale byte is decoded as E4M3 (cvt_e4m3_to_f32) instead
//       of UE8M0 (block_e<<23).
//   (3) the nibble LUT is the hard-coded E2M1 lattice (like gemv_hfp4g32),
//       NOT a per-tensor codebook.
// Structure (136-B group stride, 4-accumulator interleave, tail-by-g%4,
// (acc0+acc1)+(acc2+acc3) finalize) ports gemv_mfp4g32_lloyd.hip 1:1 so the FP
// add ordering matches the mfp4 baseline. x is the FWHT-rotated activation.
//
// E4M3 decode (OCP, UNSIGNED scale, bias 7, 3 mantissa) — bit-identical to the
// CPU `e4m3_scale_decode` in the quantizer:
//   exp = (byte>>3)&0xF ; mant = byte&0x7
//   exp==0          → subnormal: 2^-6 * (mant/8)
//   exp==15,mant==7 → NaN code (encoder never emits) → max finite 448.0
//   else (normal)   → 2^(exp-7) * (1 + mant/8)
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

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mfp4g32_p(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // Hard-coded E2M1 magnitude lattice in LDS (8 codes; sign applied via index).
    // lut[0..7] = {0,.5,1,1.5,2,3,4,6}; lut[8..15] = negatives. Matches
    // E2M1_LUT in the quantizer and e2m1_to_f32.
    __shared__ float lut[16];
    if (tid < 16) {
        const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        const int m = tid & 0x7;
        lut[tid] = (tid & 0x8) ? -mag[m] : mag[m];
    }
    __syncthreads();

    const int groups_per_row = K / 256;            // each group = 8 mfp4 blocks
    const int row_block_bytes = (K / 32) * 17;     // = groups_per_row * 136
    // mfp4+P: NO prefix; row 0 starts at byte 0.
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;                // first block of this row

    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale_f = (float)row_scale_h;

    const int block_in_iter = tid >> 2;            // 0..7
    const int nib_off = (tid & 3) * 4;             // 0/4/8/12
    const int boff = nib_off;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    const int thread_block_off = block_in_iter * 17;

    #define MFPP_DOG(pk, sc, b, a) \
        (a) += (sc * lut[(pk)        & 0xFu]) * x[(b)    ] \
             + (sc * lut[((pk) >>  4) & 0xFu]) * x[(b) + 1] \
             + (sc * lut[((pk) >>  8) & 0xFu]) * x[(b) + 2] \
             + (sc * lut[((pk) >> 12) & 0xFu]) * x[(b) + 3] \
             + (sc * lut[((pk) >> 16) & 0xFu]) * x[(b) + 4] \
             + (sc * lut[((pk) >> 20) & 0xFu]) * x[(b) + 5] \
             + (sc * lut[((pk) >> 24) & 0xFu]) * x[(b) + 6] \
             + (sc * lut[((pk) >> 28) & 0xFu]) * x[(b) + 7]

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const unsigned char e0 = *(const unsigned char*)(gp0);
        const unsigned char e1 = *(const unsigned char*)(gp1);
        const unsigned char e2 = *(const unsigned char*)(gp2);
        const unsigned char e3 = *(const unsigned char*)(gp3);

        const float sc0 = row_scale_f * cvt_e4m3_scale_to_f32(e0);
        const float sc1 = row_scale_f * cvt_e4m3_scale_to_f32(e1);
        const float sc2 = row_scale_f * cvt_e4m3_scale_to_f32(e2);
        const float sc3 = row_scale_f * cvt_e4m3_scale_to_f32(e3);

        const unsigned int pk0 = *(const unsigned int*)(gp0 + 1 + boff);
        const unsigned int pk1 = *(const unsigned int*)(gp1 + 1 + boff);
        const unsigned int pk2 = *(const unsigned int*)(gp2 + 1 + boff);
        const unsigned int pk3 = *(const unsigned int*)(gp3 + 1 + boff);

        const int base = g * 256 + tid * 8;        // (tid>>2)*32 + (tid&3)*8 == tid*8

        MFPP_DOG(pk0, sc0, base,         acc0);
        MFPP_DOG(pk1, sc1, base + 256,   acc1);
        MFPP_DOG(pk2, sc2, base + 512,   acc2);
        MFPP_DOG(pk3, sc3, base + 768,   acc3);
    }
    #undef MFPP_DOG

    #define MFPP_TAIL_DOG(pk, sc, b, a) \
        (a) += (sc * lut[(pk)        & 0xFu]) * x[(b)    ] \
             + (sc * lut[((pk) >>  4) & 0xFu]) * x[(b) + 1] \
             + (sc * lut[((pk) >>  8) & 0xFu]) * x[(b) + 2] \
             + (sc * lut[((pk) >> 12) & 0xFu]) * x[(b) + 3] \
             + (sc * lut[((pk) >> 16) & 0xFu]) * x[(b) + 4] \
             + (sc * lut[((pk) >> 20) & 0xFu]) * x[(b) + 5] \
             + (sc * lut[((pk) >> 24) & 0xFu]) * x[(b) + 6] \
             + (sc * lut[((pk) >> 28) & 0xFu]) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * cvt_e4m3_scale_to_f32(e);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8;
        MFPP_TAIL_DOG(pk, sc, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * cvt_e4m3_scale_to_f32(e);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8;
        MFPP_TAIL_DOG(pk, sc, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float sc = row_scale_f * cvt_e4m3_scale_to_f32(e);
        const unsigned int pk = *(const unsigned int*)(gp + 1 + boff);
        const int base = g * 256 + tid * 8;
        MFPP_TAIL_DOG(pk, sc, base, acc2);
    }
    #undef MFPP_TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
