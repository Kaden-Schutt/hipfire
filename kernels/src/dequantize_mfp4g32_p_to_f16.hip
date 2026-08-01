// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

// Dequantize an mfp4+P matrix [M × K] into FP16 [M × K] row-major.
//
// mfp4+P byte layout == mfp4/MFP4G32 (NO codebook prefix):
//   per-row 16-B header (row_scale_a:f16 @0, ..., flags:u8 @6 = 0x05)
//   then (K/32) blocks × 17 B (E4M3 scale_byte:u8 + 16 B packed E2M1 nibbles).
// Per element:  value = row_scale_a * e4m3_decode(scale_byte) * E2M1_LUT[nibble]
//
// Differences from dequantize_mfp4g32_lloyd_to_f16.hip:
//   (1) NO 32-B codebook prefix → row_off has no +32.
//   (2) the scale byte is E4M3 (cvt_e4m3_scale_to_f32) not UE8M0 (block_e<<23).
//   (3) the nibble LUT is the hard-coded E2M1 lattice, not a per-tensor codebook.
//
// Weights are stored FWHT-rotated (flag 0x05); this kernel emits the rotated,
// reconstructed FP16 weight and does NOT rotate. Used as the CDNA3 dequant-to-
// f16 utility (round-trip gate + any rocBLAS/MFMA consumer).
//
// Launch: grid = (M, K/256, 1), block = (32, 1, 1). Each of the 32 threads
// dequants 8 elements (one block of an 8-block, 256-element "group").

// E4M3 (OCP, UNSIGNED scale, bias 7, 3 mantissa) — bit-identical to the CPU
// `e4m3_scale_decode` in the quantizer.
#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32_dq(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32_dq(unsigned char b) {
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

extern "C" __launch_bounds__(32, 16)
__global__ void dequantize_mfp4g32_p_to_f16(
    const char*  __restrict__ A,      // [M × row_bytes], no prefix
    _Float16*    __restrict__ W_f16,  // [M × K] row-major
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // Hard-coded E2M1 magnitude lattice in LDS (sign applied via index).
    __shared__ float lut[16];
    if (tid < 16) {
        const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        const int m = tid & 0x7;
        lut[tid] = (tid & 0x8) ? -mag[m] : mag[m];
    }
    __syncthreads();

    const int groups_per_row = K / 256;            // 8 mfp4 blocks each
    const int row_block_bytes = (K / 32) * 17;     // = groups_per_row * 136
    // mfp4+P: NO prefix before row 0.
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;                // first block of this row

    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale_f = (float)row_scale_h;

    const int block_in_iter = tid >> 2;            // 0..7
    const int nib_off = (tid & 3) * 4;             // 0/4/8/12

    // 136-B group stride; this thread's block at +17*block_in_iter.
    const char* gp = row_ptr + (long long)group * 136 + block_in_iter * 17;
    const unsigned char scale_byte = *(const unsigned char*)(gp);

    const float sc = row_scale_f * cvt_e4m3_scale_to_f32_dq(scale_byte);

    const unsigned int pk = *(const unsigned int*)(gp + 1 + nib_off);

    _Float16* out = W_f16 + (long long)row * K + (long long)group * 256 + tid * 8;

    out[0] = (_Float16)(sc * lut[(pk      ) & 0xFu]);
    out[1] = (_Float16)(sc * lut[(pk >>  4) & 0xFu]);
    out[2] = (_Float16)(sc * lut[(pk >>  8) & 0xFu]);
    out[3] = (_Float16)(sc * lut[(pk >> 12) & 0xFu]);
    out[4] = (_Float16)(sc * lut[(pk >> 16) & 0xFu]);
    out[5] = (_Float16)(sc * lut[(pk >> 20) & 0xFu]);
    out[6] = (_Float16)(sc * lut[(pk >> 24) & 0xFu]);
    out[7] = (_Float16)(sc * lut[(pk >> 28) & 0xFu]);
}
