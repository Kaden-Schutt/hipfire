// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 SoA GEMV, DECODE-STRIPPED probe (bench only).
//
// Identical to gemv_mfp4g32_e8_soa EXCEPT the E8-lattice decode is replaced by a
// trivial bounded nibble extraction (keeps the codeword load live, drops the
// parity + 8 int->float conversions). Output is garbage; this measures the
// "loads + scale + reduction" ceiling to isolate the lattice-decode ALU cost in
// the cache-resident regime. If GB/s jumps far above the scalar-decode kernel,
// the decode is the wall and a cheaper decode (LUT/vec) is the lever.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 native OCP FP8 (radiowave Fp8Format::E4M3Ocp). Bit-identical to the
// scalar idiom over codes 1..=126 — the entire range e4m3_encode_roundup can
// emit; the full-domain mismatches are the NaN slot 0x7F and the negative
// half, neither reachable for a SCALE. See
// benchmarks/kernel-probes/e4m3-native-decode/.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32_strip(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return (float)v;
}
#else
__device__ __forceinline__ float cvt_e4m3_scale_to_f32_strip(unsigned char b) {
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

// STRIP: raw nibble extraction only, bounded. Keeps `idx` fully load-live but
// drops the parity/coset/sub-7 lattice ALU.
__device__ __forceinline__ void e8_decode_strip(unsigned int idx, float p[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) p[i] = (float)((idx >> (unsigned)(4 * i)) & 0xFu) * 0.0625f;
}

__launch_bounds__(32)
extern "C" __global__ void gemv_mfp4g32_e8_soa_strip(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int n_blocks       = K / 32;
    const int groups_per_row = K / 256;

    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = A + (long long)row * row_total;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const char* scale_base = hdr + 16;
    const char* cw_base    = hdr + 16 + scale_padded;

    const int block_in_group = tid >> 2;
    const int cw_slot        = tid & 3;

    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail  = groups_per_row & 1;

    for (int p = 0; p < pairs; p++) {
        const int g  = p << 1;
        const int b0 = g * 8 + block_in_group;
        const int b1 = b0 + 8;

        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32_strip(*(const unsigned char*)(scale_base + b0)) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32_strip(*(const unsigned char*)(scale_base + b1)) * 0.88f;

        const unsigned int ew0 = *(const unsigned int*)(cw_base + (long long)b0 * 16 + cw_slot * 4);
        const unsigned int ew1 = *(const unsigned int*)(cw_base + (long long)b1 * 16 + cw_slot * 4);

        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;

        float pe[8];
        e8_decode_strip(ew0, pe);
        acc0 += ssc0*pe[0]*x[base0]+ssc0*pe[1]*x[base0+1]+ssc0*pe[2]*x[base0+2]+ssc0*pe[3]*x[base0+3]
              + ssc0*pe[4]*x[base0+4]+ssc0*pe[5]*x[base0+5]+ssc0*pe[6]*x[base0+6]+ssc0*pe[7]*x[base0+7];
        e8_decode_strip(ew1, pe);
        acc1 += ssc1*pe[0]*x[base1]+ssc1*pe[1]*x[base1+1]+ssc1*pe[2]*x[base1+2]+ssc1*pe[3]*x[base1+3]
              + ssc1*pe[4]*x[base1+4]+ssc1*pe[5]*x[base1+5]+ssc1*pe[6]*x[base1+6]+ssc1*pe[7]*x[base1+7];
    }

    if (tail) {
        const int g  = pairs << 1;
        const int b  = g * 8 + block_in_group;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32_strip(*(const unsigned char*)(scale_base + b)) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(cw_base + (long long)b * 16 + cw_slot * 4);
        const int base = g * 256 + tid * 8;
        float pe[8];
        e8_decode_strip(ew, pe);
        acc0 += ssc*pe[0]*x[base]+ssc*pe[1]*x[base+1]+ssc*pe[2]*x[base+2]+ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4]+ssc*pe[5]*x[base+5]+ssc*pe[6]*x[base+6]+ssc*pe[7]*x[base+7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}
