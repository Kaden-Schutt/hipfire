// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 SoA decode-GEMV, gfx1151-specific (Strix Halo, RDNA3.5 APU).
//
// Structure-of-Arrays layout for fully-coalesced GEMV reads.
// Per-row byte layout:
//   [16B header: row_scale_a:f16 @0, n_blocks:u16 @4, flag:0x06 @6, rest 0]
//   [n_blocks bytes: E4M3 block scales, padded to 16B boundary]
//   [n_blocks * 16B: 4x u32 E8 codewords per block, 16B-aligned]
//
// Access pattern for 32 threads covering 8 consecutive blocks (one "group" = 256 K-elements):
//   Scale: each thread reads scale_base[block_in_group] → 8 consecutive bytes/group (coalesced).
//   Codewords: thread tid reads cw_base[b*16 + cw_slot*4] where b=tid>>2, cw_slot=tid&3.
//     tid=0..31 → byte offsets 0,4,8,12, 16,20,...,124 → 32×4B = 128 consecutive bytes
//     (fully coalesced 128B aligned read vs AoS stride-17 scattered reads).
//
// Bit-exact with gemv_mfp4g32_e8_gfx1151 for the same weight data.
// Target: VGPR ~62, 0 spills, occupancy 7-8 waves/SIMD32.

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
extern "C" __global__ void gemv_mfp4g32_e8_soa_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int n_blocks      = K / 32;
    const int groups_per_row = K / 256;  // each group covers 8 blocks = 256 K-elements

    // SoA per-row layout:
    //   header:       16 bytes
    //   scale array:  n_blocks bytes, padded to 16B boundary
    //   cw array:     n_blocks * 16 bytes (4 x u32 per block)
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const long long row_off   = (long long)row * row_total;

    const char* hdr = A + row_off;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const char* scale_base = hdr + 16;                 // E4M3 scale bytes (n_blocks bytes)
    const char* cw_base    = hdr + 16 + scale_padded;  // codewords (n_blocks * 16 bytes)

    // 32 threads cover one group of 8 blocks:
    //   block_in_group = tid >> 2   → block index within this group (0..7)
    //   cw_slot        = tid & 3    → which of the 4 u32s within that block (0..3)
    // This maps tid=0..31 to byte offsets 0,4,8,12, 16,20,..., 124 → 128 contiguous bytes.
    const int block_in_group = tid >> 2;
    const int cw_slot        = tid & 3;

    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail  = groups_per_row & 1;

    for (int p = 0; p < pairs; p++) {
        const int g  = p << 1;           // group index (0-based, step 2)
        const int b0 = g * 8 + block_in_group;
        const int b1 = b0 + 8;

        // Scale: one byte per block, contiguous across the group
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)(scale_base + b0)) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)(scale_base + b1)) * 0.88f;

        // Codewords: 32 threads × 4 bytes = 128 contiguous aligned bytes per group
        const unsigned int ew0 = *(const unsigned int*)(cw_base + (long long)b0 * 16 + cw_slot * 4);
        const unsigned int ew1 = *(const unsigned int*)(cw_base + (long long)b1 * 16 + cw_slot * 4);

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
        const int g  = pairs << 1;
        const int b  = g * 8 + block_in_group;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)(scale_base + b)) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(cw_base + (long long)b * 16 + cw_slot * 4);
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
