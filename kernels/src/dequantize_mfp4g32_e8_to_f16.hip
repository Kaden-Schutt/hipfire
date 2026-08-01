// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 dequantize-to-FP16 kernel for gfx942 (CDNA3 / MI300X).
//
// Same byte layout as mfp4+P (NO prefix, 16-B header + n_blocks×17B):
//   per-32-block: [1B E4M3 scale][16B data = 4×u32 E8 codewords]
// Per element: value = row_scale_a * e4m3_decode(scale_byte) * QUANT_STEP * e8_coord
//
// Launch: grid = (M, K/256, 1), block = (32, 1, 1).
// Each of the 32 threads dequants 8 elements (one codeword from one block).
// Thread tid: block_in_group = tid>>2, codeword_in_block = tid&3.
//
// Bit-identical to the CPU dequant_mfp4g32_e8 reference.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

// E4M3 (OCP, UNSIGNED scale, bias 7, 3 mantissa) — bit-identical to the
// CPU `e4m3_scale_decode` and the GEMV kernel.
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

// E8 decode: u32 index → 8 float coords.
// Bit-exact mirror of e8.rs::decode_index.
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

extern "C" __launch_bounds__(32, 16)
__global__ void dequantize_mfp4g32_e8_to_f16(
    const char*  __restrict__ A,      // [M × row_bytes], no prefix
    _Float16*    __restrict__ W_f16,  // [M × K] row-major
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;            // 8 mfp4 blocks each
    const int row_block_bytes = (K / 32) * 17;     // = groups_per_row * 136
    // mfp4-E8: NO prefix before row 0.
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;                // first block of this row

    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale_f = (float)row_scale_h;

    // 136-B group stride; this thread's block at +17*(tid>>2).
    const int block_in_iter = tid >> 2;            // 0..7
    const int cw_off = (tid & 3) * 4;             // byte offset of codeword in data region

    const char* gp = row_ptr + (long long)group * 136 + block_in_iter * 17;
    const unsigned char scale_byte = *(const unsigned char*)(gp);

    // Fold QUANT_STEP=0.88 into the scale.
    const float sc = row_scale_f * cvt_e4m3_scale_to_f32_dq(scale_byte) * 0.88f;

    const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);

    float pe[8];
    e8_decode_index(ew, pe);

    _Float16* out = W_f16 + (long long)row * K + (long long)group * 256 + tid * 8;
    for (int i = 0; i < 8; i++) {
        out[i] = (_Float16)(sc * pe[i]);
    }
}
