// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 GEMV kernel for gfx942 (CDNA3 / MI300X).
//
// Same container as mfp4+P (NO prefix, 16-B header + n_blocks×17B):
//   per-row: row_scale_a:f16 @0, n_blocks:u16 @4, flags=0x05 @6
//   per-32-block: [1B E4M3 scale][16B data = 4×u32 E8 codewords]
// Per element: value = row_scale_a * e4m3_decode(scale_byte) * QUANT_STEP * e8_coord
//
// Structure (136-B group stride, 4-acc interleave, tail-by-g&3,
// (acc0+acc1)+(acc2+acc3) finalize) mirrors gemv_mfp4g32_p.hip 1:1.
// Key changes vs mfp4+P:
//   (1) DROP the __shared__ float lut[16] / E2M1 LUT.
//   (2) Each thread reads one 32-bit E8 codeword at offset +1+(tid&3)*4.
//   (3) Decodes 8 coords via e8_decode_index, dot-products 8 at a time.
//   (4) QUANT_STEP = 0.88 folded into sc (ssc = sc * 0.88f).

#include <hip/hip_runtime.h>

// E4M3 (OCP, UNSIGNED scale, bias 7, 3 mantissa) — bit-identical to the CPU
// `e4m3_scale_decode` in the quantizer and dequantize_mfp4g32_p_to_f16.
__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) {
        return 0.015625f * ((float)mant) * 0.125f;
    }
    if (exp == 0xF && mant == 7u) {
        return 448.0f;
    }
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

// E8 decode: u32 index → 8 float coords.
// Bit-exact mirror of e8.rs::decode_index + * QUANT_STEP.
// round_tie_away / closest_d8 are encode-only; decode is pure integer + f32.
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

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mfp4g32_e8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;            // each group = 8 mfp4 blocks
    const int row_block_bytes = (K / 32) * 17;     // = groups_per_row * 136
    // mfp4-E8: NO prefix; row 0 starts at byte 0 (same as mfp4+P).
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;                // first block of this row

    const unsigned short row_scale_bits = *(const unsigned short*)(hdr);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, row_scale_bits);
    const float row_scale_f = (float)row_scale_h;

    // tid>>2 = which of the 8 blocks in this group (0..7)
    const int block_in_iter = tid >> 2;            // 0..7
    // tid&3 = which of the 4 E8 codewords in the block (each covers 8 weights)
    const int cw_off = (tid & 3) * 4;             // byte offset within the 16-B data region

    const int thread_block_off = block_in_iter * 17;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // E8_DOG: decode one codeword (8 weights) and dot-product with x.
    // ssc = row_scale_f * cvt_e4m3_scale_to_f32(e) * 0.88f
    #define E8_DOG(ew, ssc, b, a) { \
        float pe[8]; e8_decode_index((ew), pe); \
        (a) += (ssc)*pe[0]*x[(b)  ] + (ssc)*pe[1]*x[(b)+1] \
             + (ssc)*pe[2]*x[(b)+2] + (ssc)*pe[3]*x[(b)+3] \
             + (ssc)*pe[4]*x[(b)+4] + (ssc)*pe[5]*x[(b)+5] \
             + (ssc)*pe[6]*x[(b)+6] + (ssc)*pe[7]*x[(b)+7]; }

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

        // Fold QUANT_STEP=0.88 into the scale.
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(e0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(e1) * 0.88f;
        const float ssc2 = row_scale_f * cvt_e4m3_scale_to_f32(e2) * 0.88f;
        const float ssc3 = row_scale_f * cvt_e4m3_scale_to_f32(e3) * 0.88f;

        // Read this thread's E8 codeword (at +1+cw_off = +1 scale_byte + 4*(tid&3)).
        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);
        const unsigned int ew2 = *(const unsigned int*)(gp2 + 1 + cw_off);
        const unsigned int ew3 = *(const unsigned int*)(gp3 + 1 + cw_off);

        // base = g*256 + (block_in_iter)*32 + (tid&3)*8 = g*256 + tid*8
        const int base = g * 256 + tid * 8;

        E8_DOG(ew0, ssc0, base,         acc0);
        E8_DOG(ew1, ssc1, base + 256,   acc1);
        E8_DOG(ew2, ssc2, base + 512,   acc2);
        E8_DOG(ew3, ssc3, base + 768,   acc3);
    }
    #undef E8_DOG

    #define E8_TAIL_DOG(ew, ssc, b, a) { \
        float pe[8]; e8_decode_index((ew), pe); \
        (a) += (ssc)*pe[0]*x[(b)  ] + (ssc)*pe[1]*x[(b)+1] \
             + (ssc)*pe[2]*x[(b)+2] + (ssc)*pe[3]*x[(b)+3] \
             + (ssc)*pe[4]*x[(b)+4] + (ssc)*pe[5]*x[(b)+5] \
             + (ssc)*pe[6]*x[(b)+6] + (ssc)*pe[7]*x[(b)+7]; }

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(e) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        E8_TAIL_DOG(ew, ssc, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(e) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        E8_TAIL_DOG(ew, ssc, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const unsigned char e = *(const unsigned char*)(gp);
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(e) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        E8_TAIL_DOG(ew, ssc, base, acc2);
    }
    #undef E8_TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
