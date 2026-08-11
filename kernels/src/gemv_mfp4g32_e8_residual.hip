// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// mfp4-E8 dense GEMV with fused residual add: y[row] += A[row] · x.
// The o_proj / out_proj shape (attention output projection).
//
// CONTRACT mirrors gemv_mq3g256_lloyd_residual.hip 1:1:
//   (const char* A, const float* x, float* y, int M, int K)
//   grid [M,1,1], block [32,1,1], 0 dynamic LDS, __launch_bounds__(32, 16),
//   final write `y[row] += acc` instead of `y[row] = acc`.
// BODY is gemv_mfp4g32_e8.hip verbatim — same 136-B group stride, same
// 4-acc interleave, same tail-by-(g&3), same (acc0+acc1)+(acc2+acc3)
// finalize, same 5-step __shfl_down wave32 reduce. Only the store differs.
//
// Container (NO prefix; row 0 starts at byte 0 — same as mfp4+P):
//   per-row:      row_scale_a:f16 @0, n_blocks:u16 @4, flags=0x05 @6, 16 B total
//   per-32-block: [1B E4M3 scale][16B data = 4×u32 E8 codewords] = 17 B
//   per-256-group: 8 blocks = 136 B
// Per element: value = row_scale_a * e4m3_decode(scale_byte) * QUANT_STEP * e8_coord
// QUANT_STEP = 0.88 (e8.rs:15) is folded into the scale: ssc = sc * 0.88f.
//
// X is FWHT-256 pre-rotated by the caller (the encoder FWHTs each 256-segment
// of the weight row before quantizing). This kernel does NO rotation.
//
// WHY this file exists: `weight_gemv_residual` previously had no mfp4-E8 arm,
// so an E8 o_proj fell back to alloc + gemv + add_inplace_f32 + free — the same
// chain gemv_mq3g256_lloyd_residual.gfx1100.hip was written to eliminate
// (4.4% add_inplace_f32 in the 2026-05-06 decode profile). Without this kernel
// E8 cannot serve attention output projections at all.
//
// NO .gfx1100 variant, deliberately. The mq3 residual needs one because its
// baseline (gemv_mq3g256_lloyd_residual.hip) is single-accumulator + flat loop;
// the .gfx1100 sibling exists to add the K4 unroll + LDS codebook. The E8
// baseline (gemv_mfp4g32_e8.hip) is ALREADY 4-acc + quads/tails with a
// register-resident decode and no codebook to stage in LDS, so a gfx1100
// specialization would be a byte-identical copy. The only E8 gfx11 twin in
// tree (gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched.gfx11_dgpu.hip) is
// default-OFF behind HIPFIRE_E8_DGPU_TWIN because it measured ~0% on gfx1100.
// One arch-agnostic file is the correct shape here.
//
// Numerical note: bit-exact against gemv_mfp4g32_e8 for the dot product (same
// accumulator partition, same fold parentheses, same per-term ssc placement),
// so `y[row] += acc` here equals gemv + add_inplace_f32 to the ULP.

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
extern "C" __global__ void gemv_mfp4g32_e8_residual(
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

    if (tid == 0) y[row] += acc;   // fused residual add
}
