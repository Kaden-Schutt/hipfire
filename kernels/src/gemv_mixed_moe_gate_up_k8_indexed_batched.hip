// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

// Merged per-expert MIXED-PRECISION MoE gate_up GEMV.
//
// Each routed expert in a layer has one of four precision tiers.
// The grid is block-per-(row, krank, token), so all 32 threads in a block
// share ONE expert_id → ONE dtype tag → the per-block branch is
// block-uniform → NO warp divergence.
//
//   dtype_tags[expert_id] == 0 → MQ6        (200 B/group: 4 B scale + 4 B zero
//                                             + 192 B of 6-bit packed quants)
//   dtype_tags[expert_id] == 1 → MQ2-Lloyd  (72 B/group: 4×fp16 codebook
//                                             + 64 B of 2-bit indices)
//   dtype_tags[expert_id] == 2 → MQ4        (136 B/group: 4 B scale + 4 B zero
//                                             + 128 B of 4-bit packed quants)
//   dtype_tags[expert_id] == 3 → MQ3-Lloyd  (112 B/group: 8×fp16 codebook
//                                             + 96 B of 3-bit packed quants)
//   dtype_tags[expert_id] == 4 → mfp4-E8    (16 B row hdr + (K/32)*17 B;
//                                             136 B/256-group, E8 lattice VQ)
//
// All branches accumulate into a SINGLE `acc` and share one wave32 reduce
// + one gate/up split write (M = 2*MI: first half → y_gate, second half →
// y_up) — byte-identical output layout to both reference kernels.
//
// x is indexed [bid * K] (per-token, NOT per-krank — gate_up reads the
// same input for every expert in a token's top-K).
//
// Shapes:
//   expert_ptrs : [n_exp]
//   dtype_tags  : [n_exp]                     u8, 0=MQ6 / 1=MQ2-Lloyd / 2=MQ4 / 3=MQ3-Lloyd
//   topk_indices: [N × K_TOP]
//   x           : [N × K]
//   y_gate      : [N × K_TOP × MI]            (MI = M / 2)
//   y_up        : [N × K_TOP × MI]
//
// Grid: (M, K_TOP, N). Block = 32 threads.

// ───────────────────────── mfp4-E8 (tag 4) decode ─────────────────────────
// Ported byte-identical from the proven uniform-E8 kernel
// (gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched.gfx11_dgpu.hip). `e8_row_partial`
// returns the PER-THREAD partial (NO wave reduce) so this kernel's shared wave32
// reduce folds it, exactly like the MQ tag branches. E8 row layout: 16 B header
// (fp16 row scale) + (K/32)*17 B blocks; one 256-group = 8 blocks = 136 B.
//
//   dtype_tags[expert_id] == 5 → mfp3-E8   (16 B row hdr + (K/32)*13 B;
//                                             104 B/256-group, 3-bit E8 VQ)
//   dtype_tags[expert_id] == 6 → mfp2-E8   (16 B row hdr + (K/32)* 9 B;
//                                              72 B/256-group, 2-bit E8 VQ)
//
// QUANT_STEP constants MUST match crates/hipfire-quantize/src/e8.rs.
#define QUANT_STEP_MFP3 1.8f
#define QUANT_STEP_MFP2 3.8f

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
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(4 * i)) & 0xFu; sl += e[i]; }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 7); p[i] = coset ? c + 0.5f : c; }
}
__device__ __forceinline__ float e8_row_partial(
    const char* __restrict__ A, const float* __restrict__ x, int row, int K, int tid
) {
    const int groups_per_row  = K / 256;
    const int row_block_bytes = (K / 32) * 17;
    const long long row_off   = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr     = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const int block_in_iter    = tid >> 2;
    const int cw_off           = (tid & 3) * 4;
    const int thread_block_off = block_in_iter * 17;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g       * 136 + thread_block_off;
        const char* gp1 = row_ptr + (long long)(g + 1) * 136 + thread_block_off;
        const char* gp2 = row_ptr + (long long)(g + 2) * 136 + thread_block_off;
        const char* gp3 = row_ptr + (long long)(g + 3) * 136 + thread_block_off;
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * 0.88f;
        const float ssc2 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp2) * 0.88f;
        const float ssc3 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp3) * 0.88f;
        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);
        const unsigned int ew2 = *(const unsigned int*)(gp2 + 1 + cw_off);
        const unsigned int ew3 = *(const unsigned int*)(gp3 + 1 + cw_off);
        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;
        const int base2 = base0 + 512;
        const int base3 = base0 + 768;
        float pe[8];
        e8_decode_index(ew0, pe);
        acc0 += ssc0*pe[0]*x[base0  ] + ssc0*pe[1]*x[base0+1] + ssc0*pe[2]*x[base0+2] + ssc0*pe[3]*x[base0+3]
              + ssc0*pe[4]*x[base0+4] + ssc0*pe[5]*x[base0+5] + ssc0*pe[6]*x[base0+6] + ssc0*pe[7]*x[base0+7];
        e8_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x[base1  ] + ssc1*pe[1]*x[base1+1] + ssc1*pe[2]*x[base1+2] + ssc1*pe[3]*x[base1+3]
              + ssc1*pe[4]*x[base1+4] + ssc1*pe[5]*x[base1+5] + ssc1*pe[6]*x[base1+6] + ssc1*pe[7]*x[base1+7];
        e8_decode_index(ew2, pe);
        acc2 += ssc2*pe[0]*x[base2  ] + ssc2*pe[1]*x[base2+1] + ssc2*pe[2]*x[base2+2] + ssc2*pe[3]*x[base2+3]
              + ssc2*pe[4]*x[base2+4] + ssc2*pe[5]*x[base2+5] + ssc2*pe[6]*x[base2+6] + ssc2*pe[7]*x[base2+7];
        e8_decode_index(ew3, pe);
        acc3 += ssc3*pe[0]*x[base3  ] + ssc3*pe[1]*x[base3+1] + ssc3*pe[2]*x[base3+2] + ssc3*pe[3]*x[base3+3]
              + ssc3*pe[4]*x[base3+4] + ssc3*pe[5]*x[base3+5] + ssc3*pe[6]*x[base3+6] + ssc3*pe[7]*x[base3+7];
    }
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8]; e8_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8]; e8_decode_index(ew, pe);
        acc1 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8]; e8_decode_index(ew, pe);
        acc2 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

// ─────────────── mfp3-E8 (tag 5): 3-bit nibbles, center 4 ───────────────
// Bit layout (24 bits = 3 bytes LE):
//   e[i], i=0..6 → bits [3i, 3i+3)      (nibble mask 0x7)
//   e7_high (2 bits) → bits [21, 23)
//   coset           → bit 23
// High 8 bits of the storing u32 are zero — narrow 3-byte read is lossless.
// Center 4 (box [-4,3]) matches coord_bias_n(3)=4 in e8.rs (MSE-tuned recenter).
__device__ __forceinline__ void mfp3_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 23) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(3 * i)) & 0x7u; sl += e[i]; }
    const unsigned int e7_high = (idx >> 21) & 0x3u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 4); p[i] = coset ? c + 0.5f : c; }
}

// mfp3-E8 row partial: block stride 13 B, group stride 104 B, cw_off = (tid&3)*3.
// Codeword reads use 3-byte assembly to avoid over-reading past the last block.
__device__ __forceinline__ float mfp3e8_row_partial(
    const char* __restrict__ A, const float* __restrict__ x_row, int row, int K, int tid
) {
    const int groups_per_row  = K / 256;
    const int row_block_bytes = (K / 32) * 13;
    const long long row_off   = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr     = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const int block_in_iter    = tid >> 2;
    const int cw_off           = (tid & 3) * 3;
    const int thread_block_off = block_in_iter * 13;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g       * 104 + thread_block_off;
        const char* gp1 = row_ptr + (long long)(g + 1) * 104 + thread_block_off;
        const char* gp2 = row_ptr + (long long)(g + 2) * 104 + thread_block_off;
        const char* gp3 = row_ptr + (long long)(g + 3) * 104 + thread_block_off;
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * QUANT_STEP_MFP3;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * QUANT_STEP_MFP3;
        const float ssc2 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp2) * QUANT_STEP_MFP3;
        const float ssc3 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp3) * QUANT_STEP_MFP3;
        // 3-byte narrow read: assembles low 24 bits of a u32 from 3 LE bytes.
        const unsigned char* w0 = (const unsigned char*)(gp0 + 1 + cw_off);
        const unsigned char* w1 = (const unsigned char*)(gp1 + 1 + cw_off);
        const unsigned char* w2 = (const unsigned char*)(gp2 + 1 + cw_off);
        const unsigned char* w3 = (const unsigned char*)(gp3 + 1 + cw_off);
        const unsigned int ew0 = (unsigned)w0[0] | ((unsigned)w0[1] << 8) | ((unsigned)w0[2] << 16);
        const unsigned int ew1 = (unsigned)w1[0] | ((unsigned)w1[1] << 8) | ((unsigned)w1[2] << 16);
        const unsigned int ew2 = (unsigned)w2[0] | ((unsigned)w2[1] << 8) | ((unsigned)w2[2] << 16);
        const unsigned int ew3 = (unsigned)w3[0] | ((unsigned)w3[1] << 8) | ((unsigned)w3[2] << 16);
        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;
        const int base2 = base0 + 512;
        const int base3 = base0 + 768;
        float pe[8];
        mfp3_decode_index(ew0, pe);
        acc0 += ssc0*pe[0]*x_row[base0  ] + ssc0*pe[1]*x_row[base0+1] + ssc0*pe[2]*x_row[base0+2] + ssc0*pe[3]*x_row[base0+3]
              + ssc0*pe[4]*x_row[base0+4] + ssc0*pe[5]*x_row[base0+5] + ssc0*pe[6]*x_row[base0+6] + ssc0*pe[7]*x_row[base0+7];
        mfp3_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x_row[base1  ] + ssc1*pe[1]*x_row[base1+1] + ssc1*pe[2]*x_row[base1+2] + ssc1*pe[3]*x_row[base1+3]
              + ssc1*pe[4]*x_row[base1+4] + ssc1*pe[5]*x_row[base1+5] + ssc1*pe[6]*x_row[base1+6] + ssc1*pe[7]*x_row[base1+7];
        mfp3_decode_index(ew2, pe);
        acc2 += ssc2*pe[0]*x_row[base2  ] + ssc2*pe[1]*x_row[base2+1] + ssc2*pe[2]*x_row[base2+2] + ssc2*pe[3]*x_row[base2+3]
              + ssc2*pe[4]*x_row[base2+4] + ssc2*pe[5]*x_row[base2+5] + ssc2*pe[6]*x_row[base2+6] + ssc2*pe[7]*x_row[base2+7];
        mfp3_decode_index(ew3, pe);
        acc3 += ssc3*pe[0]*x_row[base3  ] + ssc3*pe[1]*x_row[base3+1] + ssc3*pe[2]*x_row[base3+2] + ssc3*pe[3]*x_row[base3+3]
              + ssc3*pe[4]*x_row[base3+4] + ssc3*pe[5]*x_row[base3+5] + ssc3*pe[6]*x_row[base3+6] + ssc3*pe[7]*x_row[base3+7];
    }
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP3;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8) | ((unsigned)w[2] << 16);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp3_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x_row[base  ] + ssc*pe[1]*x_row[base+1] + ssc*pe[2]*x_row[base+2] + ssc*pe[3]*x_row[base+3]
              + ssc*pe[4]*x_row[base+4] + ssc*pe[5]*x_row[base+5] + ssc*pe[6]*x_row[base+6] + ssc*pe[7]*x_row[base+7];
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP3;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8) | ((unsigned)w[2] << 16);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp3_decode_index(ew, pe);
        acc1 += ssc*pe[0]*x_row[base  ] + ssc*pe[1]*x_row[base+1] + ssc*pe[2]*x_row[base+2] + ssc*pe[3]*x_row[base+3]
              + ssc*pe[4]*x_row[base+4] + ssc*pe[5]*x_row[base+5] + ssc*pe[6]*x_row[base+6] + ssc*pe[7]*x_row[base+7];
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP3;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8) | ((unsigned)w[2] << 16);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp3_decode_index(ew, pe);
        acc2 += ssc*pe[0]*x_row[base  ] + ssc*pe[1]*x_row[base+1] + ssc*pe[2]*x_row[base+2] + ssc*pe[3]*x_row[base+3]
              + ssc*pe[4]*x_row[base+4] + ssc*pe[5]*x_row[base+5] + ssc*pe[6]*x_row[base+6] + ssc*pe[7]*x_row[base+7];
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

// ─────────────── mfp2-E8 (tag 6): 2-bit nibbles, center 2 ───────────────
// Bit layout (16 bits = 2 bytes LE):
//   e[i], i=0..6 → bits [2i, 2i+2)      (nibble mask 0x3)
//   e7_high (1 bit) → bit 14
//   coset           → bit 15
// High 16 bits of the storing u32 are zero — narrow 2-byte read is lossless.
// Center 2 (box [-2,1]) matches coord_bias_n(2)=2 in e8.rs (MSE-tuned recenter).
__device__ __forceinline__ void mfp2_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 15) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(2 * i)) & 0x3u; sl += e[i]; }
    const unsigned int e7_high = (idx >> 14) & 0x1u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 2); p[i] = coset ? c + 0.5f : c; }
}

// mfp2-E8 row partial: block stride 9 B, group stride 72 B, cw_off = (tid&3)*2.
// Codeword reads use 2-byte assembly to avoid over-reading past the last block.
__device__ __forceinline__ float mfp2e8_row_partial(
    const char* __restrict__ A, const float* __restrict__ x_row, int row, int K, int tid
) {
    const int groups_per_row  = K / 256;
    const int row_block_bytes = (K / 32) * 9;
    const long long row_off   = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr     = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const int block_in_iter    = tid >> 2;
    const int cw_off           = (tid & 3) * 2;
    const int thread_block_off = block_in_iter * 9;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g       * 72 + thread_block_off;
        const char* gp1 = row_ptr + (long long)(g + 1) * 72 + thread_block_off;
        const char* gp2 = row_ptr + (long long)(g + 2) * 72 + thread_block_off;
        const char* gp3 = row_ptr + (long long)(g + 3) * 72 + thread_block_off;
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * QUANT_STEP_MFP2;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * QUANT_STEP_MFP2;
        const float ssc2 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp2) * QUANT_STEP_MFP2;
        const float ssc3 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp3) * QUANT_STEP_MFP2;
        // 2-byte narrow read: assembles low 16 bits of a u32 from 2 LE bytes.
        const unsigned char* w0 = (const unsigned char*)(gp0 + 1 + cw_off);
        const unsigned char* w1 = (const unsigned char*)(gp1 + 1 + cw_off);
        const unsigned char* w2 = (const unsigned char*)(gp2 + 1 + cw_off);
        const unsigned char* w3 = (const unsigned char*)(gp3 + 1 + cw_off);
        const unsigned int ew0 = (unsigned)w0[0] | ((unsigned)w0[1] << 8);
        const unsigned int ew1 = (unsigned)w1[0] | ((unsigned)w1[1] << 8);
        const unsigned int ew2 = (unsigned)w2[0] | ((unsigned)w2[1] << 8);
        const unsigned int ew3 = (unsigned)w3[0] | ((unsigned)w3[1] << 8);
        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;
        const int base2 = base0 + 512;
        const int base3 = base0 + 768;
        float pe[8];
        mfp2_decode_index(ew0, pe);
        acc0 += ssc0*pe[0]*x_row[base0  ] + ssc0*pe[1]*x_row[base0+1] + ssc0*pe[2]*x_row[base0+2] + ssc0*pe[3]*x_row[base0+3]
              + ssc0*pe[4]*x_row[base0+4] + ssc0*pe[5]*x_row[base0+5] + ssc0*pe[6]*x_row[base0+6] + ssc0*pe[7]*x_row[base0+7];
        mfp2_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x_row[base1  ] + ssc1*pe[1]*x_row[base1+1] + ssc1*pe[2]*x_row[base1+2] + ssc1*pe[3]*x_row[base1+3]
              + ssc1*pe[4]*x_row[base1+4] + ssc1*pe[5]*x_row[base1+5] + ssc1*pe[6]*x_row[base1+6] + ssc1*pe[7]*x_row[base1+7];
        mfp2_decode_index(ew2, pe);
        acc2 += ssc2*pe[0]*x_row[base2  ] + ssc2*pe[1]*x_row[base2+1] + ssc2*pe[2]*x_row[base2+2] + ssc2*pe[3]*x_row[base2+3]
              + ssc2*pe[4]*x_row[base2+4] + ssc2*pe[5]*x_row[base2+5] + ssc2*pe[6]*x_row[base2+6] + ssc2*pe[7]*x_row[base2+7];
        mfp2_decode_index(ew3, pe);
        acc3 += ssc3*pe[0]*x_row[base3  ] + ssc3*pe[1]*x_row[base3+1] + ssc3*pe[2]*x_row[base3+2] + ssc3*pe[3]*x_row[base3+3]
              + ssc3*pe[4]*x_row[base3+4] + ssc3*pe[5]*x_row[base3+5] + ssc3*pe[6]*x_row[base3+6] + ssc3*pe[7]*x_row[base3+7];
    }
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 72 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP2;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp2_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x_row[base  ] + ssc*pe[1]*x_row[base+1] + ssc*pe[2]*x_row[base+2] + ssc*pe[3]*x_row[base+3]
              + ssc*pe[4]*x_row[base+4] + ssc*pe[5]*x_row[base+5] + ssc*pe[6]*x_row[base+6] + ssc*pe[7]*x_row[base+7];
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 72 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP2;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp2_decode_index(ew, pe);
        acc1 += ssc*pe[0]*x_row[base  ] + ssc*pe[1]*x_row[base+1] + ssc*pe[2]*x_row[base+2] + ssc*pe[3]*x_row[base+3]
              + ssc*pe[4]*x_row[base+4] + ssc*pe[5]*x_row[base+5] + ssc*pe[6]*x_row[base+6] + ssc*pe[7]*x_row[base+7];
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 72 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP2;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp2_decode_index(ew, pe);
        acc2 += ssc*pe[0]*x_row[base  ] + ssc*pe[1]*x_row[base+1] + ssc*pe[2]*x_row[base+2] + ssc*pe[3]*x_row[base+3]
              + ssc*pe[4]*x_row[base+4] + ssc*pe[5]*x_row[base+5] + ssc*pe[6]*x_row[base+6] + ssc*pe[7]*x_row[base+7];
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mixed_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const unsigned char* __restrict__ dtype_tags,        // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char tag = dtype_tags[expert_id]; // block-uniform
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 256;

    float acc = 0.0f;

    // cb_lds: 32 B shared, used only by tag==3 (MQ3-Lloyd); harmless to other branches.
    __shared__ float cb_lds[8];

    // Per-group dot (8 MACs) as a value, so the hot MQ6/MQ4 branches can carry
    // 4 INDEPENDENT accumulators across groups — restoring the quad-unroll ILP
    // the uniform gemv has (the single-`acc` chain serialized the FMA latency).
    // Self-contained: derives row_ptr from A/row/groups_per_row, offsets from tid.
#define _MQ6G(gg) ({ \
        const char* _gp = (const char*)A + ((long long)row * groups_per_row + (gg)) * 200; \
        float _s = __builtin_bit_cast(float, *(const unsigned int*)(_gp)); \
        float _z = __builtin_bit_cast(float, *(const unsigned int*)(_gp + 4)); \
        const unsigned char* _d = (const unsigned char*)(_gp + 8) + tid * 6; \
        int _bi = (gg) * 256 + tid * 8; \
        unsigned char _b0=_d[0],_b1=_d[1],_b2=_d[2],_b3=_d[3],_b4=_d[4],_b5=_d[5]; \
        int _q0=_b0&63,_q1=(_b0>>6)|((_b1&0xF)<<2),_q2=(_b1>>4)|((_b2&3)<<4),_q3=_b2>>2, \
            _q4=_b3&63,_q5=(_b3>>6)|((_b4&0xF)<<2),_q6=(_b4>>4)|((_b5&3)<<4),_q7=_b5>>2; \
        (_s*(float)_q0+_z)*x_row[_bi]   +(_s*(float)_q1+_z)*x_row[_bi+1] \
       +(_s*(float)_q2+_z)*x_row[_bi+2] +(_s*(float)_q3+_z)*x_row[_bi+3] \
       +(_s*(float)_q4+_z)*x_row[_bi+4] +(_s*(float)_q5+_z)*x_row[_bi+5] \
       +(_s*(float)_q6+_z)*x_row[_bi+6] +(_s*(float)_q7+_z)*x_row[_bi+7]; })
#define _MQ4G(gg) ({ \
        const char* _gp = (const char*)A + ((long long)row * groups_per_row + (gg)) * 136; \
        float _s = __builtin_bit_cast(float, *(const unsigned int*)(_gp)); \
        float _z = __builtin_bit_cast(float, *(const unsigned int*)(_gp + 4)); \
        unsigned int _pk = *(const unsigned int*)(_gp + 8 + tid * 4); \
        int _b = (gg) * 256 + tid * 8; \
        (_s*(float)(_pk&0xFu)+_z)*x_row[_b]        +(_s*(float)((_pk>>4)&0xFu)+_z)*x_row[_b+1] \
       +(_s*(float)((_pk>>8)&0xFu)+_z)*x_row[_b+2] +(_s*(float)((_pk>>12)&0xFu)+_z)*x_row[_b+3] \
       +(_s*(float)((_pk>>16)&0xFu)+_z)*x_row[_b+4]+(_s*(float)((_pk>>20)&0xFu)+_z)*x_row[_b+5] \
       +(_s*(float)((_pk>>24)&0xFu)+_z)*x_row[_b+6]+(_s*(float)((_pk>>28)&0xFu)+_z)*x_row[_b+7]; })

    if (tag == 0) {
        // ---- MQ6 branch (200 B/group affine dequant), 4-accumulator unroll ----
        float _a0=0.f,_a1=0.f,_a2=0.f,_a3=0.f;
        int g = 0;
        for (; g + 3 < groups_per_row; g += 4) {
            _a0 += _MQ6G(g);   _a1 += _MQ6G(g+1);
            _a2 += _MQ6G(g+2); _a3 += _MQ6G(g+3);
        }
        for (; g < groups_per_row; g++) _a0 += _MQ6G(g);
        acc = (_a0 + _a1) + (_a2 + _a3);
    } else if (tag == 1) {
        // ---- MQ2-Lloyd branch (72 B/group 2-bit codebook) ----
        const int row_bytes = groups_per_row * 72;
        const char* row_ptr = A + (long long)row * row_bytes;
        const int boff = tid * 2;
        const int xoff = tid * 8;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gptr = row_ptr + g * 72;
            const __half* cb_h = reinterpret_cast<const __half*>(gptr);
            float cb[4] = {
                __half2float(cb_h[0]),
                __half2float(cb_h[1]),
                __half2float(cb_h[2]),
                __half2float(cb_h[3]),
            };
            const unsigned char* data = (const unsigned char*)(gptr + 8);
            unsigned char b0 = data[boff];
            unsigned char b1 = data[boff + 1];
            const int base = g * 256 + xoff;
            acc += cb[b0 & 3]        * x_row[base]
                +  cb[(b0 >> 2) & 3] * x_row[base + 1]
                +  cb[(b0 >> 4) & 3] * x_row[base + 2]
                +  cb[b0 >> 6]       * x_row[base + 3]
                +  cb[b1 & 3]        * x_row[base + 4]
                +  cb[(b1 >> 2) & 3] * x_row[base + 5]
                +  cb[(b1 >> 4) & 3] * x_row[base + 6]
                +  cb[b1 >> 6]       * x_row[base + 7];
        }
    } else if (tag == 2) {
        // ---- MQ4 branch (136 B/group), 4-accumulator unroll ----
        float _a0=0.f,_a1=0.f,_a2=0.f,_a3=0.f;
        int g = 0;
        for (; g + 3 < groups_per_row; g += 4) {
            _a0 += _MQ4G(g);   _a1 += _MQ4G(g+1);
            _a2 += _MQ4G(g+2); _a3 += _MQ4G(g+3);
        }
        for (; g < groups_per_row; g++) _a0 += _MQ4G(g);
        acc = (_a0 + _a1) + (_a2 + _a3);
    } else if (tag == 3) {
        // ---- MQ3-Lloyd branch (112 B/group: [0..16) 8xfp16 codebook, [16..112) 96 B 3-bit) ----
        const int row_bytes = groups_per_row * 112;
        const char* row_ptr = A + (long long)row * row_bytes;
        const int boff = tid * 3;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gptr = row_ptr + g * 112;
            if (tid < 8) cb_lds[tid] = __half2float(((const __half*)gptr)[tid]);
            __syncthreads();
            const unsigned char* d = (const unsigned char*)(gptr + 16) + boff;
            unsigned int pk = (unsigned int)d[0]
                            | ((unsigned int)d[1] << 8)
                            | ((unsigned int)d[2] << 16);
            const int base = g * 256 + tid * 8;
            acc += cb_lds[ pk        & 7u] * x_row[base]
                 + cb_lds[(pk >>  3) & 7u] * x_row[base + 1]
                 + cb_lds[(pk >>  6) & 7u] * x_row[base + 2]
                 + cb_lds[(pk >>  9) & 7u] * x_row[base + 3]
                 + cb_lds[(pk >> 12) & 7u] * x_row[base + 4]
                 + cb_lds[(pk >> 15) & 7u] * x_row[base + 5]
                 + cb_lds[(pk >> 18) & 7u] * x_row[base + 6]
                 + cb_lds[(pk >> 21) & 7u] * x_row[base + 7];
            __syncthreads();   // protect cb_lds before next group reload
        }
    } else if (tag == 4) {
        // ---- mfp4-E8 branch (16 B row hdr + (K/32)*17 B; 136 B / 256-group) ----
        // block-uniform → no syncthreads/LDS; per-thread partial, shared reduce folds.
        acc = e8_row_partial(A, x_row, row, K, tid);
    } else if (tag == 5) {
        // ---- mfp3-E8 branch (16 B row hdr + (K/32)*13 B; 104 B / 256-group) ----
        acc = mfp3e8_row_partial(A, x_row, row, K, tid);
    } else if (tag == 6) {
        // ---- mfp2-E8 branch (16 B row hdr + (K/32)* 9 B;  72 B / 256-group) ----
        acc = mfp2e8_row_partial(A, x_row, row, K, tid);
    }
#undef _MQ6G
#undef _MQ4G

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        // M = 2 * MI, first half is gate, second half is up.
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
