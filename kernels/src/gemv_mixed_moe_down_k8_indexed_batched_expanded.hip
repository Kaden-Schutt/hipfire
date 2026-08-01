// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

// Merged per-expert MIXED-PRECISION MoE down GEMV (expanded write).
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
// + one expanded write — byte-identical to the reference expanded kernels.
// Feeds the dtype-agnostic `moe_down_combine_k8_batched`.
//
// Shapes:
//   expert_ptrs    : [n_exp]
//   dtype_tags     : [n_exp]                 u8, 0=MQ6 / 1=MQ2-Lloyd / 2=MQ4 / 3=MQ3-Lloyd
//   topk_indices   : [N × K_TOP]
//   rot_batch      : [N × K_TOP × K]
//   expert_outputs : [N × K_TOP × M]         non-atomic, one writer per cell
//
// Grid: (M, K_TOP, N). Block = 32 threads.

// ───────────────────────── mfp4-E8 (tag 4) decode ─────────────────────────
// Ported byte-identical from the proven uniform-E8 kernel
// (gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded.gfx11_dgpu.hip).
// `e8_row_partial` returns the PER-THREAD partial (NO wave reduce) so this
// kernel's shared wave32 reduce folds it, like the MQ tag branches. E8 row
// layout: 16 B header (fp16 row scale) + (K/32)*17 B blocks; 256-group = 136 B.
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

// mfp3-E8 row partial (down variant: takes `x` not `x_row`).
// Block stride 13 B, group stride 104 B, cw_off = (tid&3)*3.
__device__ __forceinline__ float mfp3e8_row_partial(
    const char* __restrict__ A, const float* __restrict__ x, int row, int K, int tid
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
        acc0 += ssc0*pe[0]*x[base0  ] + ssc0*pe[1]*x[base0+1] + ssc0*pe[2]*x[base0+2] + ssc0*pe[3]*x[base0+3]
              + ssc0*pe[4]*x[base0+4] + ssc0*pe[5]*x[base0+5] + ssc0*pe[6]*x[base0+6] + ssc0*pe[7]*x[base0+7];
        mfp3_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x[base1  ] + ssc1*pe[1]*x[base1+1] + ssc1*pe[2]*x[base1+2] + ssc1*pe[3]*x[base1+3]
              + ssc1*pe[4]*x[base1+4] + ssc1*pe[5]*x[base1+5] + ssc1*pe[6]*x[base1+6] + ssc1*pe[7]*x[base1+7];
        mfp3_decode_index(ew2, pe);
        acc2 += ssc2*pe[0]*x[base2  ] + ssc2*pe[1]*x[base2+1] + ssc2*pe[2]*x[base2+2] + ssc2*pe[3]*x[base2+3]
              + ssc2*pe[4]*x[base2+4] + ssc2*pe[5]*x[base2+5] + ssc2*pe[6]*x[base2+6] + ssc2*pe[7]*x[base2+7];
        mfp3_decode_index(ew3, pe);
        acc3 += ssc3*pe[0]*x[base3  ] + ssc3*pe[1]*x[base3+1] + ssc3*pe[2]*x[base3+2] + ssc3*pe[3]*x[base3+3]
              + ssc3*pe[4]*x[base3+4] + ssc3*pe[5]*x[base3+5] + ssc3*pe[6]*x[base3+6] + ssc3*pe[7]*x[base3+7];
    }
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP3;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8) | ((unsigned)w[2] << 16);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp3_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP3;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8) | ((unsigned)w[2] << 16);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp3_decode_index(ew, pe);
        acc1 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP3;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8) | ((unsigned)w[2] << 16);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp3_decode_index(ew, pe);
        acc2 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

// ─────────────── mfp2-E8 (tag 6): 2-bit nibbles, center 2 ───────────────
// Bit layout (16 bits = 2 bytes LE):
//   e[i], i=0..6 → bits [2i, 2i+2)      (nibble mask 0x3)
//   e7_high (1 bit) → bit 14
//   coset           → bit 15
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

// mfp2-E8 row partial (down variant: takes `x` not `x_row`).
// Block stride 9 B, group stride 72 B, cw_off = (tid&3)*2.
__device__ __forceinline__ float mfp2e8_row_partial(
    const char* __restrict__ A, const float* __restrict__ x, int row, int K, int tid
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
        acc0 += ssc0*pe[0]*x[base0  ] + ssc0*pe[1]*x[base0+1] + ssc0*pe[2]*x[base0+2] + ssc0*pe[3]*x[base0+3]
              + ssc0*pe[4]*x[base0+4] + ssc0*pe[5]*x[base0+5] + ssc0*pe[6]*x[base0+6] + ssc0*pe[7]*x[base0+7];
        mfp2_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x[base1  ] + ssc1*pe[1]*x[base1+1] + ssc1*pe[2]*x[base1+2] + ssc1*pe[3]*x[base1+3]
              + ssc1*pe[4]*x[base1+4] + ssc1*pe[5]*x[base1+5] + ssc1*pe[6]*x[base1+6] + ssc1*pe[7]*x[base1+7];
        mfp2_decode_index(ew2, pe);
        acc2 += ssc2*pe[0]*x[base2  ] + ssc2*pe[1]*x[base2+1] + ssc2*pe[2]*x[base2+2] + ssc2*pe[3]*x[base2+3]
              + ssc2*pe[4]*x[base2+4] + ssc2*pe[5]*x[base2+5] + ssc2*pe[6]*x[base2+6] + ssc2*pe[7]*x[base2+7];
        mfp2_decode_index(ew3, pe);
        acc3 += ssc3*pe[0]*x[base3  ] + ssc3*pe[1]*x[base3+1] + ssc3*pe[2]*x[base3+2] + ssc3*pe[3]*x[base3+3]
              + ssc3*pe[4]*x[base3+4] + ssc3*pe[5]*x[base3+5] + ssc3*pe[6]*x[base3+6] + ssc3*pe[7]*x[base3+7];
    }
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 72 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP2;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp2_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 72 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP2;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp2_decode_index(ew, pe);
        acc1 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 72 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * QUANT_STEP_MFP2;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_off);
        const unsigned int ew = (unsigned)w[0] | ((unsigned)w[1] << 8);
        const int base = g * 256 + tid * 8;
        float pe[8]; mfp2_decode_index(ew, pe);
        acc2 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1] + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5] + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mixed_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const unsigned char* __restrict__ dtype_tags,        // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const unsigned char tag = dtype_tags[expert_id]; // block-uniform
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;

    float acc = 0.0f;

    // cb_lds: 32 B shared, used only by tag==3 (MQ3-Lloyd); harmless to other branches.
    __shared__ float cb_lds[8];

    if (tag == 0) {
        // ---- MQ6 branch (200 B/group affine dequant) ----
        const int row_bytes = groups_per_row * 200;
        const char* row_ptr = A + (long long)row * row_bytes;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gptr = row_ptr + g * 200;
            float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
            float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
            const unsigned char* data = (const unsigned char*)(gptr + 8);

            int base_idx = g * 256 + tid * 8;
            int byte_off = tid * 6;

            unsigned char b0 = data[byte_off];
            unsigned char b1 = data[byte_off + 1];
            unsigned char b2 = data[byte_off + 2];
            unsigned char b3 = data[byte_off + 3];
            unsigned char b4 = data[byte_off + 4];
            unsigned char b5 = data[byte_off + 5];

            int q0 = b0 & 63;
            int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
            int q2 = (b1 >> 4) | ((b2 & 3) << 4);
            int q3 = b2 >> 2;
            int q4 = b3 & 63;
            int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
            int q6 = (b4 >> 4) | ((b5 & 3) << 4);
            int q7 = b5 >> 2;

            acc += (scale * (float)q0 + zero) * x[base_idx]
                 + (scale * (float)q1 + zero) * x[base_idx + 1]
                 + (scale * (float)q2 + zero) * x[base_idx + 2]
                 + (scale * (float)q3 + zero) * x[base_idx + 3]
                 + (scale * (float)q4 + zero) * x[base_idx + 4]
                 + (scale * (float)q5 + zero) * x[base_idx + 5]
                 + (scale * (float)q6 + zero) * x[base_idx + 6]
                 + (scale * (float)q7 + zero) * x[base_idx + 7];
        }
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
            acc += cb[b0 & 3]        * x[base]
                +  cb[(b0 >> 2) & 3] * x[base + 1]
                +  cb[(b0 >> 4) & 3] * x[base + 2]
                +  cb[b0 >> 6]       * x[base + 3]
                +  cb[b1 & 3]        * x[base + 4]
                +  cb[(b1 >> 2) & 3] * x[base + 5]
                +  cb[(b1 >> 4) & 3] * x[base + 6]
                +  cb[b1 >> 6]       * x[base + 7];
        }
    } else if (tag == 2) {
        // ---- MQ4 branch (136 B/group: [0..4) f32 scale, [4..8) f32 zero, [8..136) 128 B 4-bit) ----
        const int row_bytes = groups_per_row * 136;
        const char* row_ptr = A + (long long)row * row_bytes;
        const int boff = tid * 4;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gptr = row_ptr + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
            unsigned int pk = *(const unsigned int*)(gptr + 8 + boff);
            const int base = g * 256 + tid * 8;
            acc += (sc * (float)( pk        & 0xFu) + zp) * x[base]
                 + (sc * (float)((pk >>  4) & 0xFu) + zp) * x[base + 1]
                 + (sc * (float)((pk >>  8) & 0xFu) + zp) * x[base + 2]
                 + (sc * (float)((pk >> 12) & 0xFu) + zp) * x[base + 3]
                 + (sc * (float)((pk >> 16) & 0xFu) + zp) * x[base + 4]
                 + (sc * (float)((pk >> 20) & 0xFu) + zp) * x[base + 5]
                 + (sc * (float)((pk >> 24) & 0xFu) + zp) * x[base + 6]
                 + (sc * (float)((pk >> 28) & 0xFu) + zp) * x[base + 7];
        }
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
            acc += cb_lds[ pk        & 7u] * x[base]
                 + cb_lds[(pk >>  3) & 7u] * x[base + 1]
                 + cb_lds[(pk >>  6) & 7u] * x[base + 2]
                 + cb_lds[(pk >>  9) & 7u] * x[base + 3]
                 + cb_lds[(pk >> 12) & 7u] * x[base + 4]
                 + cb_lds[(pk >> 15) & 7u] * x[base + 5]
                 + cb_lds[(pk >> 18) & 7u] * x[base + 6]
                 + cb_lds[(pk >> 21) & 7u] * x[base + 7];
            __syncthreads();   // protect cb_lds before next group reload
        }
    } else if (tag == 4) {
        // ---- mfp4-E8 branch (16 B row hdr + (K/32)*17 B; 136 B / 256-group) ----
        // block-uniform → no syncthreads/LDS; per-thread partial, shared reduce folds.
        acc = e8_row_partial(A, x, row, K, tid);
    } else if (tag == 5) {
        // ---- mfp3-E8 branch (16 B row hdr + (K/32)*13 B; 104 B / 256-group) ----
        acc = mfp3e8_row_partial(A, x, row, K, tid);
    } else if (tag == 6) {
        // ---- mfp2-E8 branch (16 B row hdr + (K/32)* 9 B;  72 B / 256-group) ----
        acc = mfp2e8_row_partial(A, x, row, K, tid);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        // Direct store to per-(token, krank) row — no atomic contention.
        // Layout: expert_outputs[bid][krank][row].
        const long long out_offset =
            ((long long)bid * K_TOP + krank) * (long long)M + row;
        expert_outputs[out_offset] = acc;
    }
}
