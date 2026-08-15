// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — E8-SoA U4 raw-buffer cache-policy route.

// `gfx12_weight_cache_policy.inc` must be concatenated before this file.
#include <hip/hip_runtime.h>

#ifndef E8_SOA_U4_BUFFER_KERNEL
#define E8_SOA_U4_BUFFER_KERNEL gemv_mfp4g32_e8_soa_u4_buffer
#endif

// Optional store-time activation. The reduced row sum is already in a register
// at the store, so an elementwise epilogue costs a few ALU ops on lane 0 and
// removes a whole dependent launch from the tape. Identity by default, so every
// existing route emits byte-identical code.
#ifndef E8_SOA_U4_BUFFER_EPILOGUE
#define E8_SOA_U4_BUFFER_EPILOGUE(v) (v)
#endif

// sqrt(softplus(x)) — the DeepSeek V4 MoE routing affinity activation. Kept
// operation-for-operation identical to the standalone `sqrt_softplus_f32`
// kernel (branched formulation for numerical stability) so fusing it into the
// gate GEMV reproduces the two-kernel result exactly.
__device__ __forceinline__ float e8_soa_u4_sqrt_softplus(float v) {
    const float sp = (v > 0.0f) ? (v + logf(1.0f + expf(-v))) : logf(1.0f + expf(v));
    return sqrtf(sp);
}

__device__ __forceinline__ float cvt_e4m3_scale_to_f32_soa_buffer(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index_soa_buffer(unsigned int idx, float p[8]) {
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
        const float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

__launch_bounds__(32)
extern "C" __global__ void E8_SOA_U4_BUFFER_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int n_blocks = K / 32;
    const int groups_per_row = K / 256;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const unsigned int row_total =
        16u + (unsigned int)scale_padded + (unsigned int)n_blocks * 16u;
    const unsigned int row_offset = (unsigned int)row * row_total;
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);

    const unsigned int row_header =
        HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, row_offset);
    const float row_scale_f =
        (float)__builtin_bit_cast(_Float16, (unsigned short)(row_header & 0xffffu));
    const unsigned int scale_offset = row_offset + 16u;
    const unsigned int codeword_offset = scale_offset + (unsigned int)scale_padded;
    const int block_in_group = tid >> 2;
    const int cw_slot = tid & 3;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int b0 = g * 8 + block_in_group;
        const int b1 = b0 + 8;
        const int b2 = b0 + 16;
        const int b3 = b0 + 24;

        const unsigned char scale0 = (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)b0);
        const unsigned char scale1 = (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)b1);
        const unsigned char scale2 = (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)b2);
        const unsigned char scale3 = (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)b3);
        const float ssc0 =
            row_scale_f * cvt_e4m3_scale_to_f32_soa_buffer(scale0) * 0.88f;
        const float ssc1 =
            row_scale_f * cvt_e4m3_scale_to_f32_soa_buffer(scale1) * 0.88f;
        const float ssc2 =
            row_scale_f * cvt_e4m3_scale_to_f32_soa_buffer(scale2) * 0.88f;
        const float ssc3 =
            row_scale_f * cvt_e4m3_scale_to_f32_soa_buffer(scale3) * 0.88f;
        const unsigned int ew0 = HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc,
            codeword_offset + (unsigned int)b0 * 16u + (unsigned int)cw_slot * 4u);
        const unsigned int ew1 = HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc,
            codeword_offset + (unsigned int)b1 * 16u + (unsigned int)cw_slot * 4u);
        const unsigned int ew2 = HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc,
            codeword_offset + (unsigned int)b2 * 16u + (unsigned int)cw_slot * 4u);
        const unsigned int ew3 = HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc,
            codeword_offset + (unsigned int)b3 * 16u + (unsigned int)cw_slot * 4u);
        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;
        const int base2 = base0 + 512;
        const int base3 = base0 + 768;
        float pe[8];
        e8_decode_index_soa_buffer(ew0, pe);
        acc0 += ssc0 * pe[0] * x[base0] + ssc0 * pe[1] * x[base0 + 1]
            + ssc0 * pe[2] * x[base0 + 2] + ssc0 * pe[3] * x[base0 + 3]
            + ssc0 * pe[4] * x[base0 + 4] + ssc0 * pe[5] * x[base0 + 5]
            + ssc0 * pe[6] * x[base0 + 6] + ssc0 * pe[7] * x[base0 + 7];
        e8_decode_index_soa_buffer(ew1, pe);
        acc1 += ssc1 * pe[0] * x[base1] + ssc1 * pe[1] * x[base1 + 1]
            + ssc1 * pe[2] * x[base1 + 2] + ssc1 * pe[3] * x[base1 + 3]
            + ssc1 * pe[4] * x[base1 + 4] + ssc1 * pe[5] * x[base1 + 5]
            + ssc1 * pe[6] * x[base1 + 6] + ssc1 * pe[7] * x[base1 + 7];
        e8_decode_index_soa_buffer(ew2, pe);
        acc2 += ssc2 * pe[0] * x[base2] + ssc2 * pe[1] * x[base2 + 1]
            + ssc2 * pe[2] * x[base2 + 2] + ssc2 * pe[3] * x[base2 + 3]
            + ssc2 * pe[4] * x[base2 + 4] + ssc2 * pe[5] * x[base2 + 5]
            + ssc2 * pe[6] * x[base2 + 6] + ssc2 * pe[7] * x[base2 + 7];
        e8_decode_index_soa_buffer(ew3, pe);
        acc3 += ssc3 * pe[0] * x[base3] + ssc3 * pe[1] * x[base3 + 1]
            + ssc3 * pe[2] * x[base3 + 2] + ssc3 * pe[3] * x[base3 + 3]
            + ssc3 * pe[4] * x[base3 + 4] + ssc3 * pe[5] * x[base3 + 5]
            + ssc3 * pe[6] * x[base3 + 6] + ssc3 * pe[7] * x[base3 + 7];
    }

    for (int group = quads << 2; group < groups_per_row; group++) {
        const int block = group * 8 + block_in_group;
        const unsigned char scale = (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block);
        const float ssc =
            row_scale_f * cvt_e4m3_scale_to_f32_soa_buffer(scale) * 0.88f;
        const unsigned int ew = HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc,
            codeword_offset + (unsigned int)block * 16u + (unsigned int)cw_slot * 4u);
        const int base = group * 256 + tid * 8;
        float pe[8];
        e8_decode_index_soa_buffer(ew, pe);
        acc0 += ssc * pe[0] * x[base] + ssc * pe[1] * x[base + 1]
            + ssc * pe[2] * x[base + 2] + ssc * pe[3] * x[base + 3]
            + ssc * pe[4] * x[base + 4] + ssc * pe[5] * x[base + 5]
            + ssc * pe[6] * x[base + 6] + ssc * pe[7] * x[base + 7];
    }

    float acc = acc0 + acc1 + acc2 + acc3;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = E8_SOA_U4_BUFFER_EPILOGUE(acc);
}
