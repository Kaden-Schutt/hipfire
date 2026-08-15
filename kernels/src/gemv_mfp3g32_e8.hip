// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — dense mfp3-E8 decode GEMV.
//
// Wire layout per row:
//   16-byte header (row_scale_a:f16 at byte 0)
//   K/32 blocks, each [E4M3 scale:u8][4 x 24-bit E8 codeword]
//
// The 3-bit lattice decoder is byte-identical to the tag-5 path in
// gemv_mixed_moe_gate_up_k8_indexed_batched.hip. The activation is already
// FWHT-rotated by the caller.

#include <hip/hip_runtime.h>

#define QUANT_STEP_MFP3 1.8f

__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

// 24-bit E8 codeword:
//   e[0..6] 3 bits each; e7 high bits at 21..22; coset at bit 23.
// Center 4 is the tuned n=3 recenter from hipfire-quantize/src/e8.rs.
__device__ __forceinline__ void mfp3_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 23) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (unsigned)(3 * i)) & 0x7u;
        sl += e[i];
    }
    const unsigned int e7_high = (idx >> 21) & 0x3u;
    const unsigned int p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; i++) {
        const float c = (float)((int)e[i] - 4);
        p[i] = coset ? c + 0.5f : c;
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mfp3g32_e8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_block_bytes = (K / 32) * 13;
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale_f =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter = tid >> 2;
    const int cw_off = (tid & 3) * 3;
    const int thread_block_off = block_in_iter * 13;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#define MFP3_DOT(gp, base, acc) do { \
    const float ssc = row_scale_f \
        * cvt_e4m3_scale_to_f32(*(const unsigned char*)(gp)) \
        * QUANT_STEP_MFP3; \
    const unsigned char* w = (const unsigned char*)((gp) + 1 + cw_off); \
    const unsigned int ew = (unsigned)w[0] \
        | ((unsigned)w[1] << 8) \
        | ((unsigned)w[2] << 16); \
    float pe[8]; \
    mfp3_decode_index(ew, pe); \
    (acc) += ssc*pe[0]*x[(base)  ] + ssc*pe[1]*x[(base)+1] \
           + ssc*pe[2]*x[(base)+2] + ssc*pe[3]*x[(base)+3] \
           + ssc*pe[4]*x[(base)+4] + ssc*pe[5]*x[(base)+5] \
           + ssc*pe[6]*x[(base)+6] + ssc*pe[7]*x[(base)+7]; \
} while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 104 + thread_block_off;
        const char* gp1 = gp0 + 104;
        const char* gp2 = gp1 + 104;
        const char* gp3 = gp2 + 104;
        const int base0 = g * 256 + tid * 8;
        MFP3_DOT(gp0, base0,       acc0);
        MFP3_DOT(gp1, base0 + 256, acc1);
        MFP3_DOT(gp2, base0 + 512, acc2);
        MFP3_DOT(gp3, base0 + 768, acc3);
    }
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        MFP3_DOT(gp, g * 256 + tid * 8, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        MFP3_DOT(gp, g * 256 + tid * 8, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 104 + thread_block_off;
        MFP3_DOT(gp, g * 256 + tid * 8, acc2);
    }
#undef MFP3_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}
