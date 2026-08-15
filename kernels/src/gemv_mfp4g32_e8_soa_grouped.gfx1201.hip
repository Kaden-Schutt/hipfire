// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1201 grouped mfp4-E8 SoA decode GEMV.
//
// One 2-D launch replaces the eight O-LoRA GEMVs. The per-row body is kept
// arithmetic-identical to gemv_mfp4g32_e8_soa.gfx1201.hip; grouping changes
// only block addressing and launch structure.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float cvt_e4m3_scale_to_f32_grouped_gfx1201(
    unsigned char b
) {
    const unsigned int u0 = (unsigned int)b & 0x7Fu;
    const unsigned int u = u0 > 126u ? 126u : u0;
    const float normal = __builtin_bit_cast(float, (u + 960u) << 20);
    const float sub = (float)u * 0.001953125f;
    return u < 8u ? sub : normal;
}

__device__ __forceinline__ void e8_decode_index_grouped_gfx1201(
    unsigned int idx,
    float p[8]
) {
    const unsigned int lo = idx & 0x0F0F0F0Fu;
    const unsigned int hi = (idx >> 4) & 0x0F0F0F0Fu;
    const unsigned int e7 =
        ((idx >> 27) & 14u) | (unsigned int)(__popc(idx & 0x01111111u) & 1);
    const float off = (idx & 0x80000000u) ? -6.5f : -7.0f;
    const uchar4 lb = __builtin_bit_cast(uchar4, lo);
    const uchar4 hb = __builtin_bit_cast(uchar4, hi);
    p[0] = (float)lb.x + off;
    p[1] = (float)hb.x + off;
    p[2] = (float)lb.y + off;
    p[3] = (float)hb.y + off;
    p[4] = (float)lb.z + off;
    p[5] = (float)hb.z + off;
    p[6] = (float)lb.w + off;
    p[7] = (float)e7 + off;
}

__launch_bounds__(32)
extern "C" __global__ void gemv_mfp4g32_e8_soa_grouped_gfx1201(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int G,
    int M,
    int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M || group >= G) return;
    const int tid = threadIdx.x;

    const int n_blocks = K / 32;
    const int groups_per_row = K / 256;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const long long global_row = (long long)group * M + row;
    const char* hdr = A + global_row * row_total;
    const float* x_group = x + (long long)group * K;

    const float row_scale_f =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const char* scale_base = hdr + 16;
    const char* cw_base = hdr + 16 + scale_padded;
    const int block_in_group = tid >> 2;
    const int cw_slot = tid & 3;
    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail = groups_per_row & 1;

    for (int p = 0; p < pairs; p++) {
        const int g = p << 1;
        const int b0 = g * 8 + block_in_group;
        const int b1 = b0 + 8;
        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32_grouped_gfx1201(
            *(const unsigned char*)(scale_base + b0)) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32_grouped_gfx1201(
            *(const unsigned char*)(scale_base + b1)) * 0.88f;
        const unsigned int ew0 =
            *(const unsigned int*)(cw_base + (long long)b0 * 16 + cw_slot * 4);
        const unsigned int ew1 =
            *(const unsigned int*)(cw_base + (long long)b1 * 16 + cw_slot * 4);
        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;

        float pe[8];
        e8_decode_index_grouped_gfx1201(ew0, pe);
        acc0 += ssc0*pe[0]*x_group[base0  ] + ssc0*pe[1]*x_group[base0+1]
              + ssc0*pe[2]*x_group[base0+2] + ssc0*pe[3]*x_group[base0+3]
              + ssc0*pe[4]*x_group[base0+4] + ssc0*pe[5]*x_group[base0+5]
              + ssc0*pe[6]*x_group[base0+6] + ssc0*pe[7]*x_group[base0+7];

        e8_decode_index_grouped_gfx1201(ew1, pe);
        acc1 += ssc1*pe[0]*x_group[base1  ] + ssc1*pe[1]*x_group[base1+1]
              + ssc1*pe[2]*x_group[base1+2] + ssc1*pe[3]*x_group[base1+3]
              + ssc1*pe[4]*x_group[base1+4] + ssc1*pe[5]*x_group[base1+5]
              + ssc1*pe[6]*x_group[base1+6] + ssc1*pe[7]*x_group[base1+7];
    }

    if (tail) {
        const int g = pairs << 1;
        const int b = g * 8 + block_in_group;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32_grouped_gfx1201(
            *(const unsigned char*)(scale_base + b)) * 0.88f;
        const unsigned int ew =
            *(const unsigned int*)(cw_base + (long long)b * 16 + cw_slot * 4);
        const int base = g * 256 + tid * 8;
        float pe[8];
        e8_decode_index_grouped_gfx1201(ew, pe);
        acc0 += ssc*pe[0]*x_group[base  ] + ssc*pe[1]*x_group[base+1]
              + ssc*pe[2]*x_group[base+2] + ssc*pe[3]*x_group[base+3]
              + ssc*pe[4]*x_group[base+4] + ssc*pe[5]*x_group[base+5]
              + ssc*pe[6]*x_group[base+6] + ssc*pe[7]*x_group[base+7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[global_row] = acc;
}
