// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 two-projection E8 GEMV screen.
//
// This file is concatenated after gemv_mfp4g32_e8_soa.gfx1201.hip.  Two
// matrices consume the same activation and retain the incumbent two-chain
// accumulation and wave32 reduction independently.  The only intended saving
// is loading each activation tile once for both projections.

extern "C" __global__ __launch_bounds__(32)
void gemv_mfp4g32_e8_soa_shared_pair_gfx1201(
    const char* __restrict__ A0,
    const char* __restrict__ A1,
    const float* __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    int M0,
    int M1,
    int K
) {
    const int row = (int)blockIdx.x;
    const bool active0 = row < M0;
    const bool active1 = row < M1;
    if (!active0 && !active1) return;

    const int tid = (int)threadIdx.x;
    const int n_blocks = K / 32;
    const int groups_per_row = K / 256;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total =
        16LL + scale_padded + (long long)n_blocks * 16;

    const char* hdr0 = A0 + (long long)row * row_total;
    const char* hdr1 = A1 + (long long)row * row_total;
    float row_scale0 = 0.0f;
    float row_scale1 = 0.0f;
    if (active0) {
        row_scale0 = (float)__builtin_bit_cast(
            _Float16, *(const unsigned short*)hdr0);
    }
    if (active1) {
        row_scale1 = (float)__builtin_bit_cast(
            _Float16, *(const unsigned short*)hdr1);
    }
    const char* scale0 = hdr0 + 16;
    const char* scale1 = hdr1 + 16;
    const char* code0 = scale0 + scale_padded;
    const char* code1 = scale1 + scale_padded;

    const int block_in_group = tid >> 2;
    const int cw_slot = tid & 3;
    float a00 = 0.0f, a01 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail = groups_per_row & 1;

    for (int p = 0; p < pairs; ++p) {
        const int g = p << 1;
        const int b0 = g * 8 + block_in_group;
        const int b1 = b0 + 8;
        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;

        // Explicit vector values keep the activation loads common to both
        // projection arms.  Their component order matches the incumbent
        // scalar expression exactly.
        const float4 x00 = *(const float4*)(x + base0);
        const float4 x01 = *(const float4*)(x + base0 + 4);
        const float4 x10 = *(const float4*)(x + base1);
        const float4 x11 = *(const float4*)(x + base1 + 4);

        if (active0) {
            const float s0 = row_scale0 * cvt_e4m3_scale_to_f32_gfx1201(
                *(const unsigned char*)(scale0 + b0)) * 0.88f;
            const float s1 = row_scale0 * cvt_e4m3_scale_to_f32_gfx1201(
                *(const unsigned char*)(scale0 + b1)) * 0.88f;
            const unsigned int e0 = *(const unsigned int*)(
                code0 + (long long)b0 * 16 + cw_slot * 4);
            const unsigned int e1 = *(const unsigned int*)(
                code0 + (long long)b1 * 16 + cw_slot * 4);
            float pe[8];
            e8_decode_index_gfx1201(e0, pe);
            a00 += s0*pe[0]*x00.x + s0*pe[1]*x00.y
                 + s0*pe[2]*x00.z + s0*pe[3]*x00.w
                 + s0*pe[4]*x01.x + s0*pe[5]*x01.y
                 + s0*pe[6]*x01.z + s0*pe[7]*x01.w;
            e8_decode_index_gfx1201(e1, pe);
            a01 += s1*pe[0]*x10.x + s1*pe[1]*x10.y
                 + s1*pe[2]*x10.z + s1*pe[3]*x10.w
                 + s1*pe[4]*x11.x + s1*pe[5]*x11.y
                 + s1*pe[6]*x11.z + s1*pe[7]*x11.w;
        }

        if (active1) {
            const float s0 = row_scale1 * cvt_e4m3_scale_to_f32_gfx1201(
                *(const unsigned char*)(scale1 + b0)) * 0.88f;
            const float s1 = row_scale1 * cvt_e4m3_scale_to_f32_gfx1201(
                *(const unsigned char*)(scale1 + b1)) * 0.88f;
            const unsigned int e0 = *(const unsigned int*)(
                code1 + (long long)b0 * 16 + cw_slot * 4);
            const unsigned int e1 = *(const unsigned int*)(
                code1 + (long long)b1 * 16 + cw_slot * 4);
            float pe[8];
            e8_decode_index_gfx1201(e0, pe);
            a10 += s0*pe[0]*x00.x + s0*pe[1]*x00.y
                 + s0*pe[2]*x00.z + s0*pe[3]*x00.w
                 + s0*pe[4]*x01.x + s0*pe[5]*x01.y
                 + s0*pe[6]*x01.z + s0*pe[7]*x01.w;
            e8_decode_index_gfx1201(e1, pe);
            a11 += s1*pe[0]*x10.x + s1*pe[1]*x10.y
                 + s1*pe[2]*x10.z + s1*pe[3]*x10.w
                 + s1*pe[4]*x11.x + s1*pe[5]*x11.y
                 + s1*pe[6]*x11.z + s1*pe[7]*x11.w;
        }
    }

    if (tail) {
        const int g = pairs << 1;
        const int b = g * 8 + block_in_group;
        const int base = g * 256 + tid * 8;
        const float4 x0 = *(const float4*)(x + base);
        const float4 x1 = *(const float4*)(x + base + 4);
        if (active0) {
            const float scale = row_scale0 * cvt_e4m3_scale_to_f32_gfx1201(
                *(const unsigned char*)(scale0 + b)) * 0.88f;
            const unsigned int e = *(const unsigned int*)(
                code0 + (long long)b * 16 + cw_slot * 4);
            float pe[8];
            e8_decode_index_gfx1201(e, pe);
            a00 += scale*pe[0]*x0.x + scale*pe[1]*x0.y
                 + scale*pe[2]*x0.z + scale*pe[3]*x0.w
                 + scale*pe[4]*x1.x + scale*pe[5]*x1.y
                 + scale*pe[6]*x1.z + scale*pe[7]*x1.w;
        }
        if (active1) {
            const float scale = row_scale1 * cvt_e4m3_scale_to_f32_gfx1201(
                *(const unsigned char*)(scale1 + b)) * 0.88f;
            const unsigned int e = *(const unsigned int*)(
                code1 + (long long)b * 16 + cw_slot * 4);
            float pe[8];
            e8_decode_index_gfx1201(e, pe);
            a10 += scale*pe[0]*x0.x + scale*pe[1]*x0.y
                 + scale*pe[2]*x0.z + scale*pe[3]*x0.w
                 + scale*pe[4]*x1.x + scale*pe[5]*x1.y
                 + scale*pe[6]*x1.z + scale*pe[7]*x1.w;
        }
    }

    if (active0) {
        float sum = a00 + a01;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (tid == 0) y0[row] = sum;
    }
    if (active1) {
        float sum = a10 + a11;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (tid == 0) y1[row] = sum;
    }
}
