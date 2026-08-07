// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 MFP4G32E8SOA decode screen with LDS X reuse.

#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ float cvt_e4m3_scale_to_f32_gfx1201_ldsx(unsigned char b) {
    const unsigned int u0 = (unsigned int)b & 0x7Fu;
    const unsigned int u = u0 > 126u ? 126u : u0;
    const float normal = __builtin_bit_cast(float, (u + 960u) << 20);
    const float sub = (float)u * 0.001953125f;
    return u < 8u ? sub : normal;
}

__device__ __forceinline__ void e8_decode_index_gfx1201_ldsx(unsigned int idx, float p[8]) {
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

__device__ __forceinline__ void project_row_gfx1201_ldsx(
    const char* __restrict__ weights,
    const float* __restrict__ x_lds,
    float* __restrict__ y,
    int row,
    int k,
    int lane
) {
    const int n_blocks = k / 32;
    const int groups_per_row = k / 256;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = weights + (long long)row * row_total;
    const float row_scale_f =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const char* scale_base = hdr + 16;
    const char* cw_base = hdr + 16 + scale_padded;

    const int block_in_group = lane >> 2;
    const int cw_slot = lane & 3;
    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail = groups_per_row & 1;

    for (int pair = 0; pair < pairs; ++pair) {
        const int group = pair << 1;
        const int block0 = group * 8 + block_in_group;
        const int block1 = block0 + 8;
        const float scale0 = row_scale_f * cvt_e4m3_scale_to_f32_gfx1201_ldsx(
            *(const unsigned char*)(scale_base + block0)) * 0.88f;
        const float scale1 = row_scale_f * cvt_e4m3_scale_to_f32_gfx1201_ldsx(
            *(const unsigned char*)(scale_base + block1)) * 0.88f;
        const unsigned int code0 = *(const unsigned int*)(
            cw_base + (long long)block0 * 16 + cw_slot * 4);
        const unsigned int code1 = *(const unsigned int*)(
            cw_base + (long long)block1 * 16 + cw_slot * 4);
        const int base0 = group * 256 + lane * 8;
        const int base1 = base0 + 256;
        float decoded[8];
        e8_decode_index_gfx1201_ldsx(code0, decoded);
        acc0 += scale0*decoded[0]*x_lds[base0  ] + scale0*decoded[1]*x_lds[base0+1]
              + scale0*decoded[2]*x_lds[base0+2] + scale0*decoded[3]*x_lds[base0+3]
              + scale0*decoded[4]*x_lds[base0+4] + scale0*decoded[5]*x_lds[base0+5]
              + scale0*decoded[6]*x_lds[base0+6] + scale0*decoded[7]*x_lds[base0+7];
        e8_decode_index_gfx1201_ldsx(code1, decoded);
        acc1 += scale1*decoded[0]*x_lds[base1  ] + scale1*decoded[1]*x_lds[base1+1]
              + scale1*decoded[2]*x_lds[base1+2] + scale1*decoded[3]*x_lds[base1+3]
              + scale1*decoded[4]*x_lds[base1+4] + scale1*decoded[5]*x_lds[base1+5]
              + scale1*decoded[6]*x_lds[base1+6] + scale1*decoded[7]*x_lds[base1+7];
    }
    if (tail) {
        const int group = pairs << 1;
        const int block = group * 8 + block_in_group;
        const float scale = row_scale_f * cvt_e4m3_scale_to_f32_gfx1201_ldsx(
            *(const unsigned char*)(scale_base + block)) * 0.88f;
        const unsigned int code = *(const unsigned int*)(
            cw_base + (long long)block * 16 + cw_slot * 4);
        const int base = group * 256 + lane * 8;
        float decoded[8];
        e8_decode_index_gfx1201_ldsx(code, decoded);
        acc0 += scale*decoded[0]*x_lds[base  ] + scale*decoded[1]*x_lds[base+1]
              + scale*decoded[2]*x_lds[base+2] + scale*decoded[3]*x_lds[base+3]
              + scale*decoded[4]*x_lds[base+4] + scale*decoded[5]*x_lds[base+5]
              + scale*decoded[6]*x_lds[base+6] + scale*decoded[7]*x_lds[base+7];
    }
    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, 32);
    }
    if (lane == 0) y[row] = acc;
}

} // namespace

extern "C" __global__ __launch_bounds__(128, 2)
void gemv_mfp4g32_e8_soa_ldsx4_gfx1201(
    const char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k
) {
    extern __shared__ __align__(16) float x_lds[];
    const int t = threadIdx.x;
    for (int i = t; i < k; i += 128) {
        x_lds[i] = x[i];
    }
    __syncthreads();
    const int lane = t & 31;
    const int row = (int)blockIdx.x * 4 + (t >> 5);
    if (row < m) {
        project_row_gfx1201_ldsx(weights, x_lds, y, row, k, lane);
    }
}
