// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 four-group prefetch E8 GEMV screen.
//
// Concatenated after gemv_mfp4g32_e8_soa.gfx1201.hip.  Four adjacent groups
// are loaded before decode, but even groups still accumulate into acc0 and odd
// groups into acc1 in exactly the incumbent order.  The final add and wave32
// reduction are unchanged.

extern "C" __global__ __launch_bounds__(32)
void gemv_mfp4g32_e8_soa_prefetch4_gfx1201(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;

    const int n_blocks = K / 32;
    const int groups_per_row = K / 256;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total =
        16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = A + (long long)row * row_total;
    const float row_scale = (float)__builtin_bit_cast(
        _Float16, *(const unsigned short*)hdr);
    const char* scales = hdr + 16;
    const char* code = scales + scale_padded;
    const int block_in_group = tid >> 2;
    const int cw_slot = tid & 3;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int quads = groups_per_row >> 2;
    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const int b0 = g * 8 + block_in_group;
        const int b1 = b0 + 8;
        const int b2 = b0 + 16;
        const int b3 = b0 + 24;

        const float s0 = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + b0)) * 0.88f;
        const float s1 = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + b1)) * 0.88f;
        const float s2 = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + b2)) * 0.88f;
        const float s3 = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + b3)) * 0.88f;
        const unsigned int e0 = *(const unsigned int*)(
            code + (long long)b0 * 16 + cw_slot * 4);
        const unsigned int e1 = *(const unsigned int*)(
            code + (long long)b1 * 16 + cw_slot * 4);
        const unsigned int e2 = *(const unsigned int*)(
            code + (long long)b2 * 16 + cw_slot * 4);
        const unsigned int e3 = *(const unsigned int*)(
            code + (long long)b3 * 16 + cw_slot * 4);

        const int base0 = g * 256 + tid * 8;
        const float4 x00 = *(const float4*)(x + base0);
        const float4 x01 = *(const float4*)(x + base0 + 4);
        const float4 x10 = *(const float4*)(x + base0 + 256);
        const float4 x11 = *(const float4*)(x + base0 + 260);
        const float4 x20 = *(const float4*)(x + base0 + 512);
        const float4 x21 = *(const float4*)(x + base0 + 516);
        const float4 x30 = *(const float4*)(x + base0 + 768);
        const float4 x31 = *(const float4*)(x + base0 + 772);

        float pe[8];
        e8_decode_index_gfx1201(e0, pe);
        acc0 += s0*pe[0]*x00.x + s0*pe[1]*x00.y
              + s0*pe[2]*x00.z + s0*pe[3]*x00.w
              + s0*pe[4]*x01.x + s0*pe[5]*x01.y
              + s0*pe[6]*x01.z + s0*pe[7]*x01.w;
        e8_decode_index_gfx1201(e1, pe);
        acc1 += s1*pe[0]*x10.x + s1*pe[1]*x10.y
              + s1*pe[2]*x10.z + s1*pe[3]*x10.w
              + s1*pe[4]*x11.x + s1*pe[5]*x11.y
              + s1*pe[6]*x11.z + s1*pe[7]*x11.w;
        e8_decode_index_gfx1201(e2, pe);
        acc0 += s2*pe[0]*x20.x + s2*pe[1]*x20.y
              + s2*pe[2]*x20.z + s2*pe[3]*x20.w
              + s2*pe[4]*x21.x + s2*pe[5]*x21.y
              + s2*pe[6]*x21.z + s2*pe[7]*x21.w;
        e8_decode_index_gfx1201(e3, pe);
        acc1 += s3*pe[0]*x30.x + s3*pe[1]*x30.y
              + s3*pe[2]*x30.z + s3*pe[3]*x30.w
              + s3*pe[4]*x31.x + s3*pe[5]*x31.y
              + s3*pe[6]*x31.z + s3*pe[7]*x31.w;
    }

    // The remainder begins on an even group. Alternating destination chains
    // preserves the incumbent group-to-accumulator mapping for K values that
    // are not multiples of 1024.
    for (int group = quads << 2; group < groups_per_row; ++group) {
        const int block = group * 8 + block_in_group;
        const float scale = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + block)) * 0.88f;
        const unsigned int encoded = *(const unsigned int*)(
            code + (long long)block * 16 + cw_slot * 4);
        const int base = group * 256 + tid * 8;
        const float4 x0 = *(const float4*)(x + base);
        const float4 x1 = *(const float4*)(x + base + 4);
        float pe[8];
        e8_decode_index_gfx1201(encoded, pe);
        const float contribution =
            scale*pe[0]*x0.x + scale*pe[1]*x0.y
          + scale*pe[2]*x0.z + scale*pe[3]*x0.w
          + scale*pe[4]*x1.x + scale*pe[5]*x1.y
          + scale*pe[6]*x1.z + scale*pe[7]*x1.w;
        if (group & 1) {
            acc1 += contribution;
        } else {
            acc0 += contribution;
        }
    }

    float sum = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
