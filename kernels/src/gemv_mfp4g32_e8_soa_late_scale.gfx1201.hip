// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 late-scale E8 GEMV screen.
//
// Concatenated after gemv_mfp4g32_e8_soa.gfx1201.hip. Each E8 coordinate
// dot is formed before the block scale is applied, replacing eight repeated
// scale multiplies with one. This is mathematically equivalent but changes
// FP32 grouping, so it remains experimental until full product validation.

extern "C" __global__ __launch_bounds__(32)
void gemv_mfp4g32_e8_soa_late_scale_gfx1201(
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
    const int pairs = groups_per_row >> 1;
    const int tail = groups_per_row & 1;
    for (int pair = 0; pair < pairs; ++pair) {
        const int group = pair << 1;
        const int b0 = group * 8 + block_in_group;
        const int b1 = b0 + 8;
        const float s0 = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + b0)) * 0.88f;
        const float s1 = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + b1)) * 0.88f;
        const unsigned int e0 = *(const unsigned int*)(
            code + (long long)b0 * 16 + cw_slot * 4);
        const unsigned int e1 = *(const unsigned int*)(
            code + (long long)b1 * 16 + cw_slot * 4);
        const int base0 = group * 256 + tid * 8;
        const int base1 = base0 + 256;

        float pe[8];
        e8_decode_index_gfx1201(e0, pe);
        const float dot0 =
            pe[0]*x[base0] + pe[1]*x[base0+1]
          + pe[2]*x[base0+2] + pe[3]*x[base0+3]
          + pe[4]*x[base0+4] + pe[5]*x[base0+5]
          + pe[6]*x[base0+6] + pe[7]*x[base0+7];
        acc0 += s0 * dot0;

        e8_decode_index_gfx1201(e1, pe);
        const float dot1 =
            pe[0]*x[base1] + pe[1]*x[base1+1]
          + pe[2]*x[base1+2] + pe[3]*x[base1+3]
          + pe[4]*x[base1+4] + pe[5]*x[base1+5]
          + pe[6]*x[base1+6] + pe[7]*x[base1+7];
        acc1 += s1 * dot1;
    }

    if (tail) {
        const int group = pairs << 1;
        const int block = group * 8 + block_in_group;
        const float scale = row_scale * cvt_e4m3_scale_to_f32_gfx1201(
            *(const unsigned char*)(scales + block)) * 0.88f;
        const unsigned int encoded = *(const unsigned int*)(
            code + (long long)block * 16 + cw_slot * 4);
        const int base = group * 256 + tid * 8;
        float pe[8];
        e8_decode_index_gfx1201(encoded, pe);
        const float dot =
            pe[0]*x[base] + pe[1]*x[base+1]
          + pe[2]*x[base+2] + pe[3]*x[base+3]
          + pe[4]*x[base+4] + pe[5]*x[base+5]
          + pe[6]*x[base+6] + pe[7]*x[base+7];
        acc0 += scale * dot;
    }

    float sum = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
