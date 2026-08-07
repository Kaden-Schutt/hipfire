// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1100 MFP4-E8 SoA GEMV, two-pair VMEM window.
//
// `gfx12_weight_cache_policy.inc` is concatenated before this source. This
// candidate preserves the accepted raw-buffer kernel's two accumulator chains,
// per-chain update order, E8 decode, multiply order, and wave reduction. It
// only issues the next pair's scale/code loads before consuming the current
// pair so gfx1100 can expose more independent VMEM work.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float
cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(unsigned char b) {
  const int exp = (b >> 3) & 0xF;
  const unsigned mant = (unsigned)(b & 0x7);
  if (exp == 0)
    return 0.015625f * ((float)mant) * 0.125f;
  if (exp == 0xF && mant == 7u)
    return 448.0f;
  const float pow2 =
      __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
  return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void
e8_decode_index_soa_prefetch4_gfx1100(unsigned int idx, float p[8]) {
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

#define E8_PREFETCH4_DOT(code, scale, base, acc)                              \
  do {                                                                         \
    float points[8];                                                           \
    e8_decode_index_soa_prefetch4_gfx1100((code), points);                    \
    (acc) += (scale) * points[0] * x[(base)] +                                \
             (scale) * points[1] * x[(base) + 1] +                            \
             (scale) * points[2] * x[(base) + 2] +                            \
             (scale) * points[3] * x[(base) + 3] +                            \
             (scale) * points[4] * x[(base) + 4] +                            \
             (scale) * points[5] * x[(base) + 5] +                            \
             (scale) * points[6] * x[(base) + 6] +                            \
             (scale) * points[7] * x[(base) + 7];                             \
  } while (0)

__launch_bounds__(32) extern "C" __global__
    void gemv_mfp4g32_e8_soa_prefetch4_buffer_gfx1100(
        const char *__restrict__ A, const float *__restrict__ x,
        float *__restrict__ y, int M, int K) {
  const int row = blockIdx.x;
  if (row >= M)
    return;
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
  const float row_scale = (float)__builtin_bit_cast(
      _Float16, (unsigned short)(row_header & 0xffffu));
  const unsigned int scale_offset = row_offset + 16u;
  const unsigned int code_offset =
      scale_offset + (unsigned int)scale_padded;
  const int block_in_group = tid >> 2;
  const int code_slot = tid & 3;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  const int pairs = groups_per_row >> 1;
  int pair = 0;
  for (; pair + 1 < pairs; pair += 2) {
    const int group0 = pair << 1;
    const int block0 = group0 * 8 + block_in_group;
    const int block1 = block0 + 8;
    const int block2 = block0 + 16;
    const int block3 = block0 + 24;

    const unsigned char scale_byte0 =
        (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block0);
    const unsigned char scale_byte1 =
        (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block1);
    const unsigned char scale_byte2 =
        (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block2);
    const unsigned char scale_byte3 =
        (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block3);
    const unsigned int code0 = HIPFIRE_WEIGHT_LOAD_U32(
        weight_rsrc, code_offset + (unsigned int)block0 * 16u +
                         (unsigned int)code_slot * 4u);
    const unsigned int code1 = HIPFIRE_WEIGHT_LOAD_U32(
        weight_rsrc, code_offset + (unsigned int)block1 * 16u +
                         (unsigned int)code_slot * 4u);
    const unsigned int code2 = HIPFIRE_WEIGHT_LOAD_U32(
        weight_rsrc, code_offset + (unsigned int)block2 * 16u +
                         (unsigned int)code_slot * 4u);
    const unsigned int code3 = HIPFIRE_WEIGHT_LOAD_U32(
        weight_rsrc, code_offset + (unsigned int)block3 * 16u +
                         (unsigned int)code_slot * 4u);
    const float scale0 =
        row_scale * cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(scale_byte0) *
        0.88f;
    const float scale1 =
        row_scale * cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(scale_byte1) *
        0.88f;
    const float scale2 =
        row_scale * cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(scale_byte2) *
        0.88f;
    const float scale3 =
        row_scale * cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(scale_byte3) *
        0.88f;
    const int base0 = group0 * 256 + tid * 8;
    const int base1 = base0 + 256;
    const int base2 = base0 + 512;
    const int base3 = base0 + 768;

    // Preserve the accepted kernel's exact accumulator update order:
    // even groups update acc0 and odd groups update acc1 in ascending order.
    E8_PREFETCH4_DOT(code0, scale0, base0, acc0);
    E8_PREFETCH4_DOT(code1, scale1, base1, acc1);
    E8_PREFETCH4_DOT(code2, scale2, base2, acc0);
    E8_PREFETCH4_DOT(code3, scale3, base3, acc1);
  }

  if (pair < pairs) {
    const int group0 = pair << 1;
    const int block0 = group0 * 8 + block_in_group;
    const int block1 = block0 + 8;
    const unsigned char scale_byte0 =
        (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block0);
    const unsigned char scale_byte1 =
        (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block1);
    const float scale0 =
        row_scale * cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(scale_byte0) *
        0.88f;
    const float scale1 =
        row_scale * cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(scale_byte1) *
        0.88f;
    const unsigned int code0 = HIPFIRE_WEIGHT_LOAD_U32(
        weight_rsrc, code_offset + (unsigned int)block0 * 16u +
                         (unsigned int)code_slot * 4u);
    const unsigned int code1 = HIPFIRE_WEIGHT_LOAD_U32(
        weight_rsrc, code_offset + (unsigned int)block1 * 16u +
                         (unsigned int)code_slot * 4u);
    const int base0 = group0 * 256 + tid * 8;
    E8_PREFETCH4_DOT(code0, scale0, base0, acc0);
    E8_PREFETCH4_DOT(code1, scale1, base0 + 256, acc1);
  }

  if (groups_per_row & 1) {
    const int group = pairs << 1;
    const int block = group * 8 + block_in_group;
    const unsigned char scale_byte =
        (unsigned char)HIPFIRE_WEIGHT_LOAD_U32(
            weight_rsrc, scale_offset + (unsigned int)block);
    const float scale =
        row_scale * cvt_e4m3_scale_to_f32_soa_prefetch4_gfx1100(scale_byte) *
        0.88f;
    const unsigned int code = HIPFIRE_WEIGHT_LOAD_U32(
        weight_rsrc, code_offset + (unsigned int)block * 16u +
                         (unsigned int)code_slot * 4u);
    const int base = group * 256 + tid * 8;
    E8_PREFETCH4_DOT(code, scale, base, acc0);
  }

  float acc = acc0 + acc1;
  for (int offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down(acc, offset);
  if (tid == 0)
    y[row] = acc;
}

#undef E8_PREFETCH4_DOT
