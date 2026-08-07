// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1100 MFP4-E8 SoA GEMV with quad-broadcast scales.
//
// The SoA row stores one E4M3 scale per 32-weight block, but the incumbent
// kernel loads and converts that same byte independently in all four lanes
// which consume the block's four codewords.  This variant performs that work
// once per quad and broadcasts the resulting f32.  Row scale is likewise
// loaded once per wave.  Codeword decode, multiply order, accumulation, and
// wave reduction remain identical to the generic kernel.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float
cvt_e4m3_scale_to_f32_broadcast(unsigned char b) {
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

__device__ __forceinline__ void e8_decode_index_broadcast(unsigned int idx,
                                                           float p[8]) {
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

__device__ __forceinline__ float quad_scale(const char *__restrict__ scales,
                                             int block, int tid) {
  float value = 0.0f;
  if ((tid & 3) == 0)
    value = cvt_e4m3_scale_to_f32_broadcast(
        *(const unsigned char *)(scales + block));
  return __shfl(value, tid & ~3, 32);
}

__launch_bounds__(32) extern "C" __global__
    void gemv_mfp4g32_e8_soa_scale_broadcast_gfx1100(
        const char *__restrict__ A, const float *__restrict__ x,
        float *__restrict__ y, int M, int K) {
  const int row = blockIdx.x;
  if (row >= M)
    return;
  const int tid = threadIdx.x;

  const int n_blocks = K / 32;
  const int groups_per_row = K / 256;
  const int scale_padded = ((n_blocks + 15) >> 4) << 4;
  const long long row_total =
      16LL + scale_padded + (long long)n_blocks * 16;
  const char *hdr = A + (long long)row * row_total;

  float row_scale = 0.0f;
  if (tid == 0)
    row_scale =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short *)hdr);
  row_scale = __shfl(row_scale, 0, 32);

  const char *scale_base = hdr + 16;
  const char *cw_base = hdr + 16 + scale_padded;
  const int block_in_group = tid >> 2;
  const int cw_slot = tid & 3;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  const int pairs = groups_per_row >> 1;
  const int tail = groups_per_row & 1;

  for (int pair = 0; pair < pairs; pair++) {
    const int group = pair << 1;
    const int block0 = group * 8 + block_in_group;
    const int block1 = block0 + 8;
    const float scale0 =
        row_scale * quad_scale(scale_base, block0, tid) * 0.88f;
    const float scale1 =
        row_scale * quad_scale(scale_base, block1, tid) * 0.88f;
    const unsigned int code0 = *(const unsigned int *)(
        cw_base + (long long)block0 * 16 + cw_slot * 4);
    const unsigned int code1 = *(const unsigned int *)(
        cw_base + (long long)block1 * 16 + cw_slot * 4);
    const int base0 = group * 256 + tid * 8;
    const int base1 = base0 + 256;

    float points[8];
    e8_decode_index_broadcast(code0, points);
    acc0 += scale0 * points[0] * x[base0] +
            scale0 * points[1] * x[base0 + 1] +
            scale0 * points[2] * x[base0 + 2] +
            scale0 * points[3] * x[base0 + 3] +
            scale0 * points[4] * x[base0 + 4] +
            scale0 * points[5] * x[base0 + 5] +
            scale0 * points[6] * x[base0 + 6] +
            scale0 * points[7] * x[base0 + 7];

    e8_decode_index_broadcast(code1, points);
    acc1 += scale1 * points[0] * x[base1] +
            scale1 * points[1] * x[base1 + 1] +
            scale1 * points[2] * x[base1 + 2] +
            scale1 * points[3] * x[base1 + 3] +
            scale1 * points[4] * x[base1 + 4] +
            scale1 * points[5] * x[base1 + 5] +
            scale1 * points[6] * x[base1 + 6] +
            scale1 * points[7] * x[base1 + 7];
  }

  if (tail) {
    const int group = pairs << 1;
    const int block = group * 8 + block_in_group;
    const float scale =
        row_scale * quad_scale(scale_base, block, tid) * 0.88f;
    const unsigned int code = *(const unsigned int *)(
        cw_base + (long long)block * 16 + cw_slot * 4);
    const int base = group * 256 + tid * 8;
    float points[8];
    e8_decode_index_broadcast(code, points);
    acc0 += scale * points[0] * x[base] +
            scale * points[1] * x[base + 1] +
            scale * points[2] * x[base + 2] +
            scale * points[3] * x[base + 3] +
            scale * points[4] * x[base + 4] +
            scale * points[5] * x[base + 5] +
            scale * points[6] * x[base + 6] +
            scale * points[7] * x[base + 7];
  }

  float acc = acc0 + acc1;
  for (int offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down(acc, offset);
  if (tid == 0)
    y[row] = acc;
}
