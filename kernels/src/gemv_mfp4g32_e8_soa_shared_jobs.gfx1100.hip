// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1100 same-shape, shared-input MFP4-E8 SoA GEMV.
//
// One 2-D launch evaluates two or three projections with identical [M,K]
// shapes and a shared x[K]. Each wave preserves the incumbent per-row
// accumulation and reduction order.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float cvt_e4m3_scale_to_f32_shared(unsigned char b) {
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

__device__ __forceinline__ void e8_decode_index_shared(unsigned int idx,
                                                        float p[8]) {
  const int coset = (int)((idx >> 31) & 1u);
  unsigned int e[8];
  unsigned int sum_low = 0u;
  for (int i = 0; i < 7; i++) {
    e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
    sum_low += e[i];
  }
  const unsigned int e7_high = (idx >> 28) & 0x7u;
  const unsigned int e7_even = e7_high << 1;
  e[7] = e7_even | ((sum_low + e7_even) & 1u);
  for (int i = 0; i < 8; i++) {
    const float coordinate = (float)((int)e[i] - 7);
    p[i] = coset ? coordinate + 0.5f : coordinate;
  }
}

__device__ __forceinline__ float
e8_dot_row_shared(const char *__restrict__ row_base,
                  const float *__restrict__ x, int K) {
  const int tid = threadIdx.x;
  const int n_blocks = K / 32;
  const int groups_per_row = K / 256;
  const int scale_padded = ((n_blocks + 15) >> 4) << 4;
  const float row_scale =
      (float)__builtin_bit_cast(_Float16, *(const unsigned short *)row_base);
  const char *scale_base = row_base + 16;
  const char *code_base = row_base + 16 + scale_padded;
  const int block_in_group = tid >> 2;
  const int code_slot = tid & 3;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  const int pairs = groups_per_row >> 1;
  for (int pair = 0; pair < pairs; pair++) {
    const int group0 = pair << 1;
    const int block0 = group0 * 8 + block_in_group;
    const int block1 = block0 + 8;
    const float scale0 = row_scale *
                         cvt_e4m3_scale_to_f32_shared(
                             *(const unsigned char *)(scale_base + block0)) *
                         0.88f;
    const float scale1 = row_scale *
                         cvt_e4m3_scale_to_f32_shared(
                             *(const unsigned char *)(scale_base + block1)) *
                         0.88f;
    const unsigned int code0 =
        *(const unsigned int *)(code_base + (long long)block0 * 16 +
                                code_slot * 4);
    const unsigned int code1 =
        *(const unsigned int *)(code_base + (long long)block1 * 16 +
                                code_slot * 4);
    const int base0 = group0 * 256 + tid * 8;
    const int base1 = base0 + 256;

    float points[8];
    e8_decode_index_shared(code0, points);
    acc0 +=
        scale0 * points[0] * x[base0] + scale0 * points[1] * x[base0 + 1] +
        scale0 * points[2] * x[base0 + 2] + scale0 * points[3] * x[base0 + 3] +
        scale0 * points[4] * x[base0 + 4] + scale0 * points[5] * x[base0 + 5] +
        scale0 * points[6] * x[base0 + 6] + scale0 * points[7] * x[base0 + 7];

    e8_decode_index_shared(code1, points);
    acc1 +=
        scale1 * points[0] * x[base1] + scale1 * points[1] * x[base1 + 1] +
        scale1 * points[2] * x[base1 + 2] + scale1 * points[3] * x[base1 + 3] +
        scale1 * points[4] * x[base1 + 4] + scale1 * points[5] * x[base1 + 5] +
        scale1 * points[6] * x[base1 + 6] + scale1 * points[7] * x[base1 + 7];
  }

  if (groups_per_row & 1) {
    const int group0 = pairs << 1;
    const int block = group0 * 8 + block_in_group;
    const float scale = row_scale *
                        cvt_e4m3_scale_to_f32_shared(
                            *(const unsigned char *)(scale_base + block)) *
                        0.88f;
    const unsigned int code =
        *(const unsigned int *)(code_base + (long long)block * 16 +
                                code_slot * 4);
    const int base = group0 * 256 + tid * 8;
    float points[8];
    e8_decode_index_shared(code, points);
    acc0 += scale * points[0] * x[base] + scale * points[1] * x[base + 1] +
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
  return acc;
}

__launch_bounds__(32, 7) extern "C" __global__
    void gemv_mfp4g32_e8_soa_shared_jobs_gfx1100(
        const char *__restrict__ A0, const char *__restrict__ A1,
        const char *__restrict__ A2, const float *__restrict__ x,
        float *__restrict__ y0, float *__restrict__ y1,
        float *__restrict__ y2, int jobs, int M, int K) {
  const int row = blockIdx.x;
  const int job = blockIdx.y;
  if (row >= M || job >= jobs)
    return;
  const char *weight = job == 0 ? A0 : (job == 1 ? A1 : A2);
  float *output = job == 0 ? y0 : (job == 1 ? y1 : y2);
  const int n_blocks = K / 32;
  const int scale_padded = ((n_blocks + 15) >> 4) << 4;
  const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
  const float acc =
      e8_dot_row_shared(weight + (long long)row * row_total, x, K);
  if (threadIdx.x == 0)
    output[row] = acc;
}
