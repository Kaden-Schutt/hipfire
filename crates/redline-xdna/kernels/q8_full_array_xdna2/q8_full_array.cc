// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Kaden Schutt

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int kM = 64;
constexpr int kK = 64;
constexpr int kN = 64;
constexpr int kQ8BlockElements = 32;
constexpr int kQ8BlockBytes = 34;
constexpr int kQ8Values = 256;
constexpr int kLutParallelAccesses = 4;
constexpr int kLutBankElements = 16;
constexpr int kLutCopiesPerPair = 2;
constexpr int kLutPairElements = kQ8Values * kLutCopiesPerPair;
constexpr int kMmulM = 8;
constexpr int kMmulK = 8;
constexpr int kMmulN = 8;
constexpr int kMTiles = kM / kMmulM;
constexpr int kKTiles = kK / kMmulK;
constexpr int kNTiles = kN / kMmulN;
constexpr int kAElements = kMmulM * kMmulK;
constexpr int kBElements = kMmulK * kMmulN;
constexpr int kCElements = kMmulM * kMmulN;

using Mmul =
    aie::mmul<kMmulM, kMmulK, kMmulN, bfloat16, bfloat16, accfloat>;

inline float fp16_bits_to_f32(uint16_t bits) {
  const uint32_t sign = (uint32_t)(bits & 0x8000U) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fU;
  uint32_t fraction = bits & 0x03ffU;
  uint32_t f32_bits;

  if (exponent == 0) {
    if (fraction == 0) {
      f32_bits = sign;
    } else {
      uint32_t shift = 0;
      while ((fraction & 0x0400U) == 0) {
        fraction <<= 1;
        ++shift;
      }
      fraction &= 0x03ffU;
      f32_bits = sign | ((113U - shift) << 23) | (fraction << 13);
    }
  } else if (exponent == 0x1fU) {
    f32_bits = sign | 0x7f800000U | (fraction << 13);
  } else {
    exponent += 112U;
    f32_bits = sign | (exponent << 23) | (fraction << 13);
  }

  union {
    uint32_t bits;
    float value;
  } converted;
  converted.bits = f32_bits;
  return converted.value;
}

} // namespace

extern "C" void zero_f32_64x64(float *__restrict output) {
  const aie::vector<float, 16> zeros = aie::zeros<float, 16>();
  for (int offset = 0; offset < kM * kN; offset += 16) {
    aie::store_v(output + offset, zeros);
  }
}

extern "C" void init_q8_bf16_lut(bfloat16 *__restrict lut_ab,
                                  bfloat16 *__restrict lut_cd) {
  // AIE2P local-memory banks are 256 bits wide, so a BF16 table duplicates
  // each 16-entry bank line behind both pointers. The public API example uses
  // eight entries for older 128-bit banks; that layout is wrong on XDNA2.
  for (int base = 0; base < kQ8Values; base += kLutBankElements) {
    const int destination =
        (base / kLutBankElements) *
        kLutBankElements * kLutCopiesPerPair;
    for (int lane = 0; lane < kLutBankElements; ++lane) {
      const uint8_t raw = (uint8_t)(base + lane);
      const bfloat16 value = (bfloat16)(float)(int8_t)raw;
      lut_ab[destination + lane] = value;
      lut_ab[destination + kLutBankElements + lane] = value;
      lut_cd[destination + lane] = value;
      lut_cd[destination + kLutBankElements + lane] = value;
    }
  }
}

extern "C" void q8_decode_64x64_tile(
    const uint8_t *__restrict packed, bfloat16 *__restrict decoded,
    const bfloat16 *__restrict lut_ab,
    const bfloat16 *__restrict lut_cd) {
#ifdef Q8_FULL_ARRAY_DMA_ONLY
  return;
#elif defined(Q8_FULL_ARRAY_COMPUTE_ONLY)
  const aie::vector<bfloat16, 32> zeros = aie::zeros<bfloat16, 32>();
  for (int offset = 0; offset < kK * kN; offset += 32) {
    aie::store_v(decoded + offset, zeros);
  }
  return;
#endif

  // Input is 64 output rows, each containing two native Q8_0 K=32 blocks.
  // Output is AIE MMUL B order: [K/8][N/8][8 K][8 N].
  //
  // Decode one 8-output panel at a time. Each native Q8_0 block becomes
  // four 8x8 BF16 matrices; transposing those matrices produces the exact
  // K-major MMUL layout without scalar conversion or scalar stores.
  using Q8Lut = aie::lut<kLutParallelAccesses, bfloat16>;
  const Q8Lut q8_lut(kQ8Values, lut_ab, lut_cd);
  aie::parallel_lookup<uint8_t, Q8Lut, aie::lut_oor_policy::truncate>
      lookup(q8_lut, 0);

  for (int output_tile = 0; output_tile < kNTiles; ++output_tile) {
    for (int block = 0; block < kK / kQ8BlockElements; ++block) {
      aie::vector<bfloat16, kBElements> matrix0;
      aie::vector<bfloat16, kBElements> matrix1;
      aie::vector<bfloat16, kBElements> matrix2;
      aie::vector<bfloat16, kBElements> matrix3;

      for (int output_lane = 0; output_lane < kMmulN; ++output_lane) {
        const int output = output_tile * kMmulN + output_lane;
        const uint8_t *row =
            packed + output * (kK / kQ8BlockElements) * kQ8BlockBytes;
        const uint8_t *src = row + block * kQ8BlockBytes;
        const uint16_t scale_bits =
            (uint16_t)src[0] | ((uint16_t)src[1] << 8);
        const bfloat16 scale = (bfloat16)fp16_bits_to_f32(scale_bits);
        const auto quantized = aie::load_unaligned_v<kQ8BlockElements>(
            src + 2, 1);
        const auto unscaled = lookup.fetch(quantized);
        const auto scaled =
            aie::mul(unscaled, scale).template to_vector<bfloat16>();

        matrix0.insert(output_lane, scaled.template extract<kMmulK>(0));
        matrix1.insert(output_lane, scaled.template extract<kMmulK>(1));
        matrix2.insert(output_lane, scaled.template extract<kMmulK>(2));
        matrix3.insert(output_lane, scaled.template extract<kMmulK>(3));
      }

      const int reduction_tile = block * (kQ8BlockElements / kMmulK);
      bfloat16 *destination =
          decoded + (reduction_tile * kNTiles + output_tile) * kBElements;
      aie::store_v(destination,
                   aie::transpose(matrix0, kMmulN, kMmulK));
      aie::store_v(destination + kNTiles * kBElements,
                   aie::transpose(matrix1, kMmulN, kMmulK));
      aie::store_v(destination + 2 * kNTiles * kBElements,
                   aie::transpose(matrix2, kMmulN, kMmulK));
      aie::store_v(destination + 3 * kNTiles * kBElements,
                   aie::transpose(matrix3, kMmulN, kMmulK));
    }
  }
}

extern "C" void matmul_bf16_f32_64x64(
    const bfloat16 *__restrict activation,
    const bfloat16 *__restrict decoded_weight, float *__restrict output) {
#ifdef Q8_FULL_ARRAY_DMA_ONLY
  return;
#endif

  // A is [K/8][M/8][8][8], B is [K/8][N/8][8][8], and C is
  // [M/8][N/8][8][8]. C already contains prior K=64 partial sums. Expand
  // 2x2 across M/N so four live accumulators reuse every A/B load and give
  // the AIE2P scheduler enough independent MMULs to fill the pipeline.
  for (int m_tile = 0; m_tile < kMTiles; m_tile += 2)
    chess_prepare_for_pipelining chess_loop_range(8, 8) {
      for (int n_tile = 0; n_tile < kNTiles; n_tile += 2) {
        float *c00 =
            output + (m_tile * kNTiles + n_tile) * kCElements;
        float *c01 = c00 + kCElements;
        float *c10 =
            output + ((m_tile + 1) * kNTiles + n_tile) * kCElements;
        float *c11 = c10 + kCElements;

        Mmul accumulator00(aie::load_v<kCElements>(c00));
        Mmul accumulator01(aie::load_v<kCElements>(c01));
        Mmul accumulator10(aie::load_v<kCElements>(c10));
        Mmul accumulator11(aie::load_v<kCElements>(c11));

        // The retained A panel is streamed K-microtile-major so the
        // memory-tile DMA stays within its four-dimensional descriptor limit.
        const bfloat16 *a0 = activation + m_tile * kAElements;
        const bfloat16 *a1 = activation + (m_tile + 1) * kAElements;
        const bfloat16 *b0 =
            decoded_weight + n_tile * kBElements;
        const bfloat16 *b1 =
            decoded_weight + (n_tile + 1) * kBElements;
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
          const auto av0 = aie::load_v<kAElements>(a0);
          const auto av1 = aie::load_v<kAElements>(a1);
          const auto bv0 = aie::load_v<kBElements>(b0);
          const auto bv1 = aie::load_v<kBElements>(b1);
          accumulator00.mac(av0, bv0);
          accumulator01.mac(av0, bv1);
          accumulator10.mac(av1, bv0);
          accumulator11.mac(av1, bv1);
          a0 += kMTiles * kAElements;
          a1 += kMTiles * kAElements;
          b0 += kNTiles * kBElements;
          b1 += kNTiles * kBElements;
        }

        aie::store_v(c00, accumulator00.template to_vector<float>());
        aie::store_v(c01, accumulator01.template to_vector<float>());
        aie::store_v(c10, accumulator10.template to_vector<float>());
        aie::store_v(c11, accumulator11.template to_vector<float>());
      }
    }
}
