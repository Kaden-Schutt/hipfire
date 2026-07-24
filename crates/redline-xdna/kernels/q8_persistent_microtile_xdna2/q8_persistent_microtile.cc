// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Kaden Schutt

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef CHUNK_COUNT
#define CHUNK_COUNT 2
#endif

namespace {

constexpr int kRowsPerChunk = 8;
constexpr int kReduction = 64;
constexpr int kOutputs = 16;
constexpr int kBlockElements = 32;
constexpr int kBlockBytes = 34;
constexpr int kMmulRows = 4;
constexpr int kMmulReduction = 8;
constexpr int kMmulOutputs = 8;
constexpr int kRowTiles = kRowsPerChunk / kMmulRows;
constexpr int kReductionTiles = kReduction / kMmulReduction;
constexpr int kOutputTiles = kOutputs / kMmulOutputs;
constexpr int kMmulAElements = kMmulRows * kMmulReduction;
constexpr int kMmulBElements = kMmulReduction * kMmulOutputs;
constexpr int kMmulCElements = kMmulRows * kMmulOutputs;
constexpr int kChunkAElements = kRowsPerChunk * kReduction;
constexpr int kChunkCElements = kRowsPerChunk * kOutputs;

using Mmul =
    aie::mmul<kMmulRows, kMmulReduction, kMmulOutputs, bfloat16,
              bfloat16, accfloat>;

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

void decode_b_tile_major(const uint8_t *__restrict packed,
                         bfloat16 *__restrict decoded) {
  // Source is Hipfire Q8_0 W[output, K]. Destination is AIE MMUL B tile
  // order: [K/8 tile][output/8 tile][8 K][8 output].
  for (int output = 0; output < kOutputs; ++output) {
    const uint8_t *row = packed + output * (kReduction / 32) * kBlockBytes;
    for (int block = 0; block < kReduction / kBlockElements; ++block) {
      const uint8_t *src = row + block * kBlockBytes;
      const uint16_t scale_bits =
          (uint16_t)src[0] | ((uint16_t)src[1] << 8);
      const float scale = fp16_bits_to_f32(scale_bits);
      for (int lane = 0; lane < kBlockElements; ++lane) {
        const int reduction = block * kBlockElements + lane;
        const int reduction_tile = reduction / kMmulReduction;
        const int reduction_lane = reduction % kMmulReduction;
        const int output_tile = output / kMmulOutputs;
        const int output_lane = output % kMmulOutputs;
        const int destination =
            ((reduction_tile * kOutputTiles + output_tile) *
                 kMmulBElements +
             reduction_lane * kMmulOutputs + output_lane);
        decoded[destination] =
            (bfloat16)(scale * (float)(int8_t)src[2 + lane]);
      }
    }
  }
}

void matmul_one_chunk(const bfloat16 *__restrict activation,
                      const bfloat16 *__restrict weight,
                      float *__restrict output) {
  for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
    for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
      const bfloat16 *a =
          activation + row_tile * kReductionTiles * kMmulAElements;
      const bfloat16 *b = weight + output_tile * kMmulBElements;
      aie::vector<bfloat16, kMmulAElements> av =
          aie::load_v<kMmulAElements>(a);
      aie::vector<bfloat16, kMmulBElements> bv =
          aie::load_v<kMmulBElements>(b);
      Mmul accumulator;
      accumulator.mul(av, bv);

      for (int reduction_tile = 1; reduction_tile < kReductionTiles;
           ++reduction_tile) {
        a += kMmulAElements;
        b += kOutputTiles * kMmulBElements;
        av = aie::load_v<kMmulAElements>(a);
        bv = aie::load_v<kMmulBElements>(b);
        accumulator.mac(av, bv);
      }

      float *c =
          output +
          (row_tile * kOutputTiles + output_tile) * kMmulCElements;
      aie::store_v(c, accumulator.template to_vector<float>());
    }
  }
}

} // namespace

extern "C" void q8_decode_b_tile_major(
    const uint8_t *__restrict packed, bfloat16 *__restrict decoded_weight) {
  decode_b_tile_major(packed, decoded_weight);
}

extern "C" void bf16_persistent_microtile(
    const bfloat16 *__restrict activation,
    const bfloat16 *__restrict decoded_weight, float *__restrict output) {
  for (int chunk = 0; chunk < CHUNK_COUNT; ++chunk) {
    matmul_one_chunk(activation + chunk * kChunkAElements, decoded_weight,
                     output + chunk * kChunkCElements);
  }
}
