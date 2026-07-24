// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Kaden Schutt

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

// Peano/AIE2P does not expose native FP16 arithmetic. Convert the GGML Q8_0
// scale bits explicitly, including subnormals and IEEE special values, before
// rounding the scaled signed byte to BF16.
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
      const uint32_t f32_exponent = 113U - shift;
      f32_bits = sign | (f32_exponent << 23) | (fraction << 13);
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

extern "C" void q8_decode_bf16(const uint8_t *__restrict packed,
                                bfloat16 *__restrict decoded,
                                int32_t block_count) {
  for (int32_t block = 0; block < block_count; ++block) {
    const uint8_t *src = packed + block * 34;
    const uint16_t scale_bits =
        (uint16_t)src[0] | ((uint16_t)src[1] << 8);
    const float scale = fp16_bits_to_f32(scale_bits);

    for (int32_t lane = 0; lane < 32; ++lane) {
      const int8_t quantized = (int8_t)src[2 + lane];
      decoded[block * 32 + lane] =
          (bfloat16)(scale * (float)quantized);
    }
  }
}
