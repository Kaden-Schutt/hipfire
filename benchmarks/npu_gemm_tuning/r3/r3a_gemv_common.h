// Shared body for the R3a W4A8 GEMV super-tile partial. The two entry points
// (r3a_matvec_init reseeds pC, r3a_matvec accumulates) live in separate
// translation units so binding both kernels in one core does not collide on
// duplicate symbols; the helper is `static inline` (internal linkage).
#pragma once
#include <aie_api/aie.hpp>

#ifndef KCHUNK
#define KCHUNK 64          // weight tiles (k-steps) per streamed super-tile
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;   // MR=4, MK=16, MN=16

// Byte stride of one packed int4 weight tile: size_B int4 values, 2 per byte.
// The weight pointer MUST be advanced in BYTES (on the int8 buffer) and only then
// reinterpreted as int4 — pointer arithmetic on `const int4*` advances one BYTE
// per element (not one nibble), so `w + j*size_B` strides 2x too far, reading
// every other tile and running off the buffer end. That was the R3a "half the K"
// bug; R2a hid it by loading a single resident tile with no per-tile stride.
static constexpr int WTILE_BYTES = MMUL::size_B / 2;

// Compute this super-tile's KCHUNK-mac partial in a fresh register accumulator
// (II=1, named-accumulator recipe).
static inline aie::vector<int32, MMUL::size_C>
r3a_super_partial(const int8 *__restrict pAchunk, const int8 *__restrict wbytes) {
  MMUL c;
  c.mul(aie::load_v<MMUL::size_A>(pAchunk),
        aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(wbytes)));
  for (int j = 1; j < KCHUNK; j++)
      chess_prepare_for_pipelining {
    aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pAchunk + j * MMUL::size_A);
    const int4 *bj = reinterpret_cast<const int4 *>(wbytes + j * WTILE_BYTES);
    aie::vector<int4, MMUL::size_B> b = aie::load_v<MMUL::size_B>(bj);
    c.mac(a, b);
  }
  return c.template to_vector<int32>();
}
