// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// oq4_dequant_mm — SKETCH. In-core Oq4 (int4) weight dequant + int8 mmul for
// aie2p, grafted from mlir-aie aie_kernels/aie2p/mm.cc.
//
// Feeds int4 weights all the way to L1 (half the bytes across the feed-bound
// shim/mem DMA), then unpacks nibbles -> int8 in-core (surplus compute) and runs
// the SAME 2x2 aie::mmul<8,8,8,int8,int8> as stock. A/C paths unchanged.
//
// Phases (see oq4-npu-design.md):
//   P1 (this file, #ifndef OQ4_SCALE/OQ4_FWHT): unpack only, stock int8 mmul,
//      no scale/FWHT -> measures the FEED win with --verify false.
//   P2: define OQ4_SCALE -> multiply the per-256-group f16 scale.
//   P3: native iu4 mmul if aie2p exposes it.
//
// NOTE: AIE-API intrinsic names for the nibble unpack (arithmetic vector shift +
// interleave) must be checked against the installed aie_api; the shift/mask math
// below is the contract, adjust the exact calls to compile.

#include <aie_api/aie.hpp>
#include "zero.cc"

// Unpack 32 packed bytes (64 signed int4, low-nibble-first) -> 64 int8, in order.
// byte b = (hi<<4)|lo ; lo_s8 = (int8)(b<<4)>>4 ; hi_s8 = (int8)b>>4  (arith shr
// sign-extends the 4-bit field).
static inline aie::vector<int8, 64> unpack_int4x2(const int8 *__restrict p) {
  aie::vector<int8, 32> packed = aie::load_v<32>(p);
  aie::vector<int8, 32> lo = aie::sr_reduce((packed << 4), 4); // arith >>4 of (packed<<4)
  aie::vector<int8, 32> hi = aie::sr_reduce(packed, 4);        // arith >>4
  return aie::interleave_zip(lo, hi, 1).first_and_second_as<int8, 64>();
}

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void oq4_dequant_mm_2x2(const int8 *__restrict pA,
                                      const int8 *__restrict pB_int4,
#ifdef OQ4_SCALE
                                      const bfloat16 *__restrict pScale, // 1/256-grp
#endif
                                      int8 *__restrict pC) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8, accauto>;
  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      int8 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2) {
        const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const int8 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        // B stride is HALVED: int4 tile = MMUL::size_B/2 bytes.
        const int8 *__restrict pB1 = pB_int4 + (j) * (MMUL::size_B / 2);
        const int8 *__restrict pB2 = pB_int4 + (j + 1) * (MMUL::size_B / 2);

        MMUL C00(aie::zeros<acc32, MMUL::size_C>());
        MMUL C01(aie::zeros<acc32, MMUL::size_C>());
        MMUL C10(aie::zeros<acc32, MMUL::size_C>());
        MMUL C11(aie::zeros<acc32, MMUL::size_C>());

        for (unsigned i = 0; i < colA; ++i) {
          aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA1 += MMUL::size_A; pA2 += MMUL::size_A;

          // --- the Oq4 change: int4 -> int8 in-core ---
          aie::vector<int8, MMUL::size_B> B0 = unpack_int4x2(pB1);
          aie::vector<int8, MMUL::size_B> B1 = unpack_int4x2(pB2);
          pB1 += MMUL::size_B / 2; pB2 += MMUL::size_B / 2;
#ifdef OQ4_SCALE
          // P2: apply the per-256-group f16 scale (one scale per 32 K-steps).
          // (weight already FWHT-rotated offline; activation FWHT is upstream)
          bfloat16 s = pScale[i / 32];
          B0 = to_int8_scaled(B0, s); B1 = to_int8_scaled(B1, s);
#endif
          C00.mac(A0, B0); C01.mac(A0, B1);
          C10.mac(A1, B0); C11.mac(A1, B1);
        }

        aie::store_v(pC1, C00.template to_vector<int8>()); pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<int8>()); pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<int8>()); pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<int8>()); pC2 += MMUL::size_C;
      }
    }
  event1();
}

extern "C" {
#ifndef DIM_M
#define DIM_M 128
#endif
#ifndef DIM_K
#define DIM_K 256   // >= 1 Oq4 group
#endif
#ifndef DIM_N
#define DIM_N 128
#endif
// r=s=t=8 ; tile counts (m/r, k/s, n/t)
void oq4_dequant_mm_i4_i8(int8 *a, int8 *b_int4,
#ifdef OQ4_SCALE
                          bfloat16 *scale,
#endif
                          int8 *c) {
  oq4_dequant_mm_2x2<DIM_M / 8, DIM_K / 8, DIM_N / 8>(a, b_int4,
#ifdef OQ4_SCALE
                                                      scale,
#endif
                                                      c);
}
} // extern "C"
