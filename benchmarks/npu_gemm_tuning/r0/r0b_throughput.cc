// R0b: on-hardware VMAC throughput. 4 named independent accumulators forming 4
// distinct dependency chains (c_ij += a_i * b_j) from resident L1 tiles, timed
// with the core cycle counter. 4 chains hide the acc1=4 latency, so a saturated
// pipe runs at II=1: cycles/(ITERS*4) -> 1.0 confirms 1 VMAC/cycle.
#include <aie_api/aie.hpp>

#ifndef ITERS
#define ITERS 100000
#endif
#ifndef MR
#define MR 4
#define MK 8
#define MN 8
#endif

extern "C" void r0b_i8i8(const int8 *__restrict pA, const int8 *__restrict pB,
                         int32 *__restrict pOut) {
  using MMUL = aie::mmul<MR, MK, MN, int8, int8>;
  // 4 distinct chains from 2 A-tiles x 2 B-tiles (all resident, hoisted)
  aie::vector<int8, MMUL::size_A> a0 = aie::load_v<MMUL::size_A>(pA);
  aie::vector<int8, MMUL::size_A> a1 = aie::load_v<MMUL::size_A>(pA + MMUL::size_A);
  aie::vector<int8, MMUL::size_B> b0 = aie::load_v<MMUL::size_B>(pB);
  aie::vector<int8, MMUL::size_B> b1 = aie::load_v<MMUL::size_B>(pB + MMUL::size_B);
  MMUL c00, c01, c10, c11;
  c00.mul(a0, b0); c01.mul(a0, b1); c10.mul(a1, b0); c11.mul(a1, b1);

  // ITERS*4 vmacs; runtime scales linearly with ITERS (host measures the slope)
  for (int i = 0; i < ITERS; i++) {
    c00.mac(a0, b0);
    c01.mac(a0, b1);
    c10.mac(a1, b0);
    c11.mac(a1, b1);
  }
  auto s = aie::add(aie::add(c00.template to_vector<int32>(), c01.template to_vector<int32>()),
                    aie::add(c10.template to_vector<int32>(), c11.template to_vector<int32>()));
  aie::store_v(pOut + 4, s);         // DCE guard
  pOut[0] = ITERS * 4;               // vmac count
  pOut[1] = MR * MK * MN;            // MACs per vmac
}
