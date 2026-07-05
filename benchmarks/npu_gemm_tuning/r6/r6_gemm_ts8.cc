// R6-TS with NT=8 accumulators (vs the NT=4 r6_gemm_ts.cc). Each N-slab now covers 8·16
// = 128 N-cols, HALVING the number of slab acquires for a given N — the one lever against
// the objectfifo per-slab streaming overhead that caps the array at ~3.4 TOPS raw. The bet:
// 8 live mmul accumulators (c0..c7) still fit the aie2p acc register file at II=1, and the
// doubled W-slab (8·KCHUNK tiles) still fits L1. If either fails (spill / L1 overrun) NT=8
// is a dead end and NT=4 KCHUNK=32 stands as the ceiling. Row-major A/C via tensor streams,
// same as r6_gemm_ts.cc; only NT changes.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 8
#endif
#ifndef NT
#define NT 8
#endif
#ifndef KCHUNK
#define KCHUNK 32
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;

static inline aie::vector<int4, MMUL::size_B> ldW(const int8 *wbytes, int nt, int k) {
  return aie::load_v<MMUL::size_B>(
      reinterpret_cast<const int4 *>(wbytes + (nt * KCHUNK + k) * (MMUL::size_B / 2)));
}

extern "C" void r6_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                       int32 *__restrict pC) {
  static_assert(NT == 8, "this variant is unrolled for NT=8 accumulators");

  auto a_desc = aie::make_tensor_descriptor<int8, 16>(
      aie::tensor_dim(MT, 4 * KCHUNK), aie::tensor_dim(KCHUNK, 1), aie::tensor_dim(4u, KCHUNK));
  auto tsA = aie::make_tensor_buffer_stream(pA, a_desc);
  auto c_desc = aie::make_tensor_descriptor<int32, 16>(
      aie::tensor_dim(MT, 4 * NT), aie::tensor_dim(NT, 1), aie::tensor_dim(4u, NT));
  auto tsC = aie::make_tensor_buffer_stream(pC, c_desc);

  for (int mt = 0; mt < MT; mt++) {
    MMUL c0, c1, c2, c3, c4, c5, c6, c7;
    aie::vector<int8, 16> r0, r1, r2, r3;
    tsA >> r0 >> r1 >> r2 >> r3;
    aie::vector<int8, MMUL::size_A> a = aie::concat(r0, r1, r2, r3);
    c0.mul(a, ldW(wbytes, 0, 0));
    c1.mul(a, ldW(wbytes, 1, 0));
    c2.mul(a, ldW(wbytes, 2, 0));
    c3.mul(a, ldW(wbytes, 3, 0));
    c4.mul(a, ldW(wbytes, 4, 0));
    c5.mul(a, ldW(wbytes, 5, 0));
    c6.mul(a, ldW(wbytes, 6, 0));
    c7.mul(a, ldW(wbytes, 7, 0));
    for (int k = 1; k < KCHUNK; k++)
        chess_prepare_for_pipelining {
      tsA >> r0 >> r1 >> r2 >> r3;
      a = aie::concat(r0, r1, r2, r3);
      c0.mac(a, ldW(wbytes, 0, k));
      c1.mac(a, ldW(wbytes, 1, k));
      c2.mac(a, ldW(wbytes, 2, k));
      c3.mac(a, ldW(wbytes, 3, k));
      c4.mac(a, ldW(wbytes, 4, k));
      c5.mac(a, ldW(wbytes, 5, k));
      c6.mac(a, ldW(wbytes, 6, k));
      c7.mac(a, ldW(wbytes, 7, k));
    }
    aie::vector<int32, MMUL::size_C> v[8] = {
        c0.template to_vector<int32>(), c1.template to_vector<int32>(),
        c2.template to_vector<int32>(), c3.template to_vector<int32>(),
        c4.template to_vector<int32>(), c5.template to_vector<int32>(),
        c6.template to_vector<int32>(), c7.template to_vector<int32>()};
    for (int nt = 0; nt < NT; nt++)
      tsC << v[nt].template extract<16>(0) << v[nt].template extract<16>(1)
          << v[nt].template extract<16>(2) << v[nt].template extract<16>(3);
  }
}
