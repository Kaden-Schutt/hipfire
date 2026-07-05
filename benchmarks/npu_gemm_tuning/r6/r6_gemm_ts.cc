// R6-TS: the R6 2D-tiled W4A8 GEMM, but reading ROW-MAJOR A and writing ROW-MAJOR C via
// in-core aie::tensor_buffer_streams instead of pre-tiled load_v/store_v. The address
// generators walk the tile pattern in parallel with the vector MACs, so the reshuffle is
// free — no CPU marshaling, no strided DMA, no memtile hop. W stays pre-packed tile-major
// (static, arranged once; int4 sub-byte descriptor deferred). Buffer SIZES are identical
// to r6_gemm.cc (A = MT*KCHUNK*64 int8, C = MT*NT*64 int32) — only the A/C interpretation
// changes — so the MLIR/DMA (r6_gen.py) are unchanged (plain linear dma_bd).
//
// A stream [mt,k,m] leaf=16 (one contiguous kk row of a 4x16 tile): 4 consecutive pops =
// m0..m3 of tile (mt,k); concat -> the mmul A 64-vector a[m*16+kk]. Reproduces pack_a.
// C stream [mt,nt,m] leaf=16 (one contiguous n row of a 4x16 C tile), Nb=NT*16: push the
// 4 rows of each accumulator -> row-major c[(mt*4+m)*Nb + nt*16+n]. Reproduces unpack_c.
// (Both validated standalone in benchmarks/npu_gemm_tuning/ts before this port.)
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 8
#endif
#ifndef NT
#define NT 4
#endif
#ifndef KCHUNK
#define KCHUNK 8
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;

static inline aie::vector<int4, MMUL::size_B> ldW(const int8 *wbytes, int nt, int k) {
  return aie::load_v<MMUL::size_B>(
      reinterpret_cast<const int4 *>(wbytes + (nt * KCHUNK + k) * (MMUL::size_B / 2)));
}

// pA: ROW-MAJOR (MT*4) x (KCHUNK*16) int8.  wbytes: NT*KCHUNK packed-int4 tiles (pre-
// packed tile-major).  pC: ROW-MAJOR (MT*4) x (NT*16) int32.
extern "C" void r6_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                       int32 *__restrict pC) {
  static_assert(NT == 4, "kernel unrolled for NT=4 accumulators");

  auto a_desc = aie::make_tensor_descriptor<int8, 16>(
      aie::tensor_dim(MT, 4 * KCHUNK),   // mt
      aie::tensor_dim(KCHUNK, 1),        // k
      aie::tensor_dim(4u, KCHUNK));      // m
  auto tsA = aie::make_tensor_buffer_stream(pA, a_desc);

  auto c_desc = aie::make_tensor_descriptor<int32, 16>(
      aie::tensor_dim(MT, 4 * NT),       // mt
      aie::tensor_dim(NT, 1),            // nt
      aie::tensor_dim(4u, NT));          // m
  auto tsC = aie::make_tensor_buffer_stream(pC, c_desc);

  for (int mt = 0; mt < MT; mt++) {
    MMUL c0, c1, c2, c3;  // NT=4 N-blocks for this M-row block (II=1 chains)
    aie::vector<int8, 16> r0, r1, r2, r3;
    tsA >> r0 >> r1 >> r2 >> r3;
    aie::vector<int8, MMUL::size_A> a = aie::concat(r0, r1, r2, r3);
    c0.mul(a, ldW(wbytes, 0, 0));
    c1.mul(a, ldW(wbytes, 1, 0));
    c2.mul(a, ldW(wbytes, 2, 0));
    c3.mul(a, ldW(wbytes, 3, 0));
    for (int k = 1; k < KCHUNK; k++)
        chess_prepare_for_pipelining {
      tsA >> r0 >> r1 >> r2 >> r3;       // one activation tile reused across NT weights
      a = aie::concat(r0, r1, r2, r3);
      c0.mac(a, ldW(wbytes, 0, k));
      c1.mac(a, ldW(wbytes, 1, k));
      c2.mac(a, ldW(wbytes, 2, k));
      c3.mac(a, ldW(wbytes, 3, k));
    }
    // Push each accumulator's 4 rows (m fastest, then nt, then mt) to row-major C.
    auto v0 = c0.template to_vector<int32>();
    auto v1 = c1.template to_vector<int32>();
    auto v2 = c2.template to_vector<int32>();
    auto v3 = c3.template to_vector<int32>();
    tsC << v0.template extract<16>(0) << v0.template extract<16>(1)
        << v0.template extract<16>(2) << v0.template extract<16>(3);
    tsC << v1.template extract<16>(0) << v1.template extract<16>(1)
        << v1.template extract<16>(2) << v1.template extract<16>(3);
    tsC << v2.template extract<16>(0) << v2.template extract<16>(1)
        << v2.template extract<16>(2) << v2.template extract<16>(3);
    tsC << v3.template extract<16>(0) << v3.template extract<16>(1)
        << v3.template extract<16>(2) << v3.template extract<16>(3);
  }
}
