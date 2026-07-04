// R3a: REAL K-accumulating W4A8 GEMV (batched-decode). Unlike R2a (which repeated
// one weight tile INNER times to probe compute), this streams distinct weight
// tiles over the K contraction and accumulates a genuine result — the decode-shape
// kernel where int4's half-weight-bytes advantage actually pays off (M small ⇒
// bandwidth-bound).
//
// Shape per call: one N-block (MN=16) of output for M=MR(=4) rows. The activation
// A (MR x K int8) is resident; a super-tile of KCHUNK weight tiles (each 16x16
// int4 = 128 B) streams in; the K accumulator C (MR x MN int32) is carried in a
// resident buffer, loaded once and stored once per super-tile so the KCHUNK macs
// (II=1, named-accumulator recipe from R2a) dominate the per-call C load/store.
#include <aie_api/aie.hpp>

#ifndef KCHUNK
#define KCHUNK 64          // weight tiles (k-steps) per streamed super-tile
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;   // MR=4, MK=16, MN=16

// pAchunk: KCHUNK activation tiles for this super-tile (KCHUNK * size_A int8).
// wbytes:  KCHUNK packed-int4 weight tiles (KCHUNK * size_B/2 bytes).
// pC:      one resident accumulator tile (size_C int32), carried across calls.
extern "C" void r3a_matvec(const int8 *__restrict pAchunk,
                           const int8 *__restrict wbytes, int32 *__restrict pC) {
  const int4 *w = reinterpret_cast<const int4 *>(wbytes);
  // KCHUNK macs in a fresh register accumulator (II=1), then fold into the
  // running K partial in pC (add vector partials — mmul can't reseed its acc).
  MMUL c;
  c.mul(aie::load_v<MMUL::size_A>(pAchunk),
        aie::load_v<MMUL::size_B>(w));
  for (int j = 1; j < KCHUNK; j++)
      chess_prepare_for_pipelining {
    aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pAchunk + j * MMUL::size_A);
    aie::vector<int4, MMUL::size_B> b = aie::load_v<MMUL::size_B>(w + j * MMUL::size_B);
    c.mac(a, b);
  }
  auto partial = c.template to_vector<int32>();
  aie::store_v(pC, aie::add(aie::load_v<MMUL::size_C>(pC), partial));
}
