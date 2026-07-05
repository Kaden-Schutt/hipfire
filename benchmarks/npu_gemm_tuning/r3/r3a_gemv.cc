// R3a: REAL K-accumulating W4A8 GEMV (batched-decode). Unlike R2a (which repeated
// one weight tile INNER times to probe compute), this streams distinct weight
// tiles over the K contraction and accumulates a genuine result — the decode-shape
// kernel where int4's half-weight-bytes advantage actually pays off (M small ⇒
// bandwidth-bound).
//
// Shape per call: one N-block (MN=16) of output for M=MR(=4) rows. The activation
// A (MR x K int8) is resident; a super-tile of KCHUNK weight tiles (each 16x16
// int4 = 128 B) streams in; the K accumulator C (MR x MN int32) is carried in a
// resident buffer across the N_SUPER super-tile calls of one dispatch.
//
// r3a_matvec ADDS this super-tile's partial into the running K partial in pC. The
// first super-tile of each dispatch must use r3a_matvec_init (separate TU) to
// RESEED pC, else the resident pC carries stale state from the prior dispatch.
#include "r3a_gemv_common.h"

extern "C" void r3a_matvec(const int8 *__restrict pAchunk,
                           const int8 *__restrict wbytes, int32 *__restrict pC) {
  auto partial = r3a_super_partial(pAchunk, wbytes);
  aie::store_v(pC, aie::add(aie::load_v<MMUL::size_C>(pC), partial));
}
