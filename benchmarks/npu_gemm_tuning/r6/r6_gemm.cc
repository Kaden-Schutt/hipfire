// R6: 2D-tiled W4A8 GEMM — the M/AI lever. Real prefill is feed / arithmetic-
// intensity bound (R5 verdict): AI must clear ~58 MACs/byte to go compute-bound,
// which needs a tile that reuses BOTH operands — each weight column across MT M-row
// blocks, each activation row across NT N-col blocks. Everything before (R2a/R4/R5)
// used a 1D tile (16x16, reuse in one dimension only) and stalled at ~3-5 TOPS.
//
// One call computes an (MT*4) x (NT*16) output block, K-reduced over KCHUNK 16x16
// tiles: A[MT][KCHUNK] and W[NT][KCHUNK] are resident; the four N accumulators for a
// given M-row block share the loaded activation (A reused NT-wide), and the weight
// tiles are reused across the MT M-blocks. C is stored ONCE (per-tile overhead
// amortized over MT*NT*KCHUNK mmuls). Build -DMT= -DNT= -DKCHUNK= (NT==4 here).
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 4               // M-row blocks (each MR=4 rows)
#endif
#ifndef NT
#define NT 4               // N-col blocks (each MN=16); == number of accumulators
#endif
#ifndef KCHUNK
#define KCHUNK 8           // 16x16 K-tiles reduced per call
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;

static inline aie::vector<int8, MMUL::size_A> ldA(const int8 *pA, int mt, int k) {
  return aie::load_v<MMUL::size_A>(pA + (mt * KCHUNK + k) * MMUL::size_A);
}
static inline aie::vector<int4, MMUL::size_B> ldW(const int8 *wbytes, int nt, int k) {
  return aie::load_v<MMUL::size_B>(
      reinterpret_cast<const int4 *>(wbytes + (nt * KCHUNK + k) * (MMUL::size_B / 2)));
}

// pA: MT*KCHUNK activation tiles (row-block major). wbytes: NT*KCHUNK packed-int4
// weight tiles (col-block major). pC: MT*NT int32 output tiles (mt-major).
extern "C" void r6_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                       int32 *__restrict pC) {
  static_assert(NT == 4, "kernel unrolled for NT=4 accumulators");
  for (int mt = 0; mt < MT; mt++) {
    MMUL c0, c1, c2, c3;  // NT=4 N-blocks for this M-row block (II=1 chains)
    aie::vector<int8, MMUL::size_A> a = ldA(pA, mt, 0);
    c0.mul(a, ldW(wbytes, 0, 0));
    c1.mul(a, ldW(wbytes, 1, 0));
    c2.mul(a, ldW(wbytes, 2, 0));
    c3.mul(a, ldW(wbytes, 3, 0));
    for (int k = 1; k < KCHUNK; k++)
        chess_prepare_for_pipelining {
      a = ldA(pA, mt, k);                 // one activation reused across NT weights
      c0.mac(a, ldW(wbytes, 0, k));
      c1.mac(a, ldW(wbytes, 1, k));
      c2.mac(a, ldW(wbytes, 2, k));
      c3.mac(a, ldW(wbytes, 3, k));
    }
    int32 *c = pC + mt * NT * MMUL::size_C;
    aie::store_v(c + 0 * MMUL::size_C, c0.template to_vector<int32>());
    aie::store_v(c + 1 * MMUL::size_C, c1.template to_vector<int32>());
    aie::store_v(c + 2 * MMUL::size_C, c2.template to_vector<int32>());
    aie::store_v(c + 3 * MMUL::size_C, c3.template to_vector<int32>());
  }
}
