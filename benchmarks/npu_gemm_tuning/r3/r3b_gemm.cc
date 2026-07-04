// R3b: REAL K-accumulating W4A8 GEMM (prefill). Combines R2a's NACC-accumulator
// weight reuse with R3a's genuine K streaming: one streamed int4 weight tile is
// macced into NACC row-block accumulators (NACC*MR activation rows), so weight
// reuse = NACC and arithmetic intensity = NACC*1024/128 = NACC*8 MACs/byte. At
// NACC=8 (M=32 rows) that clears the single-stream feed, so a prefill tile is
// compute-bound (~1.8 TOPS/core) rather than feed-bound — the large-M case.
//
// Named accumulators + chess_prepare_for_pipelining (R2a) for II=1; K partials
// folded into resident C via vector add across super-tiles (aie::mmul can't reseed).
#include <aie_api/aie.hpp>

#ifndef KCHUNK
#define KCHUNK 64
#endif
#ifndef NACC
#define NACC 8              // row-blocks (M = NACC*4); 8 => compute-bound single-stream
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
#define LDA(buf, blk, k) aie::load_v<MMUL::size_A>((buf) + ((k) * NACC + (blk)) * MMUL::size_A)

// pA: KCHUNK * NACC activation tiles (row-major [k][block]). wbytes: KCHUNK packed
// int4 weight tiles. pC: NACC resident int32 accumulators, folded across calls.
extern "C" void r3b_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                        int32 *__restrict pC) {
  const int4 *w = reinterpret_cast<const int4 *>(wbytes);
  aie::vector<int4, MMUL::size_B> b0 = aie::load_v<MMUL::size_B>(w);
  MMUL c0, c1, c2, c3, c4, c5, c6, c7;
  c0.mul(LDA(pA, 0, 0), b0); c1.mul(LDA(pA, 1, 0), b0);
  c2.mul(LDA(pA, 2, 0), b0); c3.mul(LDA(pA, 3, 0), b0);
#if NACC > 4
  c4.mul(LDA(pA, 4, 0), b0); c5.mul(LDA(pA, 5, 0), b0);
  c6.mul(LDA(pA, 6, 0), b0); c7.mul(LDA(pA, 7, 0), b0);
#endif
  for (int k = 1; k < KCHUNK; k++)
      chess_prepare_for_pipelining {
    aie::vector<int4, MMUL::size_B> b = aie::load_v<MMUL::size_B>(w + k * MMUL::size_B);
    c0.mac(LDA(pA, 0, k), b); c1.mac(LDA(pA, 1, k), b);
    c2.mac(LDA(pA, 2, k), b); c3.mac(LDA(pA, 3, k), b);
#if NACC > 4
    c4.mac(LDA(pA, 4, k), b); c5.mac(LDA(pA, 5, k), b);
    c6.mac(LDA(pA, 6, k), b); c7.mac(LDA(pA, 7, k), b);
#endif
  }
#define FOLD(blk, cc) aie::store_v(pC + (blk) * MMUL::size_C, \
    aie::add(aie::load_v<MMUL::size_C>(pC + (blk) * MMUL::size_C), cc.template to_vector<int32>()))
  FOLD(0, c0); FOLD(1, c1); FOLD(2, c2); FOLD(3, c3);
#if NACC > 4
  FOLD(4, c4); FOLD(5, c5); FOLD(6, c6); FOLD(7, c7);
#endif
}
