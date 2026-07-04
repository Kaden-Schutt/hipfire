// R2a: W4A8 compute fused with weight streaming. R0b's II=1 mac_4x16_16x16 loop,
// but the int4 weight tile is STREAMED from L3 (the R1 feed) instead of resident.
// Each streamed 16x16 int4 tile (128 B) is reused (INNER+1)*NACC times against
// NACC resident int8 activation tiles, so the arithmetic intensity =
//   (INNER+1)*NACC*1024 MACs / 128 B  ==  the reuse (M-proxy) knob.
// Sweeping it moves the core from feed-bound (few MACs/byte) to compute-bound
// (II=1, ~1.84 TMAC/s/core). The crossover in MACs/byte validates R0+R1's ~M=530.
//
// NACC independent accumulators hide the acc-latency (>=4 needed for II=1, R0b).
#include <aie_api/aie.hpp>

#ifndef NACC
#define NACC 8            // resident activation tiles = independent accumulators
#endif
#ifndef INNER
#define INNER 64          // extra reuse passes of the streamed weight tile
#endif

#ifdef INT8W
using MMUL = aie::mmul<8, 8, 8, int8, int8>;     // 512 MACs/mac (dense int8 ref)
using WT = int8;
#else
using MMUL = aie::mmul<4, 16, 16, int8, int4>;   // W4A8 / Oq4 (1024 MACs nominal)
using WT = int4;
#endif

// pA: NACC resident activation tiles (NACC * size_A int8), reused every call.
// wbytes: one streamed int4 weight tile as packed int8 (size_B/2 = 128 B); there
//   is no numpy int4, so it is passed as bytes and reinterpreted here.
// pC: NACC int32 output tiles (DCE guard; fresh accumulators per streamed tile).
extern "C" void r2a_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                        int32 *__restrict pC) {
  // 4 NAMED independent accumulators (distinct dependency chains) + a pipelining
  // hint -- the R0b recipe that reaches II=1. An acc[]/a[] array loop instead
  // carries per-iteration address arithmetic and collapses the accumulators.
  aie::vector<int8, MMUL::size_A> a0 = aie::load_v<MMUL::size_A>(pA);
  aie::vector<int8, MMUL::size_A> a1 = aie::load_v<MMUL::size_A>(pA + MMUL::size_A);
  aie::vector<int8, MMUL::size_A> a2 = aie::load_v<MMUL::size_A>(pA + 2 * MMUL::size_A);
  aie::vector<int8, MMUL::size_A> a3 = aie::load_v<MMUL::size_A>(pA + 3 * MMUL::size_A);
  const WT *wtile = reinterpret_cast<const WT *>(wbytes);
  aie::vector<WT, MMUL::size_B> b = aie::load_v<MMUL::size_B>(wtile);

  MMUL c0, c1, c2, c3;
  c0.mul(a0, b); c1.mul(a1, b); c2.mul(a2, b); c3.mul(a3, b);
  for (int i = 0; i < INNER; i++)
      chess_prepare_for_pipelining {
    c0.mac(a0, b); c1.mac(a1, b); c2.mac(a2, b); c3.mac(a3, b);
  }
  aie::store_v(pC, c0.template to_vector<int32>());
  aie::store_v(pC + MMUL::size_C, c1.template to_vector<int32>());
  aie::store_v(pC + 2 * MMUL::size_C, c2.template to_vector<int32>());
  aie::store_v(pC + 3 * MMUL::size_C, c3.template to_vector<int32>());
}
