// R3c: compute-bound prefill W4A8 GEMM. R3b went A-stream-bound because it
// re-streamed the activation for one N-block; here the activation A (M x K int8,
// M = NACC*4) stays RESIDENT and is reused across every streamed N-block, so only
// weights stream. Arithmetic intensity = NACC*1024/(16*0.5) = NACC*128 MACs/byte
// (NACC=8 -> 1024) -- deep in compute-bound territory, so throughput should hit
// the ~512 MAC/cyc core ceiling rather than the feed.
//
// Per call = one full N-block: loop all KFULL k-blocks, mac NACC named
// accumulators (R2a II=1 recipe), store NACC C tiles.
#include <aie_api/aie.hpp>

#ifndef KFULL
#define KFULL 32            // k-blocks in the resident tile (K = KFULL*16)
#endif
#ifndef NACC
#define NACC 8              // row-blocks (M = NACC*4)
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
// resident A laid out [k][row-block]: A[k*NACC + r]
#define A_AT(buf, r, k) aie::load_v<MMUL::size_A>((buf) + ((k) * NACC + (r)) * MMUL::size_A)

// pA: resident M x K activations (KFULL*NACC tiles). wbytes: one N-block's weights
// (KFULL packed-int4 tiles). pC: NACC int32 outputs (fresh per N-block).
extern "C" void r3c_mac(const int8 *__restrict pA, const int8 *__restrict wbytes,
                        int32 *__restrict pC) {
  const int4 *w = reinterpret_cast<const int4 *>(wbytes);
  aie::vector<int4, MMUL::size_B> b0 = aie::load_v<MMUL::size_B>(w);
  MMUL c0, c1, c2, c3, c4, c5, c6, c7;
  c0.mul(A_AT(pA, 0, 0), b0); c1.mul(A_AT(pA, 1, 0), b0);
  c2.mul(A_AT(pA, 2, 0), b0); c3.mul(A_AT(pA, 3, 0), b0);
#if NACC > 4
  c4.mul(A_AT(pA, 4, 0), b0); c5.mul(A_AT(pA, 5, 0), b0);
  c6.mul(A_AT(pA, 6, 0), b0); c7.mul(A_AT(pA, 7, 0), b0);
#endif
  for (int k = 1; k < KFULL; k++)
      chess_prepare_for_pipelining {
    aie::vector<int4, MMUL::size_B> b = aie::load_v<MMUL::size_B>(w + k * MMUL::size_B);
    c0.mac(A_AT(pA, 0, k), b); c1.mac(A_AT(pA, 1, k), b);
    c2.mac(A_AT(pA, 2, k), b); c3.mac(A_AT(pA, 3, k), b);
#if NACC > 4
    c4.mac(A_AT(pA, 4, k), b); c5.mac(A_AT(pA, 5, k), b);
    c6.mac(A_AT(pA, 6, k), b); c7.mac(A_AT(pA, 7, k), b);
#endif
  }
  aie::store_v(pC + 0 * MMUL::size_C, c0.template to_vector<int32>());
  aie::store_v(pC + 1 * MMUL::size_C, c1.template to_vector<int32>());
  aie::store_v(pC + 2 * MMUL::size_C, c2.template to_vector<int32>());
  aie::store_v(pC + 3 * MMUL::size_C, c3.template to_vector<int32>());
#if NACC > 4
  aie::store_v(pC + 4 * MMUL::size_C, c4.template to_vector<int32>());
  aie::store_v(pC + 5 * MMUL::size_C, c5.template to_vector<int32>());
  aie::store_v(pC + 6 * MMUL::size_C, c6.template to_vector<int32>());
  aie::store_v(pC + 7 * MMUL::size_C, c7.template to_vector<int32>());
#endif
}
