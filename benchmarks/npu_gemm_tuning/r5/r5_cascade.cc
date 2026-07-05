// R5: one core of a K-cascade W4A8 column. The ROWS cores in a column split the K
// contraction; each computes its K-slice partial and the 512-bit cascade stream
// carries the running accumulator core->core (put_mcd / get_scd), so C is
// accumulated in-flight and stored ONCE by the tail core — eliminating the per-tile
// C load/store that pins the memtile dataflow (and SOTA FastFlowLM) to ~5 TOPS.
//
// Cascade API (from aie_kernels/aie2/cascade_mm.cc + aie2p_streams.h): the 512-bit
// cascade moves one v16acc32 per beat via put_mcd(v16acc32)/get_scd_v16acc32();
// the mmul<4,16,16> accumulator is size_C=64 acc32 = 4 beats. The graph wires the
// physical link with aie.cascade_flow(src,dst). Built -DROLE={0 head,1 mid,2 tail}.
#include <aie_api/aie.hpp>

#ifndef KSLICE
#define KSLICE 16          // 16x16 mmul steps this core contracts (its K-slice)
#endif
#ifndef ROLE
#define ROLE 2             // 0=head (put only), 1=middle (get+put), 2=tail (get, store C)
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
using ACC = aie::accum<acc32, MMUL::size_C>;   // 4*16 = 64 acc32 partial C
static constexpr int BEATS = MMUL::size_C / 16;  // cascade beats (16 acc32 each)

// This core's K-slice partial, in a register accumulator (II=1 recipe).
static inline ACC kslice_partial(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  MMUL c;
  const int4 *w = reinterpret_cast<const int4 *>(wbytes);
  c.mul(aie::load_v<MMUL::size_A>(pA), aie::load_v<MMUL::size_B>(w));
  for (int j = 1; j < KSLICE; j++)
      chess_prepare_for_pipelining {
    aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pA + j * MMUL::size_A);
    // Weight stride in BYTES on the int8 buffer, then reinterpret (int4* arithmetic
    // is byte-addressed — the R3a fix); size_B/2 bytes per 16x16 tile.
    const int4 *bj = reinterpret_cast<const int4 *>(wbytes + j * (MMUL::size_B / 2));
    c.mac(a, aie::load_v<MMUL::size_B>(bj));
  }
  return c.to_accum();
}

static inline void cascade_put(ACC acc) {
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    put_mcd(acc.template extract<16>(i).to_native());
}

static inline ACC cascade_get() {
  ACC acc;
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    acc.insert(i, aie::accum<acc32, 16>(get_scd_v16acc32()));
  return acc;
}

#if ROLE == 0   // HEAD: seed the cascade with this slice's partial.
extern "C" void r5_cascade_head(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  cascade_put(kslice_partial(pA, wbytes));
}
#elif ROLE == 1 // MIDDLE: add cascade-in + this slice, pass on.
extern "C" void r5_cascade_mid(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  ACC sum = add(cascade_get(), kslice_partial(pA, wbytes));
  cascade_put(sum);
}
#else           // TAIL: add cascade-in + this slice, STORE C once.
extern "C" void r5_cascade_tail(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                int32 *__restrict pC) {
  ACC sum = add(cascade_get(), kslice_partial(pA, wbytes));
  aie::store_v(pC, sum.to_vector<int32>());
}
#endif
