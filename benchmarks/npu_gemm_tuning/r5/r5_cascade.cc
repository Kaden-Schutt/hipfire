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
#ifndef INNER
#define INNER 0            // extra reuse passes over the resident K-slice (compute knob:
#endif                     // isolates the array's compute rate from the weight feed)
#ifndef ROLE
#define ROLE 2             // 0=head (put only), 1=middle (get+put), 2=tail (get, store C)
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
using ACC = aie::accum<acc32, MMUL::size_C>;   // 4*16 = 64 acc32 partial C
static constexpr int BEATS = MMUL::size_C / 16;  // cascade beats (16 acc32 each)

// Load helpers: A tile / packed-int4 W tile j (weight stride is in BYTES on the int8
// buffer, then reinterpret — int4* arithmetic is byte-addressed, the R3a fix).
static inline aie::vector<int8, MMUL::size_A> ldA(const int8 *pA, int j) {
  return aie::load_v<MMUL::size_A>(pA + j * MMUL::size_A);
}
static inline aie::vector<int4, MMUL::size_B> ldW(const int8 *wbytes, int j) {
  return aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(wbytes + j * (MMUL::size_B / 2)));
}

// This core's K-slice partial. Four NAMED accumulators (independent dependency
// chains) reach II=1 — a single accumulator serializes on mac latency (~5x slower).
// The four (a_i, b_i) K-tiles are loaded ONCE into registers and reused INNER times
// (pure register macs — reloading from L1 each mac is load-bound, the ~5-TOPS trap),
// which isolates the array's compute rate from the DDR weight feed. The four K-slice
// partials are summed before the cascade transfer.
static inline ACC kslice_partial(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  aie::vector<int8, MMUL::size_A> a0 = ldA(pA, 0), a1 = ldA(pA, 1), a2 = ldA(pA, 2), a3 = ldA(pA, 3);
  aie::vector<int4, MMUL::size_B> b0 = ldW(wbytes, 0), b1 = ldW(wbytes, 1), b2 = ldW(wbytes, 2), b3 = ldW(wbytes, 3);
  MMUL c0, c1, c2, c3;
  c0.mul(a0, b0);
  c1.mul(a1, b1);
  c2.mul(a2, b2);
  c3.mul(a3, b3);
  for (int r = 0; r < INNER; r++)
      chess_prepare_for_pipelining {
    c0.mac(a0, b0);
    c1.mac(a1, b1);
    c2.mac(a2, b2);
    c3.mac(a3, b3);
  }
  return add(add(c0.to_accum(), c1.to_accum()), add(c2.to_accum(), c3.to_accum()));
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
#elif ROLE == 2 // TAIL: add cascade-in + this slice, STORE C once.
extern "C" void r5_cascade_tail(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                int32 *__restrict pC) {
  ACC sum = add(cascade_get(), kslice_partial(pA, wbytes));
  aie::store_v(pC, sum.to_vector<int32>());
}
#else           // STANDALONE (diagnostic): compute + store, NO cascade op.
extern "C" void r5_cascade_solo(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                int32 *__restrict pC) {
  aie::store_v(pC, kslice_partial(pA, wbytes).to_vector<int32>());
}
#endif
