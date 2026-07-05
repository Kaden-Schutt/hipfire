// R5: one core of a K-cascade W4A8 column. The ROWS cores in a column split the K
// contraction; each computes its K-slice partial and the 512-bit cascade stream
// carries the running accumulator core->core, so C is accumulated in-flight and
// stored ONCE by the tail core — eliminating the per-tile C load/store that pins
// the memtile dataflow (and SOTA FastFlowLM) to ~5 TOPS. See README.md.
//
// STATUS: compute skeleton. The cascade accum plumbing (readincr/writeincr on
// aie::accum) is reverse-engineered from aie_api/adf/stream.hpp with no example to
// crib; the exact accum<->mmul seeding is to be verified on hardware once the
// low-level (non-IRON) build path is established (README step 1). Built with
// -DROLE={0 head,1 middle,2 tail} and -DKSLICE=<mmuls this core contracts>.
#include <aie_api/aie.hpp>

#ifndef KSLICE
#define KSLICE 16          // 16x16 mmul steps this core contracts (its K-slice)
#endif
#ifndef ROLE
#define ROLE 2             // 0=head (no cascade-in), 1=middle, 2=tail (stores C)
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
using ACC = aie::accum<acc32, MMUL::size_C>;   // 4*16 int32 partial C

// Compute this core's K-slice partial in a register accumulator (II=1 recipe).
static inline ACC kslice_partial(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  MMUL c;
  const int4 *w = reinterpret_cast<const int4 *>(wbytes);
  c.mul(aie::load_v<MMUL::size_A>(pA), aie::load_v<MMUL::size_B>(w));
  for (int j = 1; j < KSLICE; j++)
      chess_prepare_for_pipelining {
    aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pA + j * MMUL::size_A);
    // Weight stride in BYTES on the int8 buffer (int4* arithmetic is byte-addressed;
    // see the R3a fix), then reinterpret. size_B/2 bytes per 16x16 tile.
    const int4 *bj = reinterpret_cast<const int4 *>(wbytes + j * (MMUL::size_B / 2));
    c.mac(a, aie::load_v<MMUL::size_B>(bj));
  }
  return c.to_accum();
}

#if ROLE == 0   // HEAD: seed the cascade with this slice's partial.
extern "C" void r5_cascade_head(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                output_cascade<acc32> *cout) {
  aie::detail::cascade_stream_helper<acc32, MMUL::size_C>::writeincr(cout, kslice_partial(pA, wbytes));
}
#elif ROLE == 1 // MIDDLE: add cascade-in + this slice, pass on.
extern "C" void r5_cascade_mid(const int8 *__restrict pA, const int8 *__restrict wbytes,
                               input_cascade<acc32> *cin, output_cascade<acc32> *cout) {
  ACC in = aie::detail::cascade_stream_helper<acc32, MMUL::size_C>::readincr(cin);
  ACC sum = add(in, kslice_partial(pA, wbytes));
  aie::detail::cascade_stream_helper<acc32, MMUL::size_C>::writeincr(cout, sum);
}
#else           // TAIL: add cascade-in + this slice, STORE C (once).
extern "C" void r5_cascade_tail(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                input_cascade<acc32> *cin, int32 *__restrict pC) {
  ACC in = aie::detail::cascade_stream_helper<acc32, MMUL::size_C>::readincr(cin);
  ACC sum = add(in, kslice_partial(pA, wbytes));
  aie::store_v(pC, sum.to_vector<int32>());
}
#endif
