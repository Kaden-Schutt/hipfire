// R5: one core of a K-cascade W4A8 column. Each core computes 4 M-block partials
// with the R2a II=1 recipe (4 named accumulators sharing ONE resident weight tile
// — pure register macs, ~1.6 TOPS/core), and the 512-bit cascade stream carries all
// four accumulators core->core (put_mcd/get_scd) so C accumulates in-flight and the
// tail stores it ONCE — no per-tile C reload (the trap pinning the memtile dataflow
// and SOTA FastFlowLM to ~5 TOPS).
//
// Cascade API (aie_kernels/aie2/cascade_mm.cc + aie2p_streams.h): put_mcd(v16acc32)/
// get_scd_v16acc32(), one 512-bit beat = 16 acc32, mmul<4,16,16> = 64 acc32 = 4 beats
// per accumulator, 16 beats for the four. aie.cascade_flow(src,dst) wires the link.
// Build -DROLE={0 head,1 mid,2 tail,3 solo(no cascade)} -DINNER=<reuse passes>.
#include <aie_api/aie.hpp>

#ifndef INNER
#define INNER 0            // reuse passes of the resident tile (compute-rate knob)
#endif
#ifndef ROLE
#define ROLE 2             // 0=head, 1=mid, 2=tail (stores C), 3=solo (no cascade)
#endif

using MMUL = aie::mmul<4, 16, 16, int8, int4>;
using ACC = aie::accum<acc32, MMUL::size_C>;   // 64 acc32 = one M-block partial
static constexpr int MB = 4;                    // M-blocks (accumulators) per core
static constexpr int BEATS = MMUL::size_C / 16; // cascade beats per accumulator (4)

struct Partials { ACC c[MB]; };

// Four M-block partials: a0..a3 (4 M rows) x one shared resident weight tile b,
// reused INNER+1 times in four independent chains -> II=1. (R2a's recipe.)
static inline Partials compute(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  aie::vector<int8, MMUL::size_A> a0 = aie::load_v<MMUL::size_A>(pA);
  aie::vector<int8, MMUL::size_A> a1 = aie::load_v<MMUL::size_A>(pA + MMUL::size_A);
  aie::vector<int8, MMUL::size_A> a2 = aie::load_v<MMUL::size_A>(pA + 2 * MMUL::size_A);
  aie::vector<int8, MMUL::size_A> a3 = aie::load_v<MMUL::size_A>(pA + 3 * MMUL::size_A);
  aie::vector<int4, MMUL::size_B> b = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(wbytes));
  MMUL c0, c1, c2, c3;
  c0.mul(a0, b);
  c1.mul(a1, b);
  c2.mul(a2, b);
  c3.mul(a3, b);
  for (int r = 0; r < INNER; r++)
      chess_prepare_for_pipelining {
    c0.mac(a0, b);
    c1.mac(a1, b);
    c2.mac(a2, b);
    c3.mac(a3, b);
  }
  return {{c0.to_accum(), c1.to_accum(), c2.to_accum(), c3.to_accum()}};
}

static inline void put_acc(ACC acc) {
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    put_mcd(acc.template extract<16>(i).to_native());
}
static inline ACC get_acc() {
  ACC acc;
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    acc.insert(i, aie::accum<acc32, 16>(get_scd_v16acc32()));
  return acc;
}
static inline void store4(int32 *pC, Partials p) {
#pragma unroll
  for (int m = 0; m < MB; m++)
    aie::store_v(pC + m * MMUL::size_C, p.c[m].to_vector<int32>());
}

#if ROLE == 0   // HEAD: seed the cascade with the four partials.
extern "C" void r5_cascade_head(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  Partials p = compute(pA, wbytes);
#pragma unroll
  for (int m = 0; m < MB; m++) put_acc(p.c[m]);
}
#elif ROLE == 1 // MIDDLE: add cascade-in + local partials, pass on.
extern "C" void r5_cascade_mid(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  Partials p = compute(pA, wbytes);
#pragma unroll
  for (int m = 0; m < MB; m++) put_acc(add(get_acc(), p.c[m]));
}
#elif ROLE == 2 // TAIL: add cascade-in + local partials, STORE C once.
extern "C" void r5_cascade_tail(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                int32 *__restrict pC) {
  Partials p = compute(pA, wbytes);
  Partials out;
#pragma unroll
  for (int m = 0; m < MB; m++) out.c[m] = add(get_acc(), p.c[m]);
  store4(pC, out);
}
#else           // SOLO (diagnostic): compute + store, NO cascade.
extern "C" void r5_cascade_solo(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                int32 *__restrict pC) {
  store4(pC, compute(pA, wbytes));
}
#endif
