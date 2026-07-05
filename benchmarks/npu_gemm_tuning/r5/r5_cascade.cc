// R5: one core of a K-cascade W4A8 column. Each core computes 4 M-block partials
// with the R2a II=1 recipe (4 named accumulators sharing ONE resident weight tile
// — pure register macs, ~1.6 TOPS/core), and the 512-bit cascade stream carries all
// four accumulators core->core (put_mcd/get_scd) so C accumulates in-flight and the
// tail stores it ONCE — no per-tile C reload (the trap pinning the memtile dataflow
// and SOTA FastFlowLM to ~5 TOPS).
//
// Cascade API (from aie_kernels/aie2/cascade_mm.cc): the 512-bit cascade moves one
// v16 per beat via put_mcd / get_scd. The mmul accumulator is size_C acc32 =
// size_C/16 beats. The graph wires the physical link with aie.cascade_flow(src,dst).
// Built -DROLE={0 head,1 mid,2 tail}.
//
// ARCH: the aie2p (Strix Halo/npu2) int8xint4 mmul is <4,16,16> (size_C=64) and the
// cascade read is the get_scd_v16acc32 builtin, keeping the sum in the acc32 domain.
// XDNA1/aie2 (Phoenix/npu1) provides int8xint4 only as <4,16,8> (size_C=32) and has
// NO get_scd_v16acc32 — its cascade read is get_scd_v16int32, so we sum in the int32
// domain (partials are exact small ints; acc32->int32 at shift 0 is lossless). Select
// XDNA1 with -DNPU_AIE2 -DMMUL_N=8; aie2p is the default.
#include <aie_api/aie.hpp>

#ifndef INNER
#define INNER 0            // reuse passes of the resident tile (compute-rate knob)
#endif
#ifndef ROLE
#define ROLE 2             // 0=head, 1=mid, 2=tail (stores C), 3=solo (no cascade)
#endif
#ifndef MMUL_N
#define MMUL_N 16          // aie2p int8xint4 = <4,16,16>; XDNA1/aie2 = <4,16,8> (-DMMUL_N=8)
#endif
#ifndef INNER
#define INNER 1            // R6 probe: recompute the K-slice INNER times over the SAME
#endif                     // resident L1 tiles (no extra feed) to isolate feed- vs core-bound.
#ifndef NACC
#define NACC 4             // R7: independent K-partial accumulators to hide mmul latency
#endif                     // (II~1). NACC=1 = the R5/R6 single-accumulator chain. KSLICE%NACC==0.

using MMUL = aie::mmul<4, 16, MMUL_N, int8, int4>;
using ACC = aie::accum<acc32, MMUL::size_C>;   // 4*MMUL_N acc32 partial C
static constexpr int CN = MMUL::size_C;
static constexpr int BEATS = CN / 16;          // cascade beats (16 acc32/int32 each)

// Load this core's A / W tile for K-step j. Weight stride is in BYTES on the int8
// buffer, then reinterpret (int4* arithmetic is byte-addressed — the R3a fix);
// size_B/2 bytes per 16xN tile.
static inline aie::vector<int8, MMUL::size_A> ldA(const int8 *__restrict pA, int j) {
  return aie::load_v<MMUL::size_A>(pA + j * MMUL::size_A);
}
static inline aie::vector<int4, MMUL::size_B> ldW(const int8 *__restrict wbytes, int j) {
  return aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(wbytes + j * (MMUL::size_B / 2)));
}

// This core's K-slice partial. R5/R6 used ONE accumulator (every mac chained on the
// prior). R7 splits the K contraction across NACC independent accumulators (interleaved
// over the K loop, tree-summed at the end — exact int32). This gives a real but MODEST
// ~1.35x (NACC=4 optimal; NACC=8 spills the accumulator regs). The R7 measurement
// showed the feed-free rate is ~18ns/mmul at NACC=8 vs ~20ns at NACC=1 — i.e. the mmul
// was NOT accumulator-latency-stalled; mac_4x16_16x8_conf just runs ~32 cyc/mmul on
// AIE-ML gen1 (Phoenix). NACC helps by overlapping loads/sync, not by unstalling MACs.
static_assert(KSLICE % NACC == 0, "KSLICE must be a multiple of NACC");
static inline ACC kslice_partial(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  MMUL c[NACC];
  // Seed each independent chain with a mul (zeroes its accumulator).
#pragma unroll
  for (int p = 0; p < NACC; p++)
    c[p].mul(ldA(pA, p), ldW(wbytes, p));
  // Remaining K steps, NACC independent macs per iteration.
  for (int j = NACC; j < KSLICE; j += NACC)
      chess_prepare_for_pipelining
#pragma unroll
    for (int p = 0; p < NACC; p++)
      c[p].mac(ldA(pA, j + p), ldW(wbytes, j + p));
#if INNER > 1
  // R6 probe: extra feed-free MAC passes over the resident tiles (MACs scale, feed does not).
  for (int r = 1; r < INNER; r++)
    for (int j = 0; j < KSLICE; j += NACC)
        chess_prepare_for_pipelining
#pragma unroll
      for (int p = 0; p < NACC; p++)
        c[p].mac(ldA(pA, j + p), ldW(wbytes, j + p));
#endif
  // Tree-sum the NACC partials into one accumulator.
  ACC acc = c[0].to_accum();
#pragma unroll
  for (int p = 1; p < NACC; p++)
    acc = add(acc, c[p].to_accum());
  return acc;
}

#if defined(NPU_AIE2)
// XDNA1/aie2: cascade carries v16int32; sum in the int32 domain.
using CVEC = aie::vector<int32, CN>;
static inline CVEC to_cvec(ACC a) { return a.template to_vector<int32>(); }
static inline CVEC csum(CVEC a, CVEC b) { return aie::add(a, b); }
static inline void store_c(int32 *__restrict pC, CVEC v) { aie::store_v(pC, v); }
static inline void cascade_put(CVEC v) {
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    put_mcd((v16int32)v.template extract<16>(i));
}
static inline CVEC cascade_get() {
  CVEC v;
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    v.insert(i, aie::vector<int32, 16>(get_scd_v16int32()));
  return v;
}
#else
// aie2p: keep the accumulator in the acc32 domain end-to-end.
using CVEC = ACC;
static inline CVEC to_cvec(ACC a) { return a; }
static inline CVEC csum(CVEC a, CVEC b) { return add(a, b); }
static inline void store_c(int32 *__restrict pC, CVEC a) { aie::store_v(pC, a.template to_vector<int32>()); }
static inline void cascade_put(CVEC acc) {
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    put_mcd(acc.template extract<16>(i).to_native());
}
static inline CVEC cascade_get() {
  ACC acc;
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    acc.insert(i, aie::accum<acc32, 16>(get_scd_v16acc32()));
  return acc;
}
#endif

#if ROLE == 0   // HEAD: seed the cascade with the four partials.
extern "C" void r5_cascade_head(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  cascade_put(to_cvec(kslice_partial(pA, wbytes)));
}
#elif ROLE == 1 // MIDDLE: add cascade-in + local partials, pass on.
extern "C" void r5_cascade_mid(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  cascade_put(csum(cascade_get(), to_cvec(kslice_partial(pA, wbytes))));
}
#elif ROLE == 2 // TAIL: add cascade-in + local partials, STORE C once.
extern "C" void r5_cascade_tail(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                int32 *__restrict pC) {
  store_c(pC, csum(cascade_get(), to_cvec(kslice_partial(pA, wbytes))));
}
#endif
