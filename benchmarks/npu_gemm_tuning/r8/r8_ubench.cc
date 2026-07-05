// R8 microbench — isolate the AIE-ML gen1 (Phoenix/aie2) matrix-unit throughput.
//
// R5-R7 all used aie::mmul<4,16,8,int8,int4>, which on aie2 is a VIRTUAL op: int4 has
// no native MAC, so it unpacks to int8 and issues composite 2x8x8 real MACs. So those
// runs measured the virtual op (unpack + composite), never the matrix unit's true rate.
// This kernel hammers ONE chosen mmul shape REPEAT times over a single RESIDENT tile
// (loaded once into registers — zero feed, zero sync in the hot loop) and stores C once,
// so npu_gemm_bench's per-dispatch time / (NACC*REPEAT*M*K*N) is pure ns/MAC.
//
// SHAPE selects the op (build -DSHAPE=n):
//   0 = mmul<4,16,8,int8,int4>  virtual int4 (the R5-R7 baseline)
//   1 = mmul<2,8,8,int8,int8>   NATIVE 2x8x8 int8
//   2 = mmul<4,8,4,int8,int8>   NATIVE 4x8x4 int8
//   3 = mmul<4,16,8,int8,int8>  VIRTUAL <4,16,8> but int8 (isolates the int4-unpack tax)
//   4 = mmul<8,8,8,int8,int8>   <8,8,8> int8 (is 8x8x8 native or composite?)
#include <aie_api/aie.hpp>

#ifndef SHAPE
#define SHAPE 0
#endif
#ifndef REPEAT
#define REPEAT 20000       // hot-loop mmul iterations per accumulator (compile-time)
#endif
#ifndef NACC
#define NACC 4             // independent accumulators to hide mmul latency
#endif

#if   SHAPE == 0
using TB = int4;  using MMUL = aie::mmul<4, 16, 8, int8, int4>;
#elif SHAPE == 1
using TB = int8;  using MMUL = aie::mmul<2, 8, 8, int8, int8>;
#elif SHAPE == 2
using TB = int8;  using MMUL = aie::mmul<4, 8, 4, int8, int8>;
#elif SHAPE == 3
using TB = int8;  using MMUL = aie::mmul<4, 16, 8, int8, int8>;
#elif SHAPE == 4
using TB = int8;  using MMUL = aie::mmul<8, 8, 8, int8, int8>;
#endif
using ACC = aie::accum<acc32, MMUL::size_C>;

// A/W come in as raw int8 buffers; reinterpret W to its element type (int4* is byte
// addressed, so size_B int4 occupies size_B/2 bytes — the R3a rule; int8 is 1:1).
extern "C" void r8_ubench(const int8 *__restrict A, const int8 *__restrict Wb,
                          int32 *__restrict C) {
  auto a = aie::load_v<MMUL::size_A>(A);
  auto w = aie::load_v<MMUL::size_B>(reinterpret_cast<const TB *>(Wb));
  MMUL c[NACC];
#pragma unroll
  for (int p = 0; p < NACC; p++)
    c[p].mul(a, w);
  // Hot loop: NACC independent MACs on the SAME resident a,w — no loads, no sync.
  for (int r = 0; r < REPEAT; r++)
      chess_prepare_for_pipelining
#pragma unroll
    for (int p = 0; p < NACC; p++)
      c[p].mac(a, w);
  ACC acc = c[0].to_accum();
#pragma unroll
  for (int p = 1; p < NACC; p++)
    acc = add(acc, c[p].to_accum());
  aie::store_v(C, acc.template to_vector<int32>());
}
