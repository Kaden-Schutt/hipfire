// R10 — one output block of a REAL streaming GEMM (aie2/XDNA1). This is R9's 3x3
// load-reuse inner loop (r9_ubench) with REPEAT=1: it computes one C[MT*4 x NT*8] block
// = A[MT*4 x KT*16] . W[KT*16 x NT*8] over a KT-deep K contraction, from A/W tiles that
// the core just acquired from an objectfifo. R9 measured this rate with the working set
// L1-RESIDENT (DMA'd once, reused); R10 streams a FRESH block from DDR per call (double-
// buffered objectfifos), so end-to-end GMAC/s here includes the shim->L1 DMA feed — the
// number that says whether ~150 GMAC/s/core survives real feed or goes DMA-bound.
//
// A laid out [MT][KT] tiles, W [NT][KT] tiles, C [MT][NT] tiles. Base op int4 <4,16,8>.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 3               // output tile rows in base-op (m=4) units — R9 optimum 3x3
#endif
#ifndef NT
#define NT 3               // output tile cols in base-op (n=8) units (9 accs fit, 12 spill)
#endif
#ifndef KT
#define KT 16              // K-depth of this block, in base-op (k=16) tiles
#endif

using MMUL = aie::mmul<4, 16, 8, int8, int4>;
static constexpr int SA = MMUL::size_A;       // 64 int8 / A-tile
static constexpr int SBb = MMUL::size_B / 2;  // 64 bytes / W-tile
static constexpr int SC = MMUL::size_C;       // 32 i32 / C-tile

extern "C" void r10_gemm(const int8 *__restrict A, const int8 *__restrict Wb,
                         int32 *__restrict C) {
  MMUL c[MT][NT];
  {                                            // seed pass (k=0): mul zeroes each accumulator
    aie::vector<int8, SA> a[MT];
    aie::vector<int4, MMUL::size_B> w[NT];
#pragma unroll
    for (int i = 0; i < MT; i++) a[i] = aie::load_v<SA>(A + (i * KT) * SA);
#pragma unroll
    for (int j = 0; j < NT; j++) w[j] = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(Wb + (j * KT) * SBb));
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int j = 0; j < NT; j++) c[i][j].mul(a[i], w[j]);
  }
  for (int k = 1; k < KT; k++)
      chess_prepare_for_pipelining {
    aie::vector<int8, SA> a[MT];
    aie::vector<int4, MMUL::size_B> w[NT];
#pragma unroll
    for (int i = 0; i < MT; i++) a[i] = aie::load_v<SA>(A + (i * KT + k) * SA);
#pragma unroll
    for (int j = 0; j < NT; j++) w[j] = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(Wb + (j * KT + k) * SBb));
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int j = 0; j < NT; j++) c[i][j].mac(a[i], w[j]);
  }
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT; j++)
      aie::store_v(C + (i * NT + j) * SC, c[i][j].to_accum().template to_vector<int32>());
}
