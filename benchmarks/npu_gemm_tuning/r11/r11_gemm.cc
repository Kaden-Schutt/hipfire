// R11 — L2-tiled streaming GEMM block (aie2/XDNA1). R10 showed the single-core streaming
// GEMM is DMA-feed-bound: rate scaled ~linearly with arithmetic intensity (=NT*4 mac/byte
// for a bare MTxNT register tile), ~70 GMAC/s at intensity 12 vs R9's 150 resident. The
// fix is a second tiling level: hold a larger LMxLN base-tile output block's C in L1 and
// sweep the 3x3 register tile (MTxNT) over it. The A/W stripes for the whole block are
// DMA'd once from DDR but reused across all sub-tiles, so DMA intensity rises to
// 8*LM*LN/(LM+LN) mac/byte — climbing back toward the register-tiling rate.
//
// A laid out [LM][KT] tiles, W [LN][KT] tiles, C [LM][LN] tiles. LM%MT==0, LN%NT==0.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 3               // register tile rows (R9 optimum 3x3, 9 accumulators)
#endif
#ifndef NT
#define NT 3
#endif
#ifndef LM
#define LM 6               // L1 block rows in base-tiles (multiple of MT)
#endif
#ifndef LN
#define LN 6               // L1 block cols in base-tiles (multiple of NT)
#endif
#ifndef KT
#define KT 16              // K-depth in base-op (k=16) tiles
#endif

using MMUL = aie::mmul<4, 16, 8, int8, int4>;
static constexpr int SA = MMUL::size_A;       // 64 int8 / A-tile
static constexpr int SBb = MMUL::size_B / 2;  // 64 bytes / W-tile
static constexpr int SC = MMUL::size_C;       // 32 i32 / C-tile

extern "C" void r11_gemm(const int8 *__restrict A, const int8 *__restrict Wb,
                         int32 *__restrict C) {
  // Sweep the MTxNT register tile over the LMxLN L1 block. A[im..] rows are reused across
  // the jn loop; W[jn..] cols across the im loop; both were DMA'd once for the block.
  for (int im = 0; im < LM; im += MT)
    for (int jn = 0; jn < LN; jn += NT) {
      MMUL c[MT][NT];
      {                                        // seed k=0
        aie::vector<int8, SA> a[MT];
        aie::vector<int4, MMUL::size_B> w[NT];
#pragma unroll
        for (int i = 0; i < MT; i++) a[i] = aie::load_v<SA>(A + ((im + i) * KT) * SA);
#pragma unroll
        for (int j = 0; j < NT; j++) w[j] = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(Wb + ((jn + j) * KT) * SBb));
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
        for (int i = 0; i < MT; i++) a[i] = aie::load_v<SA>(A + ((im + i) * KT + k) * SA);
#pragma unroll
        for (int j = 0; j < NT; j++) w[j] = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(Wb + ((jn + j) * KT + k) * SBb));
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int j = 0; j < NT; j++) c[i][j].mac(a[i], w[j]);
      }
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
          aie::store_v(C + ((im + i) * LN + (jn + j)) * SC, c[i][j].to_accum().template to_vector<int32>());
    }
}
