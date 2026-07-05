// R9 — load-reuse register-tiled microbench (aie2/XDNA1). R8 proved the matrix unit is
// fast (~1 ns/mmul) when data is register-resident, and that R5-R7's streaming wall was
// LOADS: the K-loop did 2 L1 loads (A,W) per mac with zero reuse. R9 tiles the OUTPUT
// MTxNT (in base-op tiles): each K-step loads MT A-tiles + NT W-tiles from L1 and issues
// MT*NT macs, so each A row is reused across NT columns and each W tile across MT rows.
// Reuse = MT*NT/(MT+NT) macs-per-load; 1x1 = 0.5 (R5 baseline), 2x2 = 1.0, 4x4 = 2.0.
// The hot loop streams A/W from L1 (real load traffic, unlike R8's single resident tile),
// so ns/mmul here is the achievable rate under reuse — the real-GEMM number.
//
// Base op is int4 <4,16,8> (R8's fastest: ~489 GMAC/s/core). Build knobs: MT NT KT REPEAT.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 3               // output tile rows, in base-op (m=4) units (3x3 = measured optimum)
#endif
#ifndef NT
#define NT 3               // output tile cols, in base-op (n=8) units (9 accs fit; 12 spill)
#endif
#ifndef KT
#define KT 16              // K-tiles resident in L1 (working set; loads cycle over these)
#endif
#ifndef REPEAT
#define REPEAT 20000       // outer passes over the KT working set (timing length)
#endif

using MMUL = aie::mmul<4, 16, 8, int8, int4>;
static constexpr int SA = MMUL::size_A;       // 64 int8 / A-tile
static constexpr int SBb = MMUL::size_B / 2;  // 64 bytes / W-tile (two int4 per byte)
static constexpr int SC = MMUL::size_C;       // 32 i32 / C-tile

// A laid out [MT][KT] tiles, W [NT][KT] tiles, C [MT][NT] tiles.
extern "C" void r9_ubench(const int8 *__restrict A, const int8 *__restrict Wb,
                          int32 *__restrict C) {
  MMUL c[MT][NT];
  // Seed pass (k=0): mul zeroes each accumulator; all later steps mac.
  {
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
  for (int rep = 0; rep < REPEAT; rep++)
    for (int k = (rep == 0 ? 1 : 0); k < KT; k++)
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
