// R0: native aie2p MAC-conf microkernels — establish the compute ceiling.
// Each kernel keeps NACC independent mmul accumulators so consecutive VMACs are
// dependency-free; the steady-state loop II in the disasm then reveals the real
// throughput (VMACs/cycle), which reconciles 58 vs 116 TOPS.
#include <aie_api/aie.hpp>

// int8 x int8, 8x8x8 DENSE  -> mac_8x8_8x8_conf, 512 MACs/instr
extern "C" void r0_i8i8(const int8 *__restrict pA, const int8 *__restrict pB,
                        int32 *__restrict pC) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8>;
  constexpr int NACC = 8;
  aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pA);
  aie::vector<int8, MMUL::size_B> b = aie::load_v<MMUL::size_B>(pB);
  MMUL acc[NACC];
  for (int k = 0; k < NACC; k++) acc[k].mul(a, b);
  for (int i = 0; i < 64; i++)
    for (int k = 0; k < NACC; k++) acc[k].mac(a, b);
  for (int k = 0; k < NACC; k++)
    aie::store_v(pC + k * MMUL::size_C, acc[k].template to_vector<int32>());
}

// int8 x int4, 4x16x16 DENSE -> mac_4x16_16x16_conf, 1024 MACs/instr (W4A8 / Oq4)
extern "C" void r0_i8i4(const int8 *__restrict pA, const int4 *__restrict pB,
                        int32 *__restrict pC) {
  using MMUL = aie::mmul<4, 16, 16, int8, int4>;
  constexpr int NACC = 8;
  aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pA);
  aie::vector<int4, MMUL::size_B> b = aie::load_v<MMUL::size_B>(pB);
  MMUL acc[NACC];
  for (int k = 0; k < NACC; k++) acc[k].mul(a, b);
  for (int i = 0; i < 64; i++)
    for (int k = 0; k < NACC; k++) acc[k].mac(a, b);
  for (int k = 0; k < NACC; k++)
    aie::store_v(pC + k * MMUL::size_C, acc[k].template to_vector<int32>());
}
