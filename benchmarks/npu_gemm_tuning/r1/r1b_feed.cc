// R1b feed kernel: identical L3->L1 weight-feed consume to R1a. One vector-add
// per 64 B (or MINIMAL: one vector) + a single reduce as a DCE guard, so the DMA
// genuinely moves every byte and the loop is DMA-bound, not compute-bound.
//
// Note: an in-kernel core-cycle read (aie::tile::current().cycles()) was tried for
// a host-sync-immune timer but does not link on this toolchain -- it lowers to an
// undefined ::get_cycles(); the aie2p kernels timestamp via event0()/event1() +
// the trace unit instead. So on-device timing here goes through the trace path
// (r1b_run.py TRACE=1), which is the decisive host-vs-device measurement.
#include <aie_api/aie.hpp>

#ifndef TILE_N
#define TILE_N 4096            // int8 elements per streamed tile
#endif

extern "C" void feed_sum(const int8 *__restrict tile, int32 *__restrict acc) {
#ifdef MINIMAL
  // touch only the first vector: the DMA still moves the whole tile, so if the
  // rate is unchanged the bottleneck is the feed, not this consume loop.
  acc[0] = aie::reduce_add(aie::load_v<64>(tile));
#else
  aie::vector<int8, 64> s = aie::zeros<int8, 64>();
  for (int i = 0; i < TILE_N; i += 64)
    s = aie::add(s, aie::load_v<64>(tile + i));
  acc[0] = aie::reduce_add(s);   // single reduce -> DCE guard
#endif
}
