// R1a: L3->L1 weight-feed bandwidth. The worker streams int8 tiles from L3 via
// an objectFIFO; this kernel touches every byte cheaply (one vector-add per 64B,
// one reduce at the end) so the load can't be DCE'd and the DMA genuinely moves
// the data. Compute is trivial vs the transfer, so the loop is DMA-bound and the
// measured bytes/time is the feed bandwidth.
#include <aie_api/aie.hpp>

#ifndef TILE_N
#define TILE_N 4096            // int8 elements per streamed tile
#endif

extern "C" void feed_sum(const int8 *__restrict tile, int32 *__restrict acc) {
#ifdef MINIMAL
  // touch only the first vector: DMA still moves the whole tile, so if time is
  // unchanged the bottleneck is the DMA feed, not this consume loop.
  acc[0] = aie::reduce_add(aie::load_v<64>(tile));
#else
  aie::vector<int8, 64> s = aie::zeros<int8, 64>();
  for (int i = 0; i < TILE_N; i += 64)
    s = aie::add(s, aie::load_v<64>(tile + i));
  acc[0] = aie::reduce_add(s);   // single reduce -> DCE guard
#endif
}
