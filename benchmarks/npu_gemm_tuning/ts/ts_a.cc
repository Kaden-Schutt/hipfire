// Isolated tensor-buffer-stream reshuffle test. Reads ROW-MAJOR A[(MT*4)][(KCHUNK*16)]
// and writes the pack_a tile-major layout (tile = mt*KCHUNK+k, within-tile m*16+kk) via
// an in-core aie::tensor_descriptor — NO CPU marshaling, NO DMA reshuffle. This is the
// unit that de-risks porting R6 to tensor buffer streams: if the NPU output equals the
// CPU pack_a output, the descriptor strides are correct and the R6 A-read can drop the
// marshaler and consume row-major activations directly.
//
// Descriptor (steps are in LEAF-vector = 16-int8 units; make_tensor_descriptor scales by
// vector::bytes() and computes the nested-iteration rollback automatically):
//   dims[0]=mt: num=MT,     step=4*KCHUNK   (mt stride = MR*Kb bytes)
//   dims[1]=k : num=KCHUNK, step=1          (k  stride = MK=16 bytes)
//   dims[2]=m : num=4,      step=KCHUNK     (m  stride = Kb bytes)
// Rank-3 collapses to one dim_3d stream level, so a flat pop loop walks m(inner)->k->mt.
#include <aie_api/aie.hpp>

#ifndef MT
#define MT 2
#endif
#ifndef KCHUNK
#define KCHUNK 2
#endif

extern "C" void ts_a(const int8 *__restrict pA, int8 *__restrict pOut) {
  auto desc = aie::make_tensor_descriptor<int8, 16>(
      aie::tensor_dim(MT, 4 * KCHUNK),   // mt
      aie::tensor_dim(KCHUNK, 1),        // k
      aie::tensor_dim(4u, KCHUNK));      // m
  auto ts = aie::make_tensor_buffer_stream(pA, desc);
  const int ntiles = MT * KCHUNK * 4;    // one 16-vector (a tile row) per pop
  for (int t = 0; t < ntiles; ++t) {
    aie::vector<int8, 16> r;
    ts >> r;
    aie::store_v(pOut + t * 16, r);
  }
}
