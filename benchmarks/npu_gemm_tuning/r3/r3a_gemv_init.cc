// R3a init entry point (see r3a_gemv.cc): STORES the first super-tile's partial,
// reseeding the resident K accumulator pC so it does not carry stale state from a
// prior dispatch (the tile-local pC persists across invocations). Kept in its own
// translation unit so binding both r3a_matvec_init and r3a_matvec in one core does
// not collide on duplicate symbols.
#include "r3a_gemv_common.h"

extern "C" void r3a_matvec_init(const int8 *__restrict pAchunk,
                                const int8 *__restrict wbytes, int32 *__restrict pC) {
  auto partial = r3a_super_partial(pAchunk, wbytes);
  aie::store_v(pC, partial);
}
