// MMQ residual variant — mmq_x=32, MMQ_Y=64. Experimental occupancy probe.
//
// Halves the row tile (128 → 64) to cut per-WG LDS budget from ~26 KB
// to ~15 KB. At 64 KB LDS/CU, 4 WGs fit per CU instead of 2 → 16 waves/CU
// instead of 8 (~48% occupancy vs ~24%). Per-WG work also halves, so 2×
// more workgroups dispatched. Test whether the latency-hiding win from
// higher occupancy outweighs the per-WG-work-halving overhead.

#define MMQ_X 32
#define MMQ_Y 64
#define KERNEL_NAME gemm_hfq3g256_residual_mmq_x32_y64
#ifndef HIPFIRE_SDOT4
#define HIPFIRE_SDOT4
// gfx12 (RDNA4) dropped signed-only v_dot4_i32_i8 (`sdot4`, feature dot1-insts)
// but kept mixed-sign `sudot4`; both sign flags set is bit-identical to sdot4
// (verified 0/4096 vs a CPU reference on gfx1201). Defined in the wrapper as
// well as the body because the wrapper is what the JIT compiles.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIPFIRE_SDOT4_CALL(a, b, c, clamp) __builtin_amdgcn_sudot4(true, (a), true, (b), (c), (clamp))
#else
#define HIPFIRE_SDOT4_CALL(a, b, c, clamp) __builtin_amdgcn_sdot4((a), (b), (c), (clamp))
#endif
#endif
#include "gemm_hfq3g256_residual_mmq_body.cuh"
