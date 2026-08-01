// MMQ residual variant — mmq_x=32, MMQ_Y=32. Push y64 further.
//
// y64 cut LDS budget from ~26KB to ~15KB (2→4 WG/CU). y32 cuts to ~10KB
// (potentially 6 WG/CU → ~72% theoretical occupancy). Per-WG work is 1/4
// of y128, so 4× more workgroups dispatched. The risk: dispatch overhead
// and reduced latency-hiding margin per WG may exceed the occupancy win.

#define MMQ_X 32
#define MMQ_Y 32
#define KERNEL_NAME gemm_hfq3g256_residual_mmq_x32_y32
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
