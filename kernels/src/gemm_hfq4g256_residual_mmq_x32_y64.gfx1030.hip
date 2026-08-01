// MMQ residual variant — mmq_x=32, MMQ_Y=64. Applies MQ3 phase-2 finding.
//
// On HFQ3 RDNA2 the y=64 variant cuts per-WG LDS budget (~26 KB → ~15 KB),
// doubles theoretical CU occupancy (24% → 48%), and won +8% kernel-level
// over y=128. HFQ4 has the same kernel topology and LDS layout, so the
// same occupancy mechanism applies. Microbench will confirm whether the
// magnitude transfers (HFQ4 has a cheaper unpack so the relative
// memory-bound-vs-compute-bound balance differs slightly).

#define MMQ_X 32
#define MMQ_Y 64
#define KERNEL_NAME gemm_hfq4g256_residual_mmq_x32_y64
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
#include "gemm_hfq4g256_residual_mmq_body.cuh"
