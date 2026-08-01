// HFQ4 gate_up MMQ — mmq_x=32, MMQ_Y=128. Long-prefill tile.
//
// MMQ_Y stays at 128 (MQ3 phase-2 finding: y64/y96 regress on gate_up
// because per-WG work doubles vs residual, magnifying cache pressure).

#define MMQ_X 32
#define KERNEL_NAME gemm_gate_up_hfq4g256_mmq_x32
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
#include "gemm_gate_up_hfq4g256_mmq_body.cuh"
