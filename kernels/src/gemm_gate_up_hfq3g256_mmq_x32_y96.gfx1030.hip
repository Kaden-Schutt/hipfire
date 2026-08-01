// HFQ3 gate_up MMQ — mmq_x=32, MMQ_Y=96. Sweet-spot occupancy probe.
//
// The y=64 variant lost (kept 50% per-WG work, 4 WG/CU = 48% occupancy)
// and y=128 wins (full per-WG work, 2 WG/CU = 24% occupancy). y=96 tests
// the middle ground:
//   - LDS budget: 96 × 40 × 4 + 96 × 8 + 32 × 36 × 4 = 20736 B (~20 KB)
//   - 3 WGs/CU fit in 64 KB cap → ~36% occupancy
//   - Per-WG work: 75% of y=128 (vs y=64's 50%)
//
// If the residual/gate_up asymmetry is about dispatch amortization vs
// occupancy tradeoff, y=96 should sit at a local optimum for gate_up.

#define MMQ_X 32
#define MMQ_Y 96
#define KERNEL_NAME gemm_gate_up_hfq3g256_mmq_x32_y96
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
#include "gemm_gate_up_hfq3g256_mmq_body.cuh"
