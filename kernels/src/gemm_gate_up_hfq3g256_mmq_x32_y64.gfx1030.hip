// HFQ3 gate_up MMQ — mmq_x=32, MMQ_Y=64. Higher-occupancy variant.
//
// Same hypothesis as the residual y64 sibling: halve the row tile to cut
// LDS budget from ~26 KB to ~15 KB per WG, allowing 4 WGs/CU instead of
// 2 → ~48% occupancy. Gate_up is the dominant kernel by wall-time
// (55.6% of MMQ-mode prefill per the issue #300 rocprof attribution),
// so a y64 win here compounds across both the FFN preamble AND the LA
// preamble qkv+z (which routes through this same kernel via the
// qkvza split-routing path added in 05d60aa9).

#define MMQ_X 32
#define MMQ_Y 64
#define KERNEL_NAME gemm_gate_up_hfq3g256_mmq_x32_y64
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
