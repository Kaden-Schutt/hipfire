// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

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
#include "gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh"

#define MMQ_X_VAL 16

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_gate_up_hfq4g256_mmq_gfx906_x16(
    const char* __restrict__ A_a,
    const char* __restrict__ A_b,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_a,
    float* __restrict__ Y_b,
    int m_a, int m_b,
    int K, int N
) {
    gate_up_mmq_body_templated<MMQ_X_VAL, true>(A_a, A_b, Xq, Y_a, Y_b, m_a, m_b, K, N);
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_gate_up_hfq4g256_mmq_gfx906_full_set_x16(
    const char* __restrict__ A_a,
    const char* __restrict__ A_b,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_a,
    float* __restrict__ Y_b,
    int m_a, int m_b,
    int K, int N
) {
    gate_up_mmq_body_templated<MMQ_X_VAL, false>(A_a, A_b, Xq, Y_a, Y_b, m_a, m_b, K, N);
}
