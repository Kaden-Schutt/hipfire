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
#include "gemm_hfq4g256_residual_mmq_gfx906_body.cuh"

#define MMQ_X_VAL 48

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_hfq4g256_residual_mmq_gfx906_x48(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add
) {
    mmq_body_templated<MMQ_X_VAL, true, -1>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_hfq4g256_residual_mmq_gfx906_full_add_x48(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int /*add*/
) {
    mmq_body_templated<MMQ_X_VAL, false, 1>(A, Xq, Y, M, K, N, 1);
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_hfq4g256_residual_mmq_gfx906_full_set_x48(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int /*add*/
) {
    mmq_body_templated<MMQ_X_VAL, false, 0>(A, Xq, Y, M, K, N, 0);
}
