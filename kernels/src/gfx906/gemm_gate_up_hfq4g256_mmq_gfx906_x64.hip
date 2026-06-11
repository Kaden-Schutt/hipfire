// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include "gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh"

#define MMQ_X_VAL 64

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_gate_up_hfq4g256_mmq_gfx906_x64(
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
void gemm_gate_up_hfq4g256_mmq_gfx906_full_set_x64(
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
