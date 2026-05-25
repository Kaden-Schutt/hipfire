// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Y=64 occupancy variant — halves the per-WG LDS X-tile and the
// per-lane accumulator register footprint. 2× more WGs per grid →
// better gfx906 CU utilization at modest grid sizes. See plan §6
// step 5. The body asserts MMQ_Y ∈ {64, 128}; this wrapper sets
// MMQ_Y=64 before including the body.

#define MMQ_Y 64
#include "gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh"

#define MMQ_X_VAL 32

extern "C" __global__ __launch_bounds__(256, 4)
void gemm_gate_up_hfq4g256_mmq_gfx906_x32_y64(
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

extern "C" __global__ __launch_bounds__(256, 4)
void gemm_gate_up_hfq4g256_mmq_gfx906_full_set_x32_y64(
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
