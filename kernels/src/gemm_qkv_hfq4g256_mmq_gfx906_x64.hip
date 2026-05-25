// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include "gemm_qkv_hfq4g256_mmq_gfx906_body.cuh"

#define MMQ_X_VAL 64

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_qkv_hfq4g256_mmq_gfx906_x64(
    const char* __restrict__ A_q,
    const char* __restrict__ A_k,
    const char* __restrict__ A_v,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_q,
    float* __restrict__ Y_k,
    float* __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    qkv_mmq_body_templated<MMQ_X_VAL, true>(A_q, A_k, A_v, Xq, Y_q, Y_k, Y_v, q_m, k_m, v_m, K, N);
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_qkv_hfq4g256_mmq_gfx906_full_set_x64(
    const char* __restrict__ A_q,
    const char* __restrict__ A_k,
    const char* __restrict__ A_v,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y_q,
    float* __restrict__ Y_k,
    float* __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    qkv_mmq_body_templated<MMQ_X_VAL, false>(A_q, A_k, A_v, Xq, Y_q, Y_k, Y_v, q_m, k_m, v_m, K, N);
}
