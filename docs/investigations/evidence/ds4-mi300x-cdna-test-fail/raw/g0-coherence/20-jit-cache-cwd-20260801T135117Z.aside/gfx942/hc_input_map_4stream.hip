// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 mHC input mapping (Phase 3).
//
// Computes x_in = A_l · X_l where:
//   A_l ∈ R^{1 × n_hc}    (input mapping, sigmoid-bounded)
//   X_l ∈ R^{n_hc × d}    (residual streams)
//   x_in ∈ R^{1 × d}      (transform input)
//
//   x_in[d] = sum_s(A_l[s] * X_l[s, d])
//
// One thread per hidden-dim element. A_l fits in registers.

#include <hip/hip_runtime.h>

#define HC_MULT 4
#define BLOCK_HIDDEN 256

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_input_map_4stream(
    const float* __restrict__ a_vec,   // [HC_MULT] fp32 (post-sigmoid)
    const float* __restrict__ x_in,    // [HC_MULT, hidden] fp32 residual streams
    float*       __restrict__ x_out,   // [hidden] fp32 transform input
    int hidden
) {
    const int tid = threadIdx.x;
    const int d = blockIdx.x * BLOCK_HIDDEN + tid;
    if (d >= hidden) return;

    float a0 = a_vec[0];
    float a1 = a_vec[1];
    float a2 = a_vec[2];
    float a3 = a_vec[3];

    float v = a0 * x_in[0 * hidden + d]
            + a1 * x_in[1 * hidden + d]
            + a2 * x_in[2 * hidden + d]
            + a3 * x_in[3 * hidden + d];
    x_out[d] = v;
}
