// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 tree-aware conv1d + SiLU + Q/K/V split.
//
// The generic tree kernel owns one channel per thread and loops over every
// candidate token. Tree conv does not mutate conv_state, so each (token,
// channel) cell is independent after parent_indices[] has been materialized.
// This variant exposes token parallelism through grid.y.

#if defined(__gfx1151__)
#define HIPFIRE_GFX1151 1
#else
#define HIPFIRE_GFX1151 0
#endif

extern "C" __global__ void conv1d_silu_split_tree_f32_gfx1151(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ state,
    const int* __restrict__ parent_indices,
    int k_dim,
    int v_dim,
    int n_tokens
) {
#if HIPFIRE_GFX1151
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels || t >= n_tokens) return;

    const float w0 = weight[c * 4 + 0];
    const float w1 = weight[c * 4 + 1];
    const float w2 = weight[c * 4 + 2];
    const float w3 = weight[c * 4 + 3];
    const float s0 = state[c * 3];
    const float s1 = state[c * 3 + 1];
    const float s2 = state[c * 3 + 2];

    const int p1 = parent_indices[t];
    const int p2 = (p1 >= 0) ? parent_indices[p1] : p1 - 1;
    const int p3 = (p2 >= 0) ? parent_indices[p2] : p2 - 1;

    const float x_self = input[t * n_channels + c];
    const float x_p1 = (p1 >= 0) ? input[p1 * n_channels + c]
                     : (p1 == -1) ? s0
                     : (p1 == -2) ? s1
                     : s2;
    const float x_p2 = (p2 >= 0) ? input[p2 * n_channels + c]
                     : (p2 == -1) ? s0
                     : (p2 == -2) ? s1
                     : s2;
    const float x_p3 = (p3 >= 0) ? input[p3 * n_channels + c]
                     : (p3 == -1) ? s0
                     : (p3 == -2) ? s1
                     : s2;

    const float y = w3 * x_self + w2 * x_p1 + w1 * x_p2 + w0 * x_p3;
    const float result = y / (1.0f + expf(-y));

    if (c < k_dim) {
        q_out[t * k_dim + c] = result;
    } else if (c < 2 * k_dim) {
        k_out[t * k_dim + (c - k_dim)] = result;
    } else {
        v_out[t * v_dim + (c - 2 * k_dim)] = result;
    }
#else
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && n_tokens == 0) {
        q_out[0] = k_out[0] + v_out[0] + input[0] + weight[0] + state[0]
                 + (float)parent_indices[0];
    }
#endif
}
