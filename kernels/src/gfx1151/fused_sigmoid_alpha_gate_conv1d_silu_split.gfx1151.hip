// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 decode fusion for Qwen3.5/3.6 DeltaNet.
//
// Combines two independent single-token decode launches:
//   fused_sigmoid_alpha_gate_f32(beta, alpha, dt_bias, a_log, n_heads)
//   conv1d_silu_split_f32(q, k, v, qkv, conv_weight, conv_state, k_dim, v_dim, 1)
//
// The math is intentionally copied from the generic kernels. There are no
// cross-buffer dependencies between alpha/beta and qkv conv before GDN, so
// fusing only changes launch topology.

#if defined(__gfx1151__)

extern "C" __global__ void fused_sigmoid_alpha_gate_conv1d_silu_split_f32_gfx1151(
    float* __restrict__ beta,
    float* __restrict__ alpha,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int n_heads,
    int k_dim,
    int v_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_channels = 2 * k_dim + v_dim;

    if (i < n_heads) {
        beta[i] = 1.0f / (1.0f + expf(-beta[i]));

        float biased = alpha[i] + dt_bias[i];
        float sp = (biased > 20.0f) ? biased
                 : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
        alpha[i] = sp * (-expf(a_log[i]));
    }

    if (i < n_channels) {
        float x = input[i];
        float s0 = state[i * 3];
        float s1 = state[i * 3 + 1];
        float s2 = state[i * 3 + 2];

        float y = weight[i * 4 + 3] * x + weight[i * 4 + 2] * s0
                + weight[i * 4 + 1] * s1 + weight[i * 4] * s2;
        float result = y / (1.0f + expf(-y));

        state[i * 3 + 2] = s1;
        state[i * 3 + 1] = s0;
        state[i * 3] = x;

        if (i < k_dim) {
            q_out[i] = result;
        } else if (i < 2 * k_dim) {
            k_out[i - k_dim] = result;
        } else {
            v_out[i - 2 * k_dim] = result;
        }
    }
}

#else

extern "C" __global__ void fused_sigmoid_alpha_gate_conv1d_silu_split_f32_gfx1151(
    float*, float*, const float*, const float*, float*, float*, float*,
    const float*, const float*, float*, int, int, int
) {}

#endif
