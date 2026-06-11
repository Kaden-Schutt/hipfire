// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1151 parallel prefill variant of conv1d_silu_split_f32.
//
// The generic kernel owns one channel per thread and loops over every token,
// which preserves the rolling state but leaves prefill with only C lanes of
// parallel work. For a fixed 4-tap causal conv, output[t] depends only on
// input[t], input[t-1..t-3], and the initial 3-slot state. This kernel exposes
// token parallelism by computing one (token, channel) cell per thread. A second
// tiny kernel commits the final rolling state after all outputs are written.

#if defined(__gfx1151__)
#define HIPFIRE_GFX1151 1
#else
#define HIPFIRE_GFX1151 0
#endif

extern "C" __global__ void conv1d_silu_split_f32_gfx1151(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ state,
    int k_dim,
    int v_dim,
    int n_tokens
) {
#if HIPFIRE_GFX1151
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels || t >= n_tokens) return;

    const int base = t * n_channels + c;
    const float x = input[base];
    const float p1 = (t >= 1) ? input[(t - 1) * n_channels + c] : state[c * 3];
    const float p2 = (t >= 2) ? input[(t - 2) * n_channels + c] : state[c * 3 + (1 - t)];
    const float p3 = (t >= 3) ? input[(t - 3) * n_channels + c] : state[c * 3 + (2 - t)];

    const float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * p1
                  + weight[c * 4 + 1] * p2 + weight[c * 4] * p3;
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
        q_out[0] = k_out[0] + v_out[0] + input[0] + weight[0] + state[0];
    }
#endif
}

extern "C" __global__ void conv1d_silu_split_update_state_gfx1151(
    const float* __restrict__ input,
    float* __restrict__ state,
    int n_channels,
    int n_tokens
) {
#if HIPFIRE_GFX1151
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float old0 = state[c * 3];
    float old1 = state[c * 3 + 1];
    float new0 = old0;
    float new1 = old1;
    float new2 = state[c * 3 + 2];

    if (n_tokens >= 1) {
        new0 = input[(n_tokens - 1) * n_channels + c];
        new1 = old0;
        new2 = old1;
    }
    if (n_tokens >= 2) {
        new1 = input[(n_tokens - 2) * n_channels + c];
        new2 = old0;
    }
    if (n_tokens >= 3) {
        new2 = input[(n_tokens - 3) * n_channels + c];
    }

    state[c * 3] = new0;
    state[c * 3 + 1] = new1;
    state[c * 3 + 2] = new2;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_tokens == 0) {
        state[0] += input[0] + (float)n_channels;
    }
#endif
}
