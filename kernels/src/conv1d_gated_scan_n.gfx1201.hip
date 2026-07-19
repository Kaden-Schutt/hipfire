// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

#include <hip/hip_runtime.h>

extern "C" __global__ void conv1d_gated_scan_n_f32(
    const float* __restrict__ bcx,
    float* __restrict__ state,
    const float* __restrict__ weight,
    float* __restrict__ out_y,
    int n_tokens,
    int channels)
{
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;

    long long state_base = (long long)channel * 2;
    long long weight_base = (long long)channel * 3;
    float oldest = state[state_base];
    float newest = state[state_base + 1];

    for (int i = 0; i < n_tokens; ++i) {
        const float* row = bcx + (long long)i * (3 * channels);
        float b_gate = row[channel];
        float c_gate = row[channels + channel];
        float x_in = row[2 * channels + channel];
        float bx = b_gate * x_in;

        float acc = 0.0f;
        acc += oldest * weight[weight_base];
        acc += newest * weight[weight_base + 1];
        acc += bx * weight[weight_base + 2];
        out_y[(long long)i * channels + channel] = c_gate * acc;

        oldest = newest;
        newest = bx;
    }

    state[state_base] = oldest;
    state[state_base + 1] = newest;
}
