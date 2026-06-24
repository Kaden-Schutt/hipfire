// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 (nemotron_h) xBC short-conv decode step: depthwise causal conv1d
// (kernel_size=4) over `n_channels = conv_dim` channels, + per-channel bias
// (use_conv_bias=true), then SiLU over ALL channels. The caller splits the
// activated output into x / B / C afterwards (this kernel does NOT split).
// Mirrors `conv1d_decode_f32` (same K=4 weight layout + rolling 3-slot state)
// with the bias add and SiLU epilogue that nemotron_h's `act(conv1d(xBC))`
// requires. Channel-parallel, one thread per channel; state updated in place.
//
// weight layout [n_channels, 4] from safetensors [channels, 1, kernel_size]:
// weight[c*4+3] = newest (current), weight[c*4] = oldest.

#include <hip/hip_runtime.h>

extern "C" __global__ void conv1d_bias_silu_decode_f32(
    float* __restrict__ output,        // [n_channels]
    const float* __restrict__ input,   // [n_channels]
    const float* __restrict__ weight,  // [4 × n_channels] (c*4 + k)
    const float* __restrict__ bias,    // [n_channels]
    float* __restrict__ state,         // [n_channels × 3] (updated in place)
    int n_channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float x = input[c];
    float s0 = state[c * 3];
    float s1 = state[c * 3 + 1];
    float s2 = state[c * 3 + 2];

    float acc = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
              + weight[c * 4 + 1] * s1 + weight[c * 4] * s2 + bias[c];
    output[c] = acc / (1.0f + expf(-acc)); // SiLU

    state[c * 3 + 2] = s1;
    state[c * 3 + 1] = s0;
    state[c * 3]     = x;
}
