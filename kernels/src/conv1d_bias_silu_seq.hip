// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 (nemotron_h) xBC short-conv PREFILL scan: the multi-token form of
// `conv1d_bias_silu_decode_f32`. One launch processes a `seq`-token prompt
// instead of `seq` per-token launches. Depthwise causal conv1d (kernel_size=4)
// + per-channel bias + SiLU, one thread per channel, walking the sequence with
// a 3-input sliding window in registers (initialised from the rolling `state`,
// written back to the final 3 inputs for the decode hand-off). Bit-faithful to
// the decode kernel repeated over the sequence.
//
// weight layout [n_channels, 4]: weight[c*4+3]=newest (current),
// weight[c*4]=oldest. state slots: state[c*3]=in[t-1] (most recent prev),
// state[c*3+1]=in[t-2], state[c*3+2]=in[t-3].

#include <hip/hip_runtime.h>

extern "C" __global__ void conv1d_bias_silu_seq_f32(
    float* __restrict__ output,        // [seq * n_channels]
    const float* __restrict__ input,   // [seq * n_channels]
    const float* __restrict__ weight,  // [4 × n_channels] (c*4 + k)
    const float* __restrict__ bias,    // [n_channels]
    float* __restrict__ state,         // [n_channels × 3] (in/out)
    int seq,
    int n_channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float w3 = weight[c * 4 + 3];
    float w2 = weight[c * 4 + 2];
    float w1 = weight[c * 4 + 1];
    float w0 = weight[c * 4];
    float bc = bias[c];

    // Sliding window: p1=in[t-1], p2=in[t-2], p3=in[t-3] (decode slot order).
    float p1 = state[c * 3];
    float p2 = state[c * 3 + 1];
    float p3 = state[c * 3 + 2];

    for (int t = 0; t < seq; ++t) {
        float x = input[(size_t)t * n_channels + c];
        float acc = w3 * x + w2 * p1 + w1 * p2 + w0 * p3 + bc;
        output[(size_t)t * n_channels + c] = acc / (1.0f + expf(-acc)); // SiLU
        p3 = p2;
        p2 = p1;
        p1 = x;
    }

    state[c * 3] = p1;
    state[c * 3 + 1] = p2;
    state[c * 3 + 2] = p3;
}
