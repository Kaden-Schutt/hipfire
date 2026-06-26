// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// LFM2 LIV double-gated short-convolution, prompt/prefill scan.
//
// Sequence form of conv1d_gated_decode_f32. One launch walks S prompt rows for
// each channel, producing out_y[S, C] and advancing the persistent state to the
// last K-1 gated inputs so decode can continue from the same state trajectory.

#include <hip/hip_runtime.h>

#define LFM2_CONV_KMAX 8

extern "C" __global__ void conv1d_gated_seq_f32(
    const float* __restrict__ bcx,        // [S, 3*C] in_proj output (B | C_gate | xin)
    float* __restrict__ state,            // [C, K-1] rolling history, updated in place
    const float* __restrict__ weight,     // [C, K] depthwise filter
    float* __restrict__ out_y,            // [S, C] = C_gate * conv_out
    int S, int C, int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    const int hist = K - 1;
    const long long sbase = (long long)c * hist;

    float win[LFM2_CONV_KMAX];
    for (int t = 0; t < hist; ++t) {
        win[t] = state[sbase + t];
    }

    for (int s = 0; s < S; ++s) {
        const float* row = bcx + (long long)s * (3 * C);
        const float b_gate = row[c];
        const float c_gate = row[C + c];
        const float x_in = row[2 * C + c];
        const float bx = b_gate * x_in;

        win[hist] = bx;
        float acc = 0.0f;
        for (int t = 0; t < K; ++t) {
            acc += win[t] * weight[c * K + t];
        }
        out_y[(long long)s * C + c] = c_gate * acc;

        for (int t = 0; t < hist - 1; ++t) {
            win[t] = win[t + 1];
        }
        if (hist > 0) {
            win[hist - 1] = bx;
        }
    }

    for (int t = 0; t < hist; ++t) {
        state[sbase + t] = win[t];
    }
}
