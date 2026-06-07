// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// LFM2 LIV double-gated short-convolution, single-token decode step.
//
// Implements the Lfm2MoeShortConv operator fused into one launch:
//
//     BCx = in_proj(x)                       (computed by caller, gemv)
//     B, C, xin = BCx.chunk(3, dim=-1)       (each [hidden]; B|C|xin layout)
//     Bx  = B * xin                          (pre-conv multiplicative gate)
//     conv_out = depthwise_causal_conv1d(Bx) (kernel_size = K, here 3)
//     y   = C * conv_out                     (post-conv multiplicative gate)
//     out = out_proj(y)                      (computed by caller, gemv)
//
// This kernel does the middle part: read BCx, apply both gates and the
// depthwise causal conv over a per-channel rolling history of the last
// (K-1) Bx values, write y, and advance the conv-state ring buffer IN PLACE.
//
// Channel-parallel: one thread per (batch, channel). K is a RUNTIME arg so
// the same kernel serves K=3 (LFM2) without touching qwen35's compile-time
// K=4 conv1d_decode kernel. KMAX bounds the per-thread window buffer.
//
// State layout: state[(b*C + c)*(K-1) + t] holds the t-th OLDEST Bx for
// channel c (oldest at t=0, newest at t=K-2). conv_bias is always false for
// LFM2, so no bias term. In-place left-shift is safe: each thread owns its
// channel's contiguous (K-1) slots and reads slot t+1 before overwriting it.
//
// Weight layout matches PyTorch Conv1d [C,1,K] flattened to weight[c*K + t]:
// weight[0] multiplies the oldest input, weight[K-1] the current input.

#include <hip/hip_runtime.h>

#define LFM2_CONV_KMAX 8

extern "C" __global__ void conv1d_gated_decode_f32(
    const float* __restrict__ bcx,        // [B, 3*C]   in_proj output (B | C | xin)
    float* __restrict__ state,            // [B, C, K-1] rolling history (updated in place)
    const float* __restrict__ weight,     // [C, K]      depthwise filter
    float* __restrict__ out_y,            // [B, C]      = C_gate * conv_out
    int B, int C, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;
    int b = idx / C;
    int c = idx % C;

    // chunk(3, dim=-1): B = bcx[0:C], C_gate = bcx[C:2C], xin = bcx[2C:3C]
    const float* row = bcx + (long long)b * (3 * C);
    float b_gate = row[c];
    float c_gate = row[C + c];
    float x_in   = row[2 * C + c];
    float bx     = b_gate * x_in;                 // pre-conv gate

    const int hist = K - 1;
    long long sbase = ((long long)b * C + c) * hist;

    // depthwise causal conv over window = [history(K-1) ..., current bx]
    float acc = 0.0f;
    float win[LFM2_CONV_KMAX];
    for (int t = 0; t < hist; t++) win[t] = state[sbase + t];
    win[hist] = bx;
    for (int t = 0; t < K; t++) acc += win[t] * weight[c * K + t];

    out_y[idx] = c_gate * acc;                    // post-conv gate

    // roll ring buffer left in place: drop oldest, append current bx
    for (int t = 0; t < hist - 1; t++) state[sbase + t] = state[sbase + t + 1];
    if (hist > 0) state[sbase + (hist - 1)] = bx;
}
