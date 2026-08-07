// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Three/four-rank gfx1201 TP reduction fused into the Hyper-Connection consumer.
// The runtime records one reusable release event on every producer stream and
// makes every consumer stream wait before launching this kernel. HIP peer
// access therefore makes transform0..3 directly readable without materializing
// an RCCL all-reduced [hidden] tensor on every rank.

#define HC_MULT 4
#define BLOCK_HIDDEN 256

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_mix_4stream_peer3_gfx1201(
    const float *__restrict__ x_in,
    const float *__restrict__ A,
    const float *__restrict__ scale,
    const float *__restrict__ transform0,
    const float *__restrict__ transform1,
    const float *__restrict__ transform2,
    float *__restrict__ x_out,
    int hidden) {
    const int stream_out = blockIdx.y;
    const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
    if (d >= hidden || stream_out >= HC_MULT) return;

    float acc = 0.0f;
#pragma unroll
    for (int stream_in = 0; stream_in < HC_MULT; ++stream_in) {
        acc += A[stream_out * HC_MULT + stream_in] *
               x_in[stream_in * hidden + d];
    }

    float transform = transform0[d];
    transform += transform1[d];
    transform += transform2[d];
    acc += scale[stream_out] * transform;
    x_out[stream_out * hidden + d] = acc;
}

// TP3 screen: one lane owns all four output streams for one hidden element.
// The incumbent launches one (stream_out, d) thread and consequently reloads
// x_in[0..4] and the three peer transforms four times. This variant preserves
// each output stream's accumulation order while loading those shared inputs
// once. It is a distinct symbol until the captured-graph micro and product
// route prove that the lower traffic outweighs the smaller grid.
__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_mix_4stream_peer3_vec4_gfx1201(
    const float *__restrict__ x_in,
    const float *__restrict__ A,
    const float *__restrict__ scale,
    const float *__restrict__ transform0,
    const float *__restrict__ transform1,
    const float *__restrict__ transform2,
    float *__restrict__ x_out,
    int hidden) {
    const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
    if (d >= hidden) return;

    const float x0 = x_in[d];
    const float x1 = x_in[hidden + d];
    const float x2 = x_in[2 * hidden + d];
    const float x3 = x_in[3 * hidden + d];
    float transform = transform0[d];
    transform += transform1[d];
    transform += transform2[d];

    float acc0 = 0.0f;
    acc0 += A[0] * x0;
    acc0 += A[1] * x1;
    acc0 += A[2] * x2;
    acc0 += A[3] * x3;
    acc0 += scale[0] * transform;

    float acc1 = 0.0f;
    acc1 += A[4] * x0;
    acc1 += A[5] * x1;
    acc1 += A[6] * x2;
    acc1 += A[7] * x3;
    acc1 += scale[1] * transform;

    float acc2 = 0.0f;
    acc2 += A[8] * x0;
    acc2 += A[9] * x1;
    acc2 += A[10] * x2;
    acc2 += A[11] * x3;
    acc2 += scale[2] * transform;

    float acc3 = 0.0f;
    acc3 += A[12] * x0;
    acc3 += A[13] * x1;
    acc3 += A[14] * x2;
    acc3 += A[15] * x3;
    acc3 += scale[3] * transform;

    x_out[d] = acc0;
    x_out[hidden + d] = acc1;
    x_out[2 * hidden + d] = acc2;
    x_out[3 * hidden + d] = acc3;
}

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_mix_4stream_peer4_gfx1201(
    const float *__restrict__ x_in,
    const float *__restrict__ A,
    const float *__restrict__ scale,
    const float *__restrict__ transform0,
    const float *__restrict__ transform1,
    const float *__restrict__ transform2,
    const float *__restrict__ transform3,
    float *__restrict__ x_out,
    int hidden) {
    const int stream_out = blockIdx.y;
    const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
    if (d >= hidden || stream_out >= HC_MULT) return;

    float acc = 0.0f;
#pragma unroll
    for (int stream_in = 0; stream_in < HC_MULT; ++stream_in) {
        acc += A[stream_out * HC_MULT + stream_in] *
               x_in[stream_in * hidden + d];
    }

    // Fixed rank order matches the prior peer-reduction exactness oracle and
    // gives every destination rank the same arithmetic independent of device.
    float transform = transform0[d];
    transform += transform1[d];
    transform += transform2[d];
    transform += transform3[d];
    acc += scale[stream_out] * transform;
    x_out[stream_out * hidden + d] = acc;
}
