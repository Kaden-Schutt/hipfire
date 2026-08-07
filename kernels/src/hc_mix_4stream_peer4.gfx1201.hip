// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Four-rank gfx1201 TP reduction fused into the Hyper-Connection consumer.
// The runtime records one reusable release event on every producer stream and
// makes every consumer stream wait before launching this kernel. HIP peer
// access therefore makes transform0..3 directly readable without materializing
// an RCCL all-reduced [hidden] tensor on every rank.

#define HC_MULT 4
#define BLOCK_HIDDEN 256

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
