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

// Micro-only TP3 graph candidate. Each block performs the peer handshake, but
// grid-strides its HC stream so the host can screen fewer spinning blocks than
// the incumbent 16 blocks/stream without changing arithmetic.
__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_mix_4stream_peer3_sync_gridstride_gfx1201(
    const float *__restrict__ x_in,
    const float *__restrict__ A,
    const float *__restrict__ scale,
    const float *__restrict__ transform0,
    const float *__restrict__ transform1,
    const float *__restrict__ transform2,
    unsigned int *__restrict__ own_signal,
    const unsigned int *__restrict__ peer_signal0,
    const unsigned int *__restrict__ peer_signal1,
    float *__restrict__ x_out,
    int hidden,
    unsigned int epoch) {
    if (threadIdx.x == 0) {
        __hip_atomic_exchange(
            own_signal, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(
                   peer_signal0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch ||
               __hip_atomic_load(
                   peer_signal1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    const int stream_out = blockIdx.y;
    const int stride = gridDim.x * BLOCK_HIDDEN;
    for (int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x; d < hidden; d += stride) {
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
