// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Four-rank gfx1201 EP decode reduction. Each rank launches this kernel on its
// local stream after a cross-rank producer barrier. Peer access makes p0..p3
// directly readable from every rank, avoiding RCCL setup plus the separate
// local add kernel for the 4096-element DS4 routed-expert partial.
extern "C" __global__ void ep_peer_reduce_add4_f32_gfx1201(
    float *__restrict__ dst,
    const float *__restrict__ p0,
    const float *__restrict__ p1,
    const float *__restrict__ p2,
    const float *__restrict__ p3,
    int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    // Fixed rank order is deliberate: every destination observes identical
    // arithmetic, independent of which device launches the kernel.
    float sum = p0[i];
    sum += p1[i];
    sum += p2[i];
    sum += p3[i];
    dst[i] += sum;
}
