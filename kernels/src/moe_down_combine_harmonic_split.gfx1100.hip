// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Benchmark-only exact-gfx1100 combine for the harmonic DS4 expert split.
//
// local_outputs and remote_outputs are independently packed in canonical route
// order. slot_sources maps each of the six original route slots to its owner
// and packed row: bit 31 selects remote_outputs and bits 0..30 are the row.
// The reduction itself remains in canonical slot order and uses the same
// expression as moe_down_combine_k8_batched.
extern "C" __global__ void moe_down_combine_harmonic_split_gfx1100_candidate(
    const float* __restrict__ local_outputs,
    const float* __restrict__ remote_outputs,
    const unsigned int* __restrict__ slot_sources,
    const float* __restrict__ topk_weights,
    float* __restrict__ x_residual,
    int M,
    int K_TOP
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < K_TOP) {
            const unsigned int source = slot_sources[k];
            const unsigned int packed_row = source & 0x7fffffffu;
            const float* owner_outputs =
                (source & 0x80000000u) ? remote_outputs : local_outputs;
            const float w = topk_weights[k];
            const float v = owner_outputs[(long long)packed_row * M + col];
            acc += w * v;
        }
    }
    x_residual[col] += acc;
}
