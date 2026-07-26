// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Replay-safe input staging for the sequential golden MTP verifier.
//
// The host refreshes embeddings[N, dim] and positions[N] before replay.
// Each retained segment bakes only its row index, then stages that row into
// the canonical single-token scratch buffers consumed by the golden AR
// kernel family.
extern "C" __global__ void mtp_golden_stage_input_f32(
    const float* __restrict__ embeddings,
    const int* __restrict__ positions,
    float* __restrict__ x,
    int* __restrict__ pos_out,
    int row,
    int dim
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dim) {
        x[d] = embeddings[(size_t)row * dim + d];
    }
    if (d == 0) {
        pos_out[0] = positions[row];
    }
}
