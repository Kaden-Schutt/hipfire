// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched F16 embedding lookup. Copies N rows of an F16 table into
// `output[n, dim]` (F32), reading token ids from a device buffer so the
// markov head chain stays GPU-resident (no per-slot D2H+H2D round-trip).
// The F16→F32 widening is exact (no rounding), so this is byte-identical to
// the host `f16_to_f32` path it replaces.
extern "C" __global__ void embedding_f16_batched(
    const _Float16* __restrict__ table,   // [vocab × dim]
    float* __restrict__ output,           // [n × dim] row-major
    const int* __restrict__ token_ids,    // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_id = token_ids[batch_idx];
    const _Float16* row = table + (size_t)token_id * dim;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        out_row[elem] = (float)row[elem];
    }
}
