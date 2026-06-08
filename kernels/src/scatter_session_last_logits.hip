// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Copy each session's final logits row from a batch logits tensor into the
// session's persistent logits snapshot.
//
// Grid: [ceil(vocab_size / block), sessions, 1]; block [256, 1, 1].
extern "C" __global__ void scatter_session_last_logits_f32(
    const float* __restrict__ batch_logits,          // [batch_rows, vocab_size]
    const uint64_t* __restrict__ logits_ptrs,        // [sessions]
    const int* __restrict__ session_last_row_indices,// [sessions]
    int vocab_size,
    int sessions
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (s >= sessions || i >= vocab_size) return;

    int row = session_last_row_indices[s];
    if (row < 0) return;

    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(logits_ptrs[s]));
    dst[i] = batch_logits[(size_t)row * vocab_size + i];
}
