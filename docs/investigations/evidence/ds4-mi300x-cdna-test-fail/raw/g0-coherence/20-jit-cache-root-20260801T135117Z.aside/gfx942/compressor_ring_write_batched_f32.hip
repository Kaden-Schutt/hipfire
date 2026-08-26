// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 Compressor ring-buffer batched write.
//
// Replaces B sequential d2d memcpy_dtod calls (one per batch position)
// with a single launch that scatters all B positions into their slots
// in kv_state / score_state.
//
// For each batch position b ∈ [0, B):
//   slot = (slot_base + b) % R         (overlap=0 / ratio=128)
//   slot = ratio + (slot_base + b) % R (overlap=1 / ratio=4)
//   kv_state[slot, :]    ← kv_batch[b, :]
//   score_state[slot, :] ← score_batch[b, :]
//
// Note `slot_base` is computed by the caller as `start_pos % ratio`
// for the chunk's first position; per-position slot index increments
// modulo ratio.
//
// Grid:  [ceil(proj_dim / 256), B, 1]
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(256, 4)
__global__ void compressor_ring_write_batched_f32(
    const float* __restrict__ kv_batch,    // [B, proj_dim]
    const float* __restrict__ score_batch, // [B, proj_dim]
    float*       __restrict__ kv_state,    // [state_rows, proj_dim]
    float*       __restrict__ score_state, // [state_rows, proj_dim]
    int B, int proj_dim, int R, int slot_base, int overlap
) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= proj_dim || b >= B) return;

    int slot = (slot_base + b) % R;
    if (overlap) slot += R;       // overlap=true places the "current" window at rows R..2R

    kv_state   [(long long)slot * proj_dim + d] = kv_batch   [(long long)b * proj_dim + d];
    score_state[(long long)slot * proj_dim + d] = score_batch[(long long)b * proj_dim + d];
}
