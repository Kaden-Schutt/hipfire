// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 Compressor: in-place per-slot APE add over a batched score
// buffer. Mirrors the per-position add in `compressor_forward_impl`:
//
//   for b in [0, B):
//     for d in [0, proj_dim):
//       score_batch[b, d] += ape[(start_pos + b) % ratio, d]
//
// score_batch layout: [B, proj_dim] row-major.
// ape layout:         [ratio, proj_dim] row-major.
//
// Grid: [ceil(proj_dim/256), B, 1]
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(256, 4)
__global__ void compressor_add_ape_batched_f32(
    float*       __restrict__ score_batch,
    const float* __restrict__ ape,
    int B,
    int proj_dim,
    int ratio,
    int start_pos
) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= proj_dim || b >= B) return;
    int row = (start_pos + b) % ratio;
    score_batch[(long long)b * proj_dim + d] +=
        ape[(long long)row * proj_dim + d];
}

// Single-position capture/replay twin. The ring slot is staged in a device
// buffer before every forward, so neither the APE row nor its pointer is baked
// into an immutable HIP-graph/Redline tape. For overlap compressors the ring
// slot is in [ratio, 2*ratio); modulo recovers the APE row.
extern "C" __launch_bounds__(256, 4)
__global__ void compressor_add_ape_f32_buf(
    float*       __restrict__ score,
    const float* __restrict__ ape,
    const int*   __restrict__ ring_slot_buf,
    int proj_dim,
    int ratio
) {
    const int slot = ring_slot_buf[0];
    if (slot < 0) return;
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d >= proj_dim) return;
    const int row = slot % ratio;
    score[d] += ape[(long long)row * proj_dim + d];
}
