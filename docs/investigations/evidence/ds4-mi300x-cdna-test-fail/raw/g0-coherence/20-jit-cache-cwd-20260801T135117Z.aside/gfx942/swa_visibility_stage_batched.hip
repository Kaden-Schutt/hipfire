// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// SWA visibility staging — BATCHED variant (Phase B2, 2026-05-18).
//
// Given the layer's PRE-CHUNK SWA ring buffer + this chunk's per-batch
// KV, build per-batch contiguous SWA visibility windows that
// deepseek4_attn_swa_topk_batched / deepseek4_attn_swa_batched can consume directly
// (each expects `[batch, head_dim, swa_window]`).
//
// For each batch position b at absolute position p = start_pos + b:
//   visible_start = max(0, p + 1 - swa_window)
//   n_valid = (p + 1) - visible_start    (≤ swa_window)
//   For i in 0..n_valid, source slot pos_i = visible_start + i:
//     If pos_i < start_pos  → read ring[d, pos_i % swa_window]
//     Else                  → read kv_batch[pos_i - start_pos, d]
//   For i in n_valid..swa_window → leave uninitialised. The attention
//     kernel masks beyond n_valid_arr[b] so the trailing slots aren't
//     read.
//
// DeepSeek V4 K=V are tied (the same kv vector is written to both swa_k and
// swa_v rings); callers can pass the same ring + same staged-output
// pointer for both K and V, or stage twice with separate buffers.
//
// Grid:  [head_dim, batch_size, 1]
// Block: [swa_window, 1, 1]   — assumes swa_window ≤ 1024 (DeepSeek V4: 128)

#include <hip/hip_runtime.h>

extern "C" __global__ void swa_visibility_stage_batched(
    const float* __restrict__ ring,        // [head_dim, swa_window] pre-chunk
    const float* __restrict__ kv_batch,    // [B, head_dim]  this chunk's KV
    float*       __restrict__ staged,      // [B, head_dim, swa_window]
    int start_pos,
    int swa_window,
    int head_dim,
    int batch_size
) {
    const int d = blockIdx.x;
    const int b = blockIdx.y;
    const int i = threadIdx.x;
    if (d >= head_dim || b >= batch_size || i >= swa_window) return;

    const int p = start_pos + b;
    const int visible_start = max(0, p + 1 - swa_window);
    const int n_valid = (p + 1) - visible_start;
    if (i >= n_valid) return;

    const int pos_i = visible_start + i;
    float val;
    if (pos_i < start_pos) {
        const int ring_slot = pos_i % swa_window;
        val = ring[(size_t)d * swa_window + ring_slot];
    } else {
        const int chunk_b = pos_i - start_pos;
        val = kv_batch[(size_t)chunk_b * head_dim + d];
    }
    staged[((size_t)b * head_dim + d) * swa_window + i] = val;
}
