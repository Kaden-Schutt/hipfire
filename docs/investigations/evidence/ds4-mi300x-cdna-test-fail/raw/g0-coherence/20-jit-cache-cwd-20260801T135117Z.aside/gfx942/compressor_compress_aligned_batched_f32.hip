// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 Compressor batched ALIGNED compress events.
//
// Replaces the per-event chain
//   compressor_overlap_concat_f32(kv_state)
//   compressor_overlap_concat_f32(score_state)
//   compressor_softmax_pool_f32(...)
// with a single launch processing N_events compress events in parallel
// over the head_dim axis. Built specifically for the DeepSeek V4 batched
// prefill path where chunks fire multiple compress events per layer.
//
// Two modes (selected by `overlap` arg):
//   overlap=1 (ratio=4, proj_dim = 2*head_dim): per event k, the
//     2*ratio-element pool window is built from
//       prev_kv (first half of [R, 2*head_dim])             when k=0
//       kv_batch[(k-1)*R..k*R]   first half of [R, 2*head_dim]   when k≥1
//       kv_batch[k*R..(k+1)*R]   second half of [R, 2*head_dim]
//   overlap=0 (ratio=128, proj_dim = head_dim): per event k, the
//     ratio-element pool window is simply kv_batch[k*R..(k+1)*R].
//
// Output: kv_cache_out[N_events, head_dim] written to kv_cache base
// at `compressed_slot_base` (caller passes the offset pointer).
//
// Grid: [ceil(head_dim/256), N_events, 1]
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

#define BLOCK_HIDDEN 256
#define MAX_T 256   // ratio=128 worst case

extern "C" __launch_bounds__(BLOCK_HIDDEN, 4)
__global__ void compressor_compress_aligned_batched_f32(
    const float* __restrict__ prev_kv,        // [R, proj_dim] (overlap=1) or unused (overlap=0)
    const float* __restrict__ prev_score,     // [R, proj_dim] (overlap=1) or unused (overlap=0)
    const float* __restrict__ kv_batch,       // [B, proj_dim]
    const float* __restrict__ score_batch,    // [B, proj_dim]
    float*       __restrict__ kv_cache_out,   // [N_events, head_dim] (caller offsets to first event)
    int R, int head_dim, int N_events, int overlap, int B
) {
    const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
    const int event_k = blockIdx.y;
    if (d >= head_dim || event_k >= N_events) return;

    const int proj_dim = overlap ? 2 * head_dim : head_dim;
    const int T = overlap ? 2 * R : R;

    // Pass 1: find max score across T window positions for this d.
    float max_s = -INFINITY;

    if (overlap) {
        // Old window: rows 0..R-1, first half [0..head_dim) col.
        for (int r = 0; r < R; r++) {
            float s;
            if (event_k == 0) {
                s = prev_score[r * proj_dim + d];
            } else {
                int b_off = (event_k - 1) * R + r;
                s = score_batch[(long long)b_off * proj_dim + d];
            }
            if (s > max_s) max_s = s;
        }
        // New window: rows R..2R-1, second half [head_dim..2*head_dim) col.
        for (int r = 0; r < R; r++) {
            int b_off = event_k * R + r;
            float s = score_batch[(long long)b_off * proj_dim + head_dim + d];
            if (s > max_s) max_s = s;
        }
    } else {
        // No-overlap: rows 0..R-1, single half.
        for (int r = 0; r < R; r++) {
            int b_off = event_k * R + r;
            float s = score_batch[(long long)b_off * proj_dim + d];
            if (s > max_s) max_s = s;
        }
    }

    // Pass 2: softmax-weighted pool.
    float sum_exp = 0.0f;
    float weighted = 0.0f;
    if (overlap) {
        for (int r = 0; r < R; r++) {
            float s, kv;
            if (event_k == 0) {
                s  = prev_score[r * proj_dim + d];
                kv = prev_kv   [r * proj_dim + d];
            } else {
                int b_off = (event_k - 1) * R + r;
                s  = score_batch[(long long)b_off * proj_dim + d];
                kv = kv_batch   [(long long)b_off * proj_dim + d];
            }
            float e = __expf(s - max_s);
            sum_exp += e;
            weighted += e * kv;
        }
        for (int r = 0; r < R; r++) {
            int b_off = event_k * R + r;
            float s  = score_batch[(long long)b_off * proj_dim + head_dim + d];
            float kv = kv_batch   [(long long)b_off * proj_dim + head_dim + d];
            float e = __expf(s - max_s);
            sum_exp += e;
            weighted += e * kv;
        }
    } else {
        for (int r = 0; r < R; r++) {
            int b_off = event_k * R + r;
            float s  = score_batch[(long long)b_off * proj_dim + d];
            float kv = kv_batch   [(long long)b_off * proj_dim + d];
            float e = __expf(s - max_s);
            sum_exp += e;
            weighted += e * kv;
        }
    }

    kv_cache_out[(long long)event_k * head_dim + d] = weighted / sum_exp;
}
