// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 mHC input mapping — BATCHED variant (Phase A5, 2026-05-18).
//
// Batched twin of `hc_input_map_4stream`. Per batch position b:
//   x_out[b, d] = sum_s(a_vec[b, s] * x_in[b, s, d])
//
// Grid:  [hidden_blocks, batch_size, 1]
// Block: [BLOCK_HIDDEN, 1, 1]
//
// At batch_size == 1, byte-identical to hc_input_map_4stream.

#include <hip/hip_runtime.h>

#define HC_MULT 4
#define BLOCK_HIDDEN 256

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_input_map_4stream_batched(
    const float* __restrict__ a_vec,   // [batch, HC_MULT]
    const float* __restrict__ x_in,    // [batch, HC_MULT, hidden]
    float*       __restrict__ x_out,   // [batch, hidden]
    int hidden,
    int batch_size
) {
    const int tid = threadIdx.x;
    const int b   = blockIdx.y;
    const int d   = blockIdx.x * BLOCK_HIDDEN + tid;
    if (d >= hidden || b >= batch_size) return;

    const size_t a_base   = (size_t)b * HC_MULT;
    const size_t x_base   = (size_t)b * HC_MULT * hidden;
    const size_t out_base = (size_t)b * hidden;

    float a0 = a_vec[a_base + 0];
    float a1 = a_vec[a_base + 1];
    float a2 = a_vec[a_base + 2];
    float a3 = a_vec[a_base + 3];

    float v = a0 * x_in[x_base + 0 * hidden + d]
            + a1 * x_in[x_base + 1 * hidden + d]
            + a2 * x_in[x_base + 2 * hidden + d]
            + a3 * x_in[x_base + 3 * hidden + d];
    x_out[out_base + d] = v;
}
