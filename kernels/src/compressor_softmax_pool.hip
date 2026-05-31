// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DeepSeek V4 Compressor softmax-weighted pool along window dim.
//
// Per DeepSeek V4 upstream Compressor.forward (decode case, should_compress):
//   For each d in 0..head_dim:
//     max_score = max_t(score[t, d])
//     sum_exp   = sum_t(exp(score[t, d] - max_score))
//     output[d] = sum_t (exp(score[t, d] - max_score) / sum_exp * kv[t, d])
//
// Equivalent to (kv_state * score_state.softmax(dim=1)).sum(dim=1) in
// upstream PyTorch — the gated pooling that compresses `T` (= 2*ratio
// for overlap=true, ratio for overlap=false) consecutive token KVs
// into one [head_dim] vector via per-dim softmax weights.
//
// Shapes:
//   kv_state    : [T, head_dim] F32
//   score_state : [T, head_dim] F32
//   output      : [head_dim] F32
//
// Grid: 1 block per `head_dim / BLOCK_HIDDEN` chunk (or 1 block total if
// head_dim is small). Each thread handles one d, iterates T.
// T is small (≤16 for DeepSeek V4: 2*ratio=8 for ratio=4 overlap, or ratio=128
// for no-overlap).

#include <hip/hip_runtime.h>

#define MAX_T 256          // ratio=128 + safety
#define BLOCK_HIDDEN 256

extern "C" __global__ void compressor_softmax_pool_f32(
    const float* __restrict__ kv_state,    // [T, head_dim]
    const float* __restrict__ score_state, // [T, head_dim]
    float*       __restrict__ output,      // [head_dim]
    int T,
    int head_dim
) {
    const int d = blockIdx.x * BLOCK_HIDDEN + threadIdx.x;
    if (d >= head_dim) return;

    // Pass 1: max over T for this d
    float max_s = score_state[0 * head_dim + d];
    for (int t = 1; t < T; ++t) {
        float s = score_state[t * head_dim + d];
        if (s > max_s) max_s = s;
    }

    // Pass 2: sum_exp and weighted-sum-of-kv together (one pass)
    float sum_exp = 0.0f;
    float weighted = 0.0f;
    for (int t = 0; t < T; ++t) {
        float e = __expf(score_state[t * head_dim + d] - max_s);
        sum_exp += e;
        weighted += e * kv_state[t * head_dim + d];
    }

    output[d] = weighted / sum_exp;
}
