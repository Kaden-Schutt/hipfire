// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Tree-aware variant of conv1d_silu_split. For each in-block token t, the
// (KERNEL_WIDTH-1)=3 "prior" inputs come from walking t's ancestor chain
// via parent_indices[], falling back to the pre-block conv_state[] when
// the chain exits the block.
//
// Ported from SGLang's causal_conv1d_triton's HAS_EAGLE_TREE_CUSTOM_ATTN_MASK
// branch: python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
//
// Simplification: parent_indices is precomputed host-side from the tree
// structure (parent_indices[t] = parent's linear-slot index, or -1 for a
// root token whose parent is the most-recent pre-block history entry).
// SGLang computes the same mapping on-the-fly from (retrieve_next_token,
// retrieve_next_sibling); in our code path the tree is already known when
// ddtree.rs::linearize_tree runs, so we skip the derivation.
//
// Sentinel convention: parent_indices[t] < 0 means "no in-block ancestor at
// this depth". When walking further up an ancestor chain, we decrement past
// -1 → -2 → -3, which select conv_state slots 0, 1, 2 respectively.
//
// State is NOT updated. Callers that need conv_state advanced must run
// linear conv1d_silu_split on the post-acceptance spine (2-3 accepted
// tokens on typical 27B verify) separately — this is what replay_gdn
// already does.
//
// Spine topology (parent_indices = [-1, 0, 1, 2, ...]) produces outputs
// byte-exact with the linear conv1d_silu_split_f32 kernel on the same
// inputs.

extern "C" __global__ void conv1d_silu_split_tree_f32(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ state,
    const int* __restrict__ parent_indices,
    int k_dim,
    int v_dim,
    int n_tokens
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    float w0 = weight[c * 4 + 0];
    float w1 = weight[c * 4 + 1];
    float w2 = weight[c * 4 + 2];
    float w3 = weight[c * 4 + 3];
    float s0 = state[c * 3];
    float s1 = state[c * 3 + 1];
    float s2 = state[c * 3 + 2];

    for (int t = 0; t < n_tokens; t++) {
        float x_self = input[t * n_channels + c];

        int p1 = parent_indices[t];
        int p2 = (p1 >= 0) ? parent_indices[p1] : p1 - 1;
        int p3 = (p2 >= 0) ? parent_indices[p2] : p2 - 1;

        // Ancestor value: positive index → in-block input, negative → pre-block state.
        // p == -1 → s0 (most recent), p == -2 → s1, p == -3 → s2.
        float x_p1 = (p1 >= 0) ? input[p1 * n_channels + c]
                   : (p1 == -1) ? s0
                   : (p1 == -2) ? s1
                   : s2;
        float x_p2 = (p2 >= 0) ? input[p2 * n_channels + c]
                   : (p2 == -1) ? s0
                   : (p2 == -2) ? s1
                   : s2;
        float x_p3 = (p3 >= 0) ? input[p3 * n_channels + c]
                   : (p3 == -1) ? s0
                   : (p3 == -2) ? s1
                   : s2;

        float y = w3 * x_self + w2 * x_p1 + w1 * x_p2 + w0 * x_p3;
        float result = y / (1.0f + expf(-y));

        if (c < k_dim) {
            q_out[t * k_dim + c] = result;
        } else if (c < 2 * k_dim) {
            k_out[t * k_dim + (c - k_dim)] = result;
        } else {
            v_out[t * v_dim + (c - 2 * k_dim)] = result;
        }
    }
}
