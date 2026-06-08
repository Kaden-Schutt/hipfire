// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Routed conv1d + silu + Q/K/V split for independent request sessions.
//
// Existing conv1d_silu_split_f32 processes one sequence and one conv-state
// ring. This variant processes a flattened round-major prefix row list and
// routes each row to its session's conv-state pointer:
//
//   state = state_ptrs[row_session * ptr_layer_stride + delta_layer_index]
//
// Grid: [ceil(n_channels / block), n_sessions, 1]; block [256, 1, 1].
extern "C" __global__ void conv1d_silu_split_routed_f32(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const uint64_t* __restrict__ state_ptrs,
    const int* __restrict__ row_session_indices,
    int ptr_layer_stride,
    int delta_layer_index,
    int k_dim,
    int v_dim,
    int batch_rows
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int session_idx = blockIdx.y;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    float* state = reinterpret_cast<float*>(
        static_cast<uintptr_t>(state_ptrs[(size_t)session_idx * ptr_layer_stride + delta_layer_index])
    );

    for (int b = 0; b < batch_rows; b++) {
        if (row_session_indices[b] != session_idx) continue;

        float x = input[b * n_channels + c];
        float s0 = state[c * 3];
        float s1 = state[c * 3 + 1];
        float s2 = state[c * 3 + 2];

        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

        float result = y / (1.0f + expf(-y));

        state[c * 3 + 2] = s1;
        state[c * 3 + 1] = s0;
        state[c * 3] = x;

        if (c < k_dim) {
            q_out[b * k_dim + c] = result;
        } else if (c < 2 * k_dim) {
            k_out[b * k_dim + (c - k_dim)] = result;
        } else {
            v_out[b * v_dim + (c - 2 * k_dim)] = result;
        }
    }
}
