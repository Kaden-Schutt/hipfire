// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define HD 128
#define TILE_ROWS 4

// Routed FP32 Gated Delta Net recurrence for independent request sessions.
//
// Grid: [n_heads, HD/TILE_ROWS, n_sessions]; block [32].
//
// Each block owns one session/head/S-row-tile. It scans the flattened
// round-major prefix rows and applies only rows whose row_session_indices[b]
// match blockIdx.z. This is intentionally O(n_sessions * n_rows) for the first
// implementation: it preserves exact per-session recurrence order while
// removing the correctness bug of sharing one DeltaNet state across sessions.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32_routed_batch_seq(
    const float* __restrict__ q,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ k,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ v,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ gate,                 // [batch_rows * n_heads]
    const float* __restrict__ beta,                 // [batch_rows * n_heads]
    const uint64_t* __restrict__ state_ptrs,        // [n_sessions * ptr_layer_stride]
    const int* __restrict__ row_session_indices,    // [batch_rows]
    float* __restrict__ output,                     // [batch_rows * n_heads * HD]
    int ptr_layer_stride,
    int delta_layer_index,
    int batch_rows,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    const int session_idx = blockIdx.z;
    if (h >= n_heads || head_dim != HD) return;

    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];

    float* state = reinterpret_cast<float*>(
        static_cast<uintptr_t>(state_ptrs[(size_t)session_idx * ptr_layer_stride + delta_layer_index])
    );
    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_tile[i] = S_global[i];
    }

    for (int b = 0; b < batch_rows; b++) {
        if (row_session_indices[b] != session_idx) continue;

        const float* q_t = q + b * stride + h * HD;
        const float* k_t = k + b * stride + h * HD;
        const float* v_t = v + b * stride + h * HD;
        float alpha = expf(gate[b * n_heads + h]);
        float beta_v = beta[b * n_heads + h];

        float k0 = k_t[col], k1 = k_t[col + 1], k2 = k_t[col + 2], k3 = k_t[col + 3];
        float q0 = q_t[col], q1 = q_t[col + 1], q2 = q_t[col + 2], q3 = q_t[col + 3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD + col;
            float s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            float kv = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
            for (int o = 16; o > 0; o >>= 1) {
                kv += __shfl_down(kv, o);
            }
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0;
            sr[1] = s1;
            sr[2] = s2;
            sr[3] = s3;

            float out_v = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
            for (int o = 16; o > 0; o >>= 1) {
                out_v += __shfl_down(out_v, o);
            }
            if (tid == 0) {
                output[b * stride + h * HD + row] = out_v;
            }
        }
    }

    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        S_global[i] = S_tile[i];
    }
}
